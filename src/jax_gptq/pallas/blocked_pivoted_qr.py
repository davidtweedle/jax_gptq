"""
Reference structure for blocked pivoted QR.

This file intentionally starts as a documented skeleton rather than a finished
implementation. The goal is to pin down the control flow before optimizing with
Pallas.

High-level plan
---------------

Reference version:
- choose the next pivot from exact current trailing norms
- swap it into place
- form one Householder reflector
- apply that reflector to all remaining columns immediately
- recompute norms exactly

Later blocked version:
- keep the same panel factorization logic
- only update the active panel columns immediately
- update trailing norm metadata without fully transforming the trailing matrix
- when a trailing column is selected as the next pivot, first apply the previous
  panel reflectors to that incoming column
- apply the full accumulated panel update to the trailing matrix once the panel
  is complete
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from time import perf_counter
from typing import Literal

import jax
import jax.numpy as jnp

from .backend import selected_kernel_backend
from .gpu_kernels import (
    apply_compact_panel_to_block_pallas_gpu,
    apply_reflector_to_block_pallas_gpu,
)
from .tpu_kernels import (
    apply_compact_panel_to_block_pallas_tpu,
    apply_reflector_to_block_pallas_tpu,
    get_tpu_kernel_debug_counters,
)

PivotMode = Literal["largest", "smallest"]

ZERO_TOL = 1e-12
ZERO_TOL_SQ = ZERO_TOL * ZERO_TOL


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CompactPanel:
    """
    Compact WY panel representation.

    Conventions:
    - `panel_start` / `panel_end` are global matrix row/column indices
    - `y` is stored in panel-local row coordinates, i.e. over rows
      `panel_start:`
    - `tau` and `t` have one entry/column per accepted panel reflector
    """
    panel_start: int
    panel_end: int
    y: jnp.ndarray
    tau: jnp.ndarray
    t: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PanelState:
    """
    Mutable panel factorization state.

    Conventions:
    - `a`, `perm`, `norms` are global work arrays
    - `y` is panel-local over rows `k:`
    - `j` is the current global panel step
    """
    a: jnp.ndarray
    perm: jnp.ndarray
    norms: jnp.ndarray
    y: jnp.ndarray
    tau: jnp.ndarray
    t: jnp.ndarray
    k: int
    j: int
    panel_end: int
    active_cols: int
    done: bool


@dataclass(frozen=True)
class FactorPanelResult:
    """
    Contract for the future fused `factor_panel_pallas(...)` kernel.

    Fields:
    - `a`: updated work matrix after factoring one panel
    - `perm`: updated global column permutation
    - `norms`: updated trailing norm metadata
    - `reflectors`: reference-only reflector list retained for parity tests
    - `panel`: compact panel representation for deferred trailing updates,
      with `panel.y` stored in panel-local row coordinates
    """
    a: jnp.ndarray
    perm: jnp.ndarray
    norms: jnp.ndarray
    reflectors: list[tuple[int, jnp.ndarray, jnp.ndarray]]
    panel: CompactPanel


def init_panel_state(
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    k: int,
    panel_size: int,
) -> PanelState:
    """
    Initialize the mutable state for one panel factorization.

    This mirrors the eventual Pallas kernel boundary:
    - `a`, `perm`, `norms` are the evolving global work state
    - `y`, `tau`, `t` are the incremental compact panel buffers
    - `j` tracks the current in-panel step
    - `active_cols` tracks the number of accepted pivots
    - `done` allows early exit for rank-deficient `pivot_mode="smallest"`
    """
    a = jnp.asarray(a)
    perm = jnp.asarray(perm)
    norms = jnp.asarray(norms)

    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if perm.ndim != 1 or perm.shape[0] != a.shape[1]:
        raise ValueError(f"perm must be shape ({a.shape[1]},), got {perm.shape}")
    if norms.ndim != 1 or norms.shape[0] != a.shape[1]:
        raise ValueError(f"norms must be shape ({a.shape[1]},), got {norms.shape}")
    if panel_size <= 0:
        raise ValueError(f"panel_size must be positive, got {panel_size}")
    if not 0 <= k < min(a.shape[0], a.shape[1]):
        raise ValueError(f"k must be in [0, {min(a.shape[0], a.shape[1])}), got {k}")

    panel_end = min(k + panel_size, min(a.shape[0], a.shape[1]))
    width = panel_end - k
    return PanelState(
        a=a,
        perm=perm,
        norms=norms,
        y=jnp.zeros((a.shape[0] - k, width), dtype=a.dtype),
        tau=jnp.zeros((width,), dtype=a.dtype),
        t=jnp.zeros((width, width), dtype=a.dtype),
        k=k,
        j=k,
        panel_end=panel_end,
        active_cols=0,
        done=False,
    )


def choose_pivot(
    norms: jnp.ndarray,
    j: int,
    pivot_mode: PivotMode,
    zero_tol: float = ZERO_TOL,
) -> int:
    """
    Choose the next pivot column from columns j:n.

    Reference version:
    - `norms` is assumed to contain exact current trailing column norms.

    Later blocked version:
    - `norms` will be metadata for the trailing columns rather than norms
      computed from a fully updated trailing matrix.
    """
    norms = jnp.asarray(norms)
    if norms.ndim != 1:
        raise ValueError(f"norms must be 1D, got {norms.shape}")
    if not 0 <= j < norms.shape[0]:
        raise ValueError(f"j must be in [0, {norms.shape[0]}), got {j}")

    trailing_norms = norms[j:]
    zero_tol_sq = zero_tol * zero_tol
    if pivot_mode == "largest":
        pivot_offset = jnp.argmax(trailing_norms)
    elif pivot_mode == "smallest":
        masked_norms = jnp.where(trailing_norms > zero_tol_sq, trailing_norms, jnp.inf)
        pivot_offset = jnp.argmin(masked_norms)
    else:
        raise ValueError(f"unsupported pivot_mode={pivot_mode}")
    return int(j + pivot_offset)


def choose_pivot_dynamic(
    norms: jnp.ndarray,
    j: int,
    pivot_mode: PivotMode,
    zero_tol: float = ZERO_TOL,
) -> jnp.ndarray:
    idx = jnp.arange(norms.shape[0], dtype=jnp.int32)
    trailing_mask = idx >= j
    zero_tol_sq = zero_tol * zero_tol
    if pivot_mode == "largest":
        masked_norms = jnp.where(trailing_mask, norms, -jnp.inf)
        return jnp.argmax(masked_norms).astype(jnp.int32)
    elif pivot_mode == "smallest":
        positive_mask = trailing_mask & (norms > zero_tol_sq)
        masked_norms = jnp.where(positive_mask, norms, jnp.inf)
        return jnp.argmin(masked_norms).astype(jnp.int32)
    else:
        raise ValueError(f"unsupported pivot_mode={pivot_mode}")


def swap_columns(
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    j: int,
    pivot_col: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Swap columns `j` and `pivot_col` in the working matrix and permutation.

    Reference version:
    - swap the physical columns in `a`
    - swap entries in `perm`
    - swap the corresponding pivot-score entries in `norms`

    Later blocked version:
    - the same bookkeeping still applies.
    """
    a = jnp.asarray(a)
    perm = jnp.asarray(perm)
    norms = jnp.asarray(norms)

    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if perm.ndim != 1:
        raise ValueError(f"perm must be 1D, got {perm.shape}")
    if norms.ndim != 1:
        raise ValueError(f"norms must be 1D, got {norms.shape}")
    if a.shape[1] != perm.shape[0] or a.shape[1] != norms.shape[0]:
        raise ValueError("a, perm, and norms must agree on column dimension")

    if j == pivot_col:
        return a, perm, norms

    orig_pos = jnp.array([j, pivot_col])
    swap_pos = jnp.array([pivot_col, j])
    a = a.at[:, orig_pos].set(a[:, swap_pos])
    perm = perm.at[orig_pos].set(perm[swap_pos])
    norms = norms.at[orig_pos].set(norms[swap_pos])
    return a, perm, norms


def swap_columns_dynamic(
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    j: int,
    pivot_col: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def no_swap(args):
        return args

    def do_swap(args):
        a, perm, norms = args
        orig_pos = jnp.array([j, pivot_col], dtype=jnp.int32)
        swap_pos = jnp.array([pivot_col, j], dtype=jnp.int32)
        a = a.at[:, orig_pos].set(a[:, swap_pos])
        perm = perm.at[orig_pos].set(perm[swap_pos])
        norms = norms.at[orig_pos].set(norms[swap_pos])
        return a, perm, norms

    return jax.lax.cond(pivot_col == j, no_swap, do_swap, (a, perm, norms))


def householder_vector(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Form one Householder reflector for the current column vector.

    Returns:
    - `v`: Householder vector
    - `tau`: scalar coefficient
    - `alpha`: resulting diagonal value for R

    Reference version:
    - this is the standard one-column QR primitive.

    Later blocked version:
    - unchanged.
    """
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {x.shape}")

    x0 = x[0]
    x_tail = x[1:]
    sigma = jnp.dot(x_tail, x_tail)

    def trivial_case(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        v = jnp.zeros_like(x).at[0].set(1)
        tau = jnp.array(0, dtype=x.dtype)
        alpha = x0
        return v, tau, alpha

    def reflector_case(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        norm_x = jnp.sqrt(x0 * x0 + sigma)
        alpha = jnp.where(x0 <= 0, norm_x, -norm_x)
        beta = x0 - alpha
        v = x / beta
        v = v.at[0].set(1)
        tau = -beta / alpha
        return v, tau, alpha

    return jax.lax.cond(sigma == 0, trivial_case, reflector_case, operand=None)


def householder_vector_dynamic(
    a: jnp.ndarray,
    j: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    m = a.shape[0]
    col = jax.lax.dynamic_slice(a, (0, j), (m, 1)).reshape((m,))
    row_idx = jnp.arange(m, dtype=jnp.int32)
    active_mask = row_idx >= j
    tail_mask = row_idx > j

    x0 = col[j]
    x_tail = jnp.where(tail_mask, col, 0)
    sigma = jnp.dot(x_tail, x_tail)

    def trivial_case(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        v = jnp.zeros_like(col).at[j].set(1)
        tau = jnp.array(0, dtype=col.dtype)
        alpha = x0
        return v, tau, alpha

    def reflector_case(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        norm_x = jnp.sqrt(x0 * x0 + sigma)
        alpha = jnp.where(x0 <= 0, norm_x, -norm_x)
        beta = x0 - alpha
        scaled = jnp.where(active_mask, col / beta, 0)
        v = scaled.at[j].set(1)
        tau = -beta / alpha
        return v, tau, alpha

    return jax.lax.cond(sigma == 0, trivial_case, reflector_case, operand=None)


def householder_vector_panel_local_dynamic(
    a: jnp.ndarray,
    k: int,
    j: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    tail = a[k:, :]
    local_j = j - k
    return householder_vector_dynamic(tail, local_j)


def apply_reflector_to_block(
    v: jnp.ndarray,
    tau: jnp.ndarray,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply H = I - tau v v^T to a block of columns.

    Reference version:
    - used to update every remaining column immediately.

    Later blocked version:
    - still the primitive operation, but the trailing block should eventually be
      updated only once after the panel is complete.
    """
    v = jnp.asarray(v)
    block = jnp.asarray(block)
    if v.ndim != 1:
        raise ValueError(f"v must be 1D, got {v.shape}")
    if block.ndim != 2:
        raise ValueError(f"block must be 2D, got {block.shape}")
    if block.shape[0] != v.shape[0]:
        raise ValueError(
            f"block row dimension must match v length, got {block.shape[0]} and {v.shape[0]}"
        )

    w = tau * (v @ block)
    return block - jnp.outer(v, w)


def apply_reflector_to_block_pallas(
    v: jnp.ndarray,
    tau: jnp.ndarray,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    Backend-dispatching entry point for one Householder block update.

    Current behavior:
    - dispatches based on the selected kernel backend
    - `reference` uses the dense JAX helper
    - `gpu` uses the GPU kernel module
    - `tpu` raises until the TPU kernel is implemented
    """
    backend = selected_kernel_backend()
    if backend == "reference":
        return apply_reflector_to_block(v, tau, block)
    if backend == "gpu":
        return apply_reflector_to_block_pallas_gpu(v, tau, block)
    if backend == "tpu":
        return apply_reflector_to_block_pallas_tpu(v, tau, block)
    raise ValueError(f"unsupported kernel backend {backend!r}")


def update_norms_exact(a: jnp.ndarray, j: int) -> jnp.ndarray:
    """
    Recompute exact trailing norms for columns j+1:n.

    Reference version:
    - recompute from the fully updated trailing matrix for simplicity.

    Later blocked version:
    - replace this with metadata updates because the trailing matrix will not be
      explicitly updated after every Householder step.
    """
    a = jnp.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if not -1 <= j < a.shape[1]:
        raise ValueError(f"j must be in [-1, {a.shape[1]}), got {j}")

    norms = jnp.zeros((a.shape[1],), dtype=a.dtype)
    if j + 1 >= a.shape[1]:
        return norms

    trailing = a[j + 1 :, j + 1 :]
    trailing_norms_sq = jnp.sum(jnp.square(trailing), axis=0)
    norms = norms.at[j + 1 :].set(trailing_norms_sq)
    return norms


def apply_reflectors_to_trailing_view(
    a: jnp.ndarray,
    reflectors,
    start_col: int,
) -> jnp.ndarray:
    """
    Form an exact temporary view of the trailing matrix after applying the
    accumulated panel reflectors.

    This is a correctness bridge between the fully unblocked reference and the
    eventual blocked implementation:
    - the stored trailing matrix remains deferred
    - but pivot scoring still sees the exact transformed trailing columns

    Later blocked version:
    - replace this exact materialization with norm metadata updates and
      on-demand application of panel reflectors only to the selected incoming
      pivot column.
    """
    trailing = a[:, start_col:]
    for j, v, tau in reflectors:
        updated = apply_reflector_to_block(v, tau, trailing[j:, :])
        trailing = trailing.at[j:, :].set(updated)
    return trailing


def apply_reflectors_to_column(
    col: jnp.ndarray,
    global_col_index: int,
    reflectors,
) -> jnp.ndarray:
    """
    Apply the accumulated panel reflectors to one incoming trailing column.

    This is the operation the eventual blocked algorithm will need when a new
    pivot column is selected from the deferred trailing matrix and must be
    brought into the current transformed basis before entering the panel.

    Current usage:
    - helper only; not yet wired into pivot selection

    Later blocked version:
    - this should replace the expensive exact trailing-view materialization for
      selected incoming columns.
    """
    col = jnp.asarray(col)
    if col.ndim != 1:
        raise ValueError(f"col must be 1D, got {col.shape}")
    if not 0 <= global_col_index < col.shape[0] + global_col_index:
        # The helper only needs the index for row alignment with stored
        # reflectors; keep the interface explicit even though the check is loose.
        pass

    out = col
    for j, v, tau in reflectors:
        updated = apply_reflector_to_block(v, tau, out[j:].reshape(-1, 1)).reshape(-1)
        out = out.at[j:].set(updated)
    return out


def compute_exposed_trailing_row(
    a: jnp.ndarray,
    reflectors,
    row_index: int,
    start_col: int,
) -> jnp.ndarray:
    """
    Compute one exposed row of the transformed trailing block exactly.

    This avoids materializing the full transformed trailing view when only a
    single row is needed for norm downdates.

    Current behavior:
    - exact
    - still applies the reflectors sequentially

    Later blocked version:
    - replace this reflector-by-reflector row extraction with a compact panel
      representation (e.g. WY form) applied to the stale trailing block.
    """
    a = jnp.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if not 0 <= row_index < a.shape[0]:
        raise ValueError(f"row_index must be in [0, {a.shape[0]}), got {row_index}")
    if not 0 <= start_col <= a.shape[1]:
        raise ValueError(f"start_col must be in [0, {a.shape[1]}], got {start_col}")

    trailing = a[:, start_col:]
    for j, v, tau in reflectors:
        updated = apply_reflector_to_block(v, tau, trailing[j:, :])
        trailing = trailing.at[j:, :].set(updated)
    return trailing[row_index, :]


def build_compact_panel(
    reflectors,
    panel_start: int,
    panel_end: int,
    n_rows: int,
) -> CompactPanel:
    """
    Build a compact panel description from the stored reflector list.

    Current behavior:
    - stores the reflector vectors in a dense `Y` matrix, one column per panel
      reflector, aligned to panel-local row indices
    - stores the corresponding scalar coefficients in `tau`

    Later blocked version:
    - use this compact form directly for panel-to-trailing updates and exposed
      row extraction.
    """
    if not 0 <= panel_start <= panel_end <= n_rows:
        raise ValueError(
            f"expected 0 <= panel_start <= panel_end <= n_rows, got "
            f"{panel_start}, {panel_end}, {n_rows}"
        )

    width = panel_end - panel_start
    y = jnp.zeros((n_rows - panel_start, width), dtype=jnp.float32)
    tau = jnp.zeros((width,), dtype=jnp.float32)

    for local_idx, (j, v, tau_j) in enumerate(reflectors):
        if not panel_start <= j < panel_end:
            raise ValueError(
                f"reflector row index {j} is outside panel [{panel_start}, {panel_end})"
            )
        local_row = j - panel_start
        y = y.at[local_row:, local_idx].set(v)
        tau = tau.at[local_idx].set(tau_j)

    t = jnp.zeros((width, width), dtype=jnp.float32)
    for i in range(width):
        t = t.at[i, i].set(tau[i])
        if i == 0:
            continue
        y_i = y[:, i]
        w = -tau[i] * (y[:, :i].T @ y_i)
        if i > 1:
            w = t[:i, :i] @ w
        t = t.at[:i, i].set(w)

    return CompactPanel(
        panel_start=panel_start,
        panel_end=panel_end,
        y=y,
        tau=tau,
        t=t,
    )


def append_reflector_to_panel_state(
    state: PanelState,
    j: int,
    v: jnp.ndarray,
    tau_j: jnp.ndarray,
) -> PanelState:
    """
    Incrementally append one reflector to the panel-state compact buffers.

    This is the stateful counterpart to `build_compact_panel(...)`.
    `v` is interpreted in panel-local coordinates, starting at local row
    `j - state.k`. For now this only updates `y`, `tau`, and `t`; the caller
    still owns the actual panel factorization logic and any consistency checks
    against the reflector-list reference path.
    """
    v = jnp.asarray(v)
    tau_j = jnp.asarray(tau_j)
    if v.ndim != 1:
        raise ValueError(f"v must be 1D, got {v.shape}")
    if tau_j.ndim != 0:
        raise ValueError(f"tau_j must be scalar, got {tau_j.shape}")
    if not state.k <= j < state.panel_end:
        raise ValueError(f"j must be in [{state.k}, {state.panel_end}), got {j}")
    local_row = j - state.k
    if state.y.shape[0] < local_row + v.shape[0]:
        raise ValueError(
            f"state.y row dimension is too small for reflector at row {j}: "
            f"{state.y.shape[0]} < {local_row + v.shape[0]}"
        )

    local_idx = j - state.k
    y_col = jnp.zeros((state.y.shape[0],), dtype=state.y.dtype)
    y_col = jax.lax.dynamic_update_slice_in_dim(y_col, v, local_row, axis=0)
    y = jax.lax.dynamic_update_slice(
        state.y,
        y_col[:, None],
        (0, local_idx),
    )

    tau = jax.lax.dynamic_update_slice(
        state.tau,
        tau_j.reshape(1),
        (local_idx,),
    )
    t = jax.lax.dynamic_update_slice(
        state.t,
        tau_j.reshape(1, 1),
        (local_idx, local_idx),
    )

    if local_idx > 0:
        y_i = y[:, local_idx]
        w = -tau_j * (y[:, :local_idx].T @ y_i)
        if local_idx > 1:
            w = t[:local_idx, :local_idx] @ w
        t = t.at[:local_idx, local_idx].set(w)

    return PanelState(
        a=state.a,
        perm=state.perm,
        norms=state.norms,
        y=y,
        tau=tau,
        t=t,
        k=state.k,
        j=state.j,
        panel_end=state.panel_end,
        active_cols=state.active_cols,
        done=state.done,
    )


def append_reflector_to_panel_state_dynamic(
    state: PanelState,
    j: jnp.ndarray,
    v: jnp.ndarray,
    tau_j: jnp.ndarray,
) -> PanelState:
    local_idx = j - state.k
    width = state.y.shape[1]
    col_idx = jnp.arange(width, dtype=jnp.int32)
    prev_mask = col_idx < local_idx

    y = jax.lax.dynamic_update_slice(
        state.y,
        v[:, None],
        (0, local_idx),
    )

    tau = jax.lax.dynamic_update_slice(
        state.tau,
        tau_j.reshape(1),
        (local_idx,),
    )
    t = jax.lax.dynamic_update_slice(
        state.t,
        tau_j.reshape(1, 1),
        (local_idx, local_idx),
    )

    y_i = jax.lax.dynamic_slice(y, (0, local_idx), (y.shape[0], 1)).reshape((y.shape[0],))
    y_prev = jnp.where(prev_mask[None, :], y, 0)
    w_full = -tau_j * (y_prev.T @ y_i)
    t_prev = jnp.where(prev_mask[:, None] & prev_mask[None, :], t, 0)
    w_full = t_prev @ w_full
    col_update = jnp.where(prev_mask, w_full, t[:, local_idx])
    t = jax.lax.dynamic_update_slice(
        t,
        col_update.reshape((-1, 1)),
        (0, local_idx),
    )

    return PanelState(
        a=state.a,
        perm=state.perm,
        norms=state.norms,
        y=y,
        tau=tau,
        t=t,
        k=state.k,
        j=state.j,
        panel_end=state.panel_end,
        active_cols=state.active_cols,
        done=state.done,
    )


def update_panel_state_after_swap(
    state: PanelState,
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    j: int,
) -> PanelState:
    """
    Refresh the panel state after swapping the selected pivot column into
    position `j`.

    The swap itself still happens outside this helper. This only makes the
    state transition explicit so the panel loop can evolve toward a step-based
    kernel contract.
    """
    return PanelState(
        a=a,
        perm=perm,
        norms=norms,
        y=state.y,
        tau=state.tau,
        t=state.t,
        k=state.k,
        j=j,
        panel_end=state.panel_end,
        active_cols=state.active_cols,
        done=state.done,
    )


def update_panel_state_after_norms(
    state: PanelState,
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
) -> PanelState:
    """
    Refresh the panel state after the in-panel norm metadata update.

    This keeps the compact panel buffers and control state unchanged while
    replacing the synchronized matrix/permutation/norm views.
    """
    return PanelState(
        a=a,
        perm=perm,
        norms=norms,
        y=state.y,
        tau=state.tau,
        t=state.t,
        k=state.k,
        j=state.j,
        panel_end=state.panel_end,
        active_cols=state.active_cols,
        done=state.done,
    )


def apply_reflector_to_panel_block_dynamic(
    a: jnp.ndarray,
    k: int,
    j: jnp.ndarray,
    panel_end: int,
    v: jnp.ndarray,
    tau: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    m = a.shape[0] - k
    panel_block = a[k:, k:panel_end]
    updated_panel = apply_reflector_to_block_pallas(v, tau, panel_block)

    col_idx = jnp.arange(k, panel_end, dtype=jnp.int32)
    active_col_mask = col_idx >= j
    panel_block = jnp.where(active_col_mask[None, :], updated_panel, panel_block)

    local_idx = j - k
    pivot_col = jax.lax.dynamic_slice(panel_block, (0, local_idx), (m, 1)).reshape((m,))
    row_idx = jnp.arange(m, dtype=jnp.int32)
    local_row = j - k
    pivot_col = jnp.where(row_idx > local_row, 0, pivot_col)
    pivot_col = jnp.where(row_idx == local_row, alpha, pivot_col)
    panel_block = jax.lax.dynamic_update_slice(
        panel_block,
        pivot_col[:, None],
        (0, local_idx),
    )

    return a.at[k:, k:panel_end].set(panel_block)


def apply_compact_panel_to_block(
    panel: CompactPanel,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply the compact panel transform to a block.

    With the current convention, the panel transform is represented as

        Q_panel = I - Y T Y^T

    and this helper applies `Q_panel` on the left to `block`.

    Accepted block layouts:
    - panel-local tail block with row shape `panel.y.shape[0]`
    - full block with row shape `panel.panel_start + panel.y.shape[0]`
      in which case only rows `panel.panel_start:` are transformed

    Current usage:
    - parity helper only

    Later blocked version:
    - use this in place of reflector-by-reflector replay for panel-to-trailing
      updates and exposed-row extraction.
    """
    block = jnp.asarray(block)
    if block.ndim != 2:
        raise ValueError(f"block must be 2D, got {block.shape}")
    if block.shape[0] == panel.y.shape[0]:
        block_tail = block
        prefix_rows = None
    elif block.shape[0] == panel.panel_start + panel.y.shape[0]:
        block_tail = block[panel.panel_start :, :]
        prefix_rows = block[: panel.panel_start, :]
    else:
        raise ValueError(
            f"block row dimension must match panel-local or full rows, got {block.shape[0]}"
        )

    w = panel.y.T @ block_tail
    w = panel.t.T @ w
    updated_tail = block_tail - panel.y @ w
    if prefix_rows is None:
        return updated_tail
    return jnp.concatenate([prefix_rows, updated_tail], axis=0)


def apply_compact_panel_to_block_pallas(
    panel: CompactPanel,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    Backend-dispatching entry point for compact panel application.

    Current behavior:
    - dispatches based on the selected kernel backend
    - `reference` uses the dense JAX helper
    - `gpu` uses the GPU kernel module
    - `tpu` raises until the TPU kernel is implemented
    """
    block = jnp.asarray(block)
    if block.shape[0] == panel.y.shape[0]:
        block_tail = block
        prefix_rows = None
    elif block.shape[0] == panel.panel_start + panel.y.shape[0]:
        block_tail = block[panel.panel_start :, :]
        prefix_rows = block[: panel.panel_start, :]
    else:
        raise ValueError(
            f"block row dimension must match panel-local or full rows, got {block.shape[0]}"
        )

    backend = selected_kernel_backend()
    if backend == "reference":
        updated_tail = apply_compact_panel_to_block(panel, block_tail)
    elif backend == "gpu":
        updated_tail = apply_compact_panel_to_block_pallas_gpu(panel, block_tail)
    elif backend == "tpu":
        updated_tail = apply_compact_panel_to_block_pallas_tpu(panel, block_tail)
    else:
        raise ValueError(f"unsupported kernel backend {backend!r}")

    if prefix_rows is None:
        return updated_tail
    return jnp.concatenate([prefix_rows, updated_tail], axis=0)


def compute_exposed_trailing_row_from_compact_panel(
    panel: CompactPanel,
    trailing_block: jnp.ndarray,
    row_index: int,
) -> jnp.ndarray:
    """
    Compute one exposed row of the transformed trailing block using the compact
    panel representation.

    This is the compact counterpart to `compute_exposed_trailing_row`.
    `row_index` is a global row index for full blocks and a panel-local row
    index for local tail blocks.
    """
    trailing_block = jnp.asarray(trailing_block)
    if trailing_block.ndim != 2:
        raise ValueError(f"trailing_block must be 2D, got {trailing_block.shape}")
    if trailing_block.shape[0] == panel.y.shape[0]:
        local_row_index = row_index
        trailing_tail = trailing_block
    elif trailing_block.shape[0] == panel.panel_start + panel.y.shape[0]:
        local_row_index = row_index - panel.panel_start
        trailing_tail = trailing_block[panel.panel_start :, :]
    else:
        raise ValueError(
            "trailing_block row dimension must match panel-local or full rows, "
            f"got {trailing_block.shape[0]}"
        )
    # For one exposed row j, use
    # B'[j, :] = B[j, :] - (Y[j, :] T^T Y^T) B
    row_y = jax.lax.dynamic_index_in_dim(
        panel.y,
        local_row_index,
        axis=0,
        keepdims=False,
    )
    row_weights = row_y @ panel.t.T
    left = row_weights @ panel.y.T
    row_update = left @ trailing_tail
    row_b = jax.lax.dynamic_index_in_dim(
        trailing_tail, local_row_index, axis=0, keepdims=False
    )
    return row_b - row_update


def update_norms_from_reflectors(
    a: jnp.ndarray,
    j: int,
    reflectors,
) -> jnp.ndarray:
    """
    Exact norm update for the blocked-structure reference path.

    Unlike `update_norms_exact`, this computes norms from a temporary trailing
    view with the accumulated panel reflectors applied.
    """
    a = jnp.asarray(a)
    norms = jnp.zeros((a.shape[1],), dtype=a.dtype)
    if j + 1 >= a.shape[1]:
        return norms

    trailing = apply_reflectors_to_trailing_view(a, reflectors, j + 1)
    trailing_norms_sq = jnp.sum(jnp.square(trailing[j + 1 :, :]), axis=0)
    norms = norms.at[j + 1 :].set(trailing_norms_sq)
    return norms


def initialize_trailing_norm_metadata(a: jnp.ndarray) -> jnp.ndarray:
    """
    Initialize per-column trailing norm metadata from the current matrix state.

    Metadata convention:
    - `norms[c]` stores the squared norm of the active trailing portion of
      column `c`
    - for the current fully synchronized reference state, this is simply the
      full column norm

    Later blocked version:
    - this metadata will be downdated inside a panel without fully updating the
      stored trailing matrix.
    """
    a = jnp.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    return jnp.sum(jnp.square(a), axis=0)


def refresh_trailing_norm_metadata(a: jnp.ndarray, start_row: int) -> jnp.ndarray:
    """
    Recompute exact trailing norm metadata from a synchronized matrix state.

    This is the conservative metadata update used at panel boundaries for now.

    Later blocked version:
    - replace or supplement this with in-panel downdates based on the exposed
      row contributions from each Householder step.
    """
    a = jnp.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if not 0 <= start_row <= a.shape[0]:
        raise ValueError(f"start_row must be in [0, {a.shape[0]}], got {start_row}")

    norms = jnp.zeros((a.shape[1],), dtype=a.dtype)
    if start_row >= a.shape[0]:
        return norms
    norms = norms.at[:].set(jnp.sum(jnp.square(a[start_row:, :]), axis=0))
    return norms


def update_trailing_norm_metadata_in_panel(
    a: jnp.ndarray,
    norms: jnp.ndarray,
    j: int,
    panel: CompactPanel,
    panel_end: int,
    pivot_mode: PivotMode = "largest",
) -> jnp.ndarray:
    """
    Update trailing norm metadata during panel factorization.

    Current behavior:
    - uses the exact exposed trailing row induced by the accumulated panel
      compact panel
    - downdates remaining panel-column squared scores every step
    - refreshes panel-column squared scores exactly every 8 completed
      in-panel steps
    - updates deferred trailing-column squared scores from the exact exposed row

    Later blocked version:
    - replace this with true in-panel norm downdates so we do not need to
      compute the exposed row by rebuilding the compact panel every step.
    """
    a = jnp.asarray(a)
    norms = jnp.asarray(norms)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if norms.ndim != 1 or norms.shape[0] != a.shape[1]:
        raise ValueError(f"norms must be shape ({a.shape[1]},), got {norms.shape}")
    if not 0 <= panel_end <= a.shape[1]:
        raise ValueError(f"panel_end must be in [0, {a.shape[1]}], got {panel_end}")
    next_col = j + 1
    panel_stop = min(panel_end, a.shape[1])
    panel_refresh_period = 8

    def do_update(norms_in: jnp.ndarray) -> jnp.ndarray:
        def update_panel_norms(norms_inner: jnp.ndarray) -> jnp.ndarray:
            panel_start = panel.panel_start
            panel_block = a[panel_start:, panel_start:panel_stop]
            current_panel_norms = norms_inner[panel_start:panel_stop]
            col_idx = jnp.arange(panel_start, panel_stop, dtype=jnp.int32)
            panel_col_mask = col_idx >= next_col
            panel_row = jax.lax.dynamic_index_in_dim(
                panel_block,
                j - panel_start,
                axis=0,
                keepdims=False,
            )
            downdated_panel_norms = jnp.maximum(
                current_panel_norms - jnp.square(panel_row),
                0,
            )
            downdated_panel_norms = jnp.where(
                panel_col_mask,
                downdated_panel_norms,
                current_panel_norms,
            )

            def refresh_exact(panel_norms_in: jnp.ndarray) -> jnp.ndarray:
                row_idx = jnp.arange(panel_block.shape[0], dtype=jnp.int32)
                masked_panel = jnp.where(
                    row_idx[:, None] >= (next_col - panel_start),
                    panel_block,
                    0,
                )
                exact_panel_norms = jnp.sum(jnp.square(masked_panel), axis=0)
                return jnp.where(
                    panel_col_mask,
                    exact_panel_norms,
                    panel_norms_in,
                )

            completed_steps = next_col - panel_start
            updated_panel_norms = jax.lax.cond(
                (completed_steps % panel_refresh_period) == 0,
                refresh_exact,
                lambda x: x,
                downdated_panel_norms,
            )
            return jax.lax.dynamic_update_slice(
                norms_inner,
                updated_panel_norms,
                (panel_start,),
            )

        norms_inner = jax.lax.cond(
            next_col < panel_stop,
            update_panel_norms,
            lambda x: x,
            norms_in,
        )

        def update_trailing_norms(norms_inner2: jnp.ndarray) -> jnp.ndarray:
            trailing_block = a[panel.panel_start :, panel_stop:]
            exposed_row = compute_exposed_trailing_row_from_compact_panel(
                panel,
                trailing_block,
                j,
            )
            current_sq = norms_inner2[panel_stop:]
            updated_sq = jnp.maximum(current_sq - jnp.square(exposed_row), 0)
            return norms_inner2.at[panel_stop:].set(updated_sq)

        return jax.lax.cond(
            panel_stop < a.shape[1],
            update_trailing_norms,
            lambda x: x,
            norms_inner,
        )

    return jax.lax.cond(next_col < a.shape[1], do_update, lambda x: x, norms)


def panel_step(
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    reflectors,
    panel_state: PanelState,
    k: int,
    j: int,
    panel_end: int,
    pivot_mode: PivotMode,
):
    """
    Execute one panel-factorization step at global column index `j`.

    This is the semantic unit the future Pallas panel kernel should preserve:
    - optional early stop for `pivot_mode="smallest"`
    - choose and swap the next pivot
    - form one Householder reflector
    - append it to the compact panel state
    - update the active panel block
    - refresh the in-panel norm metadata
    """
    if pivot_mode == "smallest":
        trailing_norms = norms[j:]
        if not bool(jnp.any(trailing_norms > (1e-12 * 1e-12))):
            panel_state = PanelState(
                a=panel_state.a,
                perm=panel_state.perm,
                norms=panel_state.norms,
                y=panel_state.y,
                tau=panel_state.tau,
                t=panel_state.t,
                k=panel_state.k,
                j=j,
                panel_end=panel_state.panel_end,
                active_cols=panel_state.active_cols,
                done=True,
            )
            return a, perm, norms, reflectors, panel_state, True

    pivot_col = choose_pivot(norms, j, pivot_mode)
    a, perm, norms = swap_columns(a, perm, norms, j, pivot_col)
    panel_state = update_panel_state_after_swap(panel_state, a, perm, norms, j)

    v, tau, alpha = householder_vector(a[j:, j])
    reflectors.append((j, v, tau))
    panel_state = append_reflector_to_panel_state(panel_state, j, v, tau)
    panel = CompactPanel(
        panel_start=k,
        panel_end=k + panel_state.active_cols + 1,
        y=panel_state.y,
        tau=panel_state.tau,
        t=panel_state.t,
    )

    updated_block = apply_reflector_to_block_pallas(v, tau, a[j:, j:panel_end])
    updated_block = updated_block.at[1:, 0].set(0)
    updated_block = updated_block.at[0, 0].set(alpha)
    a = a.at[j:, j:panel_end].set(updated_block)
    panel_state = PanelState(
        a=a,
        perm=perm,
        norms=norms,
        y=panel_state.y,
        tau=panel_state.tau,
        t=panel_state.t,
        k=panel_state.k,
        j=j + 1,
        panel_end=panel_state.panel_end,
        active_cols=panel_state.active_cols + 1,
        done=panel_state.done,
    )

    norms = update_trailing_norm_metadata_in_panel(
        a,
        norms,
        j,
        panel,
        panel_end,
        pivot_mode=pivot_mode,
    )
    panel_state = update_panel_state_after_norms(panel_state, a, perm, norms)
    return a, perm, norms, reflectors, panel_state, False


def factor_panel(
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    k: int,
    panel_size: int,
    pivot_mode: PivotMode,
):
    """
    Factor one panel starting at global column `k`.

    Panel invariant at local step `j`:
    - columns k..j-1 are finalized
    - remaining columns j:n are candidates for the next pivot

    Reference version:
    - choose pivot from exact norms
    - swap it into place
    - form a Householder reflector
    - apply it to all remaining columns immediately
    - recompute exact norms

    Later blocked version:
    - only update the remaining panel columns immediately
    - do NOT fully update the trailing block after every step
    - maintain trailing norm metadata instead
    - if a trailing column is chosen as the next pivot, first apply the previous
      panel reflectors to that incoming column before forming the next
      Householder reflector
    - after the panel is complete, apply the accumulated panel transform to the
      trailing matrix in one shot
    """
    a = jnp.asarray(a)
    perm = jnp.asarray(perm)
    norms = jnp.asarray(norms)

    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if perm.ndim != 1 or perm.shape[0] != a.shape[1]:
        raise ValueError(f"perm must be shape ({a.shape[1]},), got {perm.shape}")
    if norms.ndim != 1 or norms.shape[0] != a.shape[1]:
        raise ValueError(f"norms must be shape ({a.shape[1]},), got {norms.shape}")
    if panel_size <= 0:
        raise ValueError(f"panel_size must be positive, got {panel_size}")
    if not 0 <= k < min(a.shape[0], a.shape[1]):
        raise ValueError(f"k must be in [0, {min(a.shape[0], a.shape[1])}), got {k}")

    state = init_panel_state(a=a, perm=perm, norms=norms, k=k, panel_size=panel_size)
    a = state.a
    perm = state.perm
    norms = state.norms
    panel_end = state.panel_end
    reflectors: list[tuple[int, jnp.ndarray, jnp.ndarray]] = []
    panel_state = state
    timing_enabled = os.environ.get("JAX_GPTQ_QR_TIMING", "0") == "1"
    choose_pivot_total = 0.0
    swap_columns_total = 0.0
    householder_total = 0.0
    append_panel_total = 0.0
    apply_reflector_total = 0.0
    update_norms_total = 0.0
    for j in range(k, panel_end):
        if pivot_mode == "smallest":
            trailing_norms = norms[j:]
            if not bool(jnp.any(trailing_norms > ZERO_TOL_SQ)):
                panel_state = PanelState(
                    a=panel_state.a,
                    perm=panel_state.perm,
                    norms=panel_state.norms,
                    y=panel_state.y,
                    tau=panel_state.tau,
                    t=panel_state.t,
                    k=panel_state.k,
                    j=j,
                    panel_end=panel_state.panel_end,
                    active_cols=panel_state.active_cols,
                    done=True,
                )
                break

        t0 = perf_counter()
        pivot_col = choose_pivot(norms, j, pivot_mode)
        if timing_enabled:
            choose_pivot_total += perf_counter() - t0

        t0 = perf_counter()
        a, perm, norms = swap_columns(a, perm, norms, j, pivot_col)
        panel_state = update_panel_state_after_swap(panel_state, a, perm, norms, j)
        if timing_enabled:
            a.block_until_ready()
            swap_columns_total += perf_counter() - t0

        t0 = perf_counter()
        v, tau, alpha = householder_vector(a[j:, j])
        if timing_enabled:
            v.block_until_ready()
            tau.block_until_ready()
            alpha.block_until_ready()
            householder_total += perf_counter() - t0

        reflectors.append((j, v, tau))

        t0 = perf_counter()
        panel_state = append_reflector_to_panel_state(panel_state, j, v, tau)
        if timing_enabled:
            panel_state.y.block_until_ready()
            append_panel_total += perf_counter() - t0

        panel = CompactPanel(
            panel_start=k,
            panel_end=k + panel_state.active_cols + 1,
            y=panel_state.y,
            tau=panel_state.tau,
            t=panel_state.t,
        )

        t0 = perf_counter()
        updated_block = apply_reflector_to_block_pallas(v, tau, a[j:, j:panel_end])
        updated_block = updated_block.at[1:, 0].set(0)
        updated_block = updated_block.at[0, 0].set(alpha)
        a = a.at[j:, j:panel_end].set(updated_block)
        if timing_enabled:
            a.block_until_ready()
            apply_reflector_total += perf_counter() - t0

        panel_state = PanelState(
            a=a,
            perm=perm,
            norms=norms,
            y=panel_state.y,
            tau=panel_state.tau,
            t=panel_state.t,
            k=panel_state.k,
            j=j + 1,
            panel_end=panel_state.panel_end,
            active_cols=panel_state.active_cols + 1,
            done=panel_state.done,
        )

        t0 = perf_counter()
        norms = update_trailing_norm_metadata_in_panel(
            a,
            norms,
            j,
            panel,
            panel_end,
            pivot_mode=pivot_mode,
        )
        panel_state = update_panel_state_after_norms(panel_state, a, perm, norms)
        if timing_enabled:
            norms.block_until_ready()
            update_norms_total += perf_counter() - t0

    final_panel = CompactPanel(
        panel_start=k,
        panel_end=k + panel_state.active_cols,
        y=panel_state.y,
        tau=panel_state.tau,
        t=panel_state.t,
    )
    if timing_enabled:
        print(
            f"factor_panel_timing panel=[{k},{final_panel.panel_end}) "
            f"active_cols={panel_state.active_cols}"
        )
        print(f"  choose_pivot_sec={choose_pivot_total:.6f}")
        print(f"  swap_columns_sec={swap_columns_total:.6f}")
        print(f"  householder_sec={householder_total:.6f}")
        print(f"  append_panel_sec={append_panel_total:.6f}")
        print(f"  apply_reflector_sec={apply_reflector_total:.6f}")
        print(f"  update_norms_sec={update_norms_total:.6f}")
    return a, perm, norms, reflectors, final_panel


def apply_panel_to_trailing(
    a: jnp.ndarray,
    panel: CompactPanel,
    panel_end: int,
) -> jnp.ndarray:
    """
    Apply the accumulated panel reflectors to the trailing matrix.

    Reference version:
    - this is effectively redundant because the trailing matrix was already
      updated at each step.

    Later blocked version:
    - this becomes the main trailing update step.
    - eventually this should become a compact block update rather than a loop
      over individual reflectors.
    """
    a = jnp.asarray(a)
    if panel_end >= a.shape[1]:
        return a

    updated = apply_compact_panel_to_block(panel, a[panel.panel_start :, panel_end:])
    a = a.at[:, panel_end:].set(updated)
    return a


def factor_panel_pallas(
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    k: int,
    panel_size: int,
    pivot_mode: PivotMode,
):
    """
    Kernel-shaped entry point for factoring a single panel.

    Current behavior:
    - thin wrapper around the reference `factor_panel(...)`

    Intended future behavior:
    - dispatch to one fused Pallas kernel that executes the full in-panel loop
      over `j` and emits the compact panel state plus updated metadata.

    Kernel contract:
    Inputs:
    - `a`: full work matrix
    - `perm`: full column permutation
    - `norms`: full per-column trailing norm metadata
    - `k`: panel start index
    - `panel_size`: nominal panel width
    - `pivot_mode`: pivot selection rule

    Outputs:
    - `FactorPanelResult` containing the updated work state and the compact
      panel object needed by the deferred trailing-update kernel
    """
    reference_timing_enabled = os.environ.get("JAX_GPTQ_QR_TIMING", "0") == "1"
    if pivot_mode in ("largest", "smallest") and not reference_timing_enabled:
        a_out, perm_out, norms_out, panel = _factor_panel_compiled(
            a=a,
            perm=perm,
            norms=norms,
            k=k,
            panel_size=panel_size,
            pivot_mode=pivot_mode,
        )
        if os.environ.get("JAX_GPTQ_RECONSTRUCT_REFLECTORS", "0") == "1":
            panel_end_int = int(jax.device_get(panel.panel_end))
            reflectors = []
            for local_idx in range(panel_end_int - k):
                j = k + local_idx
                reflectors.append((j, panel.y[local_idx:, local_idx], panel.tau[local_idx]))
        else:
            reflectors = []
    else:
        a_out, perm_out, norms_out, reflectors, panel = factor_panel(
            a=a,
            perm=perm,
            norms=norms,
            k=k,
            panel_size=panel_size,
            pivot_mode=pivot_mode,
        )
    return FactorPanelResult(
        a=a_out,
        perm=perm_out,
        norms=norms_out,
        reflectors=reflectors,
        panel=panel,
    )


@partial(jax.jit, static_argnames=("k", "panel_size", "pivot_mode"))
def _factor_panel_compiled(
    a: jnp.ndarray,
    perm: jnp.ndarray,
    norms: jnp.ndarray,
    k: int,
    panel_size: int,
    pivot_mode: PivotMode,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, CompactPanel]:
    state = init_panel_state(a=a, perm=perm, norms=norms, k=k, panel_size=panel_size)
    a = state.a
    perm = state.perm
    norms = state.norms
    panel_end = state.panel_end
    panel_state = state
    def body_fun(j, carry):
        a, perm, norms, panel_state = carry
        def skip_step(carry_in):
            return carry_in

        def run_step(carry_in):
            a, perm, norms, panel_state = carry_in

            def stop_early(state_in):
                a, perm, norms, panel_state = state_in
                panel_state = PanelState(
                    a=panel_state.a,
                    perm=panel_state.perm,
                    norms=panel_state.norms,
                    y=panel_state.y,
                    tau=panel_state.tau,
                    t=panel_state.t,
                    k=panel_state.k,
                    j=j,
                    panel_end=panel_state.panel_end,
                    active_cols=panel_state.active_cols,
                    done=True,
                )
                return a, perm, norms, panel_state

            def continue_step(state_in):
                a, perm, norms, panel_state = state_in
                pivot_col = choose_pivot_dynamic(norms, j, pivot_mode)
                a, perm, norms = swap_columns_dynamic(a, perm, norms, j, pivot_col)
                panel_state = update_panel_state_after_swap(panel_state, a, perm, norms, j)

                v, tau, alpha = householder_vector_panel_local_dynamic(a, k, j)
                panel_state = append_reflector_to_panel_state_dynamic(panel_state, j, v, tau)
                panel = CompactPanel(
                    panel_start=k,
                    panel_end=k + panel_state.active_cols + 1,
                    y=panel_state.y,
                    tau=panel_state.tau,
                    t=panel_state.t,
                )

                a = apply_reflector_to_panel_block_dynamic(a, k, j, panel_end, v, tau, alpha)

                panel_state = PanelState(
                    a=a,
                    perm=perm,
                    norms=norms,
                    y=panel_state.y,
                    tau=panel_state.tau,
                    t=panel_state.t,
                    k=panel_state.k,
                    j=j + 1,
                    panel_end=panel_state.panel_end,
                    active_cols=panel_state.active_cols + 1,
                    done=panel_state.done,
                )

                norms = update_trailing_norm_metadata_in_panel(
                    a,
                    norms,
                    j,
                    panel,
                    panel_end,
                    pivot_mode=pivot_mode,
                )
                panel_state = update_panel_state_after_norms(panel_state, a, perm, norms)
                return a, perm, norms, panel_state

            if pivot_mode == "smallest":
                idx = jnp.arange(norms.shape[0], dtype=jnp.int32)
                trailing_has_mass = jnp.any((idx >= j) & (norms > ZERO_TOL_SQ))
                return jax.lax.cond(
                    trailing_has_mass,
                    continue_step,
                    stop_early,
                    (a, perm, norms, panel_state),
                )
            return continue_step((a, perm, norms, panel_state))

        return jax.lax.cond(panel_state.done, skip_step, run_step, carry)

    a, perm, norms, panel_state = jax.lax.fori_loop(
        k,
        panel_end,
        body_fun,
        (a, perm, norms, panel_state),
    )

    final_panel = CompactPanel(
        panel_start=k,
        panel_end=k + panel_state.active_cols,
        y=panel_state.y,
        tau=panel_state.tau,
        t=panel_state.t,
    )
    return a, perm, norms, final_panel


def apply_panel_to_trailing_pallas(
    a: jnp.ndarray,
    panel: CompactPanel,
    panel_end: int,
) -> jnp.ndarray:
    """
    Kernel-shaped entry point for the deferred trailing update.

    Current behavior:
    - thin wrapper around `apply_panel_to_trailing(...)`

    Intended future behavior:
    - dispatch to a dense Pallas kernel that applies the compact panel
      transform to the trailing block.

    Kernel contract:
    Inputs:
    - `a`: full work matrix containing the stale trailing block
    - `panel`: compact panel transform emitted by `factor_panel_pallas(...)`
    - `panel_end`: first trailing column index

    Output:
    - updated work matrix with the compact panel applied to `a[:, panel_end:]`
    """
    a = jnp.asarray(a)
    if panel_end >= a.shape[1]:
        return a

    updated = apply_compact_panel_to_block_pallas(
        panel,
        a[panel.panel_start :, panel_end:],
    )
    return a.at[:, panel_end:].set(updated)


def blocked_pivoted_qr(
    a: jnp.ndarray,
    panel_size: int,
    pivot_mode: PivotMode = "largest",
):
    """
    Top-level blocked pivoted QR driver.

    Reference version:
    - initialize permutation and exact norms
    - factor panels one at a time
    - because the trailing matrix is updated immediately, panel-end trailing
      updates are conceptually present but practically redundant

    Later blocked version:
    - exact same top-level structure
    - the difference is that trailing updates move out of the inner loop and
      happen only once per completed panel
    """
    a = jnp.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if panel_size <= 0:
        raise ValueError(f"panel_size must be positive, got {panel_size}")

    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = initialize_trailing_norm_metadata(a)
    reference_timing_enabled = os.environ.get("JAX_GPTQ_QR_TIMING", "0") == "1"
    fused_timing_enabled = os.environ.get("JAX_GPTQ_QR_FUSED_TIMING", "0") == "1"
    timing_enabled = reference_timing_enabled or fused_timing_enabled
    factor_panel_total = 0.0
    apply_panel_total = 0.0
    refresh_norms_total = 0.0
    panel_widths: list[int] = []
    num_panels = 0

    work = a
    limit = min(a.shape[0], a.shape[1])
    for k in range(0, limit, panel_size):
        factor_t0 = perf_counter()
        result = factor_panel_pallas(
            a=work,
            perm=perm,
            norms=norms,
            k=k,
            panel_size=panel_size,
            pivot_mode=pivot_mode,
        )
        if timing_enabled:
            result.a.block_until_ready()
            factor_panel_total += perf_counter() - factor_t0
        work = result.a
        perm = result.perm
        norms = result.norms
        panel = result.panel

        panel_end = panel.panel_end
        panel_widths.append(panel_end - k)
        num_panels += 1
        apply_t0 = perf_counter()
        work = apply_panel_to_trailing_pallas(work, panel, panel_end)
        if timing_enabled:
            work.block_until_ready()
            apply_panel_total += perf_counter() - apply_t0
        refresh_t0 = perf_counter()
        norms = refresh_trailing_norm_metadata(work, panel_end)
        if timing_enabled:
            norms.block_until_ready()
            refresh_norms_total += perf_counter() - refresh_t0

    if timing_enabled:
        avg_panel_width = (sum(panel_widths) / num_panels) if num_panels else 0.0
        if fused_timing_enabled and not reference_timing_enabled:
            print("blocked_pivoted_qr_fused_timing")
        else:
            print("blocked_pivoted_qr_timing")
        print(f"  num_panels={num_panels}")
        print(f"  avg_panel_width={avg_panel_width:.2f}")
        print(f"  panel_widths_head={panel_widths[:8]}")
        print(f"  factor_panel_total_sec={factor_panel_total:.6f}")
        print(f"  apply_panel_total_sec={apply_panel_total:.6f}")
        print(f"  refresh_norms_total_sec={refresh_norms_total:.6f}")
        if fused_timing_enabled and not reference_timing_enabled:
            if pivot_mode in ("largest", "smallest"):
                print("  factor_panel_impl=compiled_fori_loop")
            else:
                print("  factor_panel_impl=reference_smallest")
        if os.environ.get("JAX_GPTQ_TPU_KERNEL_DEBUG", "0") == "1":
            counters = get_tpu_kernel_debug_counters()
            print("tpu_kernel_counters")
            print(f"  reflector_kernel_calls={counters['reflector_kernel_calls']}")
            print(f"  reflector_fallback_calls={counters['reflector_fallback_calls']}")
            print(f"  compact_panel_kernel_calls={counters['compact_panel_kernel_calls']}")
            print(f"  compact_panel_fallback_calls={counters['compact_panel_fallback_calls']}")

    return work, perm
