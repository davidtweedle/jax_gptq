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

from typing import Literal

import jax
import jax.numpy as jnp

PivotMode = Literal["largest", "smallest"]


def choose_pivot(norms: jnp.ndarray, j: int, pivot_mode: PivotMode) -> int:
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
    if pivot_mode == "largest":
        pivot_offset = jnp.argmax(trailing_norms)
    elif pivot_mode == "smallest":
        pivot_offset = jnp.argmin(trailing_norms)
    else:
        raise ValueError(f"unsupported pivot_mode={pivot_mode}")
    return int(j + pivot_offset)


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
    trailing_norms = jnp.linalg.norm(trailing, axis=0)
    norms = norms.at[j + 1 :].set(trailing_norms)
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
    trailing_norms = jnp.linalg.norm(trailing[j + 1 :, :], axis=0)
    norms = norms.at[j + 1 :].set(trailing_norms)
    return norms


def initialize_trailing_norm_metadata(a: jnp.ndarray) -> jnp.ndarray:
    """
    Initialize per-column trailing norm metadata from the current matrix state.

    Metadata convention:
    - `norms[c]` stores the norm of the active trailing portion of column `c`
    - for the current fully synchronized reference state, this is simply the
      full column norm

    Later blocked version:
    - this metadata will be downdated inside a panel without fully updating the
      stored trailing matrix.
    """
    a = jnp.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    return jnp.linalg.norm(a, axis=0)


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
    norms = norms.at[:].set(jnp.linalg.norm(a[start_row:, :], axis=0))
    return norms


def update_trailing_norm_metadata_in_panel(
    a: jnp.ndarray,
    norms: jnp.ndarray,
    j: int,
    reflectors,
    panel_end: int,
) -> jnp.ndarray:
    """
    Update trailing norm metadata during panel factorization.

    Current behavior:
    - uses the exact exposed trailing row induced by the accumulated panel
      reflectors
    - updates remaining panel-column norms directly from the explicitly updated
      panel block
    - updates deferred trailing-column norms from the exact exposed row

    Later blocked version:
    - replace this with true in-panel norm downdates so we do not need to
      compute the exposed row exactly from the full reflector sequence.
    """
    a = jnp.asarray(a)
    norms = jnp.asarray(norms)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got {a.shape}")
    if norms.ndim != 1 or norms.shape[0] != a.shape[1]:
        raise ValueError(f"norms must be shape ({a.shape[1]},), got {norms.shape}")
    if not 0 <= panel_end <= a.shape[1]:
        raise ValueError(f"panel_end must be in [0, {a.shape[1]}], got {panel_end}")
    if j + 1 >= a.shape[1]:
        return norms

    next_col = j + 1
    panel_stop = min(panel_end, a.shape[1])

    if next_col < panel_stop:
        panel_block = a[next_col:, next_col:panel_stop]
        panel_norms = jnp.linalg.norm(panel_block, axis=0)
        norms = norms.at[next_col:panel_stop].set(panel_norms)

    if panel_stop < a.shape[1]:
        exposed_row = compute_exposed_trailing_row(a, reflectors, j, panel_stop)
        current_sq = jnp.square(norms[panel_stop:])
        updated_sq = jnp.maximum(current_sq - jnp.square(exposed_row), 0)
        updated = jnp.sqrt(updated_sq)
        norms = norms.at[panel_stop:].set(updated)
    return norms


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

    panel_end = min(k + panel_size, min(a.shape[0], a.shape[1]))
    reflectors: list[tuple[int, jnp.ndarray, jnp.ndarray]] = []

    for j in range(k, panel_end):
        pivot_col = choose_pivot(norms, j, pivot_mode)
        a, perm, norms = swap_columns(a, perm, norms, j, pivot_col)

        v, tau, alpha = householder_vector(a[j:, j])
        reflectors.append((j, v, tau))

        updated_block = apply_reflector_to_block(v, tau, a[j:, j:panel_end])
        updated_block = updated_block.at[1:, 0].set(0)
        updated_block = updated_block.at[0, 0].set(alpha)
        a = a.at[j:, j:panel_end].set(updated_block)

        norms = update_trailing_norm_metadata_in_panel(a, norms, j, reflectors, panel_end)

    return a, perm, norms, reflectors


def apply_panel_to_trailing(
    a: jnp.ndarray,
    reflectors,
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

    for j, v, tau in reflectors:
        updated = apply_reflector_to_block(v, tau, a[j:, panel_end:])
        a = a.at[j:, panel_end:].set(updated)
    return a


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

    work = a
    limit = min(a.shape[0], a.shape[1])
    for k in range(0, limit, panel_size):
        work, perm, norms, reflectors = factor_panel(
            a=work,
            perm=perm,
            norms=norms,
            k=k,
            panel_size=panel_size,
            pivot_mode=pivot_mode,
        )
        panel_end = min(k + panel_size, limit)
        work = apply_panel_to_trailing(work, reflectors, panel_end)
        norms = refresh_trailing_norm_metadata(work, panel_end)

    return work, perm
