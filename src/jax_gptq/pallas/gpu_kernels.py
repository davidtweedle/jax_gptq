from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl
except ImportError:  # pragma: no cover - local environments may not have Pallas.
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except ImportError:  # pragma: no cover - local environments may not have Triton backend.
    pltriton = None

if TYPE_CHECKING:
    from .blocked_pivoted_qr import CompactPanel


def apply_reflector_to_block_pallas_gpu(
    v: jnp.ndarray,
    tau: jnp.ndarray,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    GPU-backed Householder block update.

    This currently uses a simple tiled Pallas kernel when explicitly enabled and
    otherwise falls back to the dense JAX expression.
    """
    v = jnp.asarray(v)
    tau = jnp.asarray(tau)
    block = jnp.asarray(block)
    if v.ndim != 1:
        raise ValueError(f"v must be 1D, got {v.shape}")
    if tau.ndim != 0:
        raise ValueError(f"tau must be scalar, got {tau.shape}")
    if block.ndim != 2:
        raise ValueError(f"block must be 2D, got {block.shape}")
    if block.shape[0] != v.shape[0]:
        raise ValueError(
            f"block row dimension must match v length, got {block.shape[0]} and {v.shape[0]}"
        )

    use_pallas = os.environ.get("JAX_GPTQ_USE_PALLAS", "0") == "1"
    if not use_pallas or pl is None or block.shape[1] == 0:
        w = tau * (v @ block)
        return block - jnp.outer(v, w)

    m, n = block.shape
    block_cols = min(128, max(1, n))
    pad_cols = (-n) % block_cols
    block_padded = jnp.pad(block, ((0, 0), (0, pad_cols)))
    n_padded = block_padded.shape[1]
    grid_n = n_padded // block_cols

    row_idx = jnp.arange(m)

    def kernel(block_ref, v_ref, out_ref):
        block_tile = block_ref[row_idx, :]
        v_local = v_ref[row_idx]
        w = tau * jnp.sum(v_local[:, None] * block_tile, axis=0)
        updated = block_tile - v_local[:, None] * w[None, :]
        out_ref[row_idx, :] = updated

    out_shape = jax.ShapeDtypeStruct((m, n_padded), block.dtype)
    block_spec = pl.BlockSpec(
        index_map=lambda j: (0, j * block_cols),
        block_shape=(m, block_cols),
    )
    full_v_spec = pl.BlockSpec(
        index_map=lambda j: (0,),
        block_shape=(m,),
    )

    updated_padded = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(grid_n,),
        in_specs=[block_spec, full_v_spec],
        out_specs=block_spec,
        compiler_params=pltriton.CompilerParams() if pltriton is not None else None,
    )(block_padded, v)

    if pad_cols:
        return updated_padded[:, :n]
    return updated_padded


def apply_compact_panel_to_block_pallas_gpu(
    panel: CompactPanel,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    GPU-backed compact WY panel application.

    This currently uses a simple tiled Pallas kernel when explicitly enabled and
    otherwise falls back to the dense JAX expression.
    """
    block = jnp.asarray(block)
    if block.ndim != 2:
        raise ValueError(f"block must be 2D, got {block.shape}")
    if block.shape[0] != panel.y.shape[0]:
        raise ValueError(
            f"block row dimension must match panel rows, got {block.shape[0]} and {panel.y.shape[0]}"
        )

    use_pallas = os.environ.get("JAX_GPTQ_USE_PALLAS", "0") == "1"
    if not use_pallas or pl is None or block.shape[1] == 0:
        w = panel.y.T @ block
        w = panel.t.T @ w
        return block - panel.y @ w

    # Keep the compact WY update on the dense JAX reference path until the
    # simpler Householder primitive is validated on Triton-backed Pallas.
    w = panel.y.T @ block
    w = panel.t.T @ w
    return block - panel.y @ w
