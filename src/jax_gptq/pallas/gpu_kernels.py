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


def _dense_reflector_update(v: jnp.ndarray, tau: jnp.ndarray, block: jnp.ndarray) -> jnp.ndarray:
    w = tau * (v @ block)
    return block - jnp.outer(v, w)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _dense_compact_panel_update(panel: CompactPanel, block: jnp.ndarray) -> jnp.ndarray:
    w = panel.y.T @ block
    w = panel.t.T @ w
    return block - panel.y @ w


def reflector_kernel_tile_cols(n_cols: int, max_block_cols: int = 128) -> int:
    block_cols = 1
    while block_cols < n_cols:
        block_cols *= 2
    block_cols = min(block_cols, max_block_cols)
    if block_cols < n_cols:
        block_cols = max_block_cols
    return block_cols


def reflector_kernel_supported_shape(block_shape: tuple[int, int]) -> bool:
    m, n = block_shape
    if m <= 0 or n <= 0:
        return False
    block_cols = reflector_kernel_tile_cols(n)
    return _is_power_of_two(m * block_cols)


def compact_panel_kernel_tile_cols(n_cols: int, max_block_cols: int = 128) -> int:
    block_cols = 1
    while block_cols < n_cols:
        block_cols *= 2
    block_cols = min(block_cols, max_block_cols)
    if block_cols < n_cols:
        block_cols = max_block_cols
    return block_cols


def compact_panel_kernel_supported_shape(panel_shape: tuple[int, int], block_shape: tuple[int, int]) -> bool:
    panel_rows, panel_width = panel_shape
    block_rows, block_cols = block_shape
    if panel_rows <= 0 or panel_width < 0 or block_rows <= 0 or block_cols <= 0:
        return False
    if panel_rows != block_rows:
        return False
    tile_cols = compact_panel_kernel_tile_cols(block_cols)
    # Triton-backed Pallas in this environment requires power-of-two array sizes.
    return (
        _is_power_of_two(panel_rows * tile_cols)
        and _is_power_of_two(panel_rows * panel_width)
        and _is_power_of_two(panel_width * panel_width)
    )


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
        return _dense_reflector_update(v, tau, block)

    m, n = block.shape
    block_cols = reflector_kernel_tile_cols(n)
    pad_cols = block_cols - n if n < block_cols else (-n) % block_cols
    block_padded = jnp.pad(block, ((0, 0), (0, pad_cols)))
    n_padded = block_padded.shape[1]
    grid_n = n_padded // block_cols

    if not reflector_kernel_supported_shape(block.shape):
        return _dense_reflector_update(v, tau, block)

    tau_buf = tau.reshape(1)

    def kernel(block_ref, v_ref, tau_ref, out_ref):
        row_idx = jnp.arange(block_ref.shape[0])
        block_tile = block_ref[row_idx, :]
        v_local = v_ref[row_idx]
        tau_local = tau_ref[0]
        w = tau_local * jnp.sum(v_local[:, None] * block_tile, axis=0)
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
    tau_spec = pl.BlockSpec(
        index_map=lambda j: (0,),
        block_shape=(1,),
    )

    updated_padded = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(grid_n,),
        in_specs=[block_spec, full_v_spec, tau_spec],
        out_specs=block_spec,
        compiler_params=pltriton.CompilerParams() if pltriton is not None else None,
    )(block_padded, v, tau_buf)

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
        return _dense_compact_panel_update(panel, block)

    if not compact_panel_kernel_supported_shape(panel.y.shape, block.shape):
        return _dense_compact_panel_update(panel, block)

    # Keep the compact WY update on the dense JAX reference path until the
    # Triton-backed version is implemented and validated.
    return _dense_compact_panel_update(panel, block)
