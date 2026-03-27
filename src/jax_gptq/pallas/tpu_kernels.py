from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax
from jax import lax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl
except ImportError:  # pragma: no cover - local environments may not have Pallas.
    pl = None

if TYPE_CHECKING:
    from .blocked_pivoted_qr import CompactPanel


def _dense_reflector_update(v: jnp.ndarray, tau: jnp.ndarray, block: jnp.ndarray) -> jnp.ndarray:
    w = tau * (v @ block)
    return block - jnp.outer(v, w)


def _dense_compact_panel_update(panel: CompactPanel, block: jnp.ndarray) -> jnp.ndarray:
    w = panel.y.T @ block
    w = panel.t.T @ w
    return block - panel.y @ w


def tpu_reflector_kernel_supported_shape(block_shape: tuple[int, int]) -> bool:
    m, n = block_shape
    if m <= 0 or n <= 0:
        return False
    # Conservative first pass for TPU Pallas:
    # - keep the row dimension fully materialized
    # - require the trailing tile width to be either 1 or a multiple of 128
    # - require the row dimension to be a multiple of 8, matching TPU block-shape guidance
    return (m % 8 == 0) and (n == 1 or n % 128 == 0)


def tpu_compact_panel_kernel_supported_shape(
    panel_shape: tuple[int, int],
    block_shape: tuple[int, int],
) -> bool:
    panel_rows, panel_width = panel_shape
    block_rows, block_cols = block_shape
    if panel_rows <= 0 or panel_width <= 0 or block_rows <= 0 or block_cols <= 0:
        return False
    if panel_rows != block_rows:
        return False
    # Conservative first pass for TPU Pallas:
    # - the row dimension should follow the TPU-friendly multiple-of-8 rule
    # - the compact panel width should be small and regular
    # - the block width should be a multiple of 128
    return (panel_rows % 8 == 0) and (panel_width % 8 == 0) and (block_cols % 128 == 0)


def apply_reflector_to_block_pallas_tpu(
    v: jnp.ndarray,
    tau: jnp.ndarray,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    TPU-backed Householder block update.

    Current behavior:
    - uses a conservative supported-shape predicate for TPU Pallas
    - falls back to the dense JAX reference path for unsupported shapes

    This keeps the first TPU path narrow while we validate the backend.
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

    if block.dtype != jnp.float32:
        return _dense_reflector_update(v, tau, block)

    if not tpu_reflector_kernel_supported_shape(block.shape):
        return _dense_reflector_update(v, tau, block)

    m, n = block.shape
    tau_buf = tau.reshape(1)

    def kernel(block_ref, v_ref, tau_ref, out_ref):
        block_tile = block_ref[:, :]
        v_local = v_ref[:]
        tau_local = tau_ref[:]
        w = tau_local * pl.dot(
            v_local[None, :],
            block_tile,
            precision=lax.Precision.HIGHEST,
        )
        updated = block_tile - v_local[:, None] * w
        out_ref[:, :] = updated

    out_shape = jax.ShapeDtypeStruct((m, n), block.dtype)
    block_spec = pl.BlockSpec(
        index_map=lambda: (0, 0),
        block_shape=(m, n),
    )
    full_v_spec = pl.BlockSpec(
        index_map=lambda: (0,),
        block_shape=(m,),
    )
    tau_spec = pl.BlockSpec(
        index_map=lambda: (0,),
        block_shape=(1,),
    )

    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[block_spec, full_v_spec, tau_spec],
        out_specs=block_spec,
    )(block, v, tau_buf)


def apply_compact_panel_to_block_pallas_tpu(
    panel: CompactPanel,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    TPU-backed compact WY panel application.

    Current behavior:
    - uses a conservative supported-shape predicate for TPU Pallas
    - falls back to the dense JAX reference path for unsupported shapes
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

    if block.dtype != jnp.float32:
        return _dense_compact_panel_update(panel, block)

    if not tpu_compact_panel_kernel_supported_shape(panel.y.shape, block.shape):
        return _dense_compact_panel_update(panel, block)

    m, n = block.shape
    b = panel.y.shape[1]

    def kernel(block_ref, y_ref, t_ref, out_ref):
        block_tile = block_ref[:, :]
        y = y_ref[:, :]
        t = t_ref[:, :]
        w = pl.dot(
            y.T,
            block_tile,
            precision=lax.Precision.HIGHEST,
        )
        w = pl.dot(
            t.T,
            w,
            precision=lax.Precision.HIGHEST,
        )
        updated = block_tile - pl.dot(
            y,
            w,
            precision=lax.Precision.HIGHEST,
        )
        out_ref[:, :] = updated

    out_shape = jax.ShapeDtypeStruct((m, n), block.dtype)
    block_spec = pl.BlockSpec(
        index_map=lambda: (0, 0),
        block_shape=(m, n),
    )
    full_y_spec = pl.BlockSpec(
        index_map=lambda: (0, 0),
        block_shape=(m, b),
    )
    full_t_spec = pl.BlockSpec(
        index_map=lambda: (0, 0),
        block_shape=(b, b),
    )

    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[block_spec, full_y_spec, full_t_spec],
        out_specs=block_spec,
    )(block, panel.y, panel.t)
