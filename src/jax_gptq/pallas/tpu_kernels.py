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


_TPU_KERNEL_DEBUG_COUNTERS = {
    "reflector_kernel_calls": 0,
    "reflector_fallback_calls": 0,
    "compact_panel_kernel_calls": 0,
    "compact_panel_fallback_calls": 0,
}


def _tpu_kernel_debug_enabled() -> bool:
    return os.environ.get("JAX_GPTQ_TPU_KERNEL_DEBUG", "0") == "1"


def _bump_tpu_kernel_counter(name: str) -> None:
    if _tpu_kernel_debug_enabled():
        _TPU_KERNEL_DEBUG_COUNTERS[name] += 1


def reset_tpu_kernel_debug_counters() -> None:
    for name in _TPU_KERNEL_DEBUG_COUNTERS:
        _TPU_KERNEL_DEBUG_COUNTERS[name] = 0


def get_tpu_kernel_debug_counters() -> dict[str, int]:
    return dict(_TPU_KERNEL_DEBUG_COUNTERS)


def _dense_reflector_update(v: jnp.ndarray, tau: jnp.ndarray, block: jnp.ndarray) -> jnp.ndarray:
    w = tau * (v @ block)
    return block - jnp.outer(v, w)


def _dense_compact_panel_update(panel: CompactPanel, block: jnp.ndarray) -> jnp.ndarray:
    w = panel.y.T @ block
    w = panel.t.T @ w
    return block - panel.y @ w


def _round_up_to_multiple(n: int, multiple: int) -> int:
    if n <= 0:
        return 0
    return ((n + multiple - 1) // multiple) * multiple


def _compact_panel_tpu_tile_cols() -> int:
    raw = os.environ.get("JAX_GPTQ_TPU_COMPACT_PANEL_TILE_COLS", "512")
    try:
        tile_cols = int(raw)
    except ValueError:
        tile_cols = 512
    return max(128, _round_up_to_multiple(tile_cols, 128))


def tpu_reflector_kernel_supported_shape(block_shape: tuple[int, int]) -> bool:
    m, n = block_shape
    if m <= 0 or n <= 0:
        return False
    # Conservative TPU rule:
    # - allow both row and width dimensions to be padded locally before the
    #   kernel
    return True


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
    # Conservative TPU rule:
    # - keep the row dimension native and aligned to 8
    # - allow panel width / block width to be padded locally before the kernel
    return panel_rows % 8 == 0


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
        _bump_tpu_kernel_counter("reflector_fallback_calls")
        return _dense_reflector_update(v, tau, block)

    if block.dtype != jnp.float32:
        _bump_tpu_kernel_counter("reflector_fallback_calls")
        return _dense_reflector_update(v, tau, block)

    if not tpu_reflector_kernel_supported_shape(block.shape):
        _bump_tpu_kernel_counter("reflector_fallback_calls")
        return _dense_reflector_update(v, tau, block)

    m, n = block.shape
    padded_m = _round_up_to_multiple(m, 8)
    padded_n = _round_up_to_multiple(n, 128)
    tau_buf = tau.reshape(1)
    v_padded = v
    block_padded = block
    if padded_m != m:
        v_padded = jnp.pad(v, ((0, padded_m - m),))
        block_padded = jnp.pad(block_padded, ((0, padded_m - m), (0, 0)))
    if padded_n != n:
        block_padded = jnp.pad(block_padded, ((0, 0), (0, padded_n - n)))

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

    out_shape = jax.ShapeDtypeStruct((padded_m, padded_n), block.dtype)
    block_spec = pl.BlockSpec(
        index_map=lambda: (0, 0),
        block_shape=(padded_m, padded_n),
    )
    full_v_spec = pl.BlockSpec(
        index_map=lambda: (0,),
        block_shape=(padded_m,),
    )
    tau_spec = pl.BlockSpec(
        index_map=lambda: (0,),
        block_shape=(1,),
    )

    _bump_tpu_kernel_counter("reflector_kernel_calls")
    updated_padded = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[block_spec, full_v_spec, tau_spec],
        out_specs=block_spec,
    )(block_padded, v_padded, tau_buf)
    return updated_padded[:m, :n]


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
        _bump_tpu_kernel_counter("compact_panel_fallback_calls")
        return _dense_compact_panel_update(panel, block)

    if block.dtype != jnp.float32:
        _bump_tpu_kernel_counter("compact_panel_fallback_calls")
        return _dense_compact_panel_update(panel, block)

    if not tpu_compact_panel_kernel_supported_shape(panel.y.shape, block.shape):
        _bump_tpu_kernel_counter("compact_panel_fallback_calls")
        return _dense_compact_panel_update(panel, block)

    m, n = block.shape
    b = panel.y.shape[1]
    padded_b = _round_up_to_multiple(b, 8)
    tile_cols = _compact_panel_tpu_tile_cols()

    y = panel.y
    t = panel.t
    if padded_b != b:
        y = jnp.pad(y, ((0, 0), (0, padded_b - b)))
        t = jnp.pad(t, ((0, padded_b - b), (0, padded_b - b)))

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

    full_y_spec = pl.BlockSpec(
        index_map=lambda: (0, 0),
        block_shape=(m, padded_b),
    )
    full_t_spec = pl.BlockSpec(
        index_map=lambda: (0, 0),
        block_shape=(padded_b, padded_b),
    )

    updated_tiles = []
    for col_start in range(0, n, tile_cols):
        col_stop = min(col_start + tile_cols, n)
        block_tile = block[:, col_start:col_stop]
        tile_n = block_tile.shape[1]
        padded_n = _round_up_to_multiple(tile_n, 128)
        block_padded = block_tile
        if padded_n != tile_n:
            block_padded = jnp.pad(block_tile, ((0, 0), (0, padded_n - tile_n)))

        out_shape = jax.ShapeDtypeStruct((m, padded_n), block.dtype)
        block_spec = pl.BlockSpec(
            index_map=lambda: (0, 0),
            block_shape=(m, padded_n),
        )

        _bump_tpu_kernel_counter("compact_panel_kernel_calls")
        updated_padded = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=[block_spec, full_y_spec, full_t_spec],
            out_specs=block_spec,
        )(block_padded, y, t)
        updated_tiles.append(updated_padded[:, :tile_n])

    return jnp.concatenate(updated_tiles, axis=1)
