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


def tpu_reflector_kernel_supported_shape(block_shape: tuple[int, int]) -> bool:
    m, n = block_shape
    if m <= 0 or n <= 0:
        return False
    # Conservative first pass for TPU Pallas:
    # - keep the row dimension fully materialized
    # - require the trailing tile width to be either 1 or a multiple of 128
    # - require the row dimension to be a multiple of 8, matching TPU block-shape guidance
    return (m % 8 == 0) and (n == 1 or n % 128 == 0)


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

    if not tpu_reflector_kernel_supported_shape(block.shape):
        return _dense_reflector_update(v, tau, block)

    m, n = block.shape
    tau_buf = tau.reshape(1)

    def kernel(block_ref, v_ref, tau_ref, out_ref):
        block_tile = block_ref[:, :]
        v_local = v_ref[:]
        tau_local = jnp.squeeze(tau_ref[:], axis=0)
        w = tau_local * jnp.dot(
            v_local,
            block_tile,
            precision=lax.Precision.HIGHEST,
        )
        updated = block_tile - v_local[:, None] * w[None, :]
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
    TPU-backed compact WY panel application stub.

    Shared QR orchestration should call this through a backend-dispatching
    wrapper once a TPU implementation exists.
    """
    raise NotImplementedError("TPU compact panel update kernel is not implemented yet")
