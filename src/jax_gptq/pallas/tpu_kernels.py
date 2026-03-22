from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from .blocked_pivoted_qr import CompactPanel


def apply_reflector_to_block_pallas_tpu(
    v: jnp.ndarray,
    tau: jnp.ndarray,
    block: jnp.ndarray,
) -> jnp.ndarray:
    """
    TPU-backed Householder block update stub.

    Shared QR orchestration should call this through a backend-dispatching
    wrapper once a TPU implementation exists.
    """
    raise NotImplementedError("TPU Householder block update kernel is not implemented yet")


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
