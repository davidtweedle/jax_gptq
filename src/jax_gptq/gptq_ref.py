import jax.numpy as jnp

from .config import QuantConfig
from .quant import quantize_dequantize


def gptq_forward_step(
    weight: jnp.ndarray,
    h_inv_sqrt: jnp.ndarray,
    perm: jnp.ndarray,
    cfg: QuantConfig,
) -> jnp.ndarray:
    """
    Minimal reference placeholder.
    Current behavior: apply permutation, quantize/dequantize, invert permutation.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got {weight.shape}")
    if perm.ndim != 1 or perm.shape[0] != weight.shape[1]:
        raise ValueError(f"perm must be shape ({weight.shape[1]},), got {perm.shape}")
    _ = h_inv_sqrt

    w_perm = weight[:, perm]
    w_q = quantize_dequantize(w_perm, cfg)
    inv_perm = jnp.argsort(perm)
    return w_q[:, inv_perm]
