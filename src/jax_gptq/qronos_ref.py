import jax.numpy as jnp

from .config import QuantConfig
from .quant import quantize_dequantize


def qronos_single_layer_update_ref(
    weight: jnp.ndarray,
    weight_orig: jnp.ndarray,
    h: jnp.ndarray,
    g: jnp.ndarray,
    cfg: QuantConfig,
) -> jnp.ndarray:
    """
    Minimal reference placeholder.
    Current behavior: returns quantize/dequantize(weight_orig).
    """
    if weight.ndim != 2 or weight_orig.ndim != 2:
        raise ValueError("weight and weight_orig must be 2D")
    if weight.shape != weight_orig.shape:
        raise ValueError("weight and weight_orig shape mismatch")
    if h.shape[0] != h.shape[1] or g.shape != h.shape:
        raise ValueError("h must be square and g must match h shape")
    _ = (weight, h, g)
    return quantize_dequantize(weight_orig, cfg)
