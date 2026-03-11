import jax.numpy as jnp

from .config import QuantConfig


def quantize_dequantize(w: jnp.ndarray, cfg: QuantConfig) -> jnp.ndarray:
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape={w.shape}")
    if cfg.group_size <= 0 or (w.shape[1] % cfg.group_size) != 0:
        raise ValueError("group_size must be positive and divide in_features exactly")

    if cfg.sym:
        qmax = float((1 << (cfg.w_bits - 1)) - 1)
        qmin = -qmax
    else:
        qmax = float((1 << cfg.w_bits) - 1)
        qmin = 0.0

    m, n = w.shape
    g = cfg.group_size
    w3 = jnp.reshape(w, (m, n // g, g))

    if cfg.sym:
        max_abs = jnp.max(jnp.abs(w3), axis=-1, keepdims=True)
        scale = jnp.maximum(max_abs / qmax, 1e-8)
        zero = jnp.zeros_like(scale)
    else:
        mn = jnp.min(w3, axis=-1, keepdims=True)
        mx = jnp.max(w3, axis=-1, keepdims=True)
        scale = jnp.maximum((mx - mn) / qmax, 1e-8)
        zero = jnp.clip(jnp.round(-mn / scale), 0.0, qmax)

    scale_exp = jnp.reshape(jnp.repeat(scale, g, axis=1), (m, n))
    zero_exp = jnp.reshape(jnp.repeat(zero, g, axis=1), (m, n))

    q = jnp.clip(jnp.round(w / scale_exp + zero_exp), qmin, qmax)
    return (q - zero_exp) * scale_exp
