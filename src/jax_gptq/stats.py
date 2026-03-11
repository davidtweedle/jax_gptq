import jax.numpy as jnp


def h_g_from_activations(
    x_quant: jnp.ndarray,
    x_float: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if x_quant.ndim != 2:
        raise ValueError(f"x_quant must be 2D [tokens, in_features], got {x_quant.shape}")
    if x_float is None:
        x_float = x_quant
    if x_float.shape != x_quant.shape:
        raise ValueError(f"x_float shape must match x_quant, got {x_float.shape} vs {x_quant.shape}")

    # Convention aligned to row-major token layout [tokens, in_features].
    h = x_quant.T @ x_quant
    g = x_float.T @ x_quant
    return h, g
