import jax.numpy as jnp

from jax_gptq.config import QuantConfig
from jax_gptq.gptq_ref import gptq_forward_step
from jax_gptq.stats import h_g_from_activations


def main():
    cfg = QuantConfig(w_bits=4, group_size=4, sym=False, rank_eps=1e-5)
    w = jnp.arange(0.0, 32.0, dtype=jnp.float32).reshape(4, 8)
    xq = jnp.arange(0.0, 80.0, dtype=jnp.float32).reshape(10, 8)
    xf = xq + 0.01
    h, g = h_g_from_activations(xq, xf)
    perm = jnp.arange(w.shape[1], dtype=jnp.int32)
    out = gptq_forward_step(w, h_inv_sqrt=h, perm=perm, cfg=cfg)
    print("ok", out.shape, h.shape, g.shape)


if __name__ == "__main__":
    main()
