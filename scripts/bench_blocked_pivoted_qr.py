import argparse
import time

import jax
import jax.numpy as jnp

from jax_gptq.pallas.blocked_pivoted_qr import blocked_pivoted_qr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--pivot-mode", choices=("largest", "smallest"), default="largest")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    a = jax.random.normal(key, (args.rows, args.cols), dtype=jnp.float32)

    t0 = time.time()
    work, perm = blocked_pivoted_qr(
        a,
        panel_size=args.panel_size,
        pivot_mode=args.pivot_mode,
    )
    work.block_until_ready()
    elapsed = time.time() - t0

    limit = min(args.rows, args.cols)
    lower_norm = jnp.linalg.norm(jnp.tril(work[:, :limit], k=-1))
    diag_abs = jnp.abs(jnp.diag(work[:limit, :limit]))

    print("shape:", a.shape)
    print("panel_size:", args.panel_size)
    print("pivot_mode:", args.pivot_mode)
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("elapsed_sec:", elapsed)
    print("perm_shape:", perm.shape)
    print("strictly_lower_norm:", float(lower_norm))
    print("diag_head:", diag_abs[:10])


if __name__ == "__main__":
    main()
