import argparse
import time

import jax
import jax.numpy as jnp

from jax_gptq.pallas.blocked_pivoted_qr import ZERO_TOL, blocked_pivoted_qr
from jax_gptq.pallas.tpu_kernels import reset_tpu_kernel_debug_counters


def _make_rank_deficient_inverse_svt(
    size: int,
    rank: int,
    seed: int,
    sigma_max: float,
    sigma_min: float,
) -> jnp.ndarray:
    if not 0 <= rank <= size:
        raise ValueError(f"rank must be in [0, {size}], got {rank}")
    if sigma_max <= 0 or sigma_min <= 0:
        raise ValueError("sigma_max and sigma_min must be positive")

    key = jax.random.PRNGKey(seed)
    gaussian = jax.random.normal(key, (size, size), dtype=jnp.float32)
    q, _ = jnp.linalg.qr(gaussian)

    if rank == 0:
        inv_sigma = jnp.zeros((size,), dtype=jnp.float32)
    else:
        kept_sigma = jnp.geomspace(
            jnp.array(sigma_max, dtype=jnp.float32),
            jnp.array(sigma_min, dtype=jnp.float32),
            rank,
        )
        kept_inv_sigma = 1.0 / kept_sigma
        inv_sigma = jnp.concatenate(
            [
                kept_inv_sigma,
                jnp.zeros((size - rank,), dtype=jnp.float32),
            ]
        )

    return inv_sigma[:, None] * q.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=3000)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--diag-tol", type=float, default=ZERO_TOL)
    args = parser.parse_args()

    a = _make_rank_deficient_inverse_svt(
        size=args.size,
        rank=args.rank,
        seed=args.seed,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
    )
    reset_tpu_kernel_debug_counters()

    t0 = time.time()
    work, perm = blocked_pivoted_qr(
        a,
        panel_size=args.panel_size,
        pivot_mode="smallest",
    )
    work.block_until_ready()
    elapsed = time.time() - t0

    diag_abs = jnp.abs(jnp.diag(work))
    positive_count = int(jnp.sum(diag_abs > args.diag_tol))
    lower_norm = jnp.linalg.norm(jnp.tril(work, k=-1))

    print("shape:", a.shape)
    print("target_rank:", args.rank)
    print("observed_positive_diagonal_count:", positive_count)
    print("panel_size:", args.panel_size)
    print("pivot_mode: smallest")
    print("sigma_range:", (args.sigma_max, args.sigma_min))
    print("diag_tol:", args.diag_tol)
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("elapsed_sec:", elapsed)
    print("perm_shape:", perm.shape)
    print("strictly_lower_norm:", float(lower_norm))
    print("diag_head:", diag_abs[:10])
    print("diag_tail:", diag_abs[-10:])


if __name__ == "__main__":
    main()
