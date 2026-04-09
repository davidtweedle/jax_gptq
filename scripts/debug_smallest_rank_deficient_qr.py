import argparse
import os

import jax
import jax.numpy as jnp

from jax_gptq.pallas.blocked_pivoted_qr import (
    ZERO_TOL,
    apply_panel_to_trailing_pallas,
    factor_panel,
    factor_panel_pallas,
    initialize_trailing_norm_metadata,
    refresh_trailing_norm_metadata,
)


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


def _positive_diag_count(work: jnp.ndarray, tol: float) -> int:
    return int(jnp.sum(jnp.abs(jnp.diag(work)) > tol))


def _diag_window(work: jnp.ndarray, start: int, width: int = 8) -> jnp.ndarray:
    diag = jnp.abs(jnp.diag(work))
    return diag[start : start + width]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=384)
    parser.add_argument("--panel-size", type=int, default=128)
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
    limit = min(a.shape[0], a.shape[1])

    ref_work = a
    ref_perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    ref_norms = initialize_trailing_norm_metadata(a)

    cmp_work = a
    cmp_perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    cmp_norms = initialize_trailing_norm_metadata(a)

    prev_timing = os.environ.get("JAX_GPTQ_QR_TIMING")
    os.environ["JAX_GPTQ_QR_TIMING"] = "1"

    try:
        for panel_idx, k in enumerate(range(0, limit, args.panel_size), start=1):
            ref_a, ref_perm, ref_norms_after_panel, _, ref_panel = factor_panel(
                a=ref_work,
                perm=ref_perm,
                norms=ref_norms,
                k=k,
                panel_size=args.panel_size,
                pivot_mode="smallest",
            )
            ref_work = apply_panel_to_trailing_pallas(
                ref_a,
                ref_panel,
                ref_panel.panel_end,
            )
            ref_norms = refresh_trailing_norm_metadata(ref_work, ref_panel.panel_end)

            os.environ["JAX_GPTQ_QR_TIMING"] = "0"
            cmp_result = factor_panel_pallas(
                a=cmp_work,
                perm=cmp_perm,
                norms=cmp_norms,
                k=k,
                panel_size=args.panel_size,
                pivot_mode="smallest",
            )
            cmp_work = apply_panel_to_trailing_pallas(
                cmp_result.a,
                cmp_result.panel,
                cmp_result.panel.panel_end,
            )
            cmp_perm = cmp_result.perm
            cmp_norms = refresh_trailing_norm_metadata(cmp_work, cmp_result.panel.panel_end)
            os.environ["JAX_GPTQ_QR_TIMING"] = "1"

            ref_count = _positive_diag_count(ref_work, args.diag_tol)
            cmp_count = _positive_diag_count(cmp_work, args.diag_tol)
            print(f"panel {panel_idx} k={k}")
            print(
                "  panel_end:",
                int(ref_panel.panel_end),
                int(cmp_result.panel.panel_end),
            )
            print("  positive_diag_count:", ref_count, cmp_count)
            print(
                "  diag_window_ref:",
                _diag_window(ref_work, k),
            )
            print(
                "  diag_window_cmp:",
                _diag_window(cmp_work, k),
            )
            print(
                "  norms_max_ref_cmp:",
                float(jnp.max(ref_norms)),
                float(jnp.max(cmp_norms)),
            )
            print(
                "  work_close:",
                bool(jnp.allclose(ref_work, cmp_work, atol=1e-4)),
            )
            print()
    finally:
        if prev_timing is None:
            os.environ.pop("JAX_GPTQ_QR_TIMING", None)
        else:
            os.environ["JAX_GPTQ_QR_TIMING"] = prev_timing


if __name__ == "__main__":
    main()
