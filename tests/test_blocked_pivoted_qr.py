import jax.numpy as jnp

from jax_gptq.pallas.blocked_pivoted_qr import blocked_pivoted_qr


def _strictly_lower_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(jnp.tril(x, k=-1))


def test_blocked_pivoted_qr_returns_upper_triangular_work_largest() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    work, perm = blocked_pivoted_qr(a, panel_size=2, pivot_mode="largest")

    assert perm.shape == (a.shape[1],)
    assert float(_strictly_lower_norm(work)) < 1e-5


def test_blocked_pivoted_qr_returns_upper_triangular_work_smallest() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    work, perm = blocked_pivoted_qr(a, panel_size=2, pivot_mode="smallest")

    assert perm.shape == (a.shape[1],)
    assert float(_strictly_lower_norm(work)) < 1e-5


def test_largest_mode_diagonal_is_nonincreasing_in_magnitude() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    work, _ = blocked_pivoted_qr(a, panel_size=2, pivot_mode="largest")
    diag_abs = jnp.abs(jnp.diag(work))
    assert jnp.all(diag_abs[:-1] >= diag_abs[1:])


def test_smallest_mode_diagonal_is_nondecreasing_in_magnitude() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    work, _ = blocked_pivoted_qr(a, panel_size=2, pivot_mode="smallest")
    diag_abs = jnp.abs(jnp.diag(work))
    assert jnp.all(diag_abs[:-1] <= diag_abs[1:])
