import jax.numpy as jnp

import os

from jax_gptq.pallas.gpu_kernels import (
    compact_panel_kernel_supported_shape,
    reflector_kernel_supported_shape,
)
from jax_gptq.pallas.tpu_kernels import tpu_reflector_kernel_supported_shape
from jax_gptq.pallas.tpu_kernels import tpu_compact_panel_kernel_supported_shape
from jax_gptq.pallas.blocked_pivoted_qr import (
    apply_compact_panel_to_block,
    apply_compact_panel_to_block_pallas,
    apply_panel_to_trailing,
    apply_reflectors_to_column,
    apply_reflectors_to_trailing_view,
    apply_reflector_to_block,
    apply_reflector_to_block_pallas,
    apply_panel_to_trailing_pallas,
    append_reflector_to_panel_state,
    blocked_pivoted_qr,
    build_compact_panel,
    compute_exposed_trailing_row_from_compact_panel,
    compute_exposed_trailing_row,
    factor_panel,
    factor_panel_pallas,
    init_panel_state,
)


def _strictly_lower_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(jnp.tril(x, k=-1))


def _backend_atol(default: float, tpu: float) -> float:
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu":
        return tpu
    return default


def _form_q_from_reflectors(m: int, reflectors) -> jnp.ndarray:
    q = jnp.eye(m, dtype=jnp.float32)
    for j, v, tau in reversed(reflectors):
        updated = apply_reflector_to_block(v, tau, q[j:, :])
        q = q.at[j:, :].set(updated)
    return q


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


def test_factor_panel_reconstructs_pivoted_input() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    work, perm, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=min(a.shape),
        pivot_mode="largest",
    )

    q = _form_q_from_reflectors(a.shape[0], reflectors)
    r = jnp.triu(work)
    lhs = a[:, perm]
    rhs = q @ r
    assert jnp.allclose(lhs, rhs, atol=_backend_atol(1e-4, 3e-2))


def test_apply_reflectors_to_column_matches_trailing_view_column() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    work, perm, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="largest",
    )

    start_col = 2
    trailing_view = apply_reflectors_to_trailing_view(work, reflectors, start_col)
    expected = trailing_view[:, 0]
    actual = apply_reflectors_to_column(work[:, start_col], start_col, reflectors)
    assert jnp.allclose(actual, expected, atol=1e-4)


def test_compute_exposed_trailing_row_matches_trailing_view_row() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    work, perm, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="largest",
    )

    start_col = 2
    row_index = 1
    trailing_view = apply_reflectors_to_trailing_view(work, reflectors, start_col)
    expected = trailing_view[row_index, :]
    actual = compute_exposed_trailing_row(work, reflectors, row_index, start_col)
    assert jnp.allclose(actual, expected, atol=1e-4)


def test_build_compact_panel_packs_reflectors_in_global_rows() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    _, _, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="largest",
    )

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    assert panel.y.shape == (a.shape[0], 2)
    assert panel.tau.shape == (2,)

    for idx, (j, v, tau_j) in enumerate(reflectors):
        expected = jnp.zeros((a.shape[0],), dtype=a.dtype).at[j:].set(v)
        assert jnp.allclose(panel.y[:, idx], expected, atol=1e-6)
        assert jnp.allclose(panel.tau[idx], tau_j, atol=1e-6)


def test_build_compact_panel_t_is_upper_triangular_with_tau_on_diagonal() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    _, _, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="largest",
    )

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    assert jnp.allclose(panel.t, jnp.triu(panel.t), atol=1e-6)
    assert jnp.allclose(jnp.diag(panel.t), panel.tau, atol=1e-6)


def test_append_reflector_to_panel_state_matches_build_compact_panel() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    _, _, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="largest",
    )

    state = init_panel_state(a=a, perm=perm, norms=norms, k=0, panel_size=2)
    for j, v, tau_j in reflectors:
        state = append_reflector_to_panel_state(state, j, v, tau_j)

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    assert jnp.allclose(state.y, panel.y, atol=1e-6)
    assert jnp.allclose(state.tau, panel.tau, atol=1e-6)
    assert jnp.allclose(state.t, panel.t, atol=1e-6)


def test_apply_compact_panel_matches_reflector_replay_on_block() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    _, _, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="largest",
    )

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    block = a[:, 2:]

    expected = block
    for j, v, tau in reflectors:
        updated = apply_reflector_to_block(v, tau, expected[j:, :])
        expected = expected.at[j:, :].set(updated)

    actual = apply_compact_panel_to_block(panel, block)
    assert jnp.allclose(actual, expected, atol=_backend_atol(1e-4, 3e-2))


def test_apply_compact_panel_to_block_pallas_matches_reference_supported_shape() -> None:
    panel = build_compact_panel(
        reflectors=[
            (
                0,
                jnp.array([1.0, 0.25, -0.5, 0.75], dtype=jnp.float32),
                jnp.array(0.8, dtype=jnp.float32),
            ),
            (
                1,
                jnp.array([1.0, -0.25, 0.5], dtype=jnp.float32),
                jnp.array(0.6, dtype=jnp.float32),
            ),
        ],
        panel_start=0,
        panel_end=2,
        n_rows=4,
    )
    block = jnp.arange(16.0, dtype=jnp.float32).reshape(4, 4)

    assert compact_panel_kernel_supported_shape(panel.y.shape, block.shape)

    expected = apply_compact_panel_to_block(panel, block)
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_compact_panel_to_block_pallas_falls_back_on_unsupported_shape() -> None:
    panel = build_compact_panel(
        reflectors=[
            (
                0,
                jnp.array([1.0, 0.25, -0.5], dtype=jnp.float32),
                jnp.array(0.8, dtype=jnp.float32),
            ),
        ],
        panel_start=0,
        panel_end=1,
        n_rows=3,
    )
    block = jnp.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=jnp.float32,
    )

    assert not compact_panel_kernel_supported_shape(panel.y.shape, block.shape)

    expected = apply_compact_panel_to_block(panel, block)
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference() -> None:
    block = jnp.array(
        [
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 5.0, 1.0],
            [0.0, 0.0, 6.0],
        ],
        dtype=jnp.float32,
    )
    v = jnp.array([1.0, 0.25, -0.5, 0.75], dtype=jnp.float32)
    tau = jnp.array(0.8, dtype=jnp.float32)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference_4x4() -> None:
    if os.environ.get("JAX_GPTQ_USE_PALLAS") != "1":
        return
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") != "gpu":
        return
    assert reflector_kernel_supported_shape((4, 4))
    block = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    v = jnp.array([1.0, 0.25, -0.5, 0.75], dtype=jnp.float32)
    tau = jnp.array(0.8, dtype=jnp.float32)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference_8x8() -> None:
    if os.environ.get("JAX_GPTQ_USE_PALLAS") != "1":
        return
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") != "gpu":
        return
    assert reflector_kernel_supported_shape((8, 8))

    block = jnp.arange(64.0, dtype=jnp.float32).reshape(8, 8)
    v = jnp.array(
        [1.0, -0.5, 0.25, 0.75, -0.125, 0.5, -0.25, 0.375],
        dtype=jnp.float32,
    )
    tau = jnp.array(0.6, dtype=jnp.float32)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_falls_back_on_unsupported_shape() -> None:
    assert not reflector_kernel_supported_shape((3, 1))

    block = jnp.array(
        [
            [3.0],
            [4.0],
            [0.0],
        ],
        dtype=jnp.float32,
    )
    v = jnp.array([1.0, 0.25, -0.5], dtype=jnp.float32)
    tau = jnp.array(0.8, dtype=jnp.float32)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference_tpu_supported_shape() -> None:
    if os.environ.get("JAX_GPTQ_USE_PALLAS") != "1":
        return
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") != "tpu":
        return

    assert tpu_reflector_kernel_supported_shape((8, 128))

    block = jnp.arange(1024.0, dtype=jnp.float32).reshape(8, 128)
    v = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float32)
    tau = jnp.array(0.6, dtype=jnp.float32)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=5e-4)


def test_apply_reflector_to_block_pallas_falls_back_on_tpu_unsupported_shape() -> None:
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu":
        assert not tpu_reflector_kernel_supported_shape((8, 64))

    block = jnp.arange(512.0, dtype=jnp.float32).reshape(8, 64)
    v = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float32)
    tau = jnp.array(0.6, dtype=jnp.float32)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference_tpu_supported_shape_float64() -> None:
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu":
        assert tpu_reflector_kernel_supported_shape((8, 128))

    block = jnp.arange(1024.0, dtype=jnp.float64).reshape(8, 128)
    v = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float64)
    tau = jnp.array(0.6, dtype=jnp.float64)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-10)


def test_apply_compact_panel_to_block_pallas_matches_reference_tpu_supported_shape() -> None:
    if os.environ.get("JAX_GPTQ_USE_PALLAS") != "1":
        return
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") != "tpu":
        return

    y = jnp.eye(8, dtype=jnp.float32)
    tau = jnp.ones((8,), dtype=jnp.float32) * 0.1
    t = jnp.diag(tau)
    panel = build_compact_panel([], panel_start=0, panel_end=0, n_rows=8)
    panel = panel.__class__(panel_start=0, panel_end=8, y=y, tau=tau, t=t)
    block = jnp.arange(1024.0, dtype=jnp.float32).reshape(8, 128)

    assert tpu_compact_panel_kernel_supported_shape(panel.y.shape, block.shape)

    # With Y = I and T = 0.1 I, the compact update reduces to 0.9 * block.
    expected = 0.9 * block
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert jnp.allclose(actual, expected, atol=5e-4)


def test_apply_compact_panel_to_block_pallas_falls_back_on_tpu_unsupported_shape() -> None:
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu":
        assert not tpu_compact_panel_kernel_supported_shape((8, 8), (8, 64))

    y = jnp.eye(8, dtype=jnp.float32)
    tau = jnp.ones((8,), dtype=jnp.float32) * 0.1
    t = jnp.diag(tau)
    panel = build_compact_panel([], panel_start=0, panel_end=0, n_rows=8)
    panel = panel.__class__(panel_start=0, panel_end=8, y=y, tau=tau, t=t)
    block = jnp.arange(512.0, dtype=jnp.float32).reshape(8, 64)

    expected = apply_compact_panel_to_block(panel, block)
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_compact_panel_exposed_row_matches_exact_helper() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    work, _, _, reflectors, _ = factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="largest",
    )

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    start_col = 2
    row_index = 1
    trailing_block = work[:, start_col:]

    expected = compute_exposed_trailing_row(work, reflectors, row_index, start_col)
    actual = compute_exposed_trailing_row_from_compact_panel(panel, trailing_block, row_index)
    assert jnp.allclose(actual, expected, atol=_backend_atol(1e-4, 3e-2))


def test_factor_panel_pallas_matches_reference_factor_panel() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)

    expected = factor_panel(
        a=a, perm=perm, norms=norms, k=0, panel_size=2, pivot_mode="largest"
    )
    actual = factor_panel_pallas(
        a=a, perm=perm, norms=norms, k=0, panel_size=2, pivot_mode="largest"
    )

    for lhs, rhs in zip(expected[:3], (actual.a, actual.perm, actual.norms)):
        assert jnp.allclose(lhs, rhs, atol=1e-6)
    assert len(expected[3]) == len(actual.reflectors)
    for (j_lhs, v_lhs, tau_lhs), (j_rhs, v_rhs, tau_rhs) in zip(expected[3], actual.reflectors):
        assert j_lhs == j_rhs
        assert jnp.allclose(v_lhs, v_rhs, atol=1e-6)
        assert jnp.allclose(tau_lhs, tau_rhs, atol=1e-6)
    assert jnp.allclose(expected[4].y, actual.panel.y, atol=1e-6)
    assert jnp.allclose(expected[4].tau, actual.panel.tau, atol=1e-6)
    assert jnp.allclose(expected[4].t, actual.panel.t, atol=1e-6)


def test_apply_panel_to_trailing_pallas_matches_reference() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    work, _, _, _, panel = factor_panel(
        a=a, perm=perm, norms=norms, k=0, panel_size=2, pivot_mode="largest"
    )

    expected = apply_compact_panel_to_block(panel, work[:, 2:])
    actual = apply_panel_to_trailing_pallas(work, panel, 2)[:, 2:]
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_blocked_pivoted_qr_matches_driver_built_from_pallas_entry_points() -> None:
    a = jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    expected_work = a
    expected_perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    expected_norms = jnp.linalg.norm(a, axis=0)
    limit = min(a.shape[0], a.shape[1])
    panel_size = 2

    for k in range(0, limit, panel_size):
        result = factor_panel_pallas(
            a=expected_work,
            perm=expected_perm,
            norms=expected_norms,
            k=k,
            panel_size=panel_size,
            pivot_mode="largest",
        )
        expected_work = apply_panel_to_trailing_pallas(
            result.a,
            result.panel,
            result.panel.panel_end,
        )
        expected_perm = result.perm
        expected_norms = jnp.linalg.norm(expected_work[result.panel.panel_end :, :], axis=0)

    actual_work, actual_perm = blocked_pivoted_qr(a, panel_size=panel_size, pivot_mode="largest")

    assert jnp.allclose(actual_work, expected_work, atol=1e-6)
    assert jnp.array_equal(actual_perm, expected_perm)


def test_apply_panel_to_trailing_uses_realized_panel_end_for_early_stop() -> None:
    a = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)

    result = factor_panel_pallas(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=2,
        pivot_mode="smallest",
    )

    assert result.panel.panel_end == 0

    actual = apply_panel_to_trailing_pallas(result.a, result.panel, result.panel.panel_end)
    expected = apply_panel_to_trailing(result.a, result.panel, result.panel.panel_end)

    assert jnp.allclose(actual, expected, atol=1e-6)
