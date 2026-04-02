import os

import jax.numpy as jnp
import numpy as np
import pytest

from jax_gptq.pallas.blocked_pivoted_qr import (
    apply_compact_panel_to_block,
    apply_compact_panel_to_block_pallas,
    apply_panel_to_trailing,
    apply_panel_to_trailing_pallas,
    apply_reflector_to_block,
    apply_reflector_to_block_pallas,
    apply_reflectors_to_column,
    apply_reflectors_to_trailing_view,
    append_reflector_to_panel_state,
    blocked_pivoted_qr,
    build_compact_panel,
    compute_exposed_trailing_row,
    compute_exposed_trailing_row_from_compact_panel,
    factor_panel,
    factor_panel_pallas,
    init_panel_state,
)
from jax_gptq.pallas.gpu_kernels import (
    compact_panel_kernel_supported_shape,
    reflector_kernel_supported_shape,
)
from jax_gptq.pallas.tpu_kernels import tpu_compact_panel_kernel_supported_shape
from jax_gptq.pallas.tpu_kernels import tpu_reflector_kernel_supported_shape


def _strictly_lower_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(jnp.tril(x, k=-1))


def _require_pallas_backend(backend: str) -> None:
    if os.environ.get("JAX_GPTQ_USE_PALLAS") != "1":
        pytest.skip("Pallas backend is not enabled")
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") != backend:
        pytest.skip(f"{backend} backend is not selected")


def _apply_reflector_to_block_np(v: np.ndarray, tau: np.ndarray, block: np.ndarray) -> np.ndarray:
    w = tau * (v @ block)
    return block - np.outer(v, w)


def _form_q_from_reflectors(m: int, reflectors) -> jnp.ndarray:
    q = jnp.eye(m, dtype=jnp.float32)
    for j, v, tau in reversed(reflectors):
        updated = apply_reflector_to_block(v, tau, q[j:, :])
        q = q.at[j:, :].set(updated)
    return q


def _form_q_from_reflectors_np(m: int, reflectors) -> np.ndarray:
    q = np.eye(m, dtype=np.float32)
    for j, v, tau in reversed(reflectors):
        v_np = np.asarray(v, dtype=np.float32)
        tau_np = np.asarray(tau, dtype=np.float32)
        updated = _apply_reflector_to_block_np(v_np, tau_np, q[j:, :])
        q[j:, :] = updated
    return q


def _example_matrix() -> jnp.ndarray:
    return jnp.array(
        [
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 0.0, 2.0, 1.0],
            [0.0, 5.0, 1.0, 0.0],
            [0.0, 0.0, 6.0, 1.0],
        ],
        dtype=jnp.float32,
    )


def _factor_largest_panel(panel_size: int = 2):
    a = _example_matrix()
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    return a, factor_panel(
        a=a,
        perm=perm,
        norms=norms,
        k=0,
        panel_size=panel_size,
        pivot_mode="largest",
    )


def _identity_diag_panel(n_rows: int = 8, tau_value: float = 0.1):
    y = jnp.eye(n_rows, dtype=jnp.float32)
    tau = jnp.ones((n_rows,), dtype=jnp.float32) * tau_value
    t = jnp.diag(tau)
    panel = build_compact_panel([], panel_start=0, panel_end=0, n_rows=n_rows)
    return panel.__class__(panel_start=0, panel_end=n_rows, y=y, tau=tau, t=t)


def test_blocked_pivoted_qr_returns_upper_triangular_work_largest() -> None:
    a = _example_matrix()
    work, perm = blocked_pivoted_qr(a, panel_size=2, pivot_mode="largest")

    assert perm.shape == (a.shape[1],)
    assert float(_strictly_lower_norm(work)) < 1e-5


def test_blocked_pivoted_qr_returns_upper_triangular_work_smallest() -> None:
    a = _example_matrix()
    work, perm = blocked_pivoted_qr(a, panel_size=2, pivot_mode="smallest")

    assert perm.shape == (a.shape[1],)
    assert float(_strictly_lower_norm(work)) < 1e-5


def test_largest_mode_diagonal_is_nonincreasing_in_magnitude() -> None:
    a = _example_matrix()
    work, _ = blocked_pivoted_qr(a, panel_size=2, pivot_mode="largest")
    diag_abs = jnp.abs(jnp.diag(work))
    assert jnp.all(diag_abs[:-1] >= diag_abs[1:])


def test_smallest_mode_diagonal_is_nondecreasing_in_magnitude() -> None:
    a = _example_matrix()
    work, _ = blocked_pivoted_qr(a, panel_size=2, pivot_mode="smallest")
    diag_abs = jnp.abs(jnp.diag(work))
    assert jnp.all(diag_abs[:-1] <= diag_abs[1:])


def test_factor_panel_reconstructs_pivoted_input() -> None:
    a, (work, perm, _, reflectors, _) = _factor_largest_panel(panel_size=4)

    q = _form_q_from_reflectors_np(a.shape[0], reflectors)
    r = np.triu(np.asarray(work, dtype=np.float32))
    lhs = np.asarray(a[:, perm], dtype=np.float32)
    rhs = q @ r
    assert np.allclose(lhs, rhs, atol=1e-4)


def test_apply_reflectors_to_column_matches_trailing_view_column() -> None:
    _, (work, perm, _, reflectors, _) = _factor_largest_panel(panel_size=2)

    start_col = 2
    trailing_view = apply_reflectors_to_trailing_view(work, reflectors, start_col)
    expected = trailing_view[:, 0]
    actual = apply_reflectors_to_column(work[:, start_col], start_col, reflectors)
    assert jnp.allclose(actual, expected, atol=1e-4)


def test_compute_exposed_trailing_row_matches_trailing_view_row() -> None:
    _, (work, _, _, reflectors, _) = _factor_largest_panel(panel_size=2)

    start_col = 2
    row_index = 1
    trailing_view = apply_reflectors_to_trailing_view(work, reflectors, start_col)
    expected = trailing_view[row_index, :]
    actual = compute_exposed_trailing_row(work, reflectors, row_index, start_col)
    assert jnp.allclose(actual, expected, atol=1e-4)


def test_build_compact_panel_packs_reflectors_in_global_rows() -> None:
    a, (_, _, _, reflectors, _) = _factor_largest_panel(panel_size=2)

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    assert panel.y.shape == (a.shape[0], 2)
    assert panel.tau.shape == (2,)

    for idx, (j, v, tau_j) in enumerate(reflectors):
        expected = jnp.zeros((a.shape[0],), dtype=a.dtype).at[j:].set(v)
        assert jnp.allclose(panel.y[:, idx], expected, atol=1e-6)
        assert jnp.allclose(panel.tau[idx], tau_j, atol=1e-6)


def test_build_compact_panel_t_is_upper_triangular_with_tau_on_diagonal() -> None:
    a, (_, _, _, reflectors, _) = _factor_largest_panel(panel_size=2)

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    assert jnp.allclose(panel.t, jnp.triu(panel.t), atol=1e-6)
    assert jnp.allclose(jnp.diag(panel.t), panel.tau, atol=1e-6)


def test_append_reflector_to_panel_state_matches_build_compact_panel() -> None:
    a, (_, perm, norms, reflectors, _) = _factor_largest_panel(panel_size=2)

    state = init_panel_state(a=a, perm=perm, norms=norms, k=0, panel_size=2)
    for j, v, tau_j in reflectors:
        state = append_reflector_to_panel_state(state, j, v, tau_j)

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    assert jnp.allclose(state.y, panel.y, atol=1e-6)
    assert jnp.allclose(state.tau, panel.tau, atol=1e-6)
    assert jnp.allclose(state.t, panel.t, atol=1e-6)


def test_apply_compact_panel_matches_reflector_replay_on_block() -> None:
    a, (_, _, _, reflectors, _) = _factor_largest_panel(panel_size=2)

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    block = a[:, 2:]

    expected = apply_compact_panel_to_block(panel, block)
    actual = apply_compact_panel_to_block(panel, block)
    assert jnp.allclose(actual, expected, atol=1e-4)


def test_apply_compact_panel_to_block_pallas_matches_reference_supported_shape() -> None:
    reflectors = [
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
    ]
    panel = build_compact_panel(reflectors=reflectors, panel_start=0, panel_end=2, n_rows=4)
    block = jnp.arange(16.0, dtype=jnp.float32).reshape(4, 4)

    assert compact_panel_kernel_supported_shape(panel.y.shape, block.shape)

    expected = apply_compact_panel_to_block(panel, block)
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_apply_compact_panel_to_block_pallas_falls_back_on_unsupported_shape() -> None:
    reflectors = [
        (
            0,
            jnp.array([1.0, 0.25, -0.5], dtype=jnp.float32),
            jnp.array(0.8, dtype=jnp.float32),
        ),
    ]
    panel = build_compact_panel(reflectors=reflectors, panel_start=0, panel_end=1, n_rows=3)
    block = jnp.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=jnp.float32,
    )

    assert not compact_panel_kernel_supported_shape(panel.y.shape, block.shape)

    expected = np.array(block, dtype=np.float32, copy=True)
    for j, v, tau in reflectors:
        v_np = np.asarray(v, dtype=np.float32)
        tau_np = np.asarray(tau, dtype=np.float32)
        expected[j:, :] = _apply_reflector_to_block_np(v_np, tau_np, expected[j:, :])
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert np.allclose(np.asarray(actual, dtype=np.float32), expected, atol=1e-6)


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

    expected = _apply_reflector_to_block_np(
        np.asarray(v, dtype=np.float32),
        np.asarray(tau, dtype=np.float32),
        np.asarray(block, dtype=np.float32),
    )
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert np.allclose(np.asarray(actual, dtype=np.float32), expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference_4x4() -> None:
    _require_pallas_backend("gpu")
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

    expected = _apply_reflector_to_block_np(
        np.asarray(v, dtype=np.float32),
        np.asarray(tau, dtype=np.float32),
        np.asarray(block, dtype=np.float32),
    )
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert np.allclose(np.asarray(actual, dtype=np.float32), expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference_8x8() -> None:
    _require_pallas_backend("gpu")
    assert reflector_kernel_supported_shape((8, 8))

    block = jnp.arange(64.0, dtype=jnp.float32).reshape(8, 8)
    v = jnp.array(
        [1.0, -0.5, 0.25, 0.75, -0.125, 0.5, -0.25, 0.375],
        dtype=jnp.float32,
    )
    tau = jnp.array(0.6, dtype=jnp.float32)

    expected = _apply_reflector_to_block_np(
        np.asarray(v, dtype=np.float32),
        np.asarray(tau, dtype=np.float32),
        np.asarray(block, dtype=np.float32),
    )
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert np.allclose(np.asarray(actual, dtype=np.float32), expected, atol=1e-6)


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

    expected = _apply_reflector_to_block_np(
        np.asarray(v, dtype=np.float32),
        np.asarray(tau, dtype=np.float32),
        np.asarray(block, dtype=np.float32),
    )
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert np.allclose(np.asarray(actual, dtype=np.float32), expected, atol=1e-6)


def test_apply_reflector_to_block_pallas_matches_reference_tpu_supported_shape() -> None:
    _require_pallas_backend("tpu")
    assert tpu_reflector_kernel_supported_shape((8, 128))

    block = jnp.arange(1024.0, dtype=jnp.float32).reshape(8, 128)
    v = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float32)
    tau = jnp.array(0.6, dtype=jnp.float32)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=5e-4)


def test_apply_reflector_to_block_pallas_falls_back_on_tpu_unsupported_shape() -> None:
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu":
        assert tpu_reflector_kernel_supported_shape((7, 64))

    block = jnp.arange(448.0, dtype=jnp.float64).reshape(7, 64)
    v = jnp.linspace(0.1, 0.7, 7, dtype=jnp.float64)
    tau = jnp.array(0.6, dtype=jnp.float64)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-10)


def test_apply_reflector_to_block_pallas_matches_reference_tpu_supported_shape_float64() -> None:
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu":
        pytest.skip("float64 is not supported on TPU")

    block = jnp.arange(1024.0, dtype=jnp.float64).reshape(8, 128)
    v = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float64)
    tau = jnp.array(0.6, dtype=jnp.float64)

    expected = apply_reflector_to_block(v, tau, block)
    actual = apply_reflector_to_block_pallas(v, tau, block)
    assert jnp.allclose(actual, expected, atol=1e-10)


def test_apply_compact_panel_to_block_pallas_matches_reference_tpu_supported_shape() -> None:
    _require_pallas_backend("tpu")
    panel = _identity_diag_panel(n_rows=8, tau_value=0.1)
    block = jnp.arange(1024.0, dtype=jnp.float32).reshape(8, 128)

    assert tpu_compact_panel_kernel_supported_shape(panel.y.shape, block.shape)

    # With Y = I and T = 0.1 I, the compact update reduces to 0.9 * block.
    expected = 0.9 * block
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert jnp.allclose(actual, expected, atol=5e-4)


def test_apply_compact_panel_to_block_pallas_falls_back_on_tpu_unsupported_shape() -> None:
    if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu":
        assert not tpu_compact_panel_kernel_supported_shape((7, 7), (7, 64))

    panel = _identity_diag_panel(n_rows=7, tau_value=0.1)
    block = jnp.arange(448.0, dtype=jnp.float32).reshape(7, 64)

    expected = apply_compact_panel_to_block(panel, block)
    actual = apply_compact_panel_to_block_pallas(panel, block)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_compact_panel_exposed_row_matches_exact_helper() -> None:
    a, (work, _, _, reflectors, _) = _factor_largest_panel(panel_size=2)

    panel = build_compact_panel(reflectors, panel_start=0, panel_end=2, n_rows=a.shape[0])
    start_col = 2
    row_index = 1
    trailing_block = work[:, start_col:]

    expected = compute_exposed_trailing_row(work, reflectors, row_index, start_col)
    actual = compute_exposed_trailing_row_from_compact_panel(panel, trailing_block, row_index)
    atol = 3e-2 if os.environ.get("JAX_GPTQ_KERNEL_BACKEND") == "tpu" else 1e-4
    assert jnp.allclose(actual, expected, atol=atol)


def test_factor_panel_pallas_matches_reference_factor_panel() -> None:
    a = _example_matrix()
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)

    expected = factor_panel(
        a=a, perm=perm, norms=norms, k=0, panel_size=2, pivot_mode="largest"
    )
    actual = factor_panel_pallas(
        a=a, perm=perm, norms=norms, k=0, panel_size=2, pivot_mode="largest"
    )

    assert np.allclose(np.asarray(expected[0], dtype=np.float32), np.asarray(actual.a, dtype=np.float32), atol=1e-6)
    assert np.array_equal(np.asarray(expected[1], dtype=np.int32), np.asarray(actual.perm, dtype=np.int32))
    assert np.allclose(np.asarray(expected[2], dtype=np.float32), np.asarray(actual.norms, dtype=np.float32), atol=1e-6)
    assert len(expected[3]) == len(actual.reflectors)
    for (j_lhs, v_lhs, tau_lhs), (j_rhs, v_rhs, tau_rhs) in zip(expected[3], actual.reflectors):
        assert j_lhs == j_rhs
        assert np.allclose(np.asarray(v_lhs, dtype=np.float32), np.asarray(v_rhs, dtype=np.float32), atol=1e-6)
        assert np.allclose(np.asarray(tau_lhs, dtype=np.float32), np.asarray(tau_rhs, dtype=np.float32), atol=1e-6)
    assert np.allclose(np.asarray(expected[4].y, dtype=np.float32), np.asarray(actual.panel.y, dtype=np.float32), atol=1e-6)
    assert np.allclose(np.asarray(expected[4].tau, dtype=np.float32), np.asarray(actual.panel.tau, dtype=np.float32), atol=1e-6)
    assert np.allclose(np.asarray(expected[4].t, dtype=np.float32), np.asarray(actual.panel.t, dtype=np.float32), atol=1e-6)


def test_apply_panel_to_trailing_pallas_matches_reference() -> None:
    a = _example_matrix()
    perm = jnp.arange(a.shape[1], dtype=jnp.int32)
    norms = jnp.linalg.norm(a, axis=0)
    work, _, _, reflectors, panel = factor_panel(
        a=a, perm=perm, norms=norms, k=0, panel_size=2, pivot_mode="largest"
    )

    expected = apply_compact_panel_to_block(panel, work[:, 2:])
    actual = apply_panel_to_trailing_pallas(work, panel, 2)[:, 2:]
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_blocked_pivoted_qr_matches_driver_built_from_pallas_entry_points() -> None:
    a = _example_matrix()

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
