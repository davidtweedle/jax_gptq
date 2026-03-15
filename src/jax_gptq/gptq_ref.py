import jax.numpy as jnp

from .config import QuantConfig
from .pallas.blocked_pivoted_qr import blocked_pivoted_qr
from .quant import quantize_dequantize


def _select_truncation_rank(eigvals_desc: jnp.ndarray, cfg: QuantConfig) -> int:
    """
    Select retained rank from descending eigenvalues according to the configured
    truncation rule.
    """
    if eigvals_desc.ndim != 1:
        raise ValueError(f"eigvals_desc must be 1D, got {eigvals_desc.shape}")

    n = eigvals_desc.shape[0]
    if n == 0:
        raise ValueError("eigvals_desc must be non-empty")

    mode = cfg.truncation_mode
    value = float(cfg.truncation_value)

    if mode == "abs":
        rank = int(jnp.sum(eigvals_desc > value))
    elif mode == "percent":
        drop_frac = min(max(value / 100.0, 0.0), 1.0)
        rank = int(jnp.ceil((1.0 - drop_frac) * n))
    elif mode == "mean_top32":
        ref_k = min(32, n)
        ref_val = jnp.mean(eigvals_desc[:ref_k])
        rank = int(jnp.sum(eigvals_desc > value * ref_val))
    elif mode == "energy":
        total = jnp.sum(eigvals_desc)
        target = (1.0 - value) * total
        rank = int(jnp.sum(jnp.cumsum(eigvals_desc) <= target))
        if rank < n:
            rank += 1
    else:
        raise ValueError(f"unsupported truncation_mode={mode}")

    return max(1, min(rank, n))


def process_hessian(
    h: jnp.ndarray,
    cfg: QuantConfig,
    panel_size: int = 32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reference Hessian processing for JAX GPTQ experiments.

    Returns:
    - `h_inv_sqrt`: upper-triangular error-propagation factor
    - `perm`: column order to use for GPTQ
    """
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"h must be square 2D, got {h.shape}")

    n = h.shape[0]

    if cfg.solver_mode == "chol":
        if cfg.ordering_mode not in {"none", "diag_h"}:
            raise ValueError(
                f"ordering_mode={cfg.ordering_mode} is not supported for solver_mode='chol'"
            )

        if cfg.ordering_mode == "diag_h":
            perm = jnp.argsort(jnp.diag(h))[::-1].astype(jnp.int32)
        else:
            perm = jnp.arange(n, dtype=jnp.int32)

        h_perm = h[perm][:, perm]
        diag = jnp.diag(h_perm)
        diag_mean = jnp.mean(diag)
        if float(diag_mean) == 0.0:
            diag_mean = jnp.array(1.0, dtype=h.dtype)

        h_inv_sqrt = None
        base_damp = cfg.damp_percent
        for damp_exp in range(5):
            try:
                damp = (10**damp_exp) * base_damp
                h_damped = h_perm + damp * diag_mean * jnp.eye(n, dtype=h.dtype)
                l = jnp.linalg.cholesky(h_damped)
                h_inv = jax.scipy.linalg.cho_solve((l, False), jnp.eye(n, dtype=h.dtype))
                h_inv_sqrt = jnp.linalg.cholesky(h_inv, upper=True)
                break
            except Exception:
                continue

        if h_inv_sqrt is None:
            h_inv_sqrt = jnp.eye(n, dtype=h.dtype)
        return h_inv_sqrt, perm

    if cfg.solver_mode != "eigh":
        raise ValueError(f"unsupported solver_mode={cfg.solver_mode}")

    eigvals, eigvecs = jnp.linalg.eigh(h)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    rank = _select_truncation_rank(eigvals, cfg)

    eigvals_kept = eigvals[:rank]
    eigvecs_kept = eigvecs[:, :rank]

    sqrt_h = jnp.sqrt(eigvals_kept)[:, None] * eigvecs_kept.T
    sqrt_hinv = jnp.power(eigvals_kept, -0.5)[:, None] * eigvecs_kept.T

    if cfg.ordering_mode == "none":
        perm = jnp.arange(n, dtype=jnp.int32)
        _, h_inv_sqrt = jnp.linalg.qr(sqrt_hinv.T[:, perm], mode="reduced")
        return h_inv_sqrt, perm

    if cfg.ordering_mode == "diag_h":
        perm = jnp.argsort(jnp.diag(h))[::-1].astype(jnp.int32)
        _, h_inv_sqrt = jnp.linalg.qr(sqrt_hinv.T[:, perm], mode="reduced")
        return h_inv_sqrt, perm

    if cfg.ordering_mode == "pivot_sqrt_h":
        _, perm = blocked_pivoted_qr(sqrt_h, panel_size=panel_size, pivot_mode="largest")
        _, h_inv_sqrt = jnp.linalg.qr(sqrt_hinv.T[:, perm], mode="reduced")
        return h_inv_sqrt, perm

    if cfg.ordering_mode == "pivot_sqrt_hinv_smallest":
        h_inv_sqrt, perm = blocked_pivoted_qr(
            sqrt_hinv,
            panel_size=panel_size,
            pivot_mode="smallest",
        )
        return h_inv_sqrt, perm

    raise ValueError(f"unsupported ordering_mode={cfg.ordering_mode}")


def gptq_forward_step(
    weight: jnp.ndarray,
    h_inv_sqrt: jnp.ndarray,
    perm: jnp.ndarray,
    cfg: QuantConfig,
) -> jnp.ndarray:
    """
    Minimal reference placeholder.
    Current behavior: apply permutation, quantize/dequantize, invert permutation.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got {weight.shape}")
    if perm.ndim != 1 or perm.shape[0] != weight.shape[1]:
        raise ValueError(f"perm must be shape ({weight.shape[1]},), got {perm.shape}")
    _ = h_inv_sqrt

    w_perm = weight[:, perm]
    w_q = quantize_dequantize(w_perm, cfg)
    inv_perm = jnp.argsort(perm)
    return w_q[:, inv_perm]
