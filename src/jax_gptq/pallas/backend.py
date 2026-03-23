from __future__ import annotations

import os

import jax


VALID_KERNEL_BACKENDS = {"auto", "gpu", "tpu", "reference"}


def pallas_enabled() -> bool:
    return os.environ.get("JAX_GPTQ_USE_PALLAS", "0") == "1"


def requested_kernel_backend() -> str:
    backend = os.environ.get("JAX_GPTQ_KERNEL_BACKEND", "auto").lower()
    if backend not in VALID_KERNEL_BACKENDS:
        raise ValueError(
            "JAX_GPTQ_KERNEL_BACKEND must be one of "
            f"{sorted(VALID_KERNEL_BACKENDS)}, got {backend!r}"
        )
    return backend


def runtime_backend() -> str:
    return jax.default_backend()


def selected_kernel_backend() -> str:
    requested = requested_kernel_backend()
    if requested != "auto":
        return requested

    runtime = runtime_backend()
    if runtime == "gpu":
        return "gpu"
    if runtime == "tpu":
        return "tpu"
    return "reference"


def describe_backend_selection() -> dict[str, object]:
    return {
        "jax_backend": runtime_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "pallas_enabled": pallas_enabled(),
        "requested_kernel_backend": requested_kernel_backend(),
        "selected_kernel_backend": selected_kernel_backend(),
    }
