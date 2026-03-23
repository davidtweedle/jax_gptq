import json
import os

import jax
import jax.numpy as jnp

from jax_gptq.pallas.backend import describe_backend_selection
from jax_gptq.pallas.blocked_pivoted_qr import (
    apply_reflector_to_block,
    apply_reflector_to_block_pallas,
)


def main() -> None:
    info = describe_backend_selection()
    info["env"] = {
        "JAX_GPTQ_USE_PALLAS": os.environ.get("JAX_GPTQ_USE_PALLAS"),
        "JAX_GPTQ_KERNEL_BACKEND": os.environ.get("JAX_GPTQ_KERNEL_BACKEND"),
    }

    v = jnp.array([1.0, 0.25, -0.5, 0.75], dtype=jnp.float32)
    tau = jnp.array(0.8, dtype=jnp.float32)
    block = jnp.ones((4, 8), dtype=jnp.float32)

    ref = apply_reflector_to_block(v, tau, block)
    out = apply_reflector_to_block_pallas(v, tau, block)
    info["reflector_path_dtype"] = str(out.dtype)
    info["reflector_path_shape"] = tuple(out.shape)
    info["reflector_max_abs_diff_vs_reference"] = float(jnp.max(jnp.abs(out - ref)))

    if hasattr(out, "device"):
        info["reflector_output_device"] = str(out.device)
    else:
        info["reflector_output_device"] = str(jax.devices()[0])

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
