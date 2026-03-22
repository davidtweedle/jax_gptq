# jax_gptq

TPU-focused JAX implementation of GPTQ-style quantization and Qronos-style updates.

## Installation

Choose the JAX backend explicitly when installing. The project does not depend
on bare `jaxlib` anymore because that can silently land on a CPU-only setup on
GPU VMs.

```bash
# CPU-only
pip install -e ".[cpu,test]"

# NVIDIA GPU with CUDA 12-compatible driver/runtime
pip install -e ".[cuda12,test]"

# NVIDIA GPU with CUDA 13-compatible driver/runtime
pip install -e ".[cuda13,test]"
```

Verify JAX picked the expected backend:

```bash
python - <<'PY'
import jax
print("default_backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

## Initial Scope

- Pure JAX reference implementations first
- Faithful algorithm path before optimizations
- Pallas kernels added only after reference parity is stable

## Package Layout

- `src/jax_gptq/config.py`: shared configs
- `src/jax_gptq/quant.py`: weight quant/dequant helpers
- `src/jax_gptq/stats.py`: H/G statistics
- `src/jax_gptq/gptq_ref.py`: reference GPTQ forward step
- `src/jax_gptq/qronos_ref.py`: reference Qronos update
- `scripts/`: runnable entry points
