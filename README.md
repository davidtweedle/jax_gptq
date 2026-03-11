# jax_gptq

TPU-focused JAX implementation of GPTQ-style quantization and Qronos-style updates.

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
