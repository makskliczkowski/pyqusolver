---
name: qes-nqs-performance
description: Use when profiling or optimizing QES NQS performance, especially for VMC sampling, TDVP solves, network evaluation, and JAX utility kernels.
---

# QES NQS Performance

Use this skill for performance work in:

- `Python/QES/Solver/MonteCarlo/vmc.py`
- `Python/QES/NQS/src/tdvp.py`
- `Python/QES/NQS/src/nqs_kernels.py`
- `Python/QES/general_python/ml/net_impl/utils/net_utils_jax.py`
- `Python/QES/general_python/ml/net_impl/networks/*`

## Priority order

1. Remove avoidable recomputation inside JITted loops.
2. Keep solver/sampler cache keys valid when static inputs change.
3. Prefer matrix-free TDVP paths over explicit Fisher formation when possible.
4. Reuse batched network apply paths before adding new vectorization layers.
5. Only add configurability when it changes steady-state performance or memory usage.

## Hotspots

- VMC: proposal loop, `log_psi_delta` fast updates, cache refresh policy, sampler recompilation triggers.
- TDVP: solver-form selection, batched Jacobian reuse, warm starts, covariance materialization.
- JAX utils: batch creation, flatten/unflatten transforms, per-sample gradient materialization.
- Networks: native batched apply, analytic gradients, fast-update support for sampler-compatible ansatze.

## Guardrails

- Benchmark the exact path you changed; do not infer wins from unrelated workloads.
- Preserve numerical stability and phase correctness when changing dtypes or accumulation strategy.
- Prefer small structural fixes over invasive rewrites unless the bottleneck is proven.
- Keep fallbacks working for both JAX and NumPy code paths unless the module is explicitly JAX-only.
