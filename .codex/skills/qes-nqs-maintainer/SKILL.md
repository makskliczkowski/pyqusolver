---
name: qes-nqs-maintainer
description: Use when modifying QES NQS, VMC, TDVP, or network code in this repository and you need to preserve the existing coding style, comments, physics conventions, and backward-compatible APIs.
---

# QES NQS Maintainer

Follow this skill when changing code under `Python/QES/NQS`, `Python/QES/Solver/MonteCarlo`, `Python/QES/general_python/ml`, or nearby algebra/lattice modules.

## Non-negotiables

- Keep existing comments unless they are factually wrong after the change.
- Do not add one-off code paths, debug-only knobs, or tutorial-only branches to production modules.
- Preserve the repository's coding style: explicit names, moderate inline documentation, and physics-facing terminology.
- Prefer extending existing abstractions over introducing parallel APIs.
- Do not silently change physics conventions, normalization, parameter ordering, or state encoding.
- Treat symmetry handling, phase tracking, and dtype behavior as compatibility-sensitive.

## Safe workflow

1. Inspect the current call graph before editing.
2. Check whether the same concept already exists in `nqs.py`, `tdvp.py`, `vmc.py`, `nqs_kernels.py`, or `net_utils_jax.py`.
3. Patch the narrowest layer that fixes the issue.
4. Add or update a focused test/example for the exact behavior you changed.
5. Run a targeted validation command before finishing.

## Design preferences

- For performance work, optimize existing hot loops first: sampler kernels, gradient flattening, TDVP solver dispatch, and batched network evaluation.
- For new physics features, prefer helper modules in `Python/QES/NQS/src/` and expose thin methods from `Python/QES/NQS/nqs.py`.
- Reuse Hamiltonian/operator/lattice interfaces from `QES.Algebra` rather than duplicating physics logic in NQS.
- Keep public APIs composable: pass explicit kwargs, return structured data, and avoid hidden global state.

## Validation checklist

- `python -m py_compile` for edited Python files.
- Relevant `pytest` targets for the touched area.
- For TDVP/VMC changes, include at least one run that exercises JAX paths.
