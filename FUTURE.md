# FUTURE.md — QES Roadmap (requested items)

This roadmap captures requested medium/long-term work. Items that would require
changes under `/general_python` are marked **Concept-only (out of scope here)**.

## 1) Tensor networks / MPS

- Add a tensor-network layer (MPS first) with clear API boundaries vs `QES.NQS`.
- Support variational optimization and (later) time evolution paths.
- Define interoperability contracts for initializing NQS from TN states.

## 2) Improved optimizers

- Extend beyond current SR-like flows with robust second-order/trust-region options.
- Add explicit optimizer capability matrix (stochastic, deterministic, constrained).
- Standardize optimizer stats payload for reproducibility and comparisons.

## 3) Determinant Monte Carlo with HS variants

- Introduce DQMC module boundary and stabilization strategy hooks.
- Support multiple Hubbard–Stratonovich decompositions under one interface.
- Add correctness checks for sign/phase conventions and thermal estimator consistency.

## 4) Chebyshev methods

- Add Chebyshev/KPM-based spectral estimation APIs.
- Define consistent moment-storage schema and broadening-kernel configuration.
- Integrate with existing quadratic and ED/Lanczos result consumers.

## 5) Polfed placeholder

- Keep a reserved roadmap slot for "polfed" pending concrete scientific spec.
- Track design decisions in ADR-style notes before implementation begins.

## 6) ED/Lanczos common API

- Unify ED and Lanczos outputs behind one solver/result protocol.
- Ensure consistent eigenpair/spectral metadata across dense and sparse backends.
- Keep backward-compatible wrappers for current entry points.

## 7) Numba recompilation avoidance

- Reduce specialization churn by stabilizing kernel signatures.
- Cache compiled kernels with explicit static/dynamic argument separation.
- Add diagnostics exposing compile vs execute time ratios.

## 8) Quadratic Hamiltonian modernization

- Align `QuadraticHamiltonian` with main Hamiltonian API conventions.
- Consolidate transformation utilities and improve documentation of basis conventions.
- Improve interoperability with solver and spectral pipelines.

## 9) Spin-1 and fermions

- Expand spin-1 operator and model coverage to parity with spin-1/2 paths.
- Improve fermionic workflows (operator/basis conventions, sign correctness checks).
- Add explicit feature support table by model/solver/backend.

## 10) Bases for Hilbert spaces in NQS

- Add explicit basis descriptors for NQS-visible Hilbert state representations.
- Support basis-aware preprocessing/rotations with documented shape/dtype contracts.
- Validate basis transforms with invariance checks on observables.

## 11) NQS + autoregressive sampling improvements

- Improve AR sampler performance/caching and diagnostics.
- Standardize AR model interfaces with current NQS trainer loops.
- Add correctness tests for exact/probability-preserving sampling semantics.

---

## Concept-only: `/general_python` dependencies (out of scope for this branch)

The following topics require direct `/general_python` edits and are therefore
recorded as concept-only in this roadmap branch:

- decomposition/splitting of `QES.general_python` into smaller packages,
- deep optimizer/network rewrites currently located under `general_python.ml`,
- backend algebra/spectral refactors rooted in `general_python.algebra`.
