# Design principles

QES is organized to support scientific workflows where reproducibility and numerical correctness are as important as speed.

## 1) Explicit contracts over implicit behavior

Module-level APIs should communicate:

- expected input ranks and shapes,
- expected dtypes,
- what is returned and in which structural form.

## 2) Backend-aware but backend-agnostic interfaces

Core user APIs should remain stable regardless of execution backend.
Backend-specific optimizations (NumPy/JAX) should preserve mathematical semantics.

## 3) Numerical stability and finite checks

Scientific routines should state known stability sensitivities such as:

- conditioning of linear solves,
- cancellation around nearly degenerate spectra,
- precision sensitivity (`float32` vs `float64`, complex precision).

## 4) Determinism as a first-class concern

Stochastic workflows should expose seed paths and deterministic modes where practical.
When backend behavior differs (for example, due to parallel reductions), docs should call out expected variance.

## 5) Low-friction interoperability

ED and NQS workflows should be composable through clear state/operator conventions, allowing users to switch methods without rewriting model definitions.
