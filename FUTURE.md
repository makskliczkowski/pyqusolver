# Future Roadmap

This document outlines the planned features, improvements, and research directions for the QES framework.

> **Note**: Items marked with `[Concept]` affecting `general_python` require careful coordination as that submodule is shared across projects.

## Methods & Solvers

### Tensor Networks & MPS
-   Implement Matrix Product States (MPS) and Tensor Train (TT) ansatzes.
-   Integrate DMRG-like optimization routines.
-   Support for contraction-based energy evaluation.

### Determinant Quantum Monte Carlo (DQMC)
-   Implement DQMC with Hubbard-Stratonovich auxiliary field variants.
-   Support for finite-temperature calculations.
-   Sign problem mitigation strategies.

### Chebyshev Methods
-   Polynomial expansion of the time evolution operator (Chebyshev propagation).
-   Spectral density calculation via Kernel Polynomial Method (KPM).

### Improved Optimizers
-   Integration of advanced second-order optimizers (e.g., K-FAC).
-   Adaptive learning rate schedulers for VMC (beyond simple decay).
-   `[Concept]` Unified optimizer interface in `general_python.ml`.

### "Polfed" (Placeholder)
-   *Details TBD*: Placeholder for upcoming research method "Polfed".

## Exact Diagonalization (ED)

### Common API for ED/Lanczos
-   Refactor `Hamiltonian.diagonalize` to use a unified `EigenSolver` interface.
-   Standardize Lanczos/Arnoldi iteration control (restarts, convergence criteria).
-   Support for spectral transformations (shift-invert) across backends.

## Performance & JIT

### Numba Recompilation Avoidance
-   Systematic review of JIT-compiled functions to ensure type stability.
-   Use of `cache=True` where appropriate.
-   Refactor dynamic function generation to avoid recompilation triggers.

## Physics & Modeling

### Quadratic Hamiltonian Modernization
-   Refactor `QuadraticHamiltonian` to align with the generic `Hamiltonian` interface.
-   Support for generic Bogoliubov transformations without assuming specific particle types.
-   Efficient "NQS-like" sampling from Gaussian states (Slater determinants/Pfaffians).

### Spin-1 & Fermionic Systems
-   Expand `Operator` catalog for Spin-S systems (S > 1/2).
-   Native support for fermionic mapping strings (Jordan-Wigner) in `NQS`.
-   Improved performance for fermionic sign handling in VMC.

### Hilbert Space Bases in NQS
-   Allow NQS models to operate in bases other than the computational (Z) basis.
-   Support for basis rotation layers (e.g., working in X-basis for transverse field models).

### NQS & Autoregressive Sampling
-   `[Concept]` Improve autoregressive sampling efficiency (caching of conditional probabilities).
-   Support for continuous variable autoregressive models in `general_python`.

## Documentation & Structure
-   Continued modularization of `QES.Algebra`.
-   Lazy loading for all heavy dependencies.
