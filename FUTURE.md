# QES Roadmap

This document captures the requested roadmap items and future directions for the Quantum EigenSolver (QES) library.

## Tensor Networks / MPS
*   Integration of Matrix Product States (MPS) and Tensor Network states as a new class of Ansatz in `QES.NQS` or a separate module `QES.TN`.
*   Support for DMRG-like optimization and TEBD time evolution.
*   Interoperability with NQS for hybrid methods (e.g., MPS-initialized NQS).

## Improved Optimizers
*   Advanced second-order optimizers beyond standard SR/MinSR.
*   Integration of K-FAC (Kronecker-Factored Approximate Curvature) for deeper networks.
*   Trust-Region methods for more stable convergence in difficult landscapes.

## Determinant Monte Carlo (DQMC) with HS Variants
*   Implementation of Determinant Quantum Monte Carlo.
*   Support for various Hubbard-Stratonovich (HS) transformations.
*   Stable stabilization techniques for low-temperature simulations.

## Chebyshev Methods
*   Chebyshev polynomial expansion for spectral functions and density of states.
*   Kernel Polynomial Method (KPM) integration.

## Polfed Placeholder
*   Placeholder for "Polfed" (Polishing/Refinement methods? Or specific physics algorithm?).
*   *Note: Needs clarification on specific scope.*

## Exact Diagonalization (ED) / Lanczos Common API
*   Unify ED and Lanczos interfaces under a common `DiagonalizationSolver`.
*   Standardize output formats (eigenvalues, eigenvectors, spectral functions) across dense and sparse solvers.
*   Lazy operator support for larger systems in ED.

## Numba Recompilation Avoidance
*   Refactor Numba-compiled kernels to avoid excessive recompilation.
*   Use caching of compiled functions based on signatures.
*   Move static arguments to closure variables or strictly typed arguments.

## Quadratic Hamiltonian Modernization
*   Modernize `QuadraticHamiltonian` to align with the generic `Hamiltonian` interface.
*   Ensure efficient Bogoliubov transformation handling.
*   Support for open boundary conditions and general graphs in quadratic solvers.

## Spin-1 / Fermions
*   Full support for Spin-1 systems (already partially in `operators_spin_1`).
*   Complete Fermionic operator support (Jordan-Wigner strings, etc.) in all NQS ansatze.
*   Fermionic neural networks (Backflow, Slater-Jastrow).

## Bases for Hilbert Spaces in NQS
*   Explicit support for working in different bases (Sz, Sx, Particle Number) within NQS.
*   Basis rotation layers in the ansatz.

## NQS + Autoregressive Sampling Improvements
*   Enhanced Autoregressive Neural Networks (AR-NN) for exact sampling.
*   Optimization of AR sampling kernels (caching, fast sampling).
*   Support for continuous variables or larger local Hilbert spaces in AR models.

## General Python (`QES.general_python`)
*   *Concept Only*: Refactoring of `general_python` to be less monolithic.
*   Splitting `general_python` into `qes_core`, `qes_utils`, etc.
*   Deprecation of legacy modules in favor of standard library or specialized packages.
