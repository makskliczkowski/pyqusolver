# Repository Audit

## Current Structure

The repository is structured as a Python package `QES` located in `Python/QES`. The core components are organized into subpackages:

*   **`Python/QES/Algebra`**: Core physics logic.
    *   `Hamil`: Hamiltonian construction and diagonalization.
    *   `Hilbert`: Hilbert space definitions.
    *   `Operator`: Quantum operators (spin, fermion, etc.).
    *   `Symmetries`: Symmetry operations (translation, parity, etc.).
    *   `Model`: Predefined physical models (Heisenberg, Ising, etc.).
    *   `backends`: Abstraction layer for NumPy/JAX backends.
*   **`Python/QES/NQS`**: Neural Quantum States.
    *   `nqs.py`: Main entry point (`NQS` class).
    *   `src`: Implementation details (kernels, training logic, integration).
*   **`Python/QES/Solver`**: Simulation solvers.
    *   `MonteCarlo`: VMC and MCMC samplers.
*   **`Python/QES/general_python`**: Shared utilities and neural network implementations (outside the scope of this audit).

## Public API Surface

The following modules and classes constitute the primary public API:

*   **`QES`**: Top-level namespace.
    *   `QES.Algebra`
    *   `QES.NQS`
    *   `QES.Solver`
    *   `QES.Hamiltonian`, `QES.HilbertSpace`, `QES.Operator` (Aliases)
    *   `QES.NQS_Model` (Alias for `NQS`)
    *   `QES.MonteCarloSolver`, `QES.Sampler` (Aliases)

*   **`QES.Algebra`**:
    *   `HilbertSpace`: Hilbert space management.
    *   `Hamiltonian`: Hamiltonian operator.
    *   `SymmetrySpec`, `HamiltonianConfig`: Configuration objects.

*   **`QES.NQS`**:
    *   `NQS`: Main solver class.
    *   `NQSTrainer`: Training manager.
    *   `NetworkFactory`: Helper for creating networks.
    *   `VMCSampler`: Variational Monte Carlo sampler.

*   **`QES.Solver`**:
    *   `MonteCarloSolver`: Base class for MC solvers.
    *   `Sampler`: Base class for samplers.
    *   `VMCSampler`: Concrete VMC sampler implementation.

## Build/CI Entry Points

*   **`Python/setup.py`**: Minimal shim for installation.
*   **`Python/pyproject.toml`**: Main configuration for build, dependencies, and metadata.
*   **`Python/Makefile`**: Commands for development (install, test, lint, docs).
*   **`Python/tox.ini`**: Configuration for `tox` to run tests across environments.

## Critical Import Paths (Do Not Break)

The following import patterns are observed or implied by the structure and must be preserved:

*   `from QES.Algebra import HilbertSpace, Hamiltonian`
*   `from QES.NQS import NQS`
*   `from QES.Solver.MonteCarlo import VMCSampler`
*   `import QES.Algebra.Operator` (should allow access to submodules like `operators_spin`)
*   `from QES.Algebra.Hamil import hamil_energy` (and related functions)
*   `from QES.general_python ...` (External dependencies rely on this structure)

## Issues Identified

1.  **`QES.Algebra.Operator`**: Submodules listed in `__all__` (e.g., `operators_spin`) are not actually imported or accessible via dot notation, requiring explicit imports like `from QES.Algebra.Operator.impl import operators_spin`.
2.  **`QES.Algebra.Hamil`**: Uses wildcard imports (`from .hamil_energy import *`), which pollutes the namespace and hinders lazy loading.
