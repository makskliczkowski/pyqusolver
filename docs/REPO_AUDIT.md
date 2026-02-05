# Repository Audit

## Current Structure

The repository is structured as a Python package `QES` located in `Python/QES`. The core components are organized into subpackages:

*   **`Python/QES/Algebra`**: Core physics logic.
    *   `Hamil`: Hamiltonian construction and diagonalization.
    *   `Hilbert`: Hilbert space definitions and symmetry management.
    *   `Operator`: Quantum operators (spin, fermion, etc.) with lazy loading support.
    *   `Symmetries`: Symmetry operations (translation, parity, etc.) - currently uses eager imports.
    *   `Model`: Predefined physical models (Heisenberg, Ising, etc.) - currently uses eager imports.
    *   `backends`: Abstraction layer for NumPy/JAX backends.
*   **`Python/QES/NQS`**: Neural Quantum States.
    *   `nqs.py`: Main entry point (`NQS` class).
    *   `src`: Implementation details (kernels, training logic, integration).
*   **`Python/QES/Solver`**: Simulation solvers.
    *   `MonteCarlo`: VMC and MCMC samplers - currently uses wildcard imports.
*   **`Python/QES/general_python`**: Shared utilities and neural network implementations (outside the scope of this audit).

## Public API Surface

The following modules and classes constitute the primary public API:

*   **`QES`**: Top-level namespace.
    *   `QES.Algebra`
    *   `QES.NQS`
    *   `QES.Solver`
    *   `QES.Hamiltonian`, `QES.HilbertSpace`, `QES.Operator` (Lazy Exports)
    *   `QES.NQS_Model` (Alias for `NQS`)
    *   `QES.MonteCarloSolver`, `QES.Sampler` (Lazy Exports)

*   **`QES.Algebra`**:
    *   `HilbertSpace`: Hilbert space management.
    *   `Hamiltonian`: Hamiltonian operator.
    *   `SymmetrySpec`, `HamiltonianConfig`: Configuration objects.
    *   `Operator`: Access to operator factories (spin, fermion).

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

*   **`Python/setup.py`**: Minimal shim delegating to `pyproject.toml`.
*   **`Python/pyproject.toml`**: Main configuration for build, dependencies, metadata, and tools (Ruff, Black, Mypy, Pytest).
*   **`Python/Makefile`**: Commands for development (install, test, lint, docs).
*   **`Python/tox.ini`**: Configuration for `tox` to run tests across environments.

## Critical Import Paths (Do Not Break)

The following import patterns are observed or implied by the structure and must be preserved:

*   `from QES.Algebra import HilbertSpace, Hamiltonian`
*   `from QES.NQS import NQS`
*   `from QES.Solver.MonteCarlo import VMCSampler` (Currently requires wildcard support in `MonteCarlo` or direct submodule access)
*   `import QES.Algebra.Operator` (and accessing `Operator.operators_spin`)
*   `from QES.Algebra.Hamil import hamil_energy` (and related functions)
*   `from QES.general_python ...` (External dependencies rely on this structure)

## Issues Identified & Planned Improvements

1.  **`QES.Solver.MonteCarlo`**: Uses `from .montecarlo import *`. Should be refactored to lazy imports to avoid namespace pollution and initialization overhead.
2.  **`QES.Solver`**: Only exports `Solver`. Should lazily export `MonteCarlo`.
3.  **`QES.Algebra.Hilbert`**: Uses wildcard imports and attempts to import a missing module `hilbert_jit_methods`, causing `__all__` to be empty. Should be refactored to lazy imports and remove dead code.
4.  **`QES.Algebra.Symmetries`**: Uses eager imports. Should be refactored to lazy imports.
5.  **`QES.Algebra.Model`**: Uses eager imports. Should be refactored to lazy imports.
6.  **`QES.Algebra`**: Uses eager imports inside a `try/except` block. Should be refactored to lazy imports.
