# API Reference

The QES API is organized into four main subpackages.

## [Algebra](api/algebra.rst)
**`QES.Algebra`**
Core components for quantum systems: Hamiltonians, Hilbert spaces, and Operators.
- `Hamiltonian`: Define physical models.
- `HilbertSpace`: Define state spaces and symmetries.
- `Operator`: General operators and observables.

## [NQS (Neural Quantum States)](api/nqs.rst)
**`QES.NQS`**
Variational Monte Carlo framework using Neural Networks.
- `NQS`: Main solver class.
- `NQSTrainer`: Training loop manager.
- `VMCSampler`: Monte Carlo sampling.

## [Solver](api/solver.rst)
**`QES.Solver`**
Simulation backends and sampling logic.
- `MonteCarloSolver`: Base class for MC methods.
- `Sampler`: Interface for state sampling.

## [Utilities](api/utilities.rst)
**`QES.general_python`**
Shared infrastructure.
- `qes_globals`: Global state (logger, backend).
- `lattices`: Lattice definitions.
