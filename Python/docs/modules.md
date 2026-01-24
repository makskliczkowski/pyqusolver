# Project Structure & Modules

QES is organized into four main submodules, reflecting the physics workflow.

## 1. `QES.Algebra`
**Core Quantum Algebra.**
Defines the "what": Hilbert spaces, Operators, and Hamiltonians.
- **Entry Points**: `Hamiltonian`, `HilbertSpace`, `Operator`.
- **Key Features**:
  - Sparse matrix generation.
  - Symmetry handling (Translation, Parity).
  - Basis management.

## 2. `QES.NQS`
**Neural Quantum States.**
Defines the "how" for variational methods.
- **Entry Points**: `NQS` (the main solver class), `NQS.train`.
- **Key Features**:
  - Wraps Flax/JAX networks for quantum states.
  - Handles Monte Carlo sampling loops.
  - Implements Stochastic Reconfiguration (SR) and TDVP.

## 3. `QES.Solver`
**Simulation Engines.**
Abstracts the optimization and sampling logic.
- **Entry Points**: `MonteCarloSolver`, `Sampler`.
- **Key Features**:
  - `VMCSampler`: Metropolis-Hastings sampling.
  - `ExactSolver`: For ED methods.

## 4. `QES.general_python`
**Utilities & Glue Code.**
Shared infrastructure.
- **Entry Points**: `QES.qes_globals` (via root import).
- **Key Features**:
  - `flog`: Advanced logging.
  - `backend_manager`: NumPy/JAX abstraction.
  - `lattices`: Lattice geometry definitions.
