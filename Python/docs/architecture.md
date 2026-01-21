# Package Architecture

Understanding the structure of QES helps you navigate the codebase and find the tools you need. The package is designed to be modular, separating the physical definitions (Algebra) from the solution methods (Solver/NQS).

## High-Level Overview

The `QES` package is organized into four main sub-packages:

1.  **`QES.Algebra`**: The language of quantum mechanics.
    *   Defines **Hilbert Spaces** (states, bases, symmetries).
    *   Defines **Hamiltonians** (energy operators, models).
    *   Defines **Operators** (spin matrices, fermionic creation/annihilation).
2.  **`QES.NQS`**: The neural network engine.
    *   Contains **Ansatzes** (RBMs, CNNs, Deep Networks).
    *   Implements **Variational Monte Carlo** logic for neural states.
    *   Handles **TDVP** (Time-Dependent Variational Principle).
3.  **`QES.Solver`**: The numerical workhorses.
    *   **Exact Diagonalization** routines.
    *   **Monte Carlo Samplers** (Metropolis-Hastings, Exchange).
    *   **Optimizers** (SGD, SR, Adam).
4.  **`QES.general_python`**: Shared utilities.
    *   **Lattice** definitions (Square, Honeycomb, etc.).
    *   **Math** helpers, random number generation.
    *   **Physics** utilities (Green's functions, entanglement entropy).
    *   **Logging** and configuration management.

## Directory Structure

Here is a simplified view of the file hierarchy:

```text
QES/
├── __init__.py             # Top-level API and lazy imports
├── qes_globals.py          # Centralized singletons (Logger, Backend, RNG)
│
├── Algebra/                # --- Physics Definitions ---
│   ├── Hilbert/            # Hilbert space logic
│   ├── Hamil/              # Base Hamiltonian classes
│   ├── Model/              # Pre-defined models (Heisenberg, Hubbard)
│   ├── Operator/           # Operator implementations
│   └── ...
│
├── NQS/                    # --- Neural Quantum States ---
│   ├── nqs.py              # Main NQS class
│   ├── ...
│
├── Solver/                 # --- Numerical Solvers ---
│   ├── MonteCarlo/         # MC samplers and solvers
│   ├── ...
│
└── general_python/         # --- Utilities ---
    ├── lattices/           # Lattice graphs
    ├── physics/            # Observables, spectral functions
    ├── ml/                 # Neural network primitives
    └── ...
```

## Design Principles

### 1. Modularity & Lazy Loading
You can import sub-packages independently. To keep startup times fast, heavy modules (like JAX-based networks) are often loaded only when first used. The top-level `import QES` is lightweight.

### 2. Global State Management
QES avoids "double initialization" issues by using a singleton pattern for:
*   **Logging**: One global logger instance ensures consistent output formatting.
*   **Backend Manager**: Handles the choice between NumPy and JAX, and manages random seeds centrally.

Access these via:
```python
import QES
logger = QES.get_logger()
backend = QES.get_backend_manager()
```

### 3. Backend Agnosticism
Many core components are written to be compatible with both NumPy and JAX. This allows you to prototype small systems on a CPU with NumPy and scale up to GPUs with JAX without rewriting your model definitions.

### 4. Registry System
QES uses a registry pattern for Hamiltonians and Models. This allows you to define a problem using a configuration dictionary (useful for running batches of experiments) or by explicitly instantiating classes.
