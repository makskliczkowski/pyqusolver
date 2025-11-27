
# Quantum Eigensolver

This project provides a comprehensive framework for simulating quantum systems using various models and functionalities. The project leverages advanced mathematical libraries and parallel computing techniques to ensure efficient and accurate simulations.

## Project Overview

Quantum Eigensolver (QES) is a modular Python framework for quantum many-body simulation, variational optimization, and exact diagonalization. It is designed for extensibility, performance, and reproducibility in research and development.

### Main Capabilities

- **Exact Diagonalization (ED):** Solve quantum Hamiltonians on lattice or Hilbert space graphs, including spin systems and random models. Supports quadratic Hamiltonians and multithreaded correlation matrix calculations.
- **Variational Monte Carlo (VMC):** Optimize neural network quantum states (NQS) using Monte Carlo sampling. Supports ground and excited state search with flexible ansatzes (RBM, RBM-PP, custom networks).
- **Symmetry Handling:** Implements point and global symmetries for efficient state space reduction and analysis.
- **TDVP Optimization:** Time-Dependent Variational Principle (TDVP) for time evolution and advanced optimization of variational states.
- **Flexible Backends:** Choose between NumPy and JAX for computation, enabling CPU and GPU acceleration.
- **Logging and Reproducibility:** Centralized global state management for logging, random number generation, and backend configuration.

### Architecture

- `QES/Algebra/` — Operator algebra, Hilbert space, and mathematical utilities.
- `QES/NQS/` — Neural Quantum State solvers, training routines, TDVP, and supporting modules.
- `QES/Solver/` — Monte Carlo samplers and optimization engines.
- `QES/general_python/` — Common utilities, logging, configuration, and backend management.
- `examples/` — Example scripts for typical workflows and model setups.
- `test/` — Comprehensive test suite for integration and correctness.

### Extensibility

QES is designed to be modular. You can:

- Add new ansatzes or neural network architectures.
- Implement custom Hamiltonians or physical models.
- Extend samplers, optimizers, or training routines.
- Integrate with external libraries via the backend manager.

### Typical Workflow

1. **Set up environment variables and Python paths** (see below).
2. **Define your model, network, and sampler.**
3. **Create a solver instance (ED, NQS, etc.).**
4. **Run optimization or simulation.**
5. **Analyze results and export data.**

For more details, see the documentation, example scripts, and the [project wiki](https://github.com/makskliczkowski/QuantumEigenSolver/wiki).

It is a general solver for physical Hamiltonians. The work is in progress. Currently, the solver includes:

- ED solutions to multiple Hamiltonians on the Lattice or Hilbert space graphs. This includes standard spin systems and random Hamiltonians. The software also enables solutions to quadratic Hamiltonians with multithreaded correlation matrices calculation.
- Implementation of point and global symmetries for the systems.
- the variational Quantum Monte Carlo solver for ansatz ground (and excited) states with RBM and RBM-PP ansatzes.

For detailed documentation and usage instructions, please refer to the [project wiki](https://github.com/makskliczkowski/QuantumEigenSolver/wiki).

## Globals and Singletons

### Centralized Global State: `QES.qes_globals`

To avoid double initialization and ensure all modules share the same logger, backend manager, and random number generator, QES now provides a single authoritative module for global state: `QES.qes_globals`.

**Key Singletons Provided:**

- Global logger: `get_logger()`
- Backend manager: `get_backend_manager()`
- NumPy RNG: `get_numpy_rng()`
- JAX key helpers: `next_jax_key()`, `split_jax_keys()`

**Usage Example:**

```python
from QES.qes_globals import get_logger, get_backend_manager, get_numpy_rng, next_jax_key

log = get_logger()
backend_mgr = get_backend_manager()
xp = backend_mgr.np
rng = get_numpy_rng()
jax_key = next_jax_key()

# Example of using a seeded scope
with backend_mgr.seed_scope(123):
... # deterministic code ...
```

**Migration Notes:**

- Do **not** create new loggers, backend managers, or RNGs in your own modules. Always import from `QES.qes_globals`.
- This pattern prevents double initialization, race conditions, and inconsistent state across the codebase.

See `Python/QES/qes_globals.py` for full documentation and rationale.

Copyright 2024-2026
Maksymilian Kliczkowski
Wroclaw University of Science and Technology
maksymilian.kliczkowski.at.pwr.edu.pl

---

## Environment Setup for QES

To run the QES package correctly, set the following environment variables before running your scripts:

- `QES_PYPATH`: Path to the QES installation directory.  
Example:

```bash
export QES_PYPATH=/path/to/QES
```

- `PY_BACKEND`: Backend for numerical operations (`numpy` or `jax`).  
Example:

```bash
export PY_BACKEND=jax
```

- `PY_GLOBAL_SEED`: (Optional) Global random seed.  
Example:

```bash
export PY_GLOBAL_SEED=42
```

- `PY_NUM_CORES`: (Optional) Number of CPU cores to use.  
Example:

```bash
export PY_NUM_CORES=8
```

- `PY_FLOATING_POINT`: (Optional) Floating point precision (`float32` or `float64`).  
Example:

```bash
export PY_FLOATING_POINT=float64
```

- `PY_JAX_DONT_USE`: (Optional) Set to `1` to disable JAX backend.  
Example:

```bash
export PY_JAX_DONT_USE=1
```

### Python Path Setup

You must ensure the QES modules are discoverable by Python. Add the QES directories to your `sys.path` at the start of your script. If you don't want the installation step, you can manually set up the paths as follows:

```python
import os
import sys
from pathlib import Path

qes_path = Path(os.environ.get("QES_PYPATH", "/usr/local/QES")).resolve()
sys.path.insert(0, str(qes_path))
sys.path.insert(0, str(qes_path / 'QES'))
sys.path.insert(0, str(qes_path / 'QES' / 'general_python'))
```

### Installation

The installation of QES requires setting up the Python environment with necessary dependencies.

- Make sure you have all dependencies installed (see `requirements/requirements.txt`).
- Install with pip if needed:

```bash
pip install -r requirements/requirements.txt
```

### Usage Example

To use QES in your Python scripts, ensure the environment variables are set and the paths are configured as shown above. Then you can import and use QES modules as needed:

```python
import os
os.environ['PY_BACKEND'] = 'jax'
os.environ['QES_PYPATH'] = '/path/to/QES'

# Add QES to sys.path as shown above
# ... import and use QES modules ...
```
