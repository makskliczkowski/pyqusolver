

# Quantum Eigensolver

This project provides a comprehensive framework for simulating quantum systems using various models and functionalities. The project leverages advanced mathematical libraries and parallel computing techniques to ensure efficient and accurate simulations.

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
from QES.qes_globals import (
	get_logger, get_backend_manager, get_numpy_rng, next_jax_key
)

log = get_logger()
backend_mgr = get_backend_manager()
xp = backend_mgr.np
rng = get_numpy_rng()

with backend_mgr.seed_scope(123):
	... # deterministic code ...
```

**Migration Notes:**
- Do **not** create new loggers, backend managers, or RNGs in your own modules. Always import from `QES.qes_globals`.
- This pattern prevents double initialization, race conditions, and inconsistent state across the codebase.

See `Python/QES/qes_globals.py` for full documentation and rationale.

Copyright 2024
Maksymilian Kliczkowski
PhD candidate
Wroclaw University of Science and Technology
maksymilian.kliczkowski.at.pwr.edu.pl 

