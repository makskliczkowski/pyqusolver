# Python Tests

## Categories

- `algebra/`
  - Hilbert spaces, custom Hamiltonians, matrix construction, operator workflows.
- `core/`
  - Registry, numerical helper invariants, lightweight module-discovery checks.
- `lattices/`
  - Lattice geometry and visualization-oriented checks.
- `models/`
  - Concrete physics model regression tests.
- `nqs/`
  - NQS entropy, RBM fast-update, and variational regression tests.
- `physics/`
  - Density matrices, entropy, spectral/statistical utilities.
- `solvers/`
  - DQMC, Monte Carlo, and solver regression tests.

## Run

From `pyqusolver/`:

```bash
PYTHONPATH=Python pytest Python/tests -q
```

Run one category:

```bash
PYTHONPATH=Python pytest Python/tests/algebra -q
```
