# Python Examples

## Organization

- `algebra/`
  - Hilbert spaces, custom many-body Hamiltonians, quadratic single-particle Hamiltonians.
  - Qiskit/OpenFermion-style quadratic round trips when optional dependencies are present.
  - Operator action examples for spin-1/2, spinless fermions, and spin-1.
  - Sparse and dense matrix construction paths.
- `physics/`
  - Density matrices, entropy measures, time evolution, level statistics.
  - Spectral broadening and statistical diagnostics (`ldos`, `dos`).
- `lattices/`
  - Square and honeycomb lattice construction and neighbor access.
- `workflows/`
  - Larger end-to-end demonstrations.
- `models/`
  - Model-specific demonstrations.
- `nqs/`
  - Variational and optimization-oriented examples.

## Run

From `pyqusolver/`:

```bash
PYTHONPATH=Python python examples/run_all_examples.py
```

Or run a specific script, for example:

```bash
PYTHONPATH=Python python examples/algebra/example_hilbert_and_custom_hamiltonian.py
```
