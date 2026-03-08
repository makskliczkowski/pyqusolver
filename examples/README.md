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
  - Random-spin diagnostics for QSM and ultrametric ensembles.
- `nqs/`
  - Variational and optimization-oriented examples.

## Reading Order

- Start with `algebra/` and `physics/` for the maintained core workflows.
- Use `lattices/` next if you want geometry-driven Hamiltonian construction.
- Move to `workflows/`, `models/`, and `nqs/` for larger end-to-end examples.

- `models/example_random_spin_models.py`
  - Deterministic QSM and ultrametric diagnostics.
  - Middle-spectrum entropy, local observables, gap ratios, and spectral functions.

## Run

From `pyqusolver/`:

```bash
PYTHONPATH=Python python examples/run_all_examples.py
```

Or run a specific script, for example:

```bash
PYTHONPATH=Python python examples/algebra/example_hilbert_and_custom_hamiltonian.py
```

## Output

- Most examples print to stdout only.
- Workflow examples may create lightweight artifacts under the repository `tmp/` area.
