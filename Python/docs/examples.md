# Examples

The Python examples are organized by module area and are intended to be runnable from the repository root with `PYTHONPATH=Python`.

## Categories

- `examples/algebra/`
  - Hilbert-space construction.
  - Custom many-body Hamiltonians.
  - Sparse and dense matrix construction.
  - Quadratic single-particle Hamiltonians.
  - Operator actions on integer and vector states.

- `examples/physics/`
  - Density matrices and entropy.
  - Time evolution and level statistics.
  - Spectral broadening and statistical helpers.

- `examples/lattices/`
  - Square and honeycomb lattices.
  - Neighbor access and region/visualization support.

- `examples/workflows/`
  - Lattice-driven Hamiltonian assembly and larger analysis scripts.

- `examples/models/`
  - Model-specific demonstrations.

- `examples/nqs/`
  - Variational and optimization examples.

## Recommended entry points

- `examples/algebra/hilbert_and_custom_hamiltonian.py`
- `examples/algebra/operators_on_states.py`
- `examples/algebra/sparse_dense_matrix_build.py`
- `examples/algebra/quadratic_single_particle.py`
- `examples/physics/entropy_density_matrix.py`
- `examples/physics/time_evolution_and_spectral_stats.py`
- `examples/physics/spectral_and_statistical_tools.py`
- `examples/lattices/lattice_neighbors_and_honeycomb.py`
- `examples/workflows/lattice_driven_hamiltonian.py`

## Run all maintained examples

```bash
cd pyqusolver
PYTHONPATH=Python python examples/run_all_examples.py
```
