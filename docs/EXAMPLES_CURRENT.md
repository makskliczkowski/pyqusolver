# Current Examples

This document lists the example scripts found in `examples/` and their current status.

| Example File | Description | Status | Type |
|--------------|-------------|--------|------|
| `example_exact_diagonalization.py` | Basic exact diagonalization of a 1D Heisenberg chain. Demonstrates Hamiltonian construction, matrix building, and diagonalization. | Meaningful | Scientific Workflow |
| `example_vmc_optimization.py` | Variational Monte Carlo optimization using NQS (RBM) on a Heisenberg chain. Demonstrates NQS interface and training loop. | Meaningful | Scientific Workflow |
| `example_entanglement.py` | Entanglement entropy and topological entanglement entropy (TEE) calculation for Heisenberg-Kitaev model. Uses symmetries. | Meaningful | Analysis |
| `example_basis_transformations.py` | Demonstrates real-space to k-space transformations for Quadratic Hamiltonians using FFT-based Bloch theorem. | Meaningful | API Demo |
| `example_kitaev_majorana.py` | Specific example for Kitaev-Gamma-Majorana model, showing explicit basis control and k-space diagonalization. | Meaningful | Scientific Workflow |
| `example_region_visualization.py` | Visualization tool for debugging Kitaev-Preskill regions used in TEE calculations. | Utility | Visualization |
| `benchmark_hamiltonian_caching.py` | Performance benchmark for Hamiltonian matrix construction caching. | Developer | Benchmark |

## Status Classifications
- **Meaningful**: Demonstrates core functionality or scientific workflow. Should be maintained.
- **Utility**: Helper script for visualization or debugging.
- **Developer**: Internal benchmark or test script.

## Notes
- `example_vmc_optimization.py` requires JAX and NQS modules.
- `example_entanglement.py` requires `HeisenbergKitaev` model and entanglement modules.
- All examples should use `QES.qes_reseed(42)` for deterministic outputs where possible.
