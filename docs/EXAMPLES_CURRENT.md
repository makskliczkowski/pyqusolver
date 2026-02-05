# Current Examples Status

This document lists the examples found in the repository as of the cleanup initiation.

| File | Status | Classification | Notes |
| :--- | :--- | :--- | :--- |
| `examples/benchmark_hamiltonian_caching.py` | Runs | Meaningful | Benchmarks Hamiltonian build caching. Essential for performance testing. |
| `examples/example_session.py` | Runs | Meaningful | Demonstrates `QES.run` context manager. Simple quickstart. |
| `Python/examples/example_basis_transformations.py` | Outdated | Tutorial/Text | Contains mostly text and placeholders. Replaced by `example_kitaev_basis_usage.py`. |
| `Python/examples/example_entanglement_entropy.py` | Redundant | Redundant | Older version of entanglement calculation. Replaced by `example_entanglement_module.py`. |
| `Python/examples/example_entanglement_module.py` | Runs | Meaningful | Advanced example using `EntanglementModule`. |
| `Python/examples/example_region_visualization.py` | Runs | Meaningful | Visualization tool for regions. |
| `Python/examples/single_particle/example_kitaev_basis_usage.py` | Runs | Meaningful | Demonstrates basis transformations in detail. |
| `Python/examples/example_symmetries.ipynb` | Unknown | Notebook | Jupyter notebook on symmetries. Not automatically tested in this pass. |

## Action Plan

The valid examples will be moved to the root `examples/` directory with standardized naming and imports. The redundant and outdated examples will be removed. New examples covering exact diagonalization and VMC optimization will be added.
