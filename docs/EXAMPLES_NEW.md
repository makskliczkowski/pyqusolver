# New QES Examples

The `examples/` directory contains a curated set of runnable scripts demonstrating key QES features.

All examples can be run directly from the repository root:
```bash
python examples/<filename.py>
```

| File | Goal | Runtime | Validates |
| :--- | :--- | :--- | :--- |
| **`00_quickstart_session.py`** | Basic configuration of QES session. | < 1s | Session manager, backend/precision config. |
| **`01_exact_diagonalization.py`** | Construct a Hamiltonian and find exact ground state. | < 5s | HilbertSpace, Hamiltonian construction, sparse diagonalization. |
| **`02_vmc_optimization.py`** | Train a Neural Quantum State (NQS) using VMC. | ~30s | NQS, RBM ansatz, VMC sampling, SR optimization. |
| **`03_entanglement_entropy.py`** | Compute EE, Mutual Information, and TEE. | ~10s | Entanglement module, symmetry handling, density matrices. |
| **`04_benchmark_hamiltonian_caching.py`** | Benchmark Hamiltonian build performance. | ~20s | Hamiltonian caching mechanism. |
| **`05_kitaev_gamma_majorana.py`** | Quadratic models and basis transformations. | < 5s | Non-interacting models, K-space transformations. |
| **`06_region_visualization.py`** | Visualize lattice regions for TEE. | < 5s | Lattice geometry, region masking tools. |

## Prerequisites

All examples require the QES package to be installed (or `Python/` directory in `PYTHONPATH`).
Dependencies include `numpy`, `scipy`, `matplotlib`, `jax`, `flax`, `optax`.

## Details

### 01_exact_diagonalization.py
Constructs a Heisenberg-Kitaev model on a small honeycomb lattice. Uses exact diagonalization to find the ground state energy and wavefunction norm.

### 02_vmc_optimization.py
Demonstrates the variational workflow. Initializes a Restricted Boltzmann Machine (RBM) and trains it to minimize the energy of the Heisenberg-Kitaev Hamiltonian using Stochastic Reconfiguration (SR).

### 03_entanglement_entropy.py
Calculates advanced quantum information quantities. It computes the entanglement spectrum of the ground state and attempts to extract the Topological Entanglement Entropy (TEE) using the Kitaev-Preskill construction.

### 05_kitaev_gamma_majorana.py
Focuses on the `QuadraticHamiltonian` interface. Shows how to work with single-particle Hamiltonians, switch between real-space and momentum-space bases, and solve for the spectrum efficiently.
