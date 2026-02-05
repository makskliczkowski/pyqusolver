# QES Usage Examples

This directory contains a set of working examples demonstrating the core functionality of the Quantum Eigen Solver (QES) framework.

## Runnable Examples

The following examples are located in the `examples/` directory. They are designed to run out-of-the-box (assuming dependencies are installed).

### 1. Exact Diagonalization
**File:** `examples/example_exact_diagonalization.py`
**Goal:** Compute exact ground state energy of a Heisenberg chain.
**Key Concepts:**
- Defining `Lattice` and `HilbertSpace`.
- Constructing a `Hamiltonian` with custom terms.
- Building the sparse matrix and using `diagonalize()`.
**Run:**
```bash
python examples/example_exact_diagonalization.py
```

### 2. Variational Monte Carlo (VMC) Optimization
**File:** `examples/example_vmc_optimization.py`
**Goal:** Optimize a Neural Quantum State (RBM) to find the ground state energy.
**Key Concepts:**
- Defining a `Hamiltonian`.
- initializing an NQS ansatz (`RBM`) and `VMCSampler`.
- Using `NQS` driver to run the optimization loop.
- Handling JAX backend and random seeds.
**Run:**
```bash
python examples/example_vmc_optimization.py
```

### 3. Entanglement Entropy & Topology
**File:** `examples/example_entanglement.py`
**Goal:** Compute entanglement entropy and topological entanglement entropy (TEE).
**Key Concepts:**
- `HeisenbergKitaev` model on a Honeycomb lattice.
- `SymmetryContainer` for translation symmetry.
- Computing Von Neumann entropy of subsystems.
- Kitaev-Preskill TEE calculation.
**Run:**
```bash
python examples/example_entanglement.py
```

### 4. Kitaev Model & Majorana Fermions
**File:** `examples/example_kitaev_majorana.py`
**Goal:** Demonstrate the Kitaev-Gamma model with Majorana fermion mapping.
**Key Concepts:**
- `KitaevGammaMajorana` model.
- Basis transformations (Real space <-> k-space).
- Explicit `HilbertSpace` basis control.
**Run:**
```bash
python examples/example_kitaev_majorana.py
```

### 5. Basis Transformations
**File:** `examples/example_basis_transformations.py`
**Goal:** detailed guide on transforming Hamiltonians between bases.
**Key Concepts:**
- `QuadraticHamiltonian`.
- `to_basis('k-space')` API.
- Understanding Bloch Hamiltonian blocks.
**Run:**
```bash
python examples/example_basis_transformations.py
```

### 6. Region Visualization
**File:** `examples/example_region_visualization.py`
**Goal:** Visualize lattice regions used for entanglement cuts.
**Key Concepts:**
- `SquareLattice` geometry.
- Defining and validating Kitaev-Preskill regions (A, B, C).
**Run:**
```bash
python examples/example_region_visualization.py
```

### 7. Hamiltonian Caching Benchmark
**File:** `examples/benchmark_hamiltonian_caching.py`
**Goal:** Verify performance gains from Hamiltonian caching.
**Key Concepts:**
- `Hamiltonian` build process.
- Caching mechanism for identical Hamiltonians.
**Run:**
```bash
python examples/benchmark_hamiltonian_caching.py
```

## Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install numpy scipy matplotlib h5py pandas jax jaxlib flax optax scienceplots
```

Set your `PYTHONPATH` to include the library root if not installed via pip:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/Python
```
