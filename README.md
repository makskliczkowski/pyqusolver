# Quantum EigenSolver (QES)

**QES** is a high-performance, modular Python framework for simulating quantum many-body systems. It unifies **Exact Diagonalization (ED)** and **Variational Monte Carlo (VMC)** with **Neural Quantum States (NQS)** into a single consistent API.

Designed for computational physicists and quantum researchers, QES leverages **JAX** for GPU-accelerated neural networks and automatic differentiation, while maintaining a robust **NumPy/SciPy** backend for exact methods and small-scale prototyping.

---

## Key Concepts

- **Backend Agnostic**: Switch between `numpy` (CPU, exact) and `jax` (GPU/TPU, AD-enabled) with a single configuration flag.
- **Global State**: Managed via `QES.qes_globals` to ensure consistent logging, random number generation (reproducibility), and backend configuration across modules.
- **Hybrid Workflow**: Seamlessly transition from solving a 10-site system exactly (to benchmark) to optimizing a variational ansatz on a 100-site system using the same Hamiltonian definitions.

## Installation

Requires Python 3.10+.

### Quick Install
```bash
pip install .
```

### For Developers (Editable)
```bash
git clone https://github.com/makskliczkowski/pyqusolver.git
cd pyqusolver
pip install -e "Python/[all]"
```

---

## Quickstart

### 1. Minimal Exact Diagonalization (ED)
Solve for the ground state of a free fermion chain (Quadratic Hamiltonian).

```python
import QES
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian

# Start a session (manages precision and backend)
with QES.run(backend='numpy'):

    # Define Hamiltonian: 10 sites
    H = QuadraticHamiltonian(ns=10)

    # Add hopping terms: -t * (c_i^dag c_{i+1} + h.c.)
    for i in range(9):
        H.add_hopping(i, i+1, -1.0)

    # Diagonalize (O(N^3) for quadratic)
    H.diagonalize()

    print(f"Ground State Energy: {sum(val for val in H.eig_val if val < 0):.6f}")
```

### 2. Minimal Neural Quantum State (VMC)
Train a Neural Network (RBM) to find the ground state of a Transverse Field Ising Model.

```python
import QES
from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
from QES.general_python.lattices.square import SquareLattice
from QES.NQS import NQS

# Use JAX for GPU acceleration and gradients
with QES.run(backend='jax', seed=42):

    # 1. Define Model (TFIM) on a 1D chain
    lat = SquareLattice(dim=1, lx=20, bc='pbc')
    H = TransverseFieldIsing(lattice=lat, j=1.0, hx=0.5)

    # 2. Initialize Solver (RBM Ansatz + Metropolis Sampler)
    psi = NQS(
        logansatz='rbm',
        model=H,
        alpha=1,             # Hidden layer density
        batch_size=1024
    )

    # 3. Train using Stochastic Reconfiguration (SR)
    stats = psi.train(n_epochs=100, lr=0.01)

    print(f"Final Variational Energy: {stats.loss_mean[-1]:.6f}")
```

---

## Project Structure

The package is located in `Python/QES`. Key submodules:

- **`QES.Algebra`**: The physics engine. Defines `Hamiltonian`, `HilbertSpace`, and `Operator`. Handles symmetries and matrix construction.
- **`QES.NQS`**: The variational engine. Implements `NQS` solver, neural networks (RBM, CNN, Transformers), and VMC logic.
- **`QES.Solver`**: Abstract simulation backends. Includes `MonteCarloSolver` and samplers (`VMCSampler`).
- **`QES.general_python`**: Shared utilities. Contains the `backend_manager` (JAX/NumPy dispatch), `flog` (logging), and `lattices` (geometry).

## Documentation

Full documentation is available in `Python/docs`. To build it locally:

```bash
cd Python/docs
make html
```

## Reproducibility

QES enforces reproducibility through its `QESSession` and global state manager.

```python
# Guaranteed reproducible run across backend resets
with QES.run(seed=123, precision='float64'):
    ...
```

## Performance Notes

- **JAX**: When using `backend='jax'`, the first call to functions (like `train_step`) will trigger JIT compilation, which may take a few seconds. Subsequent calls are highly optimized.
- **Batching**: For NQS, ensure `batch_size` is large enough to saturate your GPU but small enough to fit in VRAM.
- **Precision**: Use `precision='float32'` for NQS training (standard deep learning practice) and `precision='float64'` for high-precision ED.

## Citation & License

This project is licensed under CC-BY-4.0.

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
Copyright 2025
