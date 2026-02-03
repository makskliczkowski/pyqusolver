# Quantum EigenSolver (QES) – Python Framework

**Python Quantum EigenSolver (pyqusolver)** is a Python framework for solving quantum many-body eigenvalue problems through exact methods and neural quantum state (NQS) variational approaches. It provides researchers with tools to study quantum systems using both traditional numerical methods and modern machine learning techniques.

## Future

In the future, the plan is to adapt it to modern Tensor Networ based aproaches and Quantum Monte Carlo schemes. A potential port to more scientifically oriented and efficient languages like Julia is planned.

## Overview

QES bridges exact diagonalization and machine learning-based variational methods within a unified framework. The architecture employs:

- **NumPy Backend**: CPU-based exact diagonalization and classical computations...
- **JAX Backend**: GPU-accelerated automatic differentiation, ideal for training neural quantum states...

This design enables seamless workflows from small-scale exact benchmarks to large-scale variational simulations on identical Hamiltonian definitions.

## Core Capabilities

### 1. Exact Diagonalization

- Full Hamiltonian construction for quantum spin systems, access to various measures, quantum dynamics, expectaion values, etc.
- Exact or approximate eigenvalue/eigenvector computation via scipy sparse solvers
- Access to linear solver packages, Monte Carlo schemes, Neural Network and Machine Learning schemes
- Support for various lattice geometries (chains, square, hexagonal, triangular, honeycomb)
- Many-body Hilbert space handling with multiple symmetries implemented. Implement your own symmetry easily!
- Computation optimized with Numba JIT compilation or JAX JIT compilation

### 2. Quantum Spin Models

Pre-implemented models with automated Hamiltonian construction:

- Many Body Models
  - Spin Models:
    - **Transverse Field Ising Model (TFIM)**: $H = -J \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j - h_x \sum_i \sigma^x_i$
    - **XXZ Model**: $H = -\sum_{\langle i,j \rangle} [J_{xy}(\sigma^x_i \sigma^x_j + \sigma^y_i \sigma^y_j) + J_z \sigma^z_i \sigma^z_j] - h_x \sum_i \sigma^x_i - h_z \sum_i \sigma^z_i$
    - **Heisenberg Model**: Isotropic XXX variant ($\Delta = 1$)
    - **J1-J2 Model**: First and second nearest-neighbor interactions
    - **Kitaev-Heisenberg Model**: Combining Kitaev and Heisenberg couplings
    - **Quantum Spin Models**: General framework for custom interactions
  - Fermionic models
- Quadratic Systems:
  - Fermionic, particle conserving models
  - Random Matrix models

### 3. Neural Quantum States (NQS)

Variational ground state search using neural network ansätze:

- **Monte Carlo Sampling**: Markov chain sampling for state estimation, Autoregressive sampling
- **Training Framework**: Stochastic gradient descent with custom optimizers
- **Supported Architectures**: Restricted Boltzmann Machines (RBMs), Convolutions, Dense networks via Flax. Attach your own as well!
- **TDVP Integration**: Time-Dependent Variational Principle for parameter optimization
- **Automatic Differentiation**: Full gradient computation via JAX

### 4. Lattice Support

Built-in lattice utilities for topology-aware Hamiltonian construction:

- Chain (1D)
- Square (2D)
- Hexagonal (2D)
- Triangular (2D)
- Honeycomb (2D)
- Periodic/Open boundary conditions

### 5. General Scientific Utilities (`general_python`)

Physics and numerical computing tools available throughout QES:

- **Algebra/Linear Algebra**: Sparse matrix solvers (Lanczos, Arnoldi, MinresQLP), eigenvalue problems
- **Quantum Operations**: Density matrix calculations, entropy (von Neumann, Shannon), purity
- **Lattice Geometry**: Neighbor finding, boundary handling, visualization
- **Backend Agnosticism**: Seamless NumPy/JAX switching via centralized backend management
- **Random Number Generation**: Reproducible pseudorandom sequences

## Installation

### Prerequisites

- Python ≥ 3.10
- pip or conda

### Standard Installation

```bash
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver/pyqusolver
pip install -e "Python/[dev]"
```

### Optional Dependencies

```bash
# JAX support (GPU acceleration, automatic differentiation)
pip install -e "Python/[jax]"

# Machine learning utilities (scikit-learn, scikit-image)
pip install -e "Python/[ml]"

# HDF5 file I/O
pip install -e "Python/[hdf5]"

# All features
pip install -e "Python/[all,dev]"
```

## Getting Started

### Example 1: Exact Ground State of Transverse Field Ising Model

```python
import QES
from QES.Algebra.Model.Interacting.Spin import TransverseFieldIsing
from QES.general_python.lattices import SquareLattice

# Use NumPy backend for exact diagonalization
with QES.run(backend='numpy', seed=42):
    # 6×6 square lattice with Periodic Boundary Conditions
    lattice = SquareLattice(lx=6, ly=6)
    
    # Define TFIM with J=1.0, transverse field hx=0.5
    H = TransverseFieldIsing(lattice=lattice, j=1.0, hx=0.5)
    
    # Construct and diagonalize
    H.build()
    eigenvalues = H.diagonalize()
    E0 = eigenvalues[0]
    
    print(f"Ground state energy density: {E0 / H.Ns:.6f}")
```

### Example 2: Variational Ground State via Neural Quantum States

```python
import QES
from QES.Algebra.Model.Interacting.Spin import XXZ
from QES.general_python.lattices import SquareLattice
from QES.NQS import NQS

with QES.run(backend='jax', seed=42):
    # 8-site chain with XXZ interactions
    lattice = SquareLattice(lx=8, ly=1)
    
    # XXZ model: isotropic XY with Ising coupling, Delta=1.2
    H = XXZ(lattice=lattice, jxy=1.0, jz=1.2, hx=0.0, hz=0.0)
    
    # Initialize NQS solver with RBM ansatz (alpha=2 hidden units per visible)
    nqs = NQS(
        model=H,
        ansatz='rbm',
        alpha=2,
        batch_size=32,
        learning_rate=0.01,
        seed=42
    )
    
    # Train for 500 epochs
    results = nqs.train(n_epochs=500)
    
    print(f"Final variational energy: {results['energy_final']:.6f}")
    print(f"Estimated ground state energy: {results['E_gs']:.6f}")
```

### Example 3: Custom Hamiltonian

```python
import QES
from QES.Algebra import HilbertSpace, Hamiltonian
import numpy as np

with QES.run(backend='numpy'):
    # Create Hilbert space for 10 spins
    hs = HilbertSpace(ns=10, is_manybody=True, dtype=np.complex128)
    
    # Build custom Hamiltonian
    H = Hamiltonian(hilbert_space=hs)
    ops = H.operators
    
    # Add interactions manually
    Sz = ops.sig_z(ns=10, type_act='local')
    Sx = ops.sig_x(ns=10, type_act='local')
    
    # Ising interactions along chain
    for i in range(9):
        H.add(Sz, 1.0, sites=[i, i+1])
    
    # Transverse fields
    for i in range(10):
        H.add(Sx, 0.3, sites=[i])
    
    H.build()
    evals = H.diagonalize()
```

## Architecture & Design Principles

### Lazy Imports

QES uses deferred module loading to minimize startup overhead. Heavy dependencies (JAX, TensorFlow, etc.) are only imported when explicitly requested.

```python
import QES

# Submodules loaded on first access, not at import time
algebra_module = QES.Algebra
nqs_module = QES.NQS
```

### Session Management

Global state (backend selection, seed, precision) is managed via context managers:

```python
with QES.run(backend='jax', seed=123, precision='float32'):
    # Operations inside use JAX backend with specified seed
    pass
```

### Hilbert Space Representation

- **Binary basis**: Computational basis $|s_1 s_2 \ldots s_N\rangle$ with $s_i \in \{0,1\}$
- **Sparse matrices**: Hamiltonian stored in COO/CSR format for memory efficiency
- **Basis caching**: Identical Hamiltonians reuse cached matrix representations

## Physics Context

This framework addresses key computational challenges in quantum many-body physics:

1. **Computational Scaling**: Hilbert space dimension grows exponentially ($2^N$ for $N$ spins); larger systems require variational approaches.

2. **Benchmark Problem**: Neural quantum states provide variational upper bounds on ground state energies. Exact results on small systems validate NQS quality.

3. **Ansatz Expressibility**: RBM ansätze have limited representational capacity for some quantum phases. Framework allows testing multiple architectures and sampling strategies.

4. **Symmetry Exploitation**: Leveraging conserved quantum numbers (spin, parity) reduces computational cost; QES supports custom symmetries via Hilbert space constraints.

## Limitations & Current Status

**Development Stage**: Core functionality is stable; API may evolve.

## Documentation

Full API documentation is available in the [Python documentation](Python/docs/):

```bash
cd Python/docs
pip install -r requirements.txt
make html
# Open _build/html/index.html
```

### Key Modules

- [Algebra](Python/QES/Algebra): Hilbert spaces, operators, Hamiltonians
- [NQS](Python/QES/NQS): Neural quantum states, training, TDVP
- [Solver](Python/QES/Solver): Exact and Monte Carlo solvers
- [general_python](Python/QES/general_python): Numerical utilities and physics tools

## Testing

Run the test suite:

```bash
cd Python
pytest test/ -v
```

## Contributing

Contributions are welcome. Please ensure:

1. All tests pass
2. New features include docstrings and tests
3. No external dependencies added without discussion

## Citation

If you use QES in research, please cite:

```bibtex
@software{kliczkowski2025QuantumEigenSolver,
  author = {Kliczkowski, Maksymilian},
  title = {Quantum EigenSolver: A Framework for Quantum Many-Body Simulations},
  year = {2025},
  url = {https://github.com/makskliczkowski/QuantumEigenSolver}
}
```

## License

This project is licensed under the **CC-BY-4.0 License**. See [LICENSE.md](LICENSE.md) for details.

## References

...