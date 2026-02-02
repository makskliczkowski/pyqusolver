# Quantum EigenSolver (QES)

**Quantum EigenSolver (QES)** is a flexible and high-performance framework designed for the simulation of quantum many-body systems. We built it to bridge the gap between rigorous, exact methods and modern neural network-based approaches, giving researchers a unified environment for exploration.

At its core, QES uses a dual-backend architecture. We leverage **C++** for computationally intensive tasks and **Python (JAX/NumPy)** to provide a user-friendly, flexible interface that integrates seamlessly with modern machine learning workflows.

## Why QES?

*   **Hybrid Workflow:** You can easily transition from running exact benchmarks on small systems to performing large-scale variational simulations on the same Hamiltonian definitions.
*   **Best of Both Worlds:**
    *   **NumPy Backend:** Perfect for rapid prototyping, debugging, and exact diagonalization on CPUs.
    *   **JAX Backend:** Unlocks GPU acceleration and automatic differentiation, essential for training Neural Quantum States.
*   **Neural Quantum States (NQS):** We provide built-in support for Variational Monte Carlo (VMC) using various architectures like RBMs, CNNs, and Transformers.
*   **Robust Utilities:** The framework includes `general_python`, a comprehensive suite of physics and math tools available for any scientific computing task.

## Installation

We recommend installing QES in a dedicated virtual environment. You will need Python 3.10 or newer.

### Standard Installation

```bash
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver/pyqusolver
pip install .
```

### Development Installation

If you plan to contribute or run the test suite, install the development dependencies:

```bash
pip install -e "Python/[all,dev]"
```

## Quick Start

Here is how you can get started with QES in just a few lines of code.

### 1. Exact Diagonalization
Let's calculate the ground state energy of a Heisenberg model on a $4 \times 4$ square lattice using the exact solver.

```python
import QES
from QES.Algebra.Model.Interacting.Spin import Heisenberg
from QES.general_python.lattices import SquareLattice

# Initialize a session using the NumPy backend for exact calculations
with QES.run(backend='numpy'):
    # Define a 4x4 square lattice and the Heisenberg Hamiltonian
    lat = SquareLattice(dim=2, lx=4, ly=4)
    H = Heisenberg(lattice=lat, J=1.0)
    
    # Diagonalize the Hamiltonian
    H.diagonalize()
    print(f"Ground State Energy: {H.E0}")
```

### 2. Neural Quantum States
Now, let's train a Restricted Boltzmann Machine (RBM) to find the ground state of a Transverse Field Ising Model using the JAX backend.

```python
import QES
from QES.Algebra.Model.Interacting.Spin import TransverseFieldIsing
from QES.NQS import NQS

# Initialize a session with JAX for GPU acceleration
with QES.run(backend='jax', seed=42):
    # Define the model on a chain of 20 sites
    H = TransverseFieldIsing(lx=20, J=1.0, hx=0.5)
    
    # Initialize the Variational Solver with an RBM ansatz
    solver = NQS(model=H, ansatz="rbm", alpha=2)
    
    # Train the model
    stats = solver.train(epochs=200, learning_rate=0.01)
    print(f"Final Variational Energy: {stats.energy_mean[-1]}")
```

## Documentation

We believe in good documentation. You can build the full API reference locally to explore all available modules and methods.

```bash
cd pyqusolver/Python/docs
pip install -r requirements.txt
make html
```

Once built, open `_build/html/index.html` in your web browser.

## Contributing

We welcome contributions from the community! Whether it's fixing a bug, adding a new feature, or improving documentation, your help is appreciated. Please check our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is open-sourced under the **CC-BY-4.0** License.