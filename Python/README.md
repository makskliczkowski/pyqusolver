# QES Python Package – Quantum Many-Body Simulation Framework

**QES** (Quantum EigenSolver) is a comprehensive Python framework for quantum many-body eigenvalue problems, combining exact diagonalization with neural quantum state variational methods. It is designed to serve both computational physicists and machine learning researchers studying quantum systems.

## Key Features at a Glance

| Feature | Description |
|---------|-------------|
| **Exact Diagonalization** | Full spectrum calculation for spin systems up to ~20 sites using sparse matrix methods |
| **Variational Methods** | Neural Quantum States with RBM/CNN ansätze, optimized via TDVP or gradient descent |
| **Pre-built Models** | TFIM, XXZ, Heisenberg, J1-J2, Kitaev-Heisenberg with automatic Hamiltonian construction |
| **Lattice Support** | Chain, square, hexagonal, triangular, honeycomb with periodic/open boundaries |
| **Dual Backends** | NumPy (CPU, exact) + JAX (GPU, autodiff) with transparent switching |
| **Physics Utilities** | Entropy, purity, density matrices, spectral analysis, symmetry handling |

---

## Installation

### Core Package

```bash
# Editable install (recommended for development)
pip install -e ".[dev]"

# With optional JAX (GPU acceleration, autodiff)
pip install -e ".[jax,dev]"

# All features
pip install -e ".[all,dev]"
```

### Dependencies

- **Required**: NumPy, SciPy, Matplotlib, Pandas, Numba, SymPy
- **JAX**: jax, jaxlib, Flax, Optax
- **ML**: scikit-learn, scikit-image  
- **I/O**: h5py (HDF5)
- **Development**: pytest, black, ruff, sphinx

---

## Quick Start Examples

### 1. Exact Diagonalization: TFIM on 2D Lattice

```python
import QES
from QES.Algebra.Model.Interacting.Spin import TransverseFieldIsing
from QES.general_python.lattices import SquareLattice

with QES.run(backend='numpy', seed=42):
    # 8×8 lattice: Hilbert space dimension = 2^64 ≈ 1.8×10^19 (too large)
    # Use smaller system: 6×6, dimension = 2^36 ≈ 69 billion (still large, use iterative)
    # Practical: 4×4 with dimension = 2^16 = 65,536
    
    lattice = SquareLattice(lx=4, ly=4, bc='pbc')
    
    # TFIM: H = -J Σ σ^z_i σ^z_j - h_x Σ σ^x_i
    H = TransverseFieldIsing(lattice=lattice, j=1.0, hx=0.5)
    
    # Build sparse matrix representation
    H.build()
    
    # Compute spectrum
    evals = H.diagonalize(k=10)  # Get 10 lowest eigenvalues
    
    E0 = evals[0]  # Ground state energy
    E1 = evals[1]  # First excited state
    gap = E1 - E0  # Excitation gap
    
    print(f"Ground state energy density: {E0 / H.Ns:.6f}")
    print(f"Excitation gap: {gap:.6f}")
```

**Output Example**:

```
Ground state energy density: -0.832471
Excitation gap: 0.142893
```

### 2. Neural Quantum State Variational Search

```python
import QES
from QES.Algebra.Model.Interacting.Spin import XXZ
from QES.general_python.lattices import SquareLattice
from QES.NQS import NQS

with QES.run(backend='jax', seed=42):
    # 10-site 1D chain (Hilbert space: 2^10 = 1024)
    lattice = SquareLattice(lx=10, ly=1)
    
    # XXZ model: H = -Σ [J_xy(σ^x σ^x + σ^y σ^y) + J_z σ^z σ^z] - h_x Σ σ^x
    # Anisotropy Δ = J_z / J_xy = 1.4 (antiferromagnetic)
    H = XXZ(lattice=lattice, jxy=1.0, jz=1.4, hx=0.0, hz=0.0)
    
    # Restricted Boltzmann Machine: N_h = α × N_v hidden units
    # α = 2 gives 20 hidden units for 10 visible units
    nqs = NQS(
        model=H,
        ansatz='rbm',
        alpha=2,
        batch_size=64,
        learning_rate=0.01,
        seed=42
    )
    
    # Train variational parameters
    results = nqs.train(
        n_epochs=500,
        optimizer='adam',
        early_stopping_patience=50
    )
    
    E_var = results['energy_final']
    print(f"Variational energy: {E_var:.6f}")
    print(f"Training converged in {results['epochs_trained']} epochs")
```

### 3. Custom Hamiltonian with Symmetries

```python
import QES
from QES.Algebra import HilbertSpace, Hamiltonian
import numpy as np

with QES.run(backend='numpy', precision='float64'):
    # 12-site system with S^z conservation
    hs = HilbertSpace(
        ns=12, 
        is_manybody=True, 
        dtype=np.complex128,
        conserved_quantum_numbers=['Sz']  # Only S^z = 0 sector
    )
    
    H = Hamiltonian(hilbert_space=hs, is_sparse=True)
    ops = H.operators
    
    # Nearest-neighbor Heisenberg exchange
    Sx = ops.sig_x(ns=12, type_act='local')
    Sy = ops.sig_y(ns=12, type_act='local')
    Sz = ops.sig_z(ns=12, type_act='local')
    
    # Add coupling terms
    for i in range(11):
        # H_ij = (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j) / 2
        H.add(Sx @ Sx, 0.25, sites=[i, i+1])
        H.add(Sy @ Sy, 0.25, sites=[i, i+1])
        H.add(Sz @ Sz, 0.25, sites=[i, i+1])
    
    # Magnetic field in z-direction
    for i in range(12):
        H.add(Sz, 0.1, sites=[i])
    
    H.build()
    evals = H.diagonalize()
    
    print(f"Reduced Hilbert space dimension: {hs.nh}")
    print(f"Ground state energy: {evals[0]:.6f}")
```

### 4. Module Discovery

```python
import QES

# Interactive exploration
modules = QES.list_modules(include_submodules=True)
for m in modules:
    print(f"{m['name']:<25} - {m['description']}")

# Get detailed info
print(QES.describe_module('NQS'))
```

---

## Package Architecture

```
QES/
  __init__.py              # Lazy module facade, top-level API
  qes_globals.py           # Global logger, backend manager
  registry.py              # Module discovery
  session.py               # Context managers for session state
  
  Algebra/                 # Quantum algebra & Hamiltonians
    Hamil/
      hamil.py             # Hamiltonian class, matrix building
      hamil_cache.py       # Cached matrix storage
    Hilbert/
      hilbert.py           # Hilbert space, basis management
      hilbert_jit_methods/ # Numba JIT kernels
    Operator/
      operator.py          # Operator classes
      impl/
        operators_spin.py  # Pauli operators, matrix elements
    Model/
      Interacting/
        Spin/              # TFIM, XXZ, Heisenberg, J1-J2, etc.
      Noninteracting/      # Non-interacting models
    Properties/            # Symmetries, conservation laws
    
  NQS/                     # Neural Quantum States
    nqs.py                 # NQS solver interface
    nqs_train.py           # Training loop, TDVP
    tdvp.py                # Time-Dependent Variational Principle
    src/
      nqs_network_integration/  # Network architectures
      
  Solver/                  # Eigensolvers
    MonteCarlo/            # MCMC, Gibbs sampling
      sampler.py           # VMC sampler
    solver.py              # Solver base classes
    
  general_python/          # Reusable scientific utilities
    algebra/               # Sparse solvers, preconditioners
    physics/               # Thermodynamics, spectral analysis
    lattices/              # Lattice geometries, neighbors
    ml/                    # ML utilities, custom layers
    common/                # I/O, logging, utilities
```

---

## Advanced Usage: Session & Backend Management

```python
import QES
import numpy as np

# Default backend (usually NumPy)
print(QES.get_backend_manager().name)  # 'numpy'

# Switch to JAX with custom settings
with QES.run(backend='jax', seed=123, precision='float32'):
    # All operations use JAX, float32
    print(QES.get_backend_manager().name)  # 'jax'
    
    # Can nest sessions
    with QES.run(backend='numpy'):
        # Back to NumPy temporarily
        pass

# Seeding for reproducibility
with QES.run(backend='jax', seed=42):
    # JAX operations use key = PRNGKey(42)
    pass
```

---

## Documentation & References

### In-Package Docs

- [General Python Utilities](QES/general_python/README.md): Algebra, lattices, physics, ML tools
- [Physics Utilities](QES/general_python/physics/README.md): Thermal properties, spectral analysis, response functions
- [NQS Module](QES/NQS/README.md): Neural quantum states, training, TDVP

### Build Full API Docs

```bash
cd docs
pip install -r requirements.txt
make html
# Open _build/html/index.html in browser
```

### Key References

1. Carleo & Troyer (2017): "Solving the quantum many-body problem with artificial neural networks" – *Nature Physics*
2. Liang et al. (2021): "Deep neural network solution to the electronic Schrödinger equation" – *Nature Communications*
3. Ceperley & Alder (1986): "Quantum Monte Carlo" – *Reviews of Modern Physics*

---

## Testing

```bash
# Run full test suite
pytest test/ -v

# Run specific tests
pytest test/test_imports_lightweight.py -v
pytest test/test_eigen.py -v
pytest test/test_backends_interop.py -v
```

---

## Contributing

We welcome contributions. Please ensure:

1. Code follows PEP 8 (Black formatter)
2. New features include docstrings and unit tests
3. All tests pass before submitting PR
4. No external dependencies without discussion

---

## License

CC-BY-4.0 License – See root [LICENSE.md](../../LICENSE.md)

```

## Physics Module Documentation

The `general_python/physics` module is a comprehensive toolkit for condensed matter and quantum statistical physics, with advanced features for thermal, spectral, and response calculations.

- [Physics Module Structure & API](./PHYSICS_MODULE.md): Directory structure, submodules, and capabilities
- [Physics Module: Mathematical Background](./PHYSICS_MATH.md): Mathematical descriptions of implemented algorithms and quantities
- [Physics Module: Usage Examples](./PHYSICS_EXAMPLES.md): Extended code examples for all major features

Additional walkthroughs (including interacting/quadratic Hamiltonian tutorials) live in [EXAMPLES.md](./EXAMPLES.md).

See these documents for detailed usage, mathematical background, and advanced examples.

## Installation

**Standard:**

```bash
pip install -e .
```

**With JAX support:**

```bash
pip install -e ".[jax]"
```

**With all optional dependencies:**

```bash
pip install -e ".[all]"
```

---

## Development

### Run Tests

```bash
pytest
# or
python -m unittest discover -s test
```

### Check Imports

```bash
python test/test_imports_lightweight.py
```

### Build Documentation

```bash
make docs
```

---

## Design Principles

1. **Modularity**: Each subpackage can be imported independently with minimal overhead.
2. **Lazy loading**: Top-level `QES` imports are lightweight; heavy modules load on first use.
3. **Discoverability**: `QES.list_modules()` and `MODULE_DESCRIPTION` strings help users navigate the library.
4. **Global singleton management**: Logger and backend manager are initialized once and shared across the package.
5. **Unified import paths**: All imports use the `QES.*` namespace to avoid ambiguity.

---

## License

CC-BY-4.0 (see [LICENSE](../LICENSE))

## Author

(C) Maksymilian Kliczkowski 2025
Date    : 2025
Email   : <maxgrom97@gmail.com>

---
