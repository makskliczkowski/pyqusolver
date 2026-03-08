# Quantum EigenSolver (QES) – Python Framework

**Python Quantum EigenSolver (pyqusolver)** is a Python framework for solving quantum many-body eigenvalue problems through exact methods and neural quantum state (NQS) variational approaches. It provides researchers with tools to study quantum systems using both traditional numerical methods and modern machine learning techniques. `pyqusolver` provides the Python implementation (`QES`) for quantum many-body and quadratic Hamiltonian workflows.

## Future

In the future, the plan is to adapt it to modern Tensor Networ based aproaches and Quantum Monte Carlo schemes. A potential port to more scientifically oriented and efficient languages like Julia is planned.

## Overview

QES bridges exact diagonalization and machine learning-based variational methods within a unified framework. The architecture employs:

- **NumPy Backend**: CPU-based exact diagonalization and classical computations.
- **JAX Backend**: accelerator-ready automatic differentiation for NQS and variational workflows.

This design enables seamless workflows from small-scale exact benchmarks to large-scale variational simulations on identical Hamiltonian definitions.


## Core Capabilities

## Scope

- Many-body Hilbert spaces with optional symmetry reduction.
- Operator-based Hamiltonian construction.
- Quadratic single-particle and BdG Hamiltonians.
- Entropy, density matrix, and statistical utilities.
- Time evolution and spectral analysis helpers.

### 1. Exact Diagonalization

- Full Hamiltonian construction for quantum spin systems, together with observables, dynamics, and expectation values.
- Exact or approximate eigenvalue/eigenvector computation via SciPy sparse solvers.
- Access to linear solver packages, Monte Carlo schemes, neural-network workflows, and machine-learning utilities.
- Support for various lattice geometries (chains, square, hexagonal, triangular, honeycomb)
- Many-body Hilbert space handling with multiple symmetries implemented. Implement your own symmetry easily!
- Computation optimized with Numba JIT compilation or JAX JIT compilation.

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

## Quick Checks

```bash
cd pyqusolver
PYTHONPATH=Python pytest Python/tests -q
```

Maintained categorized tests live under `pyqusolver/Python/tests/`.

## Model Coverage and Physics Guarantees

- Maintained spin-model coverage:
  - `HeisenbergKitaev`
  - `QSM`
  - `UltrametricModel`
- Maintained interacting fermionic coverage:
  - `ManyBodyFreeFermions`
  - `HubbardModel`
- Maintained noninteracting coverage:
  - `FreeFermions`
  - `AubryAndre`
  - `SYK2`
  - `PowerLawRandomBanded`
  - `RosenzweigPorter`
- Physics invariants checked by the maintained Python suite:
  - Hermiticity
  - deterministic seeded construction for maintained random ensembles
  - coupling-update rebuild semantics
  - total-particle-number conservation for spinless many-body fermion models
  - analytic cosine-band reproduction for `FreeFermions`
  - stronger Aubry-Andre localization for large `lmbd`, checked through mean inverse participation ratio
  - finite middle-spectrum gap statistics for `SYK2`, `PowerLawRandomBanded`, and `RosenzweigPorter`
  - middle-spectrum entropy, ETH-style local-observable diagnostics, and finite spectral functions for `QSM`
  - weak-entanglement and basis-local diagnostics for `UltrametricModel`

Maintained model regression files:

- `Python/tests/models/test_heisenberg_kitaev_flux.py`
- `Python/tests/models/test_random_spin_models.py`
- `Python/tests/models/test_fermionic_and_noninteracting_models.py`

## Import Surface

- Stable top-level imports: `QES.Algebra`, `QES.NQS`, `QES.Solver`, `QES.HilbertSpace`, `QES.Hamiltonian`, `QES.Operator`.
- Compatibility-only legacy aliases: `QES.gp_*`, `QES.NQS_Model`.
- New code should import from the concrete submodule instead of the legacy alias surface.

## Getting started

Organized examples are in `pyqusolver/examples/`.

- `examples/algebra/`: Hilbert spaces, Hamiltonians, quadratic systems.
- `examples/physics/`: density matrices, entropy, time evolution, spectral stats.
- `examples/lattices/`: square/honeycomb construction and neighbors.
- `examples/workflows/`: larger end-to-end scripts.
- `examples/models/`: model-specific examples.
- `examples/nqs/`: optimization and NQS-oriented examples.

Run all lightweight examples:

```bash
cd pyqusolver
PYTHONPATH=Python python examples/run_all_examples.py
```

***Key scripts used by this runner:***

- `examples/algebra/example_operators_on_states.py`
- `examples/algebra/example_sparse_dense_matrix_build.py`
- `examples/models/example_random_spin_models.py`
- `examples/workflows/example_lattice_driven_hamiltonian.py`
- `examples/physics/example_spectral_and_statistical_tools.py`

### Examples

#### Example 1: Exact Ground State of Transverse Field Ising Model

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

#### Example 2: Variational Ground State via Neural Quantum States

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

#### Example 3: Custom Hamiltonian

```python
import QES
from QES.Algebra import HilbertSpace, Hamiltonian
import numpy as np

with QES.run(backend='numpy'):
    # Create Hilbert space for 10 spins
    hs  = HilbertSpace(ns=10, is_manybody=True, dtype=np.complex128)
    
    # Build custom Hamiltonian
    H   = Hamiltonian(hilbert_space=hs)
    ops = H.operators
    
    # Add interactions manually
    Sz  = ops.sig_z(ns=10, type_act='local')
    Sx  = ops.sig_x(ns=10, type_act='local')
    
    # Ising interactions along chain
    for i in range(9):
        H.add(Sz, 1.0, sites=[i, i+1])
    
    # Transverse fields
    for i in range(10):
        H.add(Sx, 0.3, sites=[i])
    
    H.build()
    evals = H.diagonalize()
```

## Common Usage Patterns

### 1) Hilbert space with and without symmetries

```python
from QES.Algebra.hilbert import HilbertSpace

hs_full = HilbertSpace(ns=8, is_manybody=True)
hs_sym  = HilbertSpace(ns=8, is_manybody=True, sym_gen={"translation": 0}, gen_mapping=True)
```

### 2) Custom many-body Hamiltonian from operators

```python
import numpy as np
from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice

ns      = 6
lat     = SquareLattice(dim=1, lx=ns, bc="pbc")
hs      = HilbertSpace(lattice=lat, ns=ns, is_manybody=True)

ham     = Hamiltonian(hilbert_space=hs, dtype=np.complex128)
ops     = ham.operators
sx      = ops.sig_x(ns=ns, type_act="local")
sz_corr = ops.sig_z(ns=ns, type_act="correlation")

for i in range(ns):
    j = (i + 1) % ns
    ham.add(sz_corr, sites=[i, j], multiplier=1.0)
    ham.add(sx, sites=[i], multiplier=0.35)

ham.build()
ham.diagonalize(k=4)
```

### 3) Quadratic single-particle Hamiltonian

```python
import numpy as np
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian

qh = QuadraticHamiltonian(ns=12, particle_conserving=True, particles="fermions", dtype=np.complex128)
for i in range(12):
    qh.add_hopping(i, (i + 1) % 12, -1.0)
qh.add_onsite(0, 0.25)

h_single                = qh.build_single_particle_matrix()
W_pc, eps_pc, const_pc  = qh.diagonalizing_bogoliubov_transform()

K       = np.zeros((4, 4), dtype=np.complex128)
Delta   = np.zeros((4, 4), dtype=np.complex128)
for i in range(3):
    K[i, i + 1]     = -1.0
    K[i + 1, i]     = -1.0
    Delta[i, i + 1] = 0.2
    Delta[i + 1, i] = -0.2

qh_bdg = QuadraticHamiltonian.from_bdg_matrices(
    hermitian_part=K,
    antisymmetric_part=Delta,
    constant=0.1,
    dtype=np.complex128,
)
h_bdg                     = qh_bdg.build_bdg_matrix()
W_bdg, eps_bdg, const_bdg = qh_bdg.diagonalizing_bogoliubov_transform()
```

- `build_single_particle_matrix` and `build_bdg_matrix` expose the quadratic blocks directly.
- `diagonalizing_bogoliubov_transform` returns the Qiskit-style orbital transform.
- `to_qiskit_hamiltonian` returns a Qiskit Nature `QuadraticHamiltonian` when that dependency is installed.
- `from_qiskit_hamiltonian` and `from_openfermion_hamiltonian` rebuild the quadratic model from external operator objects.
- Maintained example: `examples/algebra/example_quadratic_single_particle.py`

### 4) Density matrices and entropy

```python
import numpy as np
from QES.general_python.physics.density_matrix import rho, rho_spectrum
from QES.general_python.physics.entropy import entropy, Entanglement

psi   = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
psi  /= np.linalg.norm(psi)
rho_a = rho(psi, va=[0], ns=2)
lam   = rho_spectrum(rho_a)
svn   = entropy(lam, q=1.0, typek=Entanglement.VN)
```

### 5) Lattice neighbors and geometry

```python
from QES.general_python.lattices import SquareLattice, HoneycombLattice

sq = SquareLattice(dim=2, lx=3, ly=3, bc="pbc")
hc = HoneycombLattice(lx=2, ly=2, bc="pbc")

n0_sq = sq.get_nei(0)
n0_hc = hc.get_nei(0)
```

### 6) Time evolution and spectral statistics

```python
import numpy as np
from QES.general_python.physics.eigenlevels import gap_ratio

# after ham.diagonalize(...)
psi0 = ham.eigenstates[:, 0]
times = np.linspace(0.0, 2.0, 9)
psi_t = ham.time_evo.evolve_batch(psi0, times)
vals = np.real(np.array(ham.eigenvalues))
stats = gap_ratio(vals, fraction=0.8, use_mean_lvl_spacing=True)
```

## Module Map (Python)

- Hilbert spaces:
  - `Python/QES/Algebra/hilbert.py`
  - `Python/QES/Algebra/Hilbert/hilbert_base.py`
- Operators and Hamiltonians:
  - `Python/QES/Algebra/Operator/operator.py`
  - `Python/QES/Algebra/Operator/impl/operators_spin.py`
  - `Python/QES/Algebra/Operator/impl/operators_spinless_fermions.py`
  - `Python/QES/Algebra/hamil.py`
  - `Python/QES/Algebra/hamil_quadratic.py`
- Physics utilities:
  - `Python/QES/general_python/physics/density_matrix.py`
  - `Python/QES/general_python/physics/entropy.py`
  - `Python/QES/general_python/physics/eigenlevels.py`
- Lattices:
  - `Python/QES/general_python/lattices/lattice.py`
  - `Python/QES/general_python/lattices/square.py`
  - `Python/QES/general_python/lattices/honeycomb.py`
  
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

## Acknowledgements

During the stay at Ohio State University, the authors made significant advancements in the code, especially in the tools for frustrated systems. They extend their sincere gratitude to Nandini Trivedi and her group for their hospitality and invaluable scientific insights. The visit was supported under the Bekker Programme of the Polish National Agency for Academic Exchange, Grant no. BPN/BEK/2024/1/00115.

## License

This project is licensed under the **CC-BY-4.0 License**. See [LICENSE.md](LICENSE.md) for details.
