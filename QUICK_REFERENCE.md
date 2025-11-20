# QuadraticHamiltonian Quick Reference Guide

## Installation & Setup

```python
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
from QES.Algebra.backends import get_backend
```

## Creating a Hamiltonian

### Basic (Particle-Conserving)

```python
qh = QuadraticHamiltonian(ns=4, particle_conserving=True)
qh.add_hopping(0, 1, -1.0)          # Hopping amplitude
qh.add_onsite(0, 0.5)               # Onsite energy
qh.diagonalize()
print(qh.eig_val)                   # Eigenvalues
```

### BdG (Superconductivity)

```python
qh = QuadraticHamiltonian(ns=4, particle_conserving=False)
qh.add_hopping(0, 1, 1.0)           # Single-particle term
qh.add_pairing(0, 1, 0.5)           # Pairing/gap term
qh.diagonalize()
```

## Direct Matrix Construction

```python
import numpy as np

# From Hermitian matrix (particle-conserving)
H           = np.array([[1, -1], [-1, 1]], dtype=complex)
qh          = QuadraticHamiltonian.from_hermitian_matrix(H, constant=0.5)

# From BdG matrices
H_hopping   = np.array([[0, 1], [1, 0]], dtype=complex)
H_pairing   = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
qh          = QuadraticHamiltonian.from_bdg_matrices(H_hopping, H_pairing)
```

## Backend Control

```python
# Check available backends
backends = qh.get_backend_list()  # [('numpy', True), ('jax', True)]

# Switch backend at runtime
qh.set_backend('jax')  # Switch to JAX for autodiff

# Create with specific backend
qh = QuadraticHamiltonian(ns=4, backend='jax')
```

## Properties & Methods

### Diagonalization Results

```python
qh.diagonalize()
eig_val = qh.eig_val        # Eigenvalues (sorted)
eig_vec = qh.eig_vec        # Eigenvectors (columns)
```

### Bogoliubov Transform (BdG)

```python
W, E, c = qh.diagonalizing_bogoliubov_transform()
# W: transformation matrix
# E: quasiparticle energies
# c: constant offset
```

### Time Evolution

```python
U_t = qh.time_evolution_operator(t=1.0)  # exp(-i*H*t)
```

### Thermal Properties

```python
qh.thermal_scan(T_max=10.0, steps=50)
occ_f = qh.fermi_occupation(E=1.0, T=2.0)    # Fermi-Dirac
occ_b = qh.bose_occupation(E=1.0, T=2.0)     # Bose-Einstein
```

## Interoperability

### Qiskit

```python
try:
    qiskit_op = qh.to_qiskit_hamiltonian()
    print(f"Qiskit operator: {qiskit_op}")
except ImportError:
    print("Install: pip install qiskit-nature")
```

### OpenFermion

```python
try:
    of_ham = qh.to_openfermion_hamiltonian()
    print(f"OpenFermion: {of_ham}")
except ImportError:
    print("Install: pip install openfermion")
```

## Common Patterns

### Ring Lattice (4 sites)

```python
qh = QuadraticHamiltonian(ns=4)
for i in range(4):
    j = (i + 1) % 4
    qh.add_hopping(i, j, -1.0)
qh.diagonalize()
```

### Chain with Disorder

```python
np.random.seed(42)
qh = QuadraticHamiltonian(ns=10)
for i in range(9):
    t = np.random.uniform(-1, -0.5)
    qh.add_hopping(i, i+1, t)
for i in range(10):
    epsilon = np.random.uniform(-0.5, 0.5)
    qh.add_onsite(i, epsilon)
qh.diagonalize()
```

### Superconductor with Gap

```python
qh = QuadraticHamiltonian(ns=6, particle_conserving=False)
# Hopping
for i in range(5):
    qh.add_hopping(i, i+1, -1.0)
# s-wave pairing
for i in range(6):
    j = (i + 1) % 6
    qh.add_pairing(i, j, 0.2)
qh.diagonalize()
```

## State Calculations

### Slater Determinant Amplitude

```python
from QES.Algebra.Hilbert.hilbert_jit_states import calculate_slater_det

qh.diagonalize()
U           = qh.eig_vec
occupied    = np.array([0, 1])      # Occupy orbitals 0,1
basis_state = 0b0011                # Fock: |1,1,0,0,0,0>
amp         = calculate_slater_det(U, occupied, basis_state, ns=6)
```

### Bogoliubov Ground State

```python
from QES.Algebra.Hilbert.hilbert_jit_states import calculate_bogoliubov_amp

qh.diagonalize()
F   = qh._F  # Pairing matrix from BdG diagonalization
amp = calculate_bogoliubov_amp(basis_state, F)
```

## Performance Tips

1. **Use lazy evaluation**      : Add hopping/pairing before diagonalize()
2. **Choose backend wisely**    : Use JAX for autodiff, NumPy for standard compute
3. **Sparse matrices**          : Framework exists but not yet enabled (TODO)
4. **Large systems**            : For ns > 100, use parallel diagonalization backend

## Error Handling

```python
try:
    qh = QuadraticHamiltonian(ns=-1)
except ValueError as e:
    print(f"Invalid: {e}")

try:
    qh.set_backend('unknown')
except ValueError as e:
    print(f"Unknown backend: {e}")
```

## Testing

Run full test suite:

```bash
cd /Users/makskliczkowski/Codes/pyqusolver
python Python/test_comprehensive_suite.py      # 9/9 tests
python Python/test_backends_interop.py         # 8/8 tests
```

## Documentation

- `hamil_quadratic.py`      - Inline docstrings for all methods
- `backends/base.py`        - Backend interface documentation

## Support

Key classes and methods:

- `QuadraticHamiltonian`    - Main class
- `get_backend()`           - Get backend by name
- `QiskitInterop`           - Qiskit conversions
- `OpenFermionInterop`      - OpenFermion conversions

## Version Info

- Python                    : 3.12+
- NumPy                     : Required
- SciPy                     : Required
- JAX                       : Optional (for autodiff)
- Qiskit Nature             : Optional (for interop)
- OpenFermion               : Optional (for interop)

---

For detailed usage, see the inline docstrings in the code.
Last updated: 2025-11-01
