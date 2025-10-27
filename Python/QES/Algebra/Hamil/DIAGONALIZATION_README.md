# Modular Diagonalization System for Hamiltonians

This document describes the modular, flexible diagonalization system implemented for the `Hamiltonian` class in the QES package.

## Overview

The new diagonalization system provides:

- **Multiple Methods**      : Exact, Lanczos, Block Lanczos, and Arnoldi solvers
- **Automatic Selection**   : Intelligent method selection based on matrix properties
- **Krylov Basis Support**  : Store and transform between Krylov and original bases
- **Backend Flexibility**   : Support for NumPy, SciPy, and JAX backends
- **Standard API**          : Interface similar to SciPy and other scientific libraries
- **Detailed Diagnostics**  : Convergence information, residuals, iteration counts

## Quick Start

### Basic Usage

```python
from QES.Algebra.hamil import Hamiltonian

# Create Hamiltonian
H               = Hamiltonian(is_manybody=True, ns=12, ...)
H.build()

# Auto-select method and diagonalize
H.diagonalize(verbose=True)

# Access results
eigenvalues     = H.get_eigval()
eigenvectors    = H.get_eigvec()
```

### Specify Method

```python
# Exact diagonalization (all eigenvalues)
H.diagonalize(method='exact')

# Lanczos for 10 smallest eigenvalues
H.diagonalize(method='lanczos', k=10, which='smallest')

# Block Lanczos for clustered eigenvalues
H.diagonalize(method='block_lanczos', k=20, block_size=5)
```

## Available Methods

### 1. Auto (`method='auto'`)

Automatically selects the best method based on:

- Matrix dimension (n)
- Number of eigenvalues needed (k)
- Matrix properties (symmetric/Hermitian)

**Selection Logic**:

- `n ≤ 500`                         : Use `exact`
- `n > 500`, symmetric, `k` small   : Use `lanczos`
- `n > 500`, symmetric, `k` large   : Use `block_lanczos`
- `n > 500`, non-symmetric          : Use `arnoldi`

### 2. Exact (`method='exact'`)

Full diagonalization computing all eigenvalues.

**Use when**:

- Need complete spectrum
- Matrix is small (n < 1000)
- Memory is not a constraint

**Parameters**:

```python
H.diagonalize(
    method      = 'exact',
    hermitian   = True,         # Symmetric/Hermitian matrix
    backend     = 'numpy'       # 'numpy', 'scipy', or 'jax'
)
```

### 3. Lanczos (`method='lanczos'`)

Iterative method for sparse symmetric/Hermitian matrices.

**Use when**:

- Need few extremal eigenvalues
- Matrix is large and sparse
- Memory-efficient computation required

**Parameters**:

```python
H.diagonalize(
    method          = 'lanczos',
    k               = 10,           # Number of eigenvalues
    which           = 'smallest',   # 'smallest', 'largest', 'both'
    tol             = 1e-10,        # Convergence tolerance
    max_iter        = None,         # Maximum iterations (auto if None)
    reorthogonalize = True,
    store_basis     = True          # Store Krylov basis
)
```

### 4. Block Lanczos (`method='block_lanczos'`)

Block version of Lanczos for multiple eigenpairs simultaneously.

**Use when**:

- Need many eigenvalues
- Eigenvalues are clustered or degenerate
- Better convergence desired

**Parameters**:

```python
H.diagonalize(
    method='block_lanczos',
    k=20,               # Number of eigenvalues
    block_size=5,       # Vectors per block
    which='smallest',   # 'smallest', 'largest'
    tol=1e-10,
    reorthogonalize=True,
    store_basis=True
)
```

### 5. Arnoldi (`method='arnoldi'`)

Iterative method for general (non-symmetric) matrices.

**Use when**:

- Matrix is not symmetric
- Need extremal eigenvalues of large sparse matrix

**Parameters**:

```python
H.diagonalize(
    method='arnoldi',
    k=10,
    which='LM',  # 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
    tol=1e-10,
    max_iter=None
)
```

## Krylov Basis and Transformations

When using iterative methods (Lanczos, Block Lanczos, Arnoldi), the computation is performed in a reduced Krylov subspace. The system provides utilities to transform between this subspace and the original Hilbert space.

### Check Basis Availability

```python
H.diagonalize(method='lanczos', k=10, store_basis=True)

if H.has_krylov_basis():
    print("Krylov basis available")
```

### Get Krylov Basis

```python
V = H.get_krylov_basis()  # Returns n x k matrix
```

### Transform to Original Basis

```python
# Ritz vector in Krylov basis (k-dimensional)
ritz_vec = np.zeros(10)
ritz_vec[0] = 1.0  # Ground state in Krylov basis

# Transform to full Hilbert space (n-dimensional)
state = H.to_original_basis(ritz_vec)
```

### Transform to Krylov Basis

```python
# Arbitrary state in full Hilbert space
state = np.random.randn(H.nh)
state /= np.linalg.norm(state)

# Project onto Krylov subspace
krylov_coeffs = H.to_krylov_basis(state)
```

### Manual Transformation

```python
V = H.get_basis_transform()  # Get transformation matrix

# Krylov -> Original: v_original = V @ v_krylov
v_original = V @ v_krylov

# Original -> Krylov: v_krylov = V.H @ v_original
v_krylov = V.conj().T @ v_original
```

## Diagnostics and Information

### Get Diagonalization Method

```python
method = H.get_diagonalization_method()
print(f"Used method: {method}")
```

### Get Detailed Information

```python
info = H.get_diagonalization_info()

print(f"Method: {info['method']}")
print(f"Converged: {info['converged']}")
print(f"Iterations: {info['iterations']}")
print(f"Eigenvalues computed: {info['num_eigenvalues']}")
print(f"Has Krylov basis: {info['has_krylov_basis']}")

if info['residual_norms'] is not None:
    print(f"Max residual: {np.max(info['residual_norms']):.10e}")
```

## Backend Selection

### NumPy Backend

```python
H.diagonalize(method='exact', backend='numpy')
```

### SciPy Backend

```python
H.diagonalize(method='lanczos', backend='scipy', use_scipy=True)
```

### JAX Backend (if available)

```python
H.diagonalize(method='exact', backend='jax')
```

## Examples

### Example 1: Small System - Exact Diagonalization

```python
# Small Hamiltonian (n < 500)
H = Hamiltonian(is_manybody=True, ns=10, ...)
H.build()

# Compute all eigenvalues
H.diagonalize(method='exact', verbose=True)

evals = H.get_eigval()
print(f"Ground state energy: {evals[0]}")
print(f"First gap: {evals[1] - evals[0]}")
```

### Example 2: Large Sparse System - Lanczos

```python
# Large sparse Hamiltonian
H = Hamiltonian(is_manybody=True, ns=16, is_sparse=True, ...)
H.build()

# Compute 10 smallest eigenvalues
H.diagonalize(
    method='lanczos',
    k=10,
    which='smallest',
    tol=1e-10,
    verbose=True
)

# Check convergence
info = H.get_diagonalization_info()
if info['converged']:
    print(f"Converged in {info['iterations']} iterations")
```

### Example 3: Degenerate Spectrum - Block Lanczos

```python
# System with degeneracies
H = Hamiltonian(is_manybody=True, ns=14, ...)
H.build()

# Use Block Lanczos with larger block
H.diagonalize(
    method='block_lanczos',
    k=20,
    block_size=5,
    which='smallest',
    reorthogonalize=True,
    verbose=True
)

# Identify degeneracies
evals = H.get_eigval()
for i in range(1, len(evals)):
    gap = evals[i] - evals[i-1]
    if gap < 1e-8:
        print(f"Degeneracy at i={i}: E={evals[i]:.10f}")
```

### Example 4: Basis Transformations

```python
# Diagonalize with Krylov basis storage
H.diagonalize(method='lanczos', k=10, store_basis=True)

# Get ground state in Krylov basis
gs_krylov = np.zeros(10)
gs_krylov[0] = 1.0

# Transform to original basis
gs_original = H.to_original_basis(gs_krylov)

# Verify it's an eigenstate
H_gs = H._hamil @ gs_original
E_gs = H.get_eigval(0)
residual = np.linalg.norm(H_gs - E_gs * gs_original)
print(f"Residual: {residual:.10e}")
```

### Example 5: Method Comparison

```python
import time

H = Hamiltonian(is_manybody=True, ns=12, ...)
H.build()

methods = ['exact', 'lanczos', 'block_lanczos']
results = {}

for method in methods:
    t0 = time.perf_counter()
    H.diagonalize(method=method, k=10 if method != 'exact' else None)
    t1 = time.perf_counter()
    
    results[method] = {
        'time': t1 - t0,
        'E0': H.get_eigval(0),
        'info': H.get_diagonalization_info()
    }

for method, res in results.items():
    print(f"{method:15s}: {res['time']:.4f}s, E0={res['E0']:.10f}")
```

## Advanced Features

### Custom Convergence Criteria

```python
H.diagonalize(
    method='lanczos',
    k=10,
    tol=1e-12,              # Tighter tolerance
    max_iter=1000,          # More iterations
    reorthogonalize=True    # Better numerical stability
)
```

### Accessing Raw Results

```python
# Get the DiagonalizationEngine directly
engine = H._diag_engine

# Access raw EigenResult
result = engine.get_result()
print(result.eigenvalues)
print(result.eigenvectors)
print(result.converged)
print(result.iterations)
```

### Working with Reduced Hamiltonian

```python
# Diagonalize with Krylov basis
H.diagonalize(method='lanczos', k=10, store_basis=True)

# Get Krylov basis
V = H.get_krylov_basis()

# Build reduced Hamiltonian in Krylov basis
H_reduced = V.conj().T @ H._hamil @ V

# Reduced problem eigenvalues should match
evals_reduced = np.linalg.eigvalsh(H_reduced)
evals_full = H.get_eigval()
assert np.allclose(evals_reduced, evals_full)
```

## Performance Considerations

### Memory Usage

| Method | Memory | Speed | Best For |
|--------|--------|-------|----------|
| Exact | O(n^2) | Fast for small n | n < 1000, need all eigenvalues |
| Lanczos | O(nk) | Fast | Large sparse, few eigenvalues |
| Block Lanczos | O(n·k·b) | Moderate | Many eigenvalues, degeneracies |
| Arnoldi | O(nk) | Moderate | Non-symmetric matrices |

where:

- n = matrix dimension
- k = number of eigenvalues
- b = block size

### Method Selection Guidelines

**Use Exact when**:

- n < 500
- Need complete spectrum
- Memory is available

**Use Lanczos when**:

- n > 500
- Need k << n eigenvalues
- Matrix is sparse and symmetric

**Use Block Lanczos when**:

- Need many eigenvalues (k ~ 10-50)
- Eigenvalues are clustered
- Better convergence needed

**Use Arnoldi when**:

- Matrix is non-symmetric
- Need extremal eigenvalues

## API Reference

### Diagonalization

```python
Hamiltonian.diagonalize(
    verbose: bool = False,
    **kwargs
) -> None
```

**Parameters**:

- `method`: str - Diagonalization method
- `k`: int - Number of eigenvalues
- `which`: str - Which eigenvalues
- `tol`: float - Convergence tolerance
- `max_iter`: int - Maximum iterations
- `backend`: str - Computational backend
- `store_basis`: bool - Store Krylov basis

### Basis Transformations

```python
Hamiltonian.has_krylov_basis() -> bool
Hamiltonian.get_krylov_basis() -> Optional[np.ndarray]
Hamiltonian.to_original_basis(vec: np.ndarray) -> np.ndarray
Hamiltonian.to_krylov_basis(vec: np.ndarray) -> np.ndarray
Hamiltonian.get_basis_transform() -> Optional[np.ndarray]
```

### Diagnostics

```python
Hamiltonian.get_diagonalization_method() -> Optional[str]
Hamiltonian.get_diagonalization_info() -> dict
```

## Error Handling

```python
try:
    H.diagonalize(method='lanczos', k=10)
except RuntimeError as e:
    print(f"Diagonalization failed: {e}")
    
    # Try alternative method
    H.diagonalize(method='exact')
```

## Backward Compatibility

The new system is fully backward compatible. Existing code continues to work:

```python
# Old style (still works)
H.diagonalize()  # Uses default method='auto'
evals = H.get_eigval()
evecs = H.get_eigvec()
```

## References

- Golub & Van Loan, "Matrix Computations" (4th ed.)
- Saad, "Numerical Methods for Large Eigenvalue Problems"
- SciPy documentation: `scipy.sparse.linalg.eigsh`

---

**Last Updated**    : 2025-10-26  
**Author**          : Maksymilian Kliczkowski  
**Version**         : 1.0.0

---