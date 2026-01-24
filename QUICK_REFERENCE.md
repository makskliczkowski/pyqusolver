# QES Cheat Sheet

## 1. Setup & Configuration

**Standard Session:**
```python
import QES
# Recommended for scripts
with QES.run(backend='jax', seed=42, precision='float64'):
    ...
```

**Environment Variables:**
- `PY_BACKEND`: 'numpy' or 'jax'
- `PY_FLOATING_POINT`: 'float32' or 'float64'
- `QES_PYPATH`: Path to source (if not installed)

---

## 2. Exact Diagonalization (ED) Workflow

**Key Class:** `QES.Algebra.Hamiltonian`

**Minimal Snippet:**
```python
from QES.Algebra import Hamiltonian
H = Hamiltonian(ns=8, is_manybody=True)
H.add_heisenberg(0, 1, J=1.0)
H.build()
H.diagonalize(method='exact')  # or 'lanczos' for sparse
E_gs = H.eig_val[0]
```

**Knobs:**
- `ns`: System size.
- `is_sparse`: Set `True` (default) for sparse matrices.
- `method`: `'exact'` (dense), `'lanczos'` (sparse, extreme eigenvalues).
- `k`: Number of eigenvalues (for Lanczos).

**Common Pitfalls:**
- **Memory**: Dense diagonalization (`method='exact'`) scales as $2^{2N}$. Use Lanczos for $N > 14$.
- **Symmetries**: If `particle_conserving=True`, you must specify symmetries or ensure Hamiltonian conserves particles.

---

## 3. NQS / VMC Workflow

**Key Class:** `QES.NQS.NQS`

**Minimal Snippet:**
```python
from QES.NQS import NQS
# Assume H is defined
psi = NQS(logansatz='rbm', model=H, batch_size=512)
psi.train(n_epochs=200, lr=1e-2)
```

**Important Knobs (Training):**
- `logansatz`: `'rbm'`, `'cnn'`, `'dense'`, or custom Flax module.
- `sampler`: `'vmc'` (standard), `'exchange'` (particle conserving).
- `n_epochs`: Training duration.
- `lr`: Learning rate (try `1e-2` or `1e-3`).
- `n_batch`: Number of MC samples per step.

**Important Knobs (Sampling):**
- `num_chains`: Parallel MCMC chains (more = better decorrelation).
- `sweep_steps`: MC steps between samples (increase if autocorrelation high).
- `therm_steps`: Burn-in steps (increase if initialization bad).

**Debugging NQS:**
- **NaN Loss**: Reduce learning rate (`lr`). Check `precision='float64'`.
- **Stuck in Local Minima**: Use `sampler='exchange'` or Parallel Tempering (`replica > 1`).
- **JIT Errors**: Ensure input shapes match.

---

## 4. Time Evolution (TDVP)

**Key Method:** `psi.train(..., tdvp=True)`

**Snippet:**
```python
# Real-time evolution
stats = psi.train(
    n_epochs=1000,
    tdvp=True,
    rhs_prefactor=-1j,  # Imaginary time (optimization) is -1.0
    dt=0.01
)
```

---

## 5. Diagnostics & Analysis

**Observables:**
```python
# Compute arbitrary operator expectation
obs = psi.compute_observable(functions=[my_operator], names=['MyOp'])
print(obs['MyOp'].mean)
```

**Overlap / Fidelity:**
```python
# Overlap between two NQS states
overlap = NQS.compute_overlap(psi1, psi2)
```

**Renyi Entropy:**
```python
# Second Renyi entropy of subsystem A
S2 = NQS.compute_renyi2(psi, region=[0, 1, 2])
```
