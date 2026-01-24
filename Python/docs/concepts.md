# Key Concepts

## Backends: NumPy vs. JAX

QES is designed to be backend-agnostic.

- **NumPy (`backend='numpy'`)**:
  - Best for: Exact Diagonalization, small systems, debugging, and CPU-only workflows.
  - Uses `scipy.sparse` for efficient matrix operations.
  - Default for most Algebraic operations.

- **JAX (`backend='jax'`)**:
  - Best for: Neural Quantum States (NQS), Variational Monte Carlo, large batches, and GPU execution.
  - Enables Automatic Differentiation (AD) for gradients.
  - Requires `jax` and `flax` installed.

You can switch backends using the session manager:
```python
import QES
with QES.run(backend='jax'):
    # ... code uses JAX ...
```

## Global State (`qes_globals`)

To ensure reproducibility and correct logging across the entire library, QES uses a singleton pattern for global state.

- **`QES.get_backend_manager()`**: Handles the active backend and RNGs.
- **`QES.get_logger()`**: A unified logger that handles output formatting and verbosity.
- **`QES.get_numpy_rng()`**: The global NumPy random generator.

**Do not** create your own random generators if you want reproducibility. Use the globals.

## Seeding & Reproducibility

Reproducibility is enforced via the session or explicit seeding.

```python
# Sets seeds for NumPy, JAX, and Python random
with QES.run(seed=123):
    ...
```

Inside the code, `QES` uses these seeded generators. If you write custom layers or samplers, fetch the RNG from `get_backend_manager()`.

## Precision

You can control floating point precision globally. This affects network weights, wavefunction storage, and matrix elements.

- `'float32'` (Single): Faster, less memory. Standard for deep learning (NQS).
- `'float64'` (Double): Higher precision. Standard for Exact Diagonalization.

```python
with QES.run(precision='float64'):
    ...
```
