# Installation

QES requires Python 3.10 or later. It supports both CPU and GPU execution (via JAX).

## Quick Install (Pip)

If you have the package source, you can install it using pip:

```bash
pip install .
```

To include all optional dependencies (docs, JAX, tests):

```bash
pip install ".[all]"
```

## Developer Install (Editable)

For researchers and developers who modify the code:

1. Clone the repository:
   ```bash
   git clone https://github.com/makskliczkowski/pyqusolver.git
   cd pyqusolver
   ```

2. Install in editable mode:
   ```bash
   pip install -e "Python/[all]"
   ```

   Note: The `Python/` subdirectory contains the actual package.

## Dependencies

Core dependencies:
- `numpy`, `scipy`
- `tqdm`, `colorama` (interface)

Optional but recommended:
- `jax`, `jaxlib`, `flax`, `optax` (for NQS/VMC and GPU support)
- `numba` (for optimized CPU operations)
- `h5py` (for data storage)
- `matplotlib` (for plotting)

## Verification

After installation, verify it works by importing QES:

```python
import QES
print(QES.__version__)
```
