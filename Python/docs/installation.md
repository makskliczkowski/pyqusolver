# Installation

QES requires Python 3.10 or later. It supports both CPU and GPU execution (via JAX).

## Quick Install (Pip)

If you have the source tree, install from the `Python/` subdirectory:

```bash
cd Python
python -m pip install .
```

To include all optional dependencies (docs, JAX, tests):

```bash
python -m pip install ".[all]"
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
   cd Python
   python -m pip install -e ".[all]"
   ```

   Note: The `Python/` subdirectory contains the installable package.

## Dependencies

Core dependencies are defined in `Python/pyproject.toml` and include:
- `numpy`, `scipy`, `matplotlib`
- `pandas`, `sympy`, `tqdm`, `numba`

Optional but recommended (see `.[all]` extras):
- `jax`, `jaxlib`, `flax`, `optax` (for NQS/VMC and GPU support)
- `h5py` (for data storage)

## Verification

After installation, verify it works by importing QES:

```python
import QES
print(QES.__version__)
```
