# Installation

QES requires Python 3.10 or later. It supports both CPU and GPU execution (via JAX).

## Quick Install (Pip)

If you have the source tree, install from the `Python/` subdirectory:

```bash
cd Python
python -m pip install .
```

This gives you the lightweight minimal package. `import QES` works without pulling in the JAX stack.

Recommended standard install for NQS/VMC:

```bash
python -m pip install ".[standard]"
```

The `.[jax]` extra remains available as a compatible alias for the same accelerator-oriented stack.

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
   python -m pip install -e ".[standard,dev]"
   ```

   Note: The `Python/` subdirectory contains the installable package.

## Dependencies

Core dependencies are defined in `Python/pyproject.toml` and include:
- `numpy`, `scipy`, `matplotlib`
- `pandas`, `sympy`, `tqdm`, `numba`

Recommended standard extras:
- `standard` or `jax`: `jax`, `jaxlib`, `flax`, `optax` (for NQS/VMC and accelerator-ready workflows)

Additional optional extras:
- `hdf5`: `h5py` (for data storage)
- `ml`: `scikit-learn`, `scikit-image`
- `all`: full stack including JAX, docs, ML, HDF5, and dev tooling

## Verification

After installation, verify it works by importing QES:

```python
import QES
print(QES.__version__)
```

If you need the neural-network workflows, verify the JAX extras separately:

```python
import QES
from QES.NQS import NQS
print("QES and NQS import OK")
```
