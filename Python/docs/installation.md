# Installation

Getting started with QES is straightforward. We offer a few installation options depending on whether you just want to run simulations or if you plan to modify the codebase.

## Prerequisites

QES requires **Python 3.8** or newer. We recommend using a virtual environment (like `venv` or `conda`) to keep your dependencies clean.

## Standard Installation

If you want to use QES as a library for your projects, the easiest way is to install it via pip.

```bash
pip install QES
```

This installs the core package with standard dependencies (NumPy, SciPy, etc.).

## Development Installation

For researchers and developers who want to explore the source code, run examples, or contribute features, we recommend an "editable" installation. This allows changes in the source code to be immediately reflected without re-installing.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
    cd QuantumEigenSolver/Python
    ```

2.  **Install in editable mode:**

    ```bash
    pip install -e .
    ```

## Optional Dependencies

QES is modular, and some features require extra packages. You can install these "extras" based on your needs:

*   **JAX Support** (highly recommended for performance and neural networks):
    ```bash
    pip install ".[jax]"
    ```
    *Note: If you need GPU support for JAX, please follow the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for your specific CUDA version.*

*   **Machine Learning Utilities** (scikit-learn, etc.):
    ```bash
    pip install ".[ml]"
    ```

*   **HDF5 Support** (for handling large datasets):
    ```bash
    pip install ".[hdf5]"
    ```

*   **Development Tools** (testing, linting):
    ```bash
    pip install ".[dev]"
    ```

*   **Documentation Tools**:
    ```bash
    pip install ".[docs]"
    ```

*   **Install Everything**:
    ```bash
    pip install ".[all]"
    ```

## Verifying the Installation

To ensure everything is working correctly, you can run a simple import test in Python:

```python
import QES
print(f"QES version {QES.__version__} is installed!")
```

If you installed the development dependencies, you can also run the test suite:

```bash
pytest
```

## Environment Setup (Optional)

For advanced usage, you might want to control runtime behavior using environment variables.

*   `QES_PYPATH`: Points to your QES installation directory (useful if not installed via pip).
*   `PY_BACKEND`: Sets the default numerical backend (`numpy` or `jax`).
*   `PY_NUM_CORES`: Limits the number of CPU cores QES uses.

Example:
```bash
export PY_BACKEND=jax
export PY_NUM_CORES=8
```
