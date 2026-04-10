# QES (Quantum Eigen Solver)

## Minimal Installation

```bash
pip install QES
```

## Standard Installation

```bash
pip install "QES[standard]"
```

## Development Installation

```bash
# Clone the repository
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver/pyqusolver/Python

# Install in development mode with the recommended standard stack
pip install -e ".[standard,dev]"
```

## Optional Dependencies

### JAX Support (GPU/CPU acceleration)

```bash
pip install "QES[jax]"
```

### Standard Stack

```bash
pip install "QES[standard]"
```

### Machine Learning Utilities

```bash
pip install "QES[ml]"
```

### HDF5 File Support

```bash
pip install "QES[hdf5]"
```

### Development Tools

```bash
pip install "QES[dev]"
```

### Documentation Tools

```bash
pip install "QES[docs]"
```

### All Optional Dependencies

```bash
pip install "QES[all]"
```

## Quick Start

```python
import QES

# Lightweight top-level import
print(QES.__version__)
```

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- See `pyproject.toml` for the full dependency list

`import QES` works without JAX installed. Use the minimal install when you only need the core package. Use `QES[standard]` for the recommended JAX/Flax-enabled stack, or `QES[all]` for everything.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License.

## Contributing

Please see the main repository for contributing guidelines.
