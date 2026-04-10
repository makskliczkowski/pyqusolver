# QES Package Structure and Setup

## Overview

This document describes the package structure and setup for the QES (Quantum Eigen Solver) Python package.

## Package Structure

```bash
Python/
├── QES/                         # Main package directory
│   ├── __init__.py              # Curated top-level API with lazy imports
│   ├── Algebra/                 # Algebraic operations and models
│   ├── NQS/                     # Neural Quantum States implementation
│   ├── Solver/                  # Solver framework
│   └── general_python/          # General utilities
├── tests/                       # Maintained package-level tests
├── requirements/                # Requirements files for different scenarios
│   ├── requirements.txt         # Core dependencies
│   ├── requirements-standard.txt# Recommended standard stack with JAX
│   ├── requirements-jax.txt     # JAX ecosystem
│   ├── requirements-ml.txt      # ML utilities
│   ├── requirements-hdf5.txt    # HDF5 support
│   ├── requirements-dev.txt     # Development tools
│   ├── requirements-docs.txt    # Documentation tools
│   └── requirements-all.txt     # Full stack including JAX and dev/docs
├── setup.py                     # Legacy setup entry point
├── pyproject.toml               # Modern Python packaging configuration
├── MANIFEST.in                  # Files to include in distribution
├── INSTALL.md                   # Installation instructions
├── PACKAGE_STRUCTURE.md         # This document
├── Makefile                     # Development automation
└── tox.ini                      # Testing across Python versions
```

## Installation Methods

### 1. Minimal Installation

```bash
pip install QES
```

This installs the lightweight core package. `import QES` does not require the JAX stack.

### 2. Standard Installation

```bash
pip install "QES[standard]"
```

This is the recommended install for most users and includes JAX, JAXlib, Flax, and Optax.

### 3. Development Installation

```bash
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver/pyqusolver/Python
pip install -e ".[standard,dev]"
```

### 4. Specific Feature Sets

```bash
pip install "QES[standard]" # Recommended JAX-enabled stack
pip install "QES[jax]"     # JAX support
pip install "QES[ml]"      # ML utilities
pip install "QES[all]"     # Everything
```

## Development Workflow

### Setup

```bash
make dev-setup           # Sets up development environment
```

### Daily Development

```bash
make dev                 # Format, lint, and test
make check              # Full pre-commit checks
```

### Testing

```bash
make test               # Quick tests
make test-all           # Test across Python versions
```

### Documentation

```bash
make docs               # Build documentation
```

### Release

```bash
make build              # Build distribution packages
make upload             # Upload to PyPI (test)
```

## Key Features

### 1. Modern Python Packaging

- Uses `pyproject.toml` for configuration
- Proper dependency management with extras
- Lightweight core install plus optional JAX / ML / HDF5 / docs extras

### 2. Code Quality Tools

- Black for code formatting
- Flake8 for linting
- MyPy for type checking
- isort for import sorting
- Pre-commit hooks for automation

### 3. Testing Infrastructure

- Pytest for testing
- Tox for multi-Python testing
- Coverage reporting
- Test discovery for multiple test directories

### 4. Documentation

- Sphinx for documentation generation
- Read the Docs theme
- Jupyter notebook integration via nbsphinx

### 5. Development Automation

- Makefile for common tasks
- Pre-commit hooks
- Continuous integration ready

## Optional Dependencies Explanation

- **standard**: Recommended install target, includes `jax`, `jaxlib`, `flax`, and `optax`
- **jax**: Compatible alias for the JAX/Flax/Optax stack
- **ml**: Machine learning utilities (scikit-learn, scikit-image)
- **hdf5**: HDF5 file format support for large datasets
- **dev**: Development tools (testing, linting, formatting)
- **docs**: Documentation generation tools
- **all**: All optional dependencies combined, including JAX

## Best Practices Implemented

1. **Semantic Versioning**: Following semver.org
2. **PEP 517/518**: Modern build system
3. **PEP 621**: Project metadata in `pyproject.toml`
4. **Type Hints**: MyPy configuration for type safety and IntelliSense support
5. **Code Formatting**: Black and Ruff for consistency
6. **Documentation**: Comprehensive docs and examples

This setup provides a professional, maintainable, and scalable Python package structure suitable for scientific computing and research applications.
