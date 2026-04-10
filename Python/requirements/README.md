# QES Requirements Files

This directory contains various requirements files for different installation scenarios:

## Core Requirements

- `requirements.txt` - Minimal dependencies for basic functionality
- `requirements-standard.txt` - Recommended standard stack: core + JAX/Flax/Optax
- `requirements-all.txt` - Full stack installation, including JAX, ML, HDF5, docs, and dev tools

## Optional Requirements

- `requirements-jax.txt` - JAX ecosystem dependencies for GPU/TPU acceleration
- `requirements-ml.txt` - Machine learning utilities
- `requirements-hdf5.txt` - HDF5 file format support
- `requirements-dev.txt` - Development tools and testing
- `requirements-docs.txt` - Documentation generation tools

## Installation Examples

### Minimal installation

```bash
pip install -r requirements.txt
```

### Standard installation

```bash
pip install -r requirements-standard.txt
```

### Development environment

```bash
pip install -r requirements-standard.txt
pip install -r requirements-dev.txt
```

### Full installation with all features

```bash
pip install -r requirements-all.txt
```

### Extras via `pip`

```bash
pip install QES
pip install "QES[standard]"
pip install "QES[jax]"
pip install "QES[all]"
```

Install policy:

- `pip install QES` gives the minimal runtime without requiring JAX/Flax.
- `pip install "QES[standard]"` is the recommended standard install and includes JAX/Flax/Optax.
- `pip install "QES[jax]"` remains a compatible alias for the same accelerator-oriented stack.
- `pip install "QES[all]"` remains the explicit "install everything" path and includes the JAX stack.

## Troubleshooting

### JAX plugin error with `jax_plugins/~la_cuda12`

If you see errors like:

`Jax plugin configuration error: ... jax_plugins.~la_cuda12.initialize()`

or

`JaxRuntimeError: ALREADY_EXISTS: PJRT_Api already exists for device type cuda`

your virtual environment likely contains leftover temporary package folders
from an interrupted `pip` operation (directories starting with `~`).

Clean those directories in the active venv:

```bash
find <venv_path>/lib/python*/site-packages -maxdepth 1 -name '~*' -type d -exec rm -rf {} +
```

For CPU-only jobs, also export:

```bash
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
```
