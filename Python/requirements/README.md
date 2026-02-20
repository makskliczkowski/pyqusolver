# QES Requirements Files

This directory contains various requirements files for different installation scenarios:

## Core Requirements

- `requirements.txt` - Core dependencies for basic functionality

## Optional Requirements

- `requirements-jax.txt` - JAX ecosystem dependencies for GPU/TPU acceleration
- `requirements-ml.txt` - Machine learning utilities
- `requirements-hdf5.txt` - HDF5 file format support
- `requirements-dev.txt` - Development tools and testing
- `requirements-docs.txt` - Documentation generation tools

## Installation Examples

### Basic installation

```bash
pip install -r requirements.txt
```

### Development environment

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Full installation with all features

```bash
pip install -r requirements.txt
pip install -r requirements-jax.txt
pip install -r requirements-ml.txt
pip install -r requirements-hdf5.txt
```

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
