# QES Development Configuration

## Overview

This directory contains configuration files for development tools used in the QES project.

## Files

- `.flake8` - Flake8 linting configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `pytest.ini` - Pytest testing configuration
- `mypy.ini` - MyPy type checking configuration
- `tox.ini` - Tox testing configuration

## Setup Development Environment

1. Install development dependencies:

```bash
pip install -r requirements/requirements-dev.txt
```

1. Install pre-commit hooks:

```bash
pre-commit install
```

1. Run tests:

```bash
pytest
```

1. Run type checking:

```bash
mypy QES
```

1. Run linting:

```bash
flake8 QES
```

1. Format code:

```bash
black QES
isort QES
```
