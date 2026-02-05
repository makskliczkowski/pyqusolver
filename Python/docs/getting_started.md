# Getting started

This page gives a minimal, reliable path to run QES locally.

## 1) Install from `pyproject.toml`

From the repository root:

```bash
cd Python
python -m pip install -e .
```

Optional feature sets from `pyproject.toml`:

- JAX stack:

```bash
python -m pip install -e ".[jax]"
```

- Documentation stack:

```bash
python -m pip install -e ".[docs]"
```

- Full development stack:

```bash
python -m pip install -e ".[dev]"
```

## 2) Backend expectations (NumPy vs JAX)

- **NumPy backend** is the safest default for broad compatibility.
- **JAX backend** enables accelerator-ready and differentiable workflows, mainly for NQS/VMC paths.
- If JAX is not installed, JAX-specific modules are expected to be unavailable.

General contract:

- Inputs should be finite numeric arrays.
- Shapes should match operator or state definitions.
- Dtypes should be chosen consistently (real vs complex) across a pipeline.

## 3) Sanity-check imports

```bash
python -c "import QES; print('QES import OK')"
```

## 4) Run tests

From `Python/`:

```bash
pytest
```

To run a narrower subset:

```bash
pytest test/test_imports_lightweight.py -q
```

## 5) Build docs locally

From `Python/docs/`:

```bash
make html
```

Output is written to `Python/docs/_build/html`.
