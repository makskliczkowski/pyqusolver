# Getting started

This page gives a minimal, reliable path to run QES locally.

## 1) Install from `pyproject.toml`

From the pyqusolver repository root (the folder that contains `Python/`):

```bash
cd Python
python -m pip install -e .
```

If you are in the parent QuantumEigenSolver repository, use:

```bash
cd pyqusolver/Python
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
- If JAX is not installed, `import QES` still works and NumPy-based workflows remain available. JAX-specific execution fails only when those paths are used.

General contract:

- Inputs should be finite numeric arrays.
- Shapes should match operator or state definitions.
- Dtypes should be chosen consistently (real vs complex) across a pipeline.

## 3) Sanity-check imports

```bash
python -c "import QES; print('QES import OK')"
```

To confirm the optional NQS surface when the JAX stack is installed:

```bash
python -c "import QES; from QES.NQS import NQS; print('QES NQS import OK')"
```

Stable top-level imports are:

- `QES.Algebra`
- `QES.NQS`
- `QES.Solver`
- `QES.HilbertSpace`
- `QES.Hamiltonian`
- `QES.Operator`

Legacy `gp_*` aliases remain for compatibility, but they should be treated as deprecated convenience imports.

## 4) Run tests

From `pyqusolver/`:

```bash
PYTHONPATH=Python pytest Python/tests -q
```

To run a narrower subset:

```bash
PYTHONPATH=Python pytest Python/tests/algebra -q
```

## 5) Run examples

From `pyqusolver/`:

```bash
PYTHONPATH=Python python examples/run_all_examples.py
```

## 6) Build docs locally

From `Python/docs/`:

```bash
make html
```

Output is written to `Python/docs/_build/html`.
