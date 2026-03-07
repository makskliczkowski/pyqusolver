# Testing

Python tests are organized under `Python/tests/` by module area.

## Categories

- `Python/tests/algebra/`
- `Python/tests/core/`
- `Python/tests/lattices/`
- `Python/tests/models/`
- `Python/tests/nqs/`
- `Python/tests/physics/`
- `Python/tests/solvers/`

## Run all tests

```bash
cd pyqusolver
PYTHONPATH=Python pytest Python/tests -q
```

## Run one category

```bash
cd pyqusolver
PYTHONPATH=Python pytest Python/tests/physics -q
```
