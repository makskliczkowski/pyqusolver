# Repository Audit (non-`/general_python` scope)

## Scope and constraints

This audit covers the repository with explicit **exclusion of `/general_python` implementation changes**.
The package currently depends on that subtree for many runtime features, so compatibility-sensitive
imports are listed and treated as "do not break".

---

## 1) Current structure

### Top-level layout

- `Python/QES/` — installable Python package root.
- `Python/QES/Algebra/` — Hamiltonians, Hilbert spaces, operators, symmetries, and backend wrappers.
- `Python/QES/Solver/` — solver abstractions and Monte Carlo implementations.
- `Python/QES/NQS/` — neural quantum state API and training/evaluation internals.
- `docs/` — repository-level developer docs (examples/benchmarks/audit/future branches).
- `Python/docs/` — user/developer documentation site source (Sphinx + markdown/rst).

### Major package modules

- `QES` (top-level): lazy top-level namespace and compatibility aliases.
- `QES.Algebra`: algebraic core entry point.
- `QES.Solver`: solver package entry point.
- `QES.NQS`: NQS package entry point and training-facing API.

---

## 2) Public API surface (practical, compatibility-oriented)

The following objects/modules are currently used as public surface either by docs,
examples, or tests and should remain import-compatible.

### Top-level `QES`

- Modules: `QES.Algebra`, `QES.NQS`, `QES.Solver`.
- Common aliases exposed by `QES.__init__`: `Hamiltonian`, `HilbertSpace`, `Operator`,
  `NQS_Model`, `MonteCarloSolver`, `Sampler`.

### Algebra layer

- Main module imports:
  - `QES.Algebra.hamil`
  - `QES.Algebra.hilbert`
  - `QES.Algebra.hamil_quadratic`
  - `QES.Algebra.Hilbert.hilbert_local`
  - `QES.Algebra.Operator.operator`
  - `QES.Algebra.Operator.impl.*`
  - `QES.Algebra.Symmetries.*`

### Solver layer

- `QES.Solver.MonteCarlo.*` (diagnostics, sampler/vmc, solver base).

### NQS layer

- `QES.NQS` and `QES.NQS.nqs`
- `QES.NQS.src.*` for trainer/engine integrations used by internal tests.

---

## 3) Build + CI entry points

### Packaging/build

- `Python/pyproject.toml`: canonical package metadata/deps/tooling configuration.
- `Python/setup.py`: shim invoking setuptools build backend.

### Local automation

- `Python/Makefile`: install/test/lint/type-check/docs targets.
- `Python/tox.ini`: multi-env automation.
- `Python/check_import_hygiene.py`: import-path hygiene check script.

### CI workflow

- `.github/workflows/ci.yml`:
  - editable install (`pip install -e ".[all,dev]"` under `Python/`)
  - import smoke checks
  - Ruff linting
  - MyPy (non-blocking)
  - pytest run for `Python/test/`

---

## 4) "Do not break" import paths observed in tests/examples/docs

Observed via repository-wide search in `tests/`, `examples/`, `Python/test/`, and docs.

### Core package paths

- `import QES`
- `from QES.Algebra.hamil import Hamiltonian`
- `from QES.Algebra.hilbert import HilbertSpace`
- `from QES.Algebra.hamil_quadratic import QuadraticHamiltonian, QuadraticTerm`
- `from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType, LocalSpace`
- `from QES.Algebra.Operator.operator import ...`
- `from QES.Algebra.Operator.impl import operators_spin`
- `from QES.Algebra.Model.Interacting.Spin.* import ...`
- `from QES.Algebra.Symmetries.* import ...`
- `from QES.Solver.MonteCarlo.diagnostics import ...`
- `from QES.NQS import NQS`
- `from QES.NQS.nqs import NQS`
- `from QES.NQS.src.nqs_train import ...`

### Compatibility-sensitive (`general_python`) paths (conceptual only in this task)

- `from QES.general_python.lattices import ...`
- `from QES.general_python.algebra... import ...`
- `from QES.general_python.physics... import ...`
- `from QES.general_python.common... import ...`

Because `/general_python` is out of edit scope, all such imports are treated as hard compatibility constraints.

---

## 5) Modularization plan (compatibility-preserving)

### Goals

1. Keep `import QES` and package submodule import cost low.
2. Preserve historical import paths used by tests/examples/docs.
3. Replace eager/wildcard package imports with lazy, explicit re-exports where safe.
4. Document module invariants and I/O/dtype expectations for correctness-driven usage.

### Safe plan

- Convert package `__init__.py` files to **PEP 562 lazy attribute loading**.
- Re-export only compatibility-critical symbols and modules.
- Keep compatibility fallback for `QES.Algebra` symmetry symbols by resolving unknown
  attributes against `QES.Algebra.Symmetries`.
- Avoid renaming directories/modules in this branch.

### Deferred (future branch) plan

- Any package/directory renames (`Hamil` -> `Hamiltonian`, solver tree moves, general_python split)
  are deferred to dedicated breaking-change branches and tracked in `docs/FUTURE_BRANCHES.md`.

---

## 6) Correctness documentation priorities

For maintainers and users, major modules should document:

- expected state/config input types,
- array shape conventions (batch/chain/site axes),
- dtype expectations (real vs complex amplitudes),
- determinism/reproducibility assumptions under fixed seeds,
- symmetry-sector invariants and basis indexing invariants.

These are now reflected in refreshed module docstrings and roadmap docs in this branch.
