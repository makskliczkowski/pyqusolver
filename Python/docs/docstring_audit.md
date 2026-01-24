# Docstring Audit Report

## Public API Analysis

The public API is defined in `Python/QES/__init__.py`. The following modules and classes are exposed.

### 1. Root Level (`QES`)

| Object | Type | Docstring Status | Notes |
| :--- | :--- | :--- | :--- |
| `QESSession` | Class | ⚠️ Weak | Missing details on `num_threads` limitations and environment variable side effects. |
| `run` | Function | ⚠️ Weak | Brief. Should explain the context manager behavior more clearly. |
| `qes_globals` | Module | ✅ Adequate | Good overview. |
| `get_logger` | Function | ⚠️ Weak | Return type not explicit. |
| `get_backend_manager` | Function | ⚠️ Weak | Return type not explicit. |
| `get_numpy_rng` | Function | ⚠️ Weak | Return type not explicit. |
| `reseed_all` | Function | ⚠️ Weak | Return type not explicit. |

### 2. Algebra (`QES.Algebra`)

| Object | Type | Docstring Status | Notes |
| :--- | :--- | :--- | :--- |
| `Hamiltonian` | Class | ✅ Adequate | Comprehensive. `loc_energy_int` could use specific array shape details in Returns. |
| `HilbertSpace` | Class | ✅ Adequate | Very detailed. |
| `Operator` | Class | ⚠️ Weak | (Inferred) Needs check on `Operator` base class in `operator.py`. |

### 3. NQS (`QES.NQS`)

| Object | Type | Docstring Status | Notes |
| :--- | :--- | :--- | :--- |
| `NQS` | Class | ✅ Adequate | Very detailed `__init__`. |
| `NQS.train` | Method | ✅ Adequate | Detailed parameters. |
| `NQS.sample` | Method | ⚠️ Weak | Return type tuple structure should be explicitly defined (shapes, types). |
| `NQS.evaluate` | Method | ⚠️ Weak | Return type/shape not explicit. |

### 4. Solver (`QES.Solver`)

| Object | Type | Docstring Status | Notes |
| :--- | :--- | :--- | :--- |
| `MonteCarloSolver` | Class | ⚠️ Weak | Inconsistent style, missing explicit return types for `save_weights`, `load_weights`. |
| `Sampler` | Class | ⚠️ Weak | (Inferred) Needs check. `sample` method is critical. |

## Action Plan

1.  **QES Root**: Update `QESSession`, `run`, and `qes_globals` functions to NumPy style with explicit returns.
2.  **Algebra**: Minor touch-up on `Hamiltonian.loc_energy_*` return descriptions.
3.  **NQS**: Clarify return shapes for `sample` and `evaluate`.
4.  **Solver**: Standardize `MonteCarloSolver` docstrings.
