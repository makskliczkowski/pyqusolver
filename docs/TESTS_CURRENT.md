# Current Test Suite Status

This document summarizes the state of the test suite as of the latest engineering session.

## Test Files Overview

The following tests are located in `Python/test/`.

### Core Tests

| Test File | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| `test_hilbert_symmetries.py` | Verifies Hilbert space symmetry handling, sector construction, and Hamiltonian invariance (TFIM). | **FAILING** | **Critical Numba SegFault**. Logic bugs fixed (SymmetryContainer, ReflectionSymmetry, Hamiltonian API), but Numba crashes during matrix construction for spin operators. |
| `test_xxz_symmetries.py` | Verifies symmetries in the XXZ model (Translation, Parity, U(1)). | **FAILING** | **Critical Numba SegFault**. Fixed `AttributeError` (method call) and `RuntimeError` (missing instruction setup), but now hits the same Numba crash as TFIM. |
| `test_nqs_invariants.py` | Tests Neural Quantum State invariants. | Unknown | Not executed in this session. |
| `test_nqs_jax.py` | Tests JAX-based NQS implementations. | Unknown | Not executed in this session. |
| `test_determinism.py` | Verifies deterministic behavior of algorithms/RNG. | Unknown | Not executed in this session. |
| `test_auto_tuner_integration.py` | Tests TDVP auto-tuning integration. | Unknown | Not executed in this session. |
| `test_backend_integration.py` | Tests backend switching (NumPy/JAX). | Unknown | |
| `test_backends_interop.py` | Tests interoperability between backends. | Unknown | |
| `test_comprehensive_suite.py` | Large suite aggregating multiple checks. | Unknown | |
| `test_diagnostics.py` | Tests MCMC diagnostics (R-hat, ESS). | Unknown | |
| `test_eigen.py` | Tests diagonalization routines. | Unknown | |
| `test_hamil_optimization.py` | Tests Hamiltonian performance/optimizations. | Unknown | |
| `test_imports_lightweight.py` | Checks import time/dependencies. | Unknown | |
| `test_kitaev_symmetries.py` | Tests Kitaev model symmetries. | Unknown | Likely relies on similar operator logic as TFIM/XXZ. |
| `test_matvec_optimization.py` | Tests matrix-vector product optimizations. | Unknown | |
| `test_quadratic_hamiltonian_lazy.py`| Tests lazy evaluation for quadratic Hamiltonians. | Unknown | |
| `test_spectral_backend.py` | Tests spectral function calculations. | Unknown | |

### Known Critical Issues

#### 1. Numba JIT Segmentation Fault
**Symptoms:** `pytest` crashes with `Aborted` (SegFault) during test execution, specifically when building Hamiltonian matrices using `operators_spin.py` kernels.
**Location:** `Python/QES/Algebra/Operator/impl/operators_spin.py` -> `sigma_composition_integer` / `sigma_operator_composition_single_op`.
**Diagnosis:** The crash occurs inside the compiled Numba function execution or during late-stage compilation. It persists despite:
- Fixing `dtype` unification (float/complex) in `operators_spin.py`.
- Adding `inline='always'` to kernel functions.
- Ensuring correct operator instruction setup (`setup_instruction_codes`).
**Affected Modules:** `TransverseFieldIsing`, `XXZ`, and likely any model using `SpecialOperator` spin instructions.
**Workaround:** None currently. Requires deep debugging of Numba LLVM IR or environment checks (Numba version compatibility).

#### 2. Hamiltonian API Inconsistencies
**Issue:** `Operator.matrix` is a `@property`, but `Hamiltonian.matrix` overrides it as a **method** `def matrix(self, ...)` to accept arguments.
**Impact:** Code expecting `op.matrix` to be an array (like `Operator`) will fail with `AttributeError: 'function' object...` when interacting with a `Hamiltonian`.
**Fix:** `test_xxz_symmetries.py` was patched to call `.matrix()`. Library code should eventually standardize this (e.g., make `Hamiltonian.matrix` a property that returns a proxy or forbid arguments).

## Fixes Implemented

1.  **`ReflectionSymmetry`**: Added missing `self.perm` attribute initialization.
2.  **`SymmetryContainer`**: Fixed `build_group` to handle empty generators (Identity group) without crashing.
3.  **`QuadraticHamiltonian`**: Fixed `matrix()` argument validation to allow parameter-less calls.
4.  **`TransverseFieldIsing`**: Fixed integer casting for lattice neighbor lookups.
5.  **`XXZ`**: Added missing `self.setup_instruction_codes()` in `__init__` to prevent empty instruction function errors.
6.  **`Operator`**: Added safety check in `matrix` property to raise informative `RuntimeError` if JIT compilation failed, rather than propagating `None`.
7.  **`operators_spin.py`**: Refactored `sigma_operator_composition_single_op` to use consistent variable types and explicit casting, aiming to satisfy Numba typing (though SegFault persists).

## Future Recommendations

*   **Isolate Numba Crash:** Create a minimal reproduction script for the `sigma_composition_integer` crash involving `complex128` types and Numba `njit`.
*   **API Standardization:** Refactor `Hamiltonian.matrix` to avoid shadowing `Operator.matrix` property with incompatible signature.
*   **Legacy Cleanup:** Remove deprecated tests in `Python/test` if they are no longer relevant.
