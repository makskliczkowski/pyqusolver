# New Tests & Modifications

This document records new tests added and significant modifications to existing tests.

## Modified Tests

### `Python/test/test_xxz_symmetries.py`
- **Fix:** Updated `xxz.matrix.toarray()` calls to `xxz.matrix().toarray()` to handle `Hamiltonian.matrix` being a method (overriding `Operator.matrix` property).
- **Purpose:** Enables correct execution of the test suite (currently blocked by Numba SegFault).

### `Python/test/hilbert/test_hilbert_symmetries.py`
- **Fix:** Refactored Numba-compiled helper functions (`create_transverse_ising_operator`, `create_sigma_z_operator`) to avoid capturing external variables (`ns`) in closures, which is problematic for Numba caching.
- **Fix:** Updated `scipy.sparse` checks to use `sp.issparse` instead of deprecated `sp.sparse.issparse`.

## New Tests

*No new independent test files were added in this session. Focus was on repairing critical infrastructure failures in existing tests.*
