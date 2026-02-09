# New / Updated Tests in This Branch

## New Test File: `Python/test/test_numeric_helpers_properties.py`

### `test_autocorr_constant_series_is_well_defined`

- **Purpose**: verifies deterministic behavior for constant Monte Carlo traces.
- **Bug it would catch**: pathological ESS collapse (`tau = N`) for constant series.
- **Module guarded**: `QES/Solver/MonteCarlo/diagnostics.py`.

### `test_compute_rhat_edge_cases` (parametrized)

- **Purpose**: validates R-hat output for finite-chain edge cases (identical chains, insufficient samples, single-chain input).
- **Bug it would catch**: invalid finite-sample handling (wrong NaN policy / unstable values).
- **Module guarded**: `QES/Solver/MonteCarlo/diagnostics.py`.

### `test_count_left_int_matches_big_endian_convention` (parametrized)

- **Purpose**: checks bit-ordering convention in integer occupation helpers.
- **Bug it would catch**: left/right index inversion causing wrong fermionic signs.
- **Module guarded**: `QES/Algebra/Operator/sign.py`.

### `test_anyon_phase_and_parity_roundtrip`

- **Purpose**: verifies phase magnitude and parity codomain invariants.
- **Bug it would catch**: invalid complex phase generation or non-±1 parity output.
- **Module guarded**: `QES/Algebra/Operator/sign.py`.

---

## New Test File: `Python/test/test_registry_module_discovery.py`

### `test_list_modules_returns_stable_sorted_records`

- **Purpose**: validates stable ordering and schema of registry metadata.
- **Bug it would catch**: regression in public module-discovery API shape or sorting.
- **Module guarded**: `QES/registry.py`.

### `test_describe_module_curated_and_unknown_paths`

- **Purpose**: checks known-module description and unknown-module fallback.
- **Bug it would catch**: broken fallback behavior returning exceptions/empty strings.
- **Module guarded**: `QES/registry.py`.

---

## Library Fixes Required to Make Meaningful Tests Pass

### `Python/QES/Solver/MonteCarlo/diagnostics.py`

- Added deterministic handling for constant autocorrelation series in `compute_autocorr_time` (`tau = 1.0` for all-ones ACF) to preserve scientifically meaningful ESS behavior.
- Corrected `compute_rhat` to compute and use between-chain variance `B` explicitly in the pooled variance estimate.

These are minimal, test-driven fixes focused only on numerical-diagnostics correctness.
