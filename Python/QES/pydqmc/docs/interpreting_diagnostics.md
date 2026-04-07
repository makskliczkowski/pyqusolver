# Interpreting `pydqmc` Diagnostics

The high-level `run_dqmc(...)` result exposes two different output layers:

- `result.observables`
  - physics estimators such as energy and density
- `result.diagnostics`
  - numerical and workflow diagnostics for the Monte Carlo engine

## Core Diagnostics

### `acceptance_rate`

Fraction of proposed local HS updates accepted in the most recent sweep.

Use:

- very low values can indicate badly tuned continuous-field proposals
- values near one for continuous fields can indicate proposals are too small to decorrelate efficiently

### `green_residual_mean` and `green_residual_max`

These measure how well the stored equal-time Green's functions satisfy the defining inverse relation

```text
(I + B_M ... B_1) G = I.
```

Use:

- smaller is better
- persistent growth means fast local updates are drifting and full refreshes should happen more often

### `forced_refreshes`

How many times the sampler decided the residual drift was large enough to rebuild Green's functions from scratch.

Use:

- frequent forced refreshes mean the cheap update path is not staying sufficiently stable with the current settings

### `refresh_drift`

Difference between the residual before and after the last forced refresh.

Use:

- large positive values mean the refresh path is correcting substantial accumulated drift

### `refresh_strategy`

Full-refresh backend.

Current options:

- `jax_udt`
  - default production path
- `numpy_pivoted`
  - slower reference/debug path

### `residual_check_interval`

Number of sweeps between full residual scans.

Use:

- lower values are safer but more expensive
- higher values reduce overhead but may allow more drift before detection

## Sign Metadata

`pydqmc` now reports an explicit sign contract through `result.setup["sign"]` and `result.diagnostics`.

Current baseline:

- `sign_tracking = "measured_on_abs_weight_ensemble"`
- `reweighting = True`
- `weight_sampling = "absolute_determinant_ratio"`
- `supports_complex_phase = False`
- `sign_policy = "strict"` by default on the public solver path

Additional classification fields:

- `sign_envelope`
  - `known_sign_free` for the currently recognized safe envelopes
  - `unsupported` otherwise
- `expected_average_sign`
  - `1.0` when the code recognizes a symmetry-protected sign-free regime
- `sign_reason`
  - human-readable explanation of why the regime is treated as sign-free or unsupported

Interpretation:

- this baseline is intended for regimes where the sign structure is already benign
- complex-phase one-body matrices are rejected explicitly
- built-in equal-time observables are reweighted with the measured sign on the `|W|` ensemble
- unequal-time data collected by the built-in path is also reweighted
- per-chain custom hook outputs are reweighted automatically when hooks return one scalar per chain
- unsupported regimes are still not part of the validated production envelope
- unsupported regimes now require `sign_policy="allow_unsupported"` explicitly

## Average Sign

Two fields now matter directly:

- `average_sign`
  - running average sign over all recorded equal-time measurements
- `last_average_sign`
  - average sign of the most recent recorded measurement

If `average_sign` becomes small, the reweighted estimator remains formally correct but the statistical noise grows rapidly.

## Sampling Metadata

`result.setup["sampling"]` describes the current local-update path and future-facing capabilities.

Important fields:

- `proposal_kind`
  - `deterministic_flip` for discrete Hirsch fields
  - `local_random_walk` for current continuous fields
- `provides_measure_gradient`
  - whether the local HS measure exposes an analytic gradient, which is useful groundwork for future Langevin/HMC-style proposals
- `supports_delayed_updates`
  - currently `False`
- `update_mode`
  - currently `immediate_local`

## Analysis vs Restart

Two persistence paths now exist:

- `result.save("run.json")`
  - analysis-ready result bundle
- `solver.save_checkpoint("restart.npz")`
  - restart-oriented checkpoint containing auxiliary fields and runtime metadata

Use the JSON result bundle for postprocessing and sharing.
Use the checkpoint only to continue a compatible solver setup.
