# New Tests Documentation

This document describes the new tests added to the repository to improve coverage and reliability.

## 1. `tests/test_nqs_invariants.py`

### Purpose
To verify fundamental invariants of the NQS solver and Hamiltonian models.

### Tests Implemented
*   **`test_nqs_properties`**: Checks that the NQS object is initialized with correct properties (backend, number of visible units).
*   **`test_nqs_output_shape`**: Verifies that the NQS output (log-amplitudes) has the correct shape `(batch_size,)` and appropriate dtype (complex or float) when processing a batch of states. This catches broadcasting errors.
*   **`test_probability_positive`**: Verifies that sampled probabilities are non-negative and finite, ensuring the physical validity of the sampling process.

### Bugs Caught
*   This test would catch issues where the network output shape is inconsistent with the input batch size (e.g., broadcasting errors in JAX `vmap`).
*   It ensures that `NQS` can handle states with explicit types (e.g., `float64`).

## 2. `tests/test_determinism.py`

### Purpose
To ensure that the NQS solver is deterministic when a random seed is provided. This is crucial for reproducibility of scientific results.

### Tests Implemented
*   **`test_sampling_determinism`**: Runs the sampler twice with the same seed and asserts that the generated samples are identical.
*   **`test_sampling_randomness`**: Runs the sampler with different seeds and asserts that the samples are different.
*   **`test_optimization_step_determinism`**: Verifies that a full optimization step (sampling + gradient computation + parameter update) is deterministic given a seed. This ensures that the entire training pipeline is reproducible.

### Bugs Caught
*   Catches regressions where random number generators are not correctly seeded or propagated (e.g., in JAX `PRNGKey` handling).
*   Catches race conditions or non-deterministic behavior in parallel execution (though mainly relevant for CPU/GPU parallelism).
*   **Caught during development**: Revealed a bug in `NQS.step` where `batch_size` was passed twice to `_single_step` (once explicitly and once via `**kwargs`), causing a `TypeError`.

## Summary
These tests add a layer of defense against regressions in core NQS functionality (sampling, evaluation, optimization) and ensure that the software behaves predictably.
