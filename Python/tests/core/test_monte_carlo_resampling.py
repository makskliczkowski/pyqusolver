from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_jackknife_mean_matches_known_delete_one_result():
    diagnostics = _load_module("diagnostics_jackknife", "QES/Solver/MonteCarlo/diagnostics.py")
    samples = np.array([1.0, 2.0, 4.0, 8.0])

    result = diagnostics.jackknife_estimate(samples, estimator=np.mean, return_replicates=True)

    expected_replicates = np.array([14.0 / 3.0, 13.0 / 3.0, 11.0 / 3.0, 7.0 / 3.0])
    expected_stderr = np.sqrt((samples.size - 1) * np.mean((expected_replicates - expected_replicates.mean()) ** 2))

    np.testing.assert_allclose(result["replicates"], expected_replicates)
    assert result["estimate"] == np.mean(samples)
    assert result["bias"] == 0.0
    assert result["stderr"] == expected_stderr


def test_bootstrap_supports_vector_valued_estimators():
    diagnostics = _load_module("diagnostics_bootstrap", "QES/Solver/MonteCarlo/diagnostics.py")
    samples = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    def estimator(x):
        return np.array([np.mean(x[:, 0]), np.mean(x[:, 1])])

    result = diagnostics.bootstrap_estimate(
        samples,
        estimator=estimator,
        n_resamples=200,
        rng=123,
        return_replicates=True,
    )

    assert result["estimate"].shape == (2,)
    assert result["stderr"].shape == (2,)
    assert result["bias"].shape == (2,)
    assert result["replicates"].shape == (200, 2)
    np.testing.assert_allclose(result["estimate"], np.array([2.5, 25.0]))
    ci_low, ci_high = result["confidence_interval"]
    assert ci_low.shape == (2,)
    assert ci_high.shape == (2,)
    assert np.all(ci_low <= result["estimate"])
    assert np.all(ci_high >= result["estimate"])


def test_bootstrap_and_jackknife_respect_nonleading_sample_axis():
    diagnostics = _load_module("diagnostics_axis", "QES/Solver/MonteCarlo/diagnostics.py")
    samples = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )

    def estimator(x):
        return np.mean(x, axis=0)

    jackknife = diagnostics.jackknife_estimate(samples, estimator=estimator, sample_axis=1)
    bootstrap = diagnostics.bootstrap_estimate(samples, estimator=estimator, sample_axis=1, n_resamples=50, rng=7)

    np.testing.assert_allclose(jackknife["estimate"], np.array([2.5, 25.0]))
    np.testing.assert_allclose(bootstrap["estimate"], np.array([2.5, 25.0]))
