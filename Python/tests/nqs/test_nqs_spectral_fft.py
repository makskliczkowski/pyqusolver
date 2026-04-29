"""Regression tests for NQS spectral Fourier postprocessing."""

import numpy as np

from QES.NQS.src.nqs_spectral import (
    _NQSParamView,
    _diagonal_probe_overlap_operator,
    _enumerate_basis_states,
    _exact_expectation_value,
    _exact_wavefunction_vector,
    _materialize_trajectory_params,
)
from QES.NQS.src.spectral.fft import (
    integration_weights,
    spectrum_from_correlator,
)


def _dense_positive_transform(times, weighted_corr):
    dt = float(times[1] - times[0])
    freqs = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(times.size, d=dt))
    phases = np.exp(1.0j * np.outer(freqs, times - times[0]))
    return freqs, phases @ weighted_corr


def _dense_hermitian_transform(times, weighted_corr):
    dt = float(times[1] - times[0])
    full_times = np.concatenate([-(times[:0:-1] - times[0]), times - times[0]])
    full_corr = np.concatenate([np.conj(weighted_corr[:0:-1]), weighted_corr])
    freqs = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(full_corr.size, d=dt))
    phases = np.exp(1.0j * np.outer(freqs, full_times))
    return freqs, phases @ full_corr


def test_spectrum_positive_time_trapezoid_matches_dense_transform():
    times = np.linspace(0.25, 0.75, 6)
    corr = np.array([1.0 + 0.1j, 0.8 - 0.2j, 0.4 + 0.3j, 0.2 - 0.1j, 0.1 + 0.2j, -0.1j])
    weights = integration_weights(times, "trapezoid")

    freqs, expected = _dense_positive_transform(times, corr * weights)
    _, actual_freqs, actual_spectrum, actual = spectrum_from_correlator(
        times,
        corr,
        window=None,
        positive_frequencies_only=False,
        hermitian_extension=False,
        integration_rule="trapezoid",
    )

    np.testing.assert_allclose(actual_freqs, freqs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_spectrum, np.real(expected), rtol=1e-12, atol=1e-12)


def test_spectrum_hermitian_trapezoid_matches_dense_transform():
    times = np.linspace(0.0, 0.5, 7)
    corr = np.array(
        [1.2 + 0.0j, 0.9 - 0.3j, 0.5 - 0.4j, 0.2 - 0.2j, -0.1 + 0.1j, -0.2j, 0.05j]
    )
    weights = integration_weights(times, "trapezoid")

    freqs, expected = _dense_hermitian_transform(times, corr * weights)
    _, actual_freqs, actual_spectrum, actual = spectrum_from_correlator(
        times,
        corr,
        window=None,
        positive_frequencies_only=False,
        hermitian_extension=True,
        integration_rule="trapezoid",
    )

    np.testing.assert_allclose(actual_freqs, freqs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_spectrum, np.real(expected), rtol=1e-12, atol=1e-12)


def test_nqs_spectral_facade_keeps_legacy_helper_imports():
    assert _NQSParamView is not None
    assert _diagonal_probe_overlap_operator is not None
    assert _enumerate_basis_states is not None
    assert _exact_expectation_value is not None
    assert _exact_wavefunction_vector is not None
    assert _materialize_trajectory_params is not None
