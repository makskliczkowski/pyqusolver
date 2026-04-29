"""
Time-grid and Fourier postprocessing helpers for NQS spectral workflows.
"""

from typing import Optional, Sequence

import numpy as np

from .results import NQSSpectralResult


def as_time_array(times: Sequence[float]) -> np.ndarray:
    arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("times must contain at least one point.")
    if np.any(np.diff(arr) < -1e-12):
        raise ValueError("times must be monotonically non-decreasing.")
    return arr


def window_values(name: Optional[str], n: int) -> np.ndarray:
    if name is None or str(name).lower() in ("none", "rect", "rectangular"):
        return np.ones(n, dtype=np.float64)
    key = str(name).lower()
    if key in ("hann", "hanning"):
        return np.hanning(n)
    if key == "hamming":
        return np.hamming(n)
    if key == "blackman":
        return np.blackman(n)
    raise ValueError(f"Unsupported window '{name}'.")


def integration_weights(times_arr: np.ndarray, rule: str) -> np.ndarray:
    if times_arr.size < 2:
        raise ValueError("At least two time points are required to compute a spectrum.")
    key = str(rule).lower()
    if key == "rectangle":
        dt = np.diff(times_arr)
        if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-12):
            raise ValueError("Rectangle-rule integration requires an evenly spaced time grid.")
        return np.full(times_arr.size, float(dt[0]), dtype=np.float64)
    if key != "trapezoid":
        raise ValueError("integration_rule must be 'rectangle' or 'trapezoid'.")
    weights = np.zeros(times_arr.size, dtype=np.float64)
    delta = np.diff(times_arr)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    if times_arr.size > 2:
        weights[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return weights


def damping_profile(times_arr: np.ndarray, *, eta: float, kind: str) -> np.ndarray:
    t_rel = times_arr - times_arr[0]
    broadening = abs(float(eta))
    key = str(kind).lower()
    if broadening == 0.0 or key in ("none", "off"):
        return np.ones_like(t_rel, dtype=np.float64)
    if key in ("exp", "exponential", "lorentzian"):
        return np.exp(-broadening * t_rel)
    if key in ("gaussian", "gauss"):
        return np.exp(-(broadening * t_rel) ** 2)
    raise ValueError("broadening_kind must be 'exponential', 'gaussian', or 'none'.")


def uniform_positive_time_transform(
    weighted_values: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``sum_t weighted_values[t] exp(i omega t)`` on the FFT grid.

    The input samples are assumed to live on ``t = 0, dt, ...``.  This keeps
    the transform O(N log N) and avoids the dense frequency-by-time matrix.
    """
    weighted_values = np.asarray(weighted_values, dtype=np.complex128).reshape(-1)
    n_times = weighted_values.size
    freqs_unshifted = 2.0 * np.pi * np.fft.fftfreq(n_times, d=dt)
    raw_unshifted = np.fft.ifft(weighted_values) * n_times
    return np.fft.fftshift(freqs_unshifted), np.fft.fftshift(raw_unshifted)


def uniform_hermitian_time_transform(
    weighted_positive_values: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the Hermitian-completed positive-time transform on the FFT grid.

    ``weighted_positive_values`` stores samples for ``t >= 0``.  The negative
    branch is filled as ``C(-t)=conj(C(t))``, which is the autocorrelation
    symmetry used by the dynamical-structure-factor path.
    """
    weighted_positive_values = np.asarray(
        weighted_positive_values, dtype=np.complex128
    ).reshape(-1)
    n_times = weighted_positive_values.size
    if n_times <= 1:
        return uniform_positive_time_transform(weighted_positive_values, dt=dt)

    full_values = np.concatenate(
        [
            np.conj(weighted_positive_values[:0:-1]),
            weighted_positive_values,
        ]
    )
    n_full = full_values.size
    freqs_unshifted = 2.0 * np.pi * np.fft.fftfreq(n_full, d=dt)
    raw_unshifted = np.fft.ifft(full_values) * n_full
    raw_unshifted *= np.exp(-1.0j * freqs_unshifted * float(n_times - 1) * dt)
    return np.fft.fftshift(freqs_unshifted), np.fft.fftshift(raw_unshifted)


def spectrum_from_correlator(
    times: Sequence[float],
    correlator: np.ndarray,
    *,
    eta: float = 0.0,
    window: Optional[str] = "hann",
    subtract_initial: bool = False,
    positive_frequencies_only: bool = True,
    hermitian_extension: bool = False,
    broadening_kind: str = "exponential",
    integration_rule: str = "rectangle",
):
    """
    Convert a real-time correlator into a discrete spectral estimate.
    """
    times_arr = as_time_array(times)
    if times_arr.size < 2:
        raise ValueError("At least two time points are required to compute a spectrum.")

    dts = np.diff(times_arr)
    dt = float(dts[0])
    if not np.allclose(dts, dt, rtol=1e-6, atol=1e-12):
        raise ValueError("The spectral FFT currently requires an evenly spaced time grid.")

    corr_work = np.asarray(correlator, dtype=np.complex128)
    corr_work = corr_work * window_values(window, corr_work.size) * damping_profile(
        times_arr, eta=eta, kind=broadening_kind
    )

    if subtract_initial:
        corr_work = corr_work - corr_work[0]

    weights = integration_weights(times_arr, integration_rule)
    weighted_corr = corr_work * weights
    if hermitian_extension and corr_work.size > 1:
        freqs, raw_fft = uniform_hermitian_time_transform(weighted_corr, dt=dt)
    else:
        freqs, raw_fft = uniform_positive_time_transform(weighted_corr, dt=dt)

    spectrum = np.real(raw_fft)
    if positive_frequencies_only:
        mask = freqs >= -1e-12
        freqs = freqs[mask]
        raw_fft = raw_fft[mask]
        spectrum = spectrum[mask]
    return times_arr, freqs, spectrum, raw_fft


def spectrum_from_correlator_impl(
    times: Sequence[float],
    correlator: np.ndarray,
    *,
    eta: float = 0.0,
    window: Optional[str] = None,
    subtract_initial: bool = False,
    positive_frequencies_only: bool = True,
    hermitian_extension: bool = False,
    broadening_kind: str = "exponential",
    integration_rule: str = "rectangle",
) -> NQSSpectralResult:
    """
    Build a finite-time spectral estimate directly from a supplied correlator.
    """
    times_arr, freqs, spectrum, raw_fft = spectrum_from_correlator(
        times,
        correlator,
        eta=eta,
        window=window,
        subtract_initial=subtract_initial,
        positive_frequencies_only=positive_frequencies_only,
        hermitian_extension=hermitian_extension,
        broadening_kind=broadening_kind,
        integration_rule=integration_rule,
    )
    return NQSSpectralResult(
        times=times_arr,
        correlator=np.asarray(correlator, dtype=np.complex128),
        frequencies=freqs,
        spectrum=np.asarray(spectrum, dtype=np.float64),
        spectrum_complex=np.asarray(raw_fft, dtype=np.complex128),
        metadata={
            "eta": float(eta),
            "window": window,
            "positive_frequencies_only": bool(positive_frequencies_only),
            "subtract_initial": bool(subtract_initial),
            "hermitian_extension": bool(hermitian_extension),
            "broadening_kind": str(broadening_kind),
            "integration_rule": str(integration_rule),
        },
    )


__all__ = [
    "as_time_array",
    "damping_profile",
    "integration_weights",
    "spectrum_from_correlator",
    "spectrum_from_correlator_impl",
    "uniform_hermitian_time_transform",
    "uniform_positive_time_transform",
    "window_values",
]
