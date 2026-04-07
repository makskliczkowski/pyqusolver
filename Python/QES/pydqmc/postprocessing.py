"""
Lightweight postprocessing helpers for `QES.pydqmc`.

The goal of this module is not to replace a full analysis stack. It provides a
small stable layer for the common tasks users need immediately after a run:

- discarding warmup samples,
- rebinnig time series,
- estimating summary statistics,
- and deriving new scalar observables from measured ones.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from QES.Solver.MonteCarlo.diagnostics import compute_autocorr_time


def trim_warmup(samples: Sequence[float], warmup: int = 0) -> np.ndarray:
    """Drop the first `warmup` samples from a one-dimensional time series."""
    values = np.asarray(samples, dtype=float).reshape(-1)
    start = max(0, int(warmup))
    return values[start:]


def rebin_series(samples: Sequence[float], bin_size: int = 1) -> np.ndarray:
    """
    Rebin a one-dimensional time series into contiguous block averages.

    Any tail shorter than the requested bin size is dropped so every retained
    bin carries the same weight.
    """
    values = np.asarray(samples, dtype=float).reshape(-1)
    size = max(1, int(bin_size))
    if size == 1 or values.size == 0:
        return values.copy()
    n_blocks = values.size // size
    if n_blocks == 0:
        return np.asarray([], dtype=float)
    trimmed = values[: n_blocks * size]
    return trimmed.reshape(n_blocks, size).mean(axis=1)


def summarize_series(samples: Sequence[float], warmup: int = 0, bin_size: int = 1) -> Dict[str, Any]:
    """Return compact summary statistics for a scalar Monte Carlo time series."""
    trimmed = trim_warmup(samples, warmup=warmup)
    rebinned = rebin_series(trimmed, bin_size=bin_size)
    if rebinned.size == 0:
        return {
            "n_raw": int(np.asarray(samples).size),
            "n_used": 0,
            "warmup": max(0, int(warmup)),
            "bin_size": max(1, int(bin_size)),
            "mean": None,
            "std": None,
            "stderr": None,
            "autocorr_time": None,
        }

    mean = float(np.mean(rebinned))
    std = float(np.std(rebinned))
    stderr = float(std / np.sqrt(max(1, rebinned.size)))
    autocorr_time = None
    if trimmed.size >= 2:
        try:
            autocorr_time = float(compute_autocorr_time(trimmed))
        except Exception:
            autocorr_time = None
    return {
        "n_raw": int(np.asarray(samples).size),
        "n_used": int(rebinned.size),
        "warmup": max(0, int(warmup)),
        "bin_size": max(1, int(bin_size)),
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "autocorr_time": autocorr_time,
    }


def derive_observables(
    observables: Mapping[str, Any],
    formulas: Mapping[str, Callable[[Mapping[str, Any]], Any]],
) -> Dict[str, Any]:
    """
    Evaluate user-defined derived observables from an observable dictionary.

    Each formula receives the full observable mapping and returns the derived
    value for its key.
    """
    base = dict(observables)
    derived: Dict[str, Any] = {}
    for name, formula in formulas.items():
        derived[name] = formula(base)
    return derived


def summarize_result(result: Any, warmup: int = 0, bin_size: int = 1) -> Dict[str, Any]:
    """Return a compact workflow-oriented summary for a `DQMCResult`-like object."""
    energy_history = getattr(result, "energy_history", None)
    observables = getattr(result, "observables", {})
    diagnostics = getattr(result, "diagnostics", {})
    setup = getattr(result, "setup", {})
    return {
        "setup": dict(setup),
        "diagnostics": dict(diagnostics),
        "observables": dict(observables),
        "energy": summarize_series(energy_history or [], warmup=warmup, bin_size=bin_size),
    }
