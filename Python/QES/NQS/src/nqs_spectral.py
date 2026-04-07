"""
Public spectral facade for QES.NQS.

The implementation lives under ``QES.NQS.src.spectral``:

- ``results`` for return containers,
- ``tdvp`` for trajectory evolution helpers,
- ``mc`` for probe-state and Monte Carlo orchestration,
- ``exact`` for tiny-system deterministic checks,
- ``fft`` for time-grid and Fourier postprocessing.

This module intentionally stays thin so the public import path remains stable
while the spectral internals stay grouped by responsibility.
"""

from .spectral import (
    NQSCorrelatorResult,
    NQSSpectralMapResult,
    NQSSpectralResult,
    NQSTDVPRecord,
    dynamical_correlator_impl,
    dynamic_structure_factor_impl,
    spectral_function_impl,
    spectral_map_impl,
    spectrum_from_correlator_impl,
    time_evolve_impl,
    transition_correlator_between_impl,
    transition_correlator_between_trajectories_impl,
    transition_correlator_impl,
)

__all__ = [
    "NQSCorrelatorResult",
    "NQSSpectralMapResult",
    "NQSSpectralResult",
    "NQSTDVPRecord",
    "dynamical_correlator_impl",
    "dynamic_structure_factor_impl",
    "spectral_function_impl",
    "spectral_map_impl",
    "spectrum_from_correlator_impl",
    "time_evolve_impl",
    "transition_correlator_between_impl",
    "transition_correlator_between_trajectories_impl",
    "transition_correlator_impl",
]
