"""
Internal spectral helpers for QES.NQS.

This subpackage groups the implementation details behind the public spectral
methods exposed through ``QES.NQS.nqs`` and ``QES.NQS.src.nqs_spectral``.
"""

from .fft       import spectrum_from_correlator_impl
from .mc        import (
    dynamical_correlator_impl,
    dynamic_structure_factor_impl,
    spectral_function_impl,
    spectral_map_impl,
    transition_correlator_between_impl,
    transition_correlator_between_trajectories_impl,
    transition_correlator_impl,
)
from .results   import NQSCorrelatorResult, NQSSpectralMapResult, NQSSpectralResult, NQSTDVPRecord
from .tdvp      import time_evolve_impl

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
