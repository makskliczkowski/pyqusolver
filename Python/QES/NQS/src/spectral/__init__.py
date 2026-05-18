"""
Internal implementation package for NQS spectral workflows.

This package owns only the variational path:

1. ``tdvp`` produces a time-indexed NQS parameter trajectory.
2. ``mc`` creates operator-probe states and evaluates time-domain transition
   correlators by Monte Carlo, with exact basis summation only for tiny NQS
   regression checks.
3. ``fft`` turns uniformly sampled correlators into finite-time spectral
   estimates.
4. ``results`` defines the dataclasses shared by the public API.

Exact-diagonalization and Lehmann spectral functions are not reimplemented or
re-exported here. Use ``QES.general_python.physics.spectral.spectral_backend``
or ``hamil.spectral`` for the ED/Lanczos source of truth.

Use ``QES.NQS.src.nqs_spectral`` or the methods on ``NQS`` as the public import
surface. Import directly from this package only for tests or internal tools.
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
