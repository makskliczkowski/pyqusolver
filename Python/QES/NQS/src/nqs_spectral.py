"""
Public import facade for NQS spectral workflows.

The public ``NQS`` methods import from this module so user-facing paths remain
stable while implementation details live in ``QES.NQS.src.spectral``:

- ``tdvp`` evolves variational parameters on a requested real-time grid.
- ``mc`` builds probe states, evaluates transition correlators, and orchestrates
  dynamical structure factor calculations.
- ``fft`` converts uniformly sampled correlators into broadened spectra.

ED/Lanczos spectral functions live in
``QES.general_python.physics.spectral.spectral_backend`` and are intentionally
not re-exported here. The NQS internals contain only exact basis-summation
diagnostics for tiny variational states.

Keep this file as a facade only. New implementation code should go into the
responsibility-specific modules above, not into this compatibility layer.
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
