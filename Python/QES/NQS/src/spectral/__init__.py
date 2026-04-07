"""
Internal spectral helpers for QES.NQS.

This subpackage groups the implementation details behind the public spectral
methods exposed through ``QES.NQS.nqs`` and ``QES.NQS.src.nqs_spectral``.
"""

from .fft       import spectrum_from_correlator_impl
from .results   import NQSCorrelatorResult, NQSSpectralMapResult, NQSSpectralResult, NQSTDVPRecord

__all__ = [
    "NQSCorrelatorResult",
    "NQSSpectralMapResult",
    "NQSSpectralResult",
    "NQSTDVPRecord",
    "spectrum_from_correlator_impl",
]
