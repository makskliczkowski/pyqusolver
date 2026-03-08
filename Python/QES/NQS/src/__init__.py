"""
Some backend for NQS neural network architectures integration.
"""

from .nqs_precision import NQSPrecisionPolicy, cast_for_precision, resolve_precision_policy
from .nqs_spectral import NQSCorrelatorResult, NQSSpectralMapResult, NQSSpectralResult, NQSTDVPRecord

__all__ = [
    "NQSCorrelatorResult",
    "NQSSpectralMapResult",
    "NQSPrecisionPolicy",
    "NQSTDVPRecord",
    "NQSSpectralResult",
    "resolve_precision_policy",
    "cast_for_precision",
]

# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------
