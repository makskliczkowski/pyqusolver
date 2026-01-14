"""
Some backend for NQS neural network architectures integration.
"""

from .nqs_precision import NQSPrecisionPolicy, resolve_precision_policy, cast_for_precision

__all__ = [
    "NQSPrecisionPolicy",
    "resolve_precision_policy",
    "cast_for_precision",
]
