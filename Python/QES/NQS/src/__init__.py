"""
Some backend for NQS neural network architectures integration.
"""

from .nqs_precision import NQSPrecisionPolicy, cast_for_precision, resolve_precision_policy

__all__ = [
    "NQSPrecisionPolicy",
    "resolve_precision_policy",
    "cast_for_precision",
]
