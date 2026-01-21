"""
NQS-specific network implementations.
Overrides or extends general_python networks.
"""
from .net_approx_symmetric import AnsatzApproxSymmetric
from .net_stacked import AnsatzStacked

__all__ = [
    "AnsatzApproxSymmetric",
    "AnsatzStacked",
]
