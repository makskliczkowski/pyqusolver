"""
Compatibility surface for NQS network construction and adaptation.

The original implementation mixed representation resolution, sampler adapters,
 preset estimation, and factory/help metadata in one module. These concerns now
 live in smaller focused modules, while this file re-exports the maintained
 public names for compatibility with the rest of ``QES.NQS``.
"""

from .adapters import (
    NQSAutoregressiveAdapter,
    NQSDenseEvalAdapter,
    NQSNetAdapterBase,
    NQSRBMAdapter,
    choose_nqs_network_adapter,
    infer_network_family,
)
from .factory import NetworkFactory, NetworkInfo
from .presets import SOTANetConfig, estimate_network_params
from .representation import (
    ModelRepresentationInfo,
    apply_nqs_representation_overrides,
    canonical_network_request_family,
    resolve_model_representation,
    resolve_nqs_state_defaults,
    resolve_spin_mode_repr,
)

__all__ = [
    "ModelRepresentationInfo",
    "NQSAutoregressiveAdapter",
    "NQSDenseEvalAdapter",
    "NQSNetAdapterBase",
    "NQSRBMAdapter",
    "NetworkFactory",
    "NetworkInfo",
    "SOTANetConfig",
    "apply_nqs_representation_overrides",
    "canonical_network_request_family",
    "choose_nqs_network_adapter",
    "estimate_network_params",
    "infer_network_family",
    "resolve_model_representation",
    "resolve_nqs_state_defaults",
    "resolve_spin_mode_repr",
]
