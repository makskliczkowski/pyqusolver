"""
NQS-specific network integration package.

This package groups the NQS-side network logic that sits on top of the generic
``QES.general_python.ml`` layer:

- representation resolution
- sampler-facing adapters
- preset estimation
- user-facing factory helpers
"""

from .adapters          import NQSAutoregressiveAdapter, NQSDenseEvalAdapter, NQSNetAdapterBase, NQSRBMAdapter, choose_nqs_network_adapter, infer_network_family
from .factory           import NetworkFactory, NetworkInfo
from .presets           import SOTANetConfig, estimate_network_params
from .representation    import (
                            ModelRepresentationInfo,
                            apply_nqs_representation_overrides,
                            bind_local_energy_state_convention,
                            canonical_network_request_family,
                            convert_state_array_representation,
                            resolve_model_representation,
                            resolve_nqs_state_defaults,
                            resolve_representation_value,
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
    "bind_local_energy_state_convention",
    "canonical_network_request_family",
    "choose_nqs_network_adapter",
    "convert_state_array_representation",
    "estimate_network_params",
    "infer_network_family",
    "resolve_model_representation",
    "resolve_nqs_state_defaults",
    "resolve_representation_value",
    "resolve_spin_mode_repr",
]

# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------