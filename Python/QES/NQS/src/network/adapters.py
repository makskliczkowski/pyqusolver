"""
Small NQS-side adapter layer between network families and samplers.
"""

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp

from .representation import (
    ModelRepresentationInfo,
    resolve_representation_value,
)

def _get_nqs_metadata(net: Any) -> Dict[str, Any]:
    if hasattr(net, "get_nqs_metadata") and callable(net.get_nqs_metadata):
        try:
            meta = net.get_nqs_metadata()
            if isinstance(meta, dict):
                return meta
        except Exception:
            pass
    return {}

def infer_network_family(net: Any) -> str:
    """
    Coarse network family classifier used once during NQS setup.
    """
    meta    = _get_nqs_metadata(net)
    family  = str(meta.get("family", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if family:
        return family
    if meta.get("supports_exact_sampling", False):
        return "autoregressive"
    if meta.get("supports_fast_updates", False):
        return "rbm"
    if hasattr(net, "get_logits") and hasattr(net, "get_phase"):
        return "autoregressive"
    if hasattr(net, "log_psi_delta") or hasattr(net, "get_log_psi_delta"):
        return "rbm"

    name        = str(getattr(net, "name", type(net).__name__)).lower()
    cls_name    = type(net).__name__.lower()
    module_name = str(getattr(type(net), "__module__", "")).lower()

    if any(tok in value for value in (name, cls_name, module_name) for tok in ("autoregressive", "complexar", "made")):
        return "autoregressive"
    if "rbm" in name or "rbm" in cls_name or "net_rbm" in module_name:
        return "rbm"
    if "gcnn" in name or "gcnn" in cls_name or "net_gcnn" in module_name:
        return "gcnn"
    if "cnn" in name or "cnn" in cls_name or "net_cnn" in module_name:
        return "cnn"
    if "transformer" in name or "transformer" in cls_name or "net_transformer" in module_name:
        return "transformer"
    return "dense"

def _resolve_apply_and_params(net: Any) -> Tuple[Callable, Any]:
    """
    Resolve the callable apply path and optional parameter object for a network.
    """
    try:
        import flax.linen as nn
    except ImportError:
        nn = None

    if nn is not None and isinstance(net, nn.Module):
        return net.apply, None
    if hasattr(net, "get_apply") and callable(net.get_apply):
        return net.get_apply()
    if hasattr(net, "apply") and callable(net.apply):
        params = net.get_params() if hasattr(net, "get_params") and callable(net.get_params) else None
        return net.apply, params
    if callable(net):
        return net, None
    raise ValueError("Invalid network object provided. Needs to be callable or have an 'apply' method.")

def _flip_selected_values(values: Any, *, state_spin: bool, state_value: float):
    arr = jnp.asarray(values)
    if state_spin:
        return -arr
    return jnp.asarray(float(state_value), dtype=arr.dtype) - arr

# ------------------------------------------------------------------------------
# Sampler-facing network adapter classes
# ------------------------------------------------------------------------------

class NQSNetAdapterBase:
    """
    Minimal sampler-facing network adapter.
    """

    family = "dense"
    sampler_kind = "MCSampler"

    def __init__(self, net: Any, representation: ModelRepresentationInfo, backend: str = "jax"):
        self.net                    = net
        self.representation         = representation
        self.backend                = backend
        self.metadata               = _get_nqs_metadata(net)
        self.native_representation  = self.metadata.get("native_representation") or representation.network_representation

    def preferred_sampler_representation(self) -> str:
        return self.representation.sampler_representation

    def _resolve_sampler_value(
        self,
        representation: str | None = None,
        *,
        fallback: float = 1.0,
    ) -> float:
        return float(
            resolve_representation_value(
                representation or self.preferred_sampler_representation(),
                local_space_type=self.representation.local_space_type,
                fallback=fallback,
            )
        )

    def preferred_sampler_mode_repr(self):
        return self._resolve_sampler_value(fallback=1.0)

    def resolve_sampling_hooks(self) -> Dict[str, Any]:
        apply_fun, params = _resolve_apply_and_params(self.net)
        return {
            "family"                        : self.family,
            "sampler_kind"                  : self.sampler_kind,
            "apply_fun"                     : apply_fun,
            "parameters"                    : params,
            "native_representation"         : self.native_representation,
            "log_psi_delta"                 : getattr(self.net, "log_psi_delta", None),
            "log_psi_delta_cache_init"      : getattr(self.net, "init_log_psi_delta_cache", None),
            "log_psi_delta_supports_cache"  : hasattr(self.net, "init_log_psi_delta_cache"),
            "representation"                : self.representation,
            "metadata"                      : self.metadata,
        }

# ------------------------------------------------------------------------------

class NQSRBMAdapter(NQSNetAdapterBase):
    family = "rbm"
    sampler_kind = "MCSampler"

    def preferred_sampler_representation(self) -> str:
        return self.representation.sampler_representation

    def preferred_sampler_mode_repr(self):
        sampler_representation = self.preferred_sampler_representation().replace("_", "-")
        if sampler_representation == "spin-pm":
            # Generic RBMs operating directly on signed spin inputs expect the
            # conventional {-1, +1} encoding, not the physical spin magnitude
            # (for example +/-0.5 for spin-1/2). The latter would silently
            # rescale every visible unit and corrupt both training and exact
            # entropy evaluation when a prebuilt generic RBM is passed into NQS.
            return 1.0
        return super().preferred_sampler_mode_repr()

    def resolve_sampling_hooks(self) -> Dict[str, Any]:
        hooks                   = super().resolve_sampling_hooks()
        sampler_representation  = self.preferred_sampler_representation()
        sampler_key             = sampler_representation.replace("_", "-")
        state_spin              = sampler_key == "spin-pm"
        state_value             = self._resolve_sampler_value(sampler_representation, fallback=1.0)

        if hooks["log_psi_delta"] is None and hasattr(self.net, "get_log_psi_delta"):
            hooks["log_psi_delta"] = self.net.get_log_psi_delta()
            
        if hooks["log_psi_delta_cache_init"] is None and hasattr(self.net, "get_log_psi_delta_cache_init"):
            hooks["log_psi_delta_cache_init"]       = self.net.get_log_psi_delta_cache_init()
            hooks["log_psi_delta_supports_cache"]   = True

        if hooks["log_psi_delta"] is not None:
            hooks["log_psi_delta"] = partial(
                hooks["log_psi_delta"],
                proposal_update=partial(_flip_selected_values, state_spin=state_spin, state_value=state_value))
        return hooks

# ------------------------------------------------------------------------------

class NQSDenseEvalAdapter(NQSNetAdapterBase):
    family          = "dense"
    sampler_kind    = "MCSampler"

class NQSAutoregressiveAdapter(NQSNetAdapterBase):
    family          = "autoregressive"
    sampler_kind    = "ARSampler"

    def resolve_sampling_hooks(self) -> Dict[str, Any]:
        hooks = super().resolve_sampling_hooks()
        hooks["logits_method"] = (
            (lambda module, x: module.get_logits_binary(x))
            if hasattr(self.net, "get_logits_binary")
            else (lambda module, x: module.get_logits(x, is_binary=True))
        )
        hooks["phase_method"] = (
            (lambda module, x: module.get_phase_binary(x))
            if hasattr(self.net, "get_phase_binary")
            else (lambda module, x: module.get_phase(x, is_binary=True))
        )
        return hooks

# ------------------------------------------------------------------------------
#! Network adapter factory
# ------------------------------------------------------------------------------

def choose_nqs_network_adapter(
    net: Any,
    representation: ModelRepresentationInfo,
    backend: str = "jax",
) -> NQSNetAdapterBase:
    """
    Pick the sampler-facing adapter for a network once during setup.
    """
    family = infer_network_family(net)
    if family == "autoregressive":
        return NQSAutoregressiveAdapter(net, representation, backend=backend)
    if family == "rbm":
        return NQSRBMAdapter(net, representation, backend=backend)
    return NQSDenseEvalAdapter(net, representation, backend=backend)


__all__ = [
    "NQSAutoregressiveAdapter",
    "NQSDenseEvalAdapter",
    "NQSNetAdapterBase",
    "NQSRBMAdapter",
    "choose_nqs_network_adapter",
    "infer_network_family",
]

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------
