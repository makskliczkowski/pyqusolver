"""
Small NQS-side adapter layer between network families and samplers.
"""

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp

try:
    from .representation import ModelRepresentationInfo, resolve_representation_value
except ImportError:
    raise ImportError("Could not import representation utilities. Make sure the 'representation' module is available in the same package.")

# ----------------------------------------------------------------------

def _get_nqs_metadata(net: Any) -> Dict[str, Any]:
    ''' Try to get NQS metadata from the network if it provides it. '''
    
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
    Just a heuristic to pick the right sampler adapter, 
    not meant to be perfect or strictly correct in all cases. 
    The main goal is to detect autoregressive and RBM-like structures, 
    but it also looks for explicit metadata flags and common naming patterns.
    """
    meta    = _get_nqs_metadata(net)
    family  = str(meta.get("family", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if family:
        return family
    if meta.get("supports_exact_sampling", False):                  # Explicit metadata flag for autoregressive support
        return "autoregressive"
    if meta.get("supports_fast_updates", False):                    # Explicit metadata flag for RBM-like fast update support
        return "rbm"
    if hasattr(net, "get_logits") and hasattr(net, "get_phase"):    # Presence of separate logits/phase methods is a strong indicator of autoregressive structure
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

def resolve_apply_and_params(net: Any) -> Tuple[Callable, Any]:
    """
    Resolve the callable apply path and optional parameter object for a network.
    
    Parameters
    ----------
    net : Any
        The network object to resolve. This can be a variety of types, including:
        - A Flax nn.Module instance (in which case we return net.apply and net's parameters)
        - An object with a get_apply() method that returns (apply_fun, params)
        - An object with an apply() method and optionally a get_params() method
        - A plain callable (in which case we return it directly with None parameters)
        
    Returns
    -------
    Tuple[Callable, Any]
        A tuple containing the resolved apply function and its associated parameters (or None if not applicable).
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

def resolve_sampling_hooks_for_network(net: Any, adapter: Any = None) -> Dict[str, Any]:
    """
    Resolve sampler-facing hooks either from an explicit adapter or directly from the network.
    
    The sampling hooks are a dictionary of callables and metadata that the sampler can use to interact with the network.
    If an adapter is provided and has a resolve_sampling_hooks() method, we use that. Otherwise, 
    we try to resolve hooks directly from the network using common conventions 
    (like apply_fun, parameters, log_psi_delta, etc.).
    
    - log_psi_delta is a computation of the change in log(psi) for local updates,
    which some samplers can use for efficient acceptance probability calculations.
    - apply_fun and parameters are the core callable and its parameters that the sampler will use to evaluate the network on configurations.
    """
    if adapter is not None and hasattr(adapter, "resolve_sampling_hooks"):
        hooks = adapter.resolve_sampling_hooks()
        if hooks is not None:
            return hooks

    apply_fun, params   = resolve_apply_and_params(net)
    log_psi_delta       = getattr(net, "log_psi_delta", None)
    if log_psi_delta is None and hasattr(net, "get_log_psi_delta"):
        log_psi_delta   = net.get_log_psi_delta()

    cache_init = getattr(net, "init_log_psi_delta_cache", None)
    if cache_init is None and hasattr(net, "get_log_psi_delta_cache_init"):
        cache_init = net.get_log_psi_delta_cache_init()

    supports_cache = bool(getattr(net, "log_psi_delta_supports_cache", False))

    return {
        "apply_fun"                     : apply_fun,
        "parameters"                    : params,
        "log_psi_delta"                 : log_psi_delta,
        "log_psi_delta_cache_init"      : cache_init,
        "log_psi_delta_supports_cache"  : supports_cache,
    }

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

    family          = "dense"
    sampler_kind    = "MCSampler"

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
        apply_fun, params = resolve_apply_and_params(self.net)
        supports_cache = bool(getattr(self.net, "log_psi_delta_supports_cache", False))
        return {
            "family"                        : self.family,
            "sampler_kind"                  : self.sampler_kind,
            "apply_fun"                     : apply_fun,
            "parameters"                    : params,
            "native_representation"         : self.native_representation,
            "log_psi_delta"                 : getattr(self.net, "log_psi_delta", None),
            "log_psi_delta_cache_init"      : getattr(self.net, "init_log_psi_delta_cache", None),
            "log_psi_delta_supports_cache"  : supports_cache,
            "representation"                : self.representation,
            "metadata"                      : self.metadata,
        }

# ------------------------------------------------------------------------------
class NQSRBMAdapter(NQSNetAdapterBase):
    family = "rbm"
    sampler_kind = "MCSampler"

    def resolve_sampling_hooks(self) -> Dict[str, Any]:
        hooks = super().resolve_sampling_hooks()

        if hooks["log_psi_delta"] is None and hasattr(self.net, "get_log_psi_delta"):
            hooks["log_psi_delta"] = self.net.get_log_psi_delta()

        if hooks["log_psi_delta_cache_init"] is None and hasattr(self.net, "get_log_psi_delta_cache_init"):
            hooks["log_psi_delta_cache_init"]       = self.net.get_log_psi_delta_cache_init()
            hooks["log_psi_delta_supports_cache"]   = bool(getattr(self.net, "log_psi_delta_supports_cache", False))

        return hooks

# ------------------------------------------------------------------------------

class NQSDenseEvalAdapter(NQSNetAdapterBase):
    family          = "dense"
    sampler_kind    = "MCSampler"

class NQSAutoregressiveAdapter(NQSNetAdapterBase):
    family          = "autoregressive"
    sampler_kind    = "ARSampler"

    def resolve_sampling_hooks(self) -> Dict[str, Any]:
        hooks                   = super().resolve_sampling_hooks()
        hooks["logits_method"]  = (
            (lambda module, x: module.get_logits_binary(x))
            if hasattr(self.net, "get_logits_binary")
            else (lambda module, x: module.get_logits(x, is_binary=True))
        )
        hooks["phase_method"]   = (
            (lambda module, x: module.get_phase_binary(x))
            if hasattr(self.net, "get_phase_binary")
            else (lambda module, x: module.get_phase(x, is_binary=True))
        )
        return hooks

# ------------------------------------------------------------------------------
#! Network adapter factory
# ------------------------------------------------------------------------------

def choose_nqs_network_adapter(net: Any, representation: ModelRepresentationInfo, backend: str = "jax") -> NQSNetAdapterBase:
    """
    Pick the sampler-facing adapter for a network once during setup.
    This is a coarse classification based on heuristics and metadata, not meant to be perfect. 
    The main goal is to detect autoregressive and RBM-like structures,
    but it also looks for explicit metadata flags and common naming patterns.
    
    Parameters
    ----------
    net : Any
        The network object to adapt. This can be a variety of types, including Flax modules, custom network classes, or even plain callables.
    representation : ModelRepresentationInfo
        The model representation information that the adapter can use to resolve representation-specific values.
    backend : str, optional
        The backend to target (e.g., "jax", "torch"). This can be used by adapters to determine how to resolve certain hooks or values. Default is "jax".
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
    # Utility functions for resolving hooks and values
    "resolve_apply_and_params",
    "resolve_sampling_hooks_for_network",
]

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------
