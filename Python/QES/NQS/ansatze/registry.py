"""
Registry for ansatz names owned by ``QES.NQS``.

It allows to refer to ansatze by a canoncial name
(e.g. "rbm", "cnn", "resnet", "ar", "mlp", "gcnn", "eqgcnn", "transformer", "pp", "rbmpp", "jastrow", "mps", "amplitude_phase") instead of the full class path, 
and also to resolve ansatz requests that may be given as either a string or a direct class reference.
"""

from dataclasses import dataclass, field
import importlib
from typing import Any, Dict, Tuple

@dataclass(frozen=True)
class AnsatzEntry:
    module_path     : str
    attr_name       : str
    default_kwargs  : Dict[str, Any] = field(default_factory=dict)

_PRIMARY_ANSATZE: Tuple[str, ...] = (
    "rbm",
    "cnn",
    "resnet",
    "ar",
    "mlp",
    "eqgcnn",
    "transformer",
    "pp",
    "rbmpp",
    "jastrow",
    "mps",
    "amplitude_phase",
    "approx_symmetric",
)

_ANSATZ_REGISTRY: Dict[str, AnsatzEntry] = {
    "rbm":              AnsatzEntry("QES.general_python.ml.net_impl.networks.net_rbm",          "RBM"),
    "cnn":              AnsatzEntry("QES.general_python.ml.net_impl.networks.net_cnn",          "CNN"),
    "resnet":           AnsatzEntry("QES.general_python.ml.net_impl.networks.net_res",          "ResNet"),
    "ar":               AnsatzEntry(".autoregressive",                                          "ComplexAR"),
    "mlp":              AnsatzEntry("QES.general_python.ml.net_impl.networks.net_mlp",          "MLP"),
    "eqgcnn":           AnsatzEntry(".equivariant_gcnn", "EquivariantGCNN"),
    "transformer":      AnsatzEntry("QES.general_python.ml.net_impl.networks.net_transformer",  "Transformer"),
    "pp":               AnsatzEntry(".pair_product", "PairProduct"),
    "rbmpp":            AnsatzEntry(".pair_product", "PairProduct", default_kwargs={"use_rbm": True}),
    "jastrow":          AnsatzEntry(".jastrow", "Jastrow"),
    "mps":              AnsatzEntry(".mps", "MPS"),
    "amplitude_phase":  AnsatzEntry(".amplitude_phase", "AmplitudePhase"),
    "approx_symmetric": AnsatzEntry(".approx_symmetric", "AnsatzApproxSymmetric"),
}

# ----------------------------------------------------------------------
# Registry utilities
# ----------------------------------------------------------------------

def canonicalize_ansatz_name(name: Any) -> str:
    ''' Convert an ansatz name to a canonical form for registry lookup. This typically involves lowercasing and replacing certain characters. '''
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")

def is_registered_ansatz(name: Any) -> bool:
    ''' Check if the given name corresponds to a registered ansatz. '''
    return canonicalize_ansatz_name(name) in _ANSATZ_REGISTRY

def list_available_ansatze():
    ''' List the canonical names of all registered ansatze. '''
    return list(_PRIMARY_ANSATZE)

def load_ansatz_class(name: Any):
    ''' Load the ansatz class corresponding to the given name. Raises ValueError if the name is not registered. '''
    key     = canonicalize_ansatz_name(name)
    if key not in _ANSATZ_REGISTRY:
        raise ValueError(f"Unknown NQS ansatz '{name}'. Supported NQS ansatze: {', '.join(_PRIMARY_ANSATZE)}.")

    entry   = _ANSATZ_REGISTRY[key]
    module  = importlib.import_module(entry.module_path, package=__package__)
    return getattr(module, entry.attr_name)

def resolve_ansatz_request(ansatz: Any, kwargs: Dict[str, Any] | None = None):
    ''' Resolve an ansatz request that may be given as either a string or a direct class reference. Returns a tuple of (ansatz_class, resolved_kwargs). '''
    resolved_kwargs = dict(kwargs or {})
    if not isinstance(ansatz, str):
        return ansatz, resolved_kwargs

    key     = canonicalize_ansatz_name(ansatz)
    entry   = _ANSATZ_REGISTRY.get(key)
    if entry is None:
        return ansatz, resolved_kwargs

    for name, value in entry.default_kwargs.items():
        resolved_kwargs.setdefault(name, value)
    module = importlib.import_module(entry.module_path, package=__package__)
    return getattr(module, entry.attr_name), resolved_kwargs

def resolve_ansatz_type(ansatz: Any):
    ''' Resolve an ansatz request to its corresponding class type. Raises ValueError if the ansatz cannot be resolved. '''
    resolved_type, _ = resolve_ansatz_request(ansatz)
    return resolved_type

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
