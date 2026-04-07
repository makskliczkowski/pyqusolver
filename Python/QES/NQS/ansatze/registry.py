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
    "gcnn",
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
    "rbm":              AnsatzEntry(".shared", "RBM"),
    "cnn":              AnsatzEntry(".shared", "CNN"),
    "res":              AnsatzEntry(".shared", "ResNet"),
    "resnet":           AnsatzEntry(".shared", "ResNet"),
    "ar":               AnsatzEntry(".autoregressive", "ComplexAR"),
    "autoregressive":   AnsatzEntry(".autoregressive", "ComplexAR"),
    "mlp":              AnsatzEntry(".shared", "MLP"),
    "gcnn":             AnsatzEntry(".shared", "GCNN"),
    "eqgcnn":           AnsatzEntry(".equivariant_gcnn", "EquivariantGCNN"),
    "transformer":      AnsatzEntry(".shared", "Transformer"),
    "pp":               AnsatzEntry(".pair_product", "PairProduct"),
    "pair_product":     AnsatzEntry(".pair_product", "PairProduct"),
    "rbmpp":            AnsatzEntry(".pair_product", "PairProduct", default_kwargs={"use_rbm": True}),
    "jastrow":          AnsatzEntry(".jastrow", "Jastrow"),
    "mps":              AnsatzEntry(".mps", "MPS"),
    "amplitude_phase":  AnsatzEntry(".amplitude_phase", "AmplitudePhase"),
    "approx_symmetric": AnsatzEntry(".approx_symmetric", "AnsatzApproxSymmetric"),
    "approxsym":        AnsatzEntry(".approx_symmetric", "AnsatzApproxSymmetric"),
    "asym":             AnsatzEntry(".approx_symmetric", "AnsatzApproxSymmetric"),
}

# ----------------------------------------------------------------------
# Registry utilities
# ----------------------------------------------------------------------

def canonicalize_ansatz_name(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")

def is_registered_ansatz(name: Any) -> bool:
    return canonicalize_ansatz_name(name) in _ANSATZ_REGISTRY

def list_available_ansatze():
    return list(_PRIMARY_ANSATZE)

def load_ansatz_class(name: Any):
    key     = canonicalize_ansatz_name(name)
    if key not in _ANSATZ_REGISTRY:
        raise ValueError(f"Unknown NQS ansatz '{name}'. Supported NQS ansatze: {', '.join(_PRIMARY_ANSATZE)}.")

    entry   = _ANSATZ_REGISTRY[key]
    module  = importlib.import_module(entry.module_path, package=__package__)
    return getattr(module, entry.attr_name)

def resolve_ansatz_request(ansatz: Any, kwargs: Dict[str, Any] | None = None):
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
    resolved_type, _ = resolve_ansatz_request(ansatz)
    return resolved_type

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------