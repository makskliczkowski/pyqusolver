"""
NQS ansatz registry and exports.

This package owns string-based ansatz selection for ``QES.NQS``.
Some ansatze are fully NQS-specific, while backbone-shaped entries are thin
NQS wrappers that translate state-convention kwargs before delegating to the
general-purpose implementations.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

import importlib
from typing import Any

_LAZY_EXPORTS = {
    # Backbone-shaped wrappers
    "RBM"                           : (".shared", "RBM"),
    "CNN"                           : (".shared", "CNN"),
    "ResNet"                        : (".shared", "ResNet"),
    "MLP"                           : (".shared", "MLP"),
    "GCNN"                          : (".shared", "GCNN"),
    "Transformer"                   : (".shared", "Transformer"),
    # Fully NQS-specific ansatze
    "ComplexAR"                     : (".autoregressive",   "ComplexAR"),
    "PairProduct"                   : (".pair_product",     "PairProduct"),
    "Jastrow"                       : (".jastrow",          "Jastrow"),
    "MPS"                           : (".mps",              "MPS"),
    "AmplitudePhase"                : (".amplitude_phase",  "AmplitudePhase"),
    "AnsatzApproxSymmetric"         : (".approx_symmetric", "AnsatzApproxSymmetric"),
    "EquivariantGCNN"               : (".equivariant_gcnn", "EquivariantGCNN"),
    # Registry utilities
    "canonicalize_ansatz_name"      : (".registry",         "canonicalize_ansatz_name"),
    "is_registered_ansatz"          : (".registry",         "is_registered_ansatz"),
    "list_available_ansatze"        : (".registry",         "list_available_ansatze"),
    "load_ansatz_class"             : (".registry",         "load_ansatz_class"),
    "resolve_ansatz_request"        : (".registry",         "resolve_ansatz_request"),
    "resolve_ansatz_type"           : (".registry",         "resolve_ansatz_type"),
}

_LAZY_CACHE = {}

# ----------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_path, package=__name__)
    value = getattr(module, attr_name)
    _LAZY_CACHE[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = sorted(_LAZY_EXPORTS)

# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------
