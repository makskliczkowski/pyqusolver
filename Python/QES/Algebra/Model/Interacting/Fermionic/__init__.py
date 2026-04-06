"""Interacting fermionic models."""

from __future__ import annotations
import importlib

from ..._registry import create_model as _create_model, get_model_export_names as _get_model_export_names, resolve_model_export as _resolve_model_export

__all__ = [
    # The spinful Hubbard model with on-site interactions and spin degrees of freedom.
    "spinful_hubbard",
    "SpinfulHubbardModel",
    # The spinless Hubbard model
    "hubbard",
    "HubbardModel",
    # Free fermion models
    "free_fermion_manybody",
    "ManyBodyFreeFermions",
    "choose_model",
]

_CLASS_MAP = {
    "ManyBodyFreeFermions"      : ".free_fermion_manybody",
    "HubbardModel"              : ".hubbard",
    "SpinfulHubbardModel"       : ".spinful_hubbard",
}

def __getattr__(name: str):
    if name in _CLASS_MAP:
        module = importlib.import_module(_CLASS_MAP[name], __name__)
        return getattr(module, name)
    if name in {"free_fermion_manybody", "hubbard", "spinful_hubbard"}:
        return importlib.import_module(f".{name}", __name__)
    try:
        return _resolve_model_export(name, family="interacting_fermionic")
    except ValueError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_get_model_export_names(family="interacting_fermionic")))

def choose_model(model_name: str, **kwargs):
    return _create_model(model_name, family="interacting_fermionic", **kwargs)

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
