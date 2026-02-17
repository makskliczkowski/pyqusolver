"""Interacting fermionic models."""

from __future__ import annotations
import importlib

__all__ = [
    "free_fermion_manybody",
    "hubbard",
    "ManyBodyFreeFermions",
    "HubbardModel",
    "choose_model",
]

_CLASS_MAP = {
    "ManyBodyFreeFermions"      : ".free_fermion_manybody",
    "HubbardModel"              : ".hubbard",
}

_MODEL_NAME_MAP = {
    "manybody_free_fermions"    : "ManyBodyFreeFermions",
    "free_fermions_manybody"    : "ManyBodyFreeFermions",
    "free_fermion_manybody"     : "ManyBodyFreeFermions",
    "hubbard"                   : "HubbardModel",
    "spinless_hubbard"          : "HubbardModel",
    "hubbard_spinless"          : "HubbardModel",
}

def __getattr__(name: str):
    if name in _CLASS_MAP:
        module = importlib.import_module(_CLASS_MAP[name], __name__)
        return getattr(module, name)
    if name in {"free_fermion_manybody", "hubbard"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)

def choose_model(model_name: str, **kwargs):
    key = model_name.lower().replace(" ", "_").replace("-", "_")
    if key not in _MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown interacting fermionic model '{model_name}'. "
            f"Available: {sorted(_MODEL_NAME_MAP.keys())}"
        )
    cls = __getattr__(_MODEL_NAME_MAP[key])
    return cls(**kwargs)

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------