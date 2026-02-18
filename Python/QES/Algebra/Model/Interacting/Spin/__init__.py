"""
Spin Models Module
==================

This module contains various quantum spin models with interactions.

Modules:
--------
- heisenberg_kitaev:
    Heisenberg-Kitaev model implementations
- qsm:
    Quantum Spin Models
- transverse_ising:
    Transverse Field Ising Model
- ultrametric:
    Ultrametric spin models
- xxz:
    XXZ Spin Model

------------------------------------------------------------------------
File        : Algebra/Model/Interacting/Spin/__init__.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
License     : MIT
------------------------------------------------------------------------
"""

__all__ = [
    "heisenberg_kitaev", "qsm", "transverse_ising", "ultrametric", "j1j2", "xxz",
    "hamiltonian_spin",
    "HeisenbergKitaev", "QSM", "TransverseFieldIsing", "UltrametricModel", "J1J2Model", "XXZ",
    "HamiltonianSpin",
    "choose_model"
]

# Lazy import machinery (PEP 562 style)
import importlib

_MAPPINGS = {
    "HamiltonianSpin"      : ".hamiltonian_spin",
    "HeisenbergKitaev"      : ".heisenberg_kitaev",
    "QSM"                   : ".qsm",
    "TransverseFieldIsing"  : ".transverse_ising",
    "UltrametricModel"      : ".ultrametric",
    "J1J2Model"             : ".j1j2",
    "XXZ"                   : ".xxz",
}

def __getattr__(name):
    if name in _MAPPINGS:
        module = importlib.import_module(_MAPPINGS[name], __name__)
        return getattr(module, name)
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)

def choose_model(model_name: str, **kwargs):
    """
    Returns an instance of a spin model of the desired type.

    Args:
        model_name (str):
            Type of model ("heisenberg_kitaev", "qsm", "transverse_ising",
            "ultrametric", "j1j2", or "xxz")
        **kwargs:
            Parameters for the model constructor.

    Returns:
        Hamiltonian: An instance of the desired quantum spin model.
    """
    model_name_map = {
        "heisenberg_kitaev" : "HeisenbergKitaev",
        "kitaev"            : "HeisenbergKitaev",
        "heisenberg"        : "HeisenbergKitaev",
        "qsm"               : "QSM",
        "transverse_ising"  : "TransverseFieldIsing",
        "tfim"              : "TransverseFieldIsing",
        "ultrametric"       : "UltrametricModel",
        "j1j2"              : "J1J2Model",
        "j1_j2"             : "J1J2Model",
        "xxz"               : "XXZ",
    }

    # Normalize model name
    lookup = model_name.lower().replace(" ", "_").replace("-", "_")

    cls_name = None
    if lookup in model_name_map:
        cls_name = model_name_map[lookup]
    elif model_name in _MAPPINGS:
        cls_name = model_name

    if cls_name is None:
        raise ValueError(f"Unknown spin model '{model_name}'. Available: {list(model_name_map.keys())}")

    cls = __getattr__(cls_name)
    return cls(**kwargs)


# ----------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------
