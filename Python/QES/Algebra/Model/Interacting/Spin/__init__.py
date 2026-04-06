"""
Spin Models Module
==================

This module contains various quantum spin models with interactions.

Modules:
--------

1. General Spin Models:
- heisenberg_kitaev:
    Heisenberg-Kitaev model implementations
- transverse_ising:
    Transverse Field Ising Model
- xxz:
    XXZ Spin Model
- j1j2:
    J1-J2 Spin Model
2. Spin Random Models:
- qsm:
    Quantum Spin Models
- ultrametric:
    Ultrametric spin models
3. General Hamiltonian:
- hamiltonian_spin:
    General Hamiltonian for spin systems

------------------------------------------------------------------------
File        : Algebra/Model/Interacting/Spin/__init__.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
------------------------------------------------------------------------
"""

from __future__ import annotations
from typing     import TYPE_CHECKING

__all__ = [
    "heisenberg_kitaev", "qsm", "transverse_ising", "ultrametric", "j1j2", "xxz",
    "hamiltonian_spin",
    "HeisenbergKitaev", "QSM", "TransverseFieldIsing", "UltrametricModel", "J1J2Model", "XXZ",
    "HamiltonianSpin",
    "choose_model"
]

# Lazy import machinery (PEP 562 style)
import importlib

try:
    from ..._registry import create_model as _create_model, get_model_export_names as _get_model_export_names, resolve_model_export as _resolve_model_export
except ImportError:
    raise ImportError("Failed to import model registry utilities. Ensure the general_python package is correctly installed.")

_MAPPINGS = {
    "HamiltonianSpin"      : ".hamiltonian_spin",
    "HeisenbergKitaev"      : ".heisenberg_kitaev",
    "QSM"                   : ".qsm",
    "TransverseFieldIsing"  : ".transverse_ising",
    "UltrametricModel"      : ".ultrametric",
    "J1J2Model"             : ".j1j2",
    "XXZ"                   : ".xxz",
}

if TYPE_CHECKING:
    # Spin models
    from .hamiltonian_spin  import HamiltonianSpin
    from .heisenberg_kitaev import HeisenbergKitaev
    from .transverse_ising  import TransverseFieldIsing
    from .j1j2              import J1J2Model
    from .xxz               import XXZ
    # Spin Random models
    from .qsm               import QSM
    from .ultrametric       import UltrametricModel

def __getattr__(name):
    if name in _MAPPINGS:
        module = importlib.import_module(_MAPPINGS[name], __name__)
        return getattr(module, name)
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    try:
        return _resolve_model_export(name, family="interacting_spin")
    except ValueError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_get_model_export_names(family="interacting_spin")))

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
    return _create_model(model_name, family="interacting_spin", **kwargs)

# ----------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------
