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

__all__ = ["heisenberg_kitaev", "qsm", "transverse_ising", "ultrametric", "j1j2"]

# Lazy import machinery (PEP 562 style)
import importlib
import sys


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


# ----------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------
