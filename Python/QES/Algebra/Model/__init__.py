"""
QES Model Module
================

This module provides implementations of various quantum many-body models.

Submodules:
-----------
- Interacting: Models with particle interactions
- Noninteracting: Free particle and non-interacting models

Classes:
--------
Various quantum model implementations including:
- Spin models (Heisenberg, Ising, XY, etc.)
- Fermionic models (Hubbard, t-J, etc.)
- Bosonic models

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_IMPORTS = {
    "intr": (".Interacting", None),
    "nintr": (".Noninteracting", None),
}

if TYPE_CHECKING:
    from . import Interacting as intr
    from . import Noninteracting as nintr


def __getattr__(name: str) -> Any:
    """Lazily import submodules."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


__all__ = ["intr", "nintr"]
