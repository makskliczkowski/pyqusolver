"""
Symmetry operations for quantum many-body systems.

This module provides symmetry operators for Hilbert space construction:
- Translation symmetry (momentum sectors)
- Reflection symmetry (1D bit-reversal)
- Inversion symmetry (general spatial inversion for any lattice/dimension)
- Parity symmetry (particle-hole, spin-flip)
- Compatibility checking and automatic filtering

---------------------------------------------------
File        : QES/Algebra/Symmetries/__init__.py
Description : Symmetry operations module initialization
Author      : Maksymilian Kliczkowski
Date        : 2025-10-27
---------------------------------------------------
"""

import importlib
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Lazy Import Configuration
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # base
    "MomentumSector": (".base", "MomentumSector"),
    "SymmetryClass": (".base", "SymmetryClass"),
    "SymmetryOperator": (".base", "SymmetryOperator"),
    "SymmetryRegistry": (".base", "SymmetryRegistry"),
    # inversion
    "InversionSymmetry": (".inversion", "InversionSymmetry"),
    "build_inversion_operator_matrix": (".inversion", "build_inversion_operator_matrix"),
    # parity
    "ParitySymmetry": (".parity", "ParitySymmetry"),
    # reflection
    "ReflectionSymmetry": (".reflection", "ReflectionSymmetry"),
    # translation
    "TranslationSymmetry": (".translation", "TranslationSymmetry"),
    "build_momentum_superposition": (".translation", "build_momentum_superposition"),
    # momentum_sectors
    "MomentumSectorAnalyzer": (".momentum_sectors", "MomentumSectorAnalyzer"),
    # symmetry_container
    "CompactSymmetryData": (".symmetry_container", "CompactSymmetryData"),
    "SymmetryContainer": (".symmetry_container", "SymmetryContainer"),
    "_compact_get_phase": (".symmetry_container", "_compact_get_phase"),
    "_compact_get_repr_idx": (".symmetry_container", "_compact_get_repr_idx"),
    "_compact_get_sym_factor": (".symmetry_container", "_compact_get_sym_factor"),
    "_compact_is_in_sector": (".symmetry_container", "_compact_is_in_sector"),
}

if TYPE_CHECKING:
    from .base import MomentumSector, SymmetryClass, SymmetryOperator, SymmetryRegistry
    from .inversion import InversionSymmetry, build_inversion_operator_matrix
    from .momentum_sectors import MomentumSectorAnalyzer
    from .parity import ParitySymmetry
    from .reflection import ReflectionSymmetry
    from .symmetry_container import (
        CompactSymmetryData,
        SymmetryContainer,
        _compact_get_phase,
        _compact_get_repr_idx,
        _compact_get_sym_factor,
        _compact_is_in_sector,
    )
    from .translation import TranslationSymmetry, build_momentum_superposition


def __getattr__(name: str) -> Any:
    """Lazily import submodules and classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


####################################################################################################
# Module metadata
####################################################################################################

__version__ = "1.0.0"
__author__ = "Maksymilian Kliczkowski"
__date__ = "2025-10-27"

####################################################################################################
# Convenience function for getting available symmetries
####################################################################################################


def get_available_symmetries():
    """
    Get list of available symmetry operators.
    TODO: Expand as new symmetries are added.

    Returns
    -------
    dict
        Dictionary mapping symmetry names to their classes.
    """
    # Import classes inside the function to avoid breaking lazy loading for this utility
    from .inversion import InversionSymmetry
    from .parity import ParitySymmetry
    from .reflection import ReflectionSymmetry
    from .translation import TranslationSymmetry

    return {
        "translation": TranslationSymmetry,
        "reflection": ReflectionSymmetry,
        "inversion": InversionSymmetry,
        "parity": ParitySymmetry,
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#! Binomial coefficient function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def choose(n, k):
    """
    Binomial coefficient n choose k.

    Parameters
    ----------
    n : int
        Total number of items
    k : int
        Number of items to choose

    Returns
    -------
    int
        Binomial coefficient C(n, k)
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1

    # Use symmetry: C(n, k) = C(n, n-k)
    k = min(k, n - k)

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)

    return result


__all__ = [
    # Base classes
    "SymmetryOperator",
    "SymmetryClass",
    "MomentumSector",
    "SymmetryRegistry",
    # Symmetry operators
    "TranslationSymmetry",
    "ReflectionSymmetry",
    "InversionSymmetry",
    "ParitySymmetry",
    # Utilities
    "choose",
    "get_available_symmetries",
    # Translation utilities
    "build_momentum_superposition",
    # Momentum analysis
    "MomentumSectorAnalyzer",
    # Compact symmetry data (O(1) JIT-friendly lookups)
    "CompactSymmetryData",
    "SymmetryContainer",
    "_compact_get_sym_factor",
    "_compact_is_in_sector",
    "_compact_get_repr_idx",
    "_compact_get_phase",
]
