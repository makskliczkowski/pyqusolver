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

# Base classes and enumerations
from QES.Algebra.Symmetries.base import (
    MomentumSector,
    SymmetryClass,
    SymmetryOperator,
    SymmetryRegistry,
)
from QES.Algebra.Symmetries.inversion import (
    InversionSymmetry,
    build_inversion_operator_matrix,
)

# Momentum sector analysis
from QES.Algebra.Symmetries.momentum_sectors import (
    MomentumSectorAnalyzer,
)
from QES.Algebra.Symmetries.parity import (
    ParitySymmetry,
)
from QES.Algebra.Symmetries.reflection import (
    ReflectionSymmetry,
)

# Compact symmetry data structure
from QES.Algebra.Symmetries.symmetry_container import (
    CompactSymmetryData,
    SymmetryContainer,
    _compact_get_phase,
    _compact_get_repr_idx,
    _compact_get_sym_factor,
    _compact_is_in_sector,
)

# Symmetry operators
from QES.Algebra.Symmetries.translation import (
    TranslationSymmetry,
    build_momentum_superposition,
)

####################################################################################################
#! Public API
####################################################################################################

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#! EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
