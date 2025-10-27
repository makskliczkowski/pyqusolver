"""
Symmetry operations for quantum many-body systems.

This module provides symmetry operators for Hilbert space construction:
- Translation symmetry (momentum sectors)
- Reflection symmetry (spatial inversion)
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
    SymmetryOperator,
    SymmetryClass,
    MomentumSector,
    SymmetryRegistry,
)

# Symmetry operators
from QES.Algebra.Symmetries.translation import (
    TranslationSymmetry,
    build_momentum_superposition,
)

from QES.Algebra.Symmetries.reflection import (
    ReflectionSymmetry,
)

from QES.Algebra.Symmetries.parity import (
    ParitySymmetry,
)

# Compatibility utilities
from QES.Algebra.Symmetries.compatibility import (
    check_compatibility,
    infer_momentum_sector_from_operators,
)

# Momentum sector analysis
from QES.Algebra.Symmetries.momentum_sectors import (
    MomentumSectorAnalyzer,
)

####################################################################################################
# Public API
####################################################################################################

__all__ = [
    # Base classes
    'SymmetryOperator',
    'SymmetryClass',
    'MomentumSector',
    'SymmetryRegistry',
    
    # Symmetry operators
    'TranslationSymmetry',
    'ReflectionSymmetry',
    'ParitySymmetry',
    
    # Utilities
    'build_momentum_superposition',
    'check_compatibility',
    'infer_momentum_sector_from_operators',
    'get_available_symmetries',
    'choose',
    
    # Momentum analysis
    'MomentumSectorAnalyzer',
]

####################################################################################################
# Module metadata
####################################################################################################

__version__ = '1.0.0'
__author__ = 'Maksymilian Kliczkowski'
__date__ = '2025-10-27'

####################################################################################################
# Convenience function for getting available symmetries
####################################################################################################

def get_available_symmetries():
    """
    Get list of available symmetry operators.
    
    Returns
    -------
    dict
        Dictionary mapping symmetry names to their classes.
    """
    return {
        'translation': TranslationSymmetry,
        'reflection': ReflectionSymmetry,
        'parity': ParitySymmetry,
    }

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
