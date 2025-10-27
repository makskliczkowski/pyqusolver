"""
This module provides reflection symmetry operators for Hilbert space construction.
It is extensible and compatible with the modular symmetry framework.

File        : QES/Algebra/Symmetries/reflection.py
Description : Reflection symmetry operators and utilities for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

from typing import Tuple, TYPE_CHECKING

try:
    from QES.general_python.common.binary import rev
except ImportError:
    rev = None  # type: ignore
    
from QES.Algebra.Symmetries.base import SymmetryOperator

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice import Lattice

####################################################################################################
# Reflection symmetry operator class
####################################################################################################

class ReflectionSymmetry(SymmetryOperator):
    """
    Reflection symmetry handler for Hilbert space.
    """
    def __init__(self, lattice: 'Lattice', base: int = 2):
        self.lattice = lattice
        self.base = base

    def __call__(self, state: int) -> Tuple[int, complex]:
        return self.apply(state)

    def apply(self, state: int) -> Tuple[int, complex]:
        ns = getattr(self.lattice, 'Ns', None) or getattr(self.lattice, 'ns', None) or getattr(self.lattice, 'sites', None)
        if rev is None:
            raise ImportError("rev() not available. Ensure QES.general_python.common.binary is installed.")
        new_state = rev(state, ns, backend='default')
        # Ensure result is within ns-bit space
        new_state = int(new_state) & ((1 << ns) - 1)
        return new_state, 1.0

####################################################################################################
