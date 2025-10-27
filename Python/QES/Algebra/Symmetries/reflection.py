"""
This module provides reflection symmetry operators for Hilbert space construction.
It is extensible and compatible with the modular symmetry framework.

--------------------------------------------------
File        : QES/Algebra/Symmetries/reflection.py
Description : Reflection symmetry operators and utilities for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
--------------------------------------------------
"""

from typing import Tuple

try:
    from QES.general_python.common.binary import rev
    from QES.general_python.lattices.lattice import Lattice
except ImportError:
    rev = None  # type: ignore
    
from QES.Algebra.Symmetries.base import SymmetryOperator, SymmetryClass, MomentumSector

####################################################################################################
# Reflection symmetry operator class
####################################################################################################

class ReflectionSymmetry(SymmetryOperator):
    """
    Spatial reflection/inversion symmetry for lattice systems.
    
    Physical Context
    ----------------
    Applies to systems with mirror/inversion symmetry:
    - 1D: Reflection about center -> bit-reversal for spin chains
    - 2D: Mirror planes in square/triangular lattices
    - 3D: Inversion through lattice center
    
    Quantum numbers: Reflection parity r = ±1
    - r = +1: Even parity (symmetric states)
    - r = -1: Odd parity (antisymmetric states)
    
    Examples:
    - Open boundary chains: Always has reflection symmetry
    - PBC chains: Reflection symmetry independent of k
    - Square lattice: Can have x-reflection, y-reflection, or both
    
    Commutation Rules
    -----------------
    Always commutes with:
    - U1_PARTICLE, U1_SPIN: Conserved quantities spatial-independent
    - PARITY: Spatial reflection commutes with spin flips
    - INVERSION: Reflection is a spatial operation
    - POINT_GROUP: Part of point group symmetries
    
    Conditionally commutes with (momentum-dependent):
    - TRANSLATION: Only at k=0, pi momentum sectors
      -> Reason: sigma * T * sigma^(-1) = T^(-1), so need e^(ik) = e^(-ik)
      -> This holds only for k=0 (trivial) or k=pi (e^(i*pi)=-1=e^(-i*pi))
      -> Generic k: Reflection maps k -> -k (different momentum sectors)
    
    Physical interpretation:
    - Reflection + translation form dihedral group D_L at k=0, pi
    - At generic k: Must choose either k or -k sector (not both)
    """
    
    symmetry_class          = SymmetryClass.REFLECTION
    compatible_with         = {
                                SymmetryClass.U1_GLOBAL,
                                SymmetryClass.U1_PARTICLE,
                                SymmetryClass.U1_SPIN,
                                SymmetryClass.PARITY,
                                SymmetryClass.INVERSION,
                                SymmetryClass.POINT_GROUP,
                            }
    momentum_dependent      = {
                                MomentumSector.ZERO : {SymmetryClass.TRANSLATION},
                                MomentumSector.PI   : {SymmetryClass.TRANSLATION},
                            }
    
    # Reflection is universal - works for all local space types
    supported_local_spaces  = set()  # Empty = universal
    
    def __init__(self, lattice: 'Lattice', base: int = 2):
        self.lattice            = lattice
        self.base               = base

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~