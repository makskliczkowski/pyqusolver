"""
Parity (flip) symmetry operators and utilities for Hilbert space construction.

--------------------------------------------
File        : QES/Algebra/Symmetries/parity.py
Description : Parity (flip) symmetry operators and utilities for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
--------------------------------------------
"""

from typing import Tuple, TYPE_CHECKING
try:
    from QES.general_python.common.binary import flip_all, popcount
except ImportError:
    flip_all                = None  # type: ignore
    popcount                = None  # type: ignore

from QES.Algebra.Symmetries.base import (
    SymmetryOperator,
    SymmetryClass,
    MomentumSector,
    LocalSpaceTypes,
)

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice import Lattice

####################################################################################################
# Parity (flip) symmetry operator class
####################################################################################################

class ParitySymmetry(SymmetryOperator):
    """
    Global spin-flip (parity) symmetry operators.
    
    Physical Context
    ----------------
    Applies to discrete spin-flip symmetries in quantum spin systems:
    - Parity X (sigma^x): Flips all spins in x-basis -> (1+i)^n phase, n=spin-ups
    - Parity Y (sigma^y): Flips all spins in y-basis -> (±i)^n phase
    - Parity Z (sigma^z): Flips all spins in z-basis -> (-1)^n phase
    
    Examples:
    - XXZ model: Pz symmetry (conserves Sz but not Sx, Sy)
    - Transverse-field Ising: Px symmetry
    - XY model: May have Pz symmetry
    
    Commutation Rules
    -----------------
    Always commutes with:
    - TRANSLATION: Abelian group [T, P] = 0 for discrete lattice translations
    - REFLECTION: Spatial symmetry commutes with internal spin symmetry
    - PARITY: Different Pauli matrices commute
    - INVERSION: Spatial inversion independent of spin flip
    - U1_GLOBAL, U1_PARTICLE: Backward compatibility alias
    
    Conditionally commutes with:
    - U1_SPIN: Only Pz commutes with Sz conservation (X, Y break it)
    - U1_PARTICLE: Only if system is at half-filling N = Ns/2 (even Ns)
      -> Reason: Particle-hole transformation maps N <-> Ns - N
    """
    
    symmetry_class              = SymmetryClass.PARITY
    compatible_with             = {
                                    SymmetryClass.TRANSLATION,
                                    SymmetryClass.REFLECTION,
                                    SymmetryClass.U1_GLOBAL,  # Backward compat
                                    SymmetryClass.U1_PARTICLE,
                                    SymmetryClass.PARITY,
                                    SymmetryClass.INVERSION,
                                    SymmetryClass.POINT_GROUP,
                                }
    supported_local_spaces      = {
                                    LocalSpaceTypes.SPIN_1_2,
                                    LocalSpaceTypes.SPIN_1,
                                    # NOT supported for fermions/bosons (different parity operators needed)
                                }
    
    def __init__(self, lattice: 'Lattice', axis: str = 'z'):
        self.lattice            = lattice
        self.axis               = axis

    def __call__(self, state: int) -> Tuple[int, complex]:
        return self.apply(state)

    def apply(self, state: int) -> Tuple[int, complex]:
        ns = getattr(self.lattice, 'Ns', None) or getattr(self.lattice, 'ns', None) or getattr(self.lattice, 'sites', None)
        if flip_all is None or popcount is None:
            raise ImportError("flip_all/popcount not available. Ensure QES.general_python.common.binary is installed.")
        if self.axis == 'z':
            # Parity Z: flip all spins, phase +1
            new_state = int(flip_all(state, ns, backend='default')) & ((1 << ns) - 1)
            phase = 1.0
        elif self.axis == 'y':
            # Parity Y: flip all spins, phase depends on spin-ups
            spin_ups = ns - popcount(state, backend='default')
            phase = (1 - 2 * (spin_ups & 1)) * (1j if ns % 2 == 0 else -1j)
            new_state = int(flip_all(state, ns, backend='default')) & ((1 << ns) - 1)
        elif self.axis == 'x':
            # Parity X: flip all spins, phase +1
            new_state = int(flip_all(state, ns, backend='default')) & ((1 << ns) - 1)
            phase = 1.0
        else:
            raise ValueError(f"Unknown parity axis: {self.axis}")
        return new_state, phase

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~