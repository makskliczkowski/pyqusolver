"""
Parity (flip) symmetry operators and utilities for Hilbert space construction.

File        : QES/Algebra/Symmetries/parity.py
Description : Parity (flip) symmetry operators and utilities for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

from typing import Tuple, TYPE_CHECKING
try:
    from QES.general_python.common.binary import flip_all, popcount
except ImportError:
    flip_all = None  # type: ignore
    popcount = None  # type: ignore
from QES.Algebra.Symmetries.base import SymmetryOperator

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice import Lattice

####################################################################################################
# Parity (flip) symmetry operator class
####################################################################################################

class ParitySymmetry(SymmetryOperator):
    """
    Parity (flip) symmetry handler for Hilbert space.
    axis: 'x', 'y', or 'z'
    """
    def __init__(self, lattice: 'Lattice', axis: str = 'z'):
        self.lattice = lattice
        self.axis = axis

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
