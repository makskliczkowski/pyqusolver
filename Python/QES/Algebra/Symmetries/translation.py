"""
This module provides translation symmetry operators for 1D, 2D, and 3D lattices.
It supports block-diagonalization in momentum sectors and is extensible for new symmetry types.

File        : QES/Algebra/Symmetries/translation.py
Description : Translation symmetry operators and utilities for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

import numpy as np
from typing import Tuple, Optional

from typing import TYPE_CHECKING
try:
    from QES.general_python.lattices.lattice import LatticeDirection
except ImportError:
    LatticeDirection = None  # type: ignore
from QES.Algebra.Symmetries.base import SymmetryOperator

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice import Lattice
####################################################################################################
# Translation symmetry operator class
####################################################################################################

class TranslationSymmetry(SymmetryOperator):
    """
    General translation symmetry handler for 1D, 2D, 3D lattices.
    Provides translation operator and momentum sector projection.
    """
    def __init__(self, lattice: 'Lattice', direction: LatticeDirection = LatticeDirection.X):
        self.lattice = lattice
        self.direction = direction
        self.perm, self.crossing_mask = self._compute_translation_data()

    def _compute_translation_data(self):
        """
        Pre-compute permutation and crossing mask for translation in the given direction.
        """
        lat = self.lattice
        direction = self.direction
        axis_map = {
            LatticeDirection.X: 0,
            LatticeDirection.Y: 1,
            LatticeDirection.Z: 2,
        }
        axis = axis_map[direction]
        dims = (lat.lx, lat.ly if lat.dim > 1 else 1, lat.lz if lat.dim > 2 else 1)
        size_axis = dims[axis]
        ns = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None) or getattr(lat, 'sites', None)
        perm = np.empty(ns, dtype=np.int64)
        crossing_mask = np.zeros(ns, dtype=bool)
        for site in range(ns):
            coord = list(lat.get_coordinates(site))
            while len(coord) < 3:
                coord.append(0)
            new_coord = coord.copy()
            new_coord[axis] += 1
            wrap = False
            if new_coord[axis] >= size_axis:
                new_coord[axis] -= size_axis
                wrap = True
            dest = lat.site_index(int(new_coord[0]), int(new_coord[1]), int(new_coord[2]))
            perm[site] = dest
            if wrap:
                crossing_mask[site] = True
        return perm, crossing_mask

    def __call__(self, state: int) -> Tuple[int, complex]:
        return self.apply(state)

    def apply(self, state: int) -> Tuple[int, complex]:
        """
        Apply translation to integer state, return (new_state, phase)
        """
        lat = self.lattice
        ns = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None) or getattr(lat, 'sites', None)
        perm = self.perm
        crossing_mask = self.crossing_mask
        new_state = 0
        for src in range(ns):
            if (state >> (ns - 1 - src)) & 1:
                dest = int(perm[src])
                new_state |= 1 << (ns - 1 - dest)
        # Fermionic sign or boundary phase
        if crossing_mask.any():
            crossing_sites = np.nonzero(crossing_mask)[0]
            occ_cross = sum(1 for src in crossing_sites if (state >> (ns - 1 - src)) & 1)
        else:
            occ_cross = 0
        phase = lat.boundary_phase(self.direction, occ_cross)
        return new_state, phase

    @staticmethod
    def momentum_phase(k: int, L: int) -> complex:
        # e^{i 2pi k / L}
        return np.exp(1j * 2 * np.pi * k / L)

    # TODO: Add methods for block-diagonalization, sector projection, etc.
