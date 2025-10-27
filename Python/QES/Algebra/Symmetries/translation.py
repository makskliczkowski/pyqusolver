"""
This module provides translation symmetry operators for 1D, 2D, and 3D lattices.
It supports block-diagonalization in momentum sectors and is extensible for new symmetry types.

---------------------------------------------------
File        : QES/Algebra/Symmetries/translation.py
Description : Translation symmetry operators and utilities for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
---------------------------------------------------
"""

import numpy as np
from collections import defaultdict
from itertools import product
from typing import Tuple, Optional, Mapping, Dict, List

try:
    from QES.general_python.lattices.lattice import Lattice
    from QES.general_python.lattices.lattice import LatticeDirection
except ImportError:
    LatticeDirection = None  # type: ignore

from QES.Algebra.Symmetries.base import SymmetryOperator, SymmetryClass, MomentumSector

####################################################################################################
# Translation symmetry operator class
####################################################################################################

class TranslationSymmetry(SymmetryOperator):
    """
    Discrete translation symmetry for periodic lattice systems.
    
    Physical Context
    ----------------
    Applies to periodic boundary condition systems with discrete lattice structure:
    - 1D: Spin chains etc.
    - 2D: Square/triangular/honeycomb lattices
    - 3D: Simple cubic
    
    Quantum numbers: Crystal momentum k = 2*pi*n/L (Bloch theorem)
    - k=0           : Gamma point, fully symmetric sector
    - k=pi          : Zone boundary (L even), alternating sign
    - Generic k     : Complex phases, reduced symmetry
    
    Examples:
    - Heisenberg chain L=12     : k = 0, pi/6, pi/3, pi/2, ..., pi
    - 2D square lattice         : k_x, k_y independent quantum numbers
    
    Commutation Rules
    -----------------
    Always commutes with:
    - TRANSLATION               : Different directions form Abelian group [T_x, T_y] = 0
    - U1_PARTICLE, U1_SPIN      : Conserved quantities independent of momentum
    - PARITY                    : Global spin flips commute with spatial translations
    - INVERSION                 : [T, P] = 0 for discrete translations
    
    Conditionally commutes with (momentum-dependent):
        - REFLECTION: Only at k=0, pi due to sigma * T * sigma^(-1) = T^(-1)
            - At k != 0, pi     : e^(ikr) != e^(-ikr)   -> symmetries incompatible
            - At k=0            : e^(i*0*r) = 1         -> compatible
            - At k=pi           : e^(i*pi*r) = (-1)^r   -> compatible for integer shifts
    
    Does NOT commute with:
        - REFLECTION at generic k (broken inversion symmetry in k-space)
    """
    
    symmetry_class          = SymmetryClass.TRANSLATION
    compatible_with         = {
        SymmetryClass.TRANSLATION,
        SymmetryClass.U1_GLOBAL,
        SymmetryClass.U1_PARTICLE,
        SymmetryClass.U1_SPIN,
        SymmetryClass.PARITY,
        SymmetryClass.INVERSION,
        SymmetryClass.POINT_GROUP,
    }
    momentum_dependent      = {
        MomentumSector.ZERO : {SymmetryClass.REFLECTION},
        MomentumSector.PI   : {SymmetryClass.REFLECTION},
    }
    
    # Translation is universal - works for all local space types
    supported_local_spaces  = set()  # Empty = universal
    
    ####################################################################################################
    # INIT
    ####################################################################################################
    
    def __init__(self, lattice: 'Lattice', direction: LatticeDirection = LatticeDirection.X, momentum_index: Optional[int] = None):
        
        if lattice is None:
            raise ValueError("TranslationSymmetry requires a lattice instance.")
        self.lattice = lattice
        self.direction = direction
        self.momentum_index = momentum_index
        self.perm, self.crossing_mask = self._compute_translation_data()
    
    def get_momentum_sector(self) -> Optional[MomentumSector]:
        """Determine the momentum sector from the momentum index."""
        if self.momentum_index is None:
            return None
        
        extent_map = {
            LatticeDirection.X: getattr(self.lattice, 'lx', getattr(self.lattice, 'Lx', None)),
            LatticeDirection.Y: getattr(self.lattice, 'ly', getattr(self.lattice, 'Ly', None)),
            LatticeDirection.Z: getattr(self.lattice, 'lz', getattr(self.lattice, 'Lz', None)),
        }
        extent = extent_map.get(self.direction)
        if extent is None or extent <= 0:
            return None
        
        # k = 2\pi * momentum_index / extent
        # k=0 when momentum_index=0, k=\pi when momentum_index=extent//2
        if self.momentum_index == 0:
            return MomentumSector.ZERO
        elif extent % 2 == 0 and self.momentum_index == extent // 2:
            return MomentumSector.PI
        else:
            return MomentumSector.GENERIC

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

    def momentum_projector(self, state: int, k: Optional[int] = None) -> Dict[int, complex]:
        """
        Return normalized amplitudes for the momentum-k projection of ``state``
        along this translation direction.
        """
        if k is None:
            if self.momentum_index is None:
                raise ValueError("Momentum index not provided for momentum projector.")
            k = self.momentum_index
        orbit = self.orbit(state)
        m = len(orbit)
        if m == 0:
            return {int(state): 1.0 + 0j}
        coeffs = np.exp(-1j * 2 * np.pi * k * np.arange(m) / m) / np.sqrt(m)
        return {int(orb_state): coeffs[idx] for idx, orb_state in enumerate(orbit)}

    def apply_superposition(self, amplitudes: Mapping[int, complex]) -> Dict[int, complex]:
        """
        Apply the translation to a linear combination expressed as {state: amp}.
        """
        rotated = defaultdict(complex)
        for basis_state, amplitude in amplitudes.items():
            new_state, phase = self.apply(int(basis_state))
            rotated[int(new_state)] += amplitude * phase
        return dict(rotated)

    def orbit(self, state: int, max_steps: Optional[int] = None) -> list[int]:
        """
        Generate the translation orbit of a basis state by repeatedly applying T
        until it cycles back. Returns the list of unique states in order.
        """
        seen = {}
        seq = []
        s = int(state)
        steps = 0
        while s not in seen:
            seen[s] = steps
            seq.append(s)
            s, _ = self.apply(s)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return seq

    def orbit_length(self, state: int) -> int:
        """Number of unique states in the translation orbit of state."""
        return len(self.orbit(state))

    def projector_coefficients(self, k: Optional[int], state: int) -> Tuple[list[int], np.ndarray]:
        """
        Coefficients for the momentum-k projector acting on the orbit of `state`.
        Returns (orbit_states, coeffs) where coeffs[n] = exp(-i 2pi k n / m) / sqrt(m),
        with m = len(orbit). Useful for building momentum eigenstates in ED.
        """
        if k is None:
            if self.momentum_index is None:
                raise ValueError("Momentum index not provided for coefficient generation.")
            k = self.momentum_index
        orb = self.orbit(state)
        m = len(orb)
        phases = np.exp(-1j * 2 * np.pi * k * np.arange(m) / m) / np.sqrt(m)
        return orb, phases

    # TODO: Add methods for block-diagonalization, sector projection, etc.


def build_momentum_superposition(
    translations: Mapping['LatticeDirection', TranslationSymmetry],
    base_state: int,
    momenta: Mapping['LatticeDirection', int],
    normalize: bool = True,
) -> Dict[int, complex]:
    """
    Build a (possibly multi-dimensional) momentum superposition generated by
    the provided translations. ``translations`` and ``momenta`` must share the
    same keys mapping directions to symmetry operators and momentum indices.
    """
    if LatticeDirection is None:
        raise ImportError("LatticeDirection is not available in the current environment.")

    if set(translations.keys()) != set(momenta.keys()):
        raise ValueError("Translations and momenta must reference identical directions.")

    if not translations:
        return {int(base_state): 1.0 + 0j}

    directions: List['LatticeDirection'] = sorted(
        translations.keys(), key=lambda d: getattr(d, 'value', d)
    )
    representative = next(iter(translations.values()))
    lat = representative.lattice
    extent_lookup = {
        LatticeDirection.X: getattr(lat, 'Lx', None) or getattr(lat, 'lx', None) or 1,
        LatticeDirection.Y: getattr(lat, 'Ly', None) or getattr(lat, 'ly', None) or 1,
        LatticeDirection.Z: getattr(lat, 'Lz', None) or getattr(lat, 'lz', None) or 1,
    }

    extents: List[int] = []
    for direction in directions:
        extent = extent_lookup.get(direction)
        if extent is None or extent <= 0:
            raise ValueError(f"Cannot determine system size along direction {direction}.")
        extents.append(int(extent))

    amplitudes = defaultdict(complex)
    for step_tuple in product(*(range(extent) for extent in extents)):
        state = int(base_state)
        for direction, steps in zip(directions, step_tuple):
            translator = translations[direction]
            for _ in range(steps):
                state, _ = translator.apply(state)
        exponent = sum(
            momenta[direction] * step / extent
            for direction, step, extent in zip(directions, step_tuple, extents)
        )
        amplitudes[state] += np.exp(-1j * 2 * np.pi * exponent)

    if not amplitudes:
        raise ValueError("Momentum projector produced an empty superposition.")

    if normalize:
        norm = np.sqrt(sum(abs(val) ** 2 for val in amplitudes.values()))
        if norm <= 1e-12:
            raise ValueError("Momentum projector collapsed to zero vector for given k.")
        for basis_state in list(amplitudes.keys()):
            amplitudes[basis_state] /= norm

    return dict(amplitudes)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~