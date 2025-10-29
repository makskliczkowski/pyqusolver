"""
This module provides translation symmetry operators for 1D, 2D, and 3D lattices.
It supports block-diagonalization in momentum sectors and is extensible for new symmetry types.

This module provides translation symmetry operators and utilities for Hilbert space construction.

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

# -------------------------------------------------

try:
    from QES.general_python.lattices.lattice import Lattice
    from QES.general_python.lattices.lattice import LatticeDirection
except ImportError:
    LatticeDirection = None  # type: ignore

# -------------------------------------------------

try:
    from QES.Algebra.Symmetries.base import SymmetryOperator, SymmetryClass, MomentumSector
except ImportError:
    raise ImportError("Could not import base symmetry classes from QES.Algebra.Symmetries.base")


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
    - k=0                       : Gamma point, fully symmetric sector
    - k=pi                      : Zone boundary (L even), alternating sign
    - Generic k                 : Complex phases, reduced symmetry
    
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
    supported_local_spaces  = set() # Empty = universal
    
    ####################################################################################################
    # INIT
    ####################################################################################################
    
    def __init__(self, 
                lattice             : 'Lattice', 
                sector              : int,
                ns                  : int,
                direction           : str           = 'x',
                local_hspace_size   : Optional[int] = 2
                ):
        '''
        Initialize the translation symmetry operator.
        Translates states along the specified lattice direction.
        
        Parameters
        ----------
        lattice : Lattice
            The lattice instance defining the system geometry.
            This is used to determine the periodic boundary conditions and the lattice structure.
        sector : int
            The momentum index n for the sector k = 2*pi*n/L
        ns : int
            Number of sites in the system
        direction : str, optional
            The lattice direction along which to translate ('x', 'y', or 'z', default is 'x').
        local_hspace_size : int, optional
            The size of the local Hilbert space at each site (default is 2 for spin-1/2). This means
            that each state, for the translation operation, is represented in base `local_hspace_size`.
            Example:
            a) for spin-1/2 systems, local_hspace_size=2 (binary representation):
                - state T^1 b'1010' -> b'0101' (shift right by 1)
            b) for spin-1 systems, local_hspace_size=3 (ternary representation):
                - state T^1 b'1020' -> b'2010' (shift right by 1 in base 3)
                - Here, we need to convert the integer state to base 3, perform the shift, and convert back.
                - This requires more complex logic than binary shifts. 
                TODO: Implement base-N translation logic if needed.
            c) for fermionic systems, we can represent the state as |fermions_down|fermions_up> in binary,
                - local_hspace_size=4 (4 states per site: empty, up, down, both)
                - Translation must handle the combined state representation.
                - This requires careful bit manipulation to ensure correct translation of both spin species.
                - state T^1 b'1100 1010' -> b'0110 0101' (shift both parts right by 1)
                - Here, we need to convert the integer state to binary, perform the shift, and convert back.
                TODO: Implement fermionic translation logic if needed.

        '''

        if lattice is None:
            raise ValueError("TranslationSymmetry requires a lattice instance.")
        
        # Map string direction to LatticeDirection enum
        direction_map = {'x': LatticeDirection.X, 'y': LatticeDirection.Y, 'z': LatticeDirection.Z}
        if isinstance(direction, str):
            direction = direction_map.get(direction.lower(), LatticeDirection.X)
        
        self.lattice                    = lattice
        self.dimension                  = lattice.dim
        self.direction                  = direction
        self.sector                     = sector
        self.momentum_index             = sector  # Alias for backward compatibility
        self.ns                         = ns
        self.local_hspace_size          = local_hspace_size
        self.perm, self.crossing_mask   = self._compute_translation_data()
    
    # -----------------------------------------------------
    #! Obtain momentum sector
    # -----------------------------------------------------
    
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
        # Example:
        # 1D chain L=4:
        #   momentum_index=0 -> k=0
        #   momentum_index=1 -> k=pi/2
        #   momentum_index=2 -> k=pi
        #   momentum_index=3 -> k=3pi/2
        # 2D Honeycomb Lx=3, Ly=3:
        #   momentum_index_x=0 -> kx=0
        #   momentum_index_x=1 -> kx=2pi/3
        #   momentum_index_x=2 -> kx=4pi/3
        #   momentum_index_y=0 -> ky=0
        #   momentum_index_y=1 -> ky=2pi/3
        #   momentum_index_y=2 -> ky=4pi/3
        # regardless of the nodes per unit cell (this is handled in the lattice definition)
        
        if self.momentum_index == 0:
            return MomentumSector.ZERO
        elif extent % 2 == 0 and self.momentum_index == extent // 2:
            return MomentumSector.PI
        else:
            return MomentumSector.GENERIC
    
    # -----------------------------------------------------
    #! Translation data computation
    # -----------------------------------------------------

    def _compute_translation_data(self):
        """
        Pre-compute permutation and crossing mask for translation in the given direction.
        Returns (perm, crossing_mask)
        -----------------------------------------
        perm : np.ndarray
            Array mapping each site to its translated site index.
        crossing_mask : np.ndarray
            Boolean array indicating which sites cross the boundary during translation.
        """
        lat         = self.lattice
        direction   = self.direction
        axis_map    = {
                        LatticeDirection.X: 0,
                        LatticeDirection.Y: 1,
                        LatticeDirection.Z: 2,
                    }
        # Determine the axis index for the given direction
        axis            = axis_map[direction]
        dims            = (lat.lx, lat.ly if lat.dim > 1 else 1, lat.lz if lat.dim > 2 else 1)
        size_axis       = dims[axis]
        ns              = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None) or getattr(lat, 'sites', None)
        perm            = np.empty(ns, dtype=np.int64)      # permutation array
        crossing_mask   = np.zeros(ns, dtype=bool)          # do we cross the boundary when translating this site?
        
        # Build the permutation and crossing mask
        for site in range(ns):
            coord       = list(lat.get_coordinates(site))
            while len(coord) < 3:
                coord.append(0)
                
            # Compute new coordinates after translation
            new_coord           = coord.copy()
            new_coord[axis]    += 1
            wrap                = False
            
            # Apply periodic boundary conditions
            if new_coord[axis] >= size_axis:
                new_coord[axis] -= size_axis
                wrap             = True
            
            # Get the destination site index
            dest                = lat.site_index(int(new_coord[0]), int(new_coord[1]), int(new_coord[2]))
            perm[site]          = dest
            
            # Mark crossing if we wrapped around
            if wrap:
                crossing_mask[site] = True

        return perm, crossing_mask

    # -----------------------------------------------------
    #! Translation application and helpers
    # -----------------------------------------------------

    def __call__(self, state: int, ns: int = None, **kwargs) -> Tuple[int, complex]:
        return self.apply_int(state, ns or self.ns, **kwargs)

    def __repr__(self) -> str:
        return f"Translation direction={self.direction} k_idx={self.momentum_index}"

    def __str__(self) -> str:
        return f'T({str(self.direction)};{self.momentum_index})'

    # -----------------------------------------------------
    #! Override checks
    # -----------------------------------------------------
    
    def check_boundary_conditions(self, lattice: Optional['Lattice'] = None, **kwargs) -> Tuple[bool, str]:
        """
        Check if lattice boundary conditions are compatible with translation symmetry.
        
        Translation symmetry requires periodic boundary conditions (PBC).
        
        Parameters
        ----------
        lattice : Optional[Lattice]
            Lattice structure with boundary conditions
        **kwargs : dict
            Additional context
        
        Returns
        -------
        valid : bool
            Whether boundary conditions are compatible
        reason : str
            Explanation if invalid, "Valid" otherwise
        """
        # Use self.lattice if no lattice provided
        lat = lattice or self.lattice
        
        if lat is not None:
            try:
                from QES.general_python.lattices.lattice import LatticeBC
                if hasattr(lat, 'bc') and lat.bc != LatticeBC.PBC:
                    return False, "Translation requires periodic boundary conditions"
            except ImportError:
                # If LatticeBC not available, assume it's okay
                pass
        
        return True, "Valid"
    
    # -----------------------------------------------------
    
    def commutes_with(self, other: 'SymmetryOperator', momentum_sector: Optional[MomentumSector] = None) -> bool:
        """
        Determine if this translation symmetry commutes with another symmetry operator.
        
        Parameters
        ----------
        other : SymmetryOperator
            The other symmetry operator to check compatibility with.
        momentum_sector : MomentumSector, optional
            The momentum sector to consider for momentum-dependent commutation.
        
        Returns
        -------
        bool
            True if the symmetries commute, False otherwise.
        """
        base_check = super().commutes_with(other, momentum_sector=momentum_sector)
        
        if not base_check:
            return False

        # Check momentum-dependent compatibility
        if momentum_sector is not None and momentum_sector in self.momentum_dependent:
            if other.symmetry_class in self.momentum_dependent[momentum_sector]:
                return True
        
        # Otherwise, they do not commute
        return False
    # -----------------------------------------------------

    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """
        Apply translation to integer state, return (new_state, phase)
        Parameters
        ----------
        state : int
            The integer representation of the basis state to be translated.
        ns : int
            Number of sites (from kwargs for API compatibility)
        Returns
        -------
        new_state : int
            The integer representation of the translated basis state.
        phase : complex
            The phase factor acquired due to fermionic sign or boundary conditions.
        """
        lat             = self.lattice
        perm            = self.perm
        crossing_mask   = self.crossing_mask
        new_state       = 0
        
        # Apply permutation to the state
        for src in range(ns):
            if (state >> (ns - 1 - src)) & 1:
                dest        = int(perm[src])
                new_state  |= 1 << (ns - 1 - dest)
                
        # Fermionic sign or boundary phase
        if crossing_mask.any():
            crossing_sites  = np.nonzero(crossing_mask)[0]
            occ_cross       = sum(1 for src in crossing_sites if (state >> (ns - 1 - src)) & 1)
        else:
            occ_cross       = 0

        # Compute boundary phase
        phase = lat.boundary_phase(self.direction, occ_cross)
        return new_state, phase
    
    def apply_numpy(self, state, **kwargs):
        """Apply translation operator to numpy array state representation."""
        import numpy as np
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        # For numpy arrays, delegate to integer operations
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return np.array(new_state), phase
    
    def apply_jax(self, state, **kwargs):
        """Apply translation operator to JAX array state representation."""
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX is required for apply_jax. Install with: pip install jax")
        # For jax arrays, delegate to integer operations
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return jnp.array(new_state), phase

    # -----------------------------------------------------
    #! Additional translation utilities
    # -----------------------------------------------------

    @staticmethod
    def momentum_phase(k: int, L: int) -> complex:
        # e^{i 2pi k / L}
        return np.exp(1j * 2 * np.pi * k / L)

    def momentum_projector(self, state: int, k: Optional[int] = None) -> Dict[int, complex]:
        """
        Return normalized amplitudes for the momentum-k projection of ``state``
        along this translation direction.
        
        Parameters
        ----------
        state : int
            The integer representation of the basis state to be projected.
        k : int, optional
            The momentum index n for the sector k = 2*pi*n/L. If None, uses self.momentum_index.
        Returns
        -------
        amplitudes : Dict[int, complex]
            Dictionary mapping basis states in the translation orbit to their
            corresponding amplitudes in the momentum-k superposition.
        """
        if k is None:
            if self.momentum_index is None:
                raise ValueError("Momentum index not provided for momentum projector.")
            k = self.momentum_index
            
        orbit   = self.orbit(state)     # How many unique states in the orbit
        m       = len(orbit)            # Number of states in the orbit
        
        if m == 0:
            return {int(state): 1.0 + 0j}
        
        # Compute coefficients for the momentum superposition
        coeffs  = np.exp(-1j * 2 * np.pi * k * np.arange(m) / m) / np.sqrt(m)
        return {int(orb_state): coeffs[idx] for idx, orb_state in enumerate(orbit)}

    def apply_superposition(self, amplitudes: Mapping[int, complex]) -> Dict[int, complex]:
        """
        Apply the translation to a linear combination expressed as {state: amp}.
        """
        rotated = defaultdict(complex)
        for basis_state, amplitude in amplitudes.items():
            new_state, phase         = self.apply(int(basis_state))
            rotated[int(new_state)] += amplitude * phase
        return dict(rotated)

    # -----------------------------------------------------

    def orbit(self, state: int, max_steps: Optional[int] = None) -> list[int]:
        """
        Generate the translation orbit of a basis state by repeatedly applying T
        until it cycles back. Returns the list of unique states in order.
        
        Parameters
        ----------
        state : int
            The integer representation of the basis state.
        max_steps : int, optional
            Maximum number of translation steps to perform (to avoid infinite loops).
        """
        seen    = {}
        seq     = []
        s       = int(state)
        steps   = 0
        while s not in seen:
            seen[s] = steps
            seq.append(s)
            s, _    = self.apply(s)
            steps  += 1
            if max_steps is not None and steps >= max_steps:
                break
        return seq

    # -----------------------------------------------------

    def orbit_length(self, state: int) -> int:
        """Number of unique states in the translation orbit of state."""
        return len(self.orbit(state))

    # -----------------------------------------------------

    def projector_coefficients(self, k: Optional[int], state: int) -> Tuple[list[int], np.ndarray]:
        """
        Coefficients for the momentum-k projector acting on the orbit of `state`.
        Returns (orbit_states, coeffs) where coeffs[n] = exp(-i 2pi k n / m) / sqrt(m),
        with m = len(orbit). Useful for building momentum eigenstates in ED.
        
        Parameters
        ----------
        k : int, optional
            The momentum index n for the sector k = 2*pi*n/L. If None, uses self.momentum_index.
        state : int
            The integer representation of the basis state.
        Returns
        -------
        orbit_states : list[int]
            List of basis states in the translation orbit.
        coeffs : np.ndarray
            Array of complex coefficients for the momentum-k projection.
        """
        
        if k is None:
            if self.momentum_index is None:
                raise ValueError("Momentum index not provided for coefficient generation.")
            k = self.momentum_index
        orb     = self.orbit(state)
        m       = len(orb)
        phases = np.exp(-1j * 2 * np.pi * k * np.arange(m) / m) / np.sqrt(m)
        return orb, phases
    
# -----------------------------------------------------

def build_momentum_superposition(
    translations        : Mapping['LatticeDirection', TranslationSymmetry],
    base_state          : int,
    momenta             : Mapping['LatticeDirection', int],
    normalize           : bool = True,
    local_hspace_size   : Optional[int] = None) -> Dict[int, complex]:
    """
    Build a (possibly multi-dimensional) momentum superposition generated by
    the provided translations. It allows to handle more than one translation symmetry
    simultaneously (e.g. in 2D/3D lattices).
    
    ``translations`` and ``momenta`` must share the
    same keys mapping directions to symmetry operators and momentum indices.
    
    Parameters
    ----------
    translations : Mapping[LatticeDirection, TranslationSymmetry]
        Mapping of lattice directions to their corresponding translation symmetry operators.
    base_state : int
        The integer representation of the basis state to be projected.
    momenta : Mapping[LatticeDirection, int]
        Mapping of lattice directions to their corresponding momentum indices.
    normalize : bool, optional
        Whether to normalize the resulting superposition (default is True).
    local_hspace_size : int, optional
        The size of the local Hilbert space at each site (default is None).
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
    representative      = next(iter(translations.values()))
    lat                 = representative.lattice
    extent_lookup       = {
        LatticeDirection.X: getattr(lat, 'Lx', None) or getattr(lat, 'lx', None) or 1,
        LatticeDirection.Y: getattr(lat, 'Ly', None) or getattr(lat, 'ly', None) or 1,
        LatticeDirection.Z: getattr(lat, 'Lz', None) or getattr(lat, 'lz', None) or 1,
    }

    extents: List[int]  = []
    for direction in directions:
        extent = extent_lookup.get(direction)
        if extent is None or extent <= 0:
            raise ValueError(f"Cannot determine system size along direction {direction}.")
        extents.append(int(extent))

    # Construct the momentum superposition
    amplitudes  = defaultdict(complex)
    for step_tuple in product(*(range(extent) for extent in extents)):
        # Apply the translation steps to the base state
        state   = int(base_state)
        
        for direction, steps in zip(directions, step_tuple):
            translator      = translations[direction]
            for _ in range(steps):
                state, _    = translator.apply(state)
                
        # Compute the phase factor for this combination of steps
        exponent = sum(
            momenta[direction] * step / extent
            for direction, step, extent in zip(directions, step_tuple, extents)
        )
        
        # Accumulate the amplitude exp(-i 2pi * exponent) - notice the negative sign in the exponent
        # this comes from the definition of momentum projector and we return to the original state
        amplitudes[state] += np.exp(-1j * 2 * np.pi * exponent)

    # Validate non-empty superposition
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
#! EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~