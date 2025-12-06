"""
Spatial inversion symmetry for general lattice systems (1D, 2D, 3D).

This module provides the InversionSymmetry operator that maps each site to its 
spatially inverted position through the lattice center. Unlike reflection (bit-reversal),
this works correctly for any lattice type and dimension.

--------------------------------------------------
File        : QES/Algebra/Symmetries/inversion.py
Description : Spatial inversion symmetry operators for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-12-06
--------------------------------------------------
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from functools import lru_cache

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

try:
    from QES.general_python.lattices.lattice import Lattice
except ImportError:
    Lattice = None  # type: ignore
    
from QES.Algebra.Symmetries.base import SymmetryOperator, SymmetryClass, MomentumSector

####################################################################################################
# Numba-accelerated permutation application
####################################################################################################

@njit(cache=True)
def _apply_inversion_permutation(state: int, ns: int, base: int, perm: np.ndarray) -> int:
    """
    Apply site permutation to a quantum state in integer representation.
    
    For base=2 (spin-1/2), the state is a bitstring where bit i represents site i.
    For general base, we use mixed-radix representation.
    
    Parameters
    ----------
    state : int
        Integer representation of the quantum state
    ns : int
        Number of sites
    base : int
        Local Hilbert space dimension (2 for spin-1/2)
    perm : np.ndarray
        Permutation array: perm[i] = new position of site i
        
    Returns
    -------
    int
        New state after applying the permutation
    """
    if base == 2:
        # Fast path for spin-1/2: bit manipulation
        new_state = 0
        for i in range(ns):
            # Extract bit at position i
            bit = (state >> i) & 1
            # Place it at the inverted position
            new_state |= (bit << perm[i])
        return new_state
    else:
        # General case: mixed-radix representation
        # Extract local states
        local_states = np.zeros(ns, dtype=np.int64)
        temp = state
        for i in range(ns):
            local_states[i] = temp % base
            temp //= base
        
        # Reconstruct with permuted positions
        new_state = 0
        power = 1
        for i in range(ns):
            # Site i in new state gets value from site that maps to i
            # We need inverse permutation: find j such that perm[j] = i
            for j in range(ns):
                if perm[j] == i:
                    new_state += local_states[j] * power
                    break
            power *= base
        return new_state


def _build_inversion_permutation(lattice: 'Lattice') -> np.ndarray:
    """
    Build the site permutation array for spatial inversion.
    
    For a lattice with coordinates (nx, ny, nz), inversion maps:
        (nx, ny, nz) -> (Lx-1-nx, Ly-1-ny, Lz-1-nz)
    
    This works for any lattice type (square, honeycomb, triangular, etc.)
    as long as the lattice has coordinate information.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice instance with coordinate and indexing information
        
    Returns
    -------
    np.ndarray
        Permutation array of shape (ns,) where perm[i] = inverted site index
    """
    ns = lattice.Ns
    dim = lattice.dim
    lx = lattice.Lx
    ly = lattice.Ly if dim >= 2 else 1
    lz = lattice.Lz if dim >= 3 else 1
    
    # Check if lattice has multi-site unit cell (e.g., honeycomb)
    n_basis = len(lattice._basis) if hasattr(lattice, '_basis') and lattice._basis is not None and len(lattice._basis) > 0 else 1
    
    perm = np.zeros(ns, dtype=np.int64)
    
    for i in range(ns):
        # Get the unit cell index and sublattice index
        cell = i // n_basis
        sub = i % n_basis
        
        # Extract cell coordinates
        nx = cell % lx
        ny = (cell // lx) % ly if dim >= 2 else 0
        nz = (cell // (lx * ly)) % lz if dim >= 3 else 0
        
        # Apply inversion: (nx, ny, nz) -> (Lx-1-nx, Ly-1-ny, Lz-1-nz)
        nx_inv = (lx - 1 - nx) % lx
        ny_inv = (ly - 1 - ny) % ly if dim >= 2 else 0
        nz_inv = (lz - 1 - nz) % lz if dim >= 3 else 0
        
        # Convert back to linear index
        cell_inv = nx_inv + ny_inv * lx + nz_inv * lx * ly
        
        # For multi-site unit cells, sublattice index is preserved under inversion
        # (this is the standard convention for honeycomb, kagome, etc.)
        # Some lattices may need sublattice permutation too - can be extended
        i_inv = cell_inv * n_basis + sub
        
        perm[i] = i_inv
    
    return perm


####################################################################################################
# InversionSymmetry class
####################################################################################################

class InversionSymmetry(SymmetryOperator):
    """
    Spatial inversion symmetry for general lattice systems.
    
    Physical Context
    ----------------
    Spatial inversion (parity) symmetry P maps each site to its mirror position
    through the center of the lattice:
    
    - 1D: site i → site (L-1-i)
    - 2D: site at (x,y) → site at (Lx-1-x, Ly-1-y)
    - 3D: site at (x,y,z) → site at (Lx-1-x, Ly-1-y, Lz-1-z)
    
    Quantum numbers: Inversion parity p = +/- 1
    - p = +1: Even parity (symmetric states)
    - p = -1: Odd parity (antisymmetric states)
    
    Difference from ReflectionSymmetry
    -----------------------------------
    ReflectionSymmetry uses bit-reversal which is only correct for 1D chains.
    InversionSymmetry uses explicit lattice coordinates, making it correct
    for any lattice type and dimension.
    
    Lattice Compatibility
    ---------------------
    Works with any lattice that has:
    - Defined dimensions (Lx, Ly, Lz)
    - Optional multi-site unit cells (honeycomb, kagome, etc.)
    
    Commutation Rules
    -----------------
    Always commutes with:
    - U1_PARTICLE, U1_SPIN: Conserved quantities are spatial-independent
    - PARITY: Spatial inversion commutes with spin flips
    - REFLECTION: Both are spatial operations
    - POINT_GROUP: Part of point group symmetries
    
    Conditionally commutes with (momentum-dependent):
    - TRANSLATION: Only at k=0, k=π momentum sectors
      -> Reason: P * T * P^(-1) = T^(-1), so need e^(ik) = e^(-ik)
      -> This holds only for k=0 (trivial) or k=π
    
    Examples
    --------
    >>> from QES.Algebra.Symmetries import InversionSymmetry
    >>> from QES.general_python.lattices.square import SquareLattice
    >>> 
    >>> # 2D square lattice with inversion symmetry
    >>> lattice = SquareLattice(dim=2, lx=4, ly=4)
    >>> inv_sym = InversionSymmetry(lattice, sector=1, ns=16, base=2)
    >>> 
    >>> # Apply to a state
    >>> new_state, phase = inv_sym.apply_int(0b1010, ns=16)
    """
    
    symmetry_class          = SymmetryClass.INVERSION
    compatible_with         = {
                                SymmetryClass.U1_PARTICLE,
                                SymmetryClass.U1_SPIN,
                                SymmetryClass.PARITY,
                                SymmetryClass.REFLECTION,
                                SymmetryClass.POINT_GROUP,
                            }
    momentum_dependent      = {
                                MomentumSector.ZERO : {SymmetryClass.TRANSLATION},
                                MomentumSector.PI   : {SymmetryClass.TRANSLATION},
                            }
    
    # Inversion is universal - works for all local space types
    supported_local_spaces  = set()  # Empty = universal
    
    # Class-level cache for permutation arrays (keyed by lattice identity)
    _permutation_cache: dict = {}
    
    @staticmethod
    def get_sectors() -> List[int]:
        """
        Return the valid inversion parity sectors.
        
        Returns
        -------
        list
            Valid parity sectors: [+1, -1].
            - +1: Even parity (symmetric states)
            - -1: Odd parity (antisymmetric states)
            
        Examples
        --------
        >>> InversionSymmetry.get_sectors()
        [1, -1]
        """
        return [1, -1]
    
    def __init__(self, 
                 lattice: 'Lattice', 
                 sector: int, 
                 ns: int, 
                 base: int = 2,
                 precompute: bool = True):
        """
        Initialize inversion symmetry operator.
        
        Parameters
        ----------
        lattice : Lattice
            Lattice instance for geometry information
        sector : int
            Inversion quantum number (+1 or -1)
        ns : int
            Number of sites in the system
        base : int, optional
            Local Hilbert space dimension (default 2 for spin-1/2)
        precompute : bool, optional
            If True, precompute the permutation array at construction.
            Set to False for lazy evaluation.
        """
        if sector not in (1, -1):
            raise ValueError(f"Inversion sector must be +1 or -1, got {sector}")
        
        self.lattice    = lattice
        self.sector     = sector
        self.ns         = ns
        self.base       = base
        self._perm      = None
        
        if precompute and lattice is not None:
            self._perm = self._get_permutation()
    
    def _get_permutation(self) -> np.ndarray:
        """
        Get or compute the inversion permutation array.
        
        Uses caching to avoid recomputation for the same lattice.
        """
        if self._perm is not None:
            return self._perm
        
        # Use lattice id for caching
        cache_key = id(self.lattice)
        
        if cache_key not in InversionSymmetry._permutation_cache:
            InversionSymmetry._permutation_cache[cache_key] = _build_inversion_permutation(self.lattice)
        
        self._perm = InversionSymmetry._permutation_cache[cache_key]
        return self._perm
    
    def __call__(self, state: int, ns: int = None, **kwargs) -> Tuple[int, complex]:
        return self.apply_int(state, ns or self.ns, **kwargs)
    
    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """
        Apply inversion operator to integer state representation.
        
        Parameters
        ----------
        state : int
            Integer representation of the quantum state
        ns : int
            Number of sites
            
        Returns
        -------
        Tuple[int, complex]
            (new_state, phase) where phase is always 1.0 for the operator itself.
            The sector eigenvalue is used during symmetry sector construction.
        """
        perm = self._get_permutation()
        new_state = _apply_inversion_permutation(state, ns, self.base, perm)
        return new_state, 1.0
    
    def apply_numpy(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, complex]:
        """
        Apply inversion operator to numpy array state representation.
        
        For a state vector in the computational basis, this permutes the
        amplitudes according to the inversion transformation.
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # If it's an integer encoded as array, use apply_int
        if state.ndim == 0 or (state.ndim == 1 and state.size == 1):
            new_state, phase = self.apply_int(int(state.flat[0]), self.ns, **kwargs)
            return np.array(new_state), phase
        
        # If it's a full state vector, permute the basis states
        perm = self._get_permutation()
        nh = len(state)
        
        # Build the permutation of basis indices
        # For each computational basis state, compute its image under inversion
        permuted_state = np.zeros_like(state)
        
        for idx in range(nh):
            new_idx = _apply_inversion_permutation(idx, self.ns, self.base, perm)
            if new_idx < nh:
                permuted_state[new_idx] = state[idx]
        
        return permuted_state, 1.0
    
    def apply_jax(self, state, **kwargs):
        """Apply inversion operator using JAX."""
        try:
            from QES.general_python.algebra.utils import JAX_AVAILABLE
            if JAX_AVAILABLE:
                import jax.numpy as jnp
            else:
                raise ImportError("JAX not available")
        except ImportError:
            raise ImportError("JAX not available for apply_jax()")
        
        # Convert to numpy, apply, convert back
        np_state = np.array(state)
        result, phase = self.apply_numpy(np_state, **kwargs)
        return jnp.array(result), phase
    
    def get_character(self, count: int, sector: Union[int, float, complex], **kwargs) -> complex:
        """
        Compute the character for inversion raised to a power.
        
        For inversion: chi(P^n) = sector^n
        Since P^2 = 1, this is periodic with period 2.
        
        Parameters
        ----------
        count : int
            Power of the inversion operator
        sector : int
            Inversion parity (+1 or -1)
            
        Returns
        -------
        complex
            Character value
        """
        # P^2 = I, so only need count mod 2
        effective_count = count % 2
        return sector ** effective_count
    
    @staticmethod
    def clear_cache():
        """Clear the permutation cache (useful for memory management)."""
        InversionSymmetry._permutation_cache.clear()
    
    def __repr__(self) -> str:
        return f"InversionSymmetry(sector={self.sector:+d}, ns={self.ns}, base={self.base})"
    
    def __str__(self) -> str:
        parity_str = "even" if self.sector == 1 else "odd"
        return f"Inversion({parity_str})"
    
    @property
    def directory_name(self) -> str:
        """
        Return a clean string suitable for directory names.
        
        Format: 'inv_{p|m}' for inversion parity +1/-1
        
        Returns
        -------
        str
            Filesystem-safe string: 'inv_p' for +1, 'inv_m' for -1.
        """
        sector_str = self._sector_to_str(self.sector)
        return f"inv_{sector_str}"


####################################################################################################
# Utility functions
####################################################################################################

def build_inversion_operator_matrix(lattice: 'Lattice', base: int = 2) -> np.ndarray:
    """
    Build the full inversion operator matrix in the computational basis.
    
    This is useful for verification and small systems.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice instance
    base : int
        Local Hilbert space dimension
        
    Returns
    -------
    np.ndarray
        Inversion operator matrix of shape (nh, nh) where nh = base^ns
    """
    ns = lattice.Ns
    nh = base ** ns
    perm = _build_inversion_permutation(lattice)
    
    P = np.zeros((nh, nh), dtype=complex)
    
    for idx in range(nh):
        new_idx = _apply_inversion_permutation(idx, ns, base, perm)
        P[new_idx, idx] = 1.0
    
    return P
