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
    
    Quantum numbers: Reflection parity r = +/- 1
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
    
    @staticmethod
    def get_sectors() -> list:
        """
        Return the valid reflection parity sectors.
        
        Returns
        -------
        list
            Valid parity sectors: [+1, -1].
            - +1: Even parity (symmetric states)
            - -1: Odd parity (antisymmetric states)
            
        Examples
        --------
        >>> ReflectionSymmetry.get_sectors()
        [1, -1]
        """
        return [1, -1]
    
    def __init__(self, lattice: 'Lattice', sector: int, ns: int, base: int = 2):
        """
        Initialize reflection symmetry operator.
        
        Parameters
        ----------
        lattice : Lattice
            Lattice instance for geometry information
        sector : int
            Reflection quantum number (+1 or -1)
        ns : int
            Number of sites in the system
        base : int, optional
            Local Hilbert space dimension (default 2 for spin-1/2)
        """
        self.lattice            = lattice
        self.sector             = sector
        self.ns                 = ns
        self.base               = base

    def __call__(self, state: int, ns: int = None, **kwargs) -> Tuple[int, complex]:
        return self.apply_int(state, ns or self.ns, **kwargs)

    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """Apply reflection operator to integer state representation."""
        if rev is None:
            raise ImportError("rev() not available. Ensure QES.general_python.common.binary is installed.")
        new_state = rev(state, ns, backend='default')
        # Ensure result is within ns-bit space
        new_state = int(new_state) & ((1 << ns) - 1)
        return new_state, 1.0
    
    def apply_numpy(self, state, **kwargs):
        """Apply reflection operator to numpy array state representation."""
        import numpy as np
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        # For numpy arrays, delegate to integer operations
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return np.array(new_state), phase
    
    def apply_jax(self, state, **kwargs):
        """
        Apply reflection operator to JAX array state representation.
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX is required for apply_jax. Install with: pip install jax")
            
        # Vector encoding
        if state.ndim >= 1 and state.shape[-1] == self.ns:
            # Spatial reflection: reverse the site axis
            new_state   = jnp.flip(state, axis=-1)
            phase       = jnp.ones(state.shape[:-1], dtype=complex)
            return new_state, phase
            
        # Fallback
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return jnp.array(new_state), phase
    
    @property
    def directory_name(self) -> str:
        """
        Return a clean string suitable for directory names.
        
        Format: 'r_{p|m}' for reflection parity +1/-1
        
        Returns
        -------
        str
            Filesystem-safe string: 'r_p' for +1, 'r_m' for -1.
        """
        sector_str = self._sector_to_str(self.sector)
        return f"r_{sector_str}"
    
    def __repr__(self) -> str:
        return f"ReflectionSymmetry(sector={self.sector:+d}, ns={self.ns})"
    
    def __str__(self) -> str:
        parity_str = "even" if self.sector == 1 else "odd"
        return f"Reflection({parity_str})"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~