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

# ------------------------------------------

try:
    from QES.general_python.common.binary import flip_all, popcount
except ImportError:
    raise ImportError("QES.general_python.common.binary module is required for parity operations.")

# ------------------------------------------

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
    - Parity Y (sigma^y): Flips all spins in y-basis -> (+/- i)^n phase
    - Parity Z (sigma^z): Flips all spins in z-basis -> (-1)^n phase
    
    Examples:
    - XXZ model: Pz symmetry (conserves Sz but not Sx, Sy)
    - Transverse-field Ising: Px symmetry
    - XY model: May have Pz symmetry
    
    Commutation Rules
    -----------------
    Always commutes with:
    - TRANSLATION               : Abelian group [T, P] = 0 for discrete lattice translations
    - REFLECTION                : Spatial symmetry commutes with internal spin symmetry
    - PARITY                    : Different Pauli matrices commute
    - INVERSION                 : Spatial inversion independent of spin flip
    - U1_GLOBAL, U1_PARTICLE    : Backward compatibility alias
    
    Conditionally commutes with:
    - U1_SPIN                   : Only Pz commutes with Sz conservation (X, Y break it)
    - U1_PARTICLE               : Only if system is at half-filling N = Ns/2 (even Ns)
        -> Reason: Particle-hole transformation maps N <-> Ns - N
    """
    
    symmetry_class              = SymmetryClass.PARITY
    compatible_with             = {
                                    SymmetryClass.TRANSLATION,
                                    SymmetryClass.REFLECTION,
                                    SymmetryClass.U1_PARTICLE,
                                    SymmetryClass.U1_SPIN,
                                    SymmetryClass.PARITY,
                                    SymmetryClass.INVERSION,
                                    SymmetryClass.POINT_GROUP,
                                }
    supported_local_spaces      = {
                                    LocalSpaceTypes.SPIN_1_2,
                                    LocalSpaceTypes.SPIN_1,
                                    # NOT supported for fermions/bosons (different parity operators needed)
                                }
    
    # -------------------------
    
    def __init__(self, axis: str, sector: int, ns: int, nhl: int, lattice: 'Lattice' = None):
        """
        Initialize parity symmetry operator.
        
        Parameters
        ----------
        axis : str
            Parity axis ('x', 'y', or 'z')
        sector : int
            Parity quantum number (+1 or -1)
        ns : int
            Number of sites in the system
        nhl : int
            Hilbert space dimension (local_hspace_size^ns)
        lattice : Lattice, optional
            Lattice instance for compatibility checking
        """
        self.axis               = axis
        self.sector             = sector
        self.ns                 = ns
        self.nhl                = nhl
        self.lattice            = lattice

    def __call__(self, state: int, ns: int = None, **kwargs) -> Tuple[int, complex]:
        return self.apply_int(state, ns or self.ns, **kwargs)

    # -------------------------

    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """Apply parity operator to integer state representation."""
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
    
    def apply_numpy(self, state, **kwargs):
        """Apply parity operator to numpy array state representation."""
        import numpy as np
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        # For numpy arrays, delegate to integer operations
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return np.array(new_state), phase
    
    def apply_jax(self, state, **kwargs):
        """Apply parity operator to JAX array state representation."""
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX is required for apply_jax. Install with: pip install jax")
        # For jax arrays, delegate to integer operations
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return jnp.array(new_state), phase
    
    def is_compatible_with_global_symmetry(self, global_symmetry, **kwargs):
        """
        Check compatibility with U(1) particle number conservation.
        
        Particle-hole symmetry maps N -> Ns - N, so parity X/Y are only
        compatible at half-filling (N = Ns/2, even Ns).
        Parity Z acts trivially with U(1) so it's removed.
        """
        # Check if this is U(1) symmetry
        has_u1 = hasattr(global_symmetry, 'name') and 'U1' in str(global_symmetry.name)
        if not has_u1:
            return True, "Compatible"  # Not U(1, no restriction
        
        # Get U(1) sector (particle number) and ns from kwargs or instance
        u1_sector = getattr(global_symmetry, 'sector', None) or getattr(global_symmetry, 'val', None)
        ns = kwargs.get('ns', self.ns)
        if self.lattice:
            ns = ns or getattr(self.lattice, 'Ns', None) or getattr(self.lattice, 'ns', None)
        
        if self.axis in ['x', 'y']:
            # Parity X/Y require half-filling with U(1)
            if u1_sector is None or ns is None:
                return True, "Compatible (cannot determine filling)"
            if int(u1_sector) != ns // 2 or ns % 2 != 0:
                return False, f"Parity {self.axis.upper()} incompatible with U(1): requires half-filling N=Ns/2={ns//2} (even Ns), but N={int(u1_sector)}"
            return True, "Compatible at half-filling"
        elif self.axis == 'z':
            # Parity Z acts trivially with U(1) (preserves particle number)
            return False, "Parity Z acts trivially with U(1) (redundant)"
        
        return True, "Compatible"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~