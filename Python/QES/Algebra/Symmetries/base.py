"""
Base classes and registry for symmetry operations in Hilbert space.

--------------------------------------------
File        : QES/Algebra/Symmetries/base.py
Description : Base classes and registry for symmetry operations in Hilbert space.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
--------------------------------------------
"""

from typing import Tuple, Any, Dict, Type, Optional, Set, FrozenSet
from enum import Enum, auto

# Import LocalSpaceTypes for validation
try:
    from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
except ImportError:
    # Fallback if not available
    class LocalSpaceTypes(Enum):
        SPIN_1_2            = "spin-1/2"
        SPIN_1              = "spin-1"
        SPINLESS_FERMIONS   = "spinless-fermions"
        ANYON_ABELIAN       = "abelian-anyons"
        BOSONS              = "bosons"

####################################################################################################
#! Symmetry compatibility enumerations
####################################################################################################

class SymmetryClass(Enum):
    """
    Classification of symmetry types for compatibility checking.
    
    Symmetry Catalog & Physical Applications
    =========================================
    
    SPATIAL SYMMETRIES (lattice structure):
    ---------------------------------------
    TRANSLATION : Discrete translation symmetry T_a
        - Applies to: All periodic lattice systems (1D/2D/3D)
        - Quantum numbers: Crystal momentum k (Bloch theorem)
        - Examples: Spin chains, Hubbard models, tight-binding models
        - Commutation: Forms Abelian group, momentum-dependent with REFLECTION
    
    REFLECTION : Spatial inversion/mirror symmetry sigma
        - Applies to: Systems with inversion symmetry
        - Quantum numbers: Reflection parity (±1)
        - Examples: Symmetric lattices, centrosymmetric crystals
        - Commutation: Only at k=0,pi with TRANSLATION; always with PARITY, U1_*, SU2_*
    
    POINT_GROUP : Point group symmetries (C_n, D_n, etc.)
        - Applies to: Finite clusters, molecules, quantum dots
        - Quantum numbers: Irrep labels of point group
        - Examples: Benzene (D_6h), quantum dots (C_4v)
        - Commutation: System-dependent
    
    INVERSION : Spatial inversion through origin P
        - Applies to: Centrosymmetric systems (different from REFLECTION)
        - Quantum numbers: Parity (±1)
        - Examples: Atomic physics, 3D crystals with inversion center
        - Commutation: Always with U1_*, SU2_*; k-dependent with TRANSLATION
    
    SPIN/INTERNAL SYMMETRIES (U(1) & discrete):
    --------------------------------------------
    PARITY : Spin-flip operators (sigma^x, sigma^y, sigma^z in global basis)
        - Applies to: Spin-1/2 systems, Ising-like models
        - Quantum numbers: Parity eigenvalue (±1, ±i)
        - Examples: XXZ model (Pz), transverse-field Ising (Px)
        - Commutation: Always with TRANSLATION, REFLECTION, U1_* (if compatible axes)
    
    U1_PARTICLE : U(1) particle number conservation N
        - Applies to: Systems with fixed particle number
        - Quantum numbers: N (total particles)
        - Examples: Hubbard model, Bose-Hubbard, fermion systems
        - Commutation: Always with TRANSLATION, REFLECTION, INVERSION
        - Constraint: Parity X/Y incompatible at non-half-filling
    
    U1_SPIN : U(1) spin conservation S^z_total
        - Applies to: Systems with conserved z-component of spin
        - Quantum numbers: M (total magnetization)
        - Examples: XXZ model, Heisenberg with field, anisotropic models
        - Commutation: Always with TRANSLATION, REFLECTION; blocks PARITY_X, PARITY_Y
        - Note: Only Parity Z compatible (preserves |M| -> flips all spins)
    
    DISCRETE SYMMETRIES (non-spatial):
    ----------------------------------
    TIME_REVERSAL : Anti-unitary time-reversal T
        - Applies to: Systems without magnetic field
        - Quantum numbers: Kramers degeneracy (complex conjugation)
        - Examples: Spin systems without Zeeman term
        - Commutation: Depends on Hamiltonian details
    
    CHARGE_CONJUGATION : Particle-hole symmetry C
        - Applies to: Systems at half-filling (fermions)
        - Quantum numbers: Particle-hole parity (±1)
        - Examples: Hubbard at half-fill, graphene at neutrality
        - Commutation: Requires specific filling
    
    FERMION_PARITY : (-1)^N_fermions
        - Applies to: Fermionic systems
        - Quantum numbers: Even/odd fermion number
        - Examples: Superconductors, Majorana modes
        - Commutation: Always with U1_PARTICLE (if N fixed)
    
    GENERIC : User-defined or composite symmetries
        - Applies to: Custom symmetries, product groups
        - Quantum numbers: Model-dependent
    """
    # Spatial symmetries
    TRANSLATION         = auto()
    REFLECTION          = auto()
    POINT_GROUP         = auto()
    INVERSION           = auto()
    
    # Spin/internal symmetries
    PARITY              = auto()        # Discrete spin flips (sigma^x, sigma^y, sigma^z)
    U1_PARTICLE         = auto()        # Particle number N
    U1_SPIN             = auto()        # Spin S^z
    
    # Discrete non-spatial
    TIME_REVERSAL       = auto()
    CHARGE_CONJUGATION  = auto()
    FERMION_PARITY      = auto()
    
    # Legacy/generic
    U1_GLOBAL           = auto()        # Alias for U1_PARTICLE (backward compat)
    ROTATION            = auto()        # Continuous rotations (rare in lattice)
    GENERIC             = auto()

class MomentumSector(Enum):
    """Momentum sector classification for translation-dependent compatibility."""
    ZERO                = 0
    PI                  = 1
    GENERIC             = 2

####################################################################################################
# Abstract base class for symmetry operations
####################################################################################################

class SymmetryOperator:
    """
    Abstract base class for a symmetry operation.
    
    Attributes
    ----------
    symmetry_class : SymmetryClass
        Classification of this symmetry type for compatibility checking.
    compatible_with : Set[SymmetryClass]
        Other symmetry classes that unconditionally commute with this one.
    momentum_dependent : Dict[MomentumSector, Set[SymmetryClass]]
        Additional compatible symmetries at specific momentum sectors.
    supported_local_spaces : Set[LocalSpaceTypes]
        Local Hilbert space types that support this symmetry.
        Empty set means universal (works for all types).
    """
    
    symmetry_class              : SymmetryClass                             = SymmetryClass.GENERIC
    compatible_with             : Set[SymmetryClass]                        = set()
    momentum_dependent          : Dict[MomentumSector, Set[SymmetryClass]]  = {}
    supported_local_spaces      : Set[LocalSpaceTypes]                      = set()  # Empty = universal
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    # ------------------------------------------------
    
    def apply(self, state: int) -> Tuple[int, complex]:
        """
        Apply the symmetry to a basis state (integer representation).
        Returns (new_state, phase).
        """
        raise NotImplementedError

    
    def commutes_with(
        self,
        other: 'SymmetryOperator',
        momentum_sector: Optional[MomentumSector] = None,
    ) -> bool:
        """
        Check if this symmetry commutes with another.
        
        Parameters
        ----------
        other : SymmetryOperator
            The other symmetry operator.
        momentum_sector : Optional[MomentumSector]
            Current momentum sector (if applicable).
            
        Returns
        -------
        bool
            True if the symmetries commute in the given context.
        """
        # Same class always commutes
        if self.symmetry_class == other.symmetry_class:
            return True
        
        # Check unconditional compatibility
        if other.symmetry_class in self.compatible_with:
            return True
        
        # Check momentum-dependent compatibility
        if momentum_sector is not None and momentum_sector in self.momentum_dependent:
            if other.symmetry_class in self.momentum_dependent[momentum_sector]:
                return True
        
        return False
    
    def is_valid_for_local_space(self, local_space_type: LocalSpaceTypes) -> bool:
        """
        Check if this symmetry is valid for the given local Hilbert space type.
        
        Parameters
        ----------
        local_space_type : LocalSpaceTypes
            The type of local Hilbert space (spin-1/2, fermions, etc.)
            
        Returns
        -------
        bool
            True if the symmetry is supported for this local space type.
            
        Notes
        -----
        - Empty supported_local_spaces set means universal (works for all types)
        - Translation/Reflection are universal (spatial symmetries)
        - Parity operators require spin systems
        - Fermion parity requires fermionic systems
        """
        # Empty set means universal - works for all local space types
        if not self.supported_local_spaces:
            return True
        
        return local_space_type in self.supported_local_spaces

####################################################################################################
# Registry for symmetry operator classes
####################################################################################################

class SymmetryRegistry:
    """
    Registry for symmetry operator classes, for extensibility.
    """
    _registry: Dict[str, Type[SymmetryOperator]] = {}

    @classmethod
    def register(cls, name: str, sym_cls: Type[SymmetryOperator]):
        cls._registry[name] = sym_cls

    @classmethod
    def get(cls, name: str) -> Type[SymmetryOperator]:
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~