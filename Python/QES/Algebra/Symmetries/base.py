"""
Base classes and registry for symmetry operations in Hilbert space.
This module defines the abstract base class `SymmetryOperator` that all
symmetry operations must inherit from, along with enumerations for symmetry
classification and momentum sectors. It also includes a registry for symmetry
operator classes to facilitate extensibility.

This registry allows new symmetry types to be added dynamically and retrieved
by name, enabling flexible integration of custom symmetries into the framework.

--------------------------------------------
File        : QES/Algebra/Symmetries/base.py
Description : Base classes and registry for symmetry operations in Hilbert space.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
--------------------------------------------
"""

import  numpy       as np
import  numba
from    typing      import Tuple, Any, Dict, Type, Optional, Set, FrozenSet, Union
from    enum        import Enum, auto
from    dataclasses import dataclass

###################################################################################################

try:
    import          jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    jnp             = np
    JAX_AVAILABLE   = False
    
try:
    from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
except ImportError:
    raise ImportError("LocalSpaceTypes could not be imported from QES.Algebra.Hilbert.hilbert_local. Make sure general_python is installed correctly.")

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
        - Quantum numbers   : Crystal momentum k (Bloch theorem)
        - Examples          : Spin chains, Hubbard models, tight-binding models
        - Commutation       : Forms Abelian group, momentum-dependent with REFLECTION
    
    REFLECTION : Spatial inversion/mirror symmetry sigma
        - Applies to        : Systems with inversion symmetry
        - Quantum numbers   : Reflection parity (+/- 1)
        - Examples          : Symmetric lattices, centrosymmetric crystals
        - Commutation       : Only at k=0,pi with TRANSLATION; always with PARITY, U1_*, SU2_*
    
    POINT_GROUP : Point group symmetries (C_n, D_n, etc.)
        - Applies to        : Finite clusters, molecules, quantum dots
        - Quantum numbers   : Irrep labels of point group
        - Examples          : Benzene (D_6h), quantum dots (C_4v)
        - Commutation       : System-dependent

    INVERSION : Spatial inversion through origin P
        - Applies to        : Centrosymmetric systems (different from REFLECTION)
        - Quantum numbers   : Parity (+/- 1)
        - Examples          : Atomic physics, 3D crystals with inversion center
        - Commutation       : Always with U1_*, SU2_*; k-dependent with TRANSLATION
    
    SPIN/INTERNAL SYMMETRIES (U(1) & discrete):
    --------------------------------------------
    PARITY : Spin-flip operators (sigma^x, sigma^y, sigma^z in global basis)
        - Applies to        : Spin-1/2 systems, Ising-like models
        - Quantum numbers   : Parity eigenvalue (+/- 1, +/- i)
        - Examples          : XXZ model (Pz), transverse-field Ising (Px)
        - Commutation       : Always with TRANSLATION, REFLECTION, U1_* (if compatible axes)

    U1_PARTICLE : U(1) particle number conservation N
        - Applies to        : Systems with fixed particle number
        - Quantum numbers   : N (total particles)
        - Examples          : Hubbard model, Bose-Hubbard, fermion systems
        - Commutation       : Always with TRANSLATION, REFLECTION, INVERSION
        - Constraint        : Parity X/Y incompatible at non-half-filling

    U1_SPIN : U(1) spin conservation S^z_total
        - Applies to        : Systems with conserved z-component of spin
        - Quantum numbers   : M (total magnetization)
        - Examples          : XXZ model, Heisenberg with field, anisotropic models
        - Commutation       : Always with TRANSLATION, REFLECTION; blocks PARITY_X, PARITY_Y
        - Note              : Only Parity Z compatible (preserves |M| -> flips all spins)
    
    DISCRETE SYMMETRIES (non-spatial):
    :TODO Implement full descriptions and operational details
    ----------------------------------
    TIME_REVERSAL : Anti-unitary time-reversal T
        - Applies to        : Systems without magnetic field
        - Quantum numbers   : Kramers degeneracy (complex conjugation)
        - Examples          : Spin systems without Zeeman term
        - Commutation       : Depends on Hamiltonian details

    CHARGE_CONJUGATION : Particle-hole symmetry C
        - Applies to        : Systems at half-filling (fermions)
        - Quantum numbers   : Particle-hole parity (+/- 1)
        - Examples          : Hubbard at half-fill, graphene at neutrality
        - Commutation       : Requires specific filling
    
    FERMION_PARITY : (-1)^N_fermions
        - Applies to        : Fermionic systems
        - Quantum numbers   : Even/odd fermion number
        - Examples          : Superconductors, Majorana modes
        - Commutation       : Always with U1_PARTICLE (if N fixed)

    GENERIC : User-defined or composite symmetries
        - Applies to        : Custom symmetries, product groups
        - Quantum numbers   : Model-dependent
    """
    
    # Spatial symmetries
    TRANSLATION         = auto()        # Lattice translations          (IMPLEMENTED)
    REFLECTION          = auto()        # Spatial reflections           (IMPLEMENTED)
    POINT_GROUP         = auto()        # Point group symmetries        (NOT IMPLEMENTED)
    INVERSION           = auto()        # Spatial inversion through P   (IMPLEMENTED)
    
    # Spin/internal symmetries
    PARITY              = auto()        # Spin flips (s^x, s^y, s^z)    (IMPLEMENTED)
    U1_PARTICLE         = auto()        # Particle number N             (IMPLEMENTED)
    U1_SPIN             = auto()        # Spin S^z                      (IMPLEMENTED)
    
    # Discrete non-spatial
    TIME_REVERSAL       = auto()        # Anti-unitary time-reversal    (NOT IMPLEMENTED)
    CHARGE_CONJUGATION  = auto()        # Particle-hole symmetry        (NOT IMPLEMENTED)
    FERMION_PARITY      = auto()
    
    # Generic
    ROTATION            = auto()        # Continuous rotations          (NOT IMPLEMENTED)
    GENERIC             = auto()        # User-defined/composite symmetries (IMPLEMENTED)

class MomentumSector(Enum):
    """Momentum sector classification for translation-dependent compatibility."""
    ZERO                = 0
    PI                  = 1
    GENERIC             = 2

####################################################################################################
#! For JITTING symmetry application methods
####################################################################################################

OP_IDENTITY             = np.uint8(0)
OP_TRANSLATION          = np.uint8(1)
OP_REFLECTION           = np.uint8(2)
OP_PARITY               = np.uint8(3)
OP_INVERSION            = np.uint8(4)

@dataclass(frozen=True)
class SymmetryApplicationCodes:
    """ Codes for symmetry application methods. """
    OP_IDENTITY     = OP_IDENTITY
    OP_TRANSLATION  = OP_TRANSLATION
    OP_REFLECTION   = OP_REFLECTION
    OP_PARITY       = OP_PARITY
    OP_INVERSION    = OP_INVERSION
    
@dataclass(frozen=True)
class CompiledGroup:
    n_group             : int
    n_base              : int
    n_trans             : int
    
    # element g consists of n_ops[g] primitive ops              -> we call them in sequence
    n_ops               : np.ndarray          # (n_group,) int32

    # op_code[g, j] tells which primitive op to call            -> see SymmetryApplicationCodes
    op_code             : np.ndarray          # (n_group, max_ops) uint8

    # arguments are packed into two int64s per op call          -> depends on symmetry
    arg0                : np.ndarray          # (n_group, max_ops) int64
    arg1                : np.ndarray          # (n_group, max_ops) int64

    # character of each group element in *your* chosen irrep/sector
    chi                 : np.ndarray          # (n_group,) complex128

    # global ops
    global_op_codes     : np.ndarray          # (n_global,)     -> instructs how to apply the global ops
    global_op_vals      : np.ndarray          # (n_global,)     -> values for global symmetries
    
    @property
    def args(self) -> Tuple:
        return (
            self.n_group,
            self.n_ops,
            self.op_code,
            self.arg0,
            self.arg1,
            self.chi,
        )
    
@dataclass(frozen=True)
class SymOpTables:
    ''' 
    Precomputed tables for symmetry operations. 
    
    Each symmetry operation has its own set of tables depending on its nature.
    These tables are used to efficiently apply the symmetry to states.
    
    Examples
    --------
    - Translation tables:
        trans_perm          : Permutation of sites under translation - maps site i to site trans_perm[i]
        trans_cross_mask    : Mask for fermionic sign changes due to crossing
        trans_shift         : Shift amounts for translation powers
    - Reflection tables:
        refl_perm           : Permutation of sites under reflection - maps site i to site refl_perm[i]
    '''
    
    # Translation tables (support multiple translation generators: x/y/z etc.)
    # We store a stack of permutations; op arg0 will select which one.
    trans_perm          : np.ndarray        # (n_trans, ns) int64
    trans_cross_mask    : np.ndarray        # (n_trans, ns) uint8 or bool
    trans_shift         : np.ndarray        # (n_trans,) int32

    # Reflection tables
    refl_perm           : np.ndarray        # (n_refl, ns) int64
    
    # Inversion tables
    inv_perm            : np.ndarray        # (n_inv, ns) int64

    # Parity tables
    parity_axis         : np.ndarray        # (n_parity,) uint8   (0=x,1=y,2=z)

    # Save boundary phases
    boundary_phase      : np.ndarray        # (3, ns+1) complex128, for translation/reflection BC phases
    
    @property
    def args(self) -> Tuple:
        return (
            self.trans_perm, self.trans_cross_mask,
            self.refl_perm, 
            self.inv_perm,
            self.parity_axis,
            self.boundary_phase
        )

@numba.njit(cache=True, fastmath=True)
def _popcount64(x: np.int64) -> np.int64:
    # portable popcount (Numba supports bit ops; this is fine)
    c = np.int64(0)
    while x:
        x &= (x - 1)
        c += 1
    return c

####################################################################################################
# Abstract base class for symmetry operations
####################################################################################################

class SymmetryOperator:
    """
    Abstract base class for a symmetry operation.
    
    This class defines the interface that all symmetry operators must implement.
    Like the Operator class, symmetries support multiple backends (integer, numpy, jax)
    for maximum flexibility and performance.
    
    Backend Support
    ---------------
    Each symmetry must implement three application methods:
    - apply_int   : For integer state representation (fastest, most common)
    - apply_numpy : For numpy vector representation
    - apply_jax   : For JAX vector representation (if JAX available)
    
    This mirrors the Operator class structure and allows symmetries to work
    with different state representations seamlessly.
    
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
    sector : Union[int, float, complex]
        Quantum number (sector value) for this symmetry.
    
    Examples
    --------
    Implementing a new symmetry:
    
    >>> class MySymmetry(SymmetryOperator):
    ...     symmetry_class          = SymmetryClass.GENERIC
    ...     compatible_with         = {SymmetryClass.TRANSLATION, SymmetryClass.U1_PARTICLE}
    ...     
    ...     def __init__(self, sector, **kwargs):
    ...         self.sector         = sector
    ...     
    ...     def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
    ...         # Transform integer state
    ...         new_state   = ...   # your transformation
    ...         phase       = ...   # symmetry eigenvalue
    ...         return new_state, phase
    ...     
    ...     def apply_numpy(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, complex]:
    ...         # Transform numpy vector
    ...         new_state   = ...
    ...         phase       = ...
    ...         return new_state, phase
    ...     
    ...     def apply_jax(self, state: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, complex]:
    ...         # Transform JAX vector
    ...         if not JAX_AVAILABLE:
    ...             raise ImportError("JAX not available")
    ...         new_state   = ...
    ...         phase       = ...
    ...         return new_state, phase
    """
    
    code                        : SymmetryApplicationCodes                  = SymmetryApplicationCodes.OP_IDENTITY
    symmetry_class              : SymmetryClass                             = SymmetryClass.GENERIC     # Default to GENERIC, override in subclasses
    compatible_with             : Set[SymmetryClass]                        = set()                     # Unconditionally compatible symmetries, e.g., PARITY with U1_*
    momentum_dependent          : Dict[MomentumSector, Set[SymmetryClass]]  = {}                        # Compatibility at specific momentum sectors, e.g., TRANSLATION-REFLECTION
    supported_local_spaces      : Set[LocalSpaceTypes]                      = set()                     # Empty = universal, e.g., spin-1/2, fermions, bosons, etc.
    sector                      : Optional[Union[int, float, complex]]      = None                      # Quantum number for this symmetry
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def directory_name(self) -> str:
        """
        Return a clean string suitable for directory names.
        
        This method returns a string without special characters, spaces, or 
        characters that could cause filesystem issues. Override in subclasses
        for symmetry-specific naming.
        
        Returns
        -------
        str
            A filesystem-safe string like 'kx_0', 'pz_p', 'inv_m', etc.
            
        Examples
        --------
        >>> sym = TranslationSymmetry(lattice, sector=0, direction='x')
        >>> sym.directory_name
        'kx_0'
        
        >>> sym = ParitySymmetry(axis='z', sector=-1)
        >>> sym.directory_name
        'pz_m'
        """
        # Default implementation - subclasses should override
        sector_str = self._sector_to_str(self.sector)
        return f"{self.__class__.__name__.lower()}_{sector_str}"
    
    @staticmethod
    def _sector_to_str(sector) -> str:
        """
        Convert a sector value to a filesystem-safe string.
        
        Parameters
        ----------
        sector : int, float, or complex
            The sector value
            
        Returns
        -------
        str
            'p' for +1, 'm' for -1, or the numeric value as string
        """
        if sector is None:
            return "none"
        if sector == 1:
            return "p"
        elif sector == -1:
            return "m"
        elif isinstance(sector, (int, np.integer)):
            return str(int(sector))
        elif isinstance(sector, float):
            # Handle floats: replace decimal point with 'd'
            return str(sector).replace('.', 'd').replace('-', 'm')
        elif isinstance(sector, complex):
            # Handle complex: format as 're_im'
            return f"{sector.real:.2f}_{sector.imag:.2f}".replace('.', 'd').replace('-', 'm')
        else:
            return str(sector).replace(' ', '').replace('(', '').replace(')', '')
    
    # ------------------------------------------------
    # Core application methods (must be implemented)
    # ------------------------------------------------
    
    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """
        Apply the symmetry to a state in integer representation.
        
        Parameters
        ----------
        state : int
            Integer representation of quantum state (e.g., bitstring)
        ns : int
            Number of sites
        **kwargs : dict
            Additional parameters (e.g., nhl, lattice, etc.)
        
        Returns
        -------
        new_state : int
            Transformed state
        phase : complex
            Symmetry eigenvalue/character (typically a phase factor)
        
        Notes
        -----
        This is the most performance-critical method and should be JIT-compiled
        (numba @njit) when possible. It's used for finding representatives and
        building mappings.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_int()")
    
    def apply_numpy(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, complex]:
        """
        Apply the symmetry to a state in numpy vector representation.
        
        Parameters
        ----------
        state : np.ndarray
            State vector (shape: (ns,) or (nh,))
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        new_state : np.ndarray
            Transformed state vector
        phase : complex
            Symmetry eigenvalue
        
        Notes
        -----
        This method is used when working with explicit state vectors rather
        than integer representations. Useful for small systems or when the
        full state vector is needed.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_numpy()")
    
    def apply_jax(self, state: 'jnp.ndarray', **kwargs) -> Tuple['jnp.ndarray', complex]:
        """
        Apply the symmetry to a state in JAX vector representation.
        
        Parameters
        ----------
        state : jnp.ndarray
            JAX state vector
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        new_state : jnp.ndarray
            Transformed state vector
        phase : complex
            Symmetry eigenvalue
        
        Notes
        -----
        This method enables GPU acceleration and automatic differentiation.
        Should be decorated with @jax.jit when possible.
        
        If JAX is not available, this can raise ImportError or fall back
        to numpy implementation.
        """
        if not JAX_AVAILABLE:
            raise ImportError(f"JAX not available - cannot use apply_jax() in {self.__class__.__name__}")
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_jax()")
        
    # ------------------------------------------------
    # Character computation
    # ------------------------------------------------
    
    def get_character(self, count: int, sector: Union[int, float, complex], **kwargs) -> complex:
        r"""
        Compute the character (representation eigenvalue) for this symmetry raised to a power.
        
        This is a key quantum number that determines the eigenvalue of the symmetry operator
        in a given representation (momentum sector, parity sector, etc.). Mathetmatically,
        the character chi_sector(op^n) gives the eigenvalue when applying the symmetry
        operation 'op' 'n' times in the specified sector:
        $$
        chi_sector(op^n) = eigenvalue of op^n in sector
        $$
        
        Parameters
        ----------
        count : int
            How many times this symmetry operation is applied (power)
        sector : Union[int, float, complex]
            The quantum number/sector for this representation
        **kwargs : dict
            Additional context (lattice, ns, etc.)
        
        Returns
        -------
        character : complex
            Character value chi_sector(op^count)
        
        Notes
        -----
        Default implementation for discrete symmetries:
            chi(g^n) = sector^n
        
        For translation and other continuous symmetries, override this method.
        
        Examples
        --------
        - Parity with sector=+1:
            chi(P^2) = (+1)^2 = 1
        - Parity with sector=-1:
            chi(P^2) = (-1)^2 = 1
        - Translation:
            See TranslationSymmetry.get_character() for exp(ikn) formula
        """
        # Default: discrete symmetry with character = sector^count
        return sector ** count
        
    def get_site_permutation(self, ns: int) -> Optional[np.ndarray]:
        """
        Get the site permutation array for this symmetry operation.
        
        Returns an array `perm` such that site `i` maps to site `perm[i]`.
        If the symmetry is not a spatial permutation (e.g., spin flip), returns None.
        
        Parameters
        ----------
        ns : int
            Number of sites.
            
        Returns
        -------
        perm : np.ndarray or None
            Permutation array of shape (ns,) or None.
        """
        return None
    
    # ------------------------------------------------
    # Compatibility checking
    # ------------------------------------------------

    def is_real_sector(self, **kwargs) -> bool:
        """
        Check if the current sector is "real" (allows more compatibility).
        
        For example:
        - Translation:
            k=0 or k=pi are "real" sectors (commute with reflection)
        - Parity: 
            All sectors are real (eigenvalues +/- 1)
        - Custom symmetries: 
            Override to implement specific logic
        
        Parameters
        ----------
        **kwargs : dict
            Additional context (ns, nhl, etc.)
        
        Returns
        -------
        is_real : bool
            True if this sector has enhanced compatibility
        
        Notes
        -----
        This method should be overridden by symmetry classes that have
        sector-dependent compatibility (e.g., translation).
        Default implementation returns True (all sectors are "real").
        """
        return True
    
    def commutes_with(self, other: 'SymmetryOperator', **kwargs) -> bool:
        """
        Check if this symmetry commutes with another.
        
        Parameters
        ----------
        other : SymmetryOperator
            The other symmetry operator.
        **kwargs : dict
            Context-dependent parameters (e.g., momentum sector, ns, nhl)
            
        Returns
        -------
        commutes : bool
            True if the symmetries commute in the given context.
        
        Notes
        -----
        Compatibility rules:
        - Same class always commutes
        - Check unconditional compatibility set
        - Check if both sectors are "real" for conditional compatibility
        - Subclasses can override for custom logic
        
        Examples
        --------
        - Translation and Reflection commute only at k=0,pi
        """
        # Same class always commutes
        if self.symmetry_class == other.symmetry_class:
            return True
        
        # Check unconditional compatibility
        if other.symmetry_class in self.compatible_with:
            return True
        
        # Check conditional compatibility based on sector "realness"
        # For example, translation-reflection only commute at real sectors (k=0,pi)
        if self.is_real_sector(**kwargs) and other.is_real_sector(**kwargs):
            # Check if compatible at real sectors
            if (other.symmetry_class in self.momentum_dependent.get(MomentumSector.ZERO, set()) or
                other.symmetry_class in self.momentum_dependent.get(MomentumSector.PI,   set())):
                return True
        
        return False
    
    # ------------------------------------------------
    # Validation
    # ------------------------------------------------
    
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
        
        Examples
        --------
        - ParitySymmetry with spin-1/2 local space: 
            - valid
        - ParitySymmetry with fermionic local space: 
            - invalid
        - TranslationSymmetry with any local space: 
            - valid
        """
        # Empty set means universal - works for all local space types
        if not self.supported_local_spaces:
            return True
        
        return local_space_type in self.supported_local_spaces
    
    def check_boundary_conditions(self, lattice: Optional[Any] = None, **kwargs) -> Tuple[bool, str]:
        """
        Check if lattice boundary conditions are compatible with this symmetry.
        
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
        
        Notes
        -----
        Override this method in subclasses that have specific BC requirements.
        
        Examples
        --------
        - TranslationSymmetry requires periodic boundary conditions
        - ReflectionSymmetry may require symmetric boundaries
        
        Default implementation returns True (no BC restrictions).
        """
        return True, "Valid"
    
    def is_compatible_with_global_symmetry(self, global_symmetry: Any, **kwargs) -> Tuple[bool, str]:
        """
        Check if this symmetry is compatible with a global symmetry.
        
        Global symmetries (like U(1) particle conservation) can impose constraints
        on which local symmetries are allowed. For example:
        - U(1) particle number: Parity X/Y only allowed at half-filling
        - U(1) spin: Only Parity Z allowed (preserves magnetization)
        
        Parameters
        ----------
        global_symmetry : Any
            The global symmetry object to check against
        **kwargs : dict
            Additional context (ns, sector, etc.)
        
        Returns
        -------
        compatible : bool
            Whether this symmetry is compatible with the global symmetry
        reason : str
            Explanation if incompatible, "Compatible" otherwise
        
        Notes
        -----
        Override this in subclasses that have specific global symmetry constraints.
        
        Examples
        --------
        - ParitySymmetry should check U(1) constraints.
        - TranslationSymmetry is generally compatible with U(1).
        Default implementation returns True (no global symmetry restrictions).
        """
        return True, "Compatible"

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
#! EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~