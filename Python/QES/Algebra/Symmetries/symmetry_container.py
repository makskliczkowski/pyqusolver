"""
Symmetry container for managing symmetry operations in Hilbert spaces.

This module provides a general framework for handling symmetries in quantum many-body
systems - interacting and non-interacting alike. It supports various symmetry types
(e.g., translation, reflection, parity, etc.) through a unified interface.

Key Features
------------
1. **Uniform treatment**: 
    All symmetries (translation, reflection, parity, etc.) 
    are handled through the same interface.
2. **Automatic group construction**: 
    Builds full symmetry group from generators.
3. **Compatibility checking**: 
    Automatically determines which symmetries commute.
4. **Multiple backends**: 
    Supports integer, NumPy, and JAX state representations.
5. **Memory efficient**: 
    Computes representatives and normalizations on-the-fly.

----------------------------------------------------------------------------
File        : QES/Algebra/Symmetries/symmetry_container.py
Author      : Maksymilian Kliczkowski
Date        : 2025-10-28
Version     : 1.3.0
Changelog   :
    - 2025-10-28: Initial version
    - 2025-12-08: Added compact O(1) lookup structure for JIT compatibility, cleanup.
----------------------------------------------------------------------------
"""

from    __future__ import annotations

import  time
import  numba
import  numpy as np

# other
from    typing      import List, Tuple, Dict, Optional, Callable, Union, TYPE_CHECKING, TypeAlias
from    dataclasses import dataclass, field
from    itertools   import combinations

# --------------------------------------------------------------------------
#! Private helper functions
# --------------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def _binary_search_representative_list(mapping: np.ndarray, state: int) -> int:
    """
    Binary search to find index of state in sorted mapping array.
    Memory efficient - O(1) space, O(log Nh) time.
    
    Allows to search through mapping:
    
    full_hilbert_space_state --binary_search -> representative_index (in representative_list)
    
    Args:
        mapping: 
            Sorted array of representative states
        state: 
            State to find
        
    Returns:
        Index in mapping if found, -1 otherwise
    """
    left    = 0
    right   = len(mapping) - 1
    
    while left <= right:
        mid         = (left + right) // 2
        if mapping[mid] == state:
            return mid
        elif mapping[mid] < state:
            left    = mid + 1
        else:
            right   = mid - 1
    return numba.int64(-1)

# --------------------------------------------------------------------------
#! JIT-compiled compact symmetry lookup functions
# --------------------------------------------------------------------------

_STATE_TYPE             : TypeAlias = np.int64
_STATE_TYPE_NB          : TypeAlias = numba.int64
_REPR_MAP_DTYPE         : TypeAlias = np.uint32
_REPR_MAP_DTYPE_NB      : TypeAlias = numba.uint32
_PHASE_IDX_DTYPE        : TypeAlias = np.uint8
_PHASE_IDX_DTYPE_NB     : TypeAlias = numba.uint8

_INVALID_REPR_IDX_NB    : TypeAlias = numba.uint32(0xFFFFFFFF)  # Max uint32 for numba  (sufficient for up to ~4 billion representatives)
_INVALID_PHASE_IDX_NB   : TypeAlias = numba.uint8(0xFF)         # Max uint8 for numba   (sufficient for up to 255 distinct phases)

@numba.njit(cache=True, fastmath=True)
def _compact_get_sym_factor(
        state           : _STATE_TYPE_NB,               # new_state after operator action
        k               : _STATE_TYPE_NB,               # input representative index
        repr_map        : np.ndarray,                   # uint32                    [state -> repr int                      (nh_full,           )]
        phase_idx       : np.ndarray,                   # uint8                     [state -> phase table index             (nh_full,           )]
        phase_table     : np.ndarray,                   # complex128                [phase table index -> phase             (n_phases,          )]
        normalization   : np.ndarray,                   # float64                   [representative index -> normalization  (n_representatives, )]
    ) -> Tuple[_STATE_TYPE_NB, numba.complex128]:
    """
    JIT-compiled O(1) lookup for symmetry factor in matrix element computation.
    
    Given a new_state produced by an operator acting on representative k,
    returns (representative_index, symmetry_factor) for the matrix element.
    
    This mathematically corresponds to the matrix element <bra|O|ket> where
    |bra> is the new state and |ket> is the input representative.
    
    Parameters
    ----------
    state : int
        The new state produced by operator action, (bra in matrix element)
    k : int
        Index of the input representative - helps determine normalization (ket in matrix element)
    repr_map : ndarray[uint32]
        Maps state -> representative index
    phase_idx : ndarray[uint8]
        Maps state -> phase table index
    phase_table : ndarray[complex128]
        Array of distinct phase values
    normalization : ndarray[float64]
        Normalization factors per representative
        
    Returns
    -------
    idx : int
        Representative index (-1 if state not in sector)
    sym_factor : complex
        Symmetry factor: conj(phase) * N_idx / N_k
    """
    idx         = repr_map[state]       # representative for the new state (if exists, otherwise invalid)
    if idx == _INVALID_REPR_IDX_NB:
        return numba.int64(-1), numba.complex128(0.0)
    
    pidx        = phase_idx[state]      # phase table index for a given new state produced
    phase       = phase_table[pidx]     # corresponding phase value
    norm_idx    = normalization[idx]    # normalization for the representative (new state)
    norm_k      = normalization[k]      # normalization for the input representative (old state)
    
    sym_factor  = np.conj(phase) * norm_idx / norm_k
    return numba.int64(idx), sym_factor

@numba.njit(cache=True, fastmath=True, inline='always')
def _compact_is_in_sector(state: _STATE_TYPE_NB, repr_map: np.ndarray) -> bool:
    """Check if a state belongs to this symmetry sector (O(1) lookup)."""
    return repr_map[state] != _INVALID_REPR_IDX_NB

@numba.njit(cache=True, fastmath=True, inline='always')
def _compact_get_repr_idx(state: _STATE_TYPE_NB, repr_map: np.ndarray) -> _STATE_TYPE_NB:
    """Get representative index for a state (-1 if not in sector)."""
    idx = repr_map[state]
    if idx == _INVALID_REPR_IDX_NB:
        return numba.int64(-1)
    return numba.int64(idx)

@numba.njit(cache=True, fastmath=True, inline='always')
def _compact_get_phase(state: _STATE_TYPE_NB, phase_idx: np.ndarray, phase_table : np.ndarray) -> numba.complex128:
    """Get symmetry phase for a state (O(1) lookup)."""
    pidx = phase_idx[state]
    if pidx == _INVALID_PHASE_IDX_NB:
        return numba.complex128(0.0)
    return phase_table[pidx]

@numba.njit(cache=True, fastmath=True, inline='always')
def _compact_convert_states_to_idx(full_array: np.ndarray, repr_list: np.ndarray) -> np.ndarray:
    """Convert array of states to their representative indices using binary search without copy!"""
    n_states    = full_array.shape[0]
        
    for i in numba.prange(n_states):
        state           = full_array[i]
        if state == _INVALID_REPR_IDX or state > repr_list[-1]: # assume repr_list is sorted
            continue
        idx             = _binary_search_representative_list(repr_list, state)
        full_array[i]   = idx
    return full_array

@numba.njit(cache=True, fastmath=True, inline='always')
def _compact_convert_idx_to_states(idx_array: np.ndarray, repr_list: np.ndarray) -> np.ndarray:
    """Convert array of representative indices to their actual states using direct indexing without copy!"""
    n_states    = idx_array.shape[0]
        
    for i in numba.prange(n_states):
        idx             = idx_array[i]
        if idx == _INVALID_REPR_IDX:
            continue
        state           = repr_list[idx]
        idx_array[i]    = state
    return idx_array

# --------------------------------------------------------------------------

try:
    from QES.Algebra.Symmetries.base            import SymmetryOperator
    from QES.Algebra.Operator.operator          import SymmetryGenerators
    from QES.Algebra.globals                    import GlobalSymmetry
    from QES.general_python.lattices.lattice    import Lattice
    
    if TYPE_CHECKING:
        from QES.general_python.common.flog     import Logger
    
    try:
        import jax.numpy    as jnp
        JAX_AVAILABLE       = True
    except ImportError:
        jnp                 = np
        JAX_AVAILABLE       = False
        
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}")

#############################################################################
#! Constants
#############################################################################

_SYM_NORM_THRESHOLD = 1e-12
_INT_HUGE           = np.iinfo(np.int64).max            # THE HUGEST!!!
_INVALID_REPR_IDX   = np.iinfo(_REPR_MAP_DTYPE).max     # ~4 billion, marks state not in sector
_INVALID_PHASE_IDX  = np.iinfo(_PHASE_IDX_DTYPE).max    # 255, marks invalid phase index

#############################################################################
#! Compact Symmetry Data Structure
#############################################################################

@dataclass
class CompactSymmetryData:
    """
    Memory-efficient compact representation of symmetry data for JIT-friendly O(1) lookups.
    
    This structure stores the symmetry mapping in a compact format:
    - repr_map: 
        uint32 array mapping each state -> representative INDEX (not state value) 
        - (size = nh_full)
    - phase_idx: 
        uint8 array mapping each state -> index into phase_table 
        - (size = nh_full)
    - phase_table:
        small complex128 array of distinct phases
        - (size = group order)
    - normalization:
        float64 array of normalization factors per representative
        - (size = representatives number)
    
    Memory Usage
    ------------
    Per state: 4 bytes (repr_map) + 1 byte (phase_idx) = 5 bytes

    For a 2^20 (~1M) state Hilbert space:
    - New format: ~5 MB
    
    JIT Compatibility
    -----------------
    All arrays use fixed-size dtypes suitable for numba.njit:
    - repr_map      : np.uint32         (supports up to 4 billion representatives)
    - phase_idx     : np.uint8          (supports up to 255 distinct phases). If not enough distinct phases are needed, a larger dtype should be used.
    - phase_table   : np.complex128
    - normalization : np.float64
    
    The structure enables O(1) lookup for matrix element computation:
    ```python
    @njit
    def get_sym_factor(state, k, repr_map, phase_idx, phase_table, normalization):
        idx = repr_map[state]
        if idx == INVALID_REPR_IDX:
            return -1, 0.0 # State not in sector
            
        phase       = phase_table[phase_idx[state]]
        norm_idx    = normalization[idx]
        norm_k      = normalization[k]
        return idx, np.conj(phase) * norm_idx / norm_k # conjugate because we 'return' to representative
    ```
    
    Attributes
    ----------
    repr_map : np.ndarray
        Shape (nh_full,), dtype uint32. Maps state -> representative index.
        Value = _INVALID_REPR_IDX if state not in this symmetry sector.
    phase_idx : np.ndarray
        Shape (nh_full,), dtype uint8. Maps state -> index in phase_table.
    phase_table : np.ndarray
        Shape (n_phases,), dtype complex128. Distinct phase values.
        Typically n_phases <= group_order (often just a few values like 1, -1, i, -i).
    normalization : np.ndarray
        Shape (n_representatives,), dtype float64. Normalization per representative.
    representative_list : np.ndarray
        Shape (n_representatives,), dtype int64. Maps index -> actual state value.
    """
    repr_map            : np.ndarray  # uint32[nh_full]         : state -> representative index
    phase_idx           : np.ndarray  # uint8[nh_full]          : state -> phase table index
    phase_table         : np.ndarray  # complex128[n_phases]    : distinct phases
    normalization       : np.ndarray  # float64[n_repr]         : normalization per representative
    representative_list : np.ndarray  # int64[n_repr]           : representative state values
    
    @property
    def n_representatives(self) -> int:
        """Number of representatives (reduced Hilbert space dimension)."""
        return len(self.representative_list)
    
    @property
    def n_phases(self) -> int:
        """Number of distinct phases in the phase table."""
        return len(self.phase_table)
    
    @property 
    def nh_full(self) -> int:
        """Full Hilbert space dimension."""
        return len(self.repr_map)
    
    def __call__(self, idx: int) -> int:
        return int(self.representative_list[idx])
    
    def __getitem__(self, idx: int) -> int:
        return int(self.phase_table[idx])
    
    def __len__(self) -> int:
        return len(self.representative_list)
    
    def __contains__(self, state: int) -> bool:
        return self.is_in_sector(state)
    
    def __iter__(self):
        return iter(self.representative_list)
    
    # --------------------------------------------------
    
    def get_representative_state(self, idx: int) -> int:
        """Get the actual state value for a representative index."""
        return int(self.representative_list[idx])
    
    def get_phase(self, state: int) -> complex:
        """Get the symmetry phase for a state."""
        pidx = self.phase_idx[state]
        if pidx == _INVALID_PHASE_IDX:
            return 0.0
        return self.phase_table[pidx]
    
    def get_repr_idx(self, state: int) -> int:
        """Get the representative index for a state (-1 if not in sector)."""
        idx = self.repr_map[state]
        if idx == _INVALID_REPR_IDX:
            return -1
        return int(idx)
    
    def is_in_sector(self, state: int) -> bool:
        """Check if a state belongs to this symmetry sector."""
        return self.repr_map[state] != _INVALID_REPR_IDX

#############################################################################
#! Type Aliases
#############################################################################

StateInt            = int                                                       # Integer representation of state - normally 64-bit
StateArray          = np.ndarray                                                # or jnp.ndarray if JAX is used    
SymmetrySpec        = Tuple[SymmetryGenerators, Union[int, float, complex]]     # (Symmetry type, sector value)
GroupElement        = Tuple[SymmetryOperator, ...]                              # Tuple of operators to apply sequentially

#############################################################################
#! Symmetry Compatibility Checker
#############################################################################

class SymmetryCompatibility:
    """
    Determines which symmetry generators are compatible with each other.
    Uses the symmetries' own compatibility rules rather than hardcoded logic.
    
    This class coordinates compatibility checking but delegates the actual
    logic to the SymmetryOperator instances themselves.
    
    Examples
    --------
    >>> compat = SymmetryCompatibility(ns=4, nhl=2, lattice=lattice)
    >>> # Symmetries define their own compatibility
    >>> compat.check_compatibility(sym1, sym2)
    """
    
    def _check_logger(self, logger: Optional['Logger']) -> None:
        ''' Ensure logger is set. '''
        if logger is None:
            from QES.general_python.common.flog import get_global_logger
            logger  = get_global_logger()
        self.logger = logger
    
    def __init__(self, ns: int, nhl: int = 2, lattice: Optional['Lattice'] = None, logger: Optional['Logger'] = None):
        """
        Initialize compatibility checker.
        
        Parameters
        ----------
        ns : int
            Number of sites in the system
        nhl : int
            Local Hilbert space dimension
        lattice : Optional[Lattice]
            Lattice structure of the system
        logger : Optional[Callable[[str], None]]
            Logger function for debugging messages
        """
        self._check_logger(logger)
        self.ns         = ns
        self.nhl        = nhl
        self.lattice    = lattice

        # Cache of compatibility decisions
        self._compat_cache: Dict[Tuple, bool] = {}
    
    # -----------------------------------------------------
    #! Pairwise compatibility check
    # -----------------------------------------------------
    
    def check_pair_compatibility(
            self,
            op1     : SymmetryOperator,
            op2     : SymmetryOperator,
            spec1   : SymmetrySpec,
            spec2   : SymmetrySpec
        ) -> Tuple[bool, str]:
        """
        Check if two symmetry operators are compatible.
        
        Delegates to the symmetries' own commutes_with() methods.
        
        Parameters
        ----------
        op1, op2 : SymmetryOperator
            The symmetry operators - instances of SymmetryOperator
            with their own compatibility logic
        spec1, spec2 : SymmetrySpec
            The (generator_type, sector) specifications
        
        Returns
        -------
        compatible : bool
            Whether the operators can be used together
        reason : str
            Explanation of the decision
        """
        gen1, sector1   = spec1
        gen2, sector2   = spec2
        
        # Build context for compatibility check
        context         = {
                            'ns'        : self.ns,
                            'nhl'       : self.nhl,
                            'lattice'   : self.lattice
                        }
        
        # Check if op1 commutes with op2
        if op1.commutes_with(op2, **context):
            return True, "Compatible"
        
        # Check both sectors for "realness" - more specific reason
        if not op1.is_real_sector(**context):
            return False, f"{gen1.name} sector {sector1} incompatible with {gen2.name}"
        
        if not op2.is_real_sector(**context):
            return False, f"{gen2.name} sector {sector2} incompatible with {gen1.name}"
        
        return False, f"{gen1.name} and {gen2.name} do not commute"
    
    # -----------------------------------------------------
    
    def check_boundary_conditions(self, operator: SymmetryOperator) -> Tuple[bool, str]:
        """
        Check if lattice boundary conditions support this symmetry.
        
        Delegates to the operator's own check_boundary_conditions() method.
        
        Parameters
        ----------
        operator : SymmetryOperator
            The symmetry operator to check
        
        Returns
        -------
        valid : bool
            Whether boundary conditions are compatible
        reason : str
            Explanation
        """
        return operator.check_boundary_conditions(lattice=self.lattice, ns=self.ns, nhl=self.nhl)
    
    # -----------------------------------------------------

    def check_global_symmetry_effects(self, 
                                    operators   : List[Tuple[SymmetryOperator, SymmetrySpec]], 
                                    global_syms : List[GlobalSymmetry]) -> List[Tuple[SymmetryOperator, SymmetrySpec]]:
        """
        Remove or modify operators based on global symmetries.
        
        Delegates to each symmetry's is_compatible_with_global_symmetry() method.
        This is fully general and extensible - no hardcoded rules.

        Parameters
        ----------
        operators : List[Tuple[SymmetryOperator, SymmetrySpec]]
            List of (operator, (gen_type, sector)) tuples
        global_syms : List[GlobalSymmetry]
            Global symmetries to check against
        
        Returns
        -------
        filtered : List[Tuple[SymmetryOperator, SymmetrySpec]]
            Operators compatible with global symmetries
        """
        if not global_syms:
            return operators  # No global symmetries, no filtering needed
        
        filtered = []
        
        for op, spec in operators:
            gen, sector = spec
            keep        = True
            reason      = ""
            
            # Check compatibility with each global symmetry
            for gsym in global_syms:
                compatible, msg = op.is_compatible_with_global_symmetry(
                                    gsym, 
                                    ns      = self.ns, 
                                    nhl     = self.nhl, 
                                    lattice = self.lattice,
                                    sector  = sector
                                )
                if not compatible:
                    keep    = False
                    reason  = msg
                    break
            
            if keep:
                filtered.append((op, spec))
            elif reason:
                self.logger.info(reason)
        
        return filtered
    
    # -----------------------------------------------------
    #! Main compatibility checker
    # -----------------------------------------------------
    
    def get_compatible_operators(self,
            operators       : List[Tuple[SymmetryOperator, SymmetrySpec]],
            global_syms     : List[GlobalSymmetry]) -> List[Tuple[SymmetryOperator, SymmetrySpec]]:
        """
        Filter operators to only include compatible ones.
        
        Parameters
        ----------
        operators : List[Tuple[SymmetryOperator, SymmetrySpec]]
            Operators to filter
        global_syms : List[GlobalSymmetry]
            Global symmetries
        
        Returns
        -------
        compatible : List[Tuple[SymmetryOperator, SymmetrySpec]]
            Mutually compatible operators
        """
        
        # First filter based on global symmetries
        filtered    = self.check_global_symmetry_effects(operators, global_syms)
        
        # Now check pairwise compatibility
        result      = []
        
        for op, spec in filtered:
            gen, sector = spec
            
            # Check BC compatibility
            bc_valid, bc_reason = self.check_boundary_conditions(op)
            if not bc_valid:
                self.logger.warning(f"Removing {gen.name}: {bc_reason}")
                continue
            
            # Check pairwise compatibility with already accepted operators
            compatible  = True
            for prev_op, prev_spec in result:
                is_compat, reason = self.check_pair_compatibility(op, prev_op, spec, prev_spec)
                if not is_compat:
                    self.logger.warning(f"Removing {gen.name}: {reason}")
                    compatible = False
                    break
            
            if compatible:
                result.append((op, spec))
        
        return result # Return only compatible operators!

####################################################################################################
#! Symmetry Container
####################################################################################################

@dataclass
class SymmetryContainer:
    """
    TODO: Consider non-abelian symmetries in future.
    
    Container for all symmetry operations in a Hilbert space.
    
    This class provides a unified interface for:
    - Building symmetry groups from generators      - 'build_group()'           # e.g., TranslationSymmetry, ReflectionSymmetry
    - Finding representative states                 - 'find_representative()'   # e.g., minimal state in orbit
    - Computing normalization factors               - 'compute_normalization()' # e.g., stabilizer subgroup sums
    - Acting with symmetries on states              - 'act_with_symmetry()'     # e.g., apply operator to state

    Architecture
    ------------
    The container separates global and local symmetries:
    
    - **Global symmetries**: 
        Act as filters (e.g., U(1) particle conservation)
        see the theoretical description in the documentation...
        
        1. Check if a state satisfies the constraint
        2. Don't form groups or orbits
        3. Applied before representative finding

    - **Local symmetries**: 
        Form groups and orbits (e.g., translation, reflection)
        1. Build full symmetry group from generators
        2. Find representative (minimal state in orbit)
        3. Compute normalization (stabilizer subgroup sum)

    How to Add New Symmetries
    --------------------------
    To add a new symmetry operator:
    
    1. **Create symmetry class** in QES/Algebra/Symmetries/
       ```python
       class MySymmetry(SymmetryOperator):
           def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
               # Transform integer state, return (new_state, phase)
               ...
           
           def apply_numpy(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, complex]:
               # Transform numpy state vector
               ...
           
           def apply_jax(self, state: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, complex]:
               # Transform JAX state vector (if JAX available)
               ...
       ```
    
    2. **Add to SymmetryGenerators enum** in operator.py
       ```python
       class SymmetryGenerators(Enum):
           MY_SYMMETRY = "my_symmetry"
       ```
    
    3. **Register compatibility rules** in SymmetryCompatibility.check_pair_compatibility()
       ```python
       if {gen1, gen2} == {SymmetryGenerators.MY_SYMMETRY, SymmetryGenerators.OTHER}:
           return False, "Reason why incompatible"
       ```
       
    4. **Register global compatibility effects** in SymmetryCompatibility.check_global_symmetry_effects()
    
    5. **Use in HilbertSpace**
       ```python
       hilbert = HilbertSpace(
           ns       = 10,
           sym_gen  = [
               (SymmetryGenerators.MY_SYMMETRY, sector_value),
               ...
           ]
       )
       ```
    
    The container automatically:
    - Checks compatibility with other symmetries        - pairwise checks
    - Builds the full symmetry group                    - combinations of generators
    - Finds representatives using your apply_int method - minimal state in orbit
    - Computes normalizations                           - stabilizer subgroup sums
    
    Examples
    --------
    Basic usage:
    
    >>> from QES.Algebra.Symmetries.translation import TranslationSymmetry
    >>> container = SymmetryContainer(ns=4, lattice=lattice)
    >>> container.add_generator(SymmetryGenerators.T, sector=0, operator=TranslationSymmetry(lattice))
    >>> container.build_group()
    >>> rep, phase = container.find_representative(5) # Find rep of state |0101>
    
    Parameters
    ----------
    ns : int
        Number of sites in the system
    lattice : Optional[Lattice]
        Lattice structure (needed for spatial symmetries)
    nhl : int
        Local Hilbert space dimension (default: 2 for spin-1/2)
    backend : str
        Backend for computations ('numpy', 'jax', or 'default')
    """
    
    ns                  : int
    lattice             : Optional[Lattice]                             = None
    nhl                 : int                                           = 2
    backend             : str                                           = 'default'

    # Storage
    generators          : List[Tuple[SymmetryOperator, SymmetrySpec]]   = field(default_factory=list)
    global_symmetries   : List[GlobalSymmetry]                          = field(default_factory=list)
    symmetry_group      : List[GroupElement]                            = field(default_factory=list)
    _repr_list          : Optional[np.ndarray]                          = None      # representatives list -> state (int64), defaults to None (number of representatives, )
    _repr_norms         : Optional[np.ndarray]                          = None      # representatives normalization factors (float64), defaults to 1.0 (number of states in orbit, )
    _repr_get_norm      : Callable[[StateInt], complex]                 = lambda x  : 1.0 # function to get normalization for a state, defaults to 1.0
    _repr_active        : Optional[bool]                                = False     # if one adds new element, it stops being valid, i.e., needs recomputation of representatives
    # Compact O(1) lookup structure (primary storage, always built when symmetries present)
    _compact_data       : Optional[CompactSymmetryData]                 = None
    _compatibility      : Optional[SymmetryCompatibility]               = None      # Compatibility checker instance
    logger              : Optional[Callable[[str], None]]               = None      # Logger function
    
    # -----------------------------------------------------
    #! Initialization
    # -----------------------------------------------------

    def __post_init__(self):
        """Initialize compatibility checker and logger."""
        if self.logger is None:
            from QES.general_python.common.flog import get_global_logger
            self.logger     = get_global_logger()
        self._compatibility = SymmetryCompatibility(self.ns, self.nhl, self.lattice, self.logger)
        
    def set_repr_info(self, repr_list: np.ndarray, repr_norms: np.ndarray) -> None:
        """Set the representatives and their normalization factors."""
        # Ensure repr_list is a numpy array to avoid numba warnings
        self._repr_list     = np.asarray(repr_list,  dtype=_STATE_TYPE) if not isinstance(repr_list,  np.ndarray  ) else repr_list
        self._repr_norms    = np.asarray(repr_norms) if not isinstance(repr_norms, np.ndarray  ) else repr_norms
        self._repr_get_norm = lambda x: self._repr_norms[x]
        self._repr_active   = True
        
    # -----------------------------------------------------

    @property
    def repr_list(self) -> Optional[np.ndarray]:        return self._repr_list
    @repr_list.setter
    def repr_list(self, value: np.ndarray) -> None:     self._repr_list = value; self._repr_active = True
    
    @property
    def repr_norms(self) -> Optional[np.ndarray]:       return self._repr_norms
    @repr_norms.setter
    def repr_norms(self, value: np.ndarray) -> None:    self._repr_norms = value; self._repr_get_norm = lambda x: self._repr_norms[x]

    @property
    def repr_active(self) -> bool:                      return self._repr_active
    
    @property
    def compact_data(self) -> Optional[CompactSymmetryData]:
        """Get the compact symmetry data structure for O(1) JIT lookups."""
        return self._compact_data
    
    @property
    def has_compact_data(self) -> bool:
        """Check if compact symmetry data is available."""
        return self._compact_data is not None

    # -----------------------------------------------------
    #! Generator Management
    # -----------------------------------------------------

    def add_generator(
            self, 
            gen_type    : SymmetryGenerators, 
            sector      : Union[int, float, complex],
            operator    : SymmetryOperator
        ) -> bool:
        """
        Add a symmetry generator to the container.
        
        Parameters
        ----------
        gen_type : SymmetryGenerators
            Type of symmetry
        sector : Union[int, float, complex]
            Sector value (quantum number)
        operator : SymmetryOperator
            Symmetry operator instance
        
        Returns
        -------
        added : bool
            Whether the generator was successfully added
        """
        spec                = (gen_type, sector)
        
        # Check boundary conditions first
        bc_valid, bc_reason = self._compatibility.check_boundary_conditions(operator)
        if not bc_valid:
            self.logger.warning(f"Cannot add {gen_type.name}: {bc_reason}")
            return False
        
        # Check compatibility with existing generators
        for existing_op, existing_spec in self.generators:
            compat, reason  = self._compatibility.check_pair_compatibility(operator, existing_op, spec, existing_spec)
            if not compat:
                self.logger.warning(f"Cannot add {gen_type.name}: {reason}")
                return False
        
        self.generators.append((operator, spec))
        
        # Log addition
        self.logger.info(f"Added symmetry generator: {gen_type.name} = {sector}")
        self._repr_active = False
        return True
    
    # -----------------------------------------------------
    #! Global Symmetry Management
    # -----------------------------------------------------
    
    def add_global_symmetry(self, global_sym: GlobalSymmetry) -> None:
        """Add a global symmetry (filtering constraint)."""
        self.global_symmetries.append(global_sym)
        self.logger.info(f"Added global symmetry: {global_sym.name}")
        self._repr_active = False
    
    # -----------------------------------------------------
    #! Compatibility Filtering - built-in function
    # -----------------------------------------------------
    
    def build_group(self) -> None:
        r"""
        TODO: Make cyclic group detection more general (beyond translations).
        
        Build the full symmetry group from generators.
        
        Algorithm
        ---------
        1. Separate cyclic groups from other generators
        2. For non-translation generators, create all combinations
        3. Build translation group (product of cyclic groups for multi-D)
        4. Combine: full_group = non_translation_combos x translation_group
        
        For 2D with Tx and Ty:
            - Non-translation:
                - {E, P, R, PR, ...}
            - Translation: 
                - {Tx^i Ty^j : i=0..Nx-1, j=0..Ny-1}
            - Full: 
                - each non-translation combo x each translation combo
        """
        if not self.generators:
            self.logger.info("No symmetry generators - empty group")
            self.symmetry_group = [()]  # Identity element
            return
        
        # Separate translations from other generators
        translations: Dict[str, Tuple[SymmetryOperator, SymmetrySpec]] = {} # direction -> (op, spec)
        other_generators = []
        
        for op, spec in self.generators:
            gen_type = spec[0]
            # Check if this is any translation symmetry
            if gen_type == SymmetryGenerators.Translation_x:
                translations['x'] = (op, spec)
            elif gen_type == SymmetryGenerators.Translation_y:
                translations['y'] = (op, spec)
            elif gen_type == SymmetryGenerators.Translation_z:
                translations['z'] = (op, spec)
            else:
                other_generators.append((op, spec))
        
        # Build combinations of non-translation generators
        # Start with identity (empty tuple)
        base_elements: List[GroupElement] = [()]
        
        # Add all combinations using bitmask approach
        n_other = len(other_generators)
        for r in range(1, n_other + 1):
            for combo_indices in combinations(range(n_other), r):
                # Build operator tuple for this combination
                ops_tuple = tuple(other_generators[i][0] for i in combo_indices)
                base_elements.append(ops_tuple)
        
        # If no translations, we're done
        if not translations:
            self.symmetry_group = base_elements
            self.logger.info(f"Built symmetry group with {len(self.symmetry_group)} elements")
            return
        
        # Build translation group (product of cyclic groups)
        translation_elements = self._build_translation_group(translations)
        
        # Combine: each base element with each translation element
        full_group: List[GroupElement] = []
        for t_elem in translation_elements:
            for base_elem in base_elements:
                # Concatenate tuples: (translation ops) + (other ops)
                combined = t_elem + base_elem
                full_group.append(combined)
        
        self.symmetry_group = full_group
        self.logger.info(f"Built symmetry group with {len(self.symmetry_group)} elements ({len(translation_elements)} translation x {len(base_elements)} base)")
    
    def _build_translation_group(self, translations: Dict[str, Tuple[SymmetryOperator, SymmetrySpec]]) -> List[GroupElement]:
        r"""
        Build the translation subgroup as a product of cyclic groups.
        
        For 1D (only Tx):
        - {E, Tx, Tx^2, ..., Tx^(Nx-1)}
        For 2D (Tx, Ty):
        - {Tx^i Ty^j : i=0..Nx-1, j=0..Ny-1}
        For 3D (Tx, Ty, Tz):
        - {Tx^i Ty^j Tz^k : i,j,k over ranges}
        
        Parameters
        ----------
        translations : Dict[str, Tuple[SymmetryOperator, SymmetrySpec]]
            Dictionary mapping direction ('x', 'y', 'z') to (operator, spec)
        
        Returns
        -------
        translation_group : List[GroupElement]
            All combinations of translation powers
        """
        # Determine the period for each direction
        # For periodic BC, translation^N = identity, so we have N elements in cyclic group
        periods = {'x': self.ns, 'y': self.ns, 'z': self.ns}
        
        # If lattice has different sizes in different directions, use those
        if self.lattice is not None:
            if hasattr(self.lattice, 'lx') and 'x' in translations:
                periods['x'] = self.lattice.lx
            if hasattr(self.lattice, 'ly') and 'y' in translations:
                periods['y'] = self.lattice.ly
            if hasattr(self.lattice, 'lz') and 'z' in translations:
                periods['z'] = self.lattice.lz
        
        # Build all combinations
        translation_group: List[GroupElement] = []
        
        # Get sorted directions for consistent ordering
        directions = sorted(translations.keys())
        
        if len(directions) == 0:
            return [()]
        
        # Build product of cyclic groups
        # For each combination of powers (i, j, k, ...) create Tx^i Ty^j Tz^k ...
        ranges = [range(periods[d]) for d in directions]
        
        # Generate all combinations using itertools.product
        from itertools import product as cartesian_product
        
        # Build each group element
        for powers in cartesian_product(*ranges):
            # Build tuple of operators: Tx repeated i times, Ty repeated j times, etc.
            ops_tuple = ()
            for direction, power in zip(directions, powers):
                t_op, t_spec    = translations[direction]
                ops_tuple      += tuple([t_op] * power)
            
            # Add this translation operator 'power' times
            translation_group.append(ops_tuple)
        
        return translation_group
    
    # -----------------------------------------------------
    #! Core Functionality
    # -----------------------------------------------------
    
    def apply_group_element(self, element: GroupElement, state: StateInt) -> Tuple[StateInt, Union[complex, float]]:
        """
        Apply a group element (tuple of operators) to a state.
        
        Parameters
        ----------
        element : GroupElement
            Tuple of operators to apply sequentially
        state : StateInt
            Integer representation of state
        
        Returns
        -------
        new_state : StateInt
            Transformed state
        phase : complex
            Accumulated phase from symmetry eigenvalues
        """
        if len(element) == 0:
            return state, 1.0 # Identity element        
        
        current_state       = state
        accumulated_phase   = 1.0
        
        if isinstance(current_state, (int, np.integer)):
            for op in element:
                current_state, phase = op.apply_int(current_state, self.ns, nhl=self.nhl)
                accumulated_phase   *= phase
        elif isinstance(current_state, np.ndarray):
            for op in element:
                current_state, phase = op.apply_numpy(current_state, ns=self.ns, nhl=self.nhl)
                accumulated_phase   *= phase
        elif JAX_AVAILABLE and isinstance(current_state, jnp.ndarray):
            for op in element:
                current_state, phase = op.apply_jax(current_state, ns=self.ns, nhl=self.nhl)
                accumulated_phase   *= phase
        else:
            raise TypeError(f"Unsupported state type for symmetry application: {type(current_state)}")

        return current_state, accumulated_phase

    # -----------------------------------------------------
    #! Representative Finding
    # -----------------------------------------------------

    def _find_representative(self, state: StateInt) -> StateInt:
        r"""
        Internal method to find representative state without phase.
        This is done by applying all group elements and finding minimal state
        without using any caching. CORE LOGIC.
        
        Parameters
        ----------
        state : StateInt
            State to find representative for
        
        Returns
        -------
        representative : StateInt, complex
            Minimal state in the orbit of the given state
            and the associated symmetry eigenvalue (character)
        """
        min_state   = _INT_HUGE
        min_element = ()
        idx          = -1
        
        # Try all group elements
        for i, element in enumerate(self.symmetry_group):
            # Apply group element to get transformed state
            # (we only need the state, not the boundary phase)
            new_state, _ = self.apply_group_element(element, state)
            
            if new_state < min_state:
                min_state   = new_state # Update minimal state
                min_element = element   # Group element that gives minimal state
                idx         = i         # Index of the group element that gives minimal state
        
        # Get the character (representation eigenvalue) for the transformation
        character = self.get_character(min_element)
        return min_state, character, idx

    def find_representative(self, 
                            state           : StateInt, 
                            normalization_b : Union[float, complex] = 1.0,
                            use_cache       : bool                  = True) -> Tuple[StateInt, complex]:
        r"""
        Find the representative (minimal state) in the orbit of a given state.
        
        The representative is defined as the state with the smallest integer value
        that can be reached by applying symmetry operations.
        
        Parameters
        ----------
        state : StateInt
            State to find representative for
        use_cache : bool
            Whether to use cached representative map if available
        nb : Union[float, complex]
            Normalization of the other Hilbert space (needed for phase consistency). 
            This is necessary when we try to compute the action of observables that 
            connect different symmetry sectors. For example, when computing <psi_a|O|psi_b>, where
            psi_a and psi_b belong to different symmetry sectors, we need to ensure that the phases
            and normalizations are consistent. The normalization_b parameter allows us to
            adjust the phase of the representative state found in sector b to match that of sector a.
            
            Example:
            --------
            - spin-1/2 in computational basis
                - sector a: parity +1
                - we act with S^+_i operator (which flips spin at site i)
                - resulting state belongs to sector b: parity -1
                - we find representative of resulting state in sector b
                - to ensure correct phase, we need to divide by normalization_b
                - this ensures that the overall phase and normalization of <psi_a|O|psi_b>
                  is correct

        Returns
        -------
        representative : StateInt
            Minimal state in the orbit
        symmetry_eigenvalue : complex
            Phase accumulated when transforming state -> representative
        
        Algorithm
        ---------
        1. Check cache if available
        2. Apply all group elements to the state
        3. Find the transformation that gives minimal state
        4. Return (min_state, phase_to_reach_it)
        """

        # No symmetries - state is its own representative
        if not self._repr_active or (self._repr_list is None) or len(self.symmetry_group) == 0:
            return state, 1.0 # No mapping - return state itself

        # Check compact data cache - O(1) lookup via uint32 repr_map + uint8 phase_idx
        if use_cache and self._compact_data is not None and state < len(self._compact_data.repr_map):
            cd          = self._compact_data
            rep_idx     = int(cd.repr_map[state])
            
            # Check if state is in this sector (_INVALID_REPR_IDX = 0xFFFFFFFF means not in sector)
            if rep_idx != _INVALID_REPR_IDX and rep_idx >= 0 and rep_idx < len(self._repr_list):
                # Get the actual representative state and phase from compact data
                rep_state   = int(self._repr_list[rep_idx])
                phase       = cd.phase_table[cd.phase_idx[state]]
                norm        = cd.normalization[rep_idx]
                return rep_state, norm * np.conjugate(phase) / normalization_b
            
            # Cache indicates state not in sector - return zero
            return 0, 0.0

        try:
            from QES.general_python.common.embedded.binary_search import binary_search_numpy, _BAD_BINARY_SEARCH_STATE
        except ImportError:
            raise ImportError("binary_search_numpy not found - ensure QES.general_python.common.embedded is available.")
        
        # Check if state is already a representative
        state_in_mapping        = binary_search_numpy(self._repr_list, 0, len(self._repr_list) - 1, state)
        if state_in_mapping != _BAD_BINARY_SEARCH_STATE and state_in_mapping < len(self._repr_list):
            return int(self._repr_list[state_in_mapping]), self._repr_get_norm(state_in_mapping) / normalization_b # State is already a representative

        # Otherwise, find representative by applying group elements
        min_state, min_phase, _ = self._find_representative(state)
        state_in_mapping        = binary_search_numpy(self._repr_list, 0, len(self._repr_list) - 1, min_state)    
        if state_in_mapping != _BAD_BINARY_SEARCH_STATE and state_in_mapping < len(self._repr_list):
            return int(self._repr_list[state_in_mapping]), self._repr_get_norm(state_in_mapping) * np.conjugate(min_phase) / normalization_b

        return 0, 0.0 # State not in this sector (maybe -1?)

    # -----------------------------------------------------
    #! Character and Normalization Computation
    # -----------------------------------------------------
    
    def get_character(self, element: GroupElement) -> complex:
        r"""
        Compute the character (representation eigenvalue) for a group element.
        
        For translation T^n in momentum sector k: chi_k(T^n) = exp(2pi i * k * n / L)
        For other symmetries: chi(g) = sector_value (usually +/- 1)
        
        Parameters
        ----------
        element : GroupElement
            Tuple of symmetry operators representing a group element
        
        Returns
        -------
        character : complex
            Character value for this element in the current representation
        """
        if len(element) == 0:
            return 1.0  # Identity element
        
        from collections import Counter
        
        # Count how many times each generator appears
        character = 1.0
        op_counts = Counter(element)
        
        for op, count in op_counts.items():
            # Find the sector for this operator
            sector_value = None
            for gen_op, (gen_type, sector) in self.generators:
                if gen_op is op:
                    sector_value = sector
                    break
            
            if sector_value is None:
                continue
            
            # Use polymorphic get_character method from the operator
            # This delegates the character computation to the symmetry class itself
            character *= op.get_character(count, sector_value, lattice=self.lattice, ns=self.ns)
        
        return character
    
    def compute_normalization(self, state: StateInt) -> float:
        """
        Compute normalization factor for a representative state in the current sector.
        
        The normalization is computed using the projection formula:
        N = sqrt(sum_{g in G} chi _k(g)^* <state|g|state>)
        
        where chi _k(g) is the character of element g in irrep k.
        
        For a state to belong to momentum sector k, it must satisfy:
        |state_k> = (1/sqrt N) sum_{g in G} chi _k(g)^* g|rep>
        
        Parameters
        ----------
        state : StateInt
            Representative state
        
        Returns
        -------
        norm : float
            Normalization factor (0.0 if state not in this sector)
        
        Algorithm
        ---------
        1. Apply all group elements to state
        2. Sum characters * phases for elements that return to same state
        3. Return sqrt of sum
        
        Physical Interpretation
        -----------------------
        If norm = 0, the state does not belong to this momentum/symmetry sector.
        If norm > 0, the state is a valid representative for this sector.
        """
        if len(self.symmetry_group) == 0:
            return 1.0
        
        projection_sum = 0.0
        
        for element in self.symmetry_group:
            new_state, intrinsic_phase  = self.apply_group_element(element, state)
            
            # Only states in the orbit contribute
            # For momentum sectors, we need the character even if state changes
            character                   = self.get_character(element)
            
            # The projection operator is P_k = (1/|G|) sum_g chi _k(g)^* g
            # We compute <state|P_k|state> = (1/|G|) sum_g chi _k(g)^* <state|g|state>
            # The overlap <state|g|state> = phase * delta _{state, g(state)}
            
            if new_state == state:
                # State is invariant under this group element
                # Contribution: chi _k(g)^* * phase
                projection_sum         += np.conj(character) * intrinsic_phase
        
        # Correct formula:
        #   N_k^2(|r>) = sum_{m=0}^{s-1} e^{ikp*m}  where s = period, p = L/period
        # 
        # We've already computed this sum above as projection_sum.
        # Normalization is sqrt of the projection sum.
        # It must be real and non-negative.
        norm = np.sqrt(abs(projection_sum))
        
        # Check if normalization is non-zero (state allowed in this sector)
        if abs(norm) < _SYM_NORM_THRESHOLD:
            return 0.0
        
        return float(norm)
    
    def check_global_symmetries(self, state: StateInt) -> bool:
        """
        Check if a state satisfies all global symmetry constraints.
        
        Parameters
        ----------
        state : StateInt
            State to check
        
        Returns
        -------
        satisfies : bool
            Whether state satisfies all global symmetries
        """
        for global_sym in self.global_symmetries:
            if not global_sym(int(state)):
                return False
        return True

    # -----------------------------------------------------
    #! Full Mapping Management - IMPORTANT FOR OUR IMPLEMENTATION
    # -----------------------------------------------------

    def generate_symmetric_basis(self, 
                                nh_full         : int, 
                                global_syms     : Optional[List[GlobalSymmetry]]    = None, 
                                state_filter    : Optional[Callable[[int], bool]]   = None, 
                                return_map      : bool                              = True,
                                chunk_size      : int                               = 65536) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the symmetric basis by finding all unique representatives
        and their normalization factors. It first identifies unique representatives
        by applying symmetry operations to all states in the full Hilbert space,
        then computes normalization factors for each representative.
        
        If return_map is True, it also builds a compact mapping structure
        for O(1) lookups of representatives and phases. First pass only finds the 
        mapping state -> representative. Then, this is transformed to indices...
        
        Parameters
        ----------
        nh_full : int
            Full Hilbert space dimension
        global_syms : List[GlobalSymmetry]
            Global symmetries to filter states
        state_filter : Callable[[int], bool]
            Additional filter for states
        return_map : bool
            Whether to build and return the compact symmetry map (Pass 2).
            If False, only representatives and norms are computed (Pass 1).
            
        Returns
        -------
        representative_list : np.ndarray
            Sorted list of representative states (reduced basis)
        representative_norms : np.ndarray
            Normalization factors for representatives
        """
        
        self.logger.info(f"Generating symmetric basis for {nh_full} states...", lvl=1, color='cyan')
        t0                      = time.time()
        g_to_pidx, phase_table  = self.build_phases_map()

        if return_map:
            # Allocate FINAL arrays only (uint32 + uint8)
            # This is the peak memory usage point (along with representative_list)        
            compact_repr_map    = np.full(nh_full, _INVALID_REPR_IDX,   dtype=_REPR_MAP_DTYPE)
            compact_phase_idx   = np.full(nh_full, _INVALID_PHASE_IDX,  dtype=_PHASE_IDX_DTYPE)
            gen_map             = True
            self.logger.info(f"Allocated compact mapping arrays. Took: {time.time() - t0:.2e}s", lvl=2, color='green')
        else:
            compact_repr_map    = None
            compact_phase_idx   = None
            gen_map             = False
        
        # ---------------------------------------------------------
        # Find Unique Representatives
        # ---------------------------------------------------------
        
        self.logger.info("Identifying unique representatives...", lvl = 2, color = 'cyan')
        chunk_size              = min(chunk_size, nh_full) # 2^16 - affordable chunk size
        representative_list     = []
        representative_norms    = []
        
        state_filter            = state_filter if state_filter is not None else (lambda x: True)
        def check_global_syms(state: int) -> bool:
            bad = False
            for g in global_syms:
                if not g(state):
                    bad = True
                    break
            return bad
        
        for start_idx in range(0, nh_full, chunk_size):
            end_idx     = min(start_idx + chunk_size, nh_full)
            repr_in     = []
            norm_in     = []
            
            # Use Python loop for finding representatives
            for state in range(start_idx, end_idx):
                
                if check_global_syms(int(state)): # if global symmetries not satisfied
                    continue
                if not state_filter(int(state)):
                    continue
                
                rep_state, phase, idx = self._find_representative(int(state))
                
                # Only add the state if it is its own representative
                # (this ensures each symmetry sector is represented once)
                if rep_state == state:
                    # Calculate normalization using character-based projection
                    n = self.compute_normalization(rep_state)
                    if abs(n) > _SYM_NORM_THRESHOLD:
                        repr_in.append(rep_state)
                        norm_in.append(n)
                
                if gen_map:
                    compact_repr_map[state]     = rep_state         # Temporary - will be converted to index later
                    compact_phase_idx[state]    = g_to_pidx[idx]    # Phase index for this group element -> phase table
            
            # End of chunk processing
            representative_list.extend(repr_in)
            representative_norms.extend(norm_in)
            
        self.logger.info(f"Identified unique representatives. Took: {time.time() - t0:.2e}s", lvl=3, color='green')

        # Create sorted representative list
        pairs                   = sorted(zip(representative_list, representative_norms))
        representative_list     = np.array([p[0] for p in pairs], dtype=np.int64)
        representative_norms    = np.array([p[1] for p in pairs], dtype=np.float64)       
        
        # Last step: convert compact_repr_map from state -> index in representative_list
        if gen_map:
            self.logger.info("Building compact representative index map...", lvl=2, color='cyan')
            compact_repr_map    = _compact_convert_states_to_idx(full_array=compact_repr_map, repr_list=representative_list)
            self.logger.info(f"With built compact representative index map. Took: {time.time() - t0:.2e}s", lvl=4, color='green')
        
        # Create CompactSymmetryData
        self._compact_data = CompactSymmetryData(
            repr_map            = compact_repr_map,
            phase_idx           = compact_phase_idx,
            phase_table         = phase_table,
            normalization       = representative_norms,
            representative_list = representative_list
        )        
        
        self.logger.info(
            f"Built compact symmetry map: {nh_full} states, "
            f"{len(representative_list)} representatives, "
            f"{len(phase_table)} distinct phases, "
            f"~{(nh_full * 5 + len(phase_table) * 16 + len(representative_list) * 16) / 1024:.1f} KB, "
            f"in {time.time() - t0:.3e}s"
        )
                
        return representative_list, representative_norms

    def build_phases_map(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the phases mapping for the symmetry group.
        """
                
        # Prepare phase table logic
        # We pre-compute the phase index for each group element to avoid looking it up per state
        # Phase = character(g) * intrinsic_phase (assumed 1.0)
        group_chars         = np.array([self.get_character(g) for g in self.symmetry_group], dtype=np.complex128)
        phase_tolerance     = _SYM_NORM_THRESHOLD
        unique_phases       = []
        g_to_pidx           = np.zeros(len(self.symmetry_group), dtype=_PHASE_IDX_DTYPE)
        pidx_max            = np.iinfo(_PHASE_IDX_DTYPE).max
        
        for g_idx, char in enumerate(group_chars):
            # Find in unique phases
            phase           = char
            # phase           = np.conj(char)
            found_idx       = None
            for pidx, existing_phase in enumerate(unique_phases):
                if abs(phase - existing_phase) < phase_tolerance:
                    found_idx = pidx
                    break
            if found_idx is None:
                if len(unique_phases) >= pidx_max:
                    raise MemoryError(f"Exceeded maximum number of unique phases ({pidx_max}) for phase indexing.")
                else:
                    found_idx   = len(unique_phases)
                    unique_phases.append(phase)
            g_to_pidx[g_idx] = found_idx
            
        # Convert phase list to array
        phase_table = np.array(unique_phases, dtype=np.complex128)
        return g_to_pidx, phase_table

    # -----------------------------------------------------
    #! JIT-compatible Projector for NQS
    # -----------------------------------------------------

    def get_jittable_projector(self):
        """
        Returns a JAX-compatible callable that projects a state onto the symmetry sector.
        
        The returned function `projector(state)`:
        - Input: `state` (batch, ns) or (ns,)
        - Output: `(orbit_states, orbit_weights)`
            - `orbit_states`: (batch, group_size, ns)
            - `orbit_weights`: (batch, group_size)
            
        This unrolls the loop over group elements, making it JIT-compatible.
        It uses `apply_group_element` which must be JAX-traceable.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for get_jittable_projector.")
            
        # Capture group elements and self
        group_elements = self.symmetry_group
        
        def projector(state):
            # Ensure state is at least 2D (batch, ns)
            # If 1D, expand dims?
            is_1d = state.ndim == 1
            if is_1d:
                state = jnp.expand_dims(state, 0)
                
            orbit_states    = []
            orbit_weights   = []
            
            for elem in group_elements:
                # Apply group element (unrolled loop)
                # We need to broadcast over batch if apply_group_element doesn't handle it
                # But apply_group_element calls op.apply_jax.
                # If op.apply_jax handles batch (which it should), we are fine.
                
                # Apply symmetry: s' = g(s), phase = phi_g(s)
                new_state, phase    = self.apply_group_element(elem, state)
                
                # Character: chi(g)
                # Character is usually scalar for 1D irreps
                char                = self.get_character(elem)
                
                # Weight = char * conj(phase)
                # Phase might be (batch,), char is scalar
                weight              = char * jnp.conj(phase)
                
                # Broadcast weight to (batch,)
                weight              = jnp.broadcast_to(weight, (state.shape[0],))
                
                orbit_states.append(new_state)
                orbit_weights.append(weight)
            
            # Stack results: (batch, group_size, ns)
            states_stack        = jnp.stack(orbit_states, axis=1)
            weights_stack       = jnp.stack(orbit_weights, axis=1)
            
            if is_1d:
                states_stack    = states_stack[0]
                weights_stack   = weights_stack[0]
                
            return states_stack, weights_stack
            
        return projector

####################################################################################################
#! Utility Functions
####################################################################################################

def create_symmetry_container_from_specs(
    ns                  : int,
    generator_specs     : List[SymmetrySpec],
    global_syms         : List[GlobalSymmetry],
    lattice             : Optional[Lattice] = None,
    nhl                 : int               = 2,
    backend             : str               = 'default',
    build_group         : bool              = True
) -> SymmetryContainer:
    """
    Factory function to create and initialize a SymmetryContainer.
    
    Parameters
    ----------
    ns : int
        Number of sites
    generator_specs : List[SymmetrySpec]
        List of (generator_type, sector) tuples
    global_syms : List[GlobalSymmetry]
        List of global symmetries
    lattice : Optional[Lattice]
        Lattice structure
    nhl : int
        Local Hilbert space dimension
    backend : str
        Computation backend
    build_group : bool
        Whether to build symmetry group immediately
    
    Returns
    -------
    container : SymmetryContainer
        Initialized container
        
    Note
    ----
    Compact symmetry data (for O(1) JIT lookups) is built separately by 
    HilbertSpace after representative generation via build_compact_map().
    """
    container = SymmetryContainer(ns=ns, lattice=lattice, nhl=nhl, backend=backend)
    
    # Add global symmetries
    for gsym in global_syms:
        container.add_global_symmetry(gsym)
    
    # Create operator instances first
    operators_with_specs: List[Tuple[SymmetryOperator, SymmetrySpec]] = []
    for gen_type, sector in generator_specs:
        operator = _create_symmetry_operator(gen_type, sector, lattice, ns, nhl)
        if operator is not None:
            operators_with_specs.append((operator, (gen_type, sector)))
    
    # Filter for compatibility
    compat      = SymmetryCompatibility(ns, nhl, lattice)
    filtered    = compat.get_compatible_operators(operators_with_specs, global_syms)
    
    # Add filtered operators to container
    for operator, (gen_type, sector) in filtered:
        container.add_generator(gen_type, sector, operator)
    
    # Build group if requested
    if build_group:
        container.build_group()
    
    return container

####################################################################################################
#! Symmetry Operator Factory
####################################################################################################

def _create_symmetry_operator(
    gen_type        : SymmetryGenerators,
    sector          : Union[int, float, complex],
    lattice         : Optional[Lattice],
    ns              : int,
    nhl             : int) -> Optional[SymmetryOperator]:
    """
    Factory to create symmetry operator instances.
    
    This function imports the appropriate symmetry class and instantiates it.
    """
    try:
        # Translation symmetries (different directions)
        if gen_type in (SymmetryGenerators.Translation_x, 
                       SymmetryGenerators.Translation_y, 
                       SymmetryGenerators.Translation_z):
            from QES.Algebra.Symmetries.translation import TranslationSymmetry
            # Extract direction from enum name
            direction_map = {
                SymmetryGenerators.Translation_x: 'x',
                SymmetryGenerators.Translation_y: 'y',
                SymmetryGenerators.Translation_z: 'z'
            }
            direction = direction_map[gen_type]
            return TranslationSymmetry(lattice=lattice, sector=sector, ns=ns, direction=direction)
        
        # Reflection
        elif gen_type == SymmetryGenerators.Reflection:
            from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
            return ReflectionSymmetry(lattice=lattice, sector=sector, ns=ns)
        
        # Inversion (general spatial inversion for any lattice/dimension)
        elif gen_type == SymmetryGenerators.Inversion:
            from QES.Algebra.Symmetries.inversion import InversionSymmetry
            return InversionSymmetry(lattice=lattice, sector=sector, ns=ns, base=nhl)
        
        # Parity symmetries
        elif gen_type == SymmetryGenerators.ParityX:
            from QES.Algebra.Symmetries.parity import ParitySymmetry
            return ParitySymmetry(axis='x', sector=sector, ns=ns, nhl=nhl)
        
        elif gen_type == SymmetryGenerators.ParityY:
            from QES.Algebra.Symmetries.parity import ParitySymmetry
            return ParitySymmetry(axis='y', sector=sector, ns=ns, nhl=nhl)
        
        elif gen_type == SymmetryGenerators.ParityZ:
            from QES.Algebra.Symmetries.parity import ParitySymmetry
            return ParitySymmetry(axis='z', sector=sector, ns=ns, nhl=nhl)
        
        else:
            from QES.general_python.common.flog import get_global_logger
            get_global_logger().warning(f"Unknown symmetry type: {gen_type}")
            return None
    
    except ImportError as e:
        from QES.general_python.common.flog import get_global_logger
        get_global_logger().error(f"Failed to import symmetry {gen_type}: {e}")
        return None

####################################################################################################
#! Symmetry Parsing Utilities
####################################################################################################

def parse_symmetry_spec(sym_name: str, sym_value: Union[dict, int, float, complex]) -> List[Tuple]:
    """
    Parse symmetry specification from string name and value to (SymmetryGenerator, sector) tuples.
    
    Parameters
    ----------
    sym_name : str
        Symmetry name (e.g., 'translation', 'parity', 'reflection')
    sym_value : Union[dict, int, float, complex]
        Either a sector value (for simple symmetries) or dict with parameters
        
    Returns
    -------
    List[Tuple[SymmetryGenerators, sector_value]]
        List of symmetry generator specifications
    """
    
    try:
        from QES.Algebra.Operator.operator import SymmetryGenerators
    except ImportError:
        raise ImportError("SymmetryGenerators enum not found - ensure QES.Algebra.Operator.operator is available.")
    
    sym_name_lower  = sym_name.lower().replace('_', '').replace('-', '')
    specs           = []
    
    # Translation symmetry (can have multiple directions)
    if sym_name_lower in ['translation', 'translations', 'trans', 'momentum']:
        if isinstance(sym_value, dict):
            # Dict with kx, ky, kz sectors
            if 'kx' in sym_value or 'k_x' in sym_value:
                kx = sym_value.get('kx', sym_value.get('k_x'))
                specs.append((SymmetryGenerators.Translation_x, kx))
            if 'ky' in sym_value or 'k_y' in sym_value:
                ky = sym_value.get('ky', sym_value.get('k_y'))
                specs.append((SymmetryGenerators.Translation_y, ky))
            if 'kz' in sym_value or 'k_z' in sym_value:
                kz = sym_value.get('kz', sym_value.get('k_z'))
                specs.append((SymmetryGenerators.Translation_z, kz))
        else:
            # Single value assumed for x-direction
            specs.append((SymmetryGenerators.Translation_x, sym_value))
    
    # Parity symmetry
    elif sym_name_lower in ['parity', 'parityx', 'parityy', 'parityz', 'spin']:
        if isinstance(sym_value, dict):
            axis    = sym_value.get('axis', 'z').lower()
            sector  = sym_value.get('sector', 1)
        else:
            axis    = 'z'  # Default to z-parity
            sector  = sym_value
        
        if axis == 'x':
            specs.append((SymmetryGenerators.ParityX, sector))
        elif axis == 'y':
            specs.append((SymmetryGenerators.ParityY, sector))
        else: # 'z' or default
            specs.append((SymmetryGenerators.ParityZ, sector))
    
    # Reflection symmetry
    elif sym_name_lower in ['reflection', 'reflect', 'mirror']:
        if isinstance(sym_value, dict):
            sector = sym_value.get('sector', 1)
        else:
            sector = sym_value
        specs.append((SymmetryGenerators.Reflection, sector))
    
    # Inversion symmetry (general spatial inversion for any lattice)
    elif sym_name_lower in ['inversion', 'inv', 'spatial', 'spatialinversion']:
        if isinstance(sym_value, dict):
            sector = sym_value.get('sector', 1)
        else:
            sector = sym_value
        specs.append((SymmetryGenerators.Inversion, sector))
    
    # Fermion parity
    elif sym_name_lower in ['fermionparity', 'fermion', 'fparity']:
        if isinstance(sym_value, dict):
            sector = sym_value.get('sector', 1)
        else:
            sector = sym_value
        specs.append((SymmetryGenerators.FermionParity, sector))
    
    # Particle-hole symmetry
    elif sym_name_lower in ['particlehole', 'ph', 'chargeconjugation']:
        if isinstance(sym_value, dict):
            sector = sym_value.get('sector', 1)
        else:
            sector = sym_value
        specs.append((SymmetryGenerators.ParticleHole, sector))
    
    # Time reversal
    elif sym_name_lower in ['timereversal', 'tr', 'time']:
        if isinstance(sym_value, dict):
            sector = sym_value.get('sector', 1)
        else:
            sector = sym_value
        specs.append((SymmetryGenerators.TimeReversal, sector))
    
    else:
        raise ValueError(f"Unknown symmetry name: '{sym_name}'. "
                        f"Supported: translation, parity, reflection, inversion, "
                        f"fermion_parity, particle_hole, time_reversal")
    
    return specs

def normalize_symmetry_generators(sym_gen: Union[dict, list, None]) -> List[Tuple]:
    """
    Normalize symmetry generator input to list of (SymmetryGenerator, sector) tuples.
    
    Accepts:
    - None : No symmetries
    - dict : {'symmetry_name': value_or_dict, ...}
    - list : [SymmetryOperator instances] or [(SymmetryGenerator, sector), ...]
    
    Returns
    -------
    List[Tuple[SymmetryGenerators, sector]]
        Normalized list of symmetry specifications
    """
    try:
        from QES.Algebra.Operator.operator  import SymmetryGenerators
        from QES.Algebra.Symmetries.base    import SymmetryOperator
    except ImportError:
        raise ImportError("SymmetryGenerators or SymmetryOperator not found - ensure QES.Algebra.Operator.operator and QES.Algebra.Symmetries.base are available.")
    
    if sym_gen is None:
        return []
    
    # Already a list of tuples (SymmetryGenerators, sector)
    if isinstance(sym_gen, list):
        if len(sym_gen) == 0:
            return []
        
        # Check if already in correct format
        first_elem = sym_gen[0]
        if isinstance(first_elem, tuple) and len(first_elem) == 2:
            if isinstance(first_elem[0], SymmetryGenerators):
                return sym_gen  # Already normalized
            
            # Check if it's a string-based tuple format like ('parity', 0)
            if isinstance(first_elem[0], str):
                # Parse all string-based tuples
                specs = []
                for sym_name, sym_value in sym_gen:
                    parsed = parse_symmetry_spec(sym_name, sym_value)
                    specs.extend(parsed)
                return specs
        
        # List of SymmetryOperator instances - extract their types
        # This is less common but supported for backward compatibility
        if all(isinstance(s, SymmetryOperator) for s in sym_gen):
            # Convert to specs (this requires operator instances to have type info)
            # For now, assume they're already properly formatted
            return sym_gen
        
        raise ValueError("List format for sym_gen must be [(SymmetryGenerators, sector), ...] "
                        "or [('symmetry_name', sector), ...] or [SymmetryOperator instances]")
    
    # Dictionary format: parse each entry
    if isinstance(sym_gen, dict):
        specs = []
        for sym_name, sym_value in sym_gen.items():
            # Check if value is already a SymmetryOperator instance
            if isinstance(sym_value, SymmetryOperator):
                # Extract type and sector from operator
                # Assume operator has these attributes
                sym_type    = getattr(sym_value, 'symmetry_type', None)
                sector      = getattr(sym_value, 'sector', None)
                if sym_type and sector is not None:
                    specs.append((sym_type, sector))
                continue
            
            # Parse string name to SymmetryGenerators
            parsed = parse_symmetry_spec(sym_name, sym_value)
            specs.extend(parsed)
        
        return specs
    
    raise ValueError(f"sym_gen must be dict, list, or None, got {type(sym_gen)}")

####################################################################################################
#! End of file
####################################################################################################
