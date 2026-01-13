"""
Base Hilbert space class for quantum many-body systems.

This module provides the abstract base class for Hilbert spaces, handling:
- Basic system properties (Ns, lattice, local dimensions)
- Backend configuration
- Symmetry container management
- Logging and debugging utilities

---------------------------------------------------
File    : QES/Algebra/hilbert_base.py
Author  : Maksymilian Kliczkowski
Date    : 2025-12-08
Version : 1.0.0
---------------------------------------------------
"""
from    __future__ import annotations

import  numpy       as np
from    abc         import ABC
from    dataclasses import dataclass
from    typing      import Union, Optional, List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from QES.general_python.common.flog             import Logger
    from QES.general_python.lattices.lattice        import Lattice, LatticeDirection
    from QES.Algebra.Hilbert.hilbert_local          import LocalSpace, StateTypes
    from QES.Algebra.globals                        import GlobalSymmetry
    from QES.Algebra.Symmetries.symmetry_container  import SymmetryContainer, CompactSymmetryData

# ------------------------------------------------------------------------------------------------------
#! Base Hilbert Space Class
# ------------------------------------------------------------------------------------------------------

class BaseHilbertSpace(ABC):
    """
    Abstract base class for Hilbert spaces.
    
    Handles common functionality like backend setup, logging, and 
    symmetry container management.
    """
    
    @dataclass
    class SymBasicInfo:
        num_operators   : int   = 0
        num_gens        : int   = 0
        num_sectors     : int   = 0
        num_states      : int   = 0

    def __init__(self,
                    ns              : Optional[int],
                    lattice         : Optional['Lattice'],
                    local_space     : Optional['LocalSpace'],
                    backend         : str,
                    state_type      : Union[str, type],
                    dtype           : np.dtype,
                    *,
                    state_filter    : Optional[callable]    = None,
                    boundary_flux   : Optional[float]       = None,
                    logger          : Optional['Logger']    = None,
                    **kwargs
                ):
        """
        Initialize the base Hilbert space.
        
        Parameters
        ----------
        ns : int
            Number of sites
        lattice : Lattice
            Lattice structure
        local_space : LocalSpace
            Local Hilbert space configuration
        backend : str
            Computation backend ('numpy', 'jax', etc.)
        state_type : str or type
            State representation type
        dtype : np.dtype
            Data type for arrays
        logger : Logger
            Logger instance
        state_filter : callable
            Predicate to filter basis states
        boundary_flux : float
            Boundary flux for lattice
        kwargs : dict
            Additional arguments
        """
        
        self._logger                                            = self._check_logger(logger)
        self._backend, self._backend_str, self._state_type      = self.reset_backend(backend, state_type)
        self._dtype                                             = dtype if dtype is not None else self._backend.float64
        
        self._ns                                                = ns
        self._lattice                                           = lattice
        self._boundary_flux     = boundary_flux
        if self._lattice is not None and boundary_flux is not None:
            self._lattice.flux  = boundary_flux
        
        self._local_space                                       = local_space
        self._threadnum                                         = kwargs.get('threadnum', 1) # number of threads to use
        
        # Symmetry container (initialized by subclasses)
        self._sym_group                                         = []                            
        self._sym_container     : Optional['SymmetryContainer'] = None
        self._sym_basic_info                                    = self.SymBasicInfo()
        self._global_syms       : List['GlobalSymmetry']        = []
        self._has_complex_symmetries                            = False
        
        # Mapping properties (managed by subclasses but defined here for type safety)
        self.representative_list                                = None
        self.representative_norms                               = None
        self._nh                : int                           = 0
        self._nhfull            : Union[int, float]             = 0
        
        # State filtering - predicate to filter basis states
        self._state_filter                                      = state_filter
        
    # --------------------------------------------------------------------------------------------------
    #! Backend and Logging
    # --------------------------------------------------------------------------------------------------

    def _check_logger(self, logger: Optional['Logger']) -> 'Logger':
        ''' Check and return the logger instance '''
        if logger is None:
            from QES.general_python.common.flog import get_global_logger
            return get_global_logger()
        return logger

    def _log(self, msg: str, log: Union[int, str] = 'info', lvl: int = 0, color: str = "white", append_msg=True):
        """Log a message."""
        if self._logger is None:
            return
        
        try:
            from QES.general_python.common.flog import Logger
        except ImportError:
            return
        
        if isinstance(log, str):
            log = Logger.LEVELS_R.get(log, 20) # Default to INFO level if not found
            
        if append_msg:
            msg = f"[{self.__class__.__name__}] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)

    @staticmethod
    def reset_backend(backend: str, state_type: Union[str, type]):
        """Reset the backend for the Hilbert space."""
        if isinstance(backend, str):
            from QES.general_python.algebra.utils import get_backend
            _backend_str = backend
            _backend = get_backend(backend)
        else:
            _backend_str = 'np' if backend == np else 'jax'
            _backend = backend
        
        statetype = BaseHilbertSpace.reset_statetype(state_type, _backend)
        return _backend, _backend_str, statetype

    @staticmethod
    def reset_statetype(state_type: Union[str, type, object], backend):
        """Reset the state type for the Hilbert space."""
        if state_type is int:
            return int
            
        # Handle Enum or string-like objects
        if hasattr(state_type, "value"):
            s_val = str(state_type.value)
        else:
            s_val = str(state_type)

        if s_val.lower() in ("integer", "int"):
            return int
            
        return backend.array

    # --------------------------------------------------------------------------------------------------
    #! Properties
    # --------------------------------------------------------------------------------------------------

    @property
    def ns(self)                -> int:                     return self._ns
    @property
    def lattice(self)           -> Optional['Lattice']:     return self._lattice
    @property
    def local_space(self)       -> Optional['LocalSpace']:  return self._local_space
    @property
    def backend(self):          return self._backend
    @property
    def backend_str(self)       -> str:                     return self._backend_str
    @property
    def logger(self)            -> 'Logger':                return self._logger
    @property
    def dtype(self)             -> np.dtype:                return self._dtype

    @property
    def dim(self)               -> int:                     return self._nh
    @property
    def nh(self)                -> int:                     return self._nh
    @property
    def nhfull(self)            -> Union[int, float]:       return self._nhfull

    # --------------------------------------------------------------------------------------------------
    #! Symmetry Properties
    # --------------------------------------------------------------------------------------------------

    @property
    def sym_container(self) -> Optional['SymmetryContainer']:
        return self._sym_container

    @property
    def sym_group(self):
        """Return the symmetry group from the container."""
        if self._sym_container:
            return self._sym_container.symmetry_group
        return []

    @property
    def check_global_symmetry(self) -> bool:
        """Check if there are any global symmetries."""
        return len(self._global_syms) > 0

    @property
    def has_sym_generators(self) -> bool:
        """
        Returns True if symmetry generators are defined (even without basis reduction).
        
        Use this property when you need to know if symmetry operations are available
        for applying to states (e.g., for NQS symmetry projections).
        """
        return (self._sym_container is not None and len(self._sym_container.generators) > 0)
    
    @property  
    def has_sym_reduction(self) -> bool:
        """
        Returns True if the Hilbert space basis has been reduced by symmetries.
        
        This is the stricter check - it returns True only when representative 
        states have been computed and the effective dimension is smaller than 
        the full Hilbert space dimension.
        """
        return self._nh != self._nhfull

    @property
    def has_complex_symmetries(self) -> bool:
        """Check if symmetry phases are complex."""
        return self._has_complex_symmetries

    @property
    def has_compact_symmetry_data(self) -> bool:
        """Check if compact symmetry data is available."""
        return self._sym_container is not None and self._sym_container.has_compact_data

    @property
    def compact_symmetry_data(self) -> Optional['CompactSymmetryData']:
        """Get the CompactSymmetryData structure."""
        if self._sym_container is None:
            return None
        return self._sym_container.compact_data

    @property
    def repr_idx(self) -> Optional[np.ndarray]:
        """Representative index array: state -> representative index (uint32)."""
        if self._sym_container is not None and self._sym_container.has_compact_data:
            return self._sym_container.compact_data.repr_map
        return None
    
    @property
    def repr_phase(self) -> Optional[np.ndarray]:
        """Symmetry phases for all states."""
        if self._sym_container is not None and self._sym_container.has_compact_data:
            cd = self._sym_container.compact_data
            return cd.phase_table[cd.phase_idx]
        return None
    
    @property
    def normalization(self) -> Optional[np.ndarray]:
        """Normalization factors for representatives."""
        if self._sym_container is not None and self._sym_container.has_compact_data:
            return self._sym_container.compact_data.normalization
        return self.representative_norms

    # --------------------------------------------------------------------------------------------------
    #! Symmetry Methods (Delegated)
    # --------------------------------------------------------------------------------------------------

    def find_sym_repr(self, state, nb: float = 1.0) -> Tuple[int, Union[float, complex]]:
        """Find representative state using symmetry container."""
        if self._sym_container is None:
            return state, 1.0
        return self._sym_container.find_representative(state, nb)

    def find_sym_norm(self, state) -> Union[float, complex]:
        """Compute normalization factor using symmetry container."""
        if self._sym_container is None:
            return 1.0
        return self._sym_container.compute_normalization(state)

    # Aliases
    def find_repr(self, state, nb: float = 1.0):    return self.find_sym_repr(state, nb)
    def find_norm(self, state):                     return self.find_sym_norm(state)
    
    def norm(self, state):
        """Get normalization for a representative state index."""
        if self.representative_norms is not None and state < len(self.representative_norms):
            return self.representative_norms[state]
        return self.find_norm(state)

    def get_sym_info(self) -> str:
        """Create information string about symmetries."""
        tmp = ""
        if self._sym_container is not None and self._sym_container.generators:
            for op, (gen_type, sector) in self._sym_container.generators:
                tmp += f"{gen_type}={sector},"
                
        if self.check_global_symmetry:
            for g in self._global_syms:
                tmp += f"{g.get_name_str() if hasattr(g, 'get_name_str') else g.name}={g.get_val() if hasattr(g, 'get_val') else ''},"
        
        return tmp[:-1] if tmp else ""

    # --------------------------------------------------------------------------------------------------
    #! Directory Naming
    # --------------------------------------------------------------------------------------------------

    @property
    def symmetry_directory_name(self) -> str:
        """Return filesystem-safe string of symmetry sectors."""
        if self._sym_container is None or not self._sym_container.generators:
            return ""
        
        names = []
        for item in self._sym_container.generators:
            # item is (operator, spec)
            if isinstance(item, tuple) and len(item) >= 1:
                op = item[0]
            else:
                op = item
            
            if hasattr(op, 'directory_name'):
                names.append(op.directory_name)
            elif hasattr(op, 'sector'):
                from QES.Algebra.Symmetries.base import SymmetryOperator
                sector_str = SymmetryOperator._sector_to_str(op.sector)
                names.append(f"{op.__class__.__name__.lower()}_{sector_str}")
        
        return (",".join(names)).replace('_', "=") if names else "nosym"

    @property
    def lattice_directory_name(self) -> str:
        """Return filesystem-safe lattice string."""
        if self._lattice is not None:
            return str(self._lattice)
        elif self._ns is not None and self._ns > 0:
            return f"n={self._ns}"
        return "unknown"

    @property
    def full_directory_name(self) -> str:
        """Return complete directory name."""
        return f"{self.lattice_directory_name}/{self.symmetry_directory_name}"

    # --------------------------------------------------------------------------------------------------
    #! Magic Methods
    # --------------------------------------------------------------------------------------------------

    def __str__(self):
        info = f"HilbertSpace: Ns={self._ns}, Nh={self._nh}"
        if self._local_space:
            info += f", Local={self._local_space}"
        sym_info = self.get_sym_info()
        if sym_info:
            info += f", Symmetries=[{sym_info}]"
        return info

    def __repr__(self):
        return self.__str__()

    def transform_to_reduced_space(self, vec_full: np.ndarray) -> np.ndarray:
        """
        Transform vector(s) from full Hilbert space to reduced symmetry sector.
        
        Parameters
        ----------
        vec_full : np.ndarray
            Vector(s) in full Hilbert space. Shape (N_full,) or (N_full, N_batch).
            
        Returns
        -------
        np.ndarray
            Vector(s) in reduced Hilbert space. Shape (N_reduced,) or (N_reduced, N_batch).
        """
        if not self.has_compact_symmetry_data:
            # If no symmetries (or just global constraints), check if dimensions match
            if self.nh == self.nhfull:
                return vec_full
            else:
                raise NotImplementedError("Transform without compact symmetry data not implemented.")
        
        cd          = self.compact_symmetry_data
        repr_map    = cd.repr_map
        phase_idx   = cd.phase_idx
        phase_table = cd.phase_table
        norms       = cd.normalization
        
        # Check dimensions
        if vec_full.shape[0] != len(repr_map):
            raise ValueError(f"Vector dimension {vec_full.shape[0]} does not match full Hilbert space {len(repr_map)}")
            
        # Initialize reduced vector
        input_shape = vec_full.shape
        reduced_shape = (self.dim,) + input_shape[1:]
        
        vec_red = np.zeros(reduced_shape, dtype=self.dtype)
        
        # Ensure complex if needed (phases are complex)
        if np.iscomplexobj(vec_full) or self.has_complex_symmetries:
            vec_red = vec_red.astype(np.complex128)
            
        # 1. Find valid indices (states that belong to this sector)
        # _INVALID_REPR_IDX = 0xFFFFFFFF (uint32 max)
        valid_mask = (repr_map != 0xFFFFFFFF)
        
        if not np.any(valid_mask):
            return vec_red
            
        valid_indices = np.where(valid_mask)[0]
        target_k      = repr_map[valid_indices]
        phases        = phase_table[phase_idx[valid_indices]]
        
        # Handle broadcasting for batch dimension
        # We sum c_s * conj(chi(g_s))
        if vec_full.ndim > 1:
            # (N_valid, 1) * (N_valid, Batch) -> (N_valid, Batch)
            contributions = np.conj(phases)[:, np.newaxis] * vec_full[valid_indices]
        else:
            # (N_valid,) * (N_valid,) -> (N_valid,)
            contributions = np.conj(phases) * vec_full[valid_indices]
            
        # Accumulate: vec_red[k] += contribution
        np.add.at(vec_red, target_k, contributions)
        
        # Normalize: vec_red[k] /= N_k
        if vec_full.ndim > 1:
            vec_red /= norms[:, np.newaxis]
        else:
            vec_red /= norms
        
        return vec_red

    # --------------------------------------------------------------------------------------------------
    #! Help
    # --------------------------------------------------------------------------------------------------
    
    @staticmethod
    def help(topic: Optional[str] = None, verbose: bool = True) -> Optional[str]:
        """
        Display comprehensive help about HilbertSpace usage and capabilities.
        
        Parameters
        ----------
        topic : str, optional
            Specific topic to get help on. Options:
            - None or 'overview' : General overview and quick start
            - 'symmetries'       : Symmetry handling and momentum sectors
            - 'operators'        : Operator construction and matrix building
            - 'compact'          : Compact symmetry data structure (O(1) lookups)
            - 'properties'       : Available properties and their meanings
            - 'examples'         : Code examples for common use cases
        verbose : bool
            If True, print the help. If False, return as string.
        
        Returns
        -------
        str or None
            Help text if verbose=False, else prints and returns None.
        
        Examples
        --------
        >>> HilbertSpace.help()                    # Full overview
        >>> HilbertSpace.help('symmetries')        # Symmetry-specific help
        >>> HilbertSpace.help('examples')          # Code examples
        """
        help_texts = {
            'overview': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                          HILBERTSPACE - OVERVIEW                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

HilbertSpace class represents a quantum many-body or single-particle Hilbert space
with support for symmetry reduction, efficient matrix construction, and
multiple computational backends (NumPy, JAX).

QUICK START
-----------
    from QES.Algebra.hilbert import HilbertSpace
    from QES.general_python.lattices.square import SquareLattice
    from QES.general_python.lattices.lattice import LatticeBC
    
    # Create a lattice
    lattice = SquareLattice(dim=1, lx=6, bc=LatticeBC.PBC)
    
    # Simple Hilbert space (no symmetries)
    hilbert = HilbertSpace(lattice=lattice)
    print(f"Dimension: {hilbert.dim}")  # 2^6 = 64
    
    # With symmetries (translation symmetry, k=0 sector)
    from QES.Algebra.Operator.operator import SymmetryGenerators
    hilbert_sym = HilbertSpace(
        lattice=lattice,
        sym_gen=[(SymmetryGenerators.Translation_x, 0)],  # k=0
        gen_mapping=True
    )
    print(f"Reduced dimension: {hilbert_sym.dim}")  # ~14 states

KEY FEATURES
------------
• Many-body & quadratic (single-particle) modes
• Symmetry reduction: translation, reflection, parity, particle number
• Memory-efficient compact symmetry data (~5 bytes/state)
• O(1) JIT-compiled lookups for matrix element computation
• Multiple backends: NumPy, JAX (GPU-accelerated)

MAIN PROPERTIES
---------------
    .dim / .Nh              : Reduced Hilbert space dimension
    .nhfull                 : Full Hilbert space dimension (before symmetry reduction)
    .ns                     : Number of sites
    .representative_list    : Array of representative state integers
    .normalization          : Normalization factors for representatives
    .compact_symmetry_data  : CompactSymmetryData for O(1) lookups

Use HilbertSpace.help('symmetries') for symmetry details.
Use HilbertSpace.help('examples')   for more code examples.
''',

            'symmetries': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                     HILBERTSPACE - SYMMETRY HANDLING                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SUPPORTED SYMMETRIES
--------------------
1. Translation (momentum sectors):
    - SymmetryGenerators.Translation_x/y/z
    - Sector value = momentum quantum number k (0, 1, ..., Lx*Ly*Lz-1)

2. Reflection (spatial parity):
    - SymmetryGenerators.Reflection
    - Sector value = +-1

3. Parity (spin/particle flips):
    - SymmetryGenerators.ParityX/Y/Z
    - Sector value = +-1 or +-i

4. Inversion (sites inversion):
    - SymmetryGenerators.Inversion
    - Sector value = +-1
    
5. Global symmetries (U(1) conservation):
    - Particle number conservation (fixed N)
    - Magnetization conservation (fixed Sz)
   
Others...

USAGE
-----
    from QES.Algebra.Operator.operator import SymmetryGenerators
    
    # Translation symmetry in k=0 sector
    sym_gen = [(SymmetryGenerators.Translation_x, 0)]
    
    # Multiple symmetries
    sym_gen = [
        (SymmetryGenerators.Translation_x, 0),
        (SymmetryGenerators.Reflection, 1),
    ]
    
    hilbert = HilbertSpace(
        lattice=lattice,
        sym_gen=sym_gen,
        gen_mapping=True  # Enable compact map
    )

----------------------------------------
COMPACT SYMMETRY DATA (Memory Efficient)
----------------------------------------
When symmetries are present, a CompactSymmetryData structure is built:
    
    cd = hilbert.compact_symmetry_data
    
    • cd.repr_map       : uint32[nh_full]           - state -> repr index (O(1) lookup!)
    • cd.phase_idx      : uint8[nh_full]            - state -> phase table index
    • cd.phase_table    : complex128[~group_order]  - distinct phases
    • cd.normalization  : float64[n_repr]           - normalization factors
    
Memory: ~5 bytes/state

--------------------
JIT-COMPILED HELPERS
--------------------
    from QES.Algebra.Symmetries import _compact_get_sym_factor
    
    # O(1) symmetry factor lookup for matrix elements
    idx, sym_factor = _compact_get_sym_factor(
        new_state, k, 
        cd.repr_map, cd.phase_idx, cd.phase_table, cd.normalization
    )
''',

            'compact': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPACT SYMMETRY DATA STRUCTURE                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

The CompactSymmetryData is a memory-efficient structure for O(1) symmetry lookups.

---------
STRUCTURE
---------
    @dataclass
    class CompactSymmetryData:
        repr_map: np.ndarray            # uint32[nh_full]           - state -> repr index
        phase_idx: np.ndarray           # uint8[nh_full]            - state -> phase index
        phase_table: np.ndarray         # complex128[n]             - distinct phases
        normalization: np.ndarray       # float64[n_repr]           - per-representative
        representative_list: np.ndarray # int64[n_repr]             - repr state values

------------
MEMORY USAGE
------------
    Per state: 4 bytes (uint32) + 1 byte (uint8) = 5 bytes
    
    For 2^20 states (~1M):
    - Compact: ~5 MB

-----
USAGE
-----
    cd = hilbert.compact_symmetry_data
    
    # Check if state is in this sector
    if cd.repr_map[state] != 0xFFFFFFFF:
        idx = cd.repr_map[state]
        phase = cd.phase_table[cd.phase_idx[state]]
        norm = cd.normalization[idx]
        repr_state = cd.representative_list[idx]

-------------------------------------
JIT FUNCTIONS (numba.njit compatible)
-------------------------------------
    from QES.Algebra.Symmetries import (
        _compact_get_sym_factor,   # (state, k, ...) -> (idx, sym_factor)
        _compact_is_in_sector,     # (state, repr_map) -> bool
        _compact_get_repr_idx,     # (state, repr_map) -> int
        _compact_get_phase,        # (state, phase_idx, phase_table) -> complex
    )
''',

            'properties': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                      HILBERTSPACE - PROPERTIES                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

----------
DIMENSIONS
----------
    .dim / .Nh / .dimension     : Reduced Hilbert space dimension
    .nhfull / .Nhfull / .full   : Full dimension (before symmetry reduction)
    .ns                         : Number of sites

-------------
SYMMETRY DATA
-------------
    .representative_list        : int64[n_repr]         - representative states
    .representative_norms       : float64[n_repr]       - normalization factors
    .normalization              : Alias for representative_norms
    .compact_symmetry_data      : CompactSymmetryData structure (primary)
    .has_compact_symmetry_data  : bool                  - whether compact data exists
    .repr_idx                   : uint32[nh_full]       - state -> repr index
    .repr_phase                 : complex128[nh_full]   - symmetry phases

-----------
LOCAL SPACE
-----------
    .local / .local_space       : Local Hilbert space (spin-1/2, fermion, etc.)
    .list_local_operators()     : Available local operator names

SYSTEM TYPE
-----------
    .is_many_body / .many_body  : bool - many-body system
    .is_quadratic / .quadratic  : bool - single-particle (quadratic)
    .particle_conserving        : bool - conserves particle number

ACCESS STATES
-------------
    hilbert[k]                  : Get k-th basis state (representative)
    hilbert.state(k)            : Same as above
    iter(hilbert)               : Iterate over all representatives
''',

            'examples': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                      HILBERTSPACE - EXAMPLES                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

--------------
1. BASIC SETUP
--------------
    from QES.Algebra.hilbert import HilbertSpace
    from QES.general_python.lattices.square import SquareLattice
    from QES.general_python.lattices.lattice import LatticeBC
    
    lattice = SquareLattice(dim=1, lx=8, bc=LatticeBC.PBC)
    hilbert = HilbertSpace(lattice=lattice)

-----------------------------------------
2. WITH TRANSLATION SYMMETRY (k=0 SECTOR)
-----------------------------------------
    from QES.Algebra.Operator.operator import SymmetryGenerators
    
    hilbert = HilbertSpace(
        lattice=lattice,
        sym_gen=[(SymmetryGenerators.Translation_x, 0)], # can be also a string
        gen_mapping=True
    )
    print(f"Full: {hilbert.nhfull}, Reduced: {hilbert.dim}")

------------------------
3. BUILD OPERATOR MATRIX
------------------------
    from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix
    from numba import njit
    
    @njit
    def sz_total(state):
        ns = 8
        sz = 0.0
        for i in range(ns):
            sz += 0.5 if (state >> (ns-1-i)) & 1 else -0.5
        return np.array([state]), np.array([sz])
    
    H = build_operator_matrix(sz_total, hilbert_space=hilbert, sparse=True)

-------------------------------------
4. ACCESS COMPACT DATA FOR CUSTOM JIT
-------------------------------------
    cd = hilbert.compact_symmetry_data
    if cd is not None:
        @njit
        def my_kernel(state, repr_map, phase_idx, phase_table, normalization):
            idx = repr_map[state]
            if idx == 0xFFFFFFFF:
                return -1, 0.0
            phase = phase_table[phase_idx[state]]
            norm = normalization[idx]
            return idx, np.conj(phase) * norm
        
        # Use in your matrix building loop
        idx, factor = my_kernel(
            new_state, cd.repr_map, cd.phase_idx, 
            cd.phase_table, cd.normalization
        )

----------------------------
5. ITERATE OVER BASIS STATES
----------------------------
    for i, state in enumerate(hilbert):
        if i < 5:
            print(f"State {i}: {state:08b}")

------------------------------------------------
6. ITERATE OVER ALL SYMMETRY SECTORS (Generator)
------------------------------------------------
    from QES.Algebra.Operator.operator import SymmetryGenerators
    
    # All momentum sectors k=0..L-1
    for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        [(SymmetryGenerators.Translation_x, range(L))],
        lattice=lattice
    ):
        k   = sector[SymmetryGenerators.Translation_x]
        H   = build_hamiltonian(hilbert)
        E0  = np.linalg.eigvalsh(H.toarray())[0]
        print(f"k={k}: E0={E0:.6f}")
    
    # Convenience method for momentum sectors
    for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice):
        print(f"k={k}: dim={hilbert.dim}")
    
    # With parity
    for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice, include_parity=True):
        print(f"k={k[:-1]}, P={k[-1]}: dim={hilbert.dim}")
''',
        }
        
        # Default to overview
        if topic is None or topic == 'overview':
            text = help_texts['overview']
        elif topic in help_texts:
            text = help_texts[topic]
        else:
            available = ', '.join(f"'{k}'" for k in help_texts.keys())
            text = f"Unknown topic '{topic}'. Available topics: {available}"
        
        if verbose:
            print(text)
            return None
        return text    

# ------------------------------------------------------------------------------------------------------
#! End of File
# ------------------------------------------------------------------------------------------------------
