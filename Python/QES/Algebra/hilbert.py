
"""
High-level Hilbert space class for quantum many-body systems.

---------------------------------------------------
File    : QES/Algebra/hilbert.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-02-01
Version : 1.0.0
Changes : 
    - 2025.02.01 : 1.0.0 - Initial version of the Hilbert space class. - MK
    - 2025.10.26 : 1.1.0 - Refactored symmetry group generation and added detailed logging. - MK
    - 2025.10.28 : 1.1.1 - Working on symmetry compatibility and modular symmetries. - MK
---------------------------------------------------
"""
import sys
import math
import time
import numpy as np

from abc            import ABC
from typing         import Union, Optional, Callable, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses    import dataclass

try:
    # general thingies
    if TYPE_CHECKING:
        from QES.general_python.common.flog         import Logger
        from QES.general_python.lattices.lattice    import Lattice, LatticeDirection

    #################################################################################################
    from QES.Algebra.Hilbert.hilbert_local          import LocalSpace, StateTypes
    from QES.Algebra.globals                        import GlobalSymmetry
    from QES.Algebra.hilbert_config                 import HilbertConfig
except ImportError as e:
    raise ImportError(f"Failed to import required modules in hilbert.py: {e}") from e

#####################################################################################################
#! Hilbert space class
#####################################################################################################

class HilbertSpace(ABC):
    """
    A class to represent a Hilbert space either in Many-Body Quantum Mechanics or Quantum Information Theory and non-interacting systems.
    """
    
    # --------------------------------------------------------------------------------------------------
    
    _ERRORS = {
        "sym_gen"       : "The symmetry generators must be provided as a dictionary or list.",
        "global_syms"   : "The global symmetries must be provided as a list.",
        "gen_mapping"   : "The flag for generating the mapping must be a boolean.",
        "ns"            : "Either 'ns' or 'lattice' must be provided.",
        "lattice"       : "Either 'ns' or 'lattice' must be provided.",
        "nhl"           : "The local Hilbert space dimension must be an integer.",
        "nhl_small"     : "The local Hilbert space dimension must be >= 2.",
        "nhint"         : "The number of modes must be an integer.",
        "nh_incons"     : "The provided Nh is inconsistent with Ns and Nhl.",
        "single_part"   : "The flag for the single particle system must be a boolean.",
        "state_type"    : "The state type must be a string.",
        "backend"       : "The backend must be a string or a module.",
    }
    
    @staticmethod
    def _raise(s: str): raise ValueError(s)

    # --------------------------------------------------------------------------------------------------
    #! Internal checks and inferences
    # --------------------------------------------------------------------------------------------------

    def _check_init_sym_errors(self, sym_gen, global_syms, gen_mapping):
        ''' Check for initialization symmetry errors '''
        if sym_gen is not None and not isinstance(sym_gen, (dict, list)):
            HilbertSpace._raise(HilbertSpace._ERRORS["sym_gen"])
        if not isinstance(global_syms, list) and global_syms is not None:
            HilbertSpace._raise(HilbertSpace._ERRORS["global_syms"])
        if not isinstance(gen_mapping, bool):
            HilbertSpace._raise(HilbertSpace._ERRORS["gen_mapping"])

    def _check_ns_infer(self, lattice: 'Lattice', ns: int, nh: int):
        ''' Check and infer the system size Ns from provided parameters '''

        # import Lattice for type checking
        from QES.general_python.lattices.lattice import Lattice
        
        # infer local dimension
        _local_dim = self._local_space.local_dim if self._local_space else 2

        # handle the system physical size dimension
        if ns is not None:
            self._ns      = int(ns)
            self._lattice = lattice
            if self._lattice is not None and self._lattice.ns != self._ns:
                self._log(f"Warning: The number of sites in lattice ({self._lattice.ns}) is different than provided ({ns}).", lvl = 1, color = 'yellow', log = 'info')
        elif lattice is not None:
            self._lattice = lattice
            self._ns      = int(lattice.ns)
        elif nh is not None and self._is_many_body:
            # nh = (local_dim)^Ns  -> infer Ns from nh and local_space.local_dim
            try:
                if _local_dim <= 1:
                    HilbertSpace._raise(HilbertSpace._ERRORS["nhl_small"])

                # compute Ns via logarithm and validate exact power
                Ns_est          = int(round(math.log(float(nh), float(_local_dim))))
                if _local_dim ** Ns_est != int(nh):
                    raise ValueError(HilbertSpace._ERRORS['nh_incons'])
                
                self._ns      = Ns_est
                self._lattice = None
                self._log(f"Inferred Ns={self._ns} from Nh={nh} and local_dim={_local_dim}", log='debug', lvl=2)
            except Exception as e:
                HilbertSpace._raise(HilbertSpace._ERRORS['nh_incons'])
        elif nh is not None and self._is_quadratic:
            # Quadratic mode: treat Nh as an effective basis size; commonly Nh==Ns
            self._ns      = int(nh)
            self._lattice = None
            self._log(f"Assuming Ns={self._ns} from provided Nh={nh} in quadratic mode.", log='info', lvl=2)
        else:
            HilbertSpace._raise(HilbertSpace._ERRORS['ns'])

        try:
            self._nhfull    = _local_dim ** (self._ns) if self._ns > 0 else 0
        except OverflowError:
            self._nhfull    = float('inf')
            self._log(f"Warning: Full Hilbert space size exceeds standard limits (Ns={self._ns}).", log='warning', lvl=0)
    
    # --------------------------------------------------------------------------------------------------
    
    def _check_logger(self, logger: Optional['Logger']) -> 'Logger':
        ''' Check and return the logger instance '''
        if logger is None:
            from QES.general_python.common.flog import get_global_logger
            return get_global_logger()
        return logger
    
    # --------------------------------------------------------------------------------------------------
    
    @dataclass
    class SymBasicInfo:
        num_operators   = 0
        num_sectors     = 0
        num_states      = 0
    
    # --------------------------------------------------------------------------------------------------
    #! Initialization
    # --------------------------------------------------------------------------------------------------

    def __init__(self,
                # core definition - elements to define the modes
                ns              : Union[int, None]                                          = None,
                lattice         : Union['Lattice', None]                                    = None,
                nh              : Union[int, None]                                          = None,
                # mode specificaton
                is_manybody     : bool                                                      = True,
                part_conserv    : Optional[bool]                                            = True,
                # local space properties - for many body
                sym_gen         : Union[dict, None]                                         = None,
                global_syms     : Union[List[GlobalSymmetry], None]                         = None,
                gen_mapping     : bool                                                      = False,
                local_space     : Optional[Union[LocalSpace, str]]                          = None,
                # general parameters
                state_type      : StateTypes                                                = StateTypes.INTEGER,
                backend         : str                                                       = 'default',
                dtype           : np.dtype                                                  = np.float64,
                basis           : Optional[str]                                             = None,
                boundary_flux   : Optional[Union[float, Dict['LatticeDirection', float]]]   = None,
                state_filter    : Optional[Callable[[int], bool]]                           = None,
                logger          : Optional['Logger']                                        = None,
                **kwargs):
        r"""
        Initializes a HilbertSpace object with specified system and local space properties, symmetries, and backend configuration.
        
        Parameters:
            ns (Union[int, None], optional):
                Number of sites in the system. If not provided, inferred from `lattice` or `nh`.
            lattice (Union[Lattice, None], optional):
                Lattice object defining the system structure. If provided, `ns` is set from `lattice.ns`.
            nh (Union[int, None], optional):
                Full Hilbert space dimension. Used if neither `ns` nor `lattice` is provided.
            is_manybody (bool, optional):
                Flag indicating if the system is many-body. Default is True.
            part_conserv (Optional[bool], optional):
                Flag indicating if particle number is conserved. Default is True.
            ---------------
            sym_gen (Union[dict, None], optional):
                Dictionary or list specifying symmetry generators. Default is None.
                
                **Examples**:
                
                1. Translation symmetry (momentum sectors):
                    ```python
                    from QES.Algebra.Symmetries.translation import TranslationSymmetry
                    
                    # Dictionary format with explicit symmetry objects:
                    sym_gen = {'translation': TranslationSymmetry(kx=0, ky=0)}
                    
                    # Or using string names (if registered):
                    sym_gen = {'translation': {'kx': 0, 'ky': 0}}
                    
                    # List format:
                    sym_gen = [TranslationSymmetry(kx=0, ky=0)]
                    ```
                
                2. Parity symmetry:
                    ```python
                    from QES.Algebra.Symmetries.parity import ParitySymmetry
                    
                    sym_gen = {'parity': ParitySymmetry(sector=1)}  # Even parity
                    ```
                
                3. Multiple symmetries:
                    ```python
                    sym_gen = {
                        'translation': TranslationSymmetry(kx=np.pi, ky=0),
                        'parity': ParitySymmetry(sector=1),
                        'reflection': ReflectionSymmetry(axis='x', sector=1)
                    }
                    ```
                
                4. Particle number conservation (for fermions):
                    ```python
                    # Automatically handled when part_conserv=True
                    # Can also specify explicitly:
                    sym_gen = {'u1_particle': {'n_particles': 4}}
                    ```
                
                Supported symmetry types:
                    - 'translation'        : Discrete translation (momentum)
                    - 'parity'            : Spin/particle parity
                    - 'reflection'        : Spatial reflection
                    - 'inversion'         : Spatial inversion
                    - 'time_reversal'     : Time reversal
                    - 'u1_particle'       : Particle number (U(1))
                    - 'u1_spin'           : Spin S^z conservation
                    - Custom symmetries can be registered via SymmetryRegistry
                    
            global_syms (Union[List[GlobalSymmetry], None], optional): 
                List of global symmetry objects for additional quantum number constraints.
                These are applied before local symmetry generators. Default is None.
                
            gen_mapping (bool, optional):
                Whether to generate state mapping based on symmetries immediately.
                If False, mapping is generated on-demand. Default is False.
                Set to True for immediate symmetry reduction and representative state mapping.
                
            local_space (Optional[Union[LocalSpace, str]], optional):
                LocalSpace object or string defining local Hilbert space properties. 
                Default is None (uses spin-1/2).
                
                Supported strings: 'spin-1/2', 'spin-1', 'fermion', 'hardcore-boson', etc.
            ---------------
            state_type (str, optional):
                Type of state representation (e.g., "integer"). Default is "integer".
            ---------------
            backend (str, optional):
                Backend to use for vectors and matrices. Default is 'default'.
            dtype (optional):
                Data type for Hilbert space arrays. Default is np.float64.
            basis (Optional[str], optional):
                Initial basis representation ("real", "k-space", "fock", "sublattice", "symmetry").
                If not provided, basis is inferred from system properties. Default is None.
            ---------------
            boundary_flux (Optional[float or dict]):
                Optional Peierls phase specification applied to lattice boundary
                crossings. Accepts a scalar phase (in radians) or a mapping from
                :class:`LatticeDirection` to per-direction phases.
            state_filter (Optional[Callable[[int], bool]]):
                Optional predicate applied to integer-encoded basis labels during
                symmetry reduction.  States for which the predicate returns False
                are skipped.  Useful for enforcing additional conserved quantities.
            logger (Optional[Logger], optional):
                Logger instance for logging. Default is None.
            **kwargs:
                Additional keyword arguments, such as 'threadnum' (number of threads to use).
        Raises:
            ValueError: If provided arguments do not match expected types or required parameters are missing.
        Notes:
            - If both `ns` and `lattice` are None, and `nh` is provided, `ns` is inferred from `nh` and `nhl`.
            - Initializes symmetry groups, state mappings, and backend configuration.
            - Sets up logging and threading options.
        """
        
        #! initialize the backend for the vectors and matrices
        self._logger        = self._check_logger(logger)
        self._backend, self._backend_str, self._state_type = self.reset_backend(backend, state_type)
        self._dtype         = dtype if dtype is not None else self._backend.float64
        self._is_many_body  = is_manybody
        self._is_quadratic  = not is_manybody
        
        #! quick check
        self._check_init_sym_errors(sym_gen, global_syms, gen_mapping)

        #! set locals
        # If you have a LocalSpace.default(), use it; otherwise use your default spin-1/2 factory.
        if isinstance(local_space, LocalSpace):
            self._local_space = local_space
        elif isinstance(local_space, str):
            self._local_space = LocalSpace.from_str(local_space)
        else:
            self._local_space = LocalSpace.default() # or default_spin_half_local_space()

        #! infer the system sizes
        self._check_ns_infer(lattice=lattice, ns=ns, nh=nh)
        self._boundary_flux = boundary_flux
        if self._lattice is not None and boundary_flux is not None:
            self._lattice.flux = boundary_flux

        #! State filtering - predicate to filter basis states
        self._state_filter = state_filter

        #! =====================================================================
        #! BASIS REPRESENTATION TRACKING (infer from system properties)
        #! =====================================================================
        # Infer natural basis representation based on system properties, or use provided one
        self._infer_and_set_default_basis(explicit_basis=basis)
        
        #! Nh: Effective dimension of the *current* representation
        # Initial estimate:
        if self._is_quadratic:
            # For quadratic, the "dimension" is often Ns (or 2Ns if pairing)
            # We'll set it initially to Ns, can be adjusted later if needed (e.g., for Bogoliubov)
            self._nh = self._ns
            self._log(f"Initialized HilbertSpace in quadratic mode: Ns={self._ns}, effective Nh={self._nh}.", log='debug', lvl=1, color='green')
        else: # Many-body
            # If symmetries will be applied, Nh will be reduced later.
            # Start with the full size, potentially reduced by global syms only initially.
            self._nh = self._nhfull
            self._log(f"Initialized HilbertSpace in many-body mode: Ns={self._ns}, initial Nh={self._nh} (potentially reducible).", color='green', log='debug', lvl=1)

        #! Initialize the symmetries    
        self.representative_list            = None                          # List of representatives
        self.representative_norms           = None                          # normalization of the states - how to return to the representative
        self.full_to_representative_idx     = None                          # full to representative index mapping
        self.full_to_representative_phase   = None                          # full to representative phase mapping
        self.full_to_global_map             = None                          # full to global map
        self._sym_group                     = []                            # main symmetry group (will be populated later)
        self._sym_basic_info                = HilbertSpace.SymBasicInfo()   # basic symmetry info container
        self._global_syms                   = global_syms if global_syms is not None else []
        self._particle_conserving           = part_conserv

        # Symmetry container
        self._sym_container                 = None                          # Will be initialized in _init_representatives
        self._has_complex_symmetries        = False                         # Cached flag indicating whether any configured symmetry eigenvalues
                                                                            # jittable arrays (repr_idx, repr_phase) may be created when mapping is generated

        # setup the logger instance for the Hilbert space
        self._threadnum                     = kwargs.get('threadnum', 1)    # number of threads to use

        if self._is_many_body:
            if gen_mapping:
                self._log("Explicitly requested immediate mapping generation.", log='debug', lvl=2)
            
            # Normalize symmetry generators (supports dict/list/string formats)
            normalized_sym_gen = self._normalize_symmetry_generators(sym_gen)
            
            self._init_representatives(normalized_sym_gen, gen_mapping=gen_mapping) # gen_mapping True here enables reprmap
            # Set symmetry group from container
            if self._sym_container is not None:
                self._sym_group = list(self._sym_container.symmetry_group)
        elif self._is_quadratic:
            self._log("Quadratic mode: Skipping symmetry mapping generation.", log='debug', lvl=2)
            self.representative_list    = None
            self.representative_norms   = None
            
            # Setup sym container if generators provided
            if sym_gen:
                # Normalize symmetry generators (supports dict/list/string formats)
                normalized_sym_gen = self._normalize_symmetry_generators(sym_gen)
                
                self._init_sym_container(normalized_sym_gen)
                if self._sym_container is not None:
                    self._sym_group = list(self._sym_container.symmetry_group)

        # Ensure symmetry group always has at least the identity
        if not self._sym_group or len(self._sym_group) == 0:
            try:
                from QES.Algebra.Operator.operator import operator_identity
            except ImportError as e:
                raise RuntimeError("Failed to import operator_identity. Ensure QES.Algebra.Operator.operator is available.") from e
            self._sym_group = [operator_identity(self._backend_str)]
    
    # --------------------------------------------------------------------------------------------------
    #! Configuration from HilbertConfig
    # --------------------------------------------------------------------------------------------------
    
    @classmethod
    def from_config(cls, config: HilbertConfig, **overrides):
        """
        Instantiate a HilbertSpace from a :class:`HilbertConfig`.

        Parameters
        ----------
        config:
            Base configuration blueprint.
        **overrides:
            Keyword arguments applied on top of the blueprint before instantiation.
        """
        cfg = config.with_override(**overrides) if overrides else config
        return cls(**cfg.to_kwargs())    

    # --------------------------------------------------------------------------------------------------
    #! Help and Documentation
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

HilbertSpace represents a quantum many-body or single-particle Hilbert space
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
    .dim / .Nh          : Reduced Hilbert space dimension
    .nhfull             : Full Hilbert space dimension (before symmetry reduction)
    .ns                 : Number of sites
    .representative_list: Array of representative state integers
    .normalization      : Normalization factors for representatives
    .compact_symmetry_data: CompactSymmetryData for O(1) lookups

Use HilbertSpace.help('symmetries') for symmetry details.
Use HilbertSpace.help('examples') for more code examples.
''',

            'symmetries': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                     HILBERTSPACE - SYMMETRY HANDLING                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SUPPORTED SYMMETRIES
--------------------
1. Translation (momentum sectors):
   - SymmetryGenerators.Translation_x/y/z
   - Sector value = momentum quantum number k (0, 1, ..., L-1)
   - Example: k=0 is the Gamma point (fully symmetric)

2. Reflection (spatial parity):
   - SymmetryGenerators.Reflection
   - Sector value = ±1

3. Parity (spin/particle flips):
   - SymmetryGenerators.ParityX/Y/Z
   - Sector value = ±1 or ±i

4. Global symmetries (U(1) conservation):
   - Particle number conservation (fixed N)
   - Magnetization conservation (fixed Sz)

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

COMPACT SYMMETRY DATA (Memory Efficient)
----------------------------------------
When symmetries are present, a CompactSymmetryData structure is built:
    
    cd = hilbert.compact_symmetry_data
    
    • cd.repr_map     : uint32[nh_full] - state → repr index (O(1) lookup!)
    • cd.phase_idx    : uint8[nh_full]  - state → phase table index
    • cd.phase_table  : complex128[~group_order] - distinct phases
    • cd.normalization: float64[n_repr] - normalization factors
    
Memory: ~5 bytes/state vs ~24+ bytes for object arrays (79% reduction!)

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

STRUCTURE
---------
    @dataclass
    class CompactSymmetryData:
        repr_map: np.ndarray            # uint32[nh_full]  - state → repr index
        phase_idx: np.ndarray           # uint8[nh_full]   - state → phase index
        phase_table: np.ndarray         # complex128[n]    - distinct phases
        normalization: np.ndarray       # float64[n_repr]  - per-representative
        representative_list: np.ndarray # int64[n_repr]    - repr state values

MEMORY USAGE
------------
    Per state: 4 bytes (uint32) + 1 byte (uint8) = 5 bytes
    
    For 2^20 states (~1M):
      Old format: ~24 MB
      Compact:    ~5 MB  (79% reduction!)

USAGE
-----
    cd = hilbert.compact_symmetry_data
    
    # Check if state is in this sector
    if cd.repr_map[state] != 0xFFFFFFFF:
        idx = cd.repr_map[state]
        phase = cd.phase_table[cd.phase_idx[state]]
        norm = cd.normalization[idx]
        repr_state = cd.representative_list[idx]

JIT FUNCTIONS (numba.njit compatible)
-------------------------------------
    from QES.Algebra.Symmetries import (
        _compact_get_sym_factor,   # (state, k, ...) → (idx, sym_factor)
        _compact_is_in_sector,     # (state, repr_map) → bool
        _compact_get_repr_idx,     # (state, repr_map) → int
        _compact_get_phase,        # (state, phase_idx, phase_table) → complex
    )
''',

            'properties': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                      HILBERTSPACE - PROPERTIES                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

DIMENSIONS
----------
    .dim / .Nh / .dimension     : Reduced Hilbert space dimension
    .nhfull / .Nhfull / .full   : Full dimension (before symmetry reduction)
    .ns                         : Number of sites

SYMMETRY DATA
-------------
    .representative_list        : int64[n_repr] - representative states
    .representative_norms       : float64[n_repr] - normalization factors
    .normalization              : Alias for representative_norms
    .compact_symmetry_data      : CompactSymmetryData structure (primary)
    .has_compact_symmetry_data  : bool - whether compact data exists
    .repr_idx                   : uint32[nh_full] - state → repr index
    .repr_phase                 : complex128[nh_full] - symmetry phases

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

1. BASIC SETUP
--------------
    from QES.Algebra.hilbert import HilbertSpace
    from QES.general_python.lattices.square import SquareLattice
    from QES.general_python.lattices.lattice import LatticeBC
    
    lattice = SquareLattice(dim=1, lx=8, bc=LatticeBC.PBC)
    hilbert = HilbertSpace(lattice=lattice)

2. WITH TRANSLATION SYMMETRY (k=0 SECTOR)
-----------------------------------------
    from QES.Algebra.Operator.operator import SymmetryGenerators
    
    hilbert = HilbertSpace(
        lattice=lattice,
        sym_gen=[(SymmetryGenerators.Translation_x, 0)],
        gen_mapping=True
    )
    print(f"Full: {hilbert.nhfull}, Reduced: {hilbert.dim}")

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

5. ITERATE OVER BASIS STATES
----------------------------
    for i, state in enumerate(hilbert):
        if i < 5:
            print(f"State {i}: {state:08b}")

6. ITERATE OVER ALL SYMMETRY SECTORS (Generator)
------------------------------------------------
    from QES.Algebra.Operator.operator import SymmetryGenerators
    
    # All momentum sectors k=0..L-1
    for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        [(SymmetryGenerators.Translation_x, range(L))],
        lattice=lattice
    ):
        k = sector[SymmetryGenerators.Translation_x]
        H = build_hamiltonian(hilbert)
        E0 = np.linalg.eigvalsh(H.toarray())[0]
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

    # --------------------------------------------------------------------------------------------------
    #! Symmetry Sector Iteration (Generator)
    # --------------------------------------------------------------------------------------------------
    
    @staticmethod
    def iter_symmetry_sectors(
        symmetry_types  : 'List[Tuple[SymmetryGenerators, List]]',
        *,
        lattice         : Optional['Lattice']               = None,
        ns              : Optional[int]                     = None,
        gen_mapping     : bool                              = True,
        verbose         : bool                              = False,
        **hilbert_kwargs
    ):
        """
        Generator that yields HilbertSpaces for all symmetry sectors.
        
        This is a memory-efficient way to iterate over all symmetry sectors
        without creating all HilbertSpaces at once. Useful for:
        - Complete spectrum calculations across all k-sectors
        - Finding ground states in each symmetry sector
        - Symmetry-resolved spectroscopy
        
        Parameters
        ----------
        symmetry_types : List[Tuple[SymmetryGenerators, List]]
            List of (generator_type, allowed_sectors) pairs. Each pair specifies
            a symmetry and the list of sector values to iterate over.
            
            Examples:
            - [(SymmetryGenerators.Translation_x, range(L))]
            - [(SymmetryGenerators.Translation_x, range(Lx)), 
               (SymmetryGenerators.Translation_y, range(Ly))]
            - [(SymmetryGenerators.Translation_x, range(L)),
               (SymmetryGenerators.Reflection, [1, -1])]
               
        lattice : Lattice, optional
            Lattice structure. Either lattice or ns must be provided.
        ns : int, optional  
            Number of sites. Either lattice or ns must be provided.
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping for O(1) lookups.
        verbose : bool, default=False
            If True, log information about each sector.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            Common options: np (particle number), backend, dtype, logger.
            
        Yields
        ------
        Tuple[dict, HilbertSpace]
            A tuple of (sector_dict, hilbert_space) where:
            - sector_dict: Dictionary mapping SymmetryGenerators to sector values
              e.g., {SymmetryGenerators.Translation_x: 0}
            - hilbert_space: HilbertSpace instance for that sector
            
        Examples
        --------
        >>> from QES.Algebra.hilbert import HilbertSpace
        >>> from QES.Algebra.Operator.operator import SymmetryGenerators
        >>> from QES.general_python.lattices.square import SquareLattice
        >>> from QES.general_python.lattices.lattice import LatticeBC
        >>> 
        >>> lattice = SquareLattice(dim=1, lx=8, bc=LatticeBC.PBC)
        >>> 
        >>> # Iterate over all momentum sectors k=0,1,...,7
        >>> for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        ...     [(SymmetryGenerators.Translation_x, range(8))],
        ...     lattice=lattice
        ... ):
        ...     print(f"k={sector[SymmetryGenerators.Translation_x]}: dim={hilbert.dim}")
        k=0: dim=36
        k=1: dim=32
        ...
        
        >>> # 2D lattice: iterate over (kx, ky) sectors
        >>> lattice_2d = SquareLattice(dim=2, lx=4, ly=4, bc=LatticeBC.PBC)
        >>> for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        ...     [(SymmetryGenerators.Translation_x, range(4)),
        ...      (SymmetryGenerators.Translation_y, range(4))],
        ...     lattice=lattice_2d
        ... ):
        ...     kx, ky = sector[SymmetryGenerators.Translation_x], sector[SymmetryGenerators.Translation_y]
        ...     print(f"(kx={kx}, ky={ky}): dim={hilbert.dim}")
        
        >>> # Combined translation + reflection
        >>> for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        ...     [(SymmetryGenerators.Translation_x, range(8)),
        ...      (SymmetryGenerators.Reflection, [1, -1])],
        ...     lattice=lattice
        ... ):
        ...     k, p = sector[SymmetryGenerators.Translation_x], sector[SymmetryGenerators.Reflection]
        ...     print(f"k={k}, parity={p}: dim={hilbert.dim}")
        
        Notes
        -----
        - Uses Python generators for memory efficiency - only one HilbertSpace
          is in memory at a time (plus any you explicitly keep references to).
        - For very large systems, consider using verbose=True to track progress.
        - The order of iteration follows itertools.product over the sector lists.
        """
        from itertools import product
        
        # Validate inputs
        if lattice is None and ns is None:
            raise ValueError("Either 'lattice' or 'ns' must be provided.")
        
        if not symmetry_types or len(symmetry_types) == 0:
            raise ValueError("At least one symmetry type must be provided.")
        
        # Extract generator types and sector lists
        gen_types       = [sym_type for sym_type, _ in symmetry_types]
        sector_lists    = [list(sectors) for _, sectors in symmetry_types]
        
        # Total number of sectors (for progress reporting)
        total_sectors   = 1
        for sectors in sector_lists:
            total_sectors *= len(sectors)
        
        # Iterate over Cartesian product of all sector values
        for sector_idx, sector_values in enumerate(product(*sector_lists)):
            # Build sym_gen list for this sector
            sym_gen = list(zip(gen_types, sector_values))
            
            # Build sector dictionary for return
            sector_dict = {gen_type: value for gen_type, value in sym_gen}
            
            if verbose:
                sector_str = ", ".join(f"{gt.name}={v}" for gt, v in sym_gen)
                print(f"[{sector_idx+1}/{total_sectors}] Building sector: {sector_str}")
            
            # Create HilbertSpace for this sector
            hilbert = HilbertSpace(
                lattice     = lattice,
                ns          = ns,
                sym_gen     = sym_gen,
                gen_mapping = gen_mapping,
                **hilbert_kwargs
            )
            
            yield sector_dict, hilbert

    @staticmethod
    def iter_momentum_sectors(
        lattice         : 'Lattice',
        *,
        include_parity  : bool  = False,
        gen_mapping     : bool  = True,
        verbose         : bool  = False,
        **hilbert_kwargs
    ):
        """
        Generator that yields HilbertSpaces for all momentum sectors of a lattice.
        
        This is a convenience method for the common case of iterating over all
        translation symmetry sectors. Automatically detects lattice dimension
        and generates appropriate momentum quantum numbers.
        
        Parameters
        ----------
        lattice : Lattice
            Lattice structure with periodic boundary conditions.
        include_parity : bool, default=False
            If True, also iterate over reflection parity sectors (+1, -1).
            This doubles the number of sectors.
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping for O(1) lookups.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            Common options: np (particle number), backend, dtype, logger.
            
        Yields
        ------
        Tuple[Tuple[int, ...], HilbertSpace]
            A tuple of (k_vector, hilbert_space) where:
            - k_vector: Tuple of momentum quantum numbers (kx,) or (kx, ky) or (kx, ky, kz)
            - hilbert_space: HilbertSpace instance for that momentum sector
            
        Examples
        --------
        >>> # 1D chain with L=8 sites
        >>> lattice = SquareLattice(dim=1, lx=8, bc=LatticeBC.PBC)
        >>> for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice):
        ...     print(f"k={k}: dim={hilbert.dim}")
        k=(0,): dim=36
        k=(1,): dim=30
        ...
        
        >>> # 2D lattice 4x4
        >>> lattice_2d = SquareLattice(dim=2, lx=4, ly=4, bc=LatticeBC.PBC)
        >>> for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice_2d):
        ...     kx, ky = k
        ...     print(f"(kx={kx}, ky={ky}): dim={hilbert.dim}")
        
        >>> # With parity
        >>> for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice, include_parity=True):
        ...     kx, parity = k[:-1], k[-1]  # Last element is parity
        ...     print(f"k={kx}, P={parity}: dim={hilbert.dim}")
            
        Notes
        -----
        - Assumes lattice has lx, ly, lz attributes for lattice dimensions.
        - Only uses Translation_x, Translation_y, Translation_z generators.
        - Momentum quantum numbers run from 0 to L-1 for each direction.
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        
        # Detect lattice dimensions
        lx = getattr(lattice, 'lx', None) or getattr(lattice, 'Lx', None)
        ly = getattr(lattice, 'ly', None) or getattr(lattice, 'Ly', None)
        lz = getattr(lattice, 'lz', None) or getattr(lattice, 'Lz', None)
        
        if lx is None or lx <= 0:
            raise ValueError("Lattice must have positive lx dimension.")
        
        # Build symmetry types based on lattice dimensionality
        symmetry_types = [(SymmetryGenerators.Translation_x, range(lx))]
        
        if ly is not None and ly > 1:
            symmetry_types.append((SymmetryGenerators.Translation_y, range(ly)))
        
        if lz is not None and lz > 1:
            symmetry_types.append((SymmetryGenerators.Translation_z, range(lz)))
        
        if include_parity:
            symmetry_types.append((SymmetryGenerators.Reflection, [1, -1]))
        
        # Use the general generator
        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice     = lattice,
            gen_mapping = gen_mapping,
            verbose     = verbose,
            **hilbert_kwargs
        ):
            # Convert sector_dict to tuple for easier use
            k_vector = [sector_dict[SymmetryGenerators.Translation_x]]
            
            if ly is not None and ly > 1:
                k_vector.append(sector_dict[SymmetryGenerators.Translation_y])
            
            if lz is not None and lz > 1:
                k_vector.append(sector_dict[SymmetryGenerators.Translation_z])
            
            if include_parity:
                k_vector.append(sector_dict[SymmetryGenerators.Reflection])
            
            yield tuple(k_vector), hilbert

    @staticmethod
    def iter_reflection_sectors(
        lattice         : 'Lattice',
        *,
        momentum_sector : Optional[int]     = None,
        gen_mapping     : bool              = True,
        verbose         : bool              = False,
        **hilbert_kwargs
    ):
        """
        Generator that yields HilbertSpaces for all reflection parity sectors.
        
        Parameters
        ----------
        lattice : Lattice
            Lattice structure.
        momentum_sector : int, optional
            If provided, fix the translation sector to this value.
            Reflection is only compatible with k=0 and k=pi (L/2).
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            
        Yields
        ------
        Tuple[int, HilbertSpace]
            A tuple of (parity, hilbert_space) where parity is +1 or -1.
            
        Examples
        --------
        >>> for parity, hilbert in HilbertSpace.iter_reflection_sectors(lattice):
        ...     print(f"P={parity:+d}: dim={hilbert.dim}")
        P=+1: dim=...
        P=-1: dim=...
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
        
        symmetry_types = []
        
        if momentum_sector is not None:
            symmetry_types.append((SymmetryGenerators.Translation_x, [momentum_sector]))
        
        symmetry_types.append((SymmetryGenerators.Reflection, ReflectionSymmetry.get_sectors()))
        
        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice     = lattice,
            gen_mapping = gen_mapping,
            verbose     = verbose,
            **hilbert_kwargs
        ):
            parity = sector_dict[SymmetryGenerators.Reflection]
            yield parity, hilbert

    @staticmethod
    def iter_inversion_sectors(
        lattice         : 'Lattice',
        *,
        momentum_sector : Optional[int]     = None,
        gen_mapping     : bool              = True,
        verbose         : bool              = False,
        **hilbert_kwargs):
        """
        Generator that yields HilbertSpaces for all spatial inversion parity sectors.
        
        Unlike reflection (which uses bit-reversal), inversion uses lattice coordinates
        and works correctly for any lattice type (square, honeycomb, triangular, etc.)
        and any dimension (1D, 2D, 3D).
        
        Parameters
        ----------
        lattice : Lattice
            Lattice structure.
        momentum_sector : int, optional
            If provided, fix the translation sector to this value.
            Inversion is only compatible with k=0 and k=pi (L/2).
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            
        Yields
        ------
        Tuple[int, HilbertSpace]
            A tuple of (parity, hilbert_space) where parity is +1 or -1.
            
        Examples
        --------
        >>> # Works for any lattice type
        >>> from QES.general_python.lattices.square import SquareLattice
        >>> lattice = SquareLattice(dim=2, lx=4, ly=4)  # 2D
        >>> for parity, hilbert in HilbertSpace.iter_inversion_sectors(lattice):
        ...     print(f"P={parity:+d}: dim={hilbert.dim}")
        P=+1: dim=...
        P=-1: dim=...
        
        >>> # Also works for honeycomb, triangular, etc.
        >>> from QES.general_python.lattices.honeycomb import HoneycombLattice
        >>> lattice = HoneycombLattice(lx=3, ly=3)
        >>> for parity, hilbert in HilbertSpace.iter_inversion_sectors(lattice):
        ...     print(f"P={parity:+d}: dim={hilbert.dim}")
        """
        from QES.Algebra.Operator.operator      import SymmetryGenerators
        from QES.Algebra.Symmetries.inversion   import InversionSymmetry
        
        symmetry_types = []
        
        if momentum_sector is not None:
            symmetry_types.append((SymmetryGenerators.Translation_x, [momentum_sector]))
        
        symmetry_types.append((SymmetryGenerators.Inversion, InversionSymmetry.get_sectors()))
        
        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice     = lattice,
            gen_mapping = gen_mapping,
            verbose     = verbose,
            **hilbert_kwargs
        ):
            parity = sector_dict[SymmetryGenerators.Inversion]
            yield parity, hilbert

    @staticmethod
    def iter_parity_sectors(
        lattice         : 'Lattice',
        axis            : str               = 'z',
        *,
        gen_mapping     : bool              = True,
        verbose         : bool              = False,
        **hilbert_kwargs
    ):
        """
        Generator that yields HilbertSpaces for all spin-parity sectors.
        
        Parameters
        ----------
        lattice : Lattice
            Lattice structure.
        axis : str, default='z'
            Parity axis ('x', 'y', or 'z').
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            
        Yields
        ------
        Tuple[int, HilbertSpace]
            A tuple of (parity, hilbert_space) where parity is +1 or -1.
            
        Examples
        --------
        >>> for parity, hilbert in HilbertSpace.iter_parity_sectors(lattice, axis='z'):
        ...     print(f"Pz={parity:+d}: dim={hilbert.dim}")
        Pz=+1: dim=...
        Pz=-1: dim=...
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.parity import ParitySymmetry
        
        axis_map = {
            'x': SymmetryGenerators.ParityX,
            'y': SymmetryGenerators.ParityY,
            'z': SymmetryGenerators.ParityZ,
        }
        
        if axis.lower() not in axis_map:
            raise ValueError(f"Unknown parity axis '{axis}'. Use 'x', 'y', or 'z'.")
        
        gen_type = axis_map[axis.lower()]
        symmetry_types = [(gen_type, ParitySymmetry.get_sectors(axis))]
        
        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice     = lattice,
            gen_mapping = gen_mapping,
            verbose     = verbose,
            **hilbert_kwargs
        ):
            parity = sector_dict[gen_type]
            yield parity, hilbert

    @staticmethod
    def iter_all_sectors(
        lattice         : 'Lattice',
        symmetries      : 'List[str]'       = None,
        *,
        gen_mapping     : bool              = True,
        verbose         : bool              = False,
        **hilbert_kwargs
    ):
        """
        Generator that yields HilbertSpaces for all combinations of specified symmetries.
        
        This is a high-level convenience method that automatically determines
        valid sectors for each symmetry type based on the lattice.
        
        Parameters
        ----------
        lattice : Lattice
            Lattice structure with periodic boundary conditions.
        symmetries : List[str], optional
            List of symmetry names to include. Options:
            - 'translation' or 'momentum' : Translation symmetry (all k-sectors)
            - 'reflection' or 'parity_spatial' : Spatial reflection (+1, -1)
            - 'parity_z' or 'spin_parity' : Spin-flip parity Z (+1, -1)
            - 'parity_x' : Spin-flip parity X (+1, -1)
            
            Default: ['translation'] (all momentum sectors only).
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            
        Yields
        ------
        Tuple[dict, HilbertSpace]
            A tuple of (sector_dict, hilbert_space) where sector_dict maps
            symmetry names to their sector values.
            
        Examples
        --------
        >>> # All momentum sectors
        >>> for sectors, h in HilbertSpace.iter_all_sectors(lattice):
        ...     print(f"k={sectors['kx']}: dim={h.dim}")
        
        >>> # Momentum + reflection
        >>> for sectors, h in HilbertSpace.iter_all_sectors(
        ...     lattice, symmetries=['translation', 'reflection']
        ... ):
        ...     print(f"k={sectors['kx']}, P={sectors['reflection']}: dim={h.dim}")
        
        >>> # All symmetries for complete spectrum decomposition
        >>> for sectors, h in HilbertSpace.iter_all_sectors(
        ...     lattice, 
        ...     symmetries=['translation', 'reflection', 'parity_z'],
        ...     verbose=True
        ... ):
        ...     # Build Hamiltonian and diagonalize
        ...     H = build_hamiltonian(h)
        ...     energies = np.linalg.eigvalsh(H.toarray())
        ...     sectors['energies'] = energies
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.translation import TranslationSymmetry
        from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
        from QES.Algebra.Symmetries.parity import ParitySymmetry
        
        if symmetries is None:
            symmetries = ['translation']
        
        symmetry_types = []
        sector_names = {}  # Maps SymmetryGenerators -> friendly name
        
        lx = getattr(lattice, 'lx', None) or getattr(lattice, 'Lx', 1)
        ly = getattr(lattice, 'ly', None) or getattr(lattice, 'Ly', 1)
        lz = getattr(lattice, 'lz', None) or getattr(lattice, 'Lz', 1)
        
        for sym in symmetries:
            sym_lower = sym.lower().replace('_', '').replace('-', '')
            
            if sym_lower in ['translation', 'momentum', 'trans']:
                symmetry_types.append((SymmetryGenerators.Translation_x, TranslationSymmetry.get_sectors(lattice, 'x')))
                sector_names[SymmetryGenerators.Translation_x] = 'kx'
                
                if ly > 1:
                    symmetry_types.append((SymmetryGenerators.Translation_y, TranslationSymmetry.get_sectors(lattice, 'y')))
                    sector_names[SymmetryGenerators.Translation_y] = 'ky'
                
                if lz > 1:
                    symmetry_types.append((SymmetryGenerators.Translation_z, TranslationSymmetry.get_sectors(lattice, 'z')))
                    sector_names[SymmetryGenerators.Translation_z] = 'kz'
                    
            elif sym_lower in ['reflection', 'parityspatial', 'spatial', 'mirror']:
                symmetry_types.append((SymmetryGenerators.Reflection, ReflectionSymmetry.get_sectors()))
                sector_names[SymmetryGenerators.Reflection] = 'reflection'
            
            elif sym_lower in ['inversion', 'inv', 'spatialinversion']:
                from QES.Algebra.Symmetries.inversion import InversionSymmetry
                symmetry_types.append((SymmetryGenerators.Inversion, InversionSymmetry.get_sectors()))
                sector_names[SymmetryGenerators.Inversion] = 'inversion'
                
            elif sym_lower in ['parityz', 'spinparity', 'pz', 'spinflip']:
                symmetry_types.append((SymmetryGenerators.ParityZ, ParitySymmetry.get_sectors('z')))
                sector_names[SymmetryGenerators.ParityZ] = 'parity_z'
                
            elif sym_lower in ['parityx', 'px']:
                symmetry_types.append((SymmetryGenerators.ParityX, ParitySymmetry.get_sectors('x')))
                sector_names[SymmetryGenerators.ParityX] = 'parity_x'
                
            elif sym_lower in ['parityy', 'py']:
                symmetry_types.append((SymmetryGenerators.ParityY, ParitySymmetry.get_sectors('y')))
                sector_names[SymmetryGenerators.ParityY] = 'parity_y'
                
            else:
                raise ValueError(f"Unknown symmetry '{sym}'. Options: translation, reflection, inversion, parity_z, parity_x, parity_y")
        
        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice     = lattice,
            gen_mapping = gen_mapping,
            verbose     = verbose,
            **hilbert_kwargs
        ):
            # Convert to friendly names
            named_sectors = {sector_names[gen]: val for gen, val in sector_dict.items()}
            yield named_sectors, hilbert

    # --------------------------------------------------------------------------------------------------
    #! Resets
    # --------------------------------------------------------------------------------------------------
    
    @staticmethod
    def reset_backend(backend : str, state_type : str):
        """
        Reset the backend for the Hilbert space.
        
        Args:
            backend (str): The backend to use for the Hilbert space.
        """
        if isinstance(backend, str):
            from QES.general_python.algebra.utils import get_backend
            _backend_str   = backend
            _backend       = get_backend(backend)
        else:
            _backend_str   = 'np' if backend == np else 'jax'
            _backend       = backend
        
        statetype = HilbertSpace.reset_statetype(state_type, _backend)
        return _backend, _backend_str, statetype
    
    @staticmethod
    def reset_statetype(state_type: str, backend):
        """
        Reset the state type for the Hilbert space.
        
        Args:
            state_type (str): The state type to use for the Hilbert space.
        """
        if str(state_type).lower() == "integer" or str(state_type).lower() == "int":
            return int
        return backend.array
    
    def reset_local_symmetries(self):
        """
        Reset the local symmetries of the Hilbert space.
        """
        from QES.Algebra.Operator.operator import operator_identity
        self._log("Reseting the local symmetries. Can be now recreated.", lvl = 2, log = 'debug')
        self._sym_group = [operator_identity(self._backend_str)]
    
    # --------------------------------------------------------------------------------------------------
    
    def _log(self, msg : str, log : Union[int, str] = 'info', lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (Union[int, str]) : The flag to log the message (default is 'info').
            lvl (int) : The level of the message.
        """
        if self._logger is None:
            return
        
        try:
            from QES.general_python.common.flog import Logger
        except ImportError as e:
            return
        
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[HilbertSpace] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)

    ####################################################################################################
    #! Unified symmetry container initialization
    ####################################################################################################
    
    @staticmethod
    def _parse_symmetry_spec(sym_name: str, sym_value: Union[dict, int, float, complex]) -> List[Tuple]:
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
            
        Examples
        --------
        >>> # Translation with momentum sectors
        >>> HilbertSpace._parse_symmetry_spec('translation', {'kx': 0, 'ky': np.pi})
        [(SymmetryGenerators.Translation_x, 0), (SymmetryGenerators.Translation_y, np.pi)]
        
        >>> # Parity with sector
        >>> HilbertSpace._parse_symmetry_spec('parity', 1)
        [(SymmetryGenerators.ParityZ, 1)]
        
        >>> # Parity with axis specification
        >>> HilbertSpace._parse_symmetry_spec('parity', {'axis': 'x', 'sector': 1})
        [(SymmetryGenerators.ParityX, 1)]
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        
        sym_name_lower = sym_name.lower().replace('_', '').replace('-', '')
        specs = []
        
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
                axis = sym_value.get('axis', 'z').lower()
                sector = sym_value.get('sector', 1)
            else:
                axis = 'z'  # Default to z-parity
                sector = sym_value
            
            if axis == 'x':
                specs.append((SymmetryGenerators.ParityX, sector))
            elif axis == 'y':
                specs.append((SymmetryGenerators.ParityY, sector))
            else:  # 'z' or default
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
    
    @staticmethod
    def _normalize_symmetry_generators(sym_gen: Union[dict, list, None]) -> List[Tuple]:
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
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.base import SymmetryOperator
        
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
                        parsed = HilbertSpace._parse_symmetry_spec(sym_name, sym_value)
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
                parsed = HilbertSpace._parse_symmetry_spec(sym_name, sym_value)
                specs.extend(parsed)
            
            return specs
        
        raise ValueError(f"sym_gen must be dict, list, or None, got {type(sym_gen)}")
    
    def _init_sym_container(self, gen: list) -> None:
        """
        Initialize the unified SymmetryContainer.
        Creates and configures the SymmetryContainer based on provided symmetry generators
        and global symmetries.
        
        The SymmetryContainer handles:
        - Compatibility checking
        - Automatic group construction from generators
        - Representative finding
        - Normalization computation
        
        Parameters:
        -----------
        gen : list
            List of (SymmetryGenerator, sector_value) tuples
        """
        
        if (not gen or len(gen) == 0) and not self.check_global_symmetry:
            self._log("No symmetries provided; SymmetryContainer will use identity only.", lvl=1, log='debug', color='green')
            
            try:
                from QES.Algebra.Symmetries.symmetry_container import SymmetryContainer
            except ImportError as e:
                raise RuntimeError("Failed to import SymmetryContainer. Ensure QES.Algebra.Symmetries.symmetry_container is available.") from e
            self._sym_basic_info    = HilbertSpace.SymBasicInfo()   # basic symmetry info container
            self._sym_container     = SymmetryContainer(
                                        ns          = self._ns,
                                        lattice     = self._lattice,
                                        nhl         = self._local_space.local_dim,
                                        backend     = self._backend_str
                                    )
            return
        
        # Import the factory function
        try:
            from QES.Algebra.Symmetries.symmetry_container import create_symmetry_container_from_specs
        except ImportError as e:
            raise RuntimeError("Failed to import SymmetryContainer factory. Ensure QES.Algebra.Symmetries.symmetry_container is available.") from e
        
        # Prepare generator specs
        generator_specs = gen.copy() if gen is not None else []
        
        # Create and initialize the container
        # The factory will handle all compatibility checking and filtering
        # Note: Compact symmetry data is built later in _init_representatives after representatives are generated
        self._sym_container = create_symmetry_container_from_specs(
                                ns              = self._ns,
                                generator_specs = generator_specs,
                                global_syms     = self._global_syms,
                                lattice         = self._lattice,
                                nhl             = self._local_space.local_dim,
                                backend         = self._backend_str,
                                build_group     = True
                            )
                            
        
        # Log symmetry info
        self._sym_basic_info.num_ops    = len(self._sym_container.symmetry_group)
        self._sym_basic_info.num_gens   = len(self._sym_container.generators)
        self._log(f"SymmetryContainer initialized: {self._sym_basic_info.num_gens} generators -> {self._sym_basic_info.num_ops} group elements", lvl=1, color='green')

        # Compute and cache whether any symmetry eigenvalues/phases are complex.
        # Conservative: on unexpected errors assume True to avoid dropping
        # complex information.
        try:
            # Prefer the canonical group stored on the SymmetryContainer when present
            has_cpx     = False
            group       = getattr(self._sym_container, 'symmetry_group', None) or self._sym_group
            if group:
                for op in group:
                    eig = getattr(op, 'eigval', None)
                    if eig is None:
                        continue
                    try:
                        if not np.isreal(eig):
                            has_cpx = True
                            break
                    except Exception:
                        if isinstance(eig, complex) and getattr(eig, 'imag', 0) != 0:
                            has_cpx = True
                            break

            self._has_complex_symmetries = bool(has_cpx)
        except Exception:
            # On unexpected errors, be conservative and assume complex
            self._has_complex_symmetries = True
    
    # --------------------------------------------------------------------------------------------------
    
    def _init_representatives(self, gen : list, gen_mapping : bool = False):
        """
        Initialize the representatives list and norms based on the provided symmetry generators.

        The representative process:
        1. For each state in the full Hilbert space, check if it satisfies global symmetries (e.g., U(1))
        2. Find the representative state (minimum state in symmetry orbit) using local symmetries
        3. If the state is its own representative, calculate normalization and add to mapping
        4. Optionally build full representative map (reprmap) for fast lookup
        
        Parameters:
        -----------
            gen (list):
                A list of (SymmetryGenerator, sector_value) tuples.
            gen_mapping (bool): 
                If True, generate the full representative map for all states.
                This uses more memory but speeds up repeated representative lookups.
        """
        
        if not self._is_many_body:
            self._log("Skipping mapping initialization in quadratic mode.", log='debug', lvl=2)
            return
        
        # Trivial case: no symmetries
        if not gen and not self._global_syms and self._state_filter is None:
            # No symmetries -> trivial mapping: every full-state is its own
            # representative. Set representative_list and representative_norms to None.
            self._log("No symmetries provided, generating trivial mapping (identity).", log='debug', lvl=2)
            try:
                nh_full = int(self._nhfull)
            except Exception:
                nh_full = self.local_space.local_dim ** int(self._ns)
            self.representative_list    = None
            self.representative_norms   = None
            self._nh                    = nh_full
            self._modifies              = False
            return
        
        # Use SymmetryContainer for symmetry group construction
        t0 = time.time()
        self._init_sym_container(gen)
        
        if gen is not None and len(gen) > 0:
            self._log("Generating the mapping of the states...", lvl = 1, color = 'green')

        # generate the mapping of the states
        if self._state_type == int:
            self._generate_repr_int(gen_mapping)
        else:
            self._generate_repr_base(gen_mapping)

        if gen is not None and len(gen) > 0 or (self.representative_list is not None and len(self.representative_list) > 0):
            self.representative_list    = self._backend.array(self.representative_list, dtype = self._backend.int64)
            self.representative_norms   = self._backend.array(self.representative_norms, dtype = self._dtype)
            self._log(f"Generated the mapping of the states in {time.time() - t0:.2f} seconds.", lvl = 2, color = 'green')
        else:
            self._log("No mapping generated.", lvl = 1, color = 'green', log = 'debug')

        self._sym_container.set_repr_info(self.representative_list, self.representative_norms)
        
        # Always generate the full mapping and compact structure when symmetries are present
        # This enables O(1) JIT-friendly lookups for matrix element computation
        has_symmetries = (self.representative_list is not None and len(self.representative_list) > 0)
        
        if has_symmetries:
            # Build full mapping arrays (repr_idx, repr_phase) if not already done
            if not hasattr(self, 'full_to_representative_idx') or self.full_to_representative_idx is None:
                self._repr_kernel_int_repr()
            
            # Build the compact symmetry data structure for O(1) JIT lookups
            # This is the PRIMARY and ONLY storage for symmetry mapping data
            # Memory efficient: ~5 bytes per state (uint32 + uint8) vs ~24+ bytes
            self._sym_container.build_compact_map(
                self._nhfull,
                self.full_to_representative_idx,
                self.full_to_representative_phase
            )
            
            # Clear the temporary full arrays to save memory - data is now in compact_data
            # The compact_data contains all needed information for O(1) lookups
            del self.full_to_representative_idx
            del self.full_to_representative_phase
            self.full_to_representative_idx = None
            self.full_to_representative_phase = None


    # --------------------------------------------------------------------------------------------------
    
    def _repr_kernel_int(self, start: int, stop: int, t: int):
        """
        For a given range of states in the full Hilbert space, find those states
        that are representatives (i.e. the smallest state under symmetry operations)
        and record their normalization.
        
        This is the core algorithm for building the symmetry-reduced Hilbert space:
        1. Check if state satisfies all global symmetries (e.g., particle number conservation)
        2. Find the representative state by applying all symmetry operations
        3. If state is its own representative, calculate normalization and add to mapping
        
        The normalization factor accounts for the size of the symmetry orbit and ensures
        proper overlap calculations in the reduced basis.
        
        Parameters:
        -----------
            start (int): 
                The starting index of the range (inclusive).
            stop (int): 
                The stopping index of the range (exclusive).
            t (int): 
                The thread number (for parallel execution).

        Returns:
            tuple: (map_threaded, norm_threaded)
                - map_threaded: 
                    List of representative states in this range
                - norm_threaded: 
                    Corresponding normalization factors
        """
        map_threaded    = []
        norm_threaded   = []
        
        for j in range(start, stop):
            if self._state_filter is not None and not self._state_filter(j):
                continue
            
            # Check all global symmetries (e.g., U(1) particle number)
            global_checker = True
            if self._global_syms:
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                    
            # if the global symmetries are not satisfied, skip the state
            if not global_checker:
                continue
            
            # Find the representative of state j under local symmetries
            rep, _ = self._sym_container._find_representative(j)
            
            # Only add the state if it is its own representative
            # (this ensures each symmetry sector is represented once)
            if rep == j:
                if self._state_filter is not None and not self._state_filter(rep):
                    continue
                
                # Calculate normalization using character-based projection
                # This ensures proper momentum sector filtering
                n = self._sym_container.compute_normalization(j)
                if abs(n) > 1e-7:  # Only add if normalization is non-zero
                    map_threaded.append(j)
                    norm_threaded.append(n)
                    
        return map_threaded, norm_threaded
    
    def _repr_kernel_int_repr(self):
        """
        For all states in the full Hilbert space, determine the representative.
        For each state j, if global symmetries are not conserved, record a bad mapping.
        Otherwise, if j is already in the mapping, record trivial normalization.
        Otherwise, find the representative and record its normalization.
        
        This function is created whenever one wants to create a full map for the Hilbert space 
        and store it in the mapping.
        """
        try:
            from QES.general_python.common.binary import bin_search
        except ImportError as e:
            raise RuntimeError("Failed to import binary search utilities. Ensure QES.general_python.common.binary is available.") from e
        
        # Build jittable repr_idx and repr_phase arrays without creating
        # the legacy object-based `reprmap`. This is memory-efficient and
        # sufficient for numba builders.
        repr_idx_arr    = np.full(self._nhfull, -1, dtype=np.int64)
        repr_phase_arr  = np.zeros(self._nhfull, dtype=np.complex128)
        mapping_size    = len(self.representative_list) if self.representative_list is not None else 0

        # for j in numba.prange(self._nhfull):
        # Use regular range instead of numba.prange to allow Python function calls
        #!TODO: Consider jitting this function if performance is critical
        for j in range(self._nhfull):
            if self._state_filter is not None and not self._state_filter(j):
                continue

            # Check global symmetries
            if self._global_syms:
                ok = True
                for g in self._global_syms:
                    ok = ok and g(j)
                if not ok:
                    continue

            # If mapping explicitly lists this state, record trivial phase
            if mapping_size > 0:
                idx = bin_search.binary_search_numpy(self.representative_list, 0, mapping_size - 1, j)
                if idx != bin_search._BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                    repr_idx_arr[j]     = int(idx)
                    repr_phase_arr[j]   = 1+0j
                    continue

            # Otherwise, find representative and see if its representative is present
            rep, sym_eig = self.find_repr(j)
            if mapping_size > 0:
                idx = bin_search.binary_search_numpy(self.representative_list, 0, mapping_size - 1, rep)
                if idx != bin_search._BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                    if self._state_filter is not None and not self._state_filter(rep):
                        continue
                    repr_phase_arr[j]   = complex(sym_eig)
                    repr_idx_arr[j]     = int(idx)

        # Expose jittable arrays for builders (created only on demand).
        # Use distinct internal names to avoid confusion with the compact
        # `self.representative_list` which contains only representatives.
        self.full_to_representative_idx     = repr_idx_arr
        self.full_to_representative_phase   = repr_phase_arr
    
    # --------------------------------------------------------------------------------------------------
    
    def _generate_repr_int(self, gen_mapping : bool = False):
        """
        Generate the mapping of the states for the Hilbert space.
        
        Parameters:
        -----------
            gen_mapping (bool): A flag to generate the mapping of the representatives to the original states.
        """
        
        # no symmetries - no mapping
        if len(self._sym_container.generators) == 0 and (self._global_syms is None or len(self._global_syms) == 0):
            self._nh = self._nhfull
            return
        
        # Use self.threadNum if set; otherwise default to 1.
        fuller = self._nhfull
        if getattr(self, '_threadnum', 1) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results: List[Tuple[List[int], List[float]]] = []
            with ThreadPoolExecutor(max_workers=self._threadnum) as executor:
                futures = []
                for t in range(self._threadnum):
                    start   = int(fuller * t / self._threadnum)
                    stop    = fuller if (t + 1) == self._threadnum else int(fuller * (t + 1) / self._threadnum)
                    futures.append(executor.submit(self._repr_kernel_int, start, stop, t))
                
                # Collect results as they complete
                for future in as_completed(futures):
                    results.append(future.result())
                
            combined = []
            for mapping_chunk, norm_chunk in results:
                combined.extend(zip(mapping_chunk, norm_chunk))
                
            # Sort combined results by state index to ensure consistent ordering
            combined.sort(key=lambda item: item[0])
            self.representative_list  = [state for state, _ in combined]
            self.representative_norms = [norm for _, norm in combined]
            
            # Clear temporary results to free memory
            results.clear()
            combined.clear()
        else:
            self.representative_list, self.representative_norms = self._repr_kernel_int(0, fuller, 0)
        
        #! Set the new Hilbert space size
        self._nh = len(self.representative_list)
        
        # Set repr_info BEFORE generating full mapping so find_representative works correctly
        # Convert to backend arrays first if needed
        if isinstance(self.representative_list, list):
            repr_list_array     = self._backend.array(self.representative_list, dtype=self._backend.int64)
            repr_norms_array    = self._backend.array(self.representative_norms, dtype=self._dtype)
        else:
            repr_list_array     = self.representative_list
            repr_norms_array    = self.representative_norms
        
        self._sym_container.set_repr_info(repr_list_array, repr_norms_array)
        
        #! Generate full mapping if requested!
        if gen_mapping:
            self._repr_kernel_int_repr()

    def _generate_repr_base(self, gen_mapping : bool = False):
        """
        Generate the mapping of the states for the Hilbert space.
        
        Args:
            gen_mapping (bool): A flag to generate the mapping of the representatives to the original states.
        """
        pass
    
    ####################################################################################################
    #! BASIS REPRESENTATION MANAGEMENT
    ####################################################################################################
    
    def _infer_and_set_default_basis(self, explicit_basis: Optional[str] = None):
        """
        Infer and set the default basis for this Hilbert space based on system properties.
        
        If explicit_basis is provided, use that instead of inferring.
        
        Logic (for inference):
        - Quadratic systems with lattice        : REAL (position/momentum basis)
        - Quadratic systems without lattice     : FOCK (single-particle occupation basis)
        - Many-body systems with lattice        : REAL (position space on lattice sites)
        - Many-body systems without lattice     : COMPUTATIONAL (integer/Fock basis)

        Parameters
        ----------
        explicit_basis : Optional[str]
            If provided, use this basis type instead of inferring.
            Valid values: "real", "k-space", "fock", "sublattice", "symmetry"
        
        This is called during initialization to establish the natural basis.
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType
        
        if explicit_basis is not None:
            # Use provided basis
            if isinstance(explicit_basis, str):
                self._basis_type = HilbertBasisType.from_string(explicit_basis)
            else:
                self._basis_type = explicit_basis
            self._log(f"HilbertSpace basis explicitly set to: {self._basis_type}", lvl=2, color="cyan")
            return
        
        # Infer basis from system properties
        if self._is_quadratic:
            # Quadratic system: choose based on lattice availability
            if self._lattice is not None:
                default_basis = HilbertBasisType.REAL
                basis_reason = "quadratic-lattice"
            else:
                default_basis = HilbertBasisType.FOCK
                basis_reason = "quadratic-fock"
        else:
            # Many-body system: choose based on lattice availability
            if self._lattice is not None:
                default_basis = HilbertBasisType.REAL
                basis_reason = "many-body-lattice"
            else:
                default_basis = HilbertBasisType.FOCK
                basis_reason = "many-body-computational"

        # Set the basis type
        self._basis_type = default_basis
        self._log(f"HilbertSpace default basis inferred: {default_basis} ({basis_reason})", lvl=2, color="cyan", log='debug')
    
    ####################################################################################################
    #! Getters and checkers for the Hilbert space
    ####################################################################################################
        
    # --------------------------------------------------------------------------------------------------
    #! FLUX
    # --------------------------------------------------------------------------------------------------
    
    @property
    def boundary_flux(self):
        """Return the boundary flux specification applied to the lattice."""
        return self._boundary_flux

    @boundary_flux.setter
    def boundary_flux(self, value: Optional[Union[float, Dict['LatticeDirection', float]]]):
        self._boundary_flux = value
        if self._lattice is not None:
            self._lattice.flux = value

    @property
    def state_filter(self) -> Optional[Callable[[int], bool]]:
        """Predicate applied to integer states during mapping generation."""
        return self._state_filter

    @state_filter.setter
    def state_filter(self, value: Optional[Callable[[int], bool]]):
        self._state_filter = value    

    @property
    def modifies(self):
        """
        Return the flag for modifying the Hilbert space.
        
        Returns:
            bool: The flag for modifying the Hilbert space.
        """
        if self._is_quadratic:
            return False
        
        return self.representative_list is not None and self._nh != self._nhfull
    
    @property
    def lattice(self) -> Optional['Lattice']: 
        return self._lattice
    
    def get_basis(self):
        """
        Get the current basis representation type.
        
        Returns
        -------
        HilbertBasisType
            Current basis type (default: REAL)
        """
        return getattr(self, '_basis_type', self._get_default_basis())
    
    def set_basis(self, basis_type: str):
        """
        Set the basis representation type.
        
        Parameters
        ----------
        basis_type : str or HilbertBasisType
            Target basis ("real", "k-space", "fock", "sublattice", "symmetry")
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType
        
        if isinstance(basis_type, str):
            self._basis_type = HilbertBasisType.from_string(basis_type)
        else:
            self._basis_type = basis_type
        
        self._log(f"Hilbert space basis type set to: {self._basis_type}", log='debug', lvl=2, color='blue')
    
    def _get_default_basis(self):
        """Get default basis type."""
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType
        return HilbertBasisType.REAL
    
    @property
    def sites(self):                            return self._ns
    @property
    def n_sites(self):                          return self._ns
    @property
    def Ns(self):                               return self._ns
    @property
    def ns(self):                               return self._ns
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def mapping(self):                          return self.representative_list
    @property
    def repr_list(self):                        return self.representative_list
    @property
    def repr_norms(self):                       return self.representative_norms
    @property
    def dtype(self):                            return self._dtype
        
    @property
    def sym_group(self):
        """
        Return the symmetry group.
        
        Returns:
            list: The symmetry group.
        """
        # Expose a callable view for external consumers (tests, utilities)
        # while keeping the internal tuple representation for JIT routines.
        if self._sym_group and isinstance(self._sym_group[0], tuple):
            def wrap_ops_tuple(ops_tuple):
                def op(state):
                    st = state
                    phase = 1.0
                    for g in ops_tuple:
                        # Each g is an Operator; call and accumulate phase
                        st, val = g(st)
                        try:
                            phase = phase * val
                        except Exception:
                            # If phase types differ (e.g., real vs complex), coerce via multiplication
                            phase = phase * (val.real if hasattr(val, 'real') else val)
                    return st, phase
                return op

            return [wrap_ops_tuple(t) for t in self._sym_group]
        return self._sym_group

    @property
    def check_global_symmetry(self):
        """
        Check if there are any global symmetries.
        """
        return len(self._global_syms) > 0 if self._global_syms is not None else False
        
    def get_sym_info(self):
        """
        Creates the information string about the Hilbert space and symmetries.
        
        Returns:
            str: A string containing the information about all the symmetries.
        """
        tmp = ""
        # Use SymmetryContainer generator specs for local symmetry summary when available
        if self._sym_container is not None and getattr(self._sym_container, 'generators', None):
            for op, (gen_type, sector) in self._sym_container.generators:
                tmp += f"{gen_type}={sector},"
                
        if self.check_global_symmetry:
            # start with global symmetries
            for g in self._global_syms:
                tmp += f"{g.get_name()}={g.get_val():.2f},"
        
        # remove last ","
        if tmp:
            tmp = tmp[:-1]
        
        return tmp
    
    # --------------------------------------------------------------------------------------------------
    #! Symmetry Data Properties
    # --------------------------------------------------------------------------------------------------
    
    @property
    def repr_idx(self) -> Optional[np.ndarray]:
        """
        Representative index array: state -> representative index.
        Returns compact_data.repr_map if available (uint32), otherwise None.
        Value = 0xFFFFFFFF (4294967295) if state not in this symmetry sector.
        """
        if self._sym_container is not None and self._sym_container.has_compact_data:
            return self._sym_container.compact_data.repr_map
        return None
    
    @property
    def repr_phase(self) -> Optional[np.ndarray]:
        """
        Get symmetry phases for all states.
        Returns phase_table[phase_idx] expanded for each state, or None if no symmetries.
        For efficient JIT use, prefer compact_symmetry_data.phase_idx and .phase_table directly.
        """
        if self._sym_container is not None and self._sym_container.has_compact_data:
            cd = self._sym_container.compact_data
            # Return phases for all states (expand from phase_idx)
            return cd.phase_table[cd.phase_idx]
        return None
    
    @property
    def has_complex_symmetries(self) -> bool:   
        """Check if the symmetry phases are complex (not just real +/-1)."""
        return bool(getattr(self, '_has_complex_symmetries', False))
    
    @property
    def normalization(self) -> Optional[np.ndarray]:                    
        """Normalization factors for representatives. Same as representative_norms."""
        return self.representative_norms

    @property
    def has_compact_symmetry_data(self) -> bool:
        """Check if compact symmetry data is available for O(1) lookups."""
        return self._sym_container is not None and self._sym_container.has_compact_data
    
    @property
    def compact_symmetry_data(self):
        """
        Get the CompactSymmetryData structure for efficient O(1) JIT lookups.
        
        This is the PRIMARY storage for symmetry mapping. Memory efficient:
        ~5 bytes per state (uint32 repr_map + uint8 phase_idx) vs ~24+ bytes.
        
        Returns
        -------
        CompactSymmetryData or None
            The compact symmetry data structure with:
            - repr_map: uint32[nh_full] - state -> representative index (0xFFFFFFFF = not in sector)
            - phase_idx: uint8[nh_full] - state -> phase table index
            - phase_table: complex128[n_phases] - distinct phases (small, ~group_order)
            - normalization: float64[n_repr] - normalization per representative
            - representative_list: int64[n_repr] - representative state values
            
        Example
        -------
        >>> cd = hilbert.compact_symmetry_data
        >>> if cd is not None:
        ...     idx = cd.repr_map[state]
        ...     if idx != 0xFFFFFFFF:
        ...         phase = cd.phase_table[cd.phase_idx[state]]
        ...         norm = cd.normalization[idx]
        """
        if self._sym_container is None:
            return None
        return self._sym_container.compact_data
    
    @property
    def symmetry_directory_name(self) -> str:
        """
        Return a combined string of all symmetry sectors suitable for directory names.
        
        This property generates a filesystem-safe string by combining the directory
        names of all active symmetry generators. Useful for organizing data files
        by symmetry sector.
        
        The format is: '{sym1}_{sym2}_{...}' where each sym is the directory_name
        of the corresponding symmetry operator (e.g., 'kx_0', 'pz_p', 'inv_m').
        
        Returns 'nosym' if no symmetries are active.
        
        Returns
        -------
        str
            Filesystem-safe combined symmetry name string.
            
        Examples
        --------
        >>> # Single translation symmetry
        >>> hilbert = HilbertSpace(lattice, sym_gen={'translation': 0})
        >>> hilbert.symmetry_directory_name
        'kx_0'
        
        >>> # Multiple symmetries
        >>> hilbert = HilbertSpace(lattice, sym_gen={
        ...     'translation': {'kx': 0, 'ky': 2},
        ...     'inversion': 1
        ... })
        >>> hilbert.symmetry_directory_name
        'kx_0_ky_2_inv_p'
        
        >>> # No symmetries
        >>> hilbert = HilbertSpace(lattice)
        >>> hilbert.symmetry_directory_name
        'nosym'
        """
        if self._sym_container is None:
            return "nosym"
        
        generators = getattr(self._sym_container, 'generators', [])
        if not generators:
            return "nosym"
        
        # Collect directory names from all generator operators
        # generators is a list of (operator, (gen_type, sector)) tuples
        names = []
        for item in generators:
            if isinstance(item, tuple) and len(item) >= 1:
                op = item[0]  # First element is the operator
            else:
                op = item
            
            if hasattr(op, 'directory_name'):
                names.append(op.directory_name)
            elif hasattr(op, 'sector'):
                # Fallback for operators without directory_name
                from QES.Algebra.Symmetries.base import SymmetryOperator
                sector_str = SymmetryOperator._sector_to_str(op.sector)
                names.append(f"{op.__class__.__name__.lower()}_{sector_str}")
        
        if not names:
            return "nosym"
        
        return "_".join(names)
    
    @property
    def full_directory_name(self) -> str:
        """
        Return a complete directory name including lattice info and symmetries.
        
        Format: 'L{lx}x{ly}x{lz}_{symmetry_directory_name}'
        
        For 1D: 'L8_kx_0_pz_p'
        For 2D: 'L4x4_kx_0_ky_2'
        For 3D: 'L2x2x2_inv_p'
        
        Returns
        -------
        str
            Complete filesystem-safe directory name.
            
        Examples
        --------
        >>> hilbert = HilbertSpace(lattice_4x4, sym_gen={'translation': 0})
        >>> hilbert.full_directory_name
        'L4x4_kx_0'
        
        >>> # Without lattice, uses ns (number of sites)
        >>> hilbert = HilbertSpace(ns=8, sym_gen={'parity': 1})
        >>> hilbert.full_directory_name
        'ns8_pz_p'
        """
        return f"{self.lattice_directory_name}_{self.symmetry_directory_name}"
    
    @property
    def lattice_directory_name(self) -> str:
        """
        Return the lattice part of the directory name.
        
        Format depends on dimensionality:
        - 1D: 'L{lx}'        e.g., 'L8'
        - 2D: 'L{lx}x{ly}'   e.g., 'L4x4'
        - 3D: 'L{lx}x{ly}x{lz}' e.g., 'L2x2x2'
        - No lattice: 'ns{ns}' e.g., 'ns8'
        
        Returns
        -------
        str
            Filesystem-safe lattice string.
        """
        if self._lattice is not None:
            lx = getattr(self._lattice, 'Lx', getattr(self._lattice, 'lx', 1))
            ly = getattr(self._lattice, 'Ly', getattr(self._lattice, 'ly', 1))
            lz = getattr(self._lattice, 'Lz', getattr(self._lattice, 'lz', 1))
            dim = getattr(self._lattice, 'dim', 1)
            
            if dim == 1 or (ly == 1 and lz == 1):
                return f"L{lx}"
            elif dim == 2 or lz == 1:
                return f"L{lx}x{ly}"
            else:
                return f"L{lx}x{ly}x{lz}"
        elif self._ns is not None and self._ns > 0:
            return f"ns{self._ns}"
        else:
            return "unknown"

    # --------------------------------------------------------------------------------------------------

    @property
    def local(self):                            return self._local_space.local_dim if self._local_space else 2
    @property
    def local_space(self):                      return self._local_space

    # --------------------------------------------------------------------------------------------------
    #! Local operator builders
    # --------------------------------------------------------------------------------------------------

    def list_local_operators(self):
        """
        Return the identifiers of all onsite operators available in the local space.
        """
        if self._local_space is None:
            return tuple()
        return self._local_space.list_operator_keys()
    
    def get_operator_elem(self, col_idx: int):
        """
        Get the element of the local operator.
        """
        new_row, sym_eig = self.find_repr(col_idx)
        return new_row, sym_eig
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def dimension(self):                        return self._nh
    @property
    def dim(self):                              return self._nh
    @property
    def Nh(self):                               return self._nh
    @property
    def nh(self):                               return self._nh
    @property
    def hilbert_dim(self):                      return self._nh
    @property
    def full(self):                             return self._nhfull
    @property
    def Nhfull(self):                           return self._nhfull
    @property
    def nhfull(self):                           return self._nhfull
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def quadratic(self):                        return self._is_quadratic
    @property
    def is_quadratic(self):                     return self._is_quadratic
    @property
    def many_body(self):                        return self._is_many_body
    @property
    def is_many_body(self):                     return self._is_many_body
    @property
    def particle_conserving(self):              return self._particle_conserving
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def logger(self):                           return self._logger
    
    @property
    def operators(self):
        """
        Lazy-loaded operator module for convenient operator access.
        
        Returns
        -------
        OperatorModule
            Module providing operator factory functions based on the local space type.
            
        Examples
        --------
        >>> # For spin systems
        >>> hilbert         = HilbertSpace(ns=4, local_space='spin-1/2')
        >>> sig_x           = hilbert.operators.sig_x(sites=[0, 1])
        >>> sig_x_matrix    = sig_x.matrix

        >>> # For fermion systems
        >>> hilbert         = HilbertSpace(ns=4, local_space='fermion')
        >>> c_dag           = hilbert.operators.c_dag(sites=[0])
        >>> c_dag_matrix    = c_dag.matrix
        
        >>> # Get help on available operators
        >>> hilbert.operators.help()
        """
        if not hasattr(self, '_operator_module') or self._operator_module is None:
            from QES.Algebra.Operator.operator_loader import get_operator_module
            local_space_type = self._local_space.typ if self._local_space else None
            self._operator_module = get_operator_module(local_space_type)
        return self._operator_module
    
    ####################################################################################################
    #! Representation of the Hilbert space
    ####################################################################################################
    
    def __str__(self):
        """Short string summary of the Hilbert space."""
        mode = "Many-Body" if self._is_many_body else "Quadratic"
        info = f"{mode} Hilbert space: Ns={self._ns}, Nh={self._nh}"
        if self._is_many_body:
            info += f", Local={self._local_space}"
        if self._global_syms:
            gs = ", ".join(f"{g.get_name_str()}={g.get_val()}" for g in self._global_syms)
            info += f", GlobalSyms=[{gs}]"
        if self._sym_container and getattr(self._sym_container, 'generators', None):
            ls = ", ".join(f"{gen_type}={sector}" for (_, (gen_type, sector)) in self._sym_container.generators)
            info += f", LocalSyms=[{ls}]"
        return info

    def __repr__(self):
        sym_info = self.get_sym_info()
        base = "Single particle" if self._is_quadratic else "Many body"
        return f"{base} Hilbert space with {self._nh} states and {self._ns} sites; {self._local_space}. Symmetries: {sym_info}" if sym_info else ""

    ####################################################################################################
    #! Find the representative of a state
    ####################################################################################################

    def find_sym_repr(self, state, nb: float = 1) -> Tuple[int, Union[float, complex]]:
        """
        Find the representative (minimum state) in the orbit of a given state
        under the current symmetry sector.
        
        Parameters
        ----------
        state : int
            The state to find the representative for.
        
        Returns
        -------
        tuple
            (representative_state, symmetry_eigenvalue)
            - representative_state: The minimum state in the symmetry orbit
            - symmetry_eigenvalue: The phase factor from the symmetry operation
        """
        if self._sym_container is None:
            return state, 1.0
        return self._sym_container.find_representative(state, nb)
    
    def find_sym_norm(self, state) -> Union[float, complex]:
        """
        Compute normalization factor for a state in the current symmetry sector.
        
        Uses character-based projection formula for momentum sectors:
        N = sqrt(sum_{g in G} chi _k(g)^* <state|g|state>)
        
        Parameters
        ----------
        state : int or array
            State to compute normalization for
        
        Returns
        -------
        norm : float or complex
            Normalization factor (0 if state not in current sector)
        """
        if self._sym_container is None:
            return 1.0
        return self._sym_container.compute_normalization(state)
            
    def find_repr(self, state, nb: float = 1) -> Tuple[int, Union[float, complex]]:
        return self.find_sym_repr(state, nb)

    def find_norm(self, state):
        return self.find_sym_norm(state)
        
    def norm(self, state):
        return self.representative_norms[state] if state < len(self.representative_norms) else self.find_norm(state)
    
    ####################################################################################################
    #! Full Hilbert space generation
    ####################################################################################
    
    def get_full_map(self):
        if self.full_to_representative_idx is None or len(self.full_to_representative_idx) == 0:
            self.generate_full_map()
        return self.full_to_representative_idx
    
    def get_full_glob_map(self):
        if self.full_to_global_map is None or len(self.full_to_global_map) == 0:
            self.generate_full_glob_map()
        return self.full_to_global_map

    def generate_full_map(self):
        self._repr_kernel_int_repr()
        
    def generate_full_glob_map(self):
        if self.full_to_global_map is not None and len(self.full_to_global_map) > 0:
            return
        
        full_map = []
        
        if self._global_syms:
            for j in range(self._nhfull):
                global_checker = True
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                if global_checker:
                    full_map.append(j)
        self.full_to_global_map = np.array(full_map, dtype=np.int64)

    def expand_from_reduced_space(self, vec_reduced):
        """
        Expand a vector from the reduced symmetry sector back to the full Hilbert space.
        
        This method handles both global symmetries (via full_map) and local symmetries
        (via symmetry group expansion). If no symmetries are present, returns the input unchanged.
        """
        # Check if we have any symmetries that reduce the space
        if not self.modifies:
            return np.asarray(vec_reduced)
        
        # If we have global symmetries with a full map, use that
        if self.check_global_symmetry and self.full_to_global_map is not None:
            vec_full = np.zeros(self._nhfull, dtype=vec_reduced.dtype)
            for i, state in enumerate(self.full_to_global_map):
                if i < len(vec_reduced):
                    vec_full[state] = vec_reduced[i]
            return vec_full
        
        # For local symmetries, expand using the symmetry group
        if self.representative_list is not None and self.representative_norms is not None:
            vec_full = np.zeros(2**self._ns, dtype=vec_reduced.dtype)
            
            for i, rep in enumerate(self.representative_list):
                if i >= len(vec_reduced):
                    continue
                    
                coeff = vec_reduced[i] / self.representative_norms[i] if self.representative_norms[i] != 0 else vec_reduced[i]
                
                # Apply all symmetry operations to expand this representative
                for g in self.sym_group:
                    try:
                        state, phase = g(rep)
                        vec_full[int(state)] += coeff * np.conjugate(phase)
                    except Exception:
                        continue
                        
            return vec_full
        
        # Fallback: return unchanged
        return np.asarray(vec_reduced)

    ####################################################################################################
    #! Operators for the Hilbert space
    ####################################################################################################
    
    def __len__(self):                  return self._nh
    def __call__(self, i):              return self.find_repr(i)
    
    def __getitem__(self, i):
        """
        Return the i-th basis state of the Hilbert space.
        
        Args:
            i: The index of the basis state to return or a state to find the representative for.
        
        Returns:
            np.ndarray: The i-th basis state of the Hilbert space.
        """
        if isinstance(i, (int, np.integer)):
            return self.representative_list[i] if (self.representative_list is not None and len(self.representative_list) > 0) else i
        return self.find_repr(i)
    
    def __contains__(self, state):
        """
        Check if a state is in the Hilbert space.
        
        Args:
            state: The state to check.
        
        Returns:
            bool: True if the state is in the Hilbert space, False otherwise.
        """
        if isinstance(state, int):
            return state in self.representative_list if self.representative_list is not None else True
        
        rep, _ = self.find_repr(state)
        return rep in self.representative_list if self.representative_list is not None else True

    def __iter__(self):
        """
        Iterate over the basis states of the Hilbert space.
        
        Yields:
            int: The next basis state in the Hilbert space.
        """
        if self.representative_list is not None:
            for state in self.representative_list:
                yield state
        else:
            for state in range(self._nh):
                yield state
    
    def __array__(self, dtype=None):
        """
        Return the basis states of the Hilbert space as a NumPy array.
        
        Args:
            dtype: The desired data type of the array.
            

        Returns:
            np.ndarray: The basis states of the Hilbert space as a NumPy array.
        """
        if self.representative_list is not None:
            return np.array(self.representative_list, dtype=dtype)
        return np.arange(self._nh, dtype=dtype)

#####################################################################################################
#! End of file
#####################################################################################################