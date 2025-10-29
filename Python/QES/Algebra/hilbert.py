
"""
High-level Hilbert space class for quantum many-body systems.

File    : QES/Algebra/hilbert.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-02-01
Version : 1.0.0
Changes : 
    - 2025.02.01 : 1.0.0 - Initial version of the Hilbert space class. - MK
    - 2025.10.26 : 1.1.0 - Refactored symmetry group generation and added detailed logging. - MK
    - 2025.10.28 : 1.1.1 - Working on symmetry compatibility and modular symmetries. - MK
"""
import sys
import math
import time
import numpy as np

from abc import ABC
from functools import lru_cache
from typing import Union, Optional, Callable, List, Tuple, Dict, Any

# other
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    # general thingies
    from QES.general_python.common.flog import get_global_logger, Logger
    from QES.general_python.common.binary import bin_search
    from QES.general_python.lattices.lattice import Lattice, LatticeDirection
    
    # already imported from QES.general_python
    from QES.Algebra.Hilbert.hilbert_jit_states import get_backend, JAX_AVAILABLE, ACTIVE_INT_TYPE, maybe_jit

    #################################################################################################
    if JAX_AVAILABLE:
        from QES.general_python.algebra.utils import pad_array

    #################################################################################################
    from QES.Algebra.Operator.operator import ( 
        Operator, 
        LocalSpace, LocalSpaceTypes, 
        StateTypes,      
        SymmetryGenerators, GlobalSymmetries, OperatorTypeActing,
        operator_identity, operator_from_local
    )
    
    from QES.Algebra.globals                        import GlobalSymmetry
    from QES.Algebra.Symmetries.base                import SymmetryOperator
    from QES.Algebra.Symmetries.translation         import TranslationSymmetry
    from QES.Algebra.hilbert_config                 import HilbertConfig
    from QES.Algebra.Hilbert.hilbert_jit_methods    import get_mapping, find_representative_int, get_matrix_element, has_complex_symmetries
except ImportError as e:
    # Avoid exiting the entire test process; re-raise for clearer diagnostics upstream
    raise e

#####################################################################################################
#! Hilbert space class
#####################################################################################################

class HilbertSpace(ABC):
    """
    A class to represent a Hilbert space either in Many-Body Quantum Mechanics or Quantum Information Theory and non-interacting systems.
    """
    
    
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

    def _check_init_sym_errors(self, sym_gen, global_syms, gen_mapping):
        ''' Check for initialization symmetry errors '''
        if sym_gen is not None and not isinstance(sym_gen, (dict, list)):
            HilbertSpace._raise(HilbertSpace._ERRORS["sym_gen"])
        if not isinstance(global_syms, list) and global_syms is not None:
            HilbertSpace._raise(HilbertSpace._ERRORS["global_syms"])
        if not isinstance(gen_mapping, bool):
            HilbertSpace._raise(HilbertSpace._ERRORS["gen_mapping"])

    def _check_ns_infer(self, lattice, ns, nh):
        ''' Check and infer the system size Ns from provided parameters '''
        
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
                self._log(f"Inferred Ns={self._ns} from Nh={nh} and local_dim={_local_dim}", log='info', lvl=2)
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
    #! Initialization
    # --------------------------------------------------------------------------------------------------

    def __init__(self,
                # core definition - elements to define the modes
                ns              : Union[int, None]                  = None,
                lattice         : Union[Lattice, None]              = None,
                nh              : Union[int, None]                  = None,
                # mode specificaton
                is_manybody     : bool                              = True,
                part_conserv    : Optional[bool]                    = True,
                # local space properties - for many body
                sym_gen         : Union[dict, None]                 = None,
                global_syms     : Union[List[GlobalSymmetry], None] = None,
                gen_mapping     : bool                              = False,
                local_space     : Optional[LocalSpace]              = None,
                # general parameters
                state_type      : StateTypes                        = StateTypes.INTEGER,
                backend         : str                               = 'default',
                dtype           : np.dtype                          = np.float64,
                boundary_flux   : Optional[Union[float, Dict[LatticeDirection, float]]] = None,
                state_filter    : Optional[Callable[[int], bool]]   = None,
                logger          : Optional[Logger]                  = None,
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
            nhl (Optional[int], optional):
                Local Hilbert space dimension (per site). Default is 2.
            nhint (Optional[int], optional):
                Number of internal modes per site (e.g., fermions, bosons). Default is 1.
            sym_gen (Union[dict, None], optional):
                Dictionary specifying symmetry generators. Default is None.
            global_syms (Union[List[GlobalSymmetry], None], optional): 
                List of global symmetry objects. Default is None.
            gen_mapping (bool, optional):
                Whether to generate state mapping based on symmetries. Default is False.
            state_type (str, optional):
                Type of state representation (e.g., "integer"). Default is "integer".
            backend (str, optional):
                Backend to use for vectors and matrices. Default is 'default'.
            dtype (optional):
                Data type for Hilbert space arrays. Default is np.float64.
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
        
        self._logger        = logger if logger is not None else get_global_logger()
        
        #! initialize the backend for the vectors and matrices
        self._backend, self._backend_str, self._state_type = self.reset_backend(backend, state_type)
        self._dtype         = dtype if dtype is not None else self._backend.float64
        self._is_many_body  = is_manybody
        self._is_quadratic  = not is_manybody
        
        #! quick check
        self._check_init_sym_errors(sym_gen, global_syms, gen_mapping)

        #! set locals
        # If you have a LocalSpace.default(), use it; otherwise use your default spin-1/2 factory.
        self._local_space   = local_space if local_space is not None else LocalSpace.default()  # or default_spin_half_local_space()
        if self._local_space is None:
            raise ValueError("local_space must be provided or LocalSpace.default() must return a valid LocalSpace.")

        #! infer the system sizes
        self._check_ns_infer(lattice=lattice, ns=ns, nh=nh)
        self._boundary_flux = boundary_flux
        if self._lattice is not None and boundary_flux is not None:
            self._lattice.flux = boundary_flux

        #! State filtering - predicate to filter basis states
        self._state_filter = state_filter

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
        self._normalization         = []                            # normalization of the states - how to return to the representative
        self._sym_group             = []                            # main symmetry group (will be populated later)
        self._global_syms           = global_syms if global_syms is not None else []
        self._particle_conserving   = part_conserv

        # Symmetry container (new unified approach)
        self._sym_container         = None  # Will be initialized in _init_mapping
        
        # Cached flag indicating whether any configured symmetry eigenvalues
        # are complex. Computed during symmetry container initialization to
        # avoid repeated inspection at matrix-build time.
        self._has_complex_symmetries= False

        # initialize the properties of the Hilbert space
        self._mapping               = None
        # jittable arrays (repr_idx, repr_phase) are created when mapping is generated

        # setup the logger instance for the Hilbert space
        self._threadnum             = kwargs.get('threadnum', 1)    # number of threads to use

        if self._is_many_body:
            if gen_mapping:
                self._log("Explicitly requested immediate mapping generation.", log='debug', lvl=2)
            self._init_mapping(sym_gen, gen_mapping=gen_mapping)    # gen_mapping True here enables reprmap
        elif self._is_quadratic:
            self._log("Quadratic mode: Skipping symmetry mapping generation.", log='debug', lvl=2)
            # Ensure mapping attributes are None
            self._mapping       = None
            self._normalization = None
            # Setup sym container if generators provided, but don't build map
            if sym_gen:
                self._init_sym_container(sym_gen)
                if self._sym_container is not None:
                    self._sym_group = list(self._sym_container.symmetry_group)

        # Ensure symmetry group always has at least the identity
        if not self._sym_group or len(self._sym_group) == 0:
            self._sym_group = [operator_identity(self._backend)]
    
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
            _backend_str   = backend
            _backend       = get_backend(backend)
        else:
            _backend_str   = 'np' if backend == np else 'jax'
            _backend       = backend
        
        statetype = HilbertSpace.reset_statetype(state_type, _backend)
        return _backend, _backend_str, statetype

    # --------------------------------------------------------------------------------------------------
    #! Momentum-sector helpers
    # --------------------------------------------------------------------------------------------------

    def analyze_momentum_sectors(self, directions: Optional[List[LatticeDirection]] = None, verbose: bool = False) -> Dict:
        """
        Analyze momentum sector structure for the Hilbert space.
        
        Automatically detects dimensionality and handles:
        - 1D systems    : single k quantum number
        - 2D systems    : (k_x, k_y) quantum numbers  
        - Lattices with multiple sites per unit cell
        
        Parameters
        ----------
        directions : List[LatticeDirection], optional
            Translation directions to analyze. If None, uses all active directions.
        verbose : bool
            Print detailed sector analysis.
        
        Returns
        -------
        Dict
            Momentum sector structure with representatives and quantum numbers.
            Format depends on dimensionality:
            - 1D: {k: [(rep, info), ...]}
            - 2D: {(k_x, k_y): [(rep, info), ...]}
        """
        from QES.Algebra.Symmetries.momentum_sectors import MomentumSectorAnalyzer
        
        if self._lattice is None:
            raise ValueError("Momentum sector analysis requires a lattice.")
        
        analyzer = MomentumSectorAnalyzer(self._lattice)
        
        if analyzer.dim == 0:
            raise ValueError("Lattice has no active translation directions.")
        elif analyzer.dim == 1:
            direction = directions[0] if directions else analyzer.active_directions[0]
            return analyzer.analyze_1d_sectors(direction=direction, verbose=verbose)
        elif analyzer.dim >= 2:
            if directions and len(directions) >= 2:
                dirs = (directions[0], directions[1])
            else:
                dirs = None
            return analyzer.analyze_2d_sectors(directions=dirs, verbose=verbose)
        else:
            raise NotImplementedError("3D momentum analysis not yet implemented")
    
    def get_momentum_representatives(
        self,
        momentum_indices: Optional[Dict[LatticeDirection, int]] = None
    ) -> List[int]:
        """
        Get representative states for specified momentum sector.
        
        Parameters
        ----------
        momentum_indices : Dict[LatticeDirection, int], optional
            Momentum quantum numbers for each translation direction.
            If None, returns all representatives across all sectors.
        
        Returns
        -------
        List[int]
            Representative states in the momentum sector.
        
        Examples
        --------
        >>> # 1D chain with k=0 sector
        >>> reps = hilbert.get_momentum_representatives({LatticeDirection.X: 0})
        
        >>> # 2D lattice with (k_x, k_y) = (0, π) sector  
        >>> from QES.Algebra.Symmetries.translation import LatticeDirection
        >>> reps = hilbert.get_momentum_representatives({
        ...     LatticeDirection.X: 0,
        ...     LatticeDirection.Y: lattice.Ly // 2  # π sector
        ... })
        """
        from QES.Algebra.Symmetries.momentum_sectors import MomentumSectorAnalyzer
        
        if self._lattice is None:
            raise ValueError("Momentum sector analysis requires a lattice.")
        
        analyzer = MomentumSectorAnalyzer(self._lattice)
        return analyzer.get_sector_representatives(momentum_indices)
    
    def build_momentum_basis(
        self,
        momentum_indices: Dict[LatticeDirection, int],
        normalize: bool = True
    ) -> Dict[int, Dict[int, complex]]:
        """
        Build complete momentum-resolved basis for specified quantum numbers.
        
        Parameters
        ----------
        momentum_indices : Dict[LatticeDirection, int]
            Momentum quantum number for each translation direction.
        normalize : bool
            Whether to normalize the momentum eigenstates.
        
        Returns
        -------
        Dict[int, Dict[int, complex]]
            Mapping from representative to {basis_state: coefficient}.
            Each entry represents one momentum eigenstate.
        
        Examples
        --------
        >>> # Build k=π sector for 1D chain
        >>> basis = hilbert.build_momentum_basis({LatticeDirection.X: L//2})
        >>> # basis[rep] gives the momentum eigenstate built from representative 'rep'
        
        >>> # Build (k_x, k_y) = (0, π) sector for 2D lattice
        >>> basis = hilbert.build_momentum_basis({
        ...     LatticeDirection.X: 0,
        ...     LatticeDirection.Y: Ly//2
        ... })
        """
        from QES.Algebra.Symmetries.momentum_sectors import build_momentum_basis
        
        if self._lattice is None:
            raise ValueError("Momentum basis construction requires a lattice.")
        
        return build_momentum_basis(self._lattice, momentum_indices, normalize)

    def build_momentum_superposition(self,
                                    base_state      : int,
                                    momenta         : Optional[Dict[LatticeDirection, int]] = None,
                                    normalize       : bool = True,) -> Dict[int, complex]:
        """
        Construct a momentum-resolved superposition seeded from ``base_state``.

        If ``momenta`` is None, momentum indices are inferred from translation
        symmetries registered in the SymmetryContainer. Otherwise,
        the provided mapping must pair :class:`LatticeDirection` entries with
        the desired momentum integers (k-values).
        
        Parameters
        ----------
        base_state : int
            Seed state for the momentum superposition
        momenta : Dict[LatticeDirection, int], optional
            Momentum quantum numbers for each direction. If None, inferred from
            symmetry container.
        normalize : bool
            Whether to normalize the resulting superposition
        
        Returns
        -------
        Dict[int, complex]
            Mapping from basis states to coefficients in the momentum eigenstate
        """
        from QES.Algebra.Symmetries.translation import (
            TranslationSymmetry,
            build_momentum_superposition as _build_momentum_superposition,
        )

        if self._lattice is None:
            raise ValueError("Momentum superposition requires a lattice.")
        
        if self._sym_container is None:
            raise ValueError("No symmetry container initialized. Initialize with translation symmetries.")

        translations: Dict[LatticeDirection, TranslationSymmetry] = {}
        inferred_momenta: Dict[LatticeDirection, int] = {}

        # Extract translations from SymmetryContainer
        for op, (gen_type, sector) in self._sym_container.generators:
            if isinstance(op, TranslationSymmetry):
                translations[op.direction] = op
                if sector is not None:
                    inferred_momenta[op.direction] = int(sector)

        if momenta is None:
            momenta = inferred_momenta
        else:
            momenta = {direction: int(k) for direction, k in momenta.items()}
            for direction, translator in list(translations.items()):
                if direction not in momenta:
                    translations.pop(direction)

        if not translations:
            raise ValueError("No translation symmetries registered for momentum projection.")

        if set(momenta.keys()) != set(translations.keys()):
            missing = set(translations.keys()) - set(momenta.keys())
            raise ValueError(
                f"Momentum specification missing for directions: {', '.join(d.name for d in missing)}"
            )

        return _build_momentum_superposition(translations, int(base_state), momenta, normalize=normalize)
    
    @property
    def boundary_flux(self):
        """Return the boundary flux specification applied to the lattice."""
        return self._boundary_flux

    @boundary_flux.setter
    def boundary_flux(self, value: Optional[Union[float, Dict[LatticeDirection, float]]]):
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
        self._log("Reseting the local symmetries. Can be now recreated.", lvl = 2, log = 'debug')
        self._sym_group     = []
    
    # --------------------------------------------------------------------------------------------------
    
    def _log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (Union[int, str]) : The flag to log the message (default is 'info').
            lvl (int) : The level of the message.
        """
        
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[HilbertSpace] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)

    ####################################################################################################
    #! Unified symmetry container initialization
    ####################################################################################################
    
    def _init_sym_container(self, gen: list) -> None:
        """
        Initialize the unified SymmetryContainer (new approach).
        
        This replaces the old _gen_sym_group method with a cleaner, more modular approach.
        The SymmetryContainer handles:
        - Compatibility checking
        - Automatic group construction from generators
        - Representative finding
        - Normalization computation
        
        Args:
            gen (list): List of (SymmetryGenerator, sector_value) tuples
        """
        if (not gen or len(gen) == 0) and not self.check_global_symmetry():
            self._log("No symmetries provided; SymmetryContainer will use identity only.", 
                     lvl=1, log='debug', color='green')
            from QES.Algebra.Symmetries.symmetry_container import SymmetryContainer
            self._sym_container = SymmetryContainer(
                ns=self._ns,
                lattice=self._lattice,
                nhl=self._local_space.local_dim,
                backend=self._backend_str
            )
            return
        
        # Import the factory function
        from QES.Algebra.Symmetries.symmetry_container import create_symmetry_container_from_specs
        
        # Prepare generator specs
        generator_specs = gen.copy() if gen is not None else []
        
        # Create and initialize the container
        # The factory will handle all compatibility checking and filtering
        self._sym_container = create_symmetry_container_from_specs(
            ns=self._ns,
            generator_specs=generator_specs,
            global_syms=self._global_syms,
            lattice=self._lattice,
            nhl=self._local_space.local_dim,
            backend=self._backend_str,
            build_group=True,
            build_repr_map=False  # Build on-demand for large systems
        )
        
        
        # Log symmetry info
        num_ops     = len(self._sym_container.symmetry_group)
        num_gens    = len(self._sym_container.generators)
        self._log(f"SymmetryContainer initialized: {num_gens} generators → {num_ops} group elements", 
                 lvl=1, color='green')

        # Compute and cache whether any symmetry eigenvalues/phases are complex.
        # Conservative: on unexpected errors assume True to avoid dropping
        # complex information.
        try:
            has_cpx = False
            # Prefer the canonical group stored on the SymmetryContainer when present
            group = getattr(self._sym_container, 'symmetry_group', None) or self._sym_group
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

            # secondary/group generators recorded in SymmetryContainer
            if not has_cpx and getattr(self, '_sym_container', None) and getattr(self._sym_container, 'generators', None):
                for op, _ in self._sym_container.generators:
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
            self._has_complex_symmetries = True
    
    # --------------------------------------------------------------------------------------------------
    
    def _init_mapping(self, gen : list, gen_mapping : bool = False):
        """
        Initialize the mapping of the states. This function generates the mapping of states 
        to their representatives based on symmetry operations and global symmetries.
        
        The mapping process:
        1. For each state in the full Hilbert space, check if it satisfies global symmetries (e.g., U(1))
        2. Find the representative state (minimum state in symmetry orbit) using local symmetries
        3. If the state is its own representative, calculate normalization and add to mapping
        4. Optionally build full representative map (reprmap) for fast lookup
        
        Args:
            gen (list): A list of (SymmetryGenerator, sector_value) tuples.
            gen_mapping (bool): If True, generate the full representative map for all states.
                               This uses more memory but speeds up repeated representative lookups.
        """
        if not self._is_many_body:
            self._log("Skipping mapping initialization in quadratic mode.", log='debug', lvl=2)
            return
        if not gen and not self._global_syms and self._state_filter is None:
            self._log("No symmetries provided, mapping is trivial (identity).", log='debug', lvl=2)
            self._nh            = self._nhfull
            self._mapping       = None
            self._normalization = None
            self._modifies      = False
            return
        
        t0 = time.time()
        
        # Use SymmetryContainer for symmetry group construction
        self._init_sym_container(gen)
        
        # Convert container's group to old format for compatibility with state mapping
        if self._sym_container is not None:
            self._sym_group = list(self._sym_container.symmetry_group)
        
        if gen is not None and len(gen) > 0:
            self._log("Generating the mapping of the states...", lvl = 1, color = 'green')

        # generate the mapping of the states
        if self._state_type == int:
            self._generate_mapping_int(gen_mapping)
        else:
            self._generate_mapping_base(gen_mapping)
        
        if gen is not None and len(gen) > 0 or len(self._mapping) > 0:
            t1 = time.time()
            self._log(f"Generated the mapping of the states in {t1 - t0:.2f} seconds.", lvl = 2, color = 'green')
            self._mapping           = self._backend.array(self._mapping, dtype = self._backend.int64)
        else:
            self._log("No mapping generated.", lvl = 1, color = 'green', log = 'debug')

    ####################################################################################################
    #! Getters and checkers for the Hilbert space
    ####################################################################################################
    
    # GLOBAL SYMMETRIES
    
    def check_global_symmetry(self):
        """
        Check if there are any global symmetries.
        """
        return len(self._global_syms) > 0 if self._global_syms is not None else False
    
    #---------------------------------------------------------------------------------------------------
    
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
        if self.check_global_symmetry():
            # start with global symmetries
            for g in self._global_syms:
                tmp += f"{g.get_name()}={g.get_val():.2f},"
        
        # remove last ","
        if tmp:
            tmp = tmp[:-1]
        
        return tmp
    
    # --------------------------------------------------------------------------------------------------
    
    def get_lattice(self):
        """
        Return the lattice object.
        
        Returns:
            Lattice: The lattice object.
        """
        return self._lattice
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def dtype(self):
        """
        Return the data type of the Hilbert space.
        
        Returns:
            type: The data type of the Hilbert space.
        """
        return self._dtype
    
    @property
    def modifies(self):
        """
        Return the flag for modifying the Hilbert space.
        
        Returns:
            bool: The flag for modifying the Hilbert space.
        """
        if self._is_quadratic:
            return False
        
        if self._mapping is None:
            return False
        elif self._mapping is not None:
            return self._nh != self._nhfull
        else:
            return False
        return self._nh != self._nhfull
    
    @property
    def sites(self):
        """
        Return the number of sites in the system.
        
        Returns:
            int: The number of sites in the system.
        """
        return self._ns
    
    @property
    def Ns(self):
        """
        Return the number of sites in the system.
        
        Returns:
            int: The number of sites in the system.
        """
        return self._ns
    
    @property
    def ns(self):
        """
        Return the number of sites in the system.
        
        Returns:
            int: The number of sites in the system.
        """
        return self._ns
    
    def get_Ns(self):
        """
        Return the number of sites in the system.
        
        Returns:
            int: The number of sites in the system.
        """
        return self._ns
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def mapping(self):
        """
        Return the mapping of the states.
        
        Returns:
            list: The mapping of the states.
        """
        return self._mapping
    
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
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def repr_idx(self):
        """Return the precomputed representative index array (jittable).

        This is an array of length Nh_full where entry j is the index of the
        representative in the reduced basis for state j, or -1 if not present.
        """
        return getattr(self, '_repr_idx', None)

    @property
    def repr_phase(self):
        """Return the precomputed representative phase array (jittable).

        Entry j contains the symmetry phase (complex) relating state j to its
        representative: G_j |rep> = phase * |state_j> (phase as complex128).
        """
        return getattr(self, '_repr_phase', None)

    @property
    def has_complex_symmetries(self) -> bool:
        """
        Return cached boolean indicating whether any configured symmetry
        eigenvalues/phases are complex. This value is computed during
        `_init_sym_container` and cached in `_has_complex_symmetries`.
        """
        return bool(getattr(self, '_has_complex_symmetries', False))
    
    @property
    def normalization(self):
        """
        Return the normalization of the states.
        
        Returns:
            list: The normalization of the states.
        """
        return self._normalization

    # --------------------------------------------------------------------------------------------------

    @property
    def local(self):
        """
        Return the local Hilbert space dimension.
        
        Returns:
            int: The local Hilbert space dimension.
        """
        return self._local_space.local_dim if self._local_space else 2
    
    @property
    def local_space(self):
        """
        Return the local Hilbert space.
        
        Returns:
            LocalSpace: The local Hilbert space.
        """
        return self._local_space

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

    def build_local_operator(self,
                             key            : str,
                             *,
                             type_override  : Optional[OperatorTypeActing] = None,
                             name           : Optional[str] = None) -> Operator:
        """
        Instantiate a registered local operator by name.

        Parameters
        ----------
        key:
            Identifier of the onsite operator as provided by the catalog.
        type_override:
            Optionally force the acting type (local/global/correlation).
        name:
            Optional display name for the resulting Operator.
        """
        if self._local_space is None:
            raise ValueError("Hilbert space was constructed without a local space definition.")
        local_op = self._local_space.get_op(key)
        return operator_from_local(
            local_op,
            lattice=self._lattice,
            ns=self._ns,
            name=name,
            type_override=type_override,
        )
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def full(self):
        """
        Return the full Hilbert space dimension.
        
        Returns:
            int: The full Hilbert space dimension.
        """
        return self._nhfull
    
    @property
    def Nhfull(self):
        """
        Return the full Hilbert space dimension.
        
        Returns:
            int: The full Hilbert space dimension.
        """
        return self._nhfull

    def get_Nh_full(self):
        """
        Return the full Hilbert space dimension.
        
        Returns:
            int: The full Hilbert space dimension.
        """
        return self._nhfull
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def dimension(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    @property
    def dim(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    @property
    def Nh(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    def get_Nh(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def quadratic(self):
        return self._is_quadratic
    
    @property
    def is_quadratic(self):
        return self._is_quadratic
    
    @property
    def many_body(self):
        return self._is_many_body
    
    @property
    def is_many_body(self):
        return self._is_many_body
    
    @property
    def particle_conserving(self):
        return self._particle_conserving
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def logger(self):
        """
        Return the logger instance.
        
        Returns:
            Logger: The logger instance.
        """
        return self._logger

    # --------------------------------------------------------------------------------------------------
    
    def norm(self, state):
        """
        Return the normalization of a given state.
        
        Args:
            state (int): The state to get the normalization for.
        
        Returns:
            float: The normalization of the state.
        """
        return self._normalization[state] if state < len(self._normalization) else 1.0
    
    ####################################################################################################
    #! Representation of the Hilbert space
    ####################################################################################################
    
    def __str__(self):
        """ Return a string representation of the Hilbert space. """
        mode    =  "Many-Body" if self._is_many_body else "Quadratic"
        info    = f"Mode: {mode}, Ns: {self._ns}, Nh: {self._nh}\n"
        if self._is_many_body:
            info += f"Local: {self._local_space}"

        if self._mapping is not None:
            info    += f"Reduced Hilbert space size (Nh): {self._nh}\n"
            info    += f"Number of symmetry operators considered: {len(self._sym_group)}\n"
        elif self._is_many_body:
            info    += f"Effective Hilbert space size (Nh): {self._nh} (Full space, no reduction applied)\n"
        else: # Quadratic
            info    += f"Effective Hilbert space size (Nh): {self._nh} (Quadratic basis size)\n"

        gs_info     = [f"{g.get_name_str()}={g.get_val()}" for g in self._global_syms]
        if gs_info:
            info += f"Global Symmetries: {', '.join(gs_info)}\n"
        # Provide local symmetry summary from SymmetryContainer if present
        if self._sym_container is not None and getattr(self._sym_container, 'generators', None):
            ls_info = [f"{gen_type}={sector}" for (_, (gen_type, sector)) in self._sym_container.generators]
            if ls_info:
                info += f"Local Symmetries: {', '.join(ls_info)}\n"
        return info

    def __repr__(self):
        sym_info = self.get_sym_info()
        base = "Single particle" if self._is_quadratic else "Many body"
        return f"{base} Hilbert space with {self._nh} states and {self._ns} sites; {self._local_space}. Symmetries: {sym_info}" if sym_info else ""

    ####################################################################################################
    #! Find the representative of a state
    ####################################################################################################
    
    def find_sym_norm(self, state) -> Union[float, complex]:
        """
        Compute normalization factor for a state in the current symmetry sector.
        
        Uses character-based projection formula for momentum sectors:
        N = sqrt(sum_{g in G} χ_k(g)^* <state|g|state>)
        
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
            raise ValueError("No symmetry container initialized. Cannot compute normalization.")
        return self._sym_container.compute_normalization(state)
    
    # --------------------------------------------------------------------------------------------------
    
    def find_repr(self, state):
        """
        Find the representative (minimum state) in the orbit of a given state.
        
        The representative is the smallest state (minimum integer value) that can be
        obtained by applying all symmetry operations in the symmetry group. This
        ensures a unique canonical form for each symmetry sector.

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
            raise ValueError("No symmetry container initialized. Cannot find representative.")
        return self._sym_container.find_representative(state, use_cache=True)
    
    # --------------------------------------------------------------------------------------------------
    
    def find_representative_int(self, state, normalization_beta):
        """
        Find the representative of a given state.
        """
        return find_representative_int(state, self._mapping, self._normalization, normalization_beta, self._sym_group)
    
    def find_representative(self, state, normalization_beta):
        """
        Finds the representative for a given base index in the "sector alfa" and applies
        normalization from "sector beta".

        This procedure is used when an operator acts on a representative |r> (in sector \alpha),
        transforming it to a state |m>. We then:
        1. Find the representative |r'> for state |m> 
        2. Determine the symmetry phase φ connecting |m> to |r'>
        3. Apply normalization: N_\beta / N_\alpha * φ*
        
        The result is the matrix element contribution with proper normalization and phase.

        Args:
            state: The state (integer or array) after operator action
            normalization_beta: The normalization factor N_\beta for the target sector
            
        Returns:
            tuple: (representative_index, normalization_factor)
                - representative_index: Index in the reduced basis
                - normalization_factor: N_\beta/N_\alpha * φ* where φ is the symmetry phase
        """
        if isinstance(state, int):
            return self.find_representative_int(state, normalization_beta)
        return self.find_representative_base(state, normalization_beta)
    
    # --------------------------------------------------------------------------------------------------
    
    def _mapping_kernel_int(self, start: int, stop: int, t: int):
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
            start (int): The starting index of the range (inclusive).
            stop (int): The stopping index of the range (exclusive).
            t (int): The thread number (for parallel execution).
            
        Returns:
            tuple: (map_threaded, norm_threaded)
                - map_threaded: List of representative states in this range
                - norm_threaded: Corresponding normalization factors
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
            rep, _ = self.find_repr(j)
            
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
    
    def _mapping_kernel_int_repr(self):
        """
        For all states in the full Hilbert space, determine the representative.
        For each state j, if global symmetries are not conserved, record a bad mapping.
        Otherwise, if j is already in the mapping, record trivial normalization.
        Otherwise, find the representative and record its normalization.
        
        This function is created whenever one wants to create a full map for the Hilbert space 
        and store it in the mapping.
        """
        
        # Build jittable repr_idx and repr_phase arrays without creating
        # the legacy object-based `reprmap`. This is memory-efficient and
        # sufficient for numba builders.
        repr_idx_arr    = np.full(self._nhfull, -1, dtype=np.int64)
        repr_phase_arr  = np.zeros(self._nhfull, dtype=np.complex128)
        mapping_size    = len(self._mapping) if self._mapping is not None else 0

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
                idx = bin_search.binary_search(self._mapping, 0, mapping_size - 1, j)
                if idx != bin_search._BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                    repr_idx_arr[j] = int(idx)
                    repr_phase_arr[j] = 1+0j
                    continue

            # Otherwise, find representative and see if its representative is present
            rep, sym_eig = self.find_repr(j)
            if mapping_size > 0:
                idx = bin_search.binary_search(self._mapping, 0, mapping_size - 1, rep)
                if idx != bin_search._BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                    if self._state_filter is not None and not self._state_filter(rep):
                        continue
                    try:
                        repr_phase_arr[j] = complex(sym_eig)
                    except Exception:
                        repr_phase_arr[j] = 0+0j
                    repr_idx_arr[j] = int(idx)

        # Expose jittable arrays for builders
        self._repr_idx = repr_idx_arr
        self._repr_phase = repr_phase_arr
    
    # --------------------------------------------------------------------------------------------------
    
    def _mapping_kernel_base(self):
        """
        """
        pass
    
    def _mapping_kernel_base_repr(self):
        """
        """
        pass
    
    # --------------------------------------------------------------------------------------------------
    
    def _generate_mapping_int(self, gen_mapping : bool = False):
        """
        Generate the mapping of the states for the Hilbert space.
        
        Args:
            gen_mapping (bool): A flag to generate the mapping of the representatives to the original states.
        """
        
        # no symmetries - no mapping
        if  (self._sym_group is None or len(self._sym_group) == 0) and (self._global_syms is None or len(self._global_syms) == 0):
            self._nh = self._nhfull
            return
        
        fuller      = self._nhfull
        # For demonstration, use self.threadNum if set; otherwise default to 1.
        if self._threadnum > 1:
            results: List[Tuple[List[int], List[float]]] = []
            with ThreadPoolExecutor(max_workers=self._threadnum) as executor:
                futures = []
                for t in range(self._threadnum):
                    start   = int(fuller * t / self._threadnum)
                    stop    = fuller if (t + 1) == self._threadnum else int(fuller * (t + 1) / self._threadnum)
                    futures.append(executor.submit(self._mapping_kernel_int, start, stop, t))
                for future in as_completed(futures):
                    results.append(future.result())
            combined = []
            for mapping_chunk, norm_chunk in results:
                combined.extend(zip(mapping_chunk, norm_chunk))
            combined.sort(key=lambda item: item[0])
            self._mapping       = [state for state, _ in combined]
            self._normalization = [norm for _, norm in combined]
        else:
            self._mapping, self._normalization = self._mapping_kernel_int(0, fuller, 0)
        self._nh = len(self._mapping)
        
        if gen_mapping:
            self._mapping_kernel_int_repr()

    def _generate_mapping_base(self, gen_mapping : bool = False):
        """
        Generate the mapping of the states for the Hilbert space.
        
        Args:
            gen_mapping (bool): A flag to generate the mapping of the representatives to the original states.
        """
        pass
    
    # --------------------------------------------------------------------------------------------------
    
    def get_matrix_element(self, k, new_k, kmap = None, h_conj = False):
        r"""
        Compute the matrix element between two states in the Hilbert space.
        This method determines the matrix element corresponding to the transition between a given state |k> and a new state defined by new_k.
        It accounts for the possibility that the new state may not be in its representative form, in which case it finds the representative state
        and applies the corresponding normalization factor or symmetry eigenvalue. The ordering of the returned tuple elements may be
        reversed based on the flag h_conj; if h_conj is False, the result is ((representative, k), factor), otherwise ((k, representative), factor).

        Imagine a situation where an operator acts on a state |k> and gives a new state <new_k|.
        We use this Hilbert space to find the matrix element between these two states. It may happen
        that the new state is not the representative of the original state, so we need to find the
        representative and the normalization factor.

        Args:
            k: An index or identifier for the original state in the Hilbert space.
            new_k: An index or identifier representing the new state after the operator action.
            h_conj (bool, optional): A flag to determine the order of the tuple. If False (default), the tuple is (representative, k), 
                                    otherwise it is (k, representative).
        Returns:
            tuple: A tuple consisting of:
                - A tuple of two elements representing the indices (or identifiers) of the representative state and the original state,
                    ordered based on the value of h_conj.
                - The normalization factor or symmetry eigenvalue associated with the new state. 
        Note:
            This function uses the find_representative function from this module to find the representative.        
        """
        return get_matrix_element(k, new_k, kmap, h_conj, self._mapping, self._normalization, self._sym_group, None)
    
    ####################################################################################################
    #! Full Hilbert space generation
    ####################################################################################################
    
    def generate_full_map_int(self):
        """
        Generate the full mapping of the Hilbert space.
        """
        self._fullmap = []
        if self._global_syms:
            for j in range(self._nhfull):
                if self._state_filter is not None and not self._state_filter(j):
                    continue
                global_checker = True
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                # if the global symmetries are satisfied, add the state to the full map
                if global_checker:
                    self._fullmap.append(j)

    def get_full_map_int(self):
        """
        Generate the full mapping of the Hilbert space.
        """
        if self._fullmap is not None and len(self._fullmap) > 0:
            return self._fullmap
        self.generate_full_map_int()
        return self._fullmap
    
    ####################################################################################################
    #! Operators for the Hilbert space
    ####################################################################################################
    
    def __len__(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    def get_mapping(self, i):
        """
        Return the mapping of the states.
        
        Returns:
            list: The mapping of the states.
        Note:
            This function uses the get_mapping function from this module to get the mapping.
        """
        return get_mapping(self.mapping, i)
    
    def __getitem__(self, i):
        """
        Return the i-th basis state of the Hilbert space.
        
        Args:
            i: The index of the basis state to return or a state to find the representative for.
        
        Returns:
            np.ndarray: The i-th basis state of the Hilbert space.
        """
        if isinstance(i, (int, np.integer)):
            return self._mapping[i] if (self._mapping is not None and len(self._mapping) > 0) else i
        return self._mapping[i]
    
    def __call__(self, i):
        """
        Return the representative of the i-th basis state of the Hilbert space.
        """
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
            return state in self._mapping
        #! TODO: implement the state finding
        return NotImplementedError("Only integer indexing is supported.")
    
    ################################################################################################

#####################################################################################################
#! End of file
#####################################################################################################