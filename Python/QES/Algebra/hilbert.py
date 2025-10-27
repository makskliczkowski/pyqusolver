
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
    
"""
import sys
import math
import time
import numpy as np

from abc import ABC
from functools import lru_cache
from typing import Union, Optional, Callable, List, Tuple, Dict

# other
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    # general thingies
    from QES.general_python.common.flog import get_global_logger, Logger
    from QES.general_python.common.binary import bin_search
    from QES.general_python.lattices.lattice import Lattice, LatticeBC, LatticeDirection
    
    # already imported from QES.general_python
    from QES.Algebra.Hilbert.hilbert_jit_states import get_backend, JAX_AVAILABLE, ACTIVE_INT_TYPE, maybe_jit

    #################################################################################################
    if JAX_AVAILABLE:
        from QES.general_python.algebra.utils import pad_array
    
    #################################################################################################
    from QES.Algebra.Operator.operator import ( Operator, LocalSpace, LocalSpaceTypes, StateTypes,      
                                            SymmetryGenerators, GlobalSymmetries, OperatorTypeActing,
                                            operator_identity, operator_from_local)
    from QES.Algebra.globals import GlobalSymmetry
    from QES.Algebra.symmetries import choose, translation
    from QES.Algebra.hilbert_config import HilbertConfig
    
    #################################################################################################
    #! WRAPPER FOR JIT AND NUMBA
    #################################################################################################

    from QES.Algebra.Hilbert.hilbert_jit_methods import (
        get_mapping, find_repr_int, find_representative_int, get_matrix_element,
        jitted_find_repr_int, jitted_get_mapping, jitted_get_matrix_element
    )   
except ImportError as e:
    # Avoid exiting the entire test process; re-raise for clearer diagnostics upstream
    raise e
 
#####################################################################################################

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    
    @jax.jit
    def get_mapping_jax(mapping, state):
        """
        Get the mapping of the state.
        
        Args:
            mapping (list): The mapping of the states.
            state (int): The state to get the mapping for.
        
        Returns:
            int: The mapping of the state.
        """
        return mapping[state] if len(mapping) > state else state

#####################################################################################################


@lru_cache(maxsize=128)
def _enumerate_generator_index_combos(num_generators: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Return all index combinations (including the empty tuple) for ``num_generators`` entries.
    """
    combos: List[Tuple[int, ...]] = [tuple()]
    if num_generators <= 0:
        return tuple(combos)
    indices = tuple(range(num_generators))
    for r in range(1, num_generators + 1):
        combos.extend(tuple(combo) for combo in combinations(indices, r))
    return tuple(combos)

#####################################################################################################
#! Hilbert space class
#####################################################################################################

class HilbertSpace(ABC):
    """
    A class to represent a Hilbert space either in Many-Body Quantum Mechanics or Quantum Information Theory and non-interacting systems.
    """
    
    #################################################################################################
    
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
        """
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
        
        # initialize the backend for the vectors and matrices
        self._backend, self._backend_str, self._state_type = self.reset_backend(backend, state_type)
        self._dtype         = dtype if dtype is not None else self._backend.float64
        self._is_many_body  = is_manybody
        self._is_quadratic  = not is_manybody
        
        # quick check
        self._check_init_sym_errors(sym_gen, global_syms, gen_mapping)

        # set locals
        # If you have a LocalSpace.default(), use it; otherwise use your default spin-1/2 factory.
        self._local_space   = local_space if local_space is not None else LocalSpace.default()  # or default_spin_half_local_space()
        if self._local_space is None:
            raise ValueError("local_space must be provided or LocalSpace.default() must return a valid LocalSpace.")

        # infer the system sizes
        self._check_ns_infer(lattice=lattice, ns=ns, nh=nh)
        self._boundary_flux = boundary_flux
        if self._lattice is not None and boundary_flux is not None:
            self._lattice.flux = boundary_flux
        self._state_filter = state_filter

        # Nh: Effective dimension of the *current* representation
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

        #! Initialize the symmetries (legacy and modular)
        self._normalization         = []                            # normalization of the states - how to return to the representative
        self._sym_group             = []
        self._sym_group_sec         = []
        self._global_syms           = global_syms if global_syms is not None else []
        self._particle_conserving   = part_conserv

        # Modular symmetry group (new, extensible)
        from QES.Algebra.symmetries import choose
        self._sym_group_modular = []
        if sym_gen is not None:
            for sym in sym_gen:
                op = choose(sym, lat=self._lattice)
                self._sym_group_modular.append(op)

        # initialize the properties of the Hilbert space
        self._mapping               = None
        self._reprmap               = None                          # mapping of the representatives (vector of tuples (state, representative value))
        self._fullmap               = None                          # mapping of the full Hilbert space

        self._getmapping_fun        = None                          # function to get the mapping of the states
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
            self._reprmap        = None
            # Setup sym group if generators provided, but don't build map
            if sym_gen:
                self._gen_sym_group(sym_gen) #!TODO: How to define this?

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
        self._sym_group_sec = []
    
    # --------------------------------------------------------------------------------------------------
    
    def log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = False):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (Union[int, str]) : The flag to log the message (default is 'info').
            lvl (int) : The level of the message.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        self._log(msg, log=log, lvl=lvl, color=color, append_msg=append_msg)
    
    def _log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (int) : The flag to log the message (default is 'info').
            lvl (int) : The level of the message.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[HilbertSpace] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log = log, lvl = lvl)
    
    ####################################################################################################
    #! Generate the symmetry group and all properties of the generation
    ####################################################################################################
    
    #! Translation related
    
    def _gen_sym_group_check_t(self, sym_gen : list) -> (list, Tuple[bool, bool], Tuple[Operator, LatticeDirection]):
        '''
        Helper function to check the translation symmetry. This function is used to check the translation symmetry.
        It gets the translation generator and checks if it satisfies the symmetry conditions. If the translation
        is not possible with the boundary conditions provided, the function returns.
        
        Args:
            sym_gen (list) : A list of symmetry generators.
        Returns:
            list, tuple : A list of symmetry generators and a tuple of flags (has_translation, has_cpx_translation).
        '''
        if self._lattice is None:
            has_translation = any(g[0].has_translation() for g in sym_gen if hasattr(g,'__len__') and len(g) > 0)
            if has_translation:
                    self._log("Translation requested but no Lattice; ignoring.", log='warning', lvl=1)
            # Remove translation generators if found
            sym_gen_out = [(g, s) for (g, s) in (sym_gen or []) if not g.has_translation()]
            return sym_gen_out, (False, False), (None, LatticeDirection.X)
        
        has_cpx_translation = False
        has_translation     = False
        t                   = None  
        direction           = LatticeDirection.X
        for idx, (gen, sec) in enumerate(sym_gen):

            if not gen.has_translation():
                continue

            direction = LatticeDirection.X
            if gen == SymmetryGenerators.Translation_y:
                direction = LatticeDirection.Y
            elif gen == SymmetryGenerators.Translation_z:
                direction = LatticeDirection.Z

            sym_gen.pop(idx)
            if not self._lattice.is_periodic(direction):
                self._log(f"Translation along {direction.name} requested but boundary is not periodic.", log='warning', lvl=1)
                break

            has_translation = True
            self._sym_group_sec.append((gen, sec))

            kx = sec if direction == LatticeDirection.X else 0.0
            ky = sec if direction == LatticeDirection.Y else 0.0
            kz = sec if direction == LatticeDirection.Z else 0.0
            t = translation(self._lattice, kx=kx, ky=ky, kz=kz, direction=direction, backend=self._backend)

            if sec != 0 and not (sec == self.Ns // 2 and self.Ns % 2 == 0):
                has_cpx_translation = True
            break
            
        if has_translation:
            self._log("Translation symmetry is present.", lvl = 1)
            if has_cpx_translation:
                self._log("Translation in complex sector...", lvl = 2, color = 'blue')
        return sym_gen, (has_translation, has_cpx_translation), (t, direction)

    def _gen_sym_apply_t(self, sym_gen_op : list, t : Optional[Operator] = None, direction : LatticeDirection = LatticeDirection.X):
        """
        Apply the translation symmetry to all existing symmetry group elements.
        
        This creates new symmetry operations by combining translation with existing operations.
        For a lattice with size L in the translation direction, we generate T, T^2, ..., T^(L-1)
        and combine each with existing symmetry operations.
        
        Args:
            sym_gen_op (list): Existing list of symmetry operator tuples
            t (Optional[Operator]): Translation operator
            direction (LatticeDirection): Direction of translation
            
        Returns:
            list: Extended list of symmetry operator tuples including translation combinations
        """
        if t is not None:
            self._log("Adding translation to symmetry group combinations.", lvl = 2, color = 'yellow')
            
            # check the direction to determine lattice size
            if direction == LatticeDirection.X:
                size = self._lattice.lx
            elif direction == LatticeDirection.Y:
                size = getattr(self._lattice, 'ly', self._lattice.lx)
            else:
                size = getattr(self._lattice, 'lz', self._lattice.lx)
            
            sym_gen_out = sym_gen_op.copy()
            
            # Generate T, T^2, T^3, ..., T^(size-1) and combine with existing ops
            t_powers = [t]  # T^1
            for power in range(2, size):
                # T^power = T applied power times
                t_powers.append(tuple([t] * power))
            
            # Combine each power of T with all existing symmetry operations
            for t_pow in t_powers:
                for op_tuple in sym_gen_op:
                    # Combine: prepend t_pow operations to op_tuple
                    if isinstance(t_pow, tuple):
                        combined = t_pow + op_tuple
                    else:
                        combined = (t_pow,) + op_tuple
                    sym_gen_out.append(combined)
                    
            return sym_gen_out
        return sym_gen_op

    #! Global symmetries related
    
    def _gen_sym_group_check_u1(self) -> (bool, float):
        """
        Check if a U(1) global symmetry is present.
        Returns (has_U1, U1_value).
        """
        has_u1, u1_val = self.check_u1()
        if has_u1:
            self._log("U(1) global symmetry is present.", lvl = 2, color = 'blue')
        return has_u1, u1_val

    #! Removers for the symmetry generators

    def _gen_sym_remove_reflection(self, sym_gen : list, has_cpx_translation : bool):
        """
        Helper function to remove reflections from the symmetry generators if the complex translation is present.
        
        Args:
            sym_gen (list)         : A list of symmetry generators.
            has_cpx_translation (bool) : A flag for complex translation - momentum is different than 0 or pi.
        """
        if has_cpx_translation and sym_gen is not None and hasattr(sym_gen, "__iter__"):
            sym_gen = [gen for gen in sym_gen if not isinstance(gen[0], SymmetryGenerators.Reflection)]
            self._log("Removed reflection symmetry from the symmetry generators.", lvl = 2, color = 'blue')
        return sym_gen
    
    def _gen_sym_remove_parity(self, sym_gen : list, has_u1 : bool, has_u1_sec : float):
        """
        If U(1) is present but the system is not at half-filling (or has odd size),
        remove parity generators in the X and/or Y directions.
        """
        if has_u1:
            
            self._log("U(1) symmetry detected. Checking parity generators...", log = 1, lvl = 1, color = 'yellow')
            
            new_sym_gen = []
            for (gen, sec) in sym_gen:
                if gen in (SymmetryGenerators.ParityX, SymmetryGenerators.ParityY) and \
                   ((int(has_u1_sec) != self._ns // 2) or (self._ns % 2 != 0)):
                    self._log(f"Removing parity {gen} due to U(1) constraint.", log = 1, lvl = 2, color = 'blue')
                else:
                    new_sym_gen.append((gen, sec))
            sym_gen = new_sym_gen
        return sym_gen
    
    #! Printer
    def _gen_sym_print(self, t: Optional[Operator] = None) -> None:
        """
        Print the symmetry group.

        Parameters
        ----------
        has_t : bool
            Flag for the translation symmetry.
        t : Operator
            The translation operator.        
        """
        
        self._log("Using local symmetries:", lvl = 1, color = 'green')
        for (g, sec) in self._sym_group_sec:
            self._log(f"{g}: {sec}", lvl = 2, color = 'blue')
        if t is not None:
            self._log(f"{t}: {t.eigval}", lvl = 2, color = 'blue')
        self._log("Using global symmetries:", lvl = 1, color = 'green')
        for g in self._global_syms:
            self._log(f"{g}: {g.get_val()}", lvl = 2, color = 'blue')
    
    #! Final symmetry group generation
    
    def _gen_sym_group(self, gen : list):
        """
        Generate the symmetry group of the Hilbert space. 
        
        This method constructs the full symmetry group by:
        1. Checking and filtering generators based on lattice and global symmetries
        2. Creating all combinations of local symmetry generators
        3. Combining with translation symmetry if present
        4. Ensuring the identity element is included
        
        The symmetry group is stored in self._sym_group as a list of Operator objects.
        Each operator, when called with a state, returns (new_state, eigenvalue).
        
        Args:
            gen (list): A list of (SymmetryGenerator, sector_value) tuples defining the generators.
        """
		
        if (not gen or len(gen) == 0) and not self.check_global_symmetry():
            self._log("No local or global symmetries provided; using identity only.", lvl = 1, log = 'debug', color = 'green')
            # Ensure at least identity is in the group
            self._sym_group = [operator_identity(self._backend)]
            return
        
        # copy the generators to modify them if needed
        sym_gen                     = gen.copy() if gen is not None and hasattr(gen, "__iter__") else []
        
        # Reset symmetry groups.
        self.reset_local_symmetries()
                
        #! globals - check the global symmetries
        
        # Check global U(1) symmetry.
        has_u1, u1_val              = self._gen_sym_group_check_u1()
        
        # process translation symmetries
        sym_gen, (has_t, has_cpx_t), (t, direction) = self._gen_sym_group_check_t(sym_gen)
        
        # remove reflections from the symmetry generators if the complex translation is present
        sym_gen                     = self._gen_sym_remove_reflection(sym_gen, has_cpx_t)

        # check the existence of the parity when U(1) is present
        sym_gen                     = self._gen_sym_remove_parity(sym_gen, has_u1, u1_val)
        
        # save all sector values for convenience
        for gen, sec in sym_gen:
            self._sym_group_sec.append((gen, sec))
        
        # ------------------------------
        # Generate all combinations of the local generators.
        # For each subset of indices from sym_gen, create a list of operators to apply sequentially.
        # Instead of composing operators into one function, we store them as tuples.
        _size_gen = len(sym_gen)
        
        # Start with identity: empty tuple means no operators to apply
        self._sym_group.append(())
        
        # Generate combinations for r = 1, 2, ..., _size_gen
        if _size_gen > 0:
            generator_ops = [
                choose(sym_gen[idx], ns=self._ns, lat=self._lattice, backend=self._backend)
                for idx in range(_size_gen)
            ]
            for combo in _enumerate_generator_index_combos(_size_gen)[1:]:
                ops_tuple = tuple(generator_ops[idx] for idx in combo)
                self._sym_group.append(ops_tuple)
        
        # apply the translation symmetry by combining with existing operators
        if has_t:
            self._sym_group = self._gen_sym_apply_t(self._sym_group, t, direction)

        # Log the symmetry group information
        self._gen_sym_print(t)
    
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
        
        self._gen_sym_group(gen)    # generate the symmetry group
        
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

    # --------------------------------------------------------------------------------------------------

    ####################################################################################################
    #! Getters and checkers for the Hilbert space
    ####################################################################################################
    
    # GLOBAL SYMMETRIES
    
    def check_global_symmetry(self):
        """
        Check if there are any global symmetries.
        """
        return len(self._global_syms) > 0 if self._global_syms is not None else False
    
    def check_u1(self):
        """
        Check if there is a U(1) symmetry.
        """
        if self._global_syms is not None:
            for sym in self._global_syms:
                if sym.get_name() == GlobalSymmetries.U1:
                    return True, sym.get_val()
        return False, None
    
    #---------------------------------------------------------------------------------------------------
    
    def get_sym_info(self):
        """
        Creates the information string about the Hilbert space and symmetries.
        
        Returns:
            str: A string containing the information about all the symmetries.
        """
        tmp = ""
        if self._sym_group_sec:
            # start with local symmetries
            for gen, val in self._sym_group_sec:
                tmp += f"{gen}={val},"
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
    
    @property
    def reprmap(self):
        """
        Return the mapping of the representatives.
        
        Returns:
            list: The mapping of the representatives.
        """
        return self._reprmap
    
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

    def list_local_operators(self):
        """
        Return the identifiers of all onsite operators available in the local space.
        """
        if self._local_space is None:
            return tuple()
        return self._local_space.list_operator_keys()

    def build_local_operator(self,
                             key: str,
                             *,
                             type_override: Optional[OperatorTypeActing] = None,
                             name: Optional[str] = None) -> Operator:
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

    # --------------------------------------------------------------------------------------------------
    
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
        ls_info     = [f"{g}={sec}" for g, sec in self._sym_group_sec]
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
    
    def _find_sym_norm_base(self, state):
        r"""
        Finds the normalization for a given state (baseIdx) by summing the eigenvalues
        over all symmetry operators that return the same state.
        
        Returns sqrt(sum of eigenvalues).
        """
        pass
    
    def _find_sym_norm_int(self, state):
        r"""
        Find the symmetry normalization of a given state.
        
        The normalization factor is calculated as sqrt(sum of |eigenvalues|^2) for all
        symmetry operations that leave the state invariant (map it to itself). This ensures
        proper normalization of symmetry-adapted basis states.
        
        Args:
            state (int): The state to find the symmetry normalization for.
        
        Returns:
            float: The symmetry normalization of the state, sqrt(\Sigma |\lambda _g|^2) where g leaves state invariant
        """
        norm = 0.0
        for op_tuple in self._sym_group:
            # Apply operators sequentially
            _st = state
            _retval = 1.0
            
            # Empty tuple means identity
            if len(op_tuple) == 0:
                _st = state
                _retval = 1.0
            else:
                # Apply each operator in sequence
                for op in op_tuple:
                    _st, phase = op(_st)
                    _retval *= phase
            
            # Only count if state is invariant
            if _st == state:
                # Accumulate |eigenvalue|^2 for proper complex eigenvalue handling
                if hasattr(_retval, 'conjugate'):
                    norm += (_retval * _retval.conjugate()).real
                else:
                    norm += _retval * _retval
        return math.sqrt(norm) if norm > 0 else 0.0
    
    def find_sym_norm(self, state) -> Union[float, complex]:
        """
        Finds the normalization for a given state (baseIdx) by summing the eigenvalues
        over all symmetry operators that return the same state.
        
        Returns sqrt(sum of eigenvalues).
        """
        if isinstance(state, int):
            return self._find_sym_norm_int(state)
        return self._find_sym_norm_base(state)
    
    # --------------------------------------------------------------------------------------------------
    
    def _find_repr_base(self, state):
        pass
    
    def _find_repr_int(self, state):
        """
        Find the representative of a given state.
        
        The representative is the smallest state (minimum integer value) that can be obtained
        by applying all symmetry operations in the symmetry group. This ensures a unique
        canonical form for each symmetry sector.
        
        Args:
            state (int): The state to find the representative for.
        
        Returns:
            tuple: (representative_state, symmetry_eigenvalue)
                - representative_state: The minimum state in the symmetry orbit
                - symmetry_eigenvalue: The phase factor from the symmetry operation
        """
        # Fast-path: if a full representative map exists, translate mapping index to state
        try:
            if self._reprmap is not None and hasattr(self._reprmap, "__len__") and len(self._reprmap) > 0:
                idx     = self._reprmap[state, 0]
                sym     = self._reprmap[state, 1]
                # reprmap stores the index into self._mapping; convert to actual representative state
                from QES.general_python.common.binary import bin_search as _bs
                
                if idx != _bs._BAD_BINARY_SEARCH_STATE and self._mapping is not None and 0 <= int(idx) < len(self._mapping):
                    rep_state = int(self._mapping[int(idx)])
                    return rep_state, sym
        except Exception:
            pass

        # Fallback: compute by scanning symmetry group
        return find_repr_int(state, self._sym_group, None)
    
    def find_repr(self, state):
        """
        Find representatives of another state using various combinations of symmetry generators.

        This method computes the smallest representative possible by applying different combinations of symmetry 
        generators. It also determines the symmetry eigenvalue associated with returning to the original state.

        Args:
            state (int or state type): The state representation. If an integer is provided, an integer-based approach 
                                    is used; otherwise, a base state representation is assumed.

        Returns:
            tuple: A pair containing the representative index and the corresponding symmetry eigenvalue.
        """
        if isinstance(state, int):
            return self._find_repr_int(state)
        return self._find_repr_base(state)
    
    # --------------------------------------------------------------------------------------------------
    
    def find_representative_base(self, state, normalization_beta):
        """
        Find the representative of a given state.
        
        Args:
            state (np.ndarray): The state to find the representative for.
            normalization_beta (float): The normalization in sector beta.
        
        Returns:
            np.ndarray: The representative of the state.
        """
        pass
    
    def find_representative_int(self, state, normalization_beta):
        """
        Find the representative of a given state.
        """
        return find_representative_int(state, self._mapping,
                self._normalization, normalization_beta, self._sym_group, self._reprmap)
    
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
            rep, _ = self._find_repr_int(j)
            
            # Only add the state if it is its own representative
            # (this ensures each symmetry sector is represented once)
            if rep == j:
                if self._state_filter is not None and not self._state_filter(rep):
                    continue
                # Calculate normalization: sqrt of sum of |eigenvalues|^2 for orbit
                n = self._find_sym_norm_int(j)
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
        
        # initialize the mapping and normalization if necessary
        self._reprmap = []
        
        for j in range(self._nhfull):
            if self._state_filter is not None and not self._state_filter(j):
                self._reprmap.append((bin_search._BAD_BINARY_SEARCH_STATE, 0.0))
                continue
            global_checker = True
            
            if self._global_syms:
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                    
            # if the global symmetries are not satisfied, skip the state
            if not global_checker:
                self._reprmap.append((bin_search._BAD_BINARY_SEARCH_STATE, 0.0))
                continue
            
            mapping_size    = len(self._mapping)
            idx             = bin_search.binary_search(self._mapping, 0, mapping_size - 1, j)
            if idx != bin_search._BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                self._reprmap.append((idx, 1.0))
                continue
            
            # find the representative
            rep, sym_eig    = self.find_repr(j)
            idx             = bin_search.binary_search(self._mapping, 0, mapping_size - 1, rep)
            if idx != bin_search._BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                if self._state_filter is not None and not self._state_filter(rep):
                    self._reprmap.append((bin_search._BAD_BINARY_SEARCH_STATE, 0.0))
                    continue
                sym_eigc = sym_eig.conjugate() if hasattr(sym_eig, "conjugate") else sym_eig
                self._reprmap.append((idx, np.conj(sym_eigc)))
            else:
                self._reprmap.append((bin_search._BAD_BINARY_SEARCH_STATE, 0.0))
        
        # Convert reprmap to numpy array for efficient indexing
        self._reprmap = self._backend.array(self._reprmap, dtype=object)
    
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
        return get_matrix_element(k, new_k, kmap, h_conj, self._mapping, 
                        self._normalization, self._sym_group, self._reprmap)
    
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
    
    @maybe_jit
    def _cast_to_full_jax(self, state, backend : str = "jax"):
        """
        Cast the state to the full Hilbert space.
        
        Args:
            state: The state to cast to the full Hilbert space.
        
        Returns:
            int: The state cast to the full Hilbert space.
        """
        import jax.numpy as jnp
        # Create a full state vector of zeros.
        f_s = jnp.zeros((self._nhfull,), dtype=self._dtype)
        # Convert self._fullmap to a JAX array if it isn't one already.
        fullmap = jnp.array(self._fullmap)
        # Use the vectorized .at[].set() operation to update f_s at indices in fullmap.
        f_s = f_s.at[fullmap].set(state)
        return f_s
    
    def cast_to_full(self, state):
        """
        Cast the state to the full Hilbert space.
        
        Args:
            state: The state to cast to the full Hilbert space.
        
        Returns:
            int: The state cast to the full Hilbert space.
        """
        if not self.check_global_symmetry():
            return state
        if self._fullmap is None or len(self._fullmap) == 0:
            self.generate_full_map_int()
        if isinstance(state, np.ndarray):
            final_state = self._backend.zeros(self._nhfull, dtype=self._dtype)
            for i, idx in enumerate(self._fullmap):
                final_state[idx] = state[i]
            return final_state
        return self._cast_to_full_jax(state)
    
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

####################################################################################################

def set_operator_elem(operator, hilbert : HilbertSpace, k : int, val, new_k : int, conj = False):
    """
    Set the matrix element of the operator.
    
    Args:
        operator (Operator)     : The operator to set the matrix element for.
        hilbert (HilbertSpace)  : The Hilbert space object.
        i                       : The index of the matrix element.
        val                     : The value of the matrix element.
        j                       : The index of the matrix element.
    Returns:
        Operator: The operator with the matrix element set.
    """
    (row, col), sym_eig = hilbert.get_matrix_element(k, new_k, h_conj = conj)
    
    # check if operator is numpy array
    if isinstance(operator, np.ndarray):
        operator[row, col]  += val * sym_eig
    else:
        operator = operator.at[row, col].add(val * sym_eig)
    return operator # for convenience

def get_operator_elem(hilbert : HilbertSpace, k : int, new_k : int, conj = False):
    """
    Get the matrix element of the operator.
    
    Args:
        hilbert (HilbertSpace)  : The Hilbert space object.
        k                       : The index of the matrix element.
        new_k                   : The new index of the matrix element.
        conj                    : Whether to take the complex conjugate (default is False).
    Returns:
        float: The matrix element of the operator.
    """
    (row, col), sym_eig = hilbert.get_matrix_element(k, new_k, h_conj = conj)
    return (row, col), sym_eig

####################################################################################################

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from jax import jit, lax
    from functools import partial
    
    # @partial(jit, static_argnames=('funct', 'max_padding'))
    def process_matrix_elem_jax(funct: Callable, k, k_map, params, max_padding: int):
        """
        Process the matrix element. It is assumed that the function is returning the [rows], [cols], [vals].
        
        Args:
            funct (Callable): The function to apply.
            k (int)         : The index of the matrix element - by default it is assumed to be the row index.
            k_map (int)     : The mapped index of the matrix element - if the mapping is used.
            params          : The parameters for the matrix element - for example the lattice indices.
            max_padding (int): Maximum number of padding to apply.
        """
        # assume that the function is returning the [rows], [cols], [vals]
        all_results     = jax.vmap(lambda p: funct(k, k_map, p))(params)
        _, cols, vals   = all_results
        
        # Debug print to check shape and type
        # jax.debug.print("Shape of cols: {}", cols.shape)
        # jax.debug.print("Type of cols: {}", cols.dtype) 
        # jax.debug.print("Shape of vals: {}", vals.shape)
        # jax.debug.print("Type of vals: {}", vals.dtype) 
        
        # Flatten the results
        cols            = cols.reshape(-1)
        vals            = vals.reshape(-1)
        
        # Sort the columns and values
        sort_idx        = jnp.argsort(cols)
        cols_sorted     = cols[sort_idx]
        vals_sorted     = vals[sort_idx]
        
        # Debug print to check shape and type
        # jax.debug.print("Shape of cols_sorted: {}", cols_sorted.shape)
        # jax.debug.print("Type of cols_sorted: {}", cols_sorted.dtype)  

        # Find unique column indices and sum values.
        unique_cols, inv, counts    = jnp.unique(
            cols_sorted, return_inverse=True, return_counts=True,size=cols_sorted.shape[0]
        )
        summed_vals                 = jax.ops.segment_sum(vals_sorted, inv, num_segments=unique_cols.shape[0])

        # Padding as before.
        pad_width                   = max_padding - unique_cols.shape[0]
        # jax.debug.print("Padding width: {}", pad_width)
        # jax.debug.print("max_padding: {}", max_padding)
        
        # Calculate pad_width using JAX, keep it as JAX array
        unique_cols_padded          = pad_array(unique_cols, max_padding, -1)
        summed_vals_padded          = pad_array(summed_vals, max_padding, 0.0)

        return unique_cols_padded, summed_vals_padded, unique_cols.shape[0]

    @partial(jit, static_argnames=('funct', 'hilbert', 'max_padding', 'batch_start', 'batch_end'))
    def process_matrix_batch_jax(funct: Callable, batch_start, batch_end, hilbert : HilbertSpace, params, max_padding: int):
        '''
        Process a batch of matrix elements using JAX.
        
        Args:
            funct (Callable)    : The function to process each matrix element.
            batch_start (int)   : The starting index of the batch.
            batch_end (int)     : The ending index of the batch.
            hilbert (HilbertSpace): The Hilbert space object.
            params (Any)        : Additional parameters for processing.
            max_padding (int)   : Maximum number of padding to apply.
        '''
        ks      = jnp.arange(batch_start, batch_end, dtype=ACTIVE_INT_TYPE)
        k_maps  = jnp.array([hilbert.get_mapping(k) for k in ks], dtype=ACTIVE_INT_TYPE)
        # Vectorize process_matrix_elem_jax over the rows in the batch.
        cols_, vals_, counts_ = jax.vmap(
            lambda r, k_map: process_matrix_elem_jax(funct, r, k_map, params, max_padding)
        )(ks, k_maps)
        return cols_, vals_, counts_

def process_matrix_elem_np(funct : Callable, k, k_map, params):
    """
    Process the matrix element for a given set of parameters.
    This can be for example a set of real lattice indices 
    
    Args:
        k (int)     : The index of the matrix element - by default it is assumed to be the row index.
        k_map (int) : The mapped index of the matrix element - if the mapping is used.
        params      : The parameters for the matrix element - for example the lattice indices.
    """
    # assume that the function is returning the [rows], [cols], [vals]
    rows, cols, vals    = np.vectorize(lambda p: funct(k, k_map, p))(params)
    count               = len(rows)
    return rows, cols, vals, count

def process_matrix_batch_np(batch_start, batch_end, hilbert : HilbertSpace, funct : Callable, params):
    """
    Process the matrix batch. This function processes the matrix batch using the given function.
    This is assumed to be a start of index and end of index for the batch.
    
    Runs the process_matrix_elem_np function for each element in the batch.
    
    Args:
        batch_start (int)   : The starting index of the batch.
        batch_end (int)     : The ending index of the batch.
        funct (Callable)    : The function to process the matrix batch.
    """
    # create
    k_maps  = [hilbert.get_mapping(k) for k in range(batch_start, batch_end)]
    ks      = np.arange(batch_start, batch_end, dtype=np.int64)
    # Vectorize process_row over the rows in the batch.
    unique_cols_batch, summed_vals_batch, counts_batch  = np.vectorize(lambda k, k_map: process_matrix_elem_np(funct, k, k_map, params))(ks, k_maps)
    total_counts                                        = np.sum(counts_batch)
    return unique_cols_batch, summed_vals_batch, counts_batch, total_counts

#####################################################################################################
#! End of file
#####################################################################################################