"""
Monte Carlo samplers for quantum state space.
This module provides samplers for efficiently exploring the Hilbert space 
of quantum systems using Monte Carlo techniques. The main component is the 
MCSampler class which implements Markov Chain Monte Carlo sampling for quantum
wavefunctions.

The module supports:
- Sampling from quantum state space according to Born distribution or modified distributions
- Multiple concurrent Markov chains
- Different initial state configurations (random, ferromagnetic, antiferromagnetic)
- Customizable state update proposals
- Both NumPy and JAX backends

Classes:
    SamplerErrors       : Error messages related to sampler operations
    SolverInitState     : Enum for different types of initial states
    Sampler             : Abstract base class for samplers
    MCSampler           : Concrete MCMC sampler implementation for quantum states
    SamplerType         : Enum for different sampler types
    UpdateRule          : Enum for different update rules
Functions:
    get_sampler         : Factory function for creating samplers
    get_update_function : Factory function for creating update rules
    
---------------------------------------------------------------------------
file    : Solver/MonteCarlo/sampler.py
author  : Maksymilian Kliczkowski
date    : 2025-02-01
---------------------------------------------------------------------------
"""

import  numba
import  numpy       as np
from    functools   import partial
from    typing      import Union, Tuple, Callable, Optional, Any, List
from    abc         import ABC, abstractmethod
from    enum        import Enum, auto, unique

# from algebra
try:
    from QES.general_python.algebra.utils       import get_backend, DEFAULT_JP_INT_TYPE, DEFAULT_BACKEND_KEY, Array
    from QES.general_python.algebra.utils       import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, distinguish_type
    import QES.general_python.common.binary     as Binary
except ImportError as e:
    raise ImportError("Failed to import general_python modules. Ensure QES package is correctly installed.") from e

# from hilbert
try:
    from QES.Algebra.hilbert                    import HilbertSpace
    from QES.Algebra.Hilbert.hilbert_local      import LocalSpaceTypes
except ImportError as e:
    # raise ImportError("Failed to import HilbertSpace module. Ensure QES package is correctly installed.") from e
    HilbertSpace                                = None
    LocalSpaceTypes                             = None

# from updates
try:
    from .updates import propose_local_flip, propose_exchange, propose_global_flip, propose_multi_flip, get_neighbor_table, propose_local_flip_np, propose_multi_flip_np, propose_bond_flip
except ImportError:
    # warnings.warn("Failed to import updates module. Update rules may not be available.")
    pass

# from initialization
try:
    from .initialization import SolverInitState, initialize_state
except ImportError:
    raise ImportError("Failed to import initialization module.")

#! JAX imports
try:
    import          jax
    import          jax.numpy as jnp
    import          jax.random as random_jp
    JAX_AVAILABLE   = True
except ImportError:
    import numpy    as jnp
    jax             = None
    random_jp       = None
    JAX_AVAILABLE   = False

#########################################################################
#! Errors
#########################################################################

class SamplerErrors(Exception):
    """
    Errors for the Sampler class.
    """
    NOT_GIVEN_SIZE_ERROR        = "The size of the system is not given"
    NOT_IMPLEMENTED_ERROR       = "This feature is not implemented yet"
    NOT_A_VALID_STATE_STRING    = "The state string is not a valid state string"
    NOT_A_VALID_STATE_DISTING   = "The state is not a valid state"
    NOT_A_VALID_SAMPLER_TYPE    = "The sampler type is not a valid sampler type"
    NOT_IN_RANGE_MU             = "The parameter \\mu must be in the range [0, 2]"
    NOT_HAVING_RNG              = "Either rng or seed must be provided"

########################################################################
#! Sampler class for Monte Carlo sampling
########################################################################

class Sampler(ABC):
    """
    A base class for the sampler.     
    """
    
    _name = "BaseSampler"
    
    def __init__(self,
                shape       : Tuple[int, ...],
                upd_fun     : Callable,
                rng         : np.random.Generator               = None,
                rng_k                                           = None,
                seed        : Optional[int]                     = None,
                hilbert     : Optional[HilbertSpace]            = None,
                numsamples  : int                               = 1,
                numchains   : int                               = 1,
                initstate   : Union[np.ndarray, jnp.ndarray]    = None,
                backend     : str                               = 'default',
                statetype   : Union[np.dtype, jnp.dtype]        = None,
                makediffer  : bool                              = False,
                logger                                          = None,
                **kwargs
            ):
        """
        Abstract base class for samplers.
        Parameters:
            shape (Tuple[int, ...]):
                Shape of the system (e.g., lattice dimensions).
            upd_fun (Callable):
                Update function for proposing new states. Signature: `new_state = upd_fun(state, rng/rng_k, **kwargs)`.
                If None, defaults to single random spin flip.
            rng (np.random.Generator):
                NumPy random number generator (used if backend is NumPy or seed is provided).
            rng_k (Optional[jax.random.PRNGKey]):
                JAX random key (used if backend is JAX).
            seed (Optional[int]):
                Seed for initializing random number generators if `rng` and `rng_k` are not provided.
            hilbert (Optional[HilbertSpace]):
                Hilbert space instance (currently not fully utilized in provided code, but kept for structure).
            numsamples (int):
                Number of samples to generate per chain *after* thermalization.
            numchains (int):
                Number of parallel Markov chains.
            initstate (Array or str or SolverInitState or int):
                Initial state configuration specification. Can be an array, an integer state index,
                a predefined string ('RND', 'F_UP', 'F_DN', 'AF'), or a SolverInitState enum. Defaults to 'RND'.
            backend (str):
                Computational backend ('numpy', 'jax', or 'default'). 'default' chooses JAX if available.
            statetype (Union[np.dtype, jnp.dtype]):
                Type of the state (e.g., np.float32, jnp.float32). Not used in the provided code.
            logger (Logger):
                Logger object for logging information.
            **kwargs:
                Additional arguments passed to state generation (e.g., `modes`, `mode_repr`).

        Raises:
            SamplerErrors: If backend/initial state is invalid or RNG setup fails.

        Attributes:
            _shape (Tuple[int, ...])            : Shape of the system.
            _size (int)                         : Total number of sites/spins.
            _hilbert (Optional[HilbertSpace])   : Hilbert space instance.
            _rng, _rng_k                        : Random number generators.
            _backend                            : Computational backend module (np or jnp).
            _isjax (bool)                       : True if using JAX backend.
            _backendstr (str)                   : Name of the backend ('np' or 'jax').
            _numsamples (int)                   : Number of samples per chain.
            _numchains (int)                    : Number of chains.
            _initstate                          : Initial state configuration (single state).
            _states                             : Current states of all chains (shape: [numchains, *shape]).
            _tmpstates (np.ndarray)             : Temporary storage for NumPy proposals (if needed).
            _num_proposed, _num_accepted        : Counters for proposed and accepted moves per chain.
            _upd_fun                            : The actual update function used.
            _logger                             : Logger object.
        """
        if isinstance(backend, str):
            self._backendstr = backend
            
        self._logger = logger

            
        if rng is not None:
            self._rng       = rng
            self._rng_k     = rng_k if rng_k is not None else (DEFAULT_BACKEND_KEY if JAX_AVAILABLE else None)
            self._backend   = get_backend(backend)
        else:
            # default fallback to obtain the backend and RNGs
            self._backend, _, (self._rng, self._rng_k) = self.obtain_backend(backend, seed)
        
        is_valid_rng_k = (
            self._rng_k is not None             and             
            isinstance(self._rng_k, jax.Array)  and
            self._rng_k.shape == (2,)           and           
            self._rng_k.dtype == jnp.uint32         
        )
        
        #! check the RNG key, whether it is a valid JAX PRNGKey
        if not is_valid_rng_k:
            key_info = f"Value: {self._rng_k}, Type: {type(self._rng_k)}"
            if isinstance(self._rng_k, jax.Array):
                key_info += f", Shape: {self._rng_k.shape}, dtype: {self._rng_k.dtype}"
            raise TypeError(f"Sampler's RNG key (self._rng_k) is not a valid JAX PRNGKey. {key_info}")
    
        # check JAX
        self._isjax         = (not self._backend == np)
        if self._isjax and self._rng_k is None:
            raise SamplerErrors(SamplerErrors.NOT_HAVING_RNG + " (JAX requires rng_k)")
        if not self._isjax and self._rng is None:
            raise SamplerErrors(SamplerErrors.NOT_HAVING_RNG + " (NumPy requires rng)")
        
        # set the backend
        self._backendstr    = 'np' if not self._isjax else 'jax'
        self._makediffer    = makediffer
        
        # handle the initial state
        statetype           = int if statetype is None else statetype
        self._statetype     = distinguish_type(statetype, 'numpy' if not self._isjax else 'jax')
        
        # handle the Hilbert space - may control state initialization
        self._hilbert       = hilbert
        
        # handle the states
        self._shape         = shape
        self._size          = int(np.prod(shape)) if isinstance(shape, tuple) else shape
        self._numsamples    = numsamples
        self._numchains     = numchains
        self._states        = None
        
        # handle the initial state
        self._modes         = kwargs.get('modes', 2)
        self._mode_repr     = kwargs.get('mode_repr', Binary.BACKEND_REPR)
        state_kwargs        = {
            'different'     : self._makediffer,
            'modes'         : kwargs.pop('modes', self._modes),
            'mode_repr'     : kwargs.pop('mode_repr', self._mode_repr),
            'n_particles'   : kwargs.get('n_particles', kwargs.get('nparticles', 1)),
            'magnetization' : kwargs.get('magnetization', None),
        }
        self._initstate_t   = initstate
        self._isint         = isinstance(initstate, (int, np.integer, jnp.integer))
        self.set_initstate(self._initstate_t, **state_kwargs)
        
        # proposed state
        int_dtype           = DEFAULT_JP_INT_TYPE if self._isjax else DEFAULT_NP_INT_TYPE
        self._num_proposed  = self._backend.zeros(numchains, dtype=int_dtype)
        self._num_accepted  = self._backend.zeros(numchains, dtype=int_dtype)
        
        # handle the update function
        self._upd_fun       = upd_fun
        if self._upd_fun is None:
            if self._isjax:
                # Bind RNG arguments to the JAX updater and then wrap with JIT.
                self._upd_fun = jax.jit(propose_local_flip)
            else:
                # Use the Numba version (potentially parallelized)
                self._upd_fun = propose_local_flip_np
    
    ###################################################################
    #!ABSTRACT
    ###################################################################
    
    @abstractmethod
    def sample(self, parameters=None, num_samples=None, num_chains=None):
        """ Tries to sample the state from the Hilbert space. """
        pass
    
    ###################################################################
    #! BACKEND
    ###################################################################

    @staticmethod
    def obtain_backend(backend: str, seed: Optional[int]):
        """
        Obtain the backend for the calculations.
        Parameters:
        - backend       : backend for the calculations (default is 'default')
        - seed          : seed for the random number generator
        Returns:
        - Tuple         : backend, backend_sp, rng, rng_k 
        """
        if isinstance(backend, str):
            bck = get_backend(backend, scipy=True, random=True, seed=seed)
            if isinstance(bck, tuple):
                _backend, _backend_sp = bck[0], bck[1]
                if isinstance(bck[2], tuple):
                    _rng, _rng_k = bck[2][0], bck[2][1]
                else:
                    _rng, _rng_k = bck[2], None
            else:
                _backend, _backend_sp = bck, None
                _rng, _rng_k = None, None
            return _backend, _backend_sp, (_rng, _rng_k), backend
        _backendstr = 'np' if (backend is None or (backend == 'default' and not JAX_AVAILABLE) or backend == np) else 'jax'
        return Sampler.obtain_backend(_backendstr, seed)
    
    ###################################################################
    #! PROPERTIES
    ###################################################################
    
    @property
    def name(self): return self._name
    @property
    def hilbert(self): return self._hilbert
    @property
    def numsamples(self): return self._numsamples
    @property
    def numchains(self): return self._numchains
    @property
    def shape(self): return self._shape
    @property
    def size(self): return self._size
    @property
    def initstate(self): return self._initstate
    @property
    def upd_fun(self): return self._upd_fun
    @property
    def states(self): return self._states
    @property
    def rng(self): return self._rng
    @property
    def rng_k(self): return self._rng_k
    @property
    def backend(self): return self._backend
    @property
    def proposed(self): return self._num_proposed
    @property
    def num_proposed(self): return self._backend.sum(self._num_proposed)
    @property
    def accepted(self): return self._num_accepted
    @property
    def num_accepted(self): return self._backend.sum(self._num_accepted)
    @property
    def rejected(self): return self.proposed - self.accepted
    @property
    def num_rejected(self): return self.num_proposed - self.num_accepted
    @property
    def accepted_ratio(self):
        num_prop = self.num_proposed
        return self.num_accepted / num_prop if num_prop > 0 else self._backend.array(0.0)
    @property
    def isjax(self): return self._isjax
    
    ###################################################################
    #! SETTERS
    ###################################################################
    
    def reset(self):
        """
        Reset the sampler to its initial state.
        """
        int_dtype           = DEFAULT_JP_INT_TYPE if self._isjax else DEFAULT_NP_INT_TYPE
        self._num_proposed  = self._backend.zeros(self._numchains, dtype=int_dtype)
        self._num_accepted  = self._backend.zeros(self._numchains, dtype=int_dtype)
        state_kwargs        = {
            'modes'     : self._modes,
            'mode_repr' : self._mode_repr
        }
        self.set_initstate(self._initstate_t, **state_kwargs)
    
    # ---
    
    def set_initstate(self, initstate, **kwargs):
        """
        Set the initial state template of the system.

        Parameters:
            initstate (str, int, np.ndarray, jnp.ndarray, SolverInitState):
                The initial state specification. If None, defaults to 'RND'.
            **kwargs:
                Additional arguments passed to `initialize_state` (e.g., `modes`, `mode_repr`).

        Raises:
            SamplerErrors.NOT_A_VALID_STATE_DISTING:
                If the state type is invalid.
            ValueError:
                If underlying state generation fails.
        """

        if initstate is None:
            initstate = SolverInitState.RND 
            
        different_initials = kwargs.get('different', False)
        
        # handle the initial state
        if initstate is None or isinstance(initstate, (str, SolverInitState)):
            try:
                # Extract modes and mode_repr from kwargs
                current_bcknd_str   = 'jax' if self._isjax else 'numpy'
                modes_val           = kwargs.pop('modes', 2)
                mode_repr_val       = kwargs.pop('mode_repr', 0.5)
                
                if self._hilbert is None or True:
                    if different_initials:
                        self._states = self._backend.stack([
                                initialize_state(initstate,
                                    modes       =   modes_val,
                                    hilbert     =   self._hilbert,
                                    shape       =   self._shape,
                                    backend     =   current_bcknd_str,
                                    rng         =   self._rng,
                                    rng_k       =   self._rng_k,
                                    mode_repr   = mode_repr_val,
                                    **kwargs)
                                for _ in range(self._numchains)
                            ], axis=0)
                        self._initstate = self._states[0]
                        if not self._isint:
                            self._initstate = self._initstate.astype(self._statetype)
                            self._states = self._states.astype(self._statetype)
                        return
                    else:
                        self._initstate = initialize_state(initstate,
                                            modes       =   modes_val,
                                            hilbert     =   self._hilbert,
                                            shape       =   self._shape,
                                            backend     =   current_bcknd_str,
                                            rng         =   self._rng,
                                            rng_k       =   self._rng_k,
                                            mode_repr   = mode_repr_val,
                                            **kwargs)
                        if not self._isint:
                            self._initstate = self._initstate.astype(self._statetype)
                else:
                    raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
            except Exception as e:
                raise ValueError(f"Failed to set initial state: {e}") from e
        else:
            if isinstance(initstate, np.ndarray) and self._isjax:
                self._initstate = jnp.array(initstate).astype(self._statetype)
            elif isinstance(initstate, jnp.ndarray) and not self._isjax:
                self._initstate = np.array(initstate).astype(self._statetype)
            else:
                self._initstate = initstate
        self.set_chains(self._initstate, self._numchains)
    
    # ---
    
    def set_chains(self, initstate: Union[np.ndarray, jnp.ndarray], numchains: Optional[int] = None):
        """
        Set the chains for the sampler, replicating the initstate.

        Parameters:
            initstate (np.ndarray or jnp.ndarray):
                The single initial state configuration to replicate.
            numchains (int, optional):
                The number of chains. If None, uses the sampler's current `_numchains`.
        """
        if numchains is None:
            numchains = self._numchains
        else:
            self._numchains = numchains

        # Ensure initstate is the correct type for the backend before stacking
        if self._isjax:
            self._states = jnp.stack([jnp.array(initstate)] * numchains, axis=0)
        else:
            self._states = np.stack([np.array(initstate.copy())] * numchains, axis=0)
        if not self._isint:
            self._states = self._states.astype(self._statetype)
    
    # ---
    
    def set_numsamples(self, numsamples):
        """
        Set the number of samples.
        Parameters:
            numsamples (int):
                The number of samples
        """
        self._numsamples = numsamples
        
    def set_numchains(self, numchains):
        """
        Set the number of chains.
        Parameters:
            numchains (int):
                The number of chains
        """
        self._numchains = numchains
        self.set_chains(self._initstate, numchains)
    
    ###################################################################
    #! GETTERS
    ###################################################################
    
    @abstractmethod
    def _get_sampler_jax(self, num_samples=None, num_chains=None):
        """
        Get the JAX sampler instance.
        """
        pass
    
    @abstractmethod
    def _get_sampler_np(self, num_samples=None, num_chains=None):
        """
        Get the NumPy sampler instance.
        """
        pass

    @abstractmethod
    def get_sampler(self, num_samples=None, num_chains=None):
        """
        Get the sampler instance based on the backend.
        """
        pass

    @staticmethod
    def help(verbose: bool = True):
        """
        Display help for available update rules and sampler options.
        """
        help_msg = """
        Monte Carlo Sampler Help
        ========================
        
        Available Update Rules (upd_fun):
        ---------------------------------
        1. "LOCAL" (default):
        - Standard single-spin flip Metropolis update.
        - Suitable for non-conserved systems (e.g., Ising).
        - Backend: JAX (optimized) or NumPy.

        2. "EXCHANGE":
        - Exchanges spins/particles between connected sites (Nearest Neighbors).
        - Preserves total magnetization / particle number.
        - Requires: 'hilbert' argument with a valid 'lattice' property.
        - Backend: JAX only.

        3. "GLOBAL":
        - Flips predefined patterns of spins (e.g., plaquettes, clusters).
        - Requires: 'patterns' argument (list of lists of site indices).
        - Backend: JAX only.

        Usage Example:
        --------------
        # Local updates
        nqs         = NQS(..., upd_fun="local")
        
        # Exchange updates (conserved quantities)
        nqs         = NQS(..., upd_fun="exchange", hilbert=my_hilbert)
        
        # Global pattern updates
        patterns    = [[0, 1, 4, 5], [1, 2, 5, 6]] # defined plaquettes
        nqs         = NQS(..., upd_fun="global", patterns=patterns)
        """
        if verbose:
            print(help_msg)
        return help_msg

#######################################################################

@unique
class SamplerType(Enum):
    """
    Enum class for the sampler types.
    """
    MCSampler       = auto()
    ARSampler       = auto()
    
    @staticmethod
    def from_str(s: str) -> 'SamplerType':
        """
        Convert a string to a SamplerType enum.
        """
        try:
            return SamplerType[s]
        except KeyError:
            raise ValueError(f"Invalid SamplerType: {s}") from None

#######################################################################

def get_sampler(typek: Union[str, SamplerType, Sampler], *args, **kwargs) -> Sampler:
    """
    Get a sampler of the given type.
    
    Parameters:
    - typek (str, SamplerType, or Sampler): The type of sampler to get or an existing sampler instance
    - args: Additional arguments for the sampler
    - kwargs: Additional keyword arguments for the sampler
    
    Returns:
    - Sampler: The requested sampler or the provided sampler instance
    
    Raises:
    - ValueError: If the requested sampler type is not implemented
    
    Example:
    >>> sampler = get_sampler("MCSampler", num_chains=10, num_samples=1000)
    >>> print(sampler)
    <__main__.VMCSampler object at 0x...>
    """
    # Local imports to avoid circular dependencies
    try:
        from .vmc import VMCSampler
    except ImportError:
        VMCSampler = None

    try:
        from .arsampler import ARSampler
    except ImportError:
        ARSampler = None

    # -----------------------------------------------------------
    # 1. INSTANCE CHECK: If it's already an initialized Sampler object
    # -----------------------------------------------------------
    if isinstance(typek, Sampler):
        return typek
    
    # -----------------------------------------------------------
    # 2. CLASS CHECK: If it's a class definition (e.g. passed VMCSampler class)
    # -----------------------------------------------------------
    if isinstance(typek, type) and issubclass(typek, Sampler):
        return typek(*args, **kwargs)
    
    # -----------------------------------------------------------
    # 3. STRING TO ENUM CONVERSION
    # -----------------------------------------------------------
    
    if isinstance(typek, str):
        typek_str = typek.strip().lower()
        
        # Manually map common strings if they aren't in Enum yet
        if typek_str    in ['ar', 'arsampler', 'exact']:
            typek = SamplerType.ARSampler
        elif typek_str  in ['vmc', 'mc', 'mcsampler', 'default']:
            typek = SamplerType.MCSampler
        else:
            try:
                typek = SamplerType.from_str(typek_str)
            except ValueError as e:
                raise ValueError(f"Unknown sampler type string: {typek_str}") from e
            
    # -----------------------------------------------------------
    # 4. ENUM TO CLASS INSTANCE MAPPING
    # -----------------------------------------------------------
    
    if typek == SamplerType.MCSampler:
        return VMCSampler(*args, **kwargs)
    
    elif typek == SamplerType.ARSampler:
        if ARSampler is None:
            raise ImportError("ARSampler implementation not found.")
        return ARSampler(*args, **kwargs)
        
    raise ValueError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

#######################################################################
#! Update Rules Factory
#######################################################################

@unique
class UpdateRule(Enum):
    LOCAL       = auto()
    EXCHANGE    = auto()
    GLOBAL      = auto()
    MULTI_FLIP  = auto()
    BOND_FLIP   = auto()
    PLAQUETTE   = auto()
    SUBPLAQUETTE= auto()
    WILSON      = auto()
    
    @staticmethod
    def from_str(s: str) -> 'UpdateRule':
        try:
            return UpdateRule[s.upper()]
        except KeyError:
            # Try some common aliases
            s = s.lower()
            if s in ['local', 'spin_flip', 'flip']:                 # extra aliases for local
                return UpdateRule.LOCAL
            elif s in ['exchange', 'swap', 'particle_hop']:         # extra aliases for exchange
                return UpdateRule.EXCHANGE
            elif s in ['global', 'plaquette', 'pattern']:           # extra aliases for global
                return UpdateRule.GLOBAL
            elif s in ['multi_flip', 'multi', 'n_flip']:            # extra aliases for multi flips
                return UpdateRule.MULTI_FLIP
            elif s in ['bond_flip', 'neighbor_flip', 'bond']:       # extra aliases for bond flips
                return UpdateRule.BOND_FLIP 
            elif s in ['plaquette', 'plaquette_flip']:              # extra alias for plaquette flips
                return UpdateRule.PLAQUETTE
            elif s in ['subplaquette', 'sub_plaquette']:            # extra alias for subplaquette flips
                return UpdateRule.SUBPLAQUETTE
            elif s in ['wilson', 'wilson_loop']:                    # extra alias for wilson loops
                return UpdateRule.WILSON
            raise ValueError(f"Invalid UpdateRule: {s}") from None

def get_update_function(rule: Union[str, UpdateRule], backend='jax', **kwargs) -> Callable:
    """
    Get the update function based on the rule.
    
    Parameters:
        rule (str or UpdateRule): 
            The update rule.
        backend (str): 
            'jax' or 'numpy'. currently only 'jax' supported for advanced rules.
        **kwargs: 
            - hilbert: 
                HilbertSpace object (required for exchange/global/bond)
            - patterns: 
                List of patterns for global update (required for global if not in hilbert)
            - n_flip: 
                Number of flips for multi_flip (default 1)
            
    Returns:
        Callable: The update function.
    """
    
    def _to_padded_pattern(patterns: List[List[int]], pad_value: int = -1) -> jnp.ndarray:
        max_len     = max(len(p) for p in patterns)
        n_patterns  = len(patterns)
        
        patterns_arr = np.full((n_patterns, max_len), pad_value, dtype=np.int32)
        for i, p in enumerate(patterns):
            patterns_arr[i, :len(p)] = p
            
        return jnp.array(patterns_arr)
    
    if isinstance(rule, str):
        rule = UpdateRule.from_str(rule)
        
    if backend != 'jax' and backend != 'default':
        if rule == UpdateRule.LOCAL:
            # We need to return a function with signature (state, rng)
            return propose_local_flip_np
        raise NotImplementedError("Only JAX backend is supported for advanced update rules.")

    if rule == UpdateRule.LOCAL:
        return propose_local_flip
        
    elif rule == UpdateRule.MULTI_FLIP:
        n_flip = kwargs.get('n_flip', 1)
        return partial(propose_multi_flip, n_flip=n_flip)
        
    elif rule == UpdateRule.EXCHANGE:
        hilbert = kwargs.get('hilbert')
        if hilbert is None or hilbert.lattice is None:
            raise ValueError("Hilbert space with lattice is required for exchange updates.")
        
        # Check compatibility with local space type
        if hilbert.local_space is not None and LocalSpaceTypes is not None:
            # Exchange moves generally require particle/magnetization conservation
            # They make sense for Spins, Fermions, Bosons (Hardcore or Softcore)
            # but might not work for arbitrary custom spaces without careful checking.
            # We explicitly check for known compatible types.
            compatible_types = [
                LocalSpaceTypes.SPIN_1_2, 
                LocalSpaceTypes.SPIN_1,
                LocalSpaceTypes.SPINLESS_FERMIONS,
                LocalSpaceTypes.BOSONS
            ]
            
            if hilbert.local_space.typ not in compatible_types:
                import warnings
                warnings.warn(f"EXCHANGE update rule requested for local space type '{hilbert.local_space.typ}', "
                            f"which may not be fully supported or tested. Compatible types: {compatible_types}")

        # Precompute neighbor table
        # User can specify exchange range/order (default 1 = nearest neighbors)
        exchange_order = kwargs.get('exchange_order', kwargs.get('order', 1))
        neighbor_table = get_neighbor_table(hilbert.lattice, order=exchange_order)
        
        # Return partial function
        return partial(propose_exchange, neighbor_table=neighbor_table)
    
    elif rule == UpdateRule.BOND_FLIP:
        hilbert = kwargs.get('hilbert')
        if hilbert is None:
            raise ValueError("Hilbert space is required for bond updates.")
        
        lattice = kwargs.get('lattice')
        if hilbert.lattice is None:
            raise ValueError("Hilbert space with lattice is required for bond updates.")
            
        # Precompute neighbor table
        neighbor_table = get_neighbor_table(lattice, order=1)
        return partial(propose_bond_flip, neighbor_table=neighbor_table)
    
    ###############################################################
    #! GLOBAL UPDATES
    ###############################################################

    elif rule == UpdateRule.PLAQUETTE:
        # is like global but patterns come from lattice
        hilbert = kwargs.get('hilbert')
        if hilbert is None or hilbert.lattice is None:
            raise ValueError("Hilbert space with lattice is required for plaquette updates.")
        plaquettes = hilbert.lattice.calculate_plaquettes()
        if plaquettes is None or len(plaquettes) == 0:
            raise ValueError("No plaquettes found in the lattice for plaquette updates.")

        patterns_jax = jnp.array(plaquettes)
        patterns_jax = _to_padded_pattern(plaquettes, pad_value=-1)
        return partial(propose_global_flip, patterns=patterns_jax)
    
    elif rule == UpdateRule.SUBPLAQUETTE:
        # is like global but patterns come from lattice
        hilbert = kwargs.get('hilbert')
        if hilbert is None or hilbert.lattice is None:
            raise ValueError("Hilbert space with lattice is required for subplaquette updates.")
        plaquettes = hilbert.lattice.calculate_plaquettes()
        if plaquettes is None or len(plaquettes) == 0:
            raise ValueError("No plaquettes found in the lattice for subplaquette updates.")
        
        # Get n (sub-sequence length)
        n = kwargs.get('n_flip', 3) 
        
        subpatterns = []
        for p in plaquettes:
            L = len(p)
            eff_n = min(n, L)
            for i in range(L):
                # Cyclic slice
                sub = [p[(i + j) % L] for j in range(eff_n)]
                subpatterns.append(sub)

        patterns_jax = _to_padded_pattern(subpatterns, pad_value=-1)
        return partial(propose_global_flip, patterns=patterns_jax)
    
    elif rule == UpdateRule.WILSON:
        # is like global but patterns come from lattice
        hilbert: HilbertSpace = kwargs.get('hilbert')
        if hilbert is None or hilbert.lattice is None:
            raise ValueError("Hilbert space with lattice is required for Wilson loop updates.")
        wilson_loops = hilbert.lattice.calculate_wilson_loops()
        if wilson_loops is None or len(wilson_loops) == 0:
            raise ValueError("No Wilson loops found in the lattice for Wilson loop updates.")

        patterns_jax = jnp.array(wilson_loops)
        patterns_jax = _to_padded_pattern(wilson_loops, pad_value=-1)    
        return partial(propose_global_flip, patterns=patterns_jax)

    elif rule == UpdateRule.GLOBAL:
        patterns = kwargs.get('patterns')
        
        # Resolve string patterns (e.g. "plaquette", "wilson")
        if isinstance(patterns, str):
            hilbert = kwargs.get('hilbert')
            if hilbert is None or hilbert.lattice is None:
                raise ValueError(f"Hilbert space with lattice is required for '{patterns}' global updates.")
            
            if patterns.lower() == "plaquette":
                patterns = hilbert.lattice.calculate_plaquettes()
            elif patterns.lower() == "wilson":
                patterns = hilbert.lattice.calculate_wilson_loops()
            else:
                raise ValueError(f"Unknown global pattern type: {patterns}")

        if patterns is None:
            raise ValueError("Patterns list is required for global updates.")
        
        # Convert patterns to padded array
        patterns_jax = _to_padded_pattern(patterns, pad_value=-1)
        return partial(propose_global_flip, patterns=patterns_jax)
        
    raise ValueError(f"Unknown update rule: {rule}")

#######################################################################
#! EOF
#######################################################################