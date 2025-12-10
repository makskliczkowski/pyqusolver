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
    SamplerErrors   : Error messages related to sampler operations
    SolverInitState : Enum for different types of initial states
    Sampler         : Abstract base class for samplers
    MCSampler       : Concrete MCMC sampler implementation for quantum states
    SamplerType     : Enum for different sampler types
Functions:
    get_sampler     : Factory function for creating samplers
    
---------------------------------------------------------------------------
file    : Solver/MonteCarlo/sampler.py
author  : Maksymilian Kliczkowski
date    : 2025-02-01
---------------------------------------------------------------------------
"""

import numpy as np
import numba
from numba import prange
from functools import partial
from typing import Union, Tuple, Callable, Optional, Any
# for the abstract class
from abc import ABC, abstractmethod
from enum import Enum, auto, unique

# flax for the network
try:
    from flax import linen as nn
except ImportError as e:
    raise ImportError("Failed to import flax module. Ensure flax is correctly installed.") from e

# from algebra
try:
    from QES.general_python.algebra.utils import JAX_AVAILABLE, get_backend, DEFAULT_JP_INT_TYPE, DEFAULT_BACKEND_KEY, Array
    from QES.general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, distinguish_type
    from QES.general_python.algebra.ran_wrapper import choice, randint_np, randint_jax
    import QES.general_python.common.binary as Binary
except ImportError as e:
    raise ImportError("Failed to import general_python modules. Ensure QES package is correctly installed.") from e

# from hilbert
try:
    from QES.Algebra.hilbert import HilbertSpace
except ImportError as e:
    raise ImportError("Failed to import HilbertSpace module. Ensure QES package is correctly installed.") from e

#! JAX imports
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.random as random_jp
else:
    import numpy as jnp
    jax         = None
    random_jp   = None

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

class SolverInitState(Enum):
    """Enum for potential initial states """
    
    # -----------------------
    
    RND         = auto()    # random configuration
    F_UP        = auto()    # ferromagnetic up
    F_DN        = auto()    # ferromagnetic down
    AF          = auto()    # antiferromagnetic
    
    # -----------------------
    
    def __str__(self):
        """Return the name of the enum member."""
        return self.name

    # -----------------------

    @classmethod
    def from_str(cls, state_str: str):
        """
        Create an enum member from a string, ignoring case.
        Parameters:
            state_str (str)     : The string representation of the enum member.
        Returns:
            SolverInitState     : The enum member corresponding to the input string.
        """
        # Normalize input (upper-case) to match enum member names
        normalized = state_str.upper()
        if normalized in cls.__members__:
            return cls.__members__[normalized]
        raise ValueError(f"Unknown initial state: {state_str}")

    # -----------------------

#########################################################################
#! Propose a random flip
#########################################################################

if JAX_AVAILABLE:
    
    # @jax.jit
    def _propose_random_flip_jax(state: jnp.ndarray, rng_k):
        r'''Propose `num` random flips of a state using JAX.

        Parameters:
            state (jnp.ndarray):
                The state array (or batch of states).
            rng_k (jax.random.PRNGKey):
                The random key for JAX.

        Returns:
            jnp.ndarray: The proposed flipped state(s).
        '''
        idx = randint_jax(key=rng_k, shape = (), low=0, high=state.size)
        return Binary.jaxpy.flip_array_jax_spin(state, idx)

    # @partial(jax.jit, static_argnames=("num",))
    def _propose_random_flips_jax(state: jnp.ndarray, rng_k, num: int = 1):
        """
        Propose `num` random flips on a state or batch of states using JAX.
        If `state` is 1D -> flip single state.
        If `state` is 2D -> flip batch of states independently.
        """
        if state.ndim == 1:
            idx = randint_jax(rng_k, shape=(num,), minval=0, maxval=state.size, dtype=DEFAULT_JP_INT_TYPE)
            return Binary.jaxpy.flip_array_jax_multi(state, idx, spin=Binary.BACKEND_DEF_SPIN)

        else:
            batch_size, state_size  = state.shape
            keys                    = jax.random.split(rng_k, batch_size)

            def flip_single_state(single_state, key):
                idx = randint_jax(key, shape=(num,), minval=0, maxval=state_size, dtype=DEFAULT_JP_INT_TYPE)
                return Binary.jaxpy.flip_array_jax_multi(single_state, idx, spin=Binary.BACKEND_DEF_SPIN)

            return jax.vmap(flip_single_state)(state, keys)

    def _propose_global_flip_jax(state: jnp.ndarray, rng_k, fraction: float = 0.5):
        """
        Propose a global update by flipping each spin with probability `fraction`.
        """
        mask = jax.random.bernoulli(rng_k, p=fraction, shape=state.shape)
        
        # Check representation via a heuristic or global config
        # Assuming typical VMC case: 
        # If state contains negative values, assume +/- 1. Else 0/1.
        # This check is done per call, might be slow if not jitted out. 
        # Better to rely on Binary.BACKEND_DEF_SPIN static config.
        
        if Binary.BACKEND_DEF_SPIN: # -1 / 1
            flipper = jnp.where(mask, -1.0, 1.0).astype(state.dtype)
            return state * flipper
        else: # 0 / 1
            return jnp.where(mask, 1.0 - state, state)

    def make_hybrid_proposer(local_fun: Callable, global_fun: Callable, p_global: float = 0.1):
        """
        Creates a JIT-compiled hybrid proposer.
        
        Args:
            local_fun: Function for local updates (e.g., single flip).
            global_fun: Function for global updates.
            p_global: Probability of choosing the global update.
            
        Returns:
            A JAX-compatible proposer function.
        """
        @jax.jit
        def hybrid_proposer(state, key):
            key_choice, key_upd = jax.random.split(key)
            do_global = jax.random.bernoulli(key_choice, p=p_global)
            
            # Use lax.cond for branching
            # Note: Both branches must return same shape/type
            return jax.lax.cond(
                do_global,
                lambda s, k: global_fun(s, k),
                lambda s, k: local_fun(s, k),
                state, 
                key_upd
            )
        return hybrid_proposer


@numba.njit
def _propose_random_flips_np(state: np.ndarray, rng, num = 1):
    """
    Propose a random flip of a state using numpy.
    """
    if state.ndim == 1:
        idx = randint_np(rng=rng, low=0, high=state.size, size=num)
        return Binary.flip_array_np_multi(state, idx,
                                        spin=Binary.BACKEND_DEF_SPIN, spin_value=Binary.BACKEND_REPR)
    n_chains, state_size = state.shape[0], state.shape[1]
    for i in range(n_chains):
        idx = randint_np(rng=rng, low=0, high=state_size, size=num)
        state[i] = Binary.flip_array_np_multi(state[i], idx,
                                        spin=Binary.BACKEND_DEF_SPIN, spin_value=Binary.BACKEND_REPR)
    return state

@numba.njit(parallel=True)
def _propose_random_flip_np(state: np.ndarray, rng: np.random.Generator):
    """
    Propose a random flip of a state using numpy.
    """
    if state.ndim == 1:
        idx = randint_np(rng=rng, low=0, high=state.size, size=1)[0]
        return Binary.flip_array_np(state, idx)
    
    n_chains, state_size = state.shape[0], state.shape[1]
    for i in prange(n_chains):
        idx         = randint_np(low=0, high=state_size, size=1)[0]
        state[i]    = Binary.flip_array_np_spin(state[i], idx)
    return state

def propose_random_flip(state: Array, backend = 'default',
                        rng = None, rng_k = None, num = 1):
    """
    Propose a random flip of a state.
    """
    if backend == np or backend == 'numpy' or backend == 'np':
        return _propose_random_flip_np(state, rng) if num == 1 else _propose_random_flips_np(state, rng, num)
    return _propose_random_flip_jax(state, rng, rng_k) if num == 1 else _propose_random_flips_jax(state, rng, rng_k, num)

#########################################################################
#! Set the state of the system
#########################################################################

def _set_state_int(state        : int,
            modes               : int                           = 2,
            hilbert             : Optional[HilbertSpace]        = None,
            shape               : Union[int, Tuple[int, ...]]   = (1,),
            mode_repr           : float                         = Binary.BACKEND_REPR,
            backend             : str                           = 'default'
            ):
    '''
    Set the state configuration from the integer representation.
    
    Parameters:
    - state         : state configuration as an integer
    - modes         : modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape of the system (number of spins).
                    One may want to reshape the state to a given shape (for instance, 2D lattice).
    - mode_repr     : mode representation (default is 0.5 - for binary spins +-1)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    
    Transforms the integer to a given configuration 
    Notes:
        The states are given in binary or other representation 
            : 2 for binary (e.g. spin up/down)
            : 4 for fermions (e.g. fermions up/down)
                - The lower size bits (state & ((1 << size)-1)) encode the up orbital occupations.
                - The upper size bits (state >> size) encode the down orbital occupations.
            ...
        It uses the mode representation to determine the spin value:
        Examples:
        - spins 1/2 are created as +-0.5 when mode_repr = 0.5 (default) and _modes = 2.
                Thus, we need _size to represent the state.
        - fermions are created as 1/-1 when mode_repr = 1.0 and _modes = 2 
                and the first are up spins and the second down spins. Thus, we 
                need 2 * _size to represent the state and we have 0 and ones for the
                presence of the fermions.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from the hilbert
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    out  = None
    
    if hilbert is None:
        if modes == 2:
            # set the state from tensor
            out = Binary.int2base(state, size, backend, 
                                spin_value=mode_repr, spin=Binary.BACKEND_DEF_SPIN).reshape(shape)
        elif modes == 4:
            up_int          = state & ((1 << size) - 1)
            down_int        = state >> size
            up_array        = np.array([1 if (up_int & (1 << i)) else 0 for i in range(size)],
                                dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array      = np.array([1 if (down_int & (1 << i)) else 0 for i in range(size)],
                                dtype=DEFAULT_NP_FLOAT_TYPE)
            out             = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    return out
    
def _set_state_rand(modes       : int                           = 2,
                    hilbert     : Optional[HilbertSpace]        = None,
                    shape       : Union[int, Tuple[int, ...]]   = (1,),
                    mode_repr   : float                         = 0.5,
                    backend     : str                           = 'default',
                    rng         = None,
                    rng_k       = None
                    ):
    '''
    Generate a random state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape of the state array (number of spins).
                        For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    - rng           : random number generator for numpy
    - rng_k         : random key for JAX
    
    Returns:
    - A random state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from hilbert
    size        = shape if isinstance(shape, int) else int(np.prod(shape))
    ran_state   = None
    if hilbert is None or True:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                ran_state = choice([-1, 1], shape, rng=rng, rng_k=rng_k, backend=backend)
            else:
                ran_state = choice([0, 1], shape, rng=rng, rng_k=rng_k, backend=backend)
        elif modes == 4:
            ran_state = choice([0, 1], 2 * size, rng=rng, rng_k=rng_k, backend=backend)
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    return ran_state.astype(DEFAULT_NP_FLOAT_TYPE).reshape(shape) * mode_repr

def _set_state_up(modes         : int                           = 2,
                hilbert         : Optional[HilbertSpace]        = None,
                shape           : Union[int, Tuple[int, ...]]   = (1,),
                mode_repr       : float                         = 0.5,
                backend         : str                           = 'default'
                ):
    '''
    Generate an "all up" state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape (number of spins).
                        For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    Returns:
    - An "all up" state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    xp = get_backend(backend)
    if hilbert is None:
        if modes == 2:
            return xp.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape) * mode_repr
        elif modes == 4:
            # For fermions, "up" means up orbitals occupied and down orbitals empty.
            up_array   = np.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array = np.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            out        = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
            return out
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

def _set_state_down(modes       : int                           = 2,
                    hilbert     : Optional[HilbertSpace]        = None,
                    shape       : Union[int, Tuple[int, ...]]   = (1,),
                    mode_repr   : float                         = 0.5,
                    backend     : str                           = 'default'
                    ):
    '''
    Generate an "all down" state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape (number of spins).
                      For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    Returns:
    - An "all down" state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    xp = get_backend(backend)
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                return xp.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape) * (-mode_repr)
            else:
                return xp.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape)
        elif modes == 4:
            # For fermions, "down" means up orbitals empty and down orbitals occupied.
            up_array   = np.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array = np.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE)
            out        = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
            return out
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

def _set_state_af(modes         : int                           = 2,
                  hilbert       : Optional[HilbertSpace]        = None,
                  shape         : Union[int, Tuple[int, ...]]   = (1,),
                  mode_repr     : float                         = 0.5,
                  backend       : str                           = 'default'
                  ):
    '''
    Generate an antiferromagnetic state configuration.
    
    Parameters:
    - modes         : number of modes (default is 2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system. For modes == 2, the desired shape (number of spins).
                      For fermions (modes == 4), an integer number of sites.
    - mode_repr     : mode representation (default is 0.5 for binary spins +-1, 1.0 for fermions)
    - backend       : computational backend ('default', 'numpy', or 'jax')
    
    Returns:
    - An antiferromagnetic state array.
    '''
    
    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(SamplerErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    xp = get_backend(backend)
    if hilbert is None:
        if modes == 2:
            if Binary.BACKEND_DEF_SPIN:
                af_state = np.array([1 if i % 2 == 0 else -1 for i in range(size)],
                                    dtype=DEFAULT_NP_FLOAT_TYPE)
            else:
                af_state = np.array([1 if i % 2 == 0 else 0 for i in range(size)],
                                    dtype=DEFAULT_NP_FLOAT_TYPE)
            return af_state.reshape(shape) * mode_repr
        elif modes == 4:
            # For fermions, antiferromagnetic state:
            # up orbitals: 1 at even sites, 0 at odd sites
            # down orbitals: 0 at even sites, 1 at odd sites
            up_array   = np.array([1 if i % 2 == 0 else 0 for i in range(size)],
                                  dtype=DEFAULT_NP_FLOAT_TYPE)
            down_array = np.array([0 if i % 2 == 0 else 1 for i in range(size)],
                                  dtype=DEFAULT_NP_FLOAT_TYPE)
            out        = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
            return out
        else:
            raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(SamplerErrors.NOT_IMPLEMENTED_ERROR)

def _state_distinguish(statetype,
                    modes       : int,
                    hilbert     : Optional[HilbertSpace],
                    shape       : Union[int, Tuple[int, ...]],
                    mode_repr   : float = Binary.BACKEND_REPR,
                    rng         = None,
                    rng_k       = None,
                    backend     : str = 'default'):
    """
    Distinguishes the type of the given state and returns the appropriate state configuration.
    
    Parameters:
    - statetype (int, jnp.ndarray, np.ndarray, str, SolverInitState): The state specification.
        * If int: Converts to configuration using _set_state_int.
        * If array: Returns the array directly.
        * If str: Converts to SolverInitState and processes accordingly.
        * If SolverInitState: Processes according to the enum value.
    - modes         : number of modes (2 for binary spins, 4 for fermions)
    - hilbert       : Hilbert space object (optional)
    - shape         : shape of the system.
                        For modes == 2, the desired shape (number of spins);
                        For modes == 4, an integer number of sites.
    - mode_repr     : mode representation value.
    - rng           : random number generator for numpy.
    - rng_k         : random key for JAX.
    - backend       : computational backend ('default', 'numpy', or 'jax').
    
    Returns:
    - The corresponding state configuration as an ndarray.
    """
    if isinstance(statetype, (int, np.integer, jnp.integer)):
        return _set_state_int(statetype, modes, hilbert, shape, mode_repr, backend)
    elif isinstance(statetype, (np.ndarray, jnp.ndarray)):
        return statetype
    elif isinstance(statetype, str):
        try:
            state_enum = SolverInitState.from_str(statetype)
            return _state_distinguish(state_enum, modes, hilbert, shape, mode_repr, rng, rng_k, backend)
        except ValueError as e:
            raise ValueError(SamplerErrors.NOT_A_VALID_STATE_STRING) from e
    elif isinstance(statetype, SolverInitState):
        if statetype == SolverInitState.RND:
            return _set_state_rand(modes, hilbert, shape, mode_repr, backend, rng, rng_k)
        elif statetype == SolverInitState.F_UP:
            return _set_state_up(modes, hilbert, shape, mode_repr, backend)
        elif statetype == SolverInitState.F_DN:
            return _set_state_down(modes, hilbert, shape, mode_repr, backend)
        elif statetype == SolverInitState.AF:
            return _set_state_af(modes, hilbert, shape, mode_repr, backend)
    else:
        raise ValueError(SamplerErrors.NOT_A_VALID_STATE_DISTING)

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
            **kwargs:
                Additional arguments passed to `_state_distinguish` for state generation (e.g., `modes`, `mode_repr`).

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
        """
        if isinstance(backend, str):
            self._backendstr = backend
            
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
            'modes'     : self._modes,
            'mode_repr' : self._mode_repr,
            'different' : self._makediffer,
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
                self._upd_fun = jax.jit(_propose_random_flip_jax)
            else:
                # Use the Numba version (potentially parallelized)
                # Note:
                #   The parallel Numba version needs careful RNG handling if numchains > 1
                self._upd_fun = partial(_propose_random_flips_np, rng=self._rng, num=1)
    
    ###################################################################
    #! ABSTRACT
    ###################################################################
    
    @abstractmethod
    def sample(self, parameters=None, num_samples=None, num_chains=None):
        ''' Tries to sample the state from the Hilbert space. '''
        pass
    
    ###################################################################
    #! BACKEND
    ###################################################################

    @staticmethod
    def obtain_backend(backend: str, seed: Optional[int]):
        '''
        Obtain the backend for the calculations.
        Parameters:
        - backend       : backend for the calculations (default is 'default')
        - seed          : seed for the random number generator
        Returns:
        - Tuple         : backend, backend_sp, rng, rng_k 
        '''
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
                Additional arguments passed to `_state_distinguish` (e.g., `modes`, `mode_repr`).

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
                current_bcknd_str = 'jax' if self._isjax else 'numpy'
                if self._hilbert is None or True:
                    if different_initials:
                        self._states = self._backend.stack([
                                _state_distinguish(initstate,
                                    modes   =   kwargs.get('modes', 2),
                                    hilbert =   self._hilbert,
                                    shape   =   self._shape,
                                    backend =   current_bcknd_str,
                                    rng     =   self._rng,
                                    rng_k   =   self._rng_k)
                                for _ in range(self._numchains)
                            ], axis=0)
                        self._initstate = self._states[0]
                        if not self._isint:
                            self._initstate = self._initstate.astype(self._statetype)
                            self._states = self._states.astype(self._statetype)
                        return
                    else:
                        self._initstate = _state_distinguish(initstate,
                                            modes   =   kwargs.get('modes', 2),
                                            hilbert =   self._hilbert,
                                            shape   =   self._shape,
                                            backend =   current_bcknd_str,
                                            rng     =   self._rng,
                                            rng_k   =   self._rng_k)
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
        '''
        Set the chains for the sampler, replicating the initstate.

        Parameters:
            initstate (np.ndarray or jnp.ndarray):
                The single initial state configuration to replicate.
            numchains (int, optional):
                The number of chains. If None, uses the sampler's current `_numchains`.
        '''
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
        '''
        Set the number of samples.
        Parameters:
            numsamples (int):
                The number of samples
        '''
        self._numsamples = numsamples
        
    def set_numchains(self, numchains):
        '''
        Set the number of chains.
        Parameters:
            numchains (int):
                The number of chains
        '''
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
#! EOF
#######################################################################

