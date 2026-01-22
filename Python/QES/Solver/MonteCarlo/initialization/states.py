"""
Initial state generation for Monte Carlo simulations.

------------------------------
Author          : Maks Kliczkowski
Date            : December 2025
------------------------------
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

try:
    import QES.general_python.common.binary as Binary
    from QES.general_python.algebra.ran_wrapper import choice
    from QES.general_python.algebra.utils import DEFAULT_NP_FLOAT_TYPE, get_backend
except ImportError:
    raise ImportError("Failed to import QES modules.")

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace

# --------------------------------------------------------------------------------------------------
# ERROR CLASSES
# --------------------------------------------------------------------------------------------------


class InitStateErrors:
    NOT_GIVEN_SIZE_ERROR = "The size of the system is not given"
    NOT_IMPLEMENTED_ERROR = "This feature is not implemented yet"
    NOT_A_VALID_STATE_STRING = "The state string is not a valid state string"
    NOT_A_VALID_STATE_DISTING = "The state is not a valid state"


# --------------------------------------------------------------------------------------------------
# ENUM
# --------------------------------------------------------------------------------------------------


class SolverInitState(Enum):
    """Enum for potential initial states."""

    RND = auto()  # random configuration
    F_UP = auto()  # ferromagnetic up
    F_DN = auto()  # ferromagnetic down
    AF = auto()  # antiferromagnetic
    RND_FIXED = auto()  # random configuration with fixed particle number/magnetization

    def __str__(self):
        return self.name

    @classmethod
    def from_str(cls, state_str: str):
        normalized = state_str.upper()
        if normalized in cls.__members__:
            return cls.__members__[normalized]
        raise ValueError(f"Unknown initial state: {state_str}")


# --------------------------------------------------------------------------------------------------
# REGISTRY FOR INITIALIZERS
# --------------------------------------------------------------------------------------------------

_INITIALIZERS: Dict[Any, Callable] = {}


def register_initializer(keys):
    """Decorator to register initialization functions."""

    def decorator(func):
        for k in keys:
            _INITIALIZERS[k] = func
            if isinstance(k, str):
                _INITIALIZERS[k.upper()] = func
                _INITIALIZERS[k.lower()] = func
        return func

    return decorator


# --------------------------------------------------------------------------------------------------
# IMPLEMENTATION FUNCTIONS
# --------------------------------------------------------------------------------------------------


@register_initializer([SolverInitState.RND, "random", "rnd"])
def _set_state_rand(
    modes: int = 2,
    hilbert: Optional["HilbertSpace"] = None,
    shape: Union[int, Tuple[int, ...]] = (1,),
    mode_repr: float = 0.5,
    backend: str = "default",
    rng=None,
    rng_k=None,
    **kwargs,
):
    """Generate random state."""
    if shape is None:
        if hilbert is None:
            raise ValueError(InitStateErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns

    size = shape if isinstance(shape, int) else int(np.prod(shape))
    ran_state = None

    # Original logic: if hilbert is None or True (always True)
    if modes == 2:
        if Binary.BACKEND_DEF_SPIN:
            ran_state = choice([-1, 1], shape, rng=rng, rng_k=rng_k, backend=backend)
        else:
            ran_state = choice([0, 1], shape, rng=rng, rng_k=rng_k, backend=backend)
    elif modes == 4:
        ran_state = choice([0, 1], 2 * size, rng=rng, rng_k=rng_k, backend=backend)
    else:
        raise NotImplementedError(InitStateErrors.NOT_IMPLEMENTED_ERROR)

    return ran_state.astype(DEFAULT_NP_FLOAT_TYPE).reshape(shape) * mode_repr


@register_initializer([SolverInitState.F_UP, "ferro_up", "f_up", "up", "all_up"])
def _set_state_up(
    modes: int = 2,
    hilbert: Optional["HilbertSpace"] = None,
    shape: Union[int, Tuple[int, ...]] = (1,),
    mode_repr: float = 0.5,
    backend: str = "default",
    **kwargs,
):
    """Generate all-up state."""
    if shape is None:
        if hilbert is None:
            raise ValueError(InitStateErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    xp = get_backend(backend)

    if modes == 2:
        return xp.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape) * mode_repr
    elif modes == 4:
        up_array = np.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE)
        down_array = np.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE)
        out = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
        return xp.array(out)
    else:
        raise NotImplementedError(InitStateErrors.NOT_IMPLEMENTED_ERROR)


@register_initializer([SolverInitState.F_DN, "ferro_down", "f_down", "down", "all_down", "f_dn"])
def _set_state_down(
    modes: int = 2,
    hilbert: Optional["HilbertSpace"] = None,
    shape: Union[int, Tuple[int, ...]] = (1,),
    mode_repr: float = 0.5,
    backend: str = "default",
    **kwargs,
):
    """Generate all-down state."""
    if shape is None:
        if hilbert is None:
            raise ValueError(InitStateErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    xp = get_backend(backend)

    if modes == 2:
        if Binary.BACKEND_DEF_SPIN:
            return xp.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape) * (-mode_repr)
        else:
            return xp.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE).reshape(shape)
    elif modes == 4:
        up_array = np.zeros(size, dtype=DEFAULT_NP_FLOAT_TYPE)
        down_array = np.ones(size, dtype=DEFAULT_NP_FLOAT_TYPE)
        out = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
        return xp.array(out)
    else:
        raise NotImplementedError(InitStateErrors.NOT_IMPLEMENTED_ERROR)


@register_initializer([SolverInitState.AF, "antiferro", "af", "neel"])
def _set_state_af(
    modes: int = 2,
    hilbert: Optional["HilbertSpace"] = None,
    shape: Union[int, Tuple[int, ...]] = (1,),
    mode_repr: float = 0.5,
    backend: str = "default",
    **kwargs,
):
    """Generate antiferromagnetic state."""
    if shape is None:
        if hilbert is None:
            raise ValueError(InitStateErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    xp = get_backend(backend)

    if modes == 2:
        if Binary.BACKEND_DEF_SPIN:
            af_state = np.array(
                [1 if i % 2 == 0 else -1 for i in range(size)], dtype=DEFAULT_NP_FLOAT_TYPE
            )
        else:
            af_state = np.array(
                [1 if i % 2 == 0 else 0 for i in range(size)], dtype=DEFAULT_NP_FLOAT_TYPE
            )
        return xp.array(af_state).reshape(shape) * mode_repr
    elif modes == 4:
        up_array = np.array(
            [1 if i % 2 == 0 else 0 for i in range(size)], dtype=DEFAULT_NP_FLOAT_TYPE
        )
        down_array = np.array(
            [0 if i % 2 == 0 else 1 for i in range(size)], dtype=DEFAULT_NP_FLOAT_TYPE
        )
        out = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr
        return xp.array(out)
    else:
        raise NotImplementedError(InitStateErrors.NOT_IMPLEMENTED_ERROR)


@register_initializer([SolverInitState.RND_FIXED, "random_fixed", "fixed_n", "conserved"])
def _set_state_rand_fixed(
    modes: int = 2,
    hilbert: Optional["HilbertSpace"] = None,
    shape: Union[int, Tuple[int, ...]] = (1,),
    mode_repr: float = 0.5,
    backend: str = "default",
    rng=None,
    rng_k=None,
    n_particles: Optional[int] = None,
    magnetization: Optional[float] = None,
    **kwargs,
):
    """Generate random state with fixed particle number/magnetization."""
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    xp = get_backend(backend)

    # Determine counts for Spin-1/2 (modes=2)
    n_up = 0
    if n_particles is not None:
        n_up = int(n_particles)
    elif magnetization is not None:
        # M = N_up - N_down = 2*N_up - N  => N_up = (N + M)/2
        n_up = int((size + magnetization) // 2)
    else:
        n_up = size // 2

    n_down = size - n_up

    # Create base array
    if Binary.BACKEND_DEF_SPIN:
        # Spin representation: up = mode_repr, down = -mode_repr
        arr = np.concatenate(
            [np.full(n_up, mode_repr, dtype=float), np.full(n_down, -mode_repr, dtype=float)]
        )
    else:
        # Binary representation: up = mode_repr (usually 1), down = 0
        arr = np.concatenate(
            [np.full(n_up, mode_repr, dtype=float), np.full(n_down, 0.0, dtype=float)]
        )

    # Shuffle
    if backend == "numpy" or backend == np:
        rng.shuffle(arr)
        res = xp.array(arr, dtype=DEFAULT_NP_FLOAT_TYPE)
    else:
        # JAX
        if rng_k is None:
            raise ValueError("RNG key required for JAX backend")
        res = jax.random.permutation(rng_k, xp.array(arr, dtype=DEFAULT_NP_FLOAT_TYPE))

    return res.reshape(shape)


# Helper for direct integer conversion
def _set_state_int(
    state: int,
    modes: int = 2,
    hilbert: Optional["HilbertSpace"] = None,
    shape: Union[int, Tuple[int, ...]] = (1,),
    mode_repr: float = 0.5,
    backend: str = "default",
):
    """Set state from integer representation."""

    # check the size from the shape
    if shape is None:
        if hilbert is None:
            raise ValueError(InitStateErrors.NOT_GIVEN_SIZE_ERROR)
        else:
            shape = hilbert.ns
    # check the size from the hilbert
    size = shape if isinstance(shape, int) else int(np.prod(shape))
    out = None

    if hilbert is None:
        if modes == 2:
            # set the state from tensor
            out = Binary.int2base(
                state, size, backend, spin_value=mode_repr, spin=Binary.BACKEND_DEF_SPIN
            ).reshape(shape)
        elif modes == 4:
            # For fermions we construct using numpy then convert if needed?
            # Binary.int2base might not support modes=4 logic directly if it's just bits.
            # The original code used manual bit logic.
            up_int = state & ((1 << size) - 1)
            down_int = state >> size
            up_array = np.array(
                [1 if (up_int & (1 << i)) else 0 for i in range(size)], dtype=DEFAULT_NP_FLOAT_TYPE
            )
            down_array = np.array(
                [1 if (down_int & (1 << i)) else 0 for i in range(size)],
                dtype=DEFAULT_NP_FLOAT_TYPE,
            )
            out = np.stack((up_array, down_array), axis=0).flatten().reshape(shape) * mode_repr

            xp = get_backend(backend)
            if backend != "numpy" and backend != np:
                out = xp.array(out)
        else:
            raise NotImplementedError(InitStateErrors.NOT_IMPLEMENTED_ERROR)
    else:
        raise NotImplementedError(InitStateErrors.NOT_IMPLEMENTED_ERROR)
    return out


# --------------------------------------------------------------------------------------------------
# MAIN FACTORY FUNCTION
# --------------------------------------------------------------------------------------------------


def initialize_state(
    statetype,
    modes: int,
    hilbert: Optional["HilbertSpace"],
    shape: Union[int, Tuple[int, ...]],
    mode_repr: float = 0.5,  # Binary.BACKEND_REPR usually 0.5
    rng=None,
    rng_k=None,
    backend: str = "default",
    **kwargs,
):
    """
    Primary factory function to generate an initial state configuration.

    Parameters:
        statetype: One of {int, array, str, SolverInitState, Callable}
        modes: Number of local states (2 for spin-1/2)
        hilbert: HilbertSpace object
        shape: System shape
        mode_repr: Value representation (e.g. 0.5 for +/- 0.5 spins)
        rng: Numpy RNG
        rng_k: JAX RNG Key
        backend: 'numpy' or 'jax'
        **kwargs: Extra args for specific initializers (n_particles, etc.)
    """

    # 1. Direct Array
    if isinstance(statetype, (np.ndarray)) or (
        JAX_AVAILABLE and isinstance(statetype, jnp.ndarray)
    ):
        return statetype

    # 2. Integer
    if isinstance(statetype, (int, np.integer)):
        if JAX_AVAILABLE and isinstance(statetype, jnp.integer):
            return _set_state_int(int(statetype), modes, hilbert, shape, mode_repr, backend)
        return _set_state_int(statetype, modes, hilbert, shape, mode_repr, backend)

    # 3. String / Enum via Registry
    # Normalize string to upper case if it's a string, or check Enum

    lookup_key = statetype
    if isinstance(statetype, str):
        # Try finding in registry directly (case insensitive handled by registration)
        if statetype in _INITIALIZERS:
            pass  # found
        elif statetype.upper() in _INITIALIZERS:
            lookup_key = statetype.upper()
        elif statetype.lower() in _INITIALIZERS:
            lookup_key = statetype.lower()
        else:
            # Fallback: try converting to Enum (backward compatibility for strict Enum names)
            try:
                lookup_key = SolverInitState.from_str(statetype)
            except ValueError:
                # If still not found, check if it's a known error or just unknown
                raise ValueError(InitStateErrors.NOT_A_VALID_STATE_STRING + f": {statetype}")

    if lookup_key in _INITIALIZERS:
        return _INITIALIZERS[lookup_key](
            modes=modes,
            hilbert=hilbert,
            shape=shape,
            mode_repr=mode_repr,
            backend=backend,
            rng=rng,
            rng_k=rng_k,
            **kwargs,
        )

    # 4. Callable
    if callable(statetype):
        # Custom initialization function: f(shape, key/rng, **kwargs)
        r_obj = rng if (backend == "numpy" or backend == np) else rng_k
        return statetype(shape, r_obj, **kwargs)

    raise ValueError(InitStateErrors.NOT_A_VALID_STATE_DISTING + f": {statetype}")


# ----------------------------------------
#! EOF
# ----------------------------------------
