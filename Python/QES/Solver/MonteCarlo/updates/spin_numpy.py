"""
NumPy-based update rules for Spin-1/2 systems.

----------------------------------------------------------------------
Author          : Maks Kliczkowski
Date            : December 2025
Description     : This module provides functions to propose various types of updates
                (local flips, multi-flips) for spin-1/2 systems using NumPy/Numba.
----------------------------------------------------------------------
"""

import numba
import numpy as np
from numba import prange

try:
    import QES.general_python.common.binary as Binary
    from QES.general_python.algebra.ran_wrapper import randint_np
except ImportError:
    raise ImportError("QES.general_python modules are required for spin updates.")

# ----------------------------------------------------------------------
# Local Updates
# ----------------------------------------------------------------------


@numba.njit(parallel=True)
def propose_local_flip_np(state: np.ndarray, rng: np.random.Generator):
    """
    Propose a random flip of a state using numpy.
    """
    if state.ndim == 1:
        idx = randint_np(rng=rng, low=0, high=state.size, size=1)[0]
        return Binary.flip_array_np(state, idx)

    n_chains, state_size = state.shape[0], state.shape[1]
    for i in prange(n_chains):
        # Note: In parallel numba context, we might need thread-safe RNG or
        # allow randint_np to handle it. Assuming implicit handling as per original code.
        idx = randint_np(low=0, high=state_size, size=1)[0]
        state[i] = Binary.flip_array_np_spin(state[i], idx)
    return state


# ----------------------------------------------------------------------
# Multi-Flip Updates
# ----------------------------------------------------------------------


@numba.njit
def propose_multi_flip_np(state: np.ndarray, rng, num=1):
    """
    Propose a random flip of a state using numpy.
    """
    if state.ndim == 1:
        idx = randint_np(rng=rng, low=0, high=state.size, size=num)
        return Binary.flip_array_np_multi(
            state, idx, spin=Binary.BACKEND_DEF_SPIN, spin_value=Binary.BACKEND_REPR
        )
    n_chains, state_size = state.shape[0], state.shape[1]
    for i in range(n_chains):
        idx = randint_np(rng=rng, low=0, high=state_size, size=num)
        state[i] = Binary.flip_array_np_multi(
            state[i], idx, spin=Binary.BACKEND_DEF_SPIN, spin_value=Binary.BACKEND_REPR
        )
    return state


# ----------------------------------------
#! EOF
# ----------------------------------------
