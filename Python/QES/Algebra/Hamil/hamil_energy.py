"""
Local-energy helper assembly for Hamiltonian workflows.

This module builds the NumPy-side local-energy closures used by Hamiltonian
and variational workflows. It groups state-modifying and non-modifying
operator terms into compact helper kernels and returns one callable that
evaluates local energy contributions efficiently.

file    : Algebra/hamil_energy.py
author  :
"""

################################################################################

from typing import Any, Callable, List, Optional, Tuple

import numba
import numba.typed
import numpy as np

from QES.Algebra.Hamil.hamil_energy_helper import flatten_operator_terms, unpack_operator_terms
from QES.general_python.algebra.utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax.numpy as jnp

    from QES.Algebra.Hamil.hamil_energy_jax import local_energy_jax_wrap
else:
    local_energy_jax_wrap = None
    jax = None
    jnp = np

# -----------------------------------------------------------------------------
# NumPy array representation


# @numba.njit
def process_mod_sites(
    state: np.ndarray, sites_args: Any, multiplier: float, op_func: Callable, dtype: np.dtype
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a state-modifying operator to the given array state.
    This implementation needs sites to be passed as a list.

    Parameters
    ----------
    state : np.ndarray
        One-dimensional state array.
    sites_args : Any
        Site-specific argument payload for the operator.
    multiplier : float
        Scalar multiplier applied to the returned operator values.
    op_func : Callable
        Operator function taking `(state, sites_args)` and returning
        `(new_state, op_value)`.
    dtype : np.dtype
        Output dtype for generated coefficient arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two-dimensional output states and scaled coefficients.
    """
    if op_func is None:
        return state.reshape(1, -1), np.zeros((1,), dtype=dtype)

    new_state, op_value = op_func(state, sites_args)

    if new_state.ndim == 1:
        new_state_2d = new_state.reshape(1, new_state.shape[0])
    else:
        new_state_2d = new_state.reshape(new_state.shape[0], -1)

    # Force op_value to be at least a 1D array.
    num_states = new_state_2d.shape[0]
    op_value = np.atleast_1d(np.asarray(op_value).astype(dtype))
    op_value_1d = op_value * multiplier  # Multiply elementwise

    # If op_value has a single element but multiple states exist, broadcast it.
    if op_value_1d.shape[0] == 1 and num_states > 1:
        op_values_final = np.full(num_states, op_value_1d[0], dtype=dtype)
    elif op_value_1d.shape[0] == num_states:
        op_values_final = op_value_1d
    else:
        raise ValueError("op_value length does not match number of states in process_mod_nosites")

    return new_state_2d, op_values_final.astype(dtype)


# @numba.njit
def process_mod_nosites(
    state: np.ndarray, multiplier: float, op_func: Callable, dtype: np.dtype
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a state-modifying operator to the given array state.
    This implementation does not require sites to be passed as a list.

    Parameters
    ----------
    state : np.ndarray
        One-dimensional state array.
    multiplier : float
        Scalar multiplier applied to the returned operator values.
    op_func : Callable
        Operator function taking `(state)` and returning `(new_state, op_value)`.
    dtype : np.dtype
        Output dtype for generated coefficient arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two-dimensional output states and scaled coefficients.
    """
    if op_func is None:
        return state.reshape(1, -1), np.zeros((1,), dtype=dtype)

    new_state, op_value = op_func(state)

    if new_state.ndim == 1:
        new_state_2d = new_state.reshape(1, new_state.shape[0])
    else:
        new_state_2d = new_state.reshape(new_state.shape[0], -1)

    # Force op_value to be at least a 1D array.
    num_states = new_state_2d.shape[0]
    op_value = np.atleast_1d(np.asarray(op_value).astype(dtype))
    op_value_1d = op_value * multiplier  # Multiply elementwise

    # If op_value has a single element but multiple states exist, broadcast it.
    if op_value_1d.shape[0] == 1 and num_states > 1:
        op_values_final = np.full(num_states, op_value_1d[0], dtype=dtype)
    elif op_value_1d.shape[0] == num_states:
        op_values_final = op_value_1d
    else:
        raise ValueError("op_value length does not match number of states in process_mod_nosites")

    return new_state_2d, op_values_final.astype(dtype)


# @numba.njit
def process_nmod_sites(
    state: np.ndarray,
    sites_flat: List[np.ndarray],
    multipliers: np.ndarray,
    op_funcs: List[Callable],
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a non-modifying operator to the given array state.
    This implementation needs sites to be passed as a list.

    Parameters
    ----------
    state : np.ndarray
        One-dimensional state array.
    sites_flat : List[np.ndarray]
        Site-argument payloads for each operator contribution.
    multipliers : np.ndarray
        Scalar multipliers for each operator contribution.
    op_funcs : List[Callable]
        Operator functions taking `(state, sites)` and returning
        `(new_state, op_value)`.
    dtype : np.dtype
        Output dtype for accumulated values.

    Returns
    -------
    np.ndarray
        Accumulated non-modifying contribution.
    """

    value = np.zeros((1,), dtype=dtype)
    if op_funcs is None:
        return value

    numfunc = multipliers.shape[0]
    # for ii in numba.prange(numfunc):
    for ii in range(numfunc):
        op_func = op_funcs[ii]
        sites = sites_flat[ii]
        multiplier = multipliers[ii]
        _, op_value = op_func(state, sites)
        op_value = multiplier * op_value
        op_value = np.asarray(op_value, dtype=dtype)
        value += np.sum(op_value)
    return value


# @numba.njit
def process_nmod_nosites(
    state: np.ndarray, multipliers: np.ndarray, op_funcs: List[Callable], dtype: np.dtype
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a non-modifying operator to the given array state.
    This implementation does not require sites to be passed as a list.

    Parameters
    ----------
    state : np.ndarray
        One-dimensional state array.
    multipliers : np.ndarray
        Scalar multipliers for each operator contribution.
    op_funcs : List[Callable]
        Operator functions taking `(state)` and returning `(new_state, op_value)`.
    dtype : np.dtype
        Output dtype for accumulated values.

    Returns
    -------
    np.ndarray
        Accumulated non-modifying contribution.
    """

    value = np.zeros((1,), dtype=dtype)
    if op_funcs is None:
        return value

    numfunc = multipliers.shape[0]
    # for ii in numba.prange(numfunc):
    for ii in range(numfunc):
        op_func = op_funcs[ii]
        multiplier = multipliers[ii]
        _, op_value = op_func(state)
        op_value = multiplier * op_value
        op_value = np.asarray(op_value, dtype=dtype)
        value += np.sum(op_value)
    return value


# -----------------------------------------------------------------------------


def local_energy_np_wrap(
    ns: int,
    operator_terms_list: List,
    operator_terms_list_ns: List,
    operator_terms_list_nmod: List,
    operator_terms_list_nmod_ns: List,
    n_max: Optional[int] = 1,
    dtype: Optional[np.dtype] = np.float32,
) -> Callable:
    """
    Generate a local-energy wrapper function for NumPy states.

    This function processes various operator term lists by unpacking and flattening them, converting the multiplier
    lists to numpy arrays of the specified dtype, and calculating auxiliary parameters required for energy evaluation.
    It then defines and returns a Numba nopython-mode (njit) compiled wrapper function that executes the local energy
    calculation using the prepared data.

    Parameters
    ----------
    ns : int
        Number of sites or degrees of freedom.
    operator_terms_list : List
        Operator terms for state-modifying operators with site arguments.
    operator_terms_list_ns : List
        Operator terms for state-modifying operators without site arguments.
    operator_terms_list_nmod : List
        Operator terms for non-modifying operators with site arguments.
    operator_terms_list_nmod_ns : List
        Operator terms for non-modifying operators without site arguments.
    n_max : Optional[int], default=1
        Maximum number of output states produced by one modifying operator.
    dtype : Optional[np.dtype], default=np.float32
        Numeric dtype used for working arrays.

    Returns
    -------
    Callable
        Callable taking a state array and returning the local-energy payload.
    """

    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    # unpack the operator terms
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites = unpack_operator_terms(
        ns, operator_terms_list
    )
    _op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites = unpack_operator_terms(
        ns, operator_terms_list_ns
    )
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites = unpack_operator_terms(
        ns, operator_terms_list_nmod
    )
    _op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites = unpack_operator_terms(
        ns, operator_terms_list_nmod_ns
    )

    # flatten the operator terms for all the operators
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites = flatten_operator_terms(
        _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites
    )
    _op_f_mod_nosites, _, _op_m_mod_nosites = flatten_operator_terms(
        _op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites
    )
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites = flatten_operator_terms(
        _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites
    )
    _op_f_nmod_nosites, _, _op_m_nmod_nosites = flatten_operator_terms(
        _op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites
    )

    # change all multipliers to numpy arrays
    _op_m_mod_sites_np = np.array(_op_m_mod_sites, dtype=dtype)
    _op_m_mod_nosites_np = np.array(_op_m_mod_nosites, dtype=dtype)
    _op_m_nmod_sites_np = np.array(_op_m_nmod_sites, dtype=dtype)
    _op_m_nmod_nosites_np = np.array(_op_m_nmod_nosites, dtype=dtype)

    # calculate the number of non-local and local operators
    total_rows_sites = len(_op_f_mod_sites) * n_max
    total_rows_nosites = len(_op_f_mod_nosites) * n_max

    # convert the functions to numba lists
    _op_f_mod_sites_py = tuple(_op_f_mod_sites) if len(_op_f_mod_sites) > 0 else None
    _op_f_mod_nosites_py = tuple(_op_f_mod_nosites) if len(_op_f_mod_nosites) > 0 else None
    _op_f_nmod_sites_py = tuple(_op_f_nmod_sites) if len(_op_f_nmod_sites) > 0 else None
    _op_f_nmod_nosites_py = tuple(_op_f_nmod_nosites) if len(_op_f_nmod_nosites) > 0 else None

    def wrap_fun(_op_f, _op_i):
        sites_args = np.asarray(_op_i, dtype=np.int32)

        @numba.njit
        def wrapper(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return _op_f(state, sites_args)

        return wrapper

    _op_f_mod_sites_py = tuple(wrap_fun(f, i) for f, i in zip(_op_f_mod_sites, _op_i_mod_sites))
    _op_f_nmod_sites_py = tuple(wrap_fun(f, i) for f, i in zip(_op_f_nmod_sites, _op_i_nmod_sites))
    _total_rows_mod_sites = total_rows_sites
    _total_rows_mod_nosites = total_rows_nosites
    _nmax = n_max

    #! Create the wrapper function
    # @numba.njit
    def wrapper(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create typed lists to accumulate results.
        state_dim = state.shape[0]
        has_mod_sites = len(_op_f_mod_sites) > 0 and _op_f_mod_sites_py is not None
        has_mod_nosites = len(_op_f_mod_nosites) > 0 and _op_f_mod_nosites_py is not None
        #! start with local energy

        if has_mod_sites:
            e1 = process_nmod_nosites(state, _op_m_nmod_sites_np, _op_f_nmod_sites_py, dtype)
        else:
            e1 = np.zeros((1,), dtype=dtype)
        if has_mod_nosites:
            e2 = process_nmod_nosites(state, _op_m_nmod_nosites_np, _op_f_nmod_nosites_py, dtype)
        else:
            e2 = np.zeros((1,), dtype=dtype)
        diagonal_value = e1 + e2
        out_states = state.reshape(1, -1)
        out_values = diagonal_value

        #! continue with modifying operators
        num_mod_sites = len(_op_m_mod_sites_np)
        if num_mod_sites > 0 and _op_f_mod_sites_py is not None:
            states_list_mod_sites = np.zeros((_total_rows_mod_sites, state_dim), dtype=dtype)
            values_list_mod_sites = np.zeros((_total_rows_mod_sites,), dtype=dtype)
            mask_sites = np.zeros((_total_rows_mod_sites,), dtype=np.bool_)

            # for i in numba.prange(num_mod_sites):
            for i in range(num_mod_sites):
                new_states, op_values = process_mod_nosites(
                    state, _op_m_mod_sites_np[i], _op_f_mod_sites_py[i], dtype
                )
                # check the size
                new_states_size = new_states.shape[0]
                if new_states_size > _nmax:
                    raise ValueError(f"new_states_size ({new_states_size}) > _nmax ({_nmax})")
                start_idx = i * _nmax
                end_idx = i * _nmax + new_states_size
                states_list_mod_sites[start_idx:end_idx, :] = new_states
                values_list_mod_sites[start_idx:end_idx] = op_values
                mask_sites[start_idx:end_idx] = True

            # concatenate the results
            states_list_mod_sites = states_list_mod_sites[mask_sites]
            values_list_mod_sites = values_list_mod_sites[mask_sites]
            if states_list_mod_sites.shape[0] > 0:
                out_states = np.concatenate((out_states, states_list_mod_sites), axis=0)
                out_values = np.concatenate((out_values, values_list_mod_sites), axis=0)

        #! continue with non-modifying operators
        num_mod_nosites = len(_op_m_mod_nosites_np)
        if num_mod_nosites > 0 and _op_f_mod_nosites_py is not None:
            states_list_nosites = np.zeros((_total_rows_mod_nosites, state_dim), dtype=dtype)
            values_list_nosites = np.zeros((_total_rows_mod_nosites,), dtype=dtype)
            mask_nosites = np.zeros((_total_rows_mod_nosites,), dtype=np.bool_)

            # for i in numba.prange(num_mod_nosites):
            for i in range(num_mod_nosites):
                new_states, op_values = process_mod_nosites(
                    state, _op_m_mod_nosites_np[i], _op_f_mod_nosites_py[i], dtype
                )
                # check the size
                new_states_size = new_states.shape[0]
                if new_states_size > _nmax:
                    raise ValueError(f"new_states_size ({new_states_size}) > _nmax ({_nmax})")
                start_idx = i * _nmax
                end_idx = i * _nmax + new_states_size
                states_list_nosites[start_idx:end_idx, :] = new_states
                values_list_nosites[start_idx:end_idx] = op_values
                mask_nosites[start_idx:end_idx] = True

            # concatenate the results
            states_list_nosites = states_list_nosites[mask_nosites]
            values_list_nosites = values_list_nosites[mask_nosites]
            if states_list_nosites.shape[0] > 0:
                out_states = np.concatenate((out_states, states_list_nosites), axis=0)
                out_values = np.concatenate((out_values, values_list_nosites), axis=0)
        return out_states, out_values

    return wrapper


################################################################################
