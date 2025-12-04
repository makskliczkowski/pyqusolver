'''
Module      : Algebra/hamil_energy_jax.py
Author      : Maksymilian Kliczkowski
Date        : 2025-04-01
Description : This module implements functions for efficient
                Hamiltonian energy calculations using JAX.

-------------------------
This module implements functions for efficient Hamiltonian energy calculations using JAX.
It provides routines to compute local energy contributions of a quantum state by combining
various operator terms. The module distinguishes between operators that modify the state
with explicit site indices and those that do not, and it aggregates their contributions
using JAX's control flow primitives such as lax.scan, lax.switch, and lax.fori_loop.

The key functionalities include:
    - local_energy_jax_nonmod_sites:
        Computes the local energy contribution for a given state from operators that
        require site indices. It uses a loop (via lax.scan) to process each operator,
        applying corresponding multipliers and combining their contributions.
    - local_energy_jax_nonmod_nosites:
        Computes the local energy contribution for operators that do not involve site
        indices. Similar in structure to the sites version, it iterates over the provided
        operators and aggregates the energy contributions.
    - local_energy_jax_nonmod:
        Aggregates the local energy contributions by combining the results from the
        non-site and site-based operator evaluations.
    - _pad_indices_2d:
        Pads a list of index lists to form a uniform 2D array, returning both the padded
        indices and a boolean mask indicating the valid entries. This is crucial for handling
        variable-length site index lists during energy evaluation.
    - local_energy_jax_wrap:
        Provides a higher-level JIT-compiled wrapper function that prepares, unpacks, and
        flattens the operator terms. It preallocates output arrays for both the updated
        states and their corresponding energy contributions, and it loops through different
        operator groups (those modifying the state with or without sites, and those that do not
        modify the state) to update the energy and state arrays accordingly.
Usage:
    The functions in this module are designed for use within quantum eigen solvers or
    similar applications where one needs to compute and differentiate through local energy
    contributions. The module leverages JAX's support for just-in-time (JIT) compilation and
    automatic differentiation to enable fast, efficient computation on modern hardware.
Notes:
    - Operator terms should be provided with their corresponding multipliers.
    - Site-dependent operators use padded indices and masks to handle variable-length inputs.
    - The wrapper function, local_energy_jax_wrap, combines all operator contributions into a
        single callable that returns aggregated state updates and energy contributions.
        
This structured approach allows seamless integration of various Hamiltonian contributions while
ensuring optimal performance and differentiability using the JAX ecosystem.
'''

################################################################################

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional, List, Callable, Union
from QES.Algebra.Hamil.hamil_energy_helper import unpack_operator_terms, flatten_operator_terms

################################################################################

@partial(jax.jit, static_argnums=(1,))
def local_energy_jax_nonmod_nosites(
    state       : jnp.ndarray,
    functions   : Tuple[Callable, ...],
    multipliers : jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute local energy contributions for a given state from operators that
    do NOT depend explicitly on site indices and do NOT modify the state.

    Each function f in `functions` is assumed to satisfy

        new_state, coeff = f(state)

    where `new_state` can be ignored (typically equal to `state`) and
    `coeff` is a scalar contribution from that operator term.

    Parameters
    ----------
    state
        1D array of shape (ns,).
    functions
        Tuple of callables.
    multipliers
        1D array of shape (n_ops,) with operator multipliers.

    Returns
    -------
    local_energy : jnp.ndarray
        Shape (1,), complex or real, with the total diagonal energy.
    """

    num_local = len(functions)
    if num_local == 0:
        # Return exact zero with same dtype as state (usually complex)
        return jnp.zeros((1,), dtype=state.dtype)

    multipliers = jnp.asarray(multipliers)
    dtype_e     = jnp.result_type(state, multipliers)
    multipliers = multipliers.astype(dtype_e)

    def scan_fn(carry, i):
        _, coeff = jax.lax.switch(i, functions, state)
        coeff    = jnp.asarray(coeff).astype(dtype_e)
        contrib  = jnp.squeeze(coeff) * multipliers[i]
        return carry + contrib, 0.0

    init = jnp.zeros((), dtype=dtype_e)
    total, _ = jax.lax.scan(scan_fn, init, jnp.arange(num_local, dtype=jnp.int32))
    return total.reshape((1,))


################################################################################
# Non-modifying (diagonal) energy – with/without sites
################################################################################

@partial(jax.jit, static_argnums=(1, 3))
def local_energy_jax_nonmod(
    state              : jnp.ndarray,
    functions_no_sites : Tuple[Callable, ...],
    mult_no_sites      : jnp.ndarray,
    functions_sites    : Tuple[Callable, ...],
    mult_sites         : jnp.ndarray,
) -> jnp.ndarray:
    """
    Combine diagonal (non-modifying) energies with and without site indices.

    Parameters
    ----------
    state
        1D array of shape (ns,).
    functions_no_sites
        Tuple of callables f(state) -> (state, coeff).
    mult_no_sites
        Multipliers for `functions_no_sites`.
    functions_sites
        Tuple of callables f(state, *sites) already wrapped to depend only
        on state: f_wrapped(state) -> (state, coeff).
    mult_sites
        Multipliers for `functions_sites`.

    Returns
    -------
    local_energy : jnp.ndarray
        Shape (1,), diagonal energy.
    """

    e1 = local_energy_jax_nonmod_nosites(state, functions_no_sites, mult_no_sites)
    e2 = local_energy_jax_nonmod_nosites(state, functions_sites,    mult_sites)
    return e1 + e2


################################################################################
# Main wrapper
################################################################################

def local_energy_jax_wrap(
    ns                        : int,
    operator_terms_list       : List,
    operator_terms_list_ns    : List,
    operator_terms_list_nmod  : List,
    operator_terms_list_nmod_ns : List,
    n_max                     : Optional[int]       = 1,
    dtype                     : Optional[jnp.dtype] = jnp.complex128,
) -> Callable:
    """
    Build a JIT-compiled local-energy function

        state -> (all_states, all_energies)

    where:

        all_states   : (N_total, ns)
        all_energies : (N_total,)

    and N_total = 1 (diagonal) + sum_i K_i over all modifying operators,
    with each modifying operator allowed to return MANY states:

        f(state, *sites) -> (new_states, coeffs)
        new_states.shape == (K, ns)
        coeffs.shape     == (K,)
    """

    # Normalise dtype to a JAX dtype; default to complex for safety
    if not isinstance(dtype, jnp.dtype):
        dtype = jnp.dtype(dtype)
        
    if dtype not in (jnp.complex64, jnp.complex128):
        dtype = jnp.complex128

    # -------------------------------------------------------------------------
    # 1. Unpack and flatten all operator groups
    # -------------------------------------------------------------------------
    f_mod_sites,  i_mod_sites,  m_mod_sites  = unpack_operator_terms(ns, operator_terms_list)
    f_mod_nos,    i_mod_nos,    m_mod_nos    = unpack_operator_terms(ns, operator_terms_list_ns)
    f_nmod_sites, i_nmod_sites, m_nmod_sites = unpack_operator_terms(ns, operator_terms_list_nmod)
    f_nmod_nos,   i_nmod_nos,   m_nmod_nos   = unpack_operator_terms(ns, operator_terms_list_nmod_ns)

    f_mod_sites,  i_mod_sites,  m_mod_sites  = flatten_operator_terms(f_mod_sites,  i_mod_sites,  m_mod_sites)
    f_mod_nos,    i_mod_nos,    m_mod_nos    = flatten_operator_terms(f_mod_nos,    i_mod_nos,    m_mod_nos)
    f_nmod_sites, i_nmod_sites, m_nmod_sites = flatten_operator_terms(f_nmod_sites, i_nmod_sites, m_nmod_sites)
    f_nmod_nos,   i_nmod_nos,   m_nmod_nos   = flatten_operator_terms(f_nmod_nos,   i_nmod_nos,   m_nmod_nos)

    # multipliers as JAX arrays (energies will inherit this dtype)
    m_mod_sites   = jnp.asarray(m_mod_sites,   dtype=dtype)
    m_mod_nos     = jnp.asarray(m_mod_nos,     dtype=dtype)
    m_nmod_sites  = jnp.asarray(m_nmod_sites,  dtype=dtype)
    m_nmod_nos    = jnp.asarray(m_nmod_nos,    dtype=dtype)

    # -------------------------------------------------------------------------
    # 2. Wrap operators so that modifying ones ALWAYS return (K, ns) and (K,)
    # -------------------------------------------------------------------------
    def wrap_mod(f, sites):
        sites = tuple(int(s) for s in sites)

        def op(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # f may itself be jitted; we just shape-normalise the result
            new_states, coeffs = f(state, *sites)
            new_states = jnp.asarray(new_states, dtype=state.dtype)
            coeffs     = jnp.asarray(coeffs,     dtype=dtype)
            # Enforce 2D states: (K, ns)
            if new_states.ndim == 1:
                new_states = new_states.reshape((1, new_states.shape[0]))
            return new_states, coeffs.reshape((-1,))

        return op

    def wrap_nos(f):
        def op(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            new_states, coeffs  = f(state)
            new_states          = jnp.asarray(new_states, dtype=state.dtype)
            coeffs              = jnp.asarray(coeffs,     dtype=dtype)
            if new_states.ndim == 1:
                new_states = new_states.reshape((1, new_states.shape[0]))
            return new_states, coeffs.reshape((-1,))
        return op

    # Modifying operators (may return many states)
    f_mod_sites_wrapped  = tuple(wrap_mod(f_mod_sites[i],  i_mod_sites[i])  for i in range(len(f_mod_sites)))
    f_mod_nos_wrapped    = tuple(wrap_nos(f_mod_nos[i])                     for i in range(len(f_mod_nos)))

    # Non-modifying operators: we wrap them to match the diagonal interface
    # expected by local_energy_jax_nonmod_nosites (state -> (state, coeff))
    def wrap_nmod_sites(f, sites):
        sites = tuple(int(s) for s in sites)
        def op(state: jnp.ndarray):
            new_state, coeff = f(state, *sites)
            # new_state is ignored later; keep original state for safety
            return state, coeff
        return op

    def wrap_nmod_nos(f):
        def op(state: jnp.ndarray):
            new_state, coeff = f(state)
            return state, coeff
        return op

    f_nmod_sites_wrapped    = tuple(wrap_nmod_sites(f_nmod_sites[i], i_nmod_sites[i]) for i in range(len(f_nmod_sites)))
    f_nmod_nos_wrapped      = tuple(wrap_nmod_nos(f_nmod_nos[i]) for i in range(len(f_nmod_nos)))

    # -------------------------------------------------------------------------
    # 3. Pure Python wrapper; JAX traces through once and then all is XLA
    # -------------------------------------------------------------------------
    def init_wrapper():

        # Everything captured here is static for JAX
        f_mod_sites_t       = tuple(f_mod_sites_wrapped)
        f_mod_nos_t         = tuple(f_mod_nos_wrapped)
        f_nmod_sites_t      = tuple(f_nmod_sites_wrapped)
        f_nmod_nos_t        = tuple(f_nmod_nos_wrapped)

        m_mod_sites_arr     = jnp.asarray(m_mod_sites,  dtype=dtype)
        m_mod_nos_arr       = jnp.asarray(m_mod_nos,    dtype=dtype)
        m_nmod_sites_arr    = jnp.asarray(m_nmod_sites, dtype=dtype)
        m_nmod_nos_arr      = jnp.asarray(m_nmod_nos,   dtype=dtype)

        @jax.jit
        def wrapper(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            state = jnp.asarray(state)
            ns    = state.shape[0]

            # 1) Diagonal energy (non-modifying operators)
            diag = local_energy_jax_nonmod(
                state,
                f_nmod_nos_t,
                m_nmod_nos_arr,
                f_nmod_sites_t,
                m_nmod_sites_arr,
            ).astype(dtype)  # shape (1,)

            states_list   = [state.reshape((1, ns))]
            energies_list = [diag.reshape((1,))]

            # 2) Modifying operators WITH sites
            for op, mult in zip(f_mod_sites_t, m_mod_sites_arr):
                new_states, coeffs = op(state)          # (K, ns), (K,)
                energies_list.append(coeffs * mult)     # (K,)
                states_list.append(new_states)          # (K, ns)

            # 3) Modifying operators WITHOUT sites
            for op, mult in zip(f_mod_nos_t, m_mod_nos_arr):
                new_states, coeffs = op(state)          # (K, ns), (K,)
                energies_list.append(coeffs * mult)
                states_list.append(new_states)

            all_states   = jnp.concatenate(states_list,   axis=0)
            all_energies = jnp.concatenate(
                [e.reshape((-1,)) for e in energies_list],
                axis=0,
            )

            return all_states, all_energies

        return wrapper

    inner_wrapper = init_wrapper()

    @jax.jit
    def final_wrapper(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Public callable: state -> (all_states, all_energies)
        """
        return inner_wrapper(state)

    return final_wrapper

################################################################################