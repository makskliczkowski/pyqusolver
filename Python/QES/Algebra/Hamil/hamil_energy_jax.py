"""
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
"""

################################################################################

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError("JAX is required for this module. Please install JAX to proceed.") from e

from functools import partial
from typing import Callable, List, Optional, Tuple

try:
    from QES.Algebra.Hamil.hamil_energy_helper import flatten_operator_terms, unpack_operator_terms
except ImportError as e:
    raise ImportError(
        "Failed to import from hamil_energy_helper. Ensure the module exists and is accessible."
    ) from e

################################################################################
# Diagonal (non-modifying) contribution – no sites
################################################################################


@partial(jax.jit, static_argnums=(1,))
def local_energy_jax_nonmod_nosites(
    state: jnp.ndarray,
    functions: Tuple[Callable, ...],
    multipliers: jnp.ndarray,
) -> jnp.ndarray:
    r"""
    Diagonal local energy from operators that do **not** depend on site indices
    and do **not** modify the state.

    Each ``f`` in ``functions`` satisfies

        new_state, coeff = f(state)

    where ``new_state`` is ignored here and ``coeff`` is a scalar (one term
    in the diagonal energy).

    Parameters
    ----------
    state
        1D array, shape ``(ns,)``.
    functions
        Tuple of callables.
    multipliers
        1D array of shape ``(n_ops,)``.

    Returns
    -------
    local_energy : jnp.ndarray
        Shape ``(1,)``, same dtype as ``state``.
    """

    num_local = len(functions)
    if num_local == 0:
        return jnp.zeros((1,), dtype=state.dtype)

    # Work in the multiplying dtype.
    dtype_e = multipliers.dtype
    multipliers = jnp.asarray(multipliers, dtype=dtype_e)

    def loop_body(i, carry):
        _, coeff = jax.lax.switch(i, functions, state)
        coeff = jnp.asarray(coeff, dtype=dtype_e)
        contrib = jnp.squeeze(coeff) * multipliers[i]
        return carry + contrib

    init = jnp.zeros((), dtype=dtype_e)
    total = jax.lax.fori_loop(0, num_local, loop_body, init)
    return total.reshape((1,))


################################################################################
# Diagonal (non-modifying) contribution – with/without sites
################################################################################


@partial(jax.jit, static_argnums=(1, 3))
def local_energy_jax_nonmod(
    state: jnp.ndarray,
    functions_no_sites: Tuple[Callable, ...],
    mult_no_sites: jnp.ndarray,
    functions_sites: Tuple[Callable, ...],
    mult_sites: jnp.ndarray,
) -> jnp.ndarray:
    r"""
    Diagonal local energy from non-modifying operators, with and without sites.

    All functions passed here already have the signature

        f(state) -> (state_like, coeff)
    """

    e1 = local_energy_jax_nonmod_nosites(state, functions_no_sites, mult_no_sites)
    e2 = local_energy_jax_nonmod_nosites(state, functions_sites, mult_sites)
    return e1 + e2


################################################################################
# Main wrapper
################################################################################


def local_energy_jax_wrap(
    ns: int,
    operator_terms_list: List,
    operator_terms_list_ns: List,
    operator_terms_list_nmod: List,
    operator_terms_list_nmod_ns: List,
    n_max: Optional[int] = 1,
    dtype: Optional[jnp.dtype] = jnp.complex128,
) -> Callable:
    r"""
    Build a JIT-compiled local-energy function

        state -> (all_states, all_energies)

    where

        all_states   : (N_total, ns)
        all_energies : (N_total,)

    and

        N_total = 1 (diagonal term) + \sum_i K_i  over all modifying operators,

    with each modifying operator allowed to return **many** states:

        f(state, *sites) -> (new_states, coeffs)
        new_states.shape == (K, ns)
        coeffs.shape     == (K,)

    The returned function is JAX-compatible (pure, JIT-able, vmap-able).
    """

    # Normalise dtype; default to complex64 for speed, but honour user choice.
    if not isinstance(dtype, jnp.dtype):
        dtype = jnp.dtype(dtype)
    if dtype not in (jnp.complex64, jnp.complex128):
        dtype = jnp.complex128

    ###########################################################################
    # 1. Unpack and flatten all operator groups
    ###########################################################################

    f_mod_sites, i_mod_sites, m_mod_sites = unpack_operator_terms(ns, operator_terms_list)
    f_mod_nos, i_mod_nos, m_mod_nos = unpack_operator_terms(ns, operator_terms_list_ns)
    f_nmod_sites, i_nmod_sites, m_nmod_sites = unpack_operator_terms(ns, operator_terms_list_nmod)
    f_nmod_nos, i_nmod_nos, m_nmod_nos = unpack_operator_terms(ns, operator_terms_list_nmod_ns)

    f_mod_sites, i_mod_sites, m_mod_sites = flatten_operator_terms(
        f_mod_sites, i_mod_sites, m_mod_sites
    )
    f_mod_nos, i_mod_nos, m_mod_nos = flatten_operator_terms(f_mod_nos, i_mod_nos, m_mod_nos)
    f_nmod_sites, i_nmod_sites, m_nmod_sites = flatten_operator_terms(
        f_nmod_sites, i_nmod_sites, m_nmod_sites
    )
    f_nmod_nos, i_nmod_nos, m_nmod_nos = flatten_operator_terms(f_nmod_nos, i_nmod_nos, m_nmod_nos)

    # Multipliers live in the chosen complex dtype.
    m_mod_sites = jnp.asarray(m_mod_sites, dtype=dtype)
    m_mod_nos = jnp.asarray(m_mod_nos, dtype=dtype)
    m_nmod_sites = jnp.asarray(m_nmod_sites, dtype=dtype)
    m_nmod_nos = jnp.asarray(m_nmod_nos, dtype=dtype)

    ###########################################################################
    # 2. Wrap operators to enforce shapes
    ###########################################################################

    def wrap_mod(f, sites):
        """Wrap a modifying operator f(state, *sites)."""

        sites = tuple(int(s) for s in sites)

        def op(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            new_states, coeffs = f(state, *sites)
            new_states = jnp.asarray(new_states, dtype=state.dtype)
            coeffs = jnp.asarray(coeffs, dtype=dtype)

            # Enforce (K, ns) for states and (K,) for coeffs.
            if new_states.ndim == 1:
                new_states = new_states.reshape((1, new_states.shape[0]))

            coeffs = coeffs.reshape((-1,))
            return new_states, coeffs

        return op

    def wrap_nos(f):
        """Wrap a modifying operator f(state) without site indices."""

        def op(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            new_states, coeffs = f(state)
            new_states = jnp.asarray(new_states, dtype=state.dtype)
            coeffs = jnp.asarray(coeffs, dtype=dtype)

            if new_states.ndim == 1:
                new_states = new_states.reshape((1, new_states.shape[0]))
            coeffs = coeffs.reshape((-1,))

            return new_states, coeffs

        return op

    # Grouping operators for vectorization
    from collections import defaultdict

    import numpy as np

    def group_operators(funcs, indices, mults):
        """Groups operators by function identity and arity for vectorization."""
        groups = defaultdict(list)
        for f, idx, m in zip(funcs, indices, mults):
            # Group by (function, arity) to ensure uniform index shapes
            groups[(f, len(idx))].append((idx, m))
        return groups

    # -------------------------------------------------------------------------
    # Helper to process groups and handle failures (fallback list)
    # -------------------------------------------------------------------------
    def process_groups(groups):
        data = []
        fallback_items = []
        for (f, _), items in groups.items():
            idxs = [it[0] for it in items]
            ms = [it[1] for it in items]

            if not idxs:
                continue

            try:
                # Attempt to create uniform arrays
                idxs_arr = np.array(idxs, dtype=np.int32)
                ms_arr = (
                    np.array(ms, dtype=dtype) if not isinstance(dtype, jnp.dtype) else np.array(ms)
                )

                # Check for jagged arrays (object dtype) implies failure to vectorize
                if idxs_arr.dtype == object:
                    raise ValueError("Jagged indices array detected")

                data.append((f, jnp.array(idxs_arr), jnp.array(ms_arr, dtype=dtype)))
            except Exception:
                # Add to fallback list if vectorization fails
                for idx, m in zip(idxs, ms):
                    fallback_items.append((f, idx, m))

        return data, fallback_items

    # -------------------------------------------------------------------------
    # Modifying Operators
    # -------------------------------------------------------------------------

    # 1) Modifying WITH sites
    mod_sites_groups = group_operators(f_mod_sites, i_mod_sites, m_mod_sites)
    mod_sites_data, mod_sites_fallback = process_groups(mod_sites_groups)

    # 2) Modifying WITHOUT sites
    mod_nos_groups = group_operators(f_mod_nos, i_mod_nos, m_mod_nos)
    mod_nos_data, mod_nos_fallback = process_groups(mod_nos_groups)

    # -------------------------------------------------------------------------
    # Non-Modifying Operators (Diagonal)
    # -------------------------------------------------------------------------

    # 1) Non-modifying WITH sites
    nmod_sites_groups = group_operators(f_nmod_sites, i_nmod_sites, m_nmod_sites)
    nmod_sites_data, nmod_sites_fallback = process_groups(nmod_sites_groups)

    # 2) Non-modifying WITHOUT sites
    nmod_nos_groups = group_operators(f_nmod_nos, i_nmod_nos, m_nmod_nos)
    nmod_nos_data, nmod_nos_fallback = process_groups(nmod_nos_groups)

    # -------------------------------------------------------------------------
    # Fallback Wrappers
    # -------------------------------------------------------------------------
    # If any items failed vectorization, we need to wrap them individually
    # and pass them to the legacy handling logic or a loop.

    def wrap_nmod_sites(f, sites):
        sites = tuple(int(s) for s in sites)

        def op(state: jnp.ndarray):
            _, coeff = f(state, *sites)
            return state, coeff

        return op

    # Prepare fallback lists for non-modifying (diagonal) terms
    fallback_nmod_sites_funcs = []
    fallback_nmod_sites_mults = []

    for f, idx, m in nmod_sites_fallback:
        fallback_nmod_sites_funcs.append(wrap_nmod_sites(f, idx))
        fallback_nmod_sites_mults.append(m)

    # Also include fallback for NO SITES if any (though usually uniform)
    # Reuse wrap_nmod_sites with empty sites? Or create specific wrapper.
    def wrap_nmod_nos(f):
        def op(state: jnp.ndarray):
            _, coeff = f(state)
            return state, coeff

        return op

    fallback_nmod_nos_funcs = []
    fallback_nmod_nos_mults = []
    for f, _idx, m in nmod_nos_fallback:
        fallback_nmod_nos_funcs.append(wrap_nmod_nos(f))
        fallback_nmod_nos_mults.append(m)

    # We will use vectorized approach by default, but keep wrappers if we wanted to support mixed mode.
    # Currently we fully replace the logic for NMOD inside wrapper.

    # Modifying no-sites (usually global ops like global flip?)
    # Handled above via mod_nos_data.

    # We also need to prepare fallback wrappers for modifying operators
    fallback_mod_sites_wrappers = []
    for f, idx, m in mod_sites_fallback:
        op = wrap_mod(f, idx)
        # op(state) -> (new_states, coeffs)
        fallback_mod_sites_wrappers.append((op, m))

    fallback_mod_nos_wrappers = []
    for f, _idx, m in mod_nos_fallback:
        op = wrap_nos(f)
        fallback_mod_nos_wrappers.append((op, m))

    ###########################################################################
    # 3. Build the inner JAX function
    ###########################################################################

    def init_wrapper():

        # Capture fallback data as tuples
        fallback_nmod_sites_funcs_t = tuple(fallback_nmod_sites_funcs)
        fallback_nmod_sites_mults_arr = (
            jnp.array(fallback_nmod_sites_mults, dtype=dtype)
            if fallback_nmod_sites_mults
            else jnp.array([], dtype=dtype)
        )

        fallback_nmod_nos_funcs_t = tuple(fallback_nmod_nos_funcs)
        fallback_nmod_nos_mults_arr = (
            jnp.array(fallback_nmod_nos_mults, dtype=dtype)
            if fallback_nmod_nos_mults
            else jnp.array([], dtype=dtype)
        )

        # Capture vectorized data
        # We need to pass data into the closure.
        # They contain (f, indices_arr, mults_arr). f is static/hashable.

        @jax.jit
        def wrapper(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            state = jnp.asarray(state)
            ns = state.shape[0]

            # --------------------------------------------------------------
            # 1) Diagonal (non-modifying) contribution (Vectorized)
            # --------------------------------------------------------------

            diag = jnp.zeros((1,), dtype=dtype)

            # Non-modifying WITH sites
            for f, indices, mults in nmod_sites_data:

                # f(state, *sites) -> (state, coeff)
                # indices: (N, Arity)

                def apply_op_nmod(s, idxs, f=f):
                    if idxs.ndim == 0:
                        return f(s, idxs)
                    return f(s, *idxs)

                def get_coeff(idx):
                    _, c = apply_op_nmod(state, idx)
                    return c

                # vmap over indices
                # get_coeff returns (coeff,)
                coeffs = jax.vmap(get_coeff)(indices)

                # Sum contribution: sum(coeff * mult)
                # coeffs shape could be (N,) or (N,1)
                contrib = jnp.sum(coeffs.reshape(-1) * mults.reshape(-1))
                diag = diag + contrib

            # Non-modifying WITHOUT sites
            for f, _, mults in nmod_nos_data:
                # f(state) -> (state, coeff)
                _, coeff = f(state)
                total_mult = jnp.sum(mults)
                diag = diag + (coeff * total_mult)

            # --------------------------------------------------------------
            # 1b) Diagonal Fallback (Legacy)
            # --------------------------------------------------------------
            if len(fallback_nmod_sites_funcs_t) > 0 or len(fallback_nmod_nos_funcs_t) > 0:
                diag_fallback = local_energy_jax_nonmod(
                    state,
                    fallback_nmod_nos_funcs_t,
                    fallback_nmod_nos_mults_arr,
                    fallback_nmod_sites_funcs_t,
                    fallback_nmod_sites_mults_arr,
                )
                diag = diag + diag_fallback.reshape(())

            # Collect outputs in lists; length is static, so XLA can unroll.
            states_list = [state.reshape((1, ns))]
            energies_list = [jnp.asarray(diag, dtype=dtype).reshape((1,))]

            # --------------------------------------------------------------
            # 2) Modifying operators WITH sites (Vectorized)
            # --------------------------------------------------------------

            for f, indices, mults in mod_sites_data:
                # indices shape: (N_terms, Arity) or (N_terms,) if Arity=1
                # mults shape: (N_terms,)

                # DEBUG PRINT (Should see this only during compilation)
                # jax.debug.print("Processing group for {}: indices shape {}", f, indices.shape)

                # Helper to apply f to state and sites
                def apply_op_mod(s, idxs, f=f):
                    # Unpack indices if it's an array
                    # If Arity=1, idxs might be scalar or 0-d array
                    if idxs.ndim == 0:
                        new_states, coeffs = f(s, idxs)
                    else:
                        new_states, coeffs = f(s, *idxs)

                    # Ensure shapes are (K, ns) and (K,) to support broadcasting in vmap
                    new_states = jnp.asarray(new_states, dtype=state.dtype)
                    coeffs = jnp.asarray(coeffs, dtype=dtype)

                    if new_states.ndim == 1:
                        new_states = new_states.reshape((1, new_states.shape[0]))
                    coeffs = jnp.atleast_1d(coeffs)

                    return new_states, coeffs

                # vmap over indices (axis 0)
                # state is broadcasted (None)
                # apply_op_mod returns (new_states, coeffs)
                # new_states: (K_branch, ns), coeffs: (K_branch,)

                vmapped_f = jax.vmap(apply_op_mod, in_axes=(None, 0))

                # Result shapes:
                # new_states_batch: (N_terms, K_branch, ns)
                # coeffs_batch:     (N_terms, K_branch)
                new_states_batch, coeffs_batch = vmapped_f(state, indices)

                # Apply multipliers
                # mults: (N_terms,) -> broadcast to (N_terms, K_branch)
                energies_batch = coeffs_batch * mults[:, None]

                # Flatten batch dimension
                # (N_terms * K_branch, ns)
                states_list.append(new_states_batch.reshape(-1, ns))
                energies_list.append(energies_batch.reshape(-1))

            # --------------------------------------------------------------
            # 3) Modifying operators WITHOUT sites (Vectorized)
            # --------------------------------------------------------------
            for f, _, mults in mod_nos_data:
                # f(state) -> (new_states, coeffs)
                # new_states: (K_branch, ns)
                # coeffs: (K_branch,)

                # Since no sites vary, we just call f once and scale by sum of multipliers?
                # Or do we apply f N times?
                # If f doesn't depend on sites, f(state) is constant for all terms in this group.
                # The only difference is the multiplier.
                # So we can just sum the multipliers and apply f once!

                total_mult = jnp.sum(mults)
                new_states, coeffs = f(state)

                # Ensure shapes
                if new_states.ndim == 1:
                    new_states = new_states.reshape((1, ns))
                coeffs = jnp.atleast_1d(coeffs)

                states_list.append(new_states)
                energies_list.append(coeffs * total_mult)

            # --------------------------------------------------------------
            # 3b) Modifying Fallback
            # --------------------------------------------------------------
            for op, m in fallback_mod_sites_wrappers:
                new_states, coeffs = op(state)
                states_list.append(new_states)
                energies_list.append(coeffs * m)

            for op, m in fallback_mod_nos_wrappers:
                new_states, coeffs = op(state)
                states_list.append(new_states)
                energies_list.append(coeffs * m)

            # --------------------------------------------------------------
            # 4) Stack everything
            # --------------------------------------------------------------
            all_states = jnp.concatenate(states_list, axis=0)
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
# End of file
