"""
JAX-based update rules for Spin-1/2 systems.

----------------------------------------------------------------------
Author          : Maks Kliczkowski
Date            : December 2025
Description     : This module provides functions to propose various types of updates
                (local flips, exchanges, multi-flips, global pattern flips) for
                spin-1/2 systems using JAX for efficient computation.
----------------------------------------------------------------------
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

try:
    from QES.general_python.common.binary import BACKEND_DEF_SPIN, BACKEND_REPR, jaxpy
except ImportError:
    BACKEND_DEF_SPIN    = True
    BACKEND_REPR        = 1.0

    class jaxpy:
        @staticmethod
        def flip_array_jax_spin(state, idx):
            # val = state[idx]
            # return state.at[idx].set(-val)
            return state.at[idx].multiply(-1)


# ----------------------------------------------------------------------
# Local Updates
# ----------------------------------------------------------------------


@jax.jit
def _propose_local_flip_kernel(
    state: jnp.ndarray, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kernel for local flip. Returns new state and indices of flipped sites.
    """
    idx = jr.randint(key, shape=(), minval=0, maxval=state.size)
    if BACKEND_DEF_SPIN:
        new_state = jaxpy.flip_array_jax_spin(state, idx)
    else:
        new_state = jaxpy.flip_array_jax_nspin(state, idx, spin_value=BACKEND_REPR)
    return new_state, idx


@jax.jit
def propose_local_flip(state: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    Propose a single random spin flip.
    Returns:
        New state with one spin flipped.
    """
    new_state, _ = _propose_local_flip_kernel(state, key)
    return new_state


@jax.jit
def propose_local_flip_with_info(state: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Propose a single random spin flip.
    Returns:
        (new_state, flipped_index) where flipped_index has shape (1,)
    """
    new_state, idx = _propose_local_flip_kernel(state, key)
    return new_state, jnp.expand_dims(idx, 0)

# ----------------------------------------------------------------------
# Exchange Updates
# ----------------------------------------------------------------------

@partial(jax.jit, static_argnames=("check_conserved",))
def _propose_exchange_kernel(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray, check_conserved: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kernel for exchange. Returns new state and tuple of flipped indices.
    """
    key_site, key_neigh = jr.split(key)
    ns = state.shape[0]

    # Pick a random site i
    i = jr.randint(key_site, shape=(), minval=0, maxval=ns)

    # Pick a random neighbor j from table[i]
    max_degree = neighbor_table.shape[1]
    k = jr.randint(key_neigh, shape=(), minval=0, maxval=max_degree)
    j = neighbor_table[i, k]

    # Swap s_i and s_j if j != -1
    is_valid = j != -1

    # Safe gather: if invalid, read from i (no-op)
    j_safe = jnp.where(is_valid, j, i)

    si = state[i]
    sj = state[j_safe]

    should_swap = is_valid
    if check_conserved:
        should_swap = should_swap & (si != sj)

    # Optimized swap using scatter
    idx_pair = jnp.stack([i, j_safe])  # (2,)
    # Values to put: if swap, put [sj, si], else [si, sj]
    vals = jnp.where(should_swap, jnp.stack([sj, si]), jnp.stack([si, sj]))
    new_state = state.at[idx_pair].set(vals)

    return new_state, idx_pair

@partial(jax.jit, static_argnames=("check_conserved",))
def propose_exchange(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray, check_conserved: bool = True
) -> jnp.ndarray:
    """
    Propose an exchange between a random site and a random neighbor.
    Returns:
        New state with spins exchanged (or same state if invalid move).
    """
    new_state, _ = _propose_exchange_kernel(state, key, neighbor_table, check_conserved)
    return new_state

@partial(jax.jit, static_argnames=("check_conserved",))
def propose_exchange_with_info(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray, check_conserved: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Propose an exchange.
    Returns:
        (new_state, indices) where indices has shape (2,)
    """
    new_state, idx_pair = _propose_exchange_kernel(state, key, neighbor_table, check_conserved)
    return new_state, idx_pair

# ----------------------------------------------------------------------
# Bond Flip Updates (Flip site + random neighbor)
# ----------------------------------------------------------------------

@jax.jit
def _propose_bond_flip_kernel(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_site, key_neigh = jr.split(key)
    ns = state.shape[0]

    # Pick random site i
    i = jr.randint(key_site, shape=(), minval=0, maxval=ns)

    # Pick random neighbor j
    max_degree = neighbor_table.shape[1]
    k = jr.randint(key_neigh, shape=(), minval=0, maxval=max_degree)
    j = neighbor_table[i, k]

    # Check validity
    is_valid = j != -1
    j_safe = jnp.where(is_valid, j, i)  # if invalid, flip i twice (no-op)

    # Perform Flip
    if BACKEND_DEF_SPIN:
        # Spin +/- 1: Flip sign
        val_i = -state[i]
        val_j = -state[j_safe]
    else:
        # Binary 0/v representation
        flip_value  = jnp.asarray(BACKEND_REPR, dtype=state.dtype)
        val_i       = flip_value - state[i]
        val_j       = flip_value - state[j_safe]

    # JAX way: optimized scatter
    idx_pair = jnp.stack([i, j_safe])

    vals_new = jnp.where(
        is_valid, jnp.stack([val_i, val_j]), jnp.stack([state[i], state[j_safe]])
    )  # If not valid, keep same

    new_state = state.at[idx_pair].set(vals_new)

    return new_state, idx_pair


@jax.jit
def propose_bond_flip(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray
) -> jnp.ndarray:
    new_state, _ = _propose_bond_flip_kernel(state, key, neighbor_table)
    return new_state


@jax.jit
def propose_bond_flip_with_info(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _propose_bond_flip_kernel(state, key, neighbor_table)


# ----------------------------------------------------------------------
# Multi-Flip Updates (N-random sites)
# ----------------------------------------------------------------------


@partial(jax.jit, static_argnames=("n_flip",))
def _propose_multi_flip_kernel(
    state: jnp.ndarray, key: jax.Array, n_flip: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if n_flip == 1:
        # Re-use local flip kernel logic to avoid choice overhead
        new_state, idx = _propose_local_flip_kernel(state, key)
        return new_state, jnp.expand_dims(idx, 0)

    # Choose n_flip distinct indices
    indices = jr.choice(key, state.size, shape=(n_flip,), replace=False)

    if BACKEND_DEF_SPIN:
        new_state = state.at[indices].multiply(-1)
    else:
        # 0 -> v, v -> 0 => v - x
        val         = state[indices]
        flip_value  = jnp.asarray(BACKEND_REPR, dtype=state.dtype)
        new_state   = state.at[indices].set(flip_value - val)

    return new_state, indices


@partial(jax.jit, static_argnames=("n_flip",))
def propose_multi_flip(state: jnp.ndarray, key: jax.Array, n_flip: int = 1) -> jnp.ndarray:
    new_state, _ = _propose_multi_flip_kernel(state, key, n_flip)
    return new_state


@partial(jax.jit, static_argnames=("n_flip",))
def propose_multi_flip_with_info(
    state: jnp.ndarray, key: jax.Array, n_flip: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _propose_multi_flip_kernel(state, key, n_flip)


# ----------------------------------------------------------------------
# Global Updates (Pattern Flipping)
# ----------------------------------------------------------------------


@jax.jit
def _propose_global_flip_kernel(
    state: jnp.ndarray, key: jax.Array, patterns: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kernel for global flip.
    Returns: (new_state, safe_indices)
    Note: Multipliers are implied by the update type (flip).
    For delta_log_psi, we need indices.
    """
    # 1. Select random pattern
    n_patterns      = patterns.shape[0]
    p_idx           = jr.randint(key, shape=(), minval=0, maxval=n_patterns)
    target_indices  = patterns[p_idx]  # (PatternSize,)

    # 2. Mask valid indices (ignore -1 padding)
    mask            = target_indices != -1

    # 3. Safe indices: map -1 to 0
    safe_indices    = jnp.where(mask, target_indices, 0)

    # 4. Apply Updates
    if BACKEND_DEF_SPIN:
        # Optimized: Scatter Multiply
        multipliers = jnp.where(mask, -1, 1)
        # Apply multipliers. Index 0 might be hit multiple times if it's padding,
        # but 1*1*...*1 = 1, so no change.
        new_state   = state.at[safe_indices].multiply(multipliers)
    else:
        # Binary 0/1
        flip_counts = jnp.zeros_like(state, dtype=jnp.int32)
        adders      = jnp.where(mask, jnp.int32(1), jnp.int32(0))
        flip_counts = flip_counts.at[safe_indices].add(adders)
        should_flip = (flip_counts % 2) == 1
        flip_value  = jnp.asarray(BACKEND_REPR, dtype=state.dtype)
        new_state   = jnp.where(should_flip, flip_value - state, state)

    return new_state, safe_indices


@jax.jit
def propose_global_flip(state: jnp.ndarray, key: jax.Array, patterns: jnp.ndarray) -> jnp.ndarray:
    new_state, _ = _propose_global_flip_kernel(state, key, patterns)
    return new_state


@jax.jit
def propose_global_flip_with_info(
    state: jnp.ndarray, key: jax.Array, patterns: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _propose_global_flip_kernel(state, key, patterns)


# ----------------------------------------------------------------------
# Worm Updates (Snake / Random String)
# ----------------------------------------------------------------------


@partial(jax.jit, static_argnames=("length",))
def _propose_worm_flip_kernel(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray, length: int = 4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ns = state.shape[0]
    max_degree = neighbor_table.shape[1]

    # Split keys: one for start, one for walk
    key_start, key_walk = jr.split(key)

    # 1. Pick start site
    start_node = jr.randint(key_start, shape=(), minval=0, maxval=ns)

    indices = jnp.full((length + 1,), start_node, dtype=jnp.int32)

    # 3. Define Random Walk Step
    def step_fn(i, carry):
        curr_node, key_iter, idx_arr = carry
        key_step, key_next = jr.split(key_iter)

        # Pick random neighbor
        k           = jr.randint(key_step, shape=(), minval=0, maxval=max_degree)
        next_node   = neighbor_table[curr_node, k]

        # Check validity (neighbor_table padded with -1)
        valid       = next_node != -1

        # If valid, move there. If not, stay put (flip same site again -> identity)
        safe_node   = jnp.where(valid, next_node, curr_node)

        # Update indices array at position i + 1 (since 0 is start)
        idx_arr     = idx_arr.at[i + 1].set(safe_node)

        return (safe_node, key_next, idx_arr)

    # 4. Execute Random Walk
    carry_init          = (start_node, key_walk, indices)
    final_carry         = jax.lax.fori_loop(0, length, step_fn, carry_init)
    _, _, final_indices = final_carry

    # 5. Apply Flips
    flip_counts     = jnp.zeros_like(state, dtype=jnp.int32)
    flip_counts     = flip_counts.at[final_indices].add(jnp.int32(1))

    should_flip     = (flip_counts % 2) == 1

    if BACKEND_DEF_SPIN:
        mult        = jnp.where(should_flip, -1, 1)
        new_state   = state * mult
    else:
        flip_value  = jnp.asarray(BACKEND_REPR, dtype=state.dtype)
        new_state   = jnp.where(should_flip, flip_value - state, state)

    return new_state, final_indices


@partial(jax.jit, static_argnames=("length",))
def propose_worm_flip(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray, length: int = 4
) -> jnp.ndarray:
    new_state, _ = _propose_worm_flip_kernel(state, key, neighbor_table, length)
    return new_state


@partial(jax.jit, static_argnames=("length",))
def propose_worm_flip_with_info(
    state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray, length: int = 4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _propose_worm_flip_kernel(state, key, neighbor_table, length)


# ----------------------------------------
#! EOF
# ----------------------------------------
