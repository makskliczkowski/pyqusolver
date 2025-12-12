
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

import  jax
import  jax.numpy   as jnp
import  jax.random  as jr
from    functools   import partial
from    typing      import Optional, Tuple

try:
    from QES.general_python.common.binary import jaxpy, BACKEND_DEF_SPIN
except ImportError:
    raise ImportError("QES.general_python.common.binary.jaxpy module is required for spin updates.")

# ----------------------------------------------------------------------
# Local Updates
# ----------------------------------------------------------------------

@jax.jit
def propose_local_flip(state: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    Propose a single random spin flip.
    
    Args:
        state: Current state (Ns,).
        key: JAX PRNGKey.
        
    Returns:
        New state with one spin flipped.
    """
    # Select random site
    idx = jr.randint(key, shape=(), minval=0, maxval=state.size)
    
    # Note: jaxpy.flip_array_jax_spin is for +/- 1 spins
    if BACKEND_DEF_SPIN:
        return jaxpy.flip_array_jax_spin(state, idx)
    else:
        # Fallback for 0/1 or if nspin function has different name
        # 0 -> 1, 1 -> 0 implies 1 - x
        val = state[idx]
        return state.at[idx].set(1 - val)

# ----------------------------------------------------------------------
# Exchange Updates
# ----------------------------------------------------------------------

@partial(jax.jit, static_argnames=("check_conserved",))
def propose_exchange(state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray, check_conserved: bool = True) -> jnp.ndarray:
    """
    Propose an exchange of spins between a random site and a random neighbor.
    
    Parameters:
        state: 
            Current state (Ns,).
        key: 
            JAX PRNGKey.
        neighbor_table: 
            (Ns, max_degree) array of neighbor indices (padded with -1).
        check_conserved: 
            If True, ensures the move actually changes the state.
    
    Returns:
        New state with spins exchanged (or same state if invalid move).
    """
    key_site, key_neigh     = jr.split(key)
    ns                      = state.shape[0]
    
    # Pick a random site i
    i                       = jr.randint(key_site, shape=(), minval=0, maxval=ns)
    
    # Pick a random neighbor j from table[i]
    max_degree              = neighbor_table.shape[1]
    k                       = jr.randint(key_neigh, shape=(), minval=0, maxval=max_degree)
    j                       = neighbor_table[i, k]
    
    # Swap s_i and s_j if j != -1
    is_valid                = (j != -1)
    
    # Safe gather: if invalid, read from i (no-op)
    j_safe                  = jnp.where(is_valid, j, i)
    
    si                      = state[i]
    sj                      = state[j_safe] 
    
    should_swap             = is_valid
    if check_conserved:
        should_swap = should_swap & (si != sj)
        
    # Perform swap
    # New state has sj at pos i, si at pos j_safe
    
    # We can use jnp.where on the whole state, or conditioned updates.
    # Conditional updates are usually efficient in JAX if sparse.
    new_state = state.at[i].set(jnp.where(should_swap, sj, si))
    new_state = new_state.at[j_safe].set(jnp.where(should_swap, si, sj))
    
    return new_state

# ----------------------------------------------------------------------
# Bond Flip Updates (Flip site + random neighbor)
# ----------------------------------------------------------------------

@jax.jit
def propose_bond_flip(state: jnp.ndarray, key: jax.Array, neighbor_table: jnp.ndarray) -> jnp.ndarray:
    """
    Propose flipping a random site AND one of its random neighbors.
    Useful for topological models where single flips create excitations 
    but bond flips might just move them or preserve sectors.
    
    Args:
        state: Current state (Ns,).
        key: JAX PRNGKey.
        neighbor_table: Neighbor table.
    
    Returns:
        New state with two spins flipped.
    """
    key_site, key_neigh     = jr.split(key)
    ns                      = state.shape[0]
    
    # 1. Pick random site i
    i                       = jr.randint(key_site, shape=(), minval=0, maxval=ns)
    
    # 2. Pick random neighbor j
    max_degree              = neighbor_table.shape[1]
    k                       = jr.randint(key_neigh, shape=(), minval=0, maxval=max_degree)
    j                       = neighbor_table[i, k]
    
    # Check validity
    is_valid                = (j != -1)
    j_safe                  = jnp.where(is_valid, j, i) # if invalid, flip i twice (no-op)
    
    # Perform Flip
    if BACKEND_DEF_SPIN:
        # Spin +/- 1: Flip sign
        val_i = -state[i]
        val_j = -state[j_safe]
    else:
        # Binary 0/1: 1 - x
        val_i = 1 - state[i]
        val_j = 1 - state[j_safe]
    
    # Safe update:
    # new_state = state
    # if is_valid:
    #    state[i] = val_i
    #    state[j] = val_j
    
    # JAX way:
    new_state = state.at[i].set(jnp.where(is_valid, val_i, state[i]))
    new_state = new_state.at[j_safe].set(jnp.where(is_valid, val_j, state[j_safe]))
    
    return new_state

# ----------------------------------------------------------------------
# Multi-Flip Updates (N-random sites)
# ----------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_flip",))
def propose_multi_flip(state: jnp.ndarray, key: jax.Array, n_flip: int = 1) -> jnp.ndarray:
    """
    Propose flipping 'n_flip' distinct random sites.
    
    Args:
        state: 
            Current state (Ns,).
        key: 
            JAX PRNGKey.
        n_flip: 
            Number of sites to flip.
        
    Returns:
        New state.
    """
    if n_flip == 1:
        return propose_local_flip(state, key)
        
    # Choose n_flip distinct indices
    # jr.choice with replace=False is suitable
    indices = jr.choice(key, state.size, shape=(n_flip,), replace=False)
    
    if BACKEND_DEF_SPIN:
        # Vectorized flip
        # new_vals = -state[indices]
        # return state.at[indices].set(new_vals)
        
        # Or multiply
        return state.at[indices].multiply(-1)
    else:
        # 0 -> 1, 1 -> 0 => 1 - x
        val = state[indices]
        return state.at[indices].set(1 - val)

# ----------------------------------------------------------------------
# Global Updates (Pattern Flipping)
# ----------------------------------------------------------------------

@jax.jit
def propose_global_flip(state: jnp.ndarray, key: jax.Array, patterns: jnp.ndarray) -> jnp.ndarray:
    """
    Propose a global update by flipping a predefined pattern of spins.
    
    Example:
    >>> patterns    = lat.calculate_plaquettess()  # (NumPatterns, PatternSize)
    >>> new_state   = propose_global_flip(state, key, patterns)
    
    Args:
        state: 
            Current state (Ns,).
        key: 
            JAX PRNGKey.
        patterns: 
            (NumPatterns, PatternSize) array of site indices to flip.
            Padded with -1.
    
    Returns:
        New state with pattern flipped.
    """
    # 1. Select random pattern
    n_patterns      = patterns.shape[0]
    p_idx           = jr.randint(key, shape=(), minval=0, maxval=n_patterns)
    target_indices  = patterns[p_idx] # (PatternSize,)
    
    # 2. Mask valid indices (ignore -1 padding)
    mask            = (target_indices != -1)
    
    # 3. Safe indices: map -1 to 0 to avoid out-of-bounds access
    # Note: Updates to site 0 from padding will be identity operations
    safe_indices    = jnp.where(mask, target_indices, 0)
    
    # 4. Apply Updates
    if BACKEND_DEF_SPIN:
        # Spin +/- 1: Flip corresponds to multiplication by -1
        # Construct a flip mask of 1s (no flip)
        flip_mult   = jnp.ones_like(state)
        
        # Prepare multipliers: -1 for valid targets, 1 for padding
        multipliers = jnp.where(mask, -1, 1)
        
        # Apply multipliers to the mask
        # If site 0 is both a target and padding, it gets (-1) * (1) = -1. Correct.
        flip_mult   = flip_mult.at[safe_indices].multiply(multipliers)
        
        return state * flip_mult
        
    else:
        # Binary 0/1: Flip corresponds to adding 1 (mod 2)
        # Construct a flip count mask of 0s
        flip_counts = jnp.zeros_like(state, dtype=jnp.int32)
        
        # Prepare adders: 1 for valid targets, 0 for padding
        adders      = jnp.where(mask, 1, 0)
        
        # Accumulate flips
        # If site 0 is both target and padding, it gets 1 + 0 = 1. Correct.
        flip_counts = flip_counts.at[safe_indices].add(adders)
        
        # Determine sites to flip (odd number of flips)
        should_flip = (flip_counts % 2) == 1
        
        # Apply flip: 1 - x
        # If state is float, 1 - x works for 0.0/1.0
        return jnp.where(should_flip, 1 - state, state)

# ----------------------------------------
#! EOF
# ----------------------------------------