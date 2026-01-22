"""
Utility functions for Monte Carlo updates.

----------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : December 2025
Description     : This module provides utility functions for constructing neighbor tables
                from lattice objects and creating hybrid proposers that combine local and global updates
                for spin systems using JAX.
----------------------------------------------------------------------
"""

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice import Lattice

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    JAX_AVAILABLE = True
except ImportError:
    jnp = np
    jax = None
    jr = None
    JAX_AVAILABLE = False

# ----------------------------------------
#! Neighbor table construction
# ----------------------------------------


def get_neighbor_table(lattice: "Lattice", max_neighbors=None, order: int = 1) -> jnp.ndarray:
    """
    Constructs a padded neighbor table from a Lattice object.

    Parameters:
    -----------

    lattice:
        QES Lattice object with adjacency_matrix() method.
    max_neighbors:
        Optional override for maximum degree.
    order:
        Maximum distance (graph hops) to consider as neighbors.
        Default is 1 (nearest neighbors).


    Returns:
    --------
    jnp.ndarray: (Ns, max_degree) array where entry [i, k] is the index
                of the k-th neighbor of site i. Padded with -1.
    """

    # Get adjacency matrix (prefer dense for easier processing if small, but sparse is safer for large)
    # lattice.adjacency_matrix() returns 1 for neighbors.
    try:
        adj = lattice.adjacency_matrix(sparse=False)
    except Exception:
        adj = lattice.adjacency_matrix(sparse=True).toarray()

    # If order > 1, compute powers of adjacency matrix
    if order > 1:
        # We want to find all nodes reachable within 'order' hops.
        # Compute A + A^2 + ... + A^order
        # Use boolean arithmetic to avoid overflow and just check connectivity
        adj_bool = adj != 0
        accum_adj = adj_bool.copy()
        current_pow = adj_bool.copy()

        for _ in range(order - 1):
            current_pow = current_pow @ adj_bool  # Matrix multiplication for next hop
            accum_adj = accum_adj | current_pow  # Accumulate connectivity

        # Remove self-loops (diagonal)
        np.fill_diagonal(accum_adj, False)
        adj = accum_adj.astype(int)

    # List of lists for neighbors
    ns = adj.shape[0]
    neighbors_list = []
    max_degree = 0

    for i in range(ns):
        # Indices where A_ij != 0
        # If weighted, we might care about weights, but for exchange we usually just want connectivity.
        neighs = np.flatnonzero(adj[i])
        max_degree = max(max_degree, len(neighs))
        neighbors_list.append(neighs)

    if max_neighbors is not None:
        if max_degree > max_neighbors:
            raise ValueError(
                f"Lattice max degree {max_degree} exceeds requested max_neighbors {max_neighbors}"
            )
        max_degree = max_neighbors

    # Construct padded array
    neighbor_table = np.full((ns, max_degree), -1, dtype=np.int32)

    for i, neighs in enumerate(neighbors_list):
        k = len(neighs)
        neighbor_table[i, :k] = neighs

    return jnp.array(neighbor_table)


# ----------------------------------------
#! Hybrid Proposer
# ----------------------------------------


def make_hybrid_proposer(local_prop: Callable, global_prop: Callable, p_global: float):
    """
    Creates a hybrid proposer that chooses between local and global updates.

    Parameters:
    -----------
    local_prop:
        Function(state, key) -> new_state
    global_prop:
        Function(state, key) -> new_state
    p_global:
        Probability of choosing global_prop. The local_prop is chosen with probability (1 - p_global).
        Values should be in [0, 1].

    Returns:
        Function(state, key) -> new_state
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for hybrid proposer.")

    def hybrid_proposer(state, key):
        key_p, key_upd = jr.split(key)
        do_global = jr.bernoulli(key_p, p_global)

        # Use lax.cond to execute only one branch
        return jax.lax.cond(
            do_global, lambda k: global_prop(state, k), lambda k: local_prop(state, k), key_upd
        )

    return hybrid_proposer


# ----------------------------------------
#! EOF
# ----------------------------------------
