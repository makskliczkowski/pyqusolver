"""
Utility functions for Monte Carlo updates.
"""
from    typing import Optional, TYPE_CHECKING, Callable
import  numpy as np

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice import Lattice

try:
    import          jax
    import          jax.numpy as jnp
    import          jax.random as jr
    JAX_AVAILABLE   = True
except ImportError:
    jnp             = np
    jax             = None
    jr              = None
    JAX_AVAILABLE   = False

# ----------------------------------------
#! Neighbor table construction
# ----------------------------------------

def get_neighbor_table(lattice: 'Lattice', max_neighbors=None) -> jnp.ndarray:
    """
    Constructs a padded neighbor table from a Lattice object.
    
    Args:
        lattice: 
            QES Lattice object with adjacency_matrix() method.
        max_neighbors: 
            Optional override for maximum degree.
        
    Returns:
        jnp.ndarray: (Ns, max_degree) array where entry [i, k] is the index
                    of the k-th neighbor of site i. Padded with -1.
    """
    
    # Get adjacency matrix (prefer dense for easier processing if small, but sparse is safer for large)
    # lattice.adjacency_matrix() returns 1 for neighbors.
    try:
        adj = lattice.adjacency_matrix(sparse=False)
    except Exception:
        adj = lattice.adjacency_matrix(sparse=True).toarray()
        
    # List of lists for neighbors
    ns              = adj.shape[0]
    neighbors_list  = []
    max_degree      = 0
    
    for i in range(ns):
        # Indices where A_ij != 0
        # If weighted, we might care about weights, but for exchange we usually just want connectivity.
        neighs      = np.flatnonzero(adj[i])
        max_degree  = max(max_degree, len(neighs))
        neighbors_list.append(neighs)
        
    if max_neighbors is not None:
        if max_degree > max_neighbors:
            raise ValueError(f"Lattice max degree {max_degree} exceeds requested max_neighbors {max_neighbors}")
        max_degree = max_neighbors
        
    # Construct padded array
    neighbor_table = np.full((ns, max_degree), -1, dtype=np.int32)
    
    for i, neighs in enumerate(neighbors_list):
        k                       = len(neighs)
        neighbor_table[i, :k]   = neighs
        
    return jnp.array(neighbor_table)

# ----------------------------------------
#! Hybrid Proposer
# ----------------------------------------

def make_hybrid_proposer(local_prop: Callable, global_prop: Callable, p_global: float):
    """
    Creates a hybrid proposer that chooses between local and global updates.
    
    Args:
        local_prop: Function(state, key) -> new_state
        global_prop: Function(state, key) -> new_state
        p_global: Probability of choosing global_prop
        
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
            do_global,
            lambda k: global_prop(state, k),
            lambda k: local_prop(state, k),
            key_upd
        )
    return hybrid_proposer

# ----------------------------------------
#! EOF
# ----------------------------------------