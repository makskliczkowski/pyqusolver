"""
Initialization for update rules module.

----------------------------------------------------------------------
Author          : Maks Kliczkowski
Date            : December 2025
Description     : This module initializes the update rules for spin systems.
----------------------------------------------------------------------
"""
from .spin_jax      import propose_local_flip, propose_exchange, propose_global_flip, propose_multi_flip, propose_bond_flip
from .spin_numpy    import propose_local_flip_np, propose_multi_flip_np
from .utils         import get_neighbor_table, make_hybrid_proposer

# ----------------------------------------
#! EOF
# ----------------------------------------
