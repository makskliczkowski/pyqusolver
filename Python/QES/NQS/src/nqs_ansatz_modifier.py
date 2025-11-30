"""
NQS Ansatz Modifier Module
--------------------------
File            : NQS/src/nqs_ansatz_modifier.py
Description     : This module provides classes and functions to modify neural quantum states (NQS) ansatze.
                  It includes functionality to apply transformations, augmentations, and other modifications
                  to the ansatze used in quantum simulations.
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
--------------------------
"""

import numpy as np
import warnings
from typing import Callable, Any, Tuple, Optional, Union

try:
    import  jax
    import  jax.numpy as jnp
    from    jax.scipy.special   import logsumexp
    from    jax.tree_util       import register_pytree_node_class
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Dummy decorator to prevent import errors if JAX is missing
    def register_pytree_node_class(cls): return cls
    
# ----------------------------------------------------------

@register_pytree_node_class
class AnsatzModifier:
    """
    A JAX-native wrapper that applies a linear operator O to the ansatz.
    Formula: log <s|O|Ψ> = log ( Σ_s' <s|O|s'> * exp(log <s'|Ψ>) )
    
    Robustness:
    - Registered as a PyTree (passable to JIT).
    - Auto-detects branching factor (Diagonal vs General).
    - Handles complex logarithms safely.
    """

    def __init__(self, 
                net_apply       : Callable, 
                operator        : Any, 
                input_shape     : tuple,
                dtype           : Any   = None,
                statetype       : str   = 'float32',
                **kwargs):
        
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for AnsatzModifier.")

        self.net_apply      = net_apply
        self.input_shape    = input_shape
        self.dtype          = dtype if dtype else jnp.complex128
        self.statetype      = jnp.dtype(statetype)
        
        # Standardize the Operator Function
        # We expect op_func(state) -> (connected_states, weights)
        if hasattr(operator, 'jax'):
            self.op_func    = operator.jax
        elif callable(operator):
            self.op_func    = operator
        else:
            raise ValueError(f"Operator {operator} must be callable or implement .jax()")

        # Inspect Branching Factor (M)
        # This determines if we take the fast path (M=1) or stable path (M>1)
        self.branching_factor = self._inspect_branching()
        
        # Validation
        if self.branching_factor < 1:
            raise ValueError("Operator must return at least one connected state.")

    # ----------------------------------------------------------------------
    # Internal Utilities
    # ----------------------------------------------------------------------

    def _inspect_branching(self) -> int:
        """
        Safely determines the branching factor by tracing a dummy input.
        Returns M (int).
        """
        try:
            # Create a dummy state on CPU to avoid allocating GPU memory for metadata checks
            dummy           = jnp.ones((1, *self.input_shape), dtype=self.statetype)
            
            # We use eval_shape to trace without concrete computation (fast & safe)
            # This handles the case where the operator might crash on zeros
            abstract_out    = jax.eval_shape(self.op_func, dummy)
            
            # Output structure: (states, weights)
            # states shape: (Batch, M, ...)
            states_shape    = abstract_out[0].shape
            return states_shape[1] 
            
        except Exception as e:
            warnings.warn(f"Could not inspect branching factor automatically ({e}). Defaulting to General/Sparse mode (M>1).")
            return 2

    # ----------------------------------------------------------------------

    def __call__(self, params: Any, x: Any) -> Any:
        """
        Evaluates the modified ansatz.
        Args:
            params: Network parameters (PyTree)
            x: Input states (Batch, ...)
        Returns:
            log_amplitudes: (Batch,)
        """
        if self.branching_factor == 1:
            return self._forward_diagonal(params, x)
        else:
            return self._forward_general(params, x)

    # ----------------------------------------------------------------------
    # Execution Strategies
    # ----------------------------------------------------------------------

    def _forward_diagonal(self, params, x):
        """
        Fast path for Diagonal/Permutation operators (M=1).
        No logsumexp needed.
        """
        
        # Apply Operator
        # st: (Batch, 1, ...), w: (Batch, 1)
        st, w   = self.op_func(x)
        
        # Squeeze singleton dimension
        st      = jnp.squeeze(st, axis=1)
        w       = jnp.squeeze(w, axis=1)
        
        # Evaluate Base Ansatz
        log_psi = self.net_apply(params, st)
        
        # Combine: log(w * psi) = log(w) + log_psi
        # Ensure complex type for log to handle negative weights
        log_w   = jnp.log(w.astype(jnp.complex128))
        
        return log_psi + log_w

    def _forward_general(self, params, x):
        """
        Stable path for General Sparse operators (M>1).
        Uses LogSumExp.
        """
        batch_size = x.shape[0]
        
        # Apply Operator (Vectorized)
        # st: (Batch, M, ...), w: (Batch, M)
        # Note: If op_func is not natively vectorized, we might need jax.vmap(self.op_func)(x)
        # But usually high-performance operators in QES handle batching.
        st, w               = self.op_func(x)
        M                   = self.branching_factor
        
        # Flatten Batch and Branching dimensions
        # We need to feed (Batch * M, ...) into the network
        # Collapse dimensions 0 and 1
        flat_shape          = (batch_size * M, *st.shape[2:])
        st_flat             = jnp.reshape(st, flat_shape)
        
        # Evaluate Base Ansatz
        log_psi_flat        = self.net_apply(params, st_flat)
        
        # Reshape back to (Batch, M)
        log_psi_connected   = jnp.reshape(log_psi_flat, (batch_size, M))
        
        # Compute LogSumExp
        # formula: log( sum_k w_k * exp(log_psi_k) )
        #        = log( sum_k exp( log(w_k) + log_psi_k ) )
        
        # Handle zero weights safely (padding)
        # jnp.log(0) is -inf, which logsumexp handles correctly (term vanishes)
        log_w               = jnp.log(w.astype(jnp.complex128))
        terms               = log_w + log_psi_connected
        
        return logsumexp(terms, axis=1)

    # ----------------------------------------------------------------------
    # PyTree Registration (The Magic Sauce for JIT)
    # ----------------------------------------------------------------------

    def tree_flatten(self):
        """
        Separates dynamic data (children) from static configuration (aux_data).
        
        Children: None (Unless op_func or net_apply are PyTrees themselves, 
                  but usually they are closures or static functions. 
                  If 'params' were stored here, they would be children.)
        Aux Data: Everything needed to reconstruct the class.
        """
        children = ()
        aux_data = (self.net_apply, self.op_func, self.input_shape, self.dtype, self.branching_factor)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs the object inside JIT."""
        net_apply, op_func, input_shape, dtype, branching_factor = aux_data
        
        # Create instance without calling __init__ to avoid re-running introspection
        obj                     = cls.__new__(cls)
        obj.net_apply           = net_apply
        obj.op_func             = op_func
        obj.input_shape         = input_shape
        obj.dtype               = dtype
        obj.branching_factor    = branching_factor
        return obj
    
# ----------------------------------------------------------
#! EOF
# ----------------------------------------------------------