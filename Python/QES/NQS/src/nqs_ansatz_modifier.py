"""
NQS Ansatz Modifier Module
--------------------------
File            : NQS/src/nqs_ansatz_modifier.py
Description     : This module provides classes and functions to modify neural quantum states (NQS) ansatze.
                  It includes functionality to apply transformations, augmentations, and other modifications
                  to the ansatze used in quantum simulations.
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Version         : 2.0.0
Last Modified   : 08.03.2026
--------------------------
"""

import warnings
from typing import Any, Callable

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    from jax.tree_util import register_pytree_node_class

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

    # Dummy decorator to prevent import errors if JAX is missing
    def register_pytree_node_class(cls):
        return cls

# ----------------------------------------------------------

def _safe_log_weights(weights, dtype):
    abs_weights     = jnp.abs(weights)
    neg_inf_like    = jnp.asarray(-1.0e30, dtype=dtype)
    return jnp.where(abs_weights > 0, jnp.log(weights.astype(dtype)), neg_inf_like)

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

    def __init__(
        self,
        net_apply: Callable,
        operator: Any,
        input_shape: tuple,
        dtype: Any = None,
        statetype: str = "float32",
        **kwargs,
    ):

        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for AnsatzModifier.")

        self.net_apply      = net_apply
        self.input_shape    = input_shape
        self.dtype          = dtype if dtype else jnp.complex128
        self.statetype      = jnp.dtype(statetype)

        # Standardize the Operator Function
        # We expect op_func(state) -> (connected_states, weights)
        if hasattr(operator, "jax"):
            self.op_func = operator.jax
        elif callable(operator):
            self.op_func = operator
        else:
            raise ValueError(f"Operator {operator} must be callable or implement .jax()")

        # Inspect Branching Factor (M)
        # This determines if we take the fast path (M=1) or stable path (M>1)
        self._use_vmap_operator = False
        self.branching_factor   = self._inspect_branching()

        # Validation
        if self.branching_factor < 1:
            raise ValueError("Operator must return at least one connected state.")

    # ----------------------------------------------------------------------
    # Internal Utilities
    # ----------------------------------------------------------------------

    def _matches_state_shape(self, shape) -> bool:
        return tuple(shape) == tuple(self.input_shape)

    def _infer_branching_from_states_shape(self, states_shape) -> int:
        state_rank = len(self.input_shape)
        if len(states_shape) == state_rank and self._matches_state_shape(states_shape):
            return 1
        if len(states_shape) == state_rank + 1 and self._matches_state_shape(states_shape[-state_rank:]):
            return int(states_shape[0])
        raise ValueError(
            f"Operator output shape {states_shape} is incompatible with input_shape={self.input_shape}."
        )

    def _inspect_branching(self) -> int:
        """
        Safely determines the branching factor by tracing a dummy input.
        Returns M (int).
        """
        try:
            dummy_single = jnp.ones(self.input_shape, dtype=self.statetype)
            try:
                abstract_out = jax.eval_shape(self.op_func, dummy_single)
                branching = self._infer_branching_from_states_shape(abstract_out[0].shape)
                dummy_batch = jnp.ones((2, *self.input_shape), dtype=self.statetype)
                try:
                    batch_out = jax.eval_shape(self.op_func, dummy_batch)
                    batch_states_shape = batch_out[0].shape
                    batch_weights_shape = batch_out[1].shape
                    state_rank = len(self.input_shape)
                    native_single = (
                        len(batch_states_shape) == state_rank + 1
                        and batch_states_shape[0] == 2
                        and self._matches_state_shape(batch_states_shape[1:])
                    )
                    native_branch = (
                        len(batch_states_shape) == state_rank + 2
                        and batch_states_shape[0] == 2
                        and self._matches_state_shape(batch_states_shape[2:])
                    )
                    native_weights = (
                        len(batch_weights_shape) >= 1 and batch_weights_shape[0] == 2
                    )
                    self._use_vmap_operator = not ((native_single or native_branch) and native_weights)
                except Exception:
                    self._use_vmap_operator = True
                return branching
            except Exception:
                dummy = jnp.ones((1, *self.input_shape), dtype=self.statetype)
                abstract_out = jax.eval_shape(self.op_func, dummy)
                states_shape = abstract_out[0].shape
                state_rank = len(self.input_shape)
                if len(states_shape) == state_rank + 1 and self._matches_state_shape(states_shape[1:]):
                    return 1
                if len(states_shape) == state_rank + 2 and self._matches_state_shape(states_shape[2:]):
                    return int(states_shape[1])
                raise ValueError(
                    f"Operator output shape {states_shape} is incompatible with input_shape={self.input_shape}."
                )

        except Exception as e:
            warnings.warn(f"Could not inspect branching factor automatically ({e}). Defaulting to General/Sparse mode (M>1).")
            return 2

    def _apply_operator(self, x):
        if self._use_vmap_operator and getattr(x, "ndim", 0) > len(self.input_shape):
            return jax.vmap(self.op_func)(x)
        return self.op_func(x)

    def _broadcast_weights(self, weights, batch_size: int, branches: int):
        w = jnp.asarray(weights)
        if w.ndim == 0:
            return jnp.broadcast_to(w.reshape(1, 1), (batch_size, branches))
        if w.ndim == 1:
            if w.shape[0] == batch_size and branches == 1:
                return w.reshape(batch_size, 1)
            if w.shape[0] == branches:
                return jnp.broadcast_to(w.reshape(1, branches), (batch_size, branches))
            if w.shape[0] == 1:
                return jnp.broadcast_to(w.reshape(1, 1), (batch_size, branches))
        if w.ndim == 2:
            if w.shape == (batch_size, branches):
                return w
            if w.shape == (batch_size, 1) and branches == 1:
                return w
            if w.shape == (1, branches):
                return jnp.broadcast_to(w, (batch_size, branches))
            if w.shape == (1, 1):
                return jnp.broadcast_to(w, (batch_size, branches))
        raise ValueError(
            f"Operator weights shape {w.shape} is incompatible with batch_size={batch_size}, branches={branches}."
        )

    def _standardize_operator_output(self, x, st, w):
        state_rank = len(self.input_shape)
        x_ndim = getattr(x, "ndim", 0)
        st = jnp.asarray(st)

        if x_ndim == state_rank:
            if st.ndim == state_rank and self._matches_state_shape(st.shape):
                st = st.reshape((1, 1) + tuple(self.input_shape))
                w = self._broadcast_weights(w, 1, 1)
                return st, w
            if st.ndim == state_rank + 1 and self._matches_state_shape(st.shape[-state_rank:]):
                branches = int(st.shape[0])
                st = st.reshape((1, branches) + tuple(self.input_shape))
                w = self._broadcast_weights(w, 1, branches)
                return st, w
        else:
            batch_size = int(x.shape[0])
            if st.ndim == state_rank + 1 and st.shape[0] == batch_size and self._matches_state_shape(st.shape[1:]):
                st = st.reshape((batch_size, 1) + tuple(self.input_shape))
                w = self._broadcast_weights(w, batch_size, 1)
                return st, w
            if st.ndim == state_rank + 2 and st.shape[0] == batch_size and self._matches_state_shape(st.shape[2:]):
                branches = int(st.shape[1])
                w = self._broadcast_weights(w, batch_size, branches)
                return st, w

        raise ValueError(
            f"Operator states shape {st.shape} is incompatible with input shape {self.input_shape}."
        )

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
        st, w = self._apply_operator(x)
        st, w = self._standardize_operator_output(x, st, w)
        st = jnp.squeeze(st, axis=1)
        w = jnp.squeeze(w, axis=1)

        # Evaluate Base Ansatz
        log_psi = self.net_apply(params, st)

        # Combine: log(w * psi) = log(w) + log_psi
        # Ensure complex type for log to handle negative weights while preserving precision
        log_w_dtype = jnp.result_type(log_psi, jnp.complex64)
        log_w       = _safe_log_weights(w, log_w_dtype)

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
        st, w = self._apply_operator(x)
        st, w = self._standardize_operator_output(x, st, w)
        M = int(st.shape[1])

        # Flatten Batch and Branching dimensions
        # We need to feed (Batch * M, ...) into the network
        # Collapse dimensions 0 and 1
        flat_shape  = (batch_size * M, *self.input_shape)
        st_flat     = jnp.reshape(st, flat_shape)

        # Evaluate Base Ansatz
        log_psi_flat        = self.net_apply(params, st_flat)

        # Reshape back to (Batch, M)
        log_psi_connected   = jnp.reshape(log_psi_flat, (batch_size, M))

        # Compute LogSumExp
        # formula: log( sum_k w_k * exp(log_psi_k) )
        #        = log( sum_k exp( log(w_k) + log_psi_k ) )

        # Handle zero weights safely (padding)
        # jnp.log(0) is -inf, which logsumexp handles correctly (term vanishes)
        log_w_dtype = jnp.result_type(log_psi_connected, jnp.complex64)
        log_w       = _safe_log_weights(w, log_w_dtype)
        terms       = log_w + log_psi_connected

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
        aux_data = (
            self.net_apply,
            self.op_func,
            self.input_shape,
            self.dtype,
            self.branching_factor,
            self._use_vmap_operator,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs the object inside JIT."""
        net_apply, op_func, input_shape, dtype, branching_factor, use_vmap_operator = aux_data

        # Create instance without calling __init__ to avoid re-running introspection
        obj = cls.__new__(cls)
        obj.net_apply = net_apply
        obj.op_func = op_func
        obj.input_shape = input_shape
        obj.dtype = dtype
        obj.branching_factor = branching_factor
        obj._use_vmap_operator = use_vmap_operator
        return obj


# ----------------------------------------------------------
#! EOF
# ----------------------------------------------------------
