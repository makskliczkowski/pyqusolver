"""
Operator helpers shared across NQS spectral workflows.
"""

from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False


def identity_transition_kernel(state):
    """Return the identity transition kernel for a single basis configuration."""
    backend = jnp if JAX_AVAILABLE else np
    x = backend.asarray(state)
    if getattr(x, "ndim", 0) != 1:
        x = x.reshape(-1)
    weights = backend.ones((1,), dtype=backend.result_type(x.dtype, backend.float32))
    return x.reshape((1, -1)), weights


def materialize_apply_values(result):
    """Extract local values from the return types used by ``NQS.apply``."""
    if hasattr(result, "values"):
        return result.values
    if isinstance(result, tuple):
        return result[0]
    return result


def resolve_transition_kernel(operator):
    """Resolve a spectral operator into a connected-state transition kernel."""
    if hasattr(operator, "jax"):
        return operator.jax
    if callable(operator):
        return operator
    raise ValueError("Operator must be callable or expose a .jax transition kernel.")


def diagonal_operator_square(operator):
    """Return the diagonal kernel corresponding to ``A^dagger A`` for single-branch probes."""
    operator_fun = resolve_transition_kernel(operator)
    backend = jnp if JAX_AVAILABLE else np

    def square_kernel(state):
        connected_states, weights = operator_fun(state)
        weights_arr = backend.asarray(weights)
        if getattr(weights_arr, "ndim", 0) == 0:
            weights_arr = weights_arr.reshape(1)
        if getattr(weights_arr, "shape", (0,))[0] != 1:
            raise ValueError(
                "Automatic probe-weight estimation requires a diagonal or single-branch operator."
            )
        return connected_states, backend.conj(weights_arr) * weights_arr

    return square_kernel


def diagonal_probe_overlap_operator(bra_operator, ket_operator):
    """Return the diagonal kernel for ``A^dagger B`` when both probes are single-branch."""
    if getattr(bra_operator, "modifies", True) or getattr(ket_operator, "modifies", True):
        raise ValueError("Direct probe-overlap estimation is only valid for diagonal probes.")

    bra_fun = resolve_transition_kernel(bra_operator)
    ket_fun = resolve_transition_kernel(ket_operator)
    backend = jnp if JAX_AVAILABLE else np

    def overlap_kernel(state):
        bra_states, bra_weights = bra_fun(state)
        ket_states, ket_weights = ket_fun(state)
        bra_arr = backend.asarray(bra_weights)
        ket_arr = backend.asarray(ket_weights)
        if getattr(bra_arr, "ndim", 0) == 0:
            bra_arr = bra_arr.reshape(1)
        if getattr(ket_arr, "ndim", 0) == 0:
            ket_arr = ket_arr.reshape(1)
        if getattr(bra_arr, "shape", (0,))[0] != 1 or getattr(ket_arr, "shape", (0,))[0] != 1:
            raise ValueError(
                "Direct probe-overlap estimation requires single-branch diagonal probes."
            )
        return bra_states, backend.conj(bra_arr) * ket_arr

    return overlap_kernel


__all__ = [
    "diagonal_operator_square",
    "diagonal_probe_overlap_operator",
    "identity_transition_kernel",
    "materialize_apply_values",
    "resolve_transition_kernel",
]
