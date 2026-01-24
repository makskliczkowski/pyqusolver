"""
NQS Entropy Utilities
=====================

Provides batched entropy calculations using JAX, filling gaps in general_python.physics.entropy.
"""

from typing import Optional
try:
    import jax
    import jax.numpy as jnp
    from QES.general_python.physics import entropy_jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if JAX_AVAILABLE:

    @jax.jit
    def participation_entropy_jax(state: jnp.ndarray, q: float = 1.0, threshold: float = 1e-12) -> float:
        """
        Compute participation entropy for a single state vector.
        S_q = (1/(1-q)) ln( sum_i p_i^q ) where p_i = |psi_i|^2
        """
        probs = jnp.abs(state)**2
        # Clean small probs
        probs = jnp.where(probs < threshold, 0.0, probs)
        norm = jnp.sum(probs)
        probs = jnp.where(norm > 0, probs / norm, probs)

        def _shannon(_):
            return -jnp.sum(jnp.where(probs > threshold, probs * jnp.log(probs), 0.0))

        def _renyi(_):
            sum_pq = jnp.sum(probs**q)
            return jnp.log(sum_pq) / (1.0 - q)

        return jax.lax.cond(jnp.isclose(q, 1.0), _shannon, _renyi, operand=None)

    @jax.jit
    def entropy_batched_jax(
        lam: jnp.ndarray,
        q: float = 1.0,
        base: float = jnp.e
    ) -> jnp.ndarray:
        """
        Compute entropies for a batch of probability/eigenvalue sets.

        Parameters
        ----------
        lam : jnp.ndarray
            Input array of shape (batch_size, dim).
            Each row is a probability distribution or set of eigenvalues.
        q : float
            Rényi order (q=1 for Von Neumann).
        base : float
            Logarithm base.

        Returns
        -------
        jnp.ndarray
            Array of entropies, shape (batch_size,).
        """

        # We use vmap over the existing single-sample functions in entropy_jax

        def _compute_single(l):
            # renyi_entropy_jax handles q=1 internally by dispatching to vn_entropy_jax
            return entropy_jax.renyi_entropy_jax(l, q, base)

        return jax.vmap(_compute_single)(lam)

    @jax.jit
    def participation_entropy_batched_jax(
        states: jnp.ndarray,
        q: float = 1.0,
        threshold: float = 1e-12
    ) -> jnp.ndarray:
        """
        Batched participation entropy.

        Parameters
        ----------
        states : jnp.ndarray
            Input states batch, shape (batch_size, dim).
        q : float
            Entropy order.
        threshold : float
            Threshold for participation.

        Returns
        -------
        jnp.ndarray
            Array of participation entropies, shape (batch_size,).
        """
        return jax.vmap(lambda s: participation_entropy_jax(s, q, threshold))(states)

else:
    def entropy_batched_jax(*args, **kwargs):
        raise NotImplementedError("JAX not available")

    def participation_entropy_batched_jax(*args, **kwargs):
        raise NotImplementedError("JAX not available")
