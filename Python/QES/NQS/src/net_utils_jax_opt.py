"""
Optimized version of apply_callable_batched_jax.
To be used as an override for performance.
"""
from typing import Any, Callable, Tuple
import jax
from jax import lax
import jax.numpy as jnp

def apply_callable_batched_jax(
                            func,
                            states,
                            sample_probas,
                            logprobas_in,
                            logproba_fun,
                            parameters,
                            batch_size : int,
                            *op_args):
    r"""
    Batched application of a local estimator.
    Optimized to use lax.map and avoid redundant dummy computation.
    """

    # ----------------------------------------------------------------------
    # Flatten over (samples, chains, ...) -> (N, D)
    # ----------------------------------------------------------------------
    flat_states   = states.reshape(-1, states.shape[-1])   # (N, D)
    flat_logp_in  = logprobas_in.reshape(-1)               # (N,)
    flat_samp_p   = sample_probas.reshape(-1)              # (N,)

    # N, pad, N_pad are ordinary Python ints -> OK for jnp.pad
    N             = flat_states.shape[0]
    pad           = (-N) % batch_size          # 0 … batch_size-1
    N_pad         = N + pad
    n_batches     = N_pad // batch_size

    if pad:
        pad_spec_states  = ((0, pad), (0, 0))
        pad_spec_1d      = ((0, pad),)
        p_states         = jnp.pad(flat_states,  pad_spec_states)
        p_logp_in        = jnp.pad(flat_logp_in, pad_spec_1d)
        p_samp_p         = jnp.pad(flat_samp_p,  pad_spec_1d)
    else:
        p_states         = flat_states
        p_logp_in        = flat_logp_in
        p_samp_p         = flat_samp_p

    # ----------------------------------------------------------------------
    # Per–state estimator
    # ----------------------------------------------------------------------
    def _estimate_one(state, logp0, p_sample, *args):
        if not args:
            new_states, new_vals = func(state)
        else:
            new_states, new_vals = func(state, *args)

        new_states = jnp.asarray(new_states)
        new_vals   = jnp.asarray(new_vals)

        logp_new   = logproba_fun(parameters, new_states)
        w          = jnp.exp(logp_new - logp0)

        return p_sample * jnp.sum(jnp.conj(new_vals) * w)

    in_axes      = (0, 0, 0) + (None,) * len(op_args)
    batch_kernel = jax.vmap(_estimate_one, in_axes=in_axes, out_axes=0)

    # ----------------------------------------------------------------------
    # Scan over batches using lax.map
    # ----------------------------------------------------------------------
    # Reshape for lax.map: (n_batches, batch_size, ...)
    # Last dimension is assumed to be state_size if any, or -1
    batched_states  = p_states.reshape(n_batches, batch_size, -1)
    batched_logp    = p_logp_in.reshape(n_batches, batch_size)
    batched_samp    = p_samp_p.reshape(n_batches, batch_size)

    def process_batch(args):
        b_s, b_lp, b_sp = args
        return batch_kernel(b_s, b_lp, b_sp, *op_args)

    # lax.map applies process_batch to each slice along axis 0
    estimates_stacked = lax.map(process_batch, (batched_states, batched_logp, batched_samp))
    estimates         = estimates_stacked.reshape(-1)[:N]

    return estimates, jnp.mean(estimates), jnp.std(estimates)
