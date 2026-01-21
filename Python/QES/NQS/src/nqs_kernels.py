"""
Kernels for NQS.
Separated for modularity and JIT compilation cache stability.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Any, Optional, Tuple, List, Union
from functools import partial
from dataclasses import dataclass

# Import net_utils
try:
    import QES.general_python.ml.net_impl.utils.net_utils as net_utils
except ImportError:
    # Fallback or error - assume strict dependency
    raise ImportError("Could not import net_utils from QES.general_python.ml.net_impl.utils.")

from .nqs_precision import cast_for_precision

Array = Any

@dataclass
class NQSSingleStepResult:
    '''
    Data class to hold the results of a single optimization step in the NQS solver.
    '''
    loss                : float                     # Estimated energy or loss value
    loss_mean           : float                     # Mean of the energy or loss
    loss_std            : float                     # Standard deviation of the energy or loss
    grad_flat           : Array                     # Flattened gradient vector
    params_shapes       : List[Tuple]               # Shapes of the parameters
    params_sizes        : List[int]                 # Sizes of the parameters
    params_cpx          : List[bool]                # Whether each parameter is complex

# Register as PyTree node for JIT compatibility
jax.tree_util.register_pytree_node(
    NQSSingleStepResult,
    lambda n: ((n.loss, n.loss_mean, n.loss_std, n.grad_flat), (n.params_shapes, n.params_sizes, n.params_cpx)),
    lambda aux, children: NQSSingleStepResult(*children, *aux)
)

@partial(jax.jit, static_argnames=['func', 'logproba_fun', 'batch_size'])
def _apply_fun_jax(func   : Callable,                           # function to be evaluated (e.g., local energy f: s -> E_loc(s))
            states        : Array,                              # input states (shape: (N, ...))
            probabilities : Array,                              # probabilities associated with the states (shape: (N,))
            logproba_in   : Array,                              # logarithm of the probabilities for the input states (\log p(s))
            logproba_fun  : Callable,                           # function to compute the logarithm of probabilities (\log p(s') -> to evaluate)
            parameters    : Any,                                # parameters to be passed to the function - for the Networks ansatze
            batch_size    : Optional[int] = None,               # batch size for evaluation
            *op_args):                                          # additional arguments to pass to func
    """
    Evaluates a given function on a set of states and probabilities, with optional batching.
    """
    if batch_size is None or batch_size == 1:
        funct_in = net_utils.jaxpy.apply_callable_jax
        return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters, 1, *op_args)
    else:
        funct_in = net_utils.jaxpy.apply_callable_batched_jax
        return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters, batch_size, *op_args)

def _apply_fun_np(func    : Callable,
            states        : np.ndarray,
            probabilities : np.ndarray,
            logproba_in   : np.ndarray,
            logproba_fun  : Callable,
            parameters    : Any,
            batch_size    : Optional[int] = None):
    """
    Evaluates a given function on a set of states and probabilities, with optional batching.
    """

    if batch_size is None:
        funct_in = net_utils.numpy.apply_callable_np
        return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters)

    # otherwise, we shall use the batched version
    funct_in = net_utils.numpy.apply_callable_batched_np
    return funct_in(func            = func,
                    states          = states,
                    sample_probas   = probabilities,
                    logprobas_in    = logproba_in,
                    logproba_fun    = logproba_fun,
                    parameters      = parameters,
                    batch_size      = batch_size)

@partial(jax.jit, static_argnames=['net_apply', 'single_sample_flat_grad_fun', 'batch_size'])
def log_derivative_jax(
        net_apply                   : Callable,
        params                      : Any,
        states                      : jnp.ndarray,
        single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray],
        batch_size                  : int = 1) -> Tuple[Array, List, List, List]:
    '''
    Compute the batch of flattened gradients using JAX (JIT compiled).
    '''
    gradients_batch, shapes, sizes, is_cpx = net_utils.jaxpy.compute_gradients_batched(net_apply, params, states, single_sample_flat_grad_fun, batch_size)
    return gradients_batch, shapes, sizes, is_cpx

def log_derivative_np(net, params, batch_size, states, flat_grad) -> np.ndarray:
    r'''
    Optimized numpy log derivative.
    '''
    sb = net_utils.numpy.create_batches_np(states, batch_size)

    # Need to handle empty states?
    if len(states) == 0:
        return np.array([])

    # Determine gradient shape from first sample/batch
    # Probe with 1 sample
    # Note: flat_grad usually returns 1D array for a single sample, or (Batch, ...) for batch
    # But checking original code: flat_grad(net, params, b)
    # If b is a batch, it should return (Batch, N_params).

    grads_list = []
    for b in sb:
        g_batch = flat_grad(net, params, b)
        grads_list.append(g_batch)

    if grads_list:
        return np.concatenate(grads_list, axis=0)
    else:
        return np.array([])

@partial(jax.jit, static_argnames=['ansatz_fn', 'apply_fn', 'local_loss_fn', 'flat_grad_fn', 'compute_grad_f', 'accum_real_dtype', 'accum_complex_dtype', 'use_jax', 'batch_size'])
def _single_step(params         : Any,
                configs         : Array,
                configs_ansatze : Any,
                probabilities   : Any,
                # functions (Static input for JIT)
                ansatz_fn       : Callable  = None,
                apply_fn        : Callable  = None,
                local_loss_fn   : Callable  = None,
                flat_grad_fn    : Callable  = None,
                compute_grad_f  : Callable  = net_utils.jaxpy.compute_gradients_batched,
                # precision (Static input for JIT)
                accum_real_dtype    : Any   = None,
                accum_complex_dtype : Any   = None,
                use_jax             : bool  = True,
                # Static for evaluation
                batch_size      : int       = None,
                t               : float     = None,
                int_step        : int       = 0) -> NQSSingleStepResult:
    '''
    Perform a single training step. Pure function.
    '''

    #! a) Compute Local Loss
    (v, means, stds) = apply_fn(
                        func            = local_loss_fn,
                        states          = configs,
                        sample_probas   = probabilities,
                        logprobas_in    = configs_ansatze,
                        logproba_fun    = ansatz_fn,
                        parameters      = params,
                        batch_size      = batch_size
                    )

    #! b) Compute Gradients
    flat_grads, shapes, sizes, iscpx = compute_grad_f(
                                        net_apply                   = ansatz_fn,
                                        params                      = params,
                                        states                      = configs,
                                        single_sample_flat_grad_fun = flat_grad_fn,
                                        batch_size                  = batch_size
                                    )

    # Promote numerically sensitive outputs
    v          = cast_for_precision(v, accum_real_dtype, accum_complex_dtype, use_jax)
    means      = cast_for_precision(means, accum_real_dtype, accum_complex_dtype, use_jax)
    stds       = cast_for_precision(stds, accum_real_dtype, accum_complex_dtype, use_jax)
    flat_grads = cast_for_precision(flat_grads, accum_real_dtype, accum_complex_dtype, use_jax)

    return NQSSingleStepResult(loss=v, loss_mean=means, loss_std=stds, grad_flat=flat_grads, params_shapes=shapes, params_sizes=sizes, params_cpx=iscpx)
