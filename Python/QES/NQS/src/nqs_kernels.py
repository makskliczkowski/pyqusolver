"""
Kernels for NQS.
Separated for modularity and JIT compilation cache stability.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

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
    """
    Data class to hold the results of a single optimization step in the NQS solver.
    """

    loss: float  # Estimated energy or loss value
    loss_mean: float  # Mean of the energy or loss
    loss_std: float  # Standard deviation of the energy or loss
    grad_flat: Array  # Flattened gradient vector
    params_shapes: List[Tuple]  # Shapes of the parameters
    params_sizes: List[int]  # Sizes of the parameters
    params_cpx: List[bool]  # Whether each parameter is complex


# Register as PyTree node for JIT compatibility
if JAX_AVAILABLE:
    jax.tree_util.register_pytree_node(
        NQSSingleStepResult,
        lambda n: (
            (n.loss, n.loss_mean, n.loss_std, n.grad_flat),
            (n.params_shapes, n.params_sizes, n.params_cpx),
        ),
        lambda aux, children: NQSSingleStepResult(*children, *aux),
    )

# ----------------------------------------------------------------------
#! Unified Function Application Kernel
# ----------------------------------------------------------------------


def apply_fun(
    func: Callable,
    states: Array,
    probabilities: Array,
    logproba_in: Array,
    logproba_fun: Callable,
    parameters: Any,
    batch_size: Optional[int] = None,
    is_jax: bool = True,
    *op_args,
):
    """
    Unified dispatcher for applying a function (e.g. Local Energy) to a batch of states.
    Dispatches to JAX or NumPy implementation based on `is_jax`.
    """
    if is_jax and JAX_AVAILABLE:
        return _apply_fun_jax(
            func, states, probabilities, logproba_in, logproba_fun, parameters, batch_size, *op_args
        )
    else:
        return _apply_fun_np(
            func, states, probabilities, logproba_in, logproba_fun, parameters, batch_size, *op_args
        )


@partial(jax.jit, static_argnames=["func", "logproba_fun", "batch_size"])
def _apply_fun_jax(
    func: Callable,  # function to be evaluated (e.g., local energy f: s -> E_loc(s))
    states: Array,  # input states (shape: (N, ...))
    probabilities: Array,  # probabilities associated with the states (shape: (N,))
    logproba_in: Array,  # logarithm of the probabilities for the input states (\log p(s))
    logproba_fun: Callable,  # function to compute the logarithm of probabilities (\log p(s') -> to evaluate)
    parameters: Any,  # parameters to be passed to the function - for the Networks ansatze
    batch_size: Optional[int] = None,  # batch size for evaluation
    *op_args,
):  # additional arguments to pass to func (broadcast across states)
    r"""
    Evaluates a given function on a set of states and probabilities, with optional batching.
    Args:
        func (Callable):
            The function to be evaluated.
        states (jnp.ndarray):
            The input states for the function.
        probabilities (jnp.ndarray):
            The probabilities associated with the states.
        logproba_in (jnp.ndarray):
            The logarithm of the probabilities for the input states.
        logproba_fun (Callable):
            A function to compute the logarithm of probabilities.
        parameters (Union[dict, list, jnp.ndarray]):
            Parameters to be passed to the function.
        batch_size (Optional[int], optional):
            The size of batches for evaluation.
            If None, the function is evaluated without batching. Defaults to None.
        *op_args:
            Additional arguments to pass to func. These are broadcast (not vmapped) across states.
            Useful for operator indices (i, j) in correlation functions.
    Returns:
        The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
    """
    if batch_size is None or batch_size == 1:
        funct_in = net_utils.jaxpy.apply_callable_jax
        return funct_in(
            func, states, probabilities, logproba_in, logproba_fun, parameters, 1, *op_args
        )
    else:
        funct_in = net_utils.jaxpy.apply_callable_batched_jax
        return funct_in(
            func, states, probabilities, logproba_in, logproba_fun, parameters, batch_size, *op_args
        )


def _apply_fun_np(
    func: Callable,
    states: np.ndarray,
    probabilities: np.ndarray,
    logproba_in: np.ndarray,
    logproba_fun: Callable,
    parameters: Any,
    batch_size: Optional[int] = None,
    *op_args,
):
    """
    Evaluates a given function on a set of states and probabilities, with optional batching.

    Args:
        func (Callable):
            The function to be evaluated.
        states (np.ndarray):
            The input states for the function.
        probabilities (np.ndarray):
            The probabilities associated with the states.
        logproba_in (np.ndarray):
            The logarithm of the probabilities for the input states.
        logproba_fun (Callable):
            A function to compute the logarithm of probabilities.
        parameters (Union[dict, list, np.ndarray]):
            Parameters to be passed to the function.
        batch_size (Optional[int], optional):
            The size of batches for evaluation.
            If None, the function is evaluated without batching. Defaults to None.
        *op_args:
            Additional arguments to pass to the function.
            Note: NumPy backend currently might not support op_args in all implementations.
            They are passed to the underlying net_utils function if supported.
    Returns:
        The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
    """

    # We assume net_utils.numpy.apply_callable_np/batched_np support *op_args.
    # If not, this might fail or we should ignore them.
    # Given the JAX implementation supports it, it's safer to pass them.

    if batch_size is None:
        funct_in = net_utils.numpy.apply_callable_np
        return funct_in(
            func, states, probabilities, logproba_in, logproba_fun, parameters, *op_args
        )

    # otherwise, we shall use the batched version
    funct_in = net_utils.numpy.apply_callable_batched_np
    return funct_in(
        func=func,
        states=states,
        sample_probas=probabilities,
        logprobas_in=logproba_in,
        logproba_fun=logproba_fun,
        parameters=parameters,
        batch_size=batch_size,
        *op_args,
    )


# ----------------------------------------------------------------------
#! Log Derivative Kernels
# ----------------------------------------------------------------------


@partial(jax.jit, static_argnames=["net_apply", "single_sample_flat_grad_fun", "batch_size"])
def log_derivative_jax(
    net_apply: Callable,  # The network's apply function f(p, x)
    params: Any,  # Network parameters p
    states: jnp.ndarray,  # Input states s_i, shape (num_samples, ...)
    single_sample_flat_grad_fun: Callable[
        [Callable, Any, Any], jnp.ndarray
    ],  # JAX-traceable function computing the flattened gradient for one sample.
    batch_size: int = 1,
) -> Tuple[Array, List, List, List]:  # Batch size
    r"""
    Compute the batch of flattened gradients using JAX (JIT compiled).

    Returns the gradients (e.g., :math:`O_k = \\nabla \\ln \\psi(s)`)
    for each state in the batch. The output format (complex/real)
    depends on the `single_sample_flat_grad_fun` used.

    Parameters
    ----------
    net_apply : Callable
        The network's apply function `f(p, x)`. Static argument for JIT.
    params : Any
        Network parameters `p`.
    states : jnp.ndarray
        Input states `s_i`, shape `(num_samples, ...)`.
    single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray]
        JAX-traceable function computing the flattened gradient for one sample.
        Signature: `fun(net_apply, params, single_state) -> flat_gradient_vector`.
        Static argument for JIT.
    batch_size : int
        Batch size. Static argument for JIT.

    Returns
    -------
    jnp.ndarray
        Array of flattened gradients, shape `(num_samples, num_flat_params)`.
        Dtype matches the output of `single_sample_flat_grad_fun`.
    """
    gradients_batch, shapes, sizes, is_cpx = net_utils.jaxpy.compute_gradients_batched(
        net_apply, params, states, single_sample_flat_grad_fun, batch_size
    )
    return gradients_batch, shapes, sizes, is_cpx


def log_derivative_np(net, params, batch_size, states, flat_grad) -> np.ndarray:
    r"""
    !TODO: Add the precomputed gradient vector - memory efficient
    """
    sb = net_utils.numpy.create_batches_np(states, batch_size)
    if len(states) == 0:
        return np.array([])

    g = None
    idx = 0

    # Process each batch
    for b in sb:
        g_batch = flat_grad(net, params, b)

        if g is None:
            # Pre-allocate based on first batch result for memory efficiency
            # g_batch shape is (batch_size, n_params)
            param_shape = g_batch.shape[1:]
            g = np.zeros((len(states),) + param_shape, dtype=g_batch.dtype)

        # Copy batch into main array
        n_in_batch = g_batch.shape[0]
        g[idx : idx + n_in_batch] = g_batch
        idx += n_in_batch

    return g


# ----------------------------------------------------------------------
#! NQS Single Step Kernel
# ----------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "ansatz_fn",
        "apply_fn",
        "local_loss_fn",
        "flat_grad_fn",
        "compute_grad_f",
        "accum_real_dtype",
        "accum_complex_dtype",
        "use_jax",
        "batch_size",
    ],
)
def _single_step(
    params: Any,
    configs: Array,
    configs_ansatze: Any,
    probabilities: Any,
    # functions (Static input for JIT)
    ansatz_fn: Callable = None,
    apply_fn: Callable = None,
    local_loss_fn: Callable = None,
    flat_grad_fn: Callable = None,
    compute_grad_f: Callable = net_utils.jaxpy.compute_gradients_batched,
    # precision (Static input for JIT)
    accum_real_dtype: Any = None,
    accum_complex_dtype: Any = None,
    use_jax: bool = True,
    # Static for evaluation
    batch_size: int = None,
    t: float = None,
    int_step: int = 0,
) -> NQSSingleStepResult:
    """
    Perform a single training step. Pure function.
    """

    #! a) Compute Local Loss
    # For example, E_loc = <s|H|psi> / <s|psi> or loss for density matrix
    v, means, stds = apply_fn(
        func=local_loss_fn,
        states=configs,
        sample_probas=probabilities,
        logprobas_in=configs_ansatze,
        logproba_fun=ansatz_fn,
        parameters=params,
        batch_size=batch_size,
    )

    #! b) Compute Gradients
    # O_k = nabla log psi
    flat_grads, shapes, sizes, iscpx = compute_grad_f(
        net_apply=ansatz_fn,
        params=params,
        states=configs,
        single_sample_flat_grad_fun=flat_grad_fn,
        batch_size=batch_size,
    )

    # Promote numerically sensitive outputs - e.g., loss mean/std and gradients if user wants to have fast ansatz and high-precision accumulation
    v = cast_for_precision(v, accum_real_dtype, accum_complex_dtype, use_jax)
    means = cast_for_precision(means, accum_real_dtype, accum_complex_dtype, use_jax)
    stds = cast_for_precision(stds, accum_real_dtype, accum_complex_dtype, use_jax)
    flat_grads = cast_for_precision(flat_grads, accum_real_dtype, accum_complex_dtype, use_jax)

    return NQSSingleStepResult(
        loss=v,
        loss_mean=means,
        loss_std=stds,
        grad_flat=flat_grads,
        params_shapes=shapes,
        params_sizes=sizes,
        params_cpx=iscpx,
    )


# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------
