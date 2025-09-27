'''
Backend interface for Neural Quantum States (NQS) implementations.
'''
import os
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Union
from enum import Enum

#! -------------------------------------------------------------
JAX_AVAILABLE = os.environ.get('JAX_AVAILABLE', '1') == '1'
#! -------------------------------------------------------------

# --------------------------------------------------------------
#! Import net_utils from QES.general_python
# --------------------------------------------------------------

try:
    import QES.general_python.ml.net_impl.utils.net_utils as net_utils
except ImportError as e:
    warnings.warn("Could not import net_utils from QES.general_python.ml.net_impl.utils. Ensure QES.general_python is installed correctly.", ImportWarning)
    raise e

# --------------------------------------------------------------
#! QES Imports
# --------------------------------------------------------------

try:
    from QES.Algebra.hamil import Hamiltonian
    from QES.general_python.algebra.utils import Array
except ImportError as e:
    warnings.warn("Could not import Hamiltonian from QES.Algebra.hamil. Ensure QES is installed correctly.", ImportWarning)
    raise e

# Network
try:
    #! Neural Networks
    from QES.general_python.ml import networks as Networks
    
    #! For the gradients and stuff
    import QES.general_python.ml.net_impl.net_general as net_general

    if JAX_AVAILABLE:
        import QES.general_python.ml.net_impl.interface_net_flax as net_flax

    # schedulers and preconditioners and solvers
    # import QES.general_python.ml.schedulers as scheduler_mod
    # import QES.general_python.algebra.solvers as solvers_mod
except ImportError as e:
    warnings.warn("Some general_python.ml modules could not be imported. Ensure general_python.ml is installed correctly.", ImportWarning)
    raise e

# Different backends imports
try:
    # JAX imports
    if JAX_AVAILABLE:
        import jax
        from jax import jit as jax_jit
        from jax import numpy as jnp
    
        # jax tree
        try:
            from jax.tree import tree_flatten
            from orbax.checkpoint import CheckpointManager, PyTreeCheckpointHandler
        except ImportError:
            from jax.tree_util import tree_flatten
            CheckpointManager, PyTreeCheckpointHandler = None, None
        
        # use flax
        import flax
        import flax.linen as nn
        import flax.training.train_state
    else:
        jax, jnp, flax                              = None, None, None
        tree_flatten                                = None
        CheckpointManager, PyTreeCheckpointHandler  = None, None
except ImportError as e:
    warnings.warn("JAX or Flax could not be imported. Ensure they are installed correctly.", ImportWarning)
    raise e


class BackendInterface(ABC):
    """Abstract backend (NumPy, JAX, etc.)."""
    
    name: str = "abstract"

    @abstractmethod
    def eval_batched(self, net, params, states):
        """Evaluate ansatz on a batch of states."""
        pass

    @abstractmethod
    def apply_callable(self, func, params, states):
        """Apply an arbitrary callable to states."""
        pass

    @abstractmethod
    def local_energy(self, hamiltonian: Hamiltonian) -> Callable[[Any, Any], Any]:
        """Return a local energy function compatible with backend."""
        pass

    # ------
    
    @abstractmethod
    def compile_functions(self, net, batch_size: int):
        """
        Returns (ansatz_func, eval_func, apply_func).
        
        - ansatz_func(params, states)   -> e.g. log_psi - Neural Network output
        - eval_func(params, states)     -> e.g. evaluates a function on batched states
        - apply_func(params, states)    -> e.g. applies an arbitrary callable on batched states using MCMC
        """
        pass

    @abstractmethod
    def prepare_gradients(self, net):
        """
        Returns a dict with:
        - flat_grad_func            -> function to compute flattened gradients (log psi)
        - dict_grad_types           -> dict with gradient types per parameter
        - slice_metadata            -> metadata for slicing gradients - e.g. information about the structure of the gradients
        - leaf_info                 -> information about leaves in the parameter pytree
        - tree_def                  -> definition of the parameter pytree
        - shapes                    -> shapes of the parameters
        - sizes                     -> sizes of the parameters - useful for flattening
        - is_complex_per_leaf       -> whether each leaf is complex - helpful for complex differentiation
        - total_size                -> total size of the parameter pytree - useful for flattening
        """
        pass

# --------------------------------------------------------------
#! NumPy backend
# --------------------------------------------------------------

class NumpyBackend(BackendInterface):
    '''NumPy backend implementation.'''
    
    name: str = "numpy"
    
    def eval_batched(self, net: Any, params: Array, states: Array):
        return net_utils.numpy.eval_batched_np(net, params, states)

    def apply_callable(self, func: Callable, params: Array, states: Array):
        return net_utils.numpy.apply_callable_batched_np(func, params, states)

    def local_energy(self, hamiltonian: Hamiltonian) -> Callable[[Any, Any], Any]:
        return hamiltonian.get_loc_energy_np_fun()

    # ---

    def compile_functions(self, net, batch_size: int):
        # net.get_apply(use_jax=False) should return (apply_fn, params)
        ansatz_func, _params    = net.get_apply(use_jax=False)
        eval_func               = net_utils.numpy.eval_batched_np
        apply_func              = net_utils.numpy.apply_callable_batched_np
        return ansatz_func, eval_func, apply_func

    def prepare_gradients(self, net: Networks.GeneralNet):
        flat_grad_func, dict_grad_type = net_utils.decide_grads(
            iscpx           =   net.is_complex, 
            isjax           =   False,
            isanalytic      =   net.has_analytic_grad,
            isholomorphic   =   net.is_holomorphic
        )
        return dict(
            flat_grad_func      =   flat_grad_func,
            dict_grad_type      =   dict_grad_type,
            slice_metadata      =   None,
            leaf_info           =   None,
            tree_def            =   None,
            shapes              =   getattr(net, "shapes", None),
            sizes               =   None,
            is_complex_per_leaf =   None,
            total_size          =   getattr(net, "nparams", None),
        )

    @staticmethod
    def stable_ratio(exp_top, exp_bot):
        # exp_top/log_psi_top and exp_bot/log_psi_bot are log-amplitudes if you store logs;
        # compute ratio r = Ψ_top / Ψ_bot in log-space for stability.
        return np.exp(exp_top - exp_bot)

# --------------------------------------------------------------
#! JAX backend
# --------------------------------------------------------------

class JAXBackend(BackendInterface):
    '''JAX backend implementation.'''

    name: str = "jax"

    def eval_batched(self, net: nn.Module, params: Array, states: Array):
        return jax.jit(net_utils.jaxpy.eval_batched_jax, static_argnums=(0,1))(net, params, states)

    def apply_callable(self, func: Callable, params: Array, states: Array):
        return jax.jit(net_utils.jaxpy.apply_callable_batched_jax, static_argnums=(0,4,6))(func, params, states)

    def local_energy(self, hamiltonian: Hamiltonian) -> Callable[[Any, Any], Any]:
        return hamiltonian.get_loc_energy_jax_fun()

    def compile_functions(self, net: Networks.GeneralNet, batch_size: int):
        '''
        The JAX functions need to be compiled. The compilation depends on whether we use batching or not.
        Returns (ansatz_func, eval_func, apply_func).
        
        Parameters:
        -----------
        net : Networks.GeneralNet
            The neural network ansatz.        
        batch_size : int
            The batch size for MCMC sampling. If >1, use batched functions.
        '''
        
        # get the ansatz function
        ansatz_func, _ = net.get_apply(use_jax=True)
        
        # depending on batch size, compile the functions differently
        if batch_size and batch_size > 1:
            apply_func = jax.jit(net_utils.jaxpy.apply_callable_batched_jax, static_argnums=(0,4,6))
        else:
            apply_func = jax.jit(net_utils.jaxpy.apply_callable_jax, static_argnums=(0,4,6))
            
        # get the eval function
        eval_func = jax.jit(net_utils.jaxpy.eval_batched_jax, static_argnums=(0,1))
        
        return ansatz_func, eval_func, apply_func

    def prepare_gradients(self, net: Networks.GeneralNet, analytic: bool = None):
        params              = net.get_params()
        leaves, tree_def    = tree_flatten(params)
        leaf_info           = net_utils.jaxpy.prepare_leaf_info(params) # list of (name, shape, is_complex)
        slice_metadata      = net_utils.jaxpy.prepare_unflatten_metadata_from_leaf_info(leaf_info) # list of slice metadata
        sizes               = [s.size for s in slice_metadata]
        shapes              = [s.shape for s in slice_metadata]
        is_cplx             = [s.is_complex for s in slice_metadata]
        total_size          = slice_metadata[-1].start + slice_metadata[-1].size if slice_metadata else 0

        flat_grad_func, dict_grad_type = net_utils.decide_grads(
            iscpx           = net.is_complex, 
            isjax           = True,
            isanalytic      = net.has_analytic_grad,
            isholomorphic   = net.is_holomorphic
        )
        
        # prepare the dictionary
        out = dict(
            flat_grad_func      = flat_grad_func,
            dict_grad_type      = dict_grad_type,
            slice_metadata      = slice_metadata,
            leaf_info           = leaf_info,
            tree_def            = tree_def,
            shapes              = shapes,
            sizes               = sizes,
            is_complex_per_leaf = is_cplx,
            total_size          = total_size,
        )
        
        if analytic is not None and analytic:
            analytical_pytree_fun, _ = net.get_gradient(use_jax=True)
            if analytical_pytree_fun is None:
                raise ValueError("Analytical gradient function is not available.")
            out["analytical_pytree_fun"] = analytical_pytree_fun
        return out

    # ----------------------------------------------------------
    
    @staticmethod
    def stable_ratio(log_top, log_bot):
        '''Compute stable ratio of two exponentials in log-space.'''
        # log_top and log_bot are log-amplitudes if you store logs;
        # compute ratio r = Ψ_top / Ψ_bot in log-space for stability.
        return jnp.exp(log_top - log_bot)

# --------------------------------------------------------------
#! Summary of JAX functions
# --------------------------------------------------------------

class NQSBackendType(Enum):
    NUMPY = 'numpy'
    JAX   = 'jax'

# --------------------------------------------------------------

def nqs_get_backend(backend_type: Union[NQSBackendType, str]) -> BackendInterface:
    """Factory function to get the appropriate backend instance."""
    
    if isinstance(backend_type, str):
        b = (backend_type or "").lower()
        if b in ("jax", "gpu", "xla"):
            return JAXBackend()
        if b in ("numpy", "np", "cpu", "default"):
            return NumpyBackend()
        raise ValueError(f"Unsupported backend: {backend_type}")
    
    # otherwise, it's an enum
    if backend_type == NQSBackendType.NUMPY:
        return NumpyBackend()
    elif backend_type == NQSBackendType.JAX:
        if not JAX_AVAILABLE:
            raise ImportError("JAX backend requested but JAX is not available.")
        return JAXBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    return None

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------

# ---

# def init_state(self):
#     '''
#     Initialize the state of the network. This is done by creating a new
#     TrainState object with the network's parameters and the provided shape.
#     Returns:
#         flax.training.train_state.TrainState
#             The initialized TrainState object.
#     Note:
#         This method is only applicable if the backend is JAX and the network
#         is a Flax network. If the backend is not JAX, this method will return None.
        
#     '''

#     if JAX_AVAILABLE and self._isjax and issubclass(type(self._net), net_flax.FlaxInterface):
#         params   = self._net.init(self._rng_k, jnp.ones(self._shape, dtype=jnp.int32))
#         return flax.training.train_state.TrainState.create(
#             apply_fn = self._ansatz_func,
#             params   = params,
#             tx       = None
#         )
#     else:
#         return None