'''
Backend interface for Neural Quantum States (NQS) implementations.

This module defines an abstract backend interface and concrete implementations
for different computational backends such as NumPy and JAX.

--------------------------------------------------------------
File                : NQS/src/nqs_backend.py
Author              : Maksymilian Kliczkowski
--------------------------------------------------------------
'''
import os
import warnings
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Union

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
    from QES.Algebra.hamil                  import Hamiltonian
    from QES.general_python.algebra.utils   import Array
except ImportError as e:
    warnings.warn("Could not import Hamiltonian from QES.Algebra.hamil. Ensure QES is installed correctly.", ImportWarning)
    raise e

# Network
try:
    #! Neural Networks
    from QES.general_python.ml import networks as Networks
except ImportError as e:
    raise ImportError("Could not import general_python.ml.networks modules. Make sure general_python.ml is installed correctly.") from e

# Different backends imports
try:
    # JAX imports
    import jax
    from jax import numpy as jnp
    
    # jax tree
    try:
        from jax.tree           import tree_flatten
        from orbax.checkpoint   import CheckpointManager, PyTreeCheckpointHandler
    except ImportError:
        from jax.tree_util      import tree_flatten
        CheckpointManager, PyTreeCheckpointHandler = None, None
    
    # use flax
    import flax.linen as nn
    
    JAX_AVAILABLE               = True
except ImportError as e:
    JAX_AVAILABLE               = False
    jax, jnp, flax              = None, None, None
    tree_flatten                = None
    CheckpointManager           = None
    PyTreeCheckpointHandler     = None
    warnings.warn("JAX or Flax could not be imported. Ensure they are installed correctly.", ImportWarning)
    raise e

# --------------------------------------------------------------
#! Backend Interface
# --------------------------------------------------------------

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
    #! Compilation and gradient preparation
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
    def prepare_gradients(self, net, analytic: bool = None):
        """
        Returns a dict with:
        - analytic_grad_fun             -> function to compute analytical gradients in pytree form (if available)
        - flat_grad_func                -> function to compute flattened gradients (log psi)
        - dict_grad_types               -> dict with gradient types per parameter
        - slice_metadata                -> metadata for slicing gradients - e.g. information about the structure of the gradients
        - leaf_info                     -> information about leaves in the parameter pytree
        - tree_def                      -> definition of the parameter pytree
        - shapes                        -> shapes of the parameters
        - sizes                         -> sizes of the parameters - useful for flattening
        - is_complex_per_leaf           -> whether each leaf is complex - helpful for complex differentiation
        - total_size                    -> total size of the parameter pytree - useful for flattening
        """
        pass
    
    @staticmethod
    def stable_ratio(exp_top, exp_bot):
        """Compute stable ratio of two exponentials in log-space."""
        return exp_top / exp_bot
    
    # ----------------------------------------------------------
    
    @staticmethod
    def to_numpy(x):
        """Convert backend array to NumPy array."""
        return np.array(x)
    
    @staticmethod
    def to_jax(x, dtype=None, **kwargs):
        """Convert input to JAX array with optional dtype."""
        return x    

    @staticmethod
    def is_jax_array(x):
        """Check if input is a JAX array."""
        return False
    
    @staticmethod
    def is_numpy_array(x):
        """Check if input is a NumPy array."""
        return isinstance(x, np.ndarray)
    
    # ----------------------------------------------------------    
    
    @staticmethod
    def asarray(x, dtype=None, **kwargs):
        """Convert input to backend array with optional dtype."""
        return np.asarray(x, dtype=dtype, **kwargs)
    
    @staticmethod
    def device_put(x):
        """Put array on JAX device."""
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

    # ----------------------------------------------------------

    def compile_functions(self, net, batch_size: int):
        # net.get_apply(use_jax=False) should return (apply_fn, params)
        ansatz_func, _params    = net.get_apply(use_jax=False)
        eval_func               = net_utils.numpy.eval_batched_np
        apply_func              = net_utils.numpy.apply_callable_batched_np
        return ansatz_func, eval_func, apply_func

    def prepare_gradients(self, net: Networks.GeneralNet, analytic: bool = None):
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
            analytic_grad_func  =   None if analytic is None or not analytic else net.get_gradient(use_jax=False)[0],
        )

    @staticmethod
    def stable_ratio(exp_top, exp_bot):
        # exp_top/log_psi_top and exp_bot/log_psi_bot are log-amplitudes if you store logs;
        # compute ratio r = Psi_top / Psi_bot in log-space for stability.
        return np.exp(exp_top - exp_bot)
    
    # ----------------------------------------------------------
    
    @staticmethod
    def to_numpy(x):
        '''Convert NumPy array to NumPy array.'''
        return np.array(x)

    @staticmethod
    def asarray(x, dtype=None, **kwargs):
        '''Convert input to NumPy array with optional dtype.'''
        if dtype is not None:
            return np.asarray(x, dtype=dtype, **kwargs)
        return np.asarray(x, **kwargs)

# --------------------------------------------------------------
#! JAX backend
# --------------------------------------------------------------

class JAXBackend(BackendInterface):
    '''JAX backend implementation.'''

    name: str   = "jax"
    
    # Class-level cache for JIT'd functions to prevent recompilation
    _jit_cache  = {}
    
    @classmethod
    def _get_cached_jit(cls, func, static_argnums):
        """Get or create cached JIT'd function to avoid recompilation."""
        key = (id(func), static_argnums)
        if key not in cls._jit_cache:
            cls._jit_cache[key] = jax.jit(func, static_argnums=static_argnums)
        return cls._jit_cache[key]

    def eval_batched(self, net: nn.Module, params: Array, states: Array):
        jitted = self._get_cached_jit(net_utils.jaxpy.eval_batched_jax, (0,1))
        return jitted(net, params, states)

    def apply_callable(self, func: Callable, params: Array, states: Array):
        jitted = self._get_cached_jit(net_utils.jaxpy.apply_callable_batched_jax, (0,4,6))
        return jitted(func, params, states)

    def local_energy(self, hamiltonian: Hamiltonian) -> Callable[[Any, Any], Any]:
        return hamiltonian.get_loc_energy_jax_fun()

    # ----------------------------------------------------------

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
            analytic_grad_func  = None,
            flat_grad_func      = flat_grad_func,
            dict_grad_type      = dict_grad_type,
            slice_metadata      = slice_metadata,
            leaf_info           = leaf_info,
            leaves              = leaves,
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
            out["analytic_grad_func"] = analytical_pytree_fun
        return out

    # ----------------------------------------------------------
    
    @staticmethod
    def stable_ratio(log_top, log_bot):
        '''Compute stable ratio of two exponentials in log-space.'''
        # log_top and log_bot are log-amplitudes if you store logs;
        # compute ratio r = Psi_top / Psi_bot in log-space for stability.
        return jnp.exp(log_top - log_bot)

    # ----------------------------------------------------------

    @staticmethod
    def to_numpy(x):
        '''Convert JAX array to NumPy array.'''
        return np.array(x)
    
    @staticmethod
    def to_jax(x, dtype=None, **kwargs):
        '''Convert input to JAX array with optional dtype.'''
        if dtype is not None:
            return jnp.array(x, dtype=dtype, **kwargs)
        return jnp.array(x, **kwargs)
    
    @staticmethod
    def is_jax_array(x):
        '''Check if input is a JAX array.'''
        return isinstance(x, jnp.ndarray)

    # ----------------------------------------------------------
    
    @staticmethod
    def asarray(x, dtype=None, **kwargs):
        '''Convert input to JAX array with optional dtype.'''
        if dtype is not None:
            return jnp.asarray(x, dtype=dtype, **kwargs)
        return jnp.asarray(x, **kwargs)
    
    @staticmethod
    def device_put(x):
        '''Put array on JAX device.'''
        return jax.device_put(x)

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