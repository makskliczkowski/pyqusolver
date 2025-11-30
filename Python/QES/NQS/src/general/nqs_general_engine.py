"""
Unified Evaluation Engine for Neural Quantum States

This module provides a unified interface for evaluating NQS quantities with automatic
backend dispatch (JAX/NumPy).

Architecture:
    EvaluationBackend (abstract)
        ├── JAXBackend      (JIT-compiled vectorized evaluation)
        ├── NumpyBackend    (efficient batch processing)
        └── AutoBackend     (dispatch based on input type)

    BatchProcessor       (handles batching logic uniformly)
    EvaluationConfig     (configuration dataclass)
    EvaluationResult     (result dataclass)

Key Features:
    - Unified dispatcher eliminates duplicated backend logic
    - Batch processing abstraction for consistent behavior
    - JAX JIT compilation support
    - Error handling and validation
    - Backwards compatible with existing NQS interface

-------------------------------------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 2025-11-01
-------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Tuple, Any, Dict

import os
import numpy as np

# ---- JAX imports ----
try:
    from QES.general_python.algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    JAX_AVAILABLE = False

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap
    except ImportError as e:
        pass
else:
    raise ImportError("JAX is not available but required for JAXBackend.")

__all__ = [
    'EvaluationBackend',
    'JAXBackend', 
    'NumpyBackend',
    'AutoBackend',
    'BatchProcessor',
    'EvaluationConfig',
    'EvaluationResult',
    'UnifiedEvaluationEngine',
]

#####################################################################################################
#! CONFIGURATION AND RESULT DATACLASSES
#####################################################################################################

@dataclass
class EvaluationConfig:
    """Configuration for evaluation operations."""
    
    backend         : str = 'auto'              # 'jax', 'numpy', or 'auto'
    batch_size      : Optional[int] = None      # None means no batching
    jit_compile     : bool = True               # JIT compile for JAX backend
    return_stats    : bool = True               # Whether to compute and return statistics
    validate_inputs : bool = True               # Validate input states
    verbose         : bool = False              # Verbose logging
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.backend not in ['auto', 'jax', 'numpy']:
            raise ValueError(f"Invalid backend '{self.backend}'. Must be 'auto', 'jax', or 'numpy'.")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

# ----------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Result of an evaluation operation."""
    
    values          : Array                         # Computed values array
    mean            : Optional[float] = None        # Mean of values
    std             : Optional[float] = None        # Standard deviation of values
    min_val         : Optional[float] = None        # Minimum value
    max_val         : Optional[float] = None        # Maximum value
    n_samples       : int             = 0           # Number of samples evaluated
    
    variance        : Optional[float] = None        # Variance of values
    error_of_mean   : Optional[float] = None        # Error of the mean    
    
    backend_used    : str             = 'unknown'   # Backend used for evaluation
    metadata        : Dict[str, Any]  = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived statistics."""
        self.variance       = self.std ** 2                                  # Variance
        self.error_of_mean  = self.std / np.sqrt(max(self.n_samples - 1, 1)) # Monte Carlo error

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        return {
            'values_shape'  : self.values.shape,
            'n_samples'     : self.n_samples,
            'mean'          : self.mean,
            'std'           : self.std,
            'min'           : self.min_val,
            'max'           : self.max_val,
            'backend'       : self.backend_used,
        }
        
    def __str__(self) -> str:
        ''' As a single line summary '''
        return ','.join([f"{k}={v}" for k, v in self.summary().items()])
    
    def __repr__(self) -> str:
        return self.__str__()

#####################################################################################################
#! BATCH PROCESSOR
#####################################################################################################

class BatchProcessor:
    """
    Handles batch processing with uniform interface across backends.
    
    Strategies:
    - No batching (batch_size=None or len(states) <= batch_size)
    - Automatic batching (split into chunks, process, recombine)
    """
    
    @staticmethod
    def create_batches(data: Array, batch_size: Optional[int]) -> List[Array]:
        """
        Split data into batches.
        
        Parameters:
            data: Array of shape (N, ...)
            batch_size: Size of each batch. If None, return [data]
            
        Returns:
            List of batch arrays
        """
        if batch_size is None or len(data) <= batch_size:
            return [data]
        
        n_batches   = (len(data) + batch_size - 1) // batch_size
        batches     = []
        for i in range(n_batches):
            start   = i * batch_size
            end     = min((i + 1) * batch_size, len(data))
            batches.append(data[start:end])
        return batches
    
    @staticmethod
    def recombine_results(batch_results: List[Array]) -> Array:
        """
        Recombine results from multiple batches.
        
        Parameters:
            batch_results: List of batch result arrays
            
        Returns:
            Concatenated results array
        """
        if len(batch_results) == 1:
            return batch_results[0]
        return np.concatenate(batch_results, axis=0)
    
    @staticmethod
    def compute_statistics(values: Array, return_stats: bool = True) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute statistics on values.
        
        Parameters:
            values: Array of computed values
            return_stats: Whether to compute statistics
            
        Returns:
            Tuple of (mean, std, min, max) or (None, None, None, None) if return_stats=False
        """
        if not return_stats:
            return None, None, None, None
        
        mean        = float(np.mean(values))
        std         = float(np.std(values))
        min_val     = float(np.min(values))
        max_val     = float(np.max(values))
        return mean, std, min_val, max_val

#####################################################################################################
#! BACKEND ABSTRACTION
#####################################################################################################

class EvaluationBackend(ABC):
    """Abstract base class for evaluation backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        
    @abstractmethod
    def evaluate_ansatz(self,
                       ansatz_func  : Callable[[Callable, Array, int, Any], Array],
                       eval_func    : Callable[[Callable, Array, Any, int], Array], # func, states, params, batch_size -> Array [log ansatz values]
                       states       : Array,
                       params       : Any,
                       batch_size   : Optional[int] = None) -> Array:
        """
        Evaluate log-ansatz on states.
        
        Parameters:
        ----------
            ansatz_func: 
                Function (params, states) -> log_ansatz
            states: 
                Array of shape (N, ...)
            params: 
                Network parameters
            batch_size: 
                Optional batch size

        Returns:
            Array of shape (N,) with log-ansatz values
        """
        
    @abstractmethod
    def apply_function(self,
                    func          : Callable[[Array], [Array, float, float]],
                    states        : Array,
                    eval_func     : Callable[[Callable, Array, Any, int], Array], # func, states, params, batch_size -> Array [log ansatz values]
                    params        : Any,
                    probabilities : Optional[Array] = None,
                    batch_size    : Optional[int] = None) -> Array:
        """
        Apply a function to states with optional probability weighting.
        
        Parameters:
        ----------
            func: 
                Function to apply (s) -> value(s)
            states: 
                Array of shape (N, ...)
            ansatz_func: 
                Function (params, s) -> log_ansatz (for gradients/normalization)
            params: 
                Network parameters
            probabilities: 
                Optional probability weights shape (N,)
            batch_size: 
                Optional batch size
            
        Returns:
            Array of computed function values
        """

class JAXBackend(EvaluationBackend):
    """JAX-based evaluation backend with JIT compilation."""
    
    def __init__(self, jit_compile: bool = True):
        self.jit_compile    = jit_compile
        self._cached_funcs  = {}  # Cache for JIT-compiled functions
    
    @property
    def name(self) -> str:
        return 'jax'
    
    def evaluate_ansatz(self,
                       ansatz_func  : Callable[[Callable, Array, int, Any], Array],
                       states       : Array,
                       params       : Any,
                       eval_func    : Optional[Callable]    = None,
                       batch_size   : Optional[int]         = None) -> Array:
        """Evaluate log-ansatz using JAX with optional batching."""
        
        # Fallback to native evaluation if no eval_func provided
        if eval_func is None:
            def single_eval(state):
                return ansatz_func(params, state)

            if self.jit_compile:
                single_eval = jit(single_eval)
            
            eval_func = vmap(single_eval)
            # Handle batching
            if batch_size is None or len(states) <= batch_size:
                return eval_func(states)
            else:
                # Process in batches
                batches = BatchProcessor.create_batches(states, batch_size)
                results = [eval_func(batch) for batch in batches]
                return jnp.concatenate(results, axis=0)
        else:
            return eval_func(func=ansatz_func, data=states, params=params, batch_size=batch_size)
    
    def apply_function(self,
                      func          : Callable,
                      states        : Array,
                      ansatz_func   : Callable,
                      params        : Any,
                      probabilities : Optional[Array] = None,
                      batch_size    : Optional[int] = None) -> Array:
        """Apply function to states using JAX."""
        
        # Define vectorized function application
        def single_apply(state):
            return func(state)
        
        if self.jit_compile:
            single_apply = jit(single_apply)
        
        vmapped_apply = vmap(single_apply)
        
        # Handle batching
        if batch_size is None or len(states) <= batch_size:
            result  = vmapped_apply(states)
        else:
            batches = BatchProcessor.create_batches(states, batch_size)
            results = [vmapped_apply(batch) for batch in batches]
            result  = jnp.concatenate(results, axis=0)
        
        # Apply probability weighting if provided
        if probabilities is not None:
            result = result * probabilities
        
        return result

class NumpyBackend(EvaluationBackend):
    """NumPy-based evaluation backend for CPU-only use."""
    
    @property
    def name(self) -> str:
        return 'numpy'
    
    # ----------------------------------------
    
    def evaluate_ansatz(self,
                    ansatz_func  : Callable,
                    eval_func    : Callable,
                    states       : Array,
                    params       : Any,
                    batch_size   : Optional[int] = None) -> Array:
        """Evaluate log-ansatz using NumPy."""
        
        # Convert states to NumPy if needed
        states_np = np.asarray(states)
        
        if eval_func is not None:
            return eval_func(func=ansatz_func, data=states_np, params=params, batch_size=batch_size)
        
        
        # Handle batching
        if batch_size is None or len(states_np) <= batch_size:
            # Direct evaluation
            return np.array([ansatz_func(params, s) for s in states_np])
        else:
            # Batched evaluation
            batches = BatchProcessor.create_batches(states_np, batch_size)
            results = []
            for batch in batches:
                batch_results = np.array([ansatz_func(params, s) for s in batch])
                results.append(batch_results)
            return np.concatenate(results, axis=0)
    
    def apply_function(self,
                      func              : Callable,
                      states            : Array,
                      ansatz_func       : Callable,
                      params            : Any,
                      probabilities     : Optional[Array] = None,
                      batch_size        : Optional[int] = None) -> Array:
        """Apply function to states using NumPy."""
        
        states_np = np.asarray(states)
        
        # Handle batching
        if batch_size is None or len(states_np) <= batch_size:
            result = np.array([func(s) for s in states_np])
        else:
            batches = BatchProcessor.create_batches(states_np, batch_size)
            results = []
            for batch in batches:
                batch_results = np.array([func(s) for s in batch])
                results.append(batch_results)
            result = np.concatenate(results, axis=0)
        
        # Apply probability weighting if provided
        if probabilities is not None:
            result = result * np.asarray(probabilities)
        
        return result

class AutoBackend(EvaluationBackend):
    """
    Automatic backend dispatcher that selects based on input type.
    
    Strategy:
    - JAX arrays    -> JAXBackend
    - NumPy arrays  -> NumpyBackend
    - Python lists  -> NumpyBackend
    """
    
    def __init__(self):
        self.jax_backend   = JAXBackend()
        self.numpy_backend = NumpyBackend()
    
    @property
    def name(self) -> str:
        return 'auto'
    
    def _select_backend(self, states: Array) -> EvaluationBackend:
        """Select backend based on input type."""
        if isinstance(states, jnp.ndarray):
            return self.jax_backend
        else:
            return self.numpy_backend
    
    def evaluate_ansatz(self,
                       ansatz_func  : Callable,
                       eval_func    : Callable,
                       states       : Array,
                       params       : Any,
                       batch_size   : Optional[int] = None) -> Array:
        """Dispatch to appropriate backend."""
        backend = self._select_backend(states)
        return backend.evaluate_ansatz(ansatz_func, eval_func, states, params, batch_size)
    
    def apply_function(self,
                      func          : Callable,
                      states        : Array,
                      ansatz_func   : Callable,
                      params        : Any,
                      probabilities : Optional[Array] = None,
                      batch_size    : Optional[int] = None) -> Array:
        """Dispatch to appropriate backend."""
        backend = self._select_backend(states)
        return backend.apply_function(func, states, ansatz_func, params, probabilities, batch_size)

#####################################################################################################
#! UNIFIED EVALUATION ENGINE
#####################################################################################################

class UnifiedEvaluationEngine:
    """
    Unified evaluation engine for NQS computations.
    
    This class consolidates all evaluation logic scattered across NQS into a single,
    modular interface. It handles:
    - Ansatz evaluation (log-wavefunction)
    - Arbitrary function application
    - Batch processing
    - Automatic backend dispatch
    
    Example:
        >>> engine = UnifiedEvaluationEngine(nqs, backend='auto')
        >>> result = engine.evaluate_ansatz(states, batch_size=32)
        >>> print(result.mean, result.std)
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None, backend: str = 'default', **kwargs):
        """
        Initialize the evaluation engine.
        
        Parameters:
            config: EvaluationConfig instance. If None, uses defaults.
        """
        self.config = config or EvaluationConfig(backend        =   backend, 
                                                batch_size      =   kwargs.get('batch_size', None),
                                                jit_compile     =   kwargs.get('jit_compile', True),
                                                return_stats    =   kwargs.get('return_stats', True),
                                                validate_inputs =   kwargs.get('validate_inputs', True),
                                                verbose         =   kwargs.get('verbose', False))
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the evaluation backend."""
        if isinstance(self.config.backend, str) and (self.config.backend.lower() == 'jax') or self.config.backend == jax:
            self.backend = JAXBackend(jit_compile=self.config.jit_compile)
        elif isinstance(self.config.backend, str) and (self.config.backend.lower() == 'numpy') or self.config.backend == np:
            self.backend = NumpyBackend()
        else:  # 'auto'
            self.backend = AutoBackend()
    
    # -----------------------
    
    def evaluate_ansatz(self,
                       ansatz_func  : Callable,
                       states       : Array,
                       params       : Any,
                       *,
                       eval_func    : Optional[Callable]    = None,
                       return_stats : bool                  = True
                       ) -> Union[Array, EvaluationResult]:
        """
        Evaluate log-ansatz log|psi(s)| on states.
        
        Parameters:
            ansatz_func: 
                Function (params, state) -> log_ansatz value
            states: 
                Array of shape (N, ...)
            params: 
                Network parameters (pytree)
            
        Returns:
            EvaluationResult with log-ansatz values
        """
        
        if self.config.validate_inputs:
            if len(states) == 0:
                raise ValueError("Cannot evaluate on empty state array")
        
        # Compute ansatz values
        values                      = self.backend.evaluate_ansatz(ansatz_func, eval_func=eval_func, 
                                            states=states, params=params, batch_size=self.config.batch_size)
        mean, std, min_val, max_val = BatchProcessor.compute_statistics(values, return_stats=self.config.return_stats)
        
        if not return_stats:
            return values
        
        return EvaluationResult(
            values          =   values,
            mean            =   mean,
            std             =   std,
            min_val         =   min_val,
            max_val         =   max_val,
            n_samples       =   len(states),
            backend_used    =   self.backend.name,
            metadata        =   {'type': 'ansatz_evaluation'}
        )
    
    def evaluate_function(self,
                        func           : Callable,
                        states         : Array,
                        ansatz_func    : Callable,
                        params         : Any,
                        probabilities  : Optional[Array],
                        *,
                        return_stats   : bool = True
                        ) -> Union[Array, EvaluationResult]:
        """
        Evaluate a function on states.
        
        Parameters:
            func: 
                Function (state) -> value(s)
            states: 
                Array of shape (N, ...)
            ansatz_func: 
                Ansatz function (for reference/context)
            params: 
                Network parameters
            probabilities: 
                Optional probability weights shape (N,)
            
        Returns:
            EvaluationResult with function values
        """
        
        if self.config.validate_inputs:
            if len(states) == 0:
                raise ValueError("Cannot evaluate on empty state array")
        
        # Apply function
        values = self.backend.apply_function(func, 
                            states, ansatz_func, params, probabilities, batch_size=self.config.batch_size)
        
        # Compute statistics
        mean, std, min_val, max_val = BatchProcessor.compute_statistics(values, return_stats=self.config.return_stats)
        
        if return_stats:        
            return EvaluationResult(
                values          =   values,
                mean            =   mean,
                std             =   std,
                min_val         =   min_val,
                max_val         =   max_val,
                n_samples       =   len(states),
                backend_used    =   self.backend.name,
                metadata        =   {'type': 'function_evaluation', 'func': func.__name__ if hasattr(func, '__name__') else 'lambda'}
            )
        
        return values
    
    # ----------------------------------
    
    def set_backend(self, backend: str):
        """
        Change the evaluation backend.
        
        Parameters:
            backend: 'jax', 'numpy', or 'auto'
        """
        if backend not in ['jax', 'numpy', 'auto']:
            raise ValueError(f"Invalid backend '{backend}'")
        self.config.backend = backend
        self._init_backend()
    
    def set_batch_size(self, batch_size: Optional[int]):
        """Set batch size for evaluation."""
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.config.batch_size = batch_size
    
    def get_config(self) -> EvaluationConfig:
        """Get current configuration."""
        return self.config
    
    def get_backend_name(self) -> str:
        """Get name of currently active backend."""
        return self.backend.name

#####################################################################################################
#! FACTORY FUNCTIONS
#####################################################################################################

def create_evaluation_engine(backend        : str = 'auto',
                            batch_size      : Optional[int] = None,
                            jit_compile     : bool          = True,
                            return_stats    : bool          = True,
                            validate_inputs : bool          = True,
                            verbose         : bool          = False
                            
                            ) -> UnifiedEvaluationEngine:
    """
    Factory function to create a configured evaluation engine.
    
    Parameters:
        backend: 
            'jax', 'numpy', or 'auto'
        batch_size: 
            Optional batch size for processing
        jit_compile: 
            Whether to JIT compile (only for JAX backend)
        
    Returns:
        Configured UnifiedEvaluationEngine instance
    """
    config = EvaluationConfig(
        backend         = backend,
        batch_size      = batch_size,
        jit_compile     = jit_compile,
        return_stats    = return_stats,
        validate_inputs = validate_inputs,
        verbose         = verbose
    )
    return UnifiedEvaluationEngine(config)

# ---------------------------------------------------------------------------------------------------
