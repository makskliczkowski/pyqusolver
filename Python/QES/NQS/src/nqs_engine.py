"""
ComputeLocalEnergy: High-level interface for NQS energy and observable computations

This module provides the ComputeLocalEnergy class, which wraps UnifiedEvaluationEngine
and provides NQS-specific functionality like local energy computation, observable
evaluation, and energy statistics.

This consolidates high-level evaluation logic while delegating to UnifiedEvaluationEngine
for low-level backend dispatch.

Architecture:
    ComputeLocalEnergy (NQS-specific interface)
        └─ UnifiedEvaluationEngine (backend abstraction)
            ├─ JAXBackend
            ├─ NumpyBackend
            └─ AutoBackend

Key Methods:
    - compute_local_energy(): 
        Compute H*psi/psi
    - compute_observables(): 
        Evaluate multiple observables
    - compute_statistics(): 
        Get energy statistics
    - evaluate_ansatz(): 
        Evaluate log-wavefunction

------------------------------------------------
File            : QES/NQS/src/nqs_engine.py
Author          : Maksymilian Kliczkowski
Date            : 2025-11-01
License         : MIT
------------------------------------------------
"""

from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Dict, Tuple, Any
from functools import partial
import numpy as np

# -------------------
try:
    if TYPE_CHECKING:
        from ..nqs import NQS
except ImportError:
    pass

try:
    from QES.general_python.algebra.utils import JAX_AVAILABLE, Array
    from .general.nqs_general_engine import UnifiedEvaluationEngine, EvaluationConfig, EvaluationResult, create_evaluation_engine
except:
    raise RuntimeError("QES.general_python.algebra.utils could not be imported. Ensure QES is properly installed.")

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    raise ImportError("JAX is not available. Please install JAX to use this module.")

__all__ = [
    'EnergyStatistics',
    'ObservableResult',
    'ComputeLocalEnergy',
]

#####################################################################################################
#! DATA STRUCTURES
#####################################################################################################

@dataclass
class NQSEnergyResult(EvaluationResult):
    """Statistics about local energy computations."""
    
    @property
    def local_energies(self)                -> Array: return self.values
    @local_energies.setter
    def local_energies(self, v: Array):     self.values = v
    
    @property
    def mean_energy(self)                   -> float: return self.mean
    @mean_energy.setter
    def mean_energy(self, v: float):        self.mean = v
    
    @property
    def std_energy(self)                    -> float: return self.std
    @std_energy.setter
    def std_energy(self, v: float):         self.std = v
    
    @property
    def min_energy(self)                    -> float: return self.min_val
    @min_energy.setter
    def min_energy(self, v: float):         self.min_val = v

    @property
    def max_energy(self)                    -> float: return self.max_val
    @max_energy.setter
    def max_energy(self, v: float):         self.max_val = v
    
    variance            : float = field(init=False)     # Variance of energy
    error_of_mean       : float = field(init=False)     # Error of the mean

    def summary(self) -> Dict[str, float]:
        """Get a summary of energy statistics."""
        return {
            'mean_energy'   : self.mean_energy,
            'std_energy'    : self.std_energy,
            'min_energy'    : self.min_energy,
            'max_energy'    : self.max_energy,
            'variance'      : self.variance,
            'error_of_mean' : self.error_of_mean,
            'n_samples'     : self.n_samples,
        }

@dataclass
class ObservableResult(EvaluationResult):
    """
    Result from observable evaluation.
    
    The observable is defined by a function O(s) that computes the local value
    for a given state configuration s. Namely, given a set of samples {s_i}, we compute:
        - Local values: O(s_i)
        - Expectation value: <O> = (1/N) * sum_i O(s_i)
        - Sampled with respect to |psi(s_i)|^2
    
    This can be used to evaluate arbitrary observables on the NQS wavefunction, such as:
        - Spin correlations
        - Magnetization
        - Custom operators
        - Energy
    """
    
    observable_name : str = ""  # Name of the observable

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the observable evaluation."""
        return {
            'observable_name'       : self.observable_name,
            'expectation'           : self.mean,
            'std'                   : self.std,
            'n_samples'             : self.n_samples,
        }

#####################################################################################################
#! COMPUTE LOCAL ENERGY CLASS
#####################################################################################################

class NQSEvalEngine(UnifiedEvaluationEngine):
    """
    High-level interface for NQS energy and observable computations.
    
    This class provides methods for:
    - Computing local energies 
        E_loc(s) = <s|H|psi>/<s|psi>
    - Evaluating observables 
        <psi|O|psi>
    - Getting energy statistics
    - Evaluating the NQS ansatz log|psi(s)|
    - Efficient batch processing
    
    Example:
        >>> computer        = NQSEvalEngine(nqs, backend='auto', batch_size=32)
        >>> energy_stats    = computer.compute_local_energy(states, ham_action_func)
        >>> print(f"E = {energy_stats.mean_energy:.6f} ± {energy_stats.std_energy:.6f}")
    """
    
    def __init__(self, nqs: 'NQS', backend: str = 'auto',  batch_size: Optional[int] = None, jit_compile: bool = True, **kwargs):
        """
        Initialize ComputeLocalEnergy.
        
        Parameters:
            nqs: 
                The NQS instance to compute with
            backend: 
                'jax', 'numpy', or 'auto' for backend selection
            batch_size: 
                Optional batch size for evaluation
            jit_compile: 
                Whether to JIT compile (JAX backend only)
        """
        super().__init__(backend=backend, batch_size=batch_size, jit_compile=jit_compile)
        self.nqs                = nqs
        self._cached_results    = {}
    
    #####################################################################################################
    #! ANSATZ EVALUATION
    #####################################################################################################
    
    def evaluate_ansatz(self,
                       states       : np.ndarray,
                       params       : Optional[Any] = None,
                       batch_size   : Optional[int] = None) -> EvaluationResult:
        """
        Evaluate the NQS ansatz log|psi(s)| on states.
        
        Comments:
            This consolidates the old _eval_jax, _eval_np, and ansatz methods.
        
        Parameters:
        -----------
            states: 
                Array of shape (N, ...) containing state configurations
            params: 
                Network parameters. If None, uses current NQS parameters
            batch_size: 
                Optional batch size override

        Returns:
            EvaluationResult with ansatz values and statistics
        """
        if params is None:
            params = self.nqs.get_params()
        
        # Set batch size if override provided
        old_batch_size = self.config.batch_size
        if batch_size is not None:
            self.set_batch_size(batch_size)
        
        try:
            result = self.evaluate_ansatz(self.nqs._ansatz_func, states, params)
        finally:
            if batch_size is not None:
                self.set_batch_size(old_batch_size)
        
        return result
    
    #####################################################################################################
    #! LOCAL ENERGY COMPUTATION
    #####################################################################################################
    
    def compute_local_energy(self,
                            states          : Array,
                            ham_action_func : Callable,
                            params          : Optional[Any]         = None,
                            probabilities   : Optional[np.ndarray]  = None,
                            batch_size      : Optional[int]         = None,
                            *,
                            return_stats    : bool                  = True
                            ) -> NQSEnergyResult:
        """
        Compute local energies E_loc(s) = <s|H|psi>/<s|psi>.
        
        This consolidates the old _single_step_groundstate and energy evaluation logic.
        
        Parameters:
            states: 
                Array of configurations shape (N, ...)
            ham_action_func: 
                Function (state) -> E_loc(state) or callable with signature
                that takes state and returns Hamiltonian matrix elements
            params: 
                Network parameters. If None, uses current NQS parameters
            probabilities: 
                Optional probability weights for importance sampling
            batch_size: 
                Optional batch size override

        Returns:
            EnergyStatistics object with energy values and statistics
        """
        if params is None:
            params = self.nqs.get_params()
        
        # Set batch size if override provided
        old_batch_size = self.config.batch_size
        if batch_size is not None:
            self.set_batch_size(batch_size)
        
        try:
            # Compute local energies using the engine
            result = self.evaluate_function(
                        ham_action_func,
                        states,
                        self.nqs._ansatz_func,
                        params,
                        probabilities=probabilities
                    )
                    
            # Convert to EnergyStatistics
            local_energies = np.asarray(result.values)
            
            if not return_stats:
                return local_energies
            
            return result
            
        finally:
            # Restore original batch size
            if batch_size is not None:
                self.set_batch_size(old_batch_size)
    
    #####################################################################################################
    #! OBSERVABLE EVALUATION
    #####################################################################################################
    
    def compute_observable(self,
                          observable_func       : Callable,
                          states                : np.ndarray,
                          observable_name       : str                   = "Observable",
                          params                : Optional[Any]         = None,
                          batch_size            : Optional[int]         = None,
                          probabilities         : Optional[np.ndarray]  = None,
                          *,
                          return_stats          : bool                  = False
                          ) -> Union[Array, ObservableResult]:
        """
        Evaluate an observable O on states.
        
        This consolidates eval_observables and operator evaluation logic.
        
        Parameters:
            observable_func: 
                Function (state) -> observable_value
            states: 
                Array of configurations shape (N, ...)
            observable_name: 
                Name of the observable for documentation
            params: 
                Network parameters. If None, uses current NQS parameters
            compute_expectation: 
                If True, compute <psi|O|psi> (requires specialized impl)
            batch_size: 
                Optional batch size override

        Returns:
            ObservableResult with local values and statistics
        """
        if params is None:
            params = self.nqs.get_params()
        
        # Set batch size if override provided
        old_batch_size = self.config.batch_size
        if batch_size is not None:
            self.set_batch_size(batch_size)
        
        try:
            result = super().evaluate_function(
                observable_func,
                states,
                self.nqs._ansatz_func,
                params,
                probabilities=probabilities
            )
            
            local_values = np.asarray(result.values)
            
            if not return_stats:
                return local_values
            
            return ObservableResult(
                observable_name     = observable_name,
                values              = local_values,
                mean                = float(result.mean),
                std                 = float(result.std),
                min_val             = float(result.min_val),
                max_val             = float(result.max_val),
                n_samples           = result.n_samples,
                backend_used        = result.backend_used,
                metadata            = result.metadata,
            )
        finally:
            # Restore original batch size
            if batch_size is not None:
                self.set_batch_size(old_batch_size)
    
    def compute_observables(self,
                            observable_funcs     : Dict[str, Callable],
                            states               : np.ndarray,
                            params               : Optional[Any] = None,
                            batch_size           : Optional[int] = None,
                            *,
                            return_stats        : bool = True
                            ) -> Dict[str, ObservableResult]:
        """
        Evaluate multiple observables efficiently.
        
        Parameters:
        ------------
            observable_funcs:
                Dictionary mapping observable name to function
            states: 
                Array of configurations shape (N, ...)
            params: 
                Network parameters
            batch_size: 
                Optional batch size override
            return_stats:             
                If True, return detailed statistics

        Returns:
            Dictionary mapping observable name to ObservableResult
        """
        results = {}
        for name, func in observable_funcs.items():
            results[name] = self.compute_observable(
                func, states, name, params, batch_size=batch_size,
                return_stats=return_stats
            )
        return results
    
    #####################################################################################################
    #! BATCH FUNCTION EVALUATION
    #####################################################################################################
    
    def evaluate_function(self,
                         func           : Callable,
                         states         : Optional[Array]       = None,
                         probabilities  : Optional[Array]       = None,
                         params         : Optional[Any]         = None,
                         batch_size     : Optional[int]         = None) -> EvaluationResult:
        """
        General-purpose function evaluation on state batches.
        
        This consolidates the old _apply_fun* methods.
        
        Parameters:
            func: 
                Function (state) -> value
            states:
                Array of configurations shape (N, ...)
            params: 
                Network parameters (for context)
            probabilities: 
                Optional probability weights
            batch_size: 
                Optional batch size override

        Returns:
            EvaluationResult with computed values
        """
        if params is None:
            params = self.nqs.get_params()
            
        if states is None or len(states) == 0:
            (last_configs, last_ansatze), (states, all_ansatze), (all_probabilities) = self.nqs.sample()
        
        # Set batch size if override provided
        old_batch_size = self.config.batch_size
        if batch_size is not None:
            self.set_batch_size(batch_size)
        
        try:
            return super().evaluate_function(func, states, self.nqs._ansatz_func, params, probabilities=probabilities)
        finally:
            if batch_size is not None:
                self.set_batch_size(old_batch_size)
    
    #####################################################################################################
    #! CONFIGURATION MANAGEMENT
    #####################################################################################################
    
    def set_backend(self, backend: str):
        """
        Change the evaluation backend.
        
        Parameters:
            backend: 'jax', 'numpy', or 'auto'
        """
        super().set_backend(backend)
    
    def set_batch_size(self, batch_size: Optional[int]):
        """Set batch size for evaluation."""
        super().set_batch_size(batch_size)
    
    def get_config(self) -> EvaluationConfig:
        """Get current evaluation configuration."""
        return super().get_config()
    
    def get_backend_name(self) -> str:
        """Get name of currently active backend."""
        return super().get_backend_name()
    
    #####################################################################################################
    #! UTILITIES
    #####################################################################################################
    
    def clear_cache(self):
        """Clear cached computation results."""
        self._cached_results.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the compute engine configuration."""
        config = self.get_config()
        return {
            'backend'           : config.backend,
            'actual_backend'    : self.get_backend_name(),
            'batch_size'        : config.batch_size,
            'jit_compile'       : config.jit_compile,
            'cached_results'    : len(self._cached_results),
        }

#####################################################################################################
#! FACTORY FUNCTION
#####################################################################################################

def create_compute_local_energy(nqs         : 'NQS',
                               backend      : str = 'auto',
                               batch_size   : Optional[int] = None) -> NQSEvalEngine:
    """
    Factory function to create a ComputeLocalEnergy instance.
    
    Parameters:
        nqs: 
            The NQS instance
        backend: 
            Backend selection ('jax', 'numpy', or 'auto')
        batch_size: 
            Optional batch size for computations
        
    Returns:
        Configured NQSEvalEngine instance
    """
    return NQSEvalEngine(nqs, backend=backend, batch_size=batch_size)

# ------------------------------------------------------------------------------------
#! END OF FILE
# ------------------------------------------------------------------------------------
