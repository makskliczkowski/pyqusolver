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
    - compute_local_energy(): Compute H*ψ/ψ
    - compute_observables(): Evaluate multiple observables
    - compute_statistics(): Get energy statistics
    - evaluate_ansatz(): Evaluate log-wavefunction

Author: Automated Session 5 Task 5.3
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Dict, Tuple, Any
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .unified_evaluation_engine import (
    UnifiedEvaluationEngine,
    EvaluationConfig,
    EvaluationResult,
    create_evaluation_engine,
)

__all__ = [
    'EnergyStatistics',
    'ObservableResult',
    'ComputeLocalEnergy',
]

#####################################################################################################
#! DATA STRUCTURES
#####################################################################################################

@dataclass
class EnergyStatistics:
    """Statistics about local energy computations."""
    
    local_energies: np.ndarray  # Local energy for each sample
    mean_energy: float
    std_energy: float
    min_energy: float
    max_energy: float
    n_samples: int
    variance: float = field(init=False)
    error_of_mean: float = field(init=False)
    
    def __post_init__(self):
        """Compute derived statistics."""
        self.variance = self.std_energy ** 2
        self.error_of_mean = self.std_energy / np.sqrt(max(self.n_samples - 1, 1))
    
    def summary(self) -> Dict[str, float]:
        """Get a summary of energy statistics."""
        return {
            'mean_energy': self.mean_energy,
            'std_energy': self.std_energy,
            'min_energy': self.min_energy,
            'max_energy': self.max_energy,
            'variance': self.variance,
            'error_of_mean': self.error_of_mean,
            'n_samples': self.n_samples,
        }


@dataclass
class ObservableResult:
    """Result from observable evaluation."""
    
    observable_name: str
    expectation_value: Optional[float]  # <ψ|O|ψ>
    local_values: np.ndarray  # Local values O(s) for each sample
    mean_local_value: float
    std_local_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the observable result."""
        return {
            'observable': self.observable_name,
            'expectation_value': self.expectation_value,
            'mean_local_value': self.mean_local_value,
            'std_local_value': self.std_local_value,
            'n_samples': len(self.local_values),
        }

#####################################################################################################
#! COMPUTE LOCAL ENERGY CLASS
#####################################################################################################

class ComputeLocalEnergy:
    """
    High-level interface for NQS energy and observable computations.
    
    This class provides methods for:
    - Computing local energies E_loc(s) = <s|H|ψ>/<s|ψ>
    - Evaluating observables <ψ|O|ψ>
    - Getting energy statistics
    - Efficient batch processing
    
    It consolidates logic that was scattered across 19+ methods in nqs.py
    and provides a unified API that's easier to test and maintain.
    
    Example:
        >>> computer = ComputeLocalEnergy(nqs, backend='auto', batch_size=32)
        >>> energy_stats = computer.compute_local_energy(states, ham_action_func)
        >>> print(f"E = {energy_stats.mean_energy:.6f} ± {energy_stats.std_energy:.6f}")
    """
    
    def __init__(self, nqs: 'NQS', backend: str = 'auto', 
                 batch_size: Optional[int] = None, jit_compile: bool = True):
        """
        Initialize ComputeLocalEnergy.
        
        Parameters:
            nqs: The NQS instance to compute with
            backend: 'jax', 'numpy', or 'auto' for backend selection
            batch_size: Optional batch size for evaluation
            jit_compile: Whether to JIT compile (JAX backend only)
        """
        self.nqs = nqs
        self.engine = create_evaluation_engine(
            backend=backend,
            batch_size=batch_size,
            jit_compile=jit_compile
        )
        self._cached_results = {}
    
    #####################################################################################################
    #! ANSATZ EVALUATION
    #####################################################################################################
    
    def evaluate_ansatz(self,
                       states: np.ndarray,
                       params: Optional[Any] = None,
                       batch_size: Optional[int] = None) -> EvaluationResult:
        """
        Evaluate the NQS ansatz log|ψ(s)| on states.
        
        This consolidates the old _eval_jax, _eval_np, and ansatz methods.
        
        Parameters:
            states: Array of shape (N, ...) containing state configurations
            params: Network parameters. If None, uses current NQS parameters
            batch_size: Optional batch size override
            
        Returns:
            EvaluationResult with ansatz values and statistics
        """
        if params is None:
            params = self.nqs.get_params()
        
        # Set batch size if override provided
        old_batch_size = self.engine.config.batch_size
        if batch_size is not None:
            self.engine.set_batch_size(batch_size)
        
        try:
            result = self.engine.evaluate_ansatz(
                self.nqs._ansatz_func,
                states,
                params
            )
        finally:
            # Restore original batch size
            if batch_size is not None:
                self.engine.set_batch_size(old_batch_size)
        
        return result
    
    #####################################################################################################
    #! LOCAL ENERGY COMPUTATION
    #####################################################################################################
    
    def compute_local_energy(self,
                            states: np.ndarray,
                            ham_action_func: Callable,
                            params: Optional[Any] = None,
                            probabilities: Optional[np.ndarray] = None,
                            batch_size: Optional[int] = None) -> EnergyStatistics:
        """
        Compute local energies E_loc(s) = <s|H|ψ>/<s|ψ>.
        
        This consolidates the old _single_step_groundstate and energy evaluation logic.
        
        Parameters:
            states: Array of configurations shape (N, ...)
            ham_action_func: Function (state) -> E_loc(state) or callable with signature
                            that takes state and returns Hamiltonian matrix elements
            params: Network parameters. If None, uses current NQS parameters
            probabilities: Optional probability weights for importance sampling
            batch_size: Optional batch size override
            
        Returns:
            EnergyStatistics object with energy values and statistics
        """
        if params is None:
            params = self.nqs.get_params()
        
        # Set batch size if override provided
        old_batch_size = self.engine.config.batch_size
        if batch_size is not None:
            self.engine.set_batch_size(batch_size)
        
        try:
            # Compute local energies using the engine
            result = self.engine.evaluate_function(
                ham_action_func,
                states,
                self.nqs._ansatz_func,
                params,
                probabilities=probabilities
            )
            
            # Convert to EnergyStatistics
            local_energies = np.asarray(result.values)
            
            return EnergyStatistics(
                local_energies=local_energies,
                mean_energy=float(result.mean),
                std_energy=float(result.std),
                min_energy=float(result.min_val),
                max_energy=float(result.max_val),
                n_samples=result.n_samples
            )
        finally:
            # Restore original batch size
            if batch_size is not None:
                self.engine.set_batch_size(old_batch_size)
    
    #####################################################################################################
    #! OBSERVABLE EVALUATION
    #####################################################################################################
    
    def compute_observable(self,
                          observable_func: Callable,
                          states: np.ndarray,
                          observable_name: str = "Observable",
                          params: Optional[Any] = None,
                          compute_expectation: bool = False,
                          batch_size: Optional[int] = None) -> ObservableResult:
        """
        Evaluate an observable O on states.
        
        This consolidates eval_observables and operator evaluation logic.
        
        Parameters:
            observable_func: Function (state) -> observable_value
            states: Array of configurations shape (N, ...)
            observable_name: Name of the observable for documentation
            params: Network parameters. If None, uses current NQS parameters
            compute_expectation: If True, compute <ψ|O|ψ> (requires specialized impl)
            batch_size: Optional batch size override
            
        Returns:
            ObservableResult with local values and statistics
        """
        if params is None:
            params = self.nqs.get_params()
        
        # Set batch size if override provided
        old_batch_size = self.engine.config.batch_size
        if batch_size is not None:
            self.engine.set_batch_size(batch_size)
        
        try:
            result = self.engine.evaluate_function(
                observable_func,
                states,
                self.nqs._ansatz_func,
                params,
                probabilities=None
            )
            
            local_values = np.asarray(result.values)
            
            return ObservableResult(
                observable_name=observable_name,
                expectation_value=float(result.mean) if compute_expectation else None,
                local_values=local_values,
                mean_local_value=float(result.mean),
                std_local_value=float(result.std),
                metadata={
                    'n_samples': result.n_samples,
                    'backend_used': result.backend_used,
                }
            )
        finally:
            # Restore original batch size
            if batch_size is not None:
                self.engine.set_batch_size(old_batch_size)
    
    def compute_observables(self,
                           observable_funcs: Dict[str, Callable],
                           states: np.ndarray,
                           params: Optional[Any] = None,
                           batch_size: Optional[int] = None) -> Dict[str, ObservableResult]:
        """
        Evaluate multiple observables efficiently.
        
        Parameters:
            observable_funcs: Dictionary mapping observable name to function
            states: Array of configurations shape (N, ...)
            params: Network parameters
            batch_size: Optional batch size override
            
        Returns:
            Dictionary mapping observable name to ObservableResult
        """
        results = {}
        for name, func in observable_funcs.items():
            results[name] = self.compute_observable(
                func, states, name, params, 
                compute_expectation=True, batch_size=batch_size
            )
        return results
    
    #####################################################################################################
    #! BATCH FUNCTION EVALUATION
    #####################################################################################################
    
    def evaluate_function(self,
                         func: Callable,
                         states: np.ndarray,
                         params: Optional[Any] = None,
                         probabilities: Optional[np.ndarray] = None,
                         batch_size: Optional[int] = None) -> EvaluationResult:
        """
        General-purpose function evaluation on state batches.
        
        This consolidates the old _apply_fun* methods.
        
        Parameters:
            func: Function (state) -> value
            states: Array of configurations shape (N, ...)
            params: Network parameters (for context)
            probabilities: Optional probability weights
            batch_size: Optional batch size override
            
        Returns:
            EvaluationResult with computed values
        """
        if params is None:
            params = self.nqs.get_params()
        
        # Set batch size if override provided
        old_batch_size = self.engine.config.batch_size
        if batch_size is not None:
            self.engine.set_batch_size(batch_size)
        
        try:
            return self.engine.evaluate_function(
                func, states, self.nqs._ansatz_func, params,
                probabilities=probabilities
            )
        finally:
            # Restore original batch size
            if batch_size is not None:
                self.engine.set_batch_size(old_batch_size)
    
    #####################################################################################################
    #! CONFIGURATION MANAGEMENT
    #####################################################################################################
    
    def set_backend(self, backend: str):
        """
        Change the evaluation backend.
        
        Parameters:
            backend: 'jax', 'numpy', or 'auto'
        """
        self.engine.set_backend(backend)
    
    def set_batch_size(self, batch_size: Optional[int]):
        """Set batch size for evaluation."""
        self.engine.set_batch_size(batch_size)
    
    def get_config(self) -> EvaluationConfig:
        """Get current evaluation configuration."""
        return self.engine.get_config()
    
    def get_backend_name(self) -> str:
        """Get name of currently active backend."""
        return self.engine.get_backend_name()
    
    #####################################################################################################
    #! UTILITIES
    #####################################################################################################
    
    def clear_cache(self):
        """Clear cached computation results."""
        self._cached_results.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the compute engine configuration."""
        config = self.engine.get_config()
        return {
            'backend': config.backend,
            'actual_backend': self.get_backend_name(),
            'batch_size': config.batch_size,
            'jit_compile': config.jit_compile,
            'cached_results': len(self._cached_results),
        }


#####################################################################################################
#! FACTORY FUNCTION
#####################################################################################################

def create_compute_local_energy(nqs: 'NQS',
                               backend: str = 'auto',
                               batch_size: Optional[int] = None) -> ComputeLocalEnergy:
    """
    Factory function to create a ComputeLocalEnergy instance.
    
    Parameters:
        nqs: The NQS instance
        backend: Backend selection ('jax', 'numpy', or 'auto')
        batch_size: Optional batch size for computations
        
    Returns:
        Configured ComputeLocalEnergy instance
    """
    return ComputeLocalEnergy(nqs, backend=backend, batch_size=batch_size)
