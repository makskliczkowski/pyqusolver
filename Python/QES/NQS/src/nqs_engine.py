"""
ComputeLocalEnergy: High-level interface for NQS energy and observable computations

This module provides the ComputeLocalEnergy class, which wraps UnifiedEvaluationEngine
and provides NQS-specific functionality like local energy computation, observable
evaluation, and energy statistics.

This consolidates high-level evaluation logic while delegating to UnifiedEvaluationEngine
for low-level backend dispatch.

Architecture:
- NQSEvalEngine     : High-level interface for NQS computations
- EvaluationResult  : Data structure for evaluation results and statistics

------------------------------------------------
File            : QES/NQS/src/nqs_engine.py
Author          : Maksymilian Kliczkowski
Date            : 2025-11-01
License         : MIT
------------------------------------------------
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Dict, TypeAlias, Any
from functools import partial
from unittest import result
import numpy as np

# -------------------
try:
    if TYPE_CHECKING:
        from ..nqs import NQS, EvalFunctionType, AnsatzFunctionType, CallableFunctionType
except ImportError:
    raise ImportError("NQS module not found. Ensure that the QES/NQS package is properly installed.")

# -------------------

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax             = None
    jnp             = None
    JAX_AVAILABLE   = False

__all__ = [
    'EvaluationResult',
    'NQSLoss',
    'NQSObservable',
    'NQSEvalEngine',
]

#! Type Aliases
if JAX_AVAILABLE and jnp:
    Array       : TypeAlias                 = Union[np.ndarray, jnp.ndarray]
    PRNGKey     : TypeAlias                 = Any # jax.random.PRNGKeyArray
    JaxDevice   : TypeAlias                 = Any # Placeholder for jax device type
else:
    Array       : TypeAlias                 = np.ndarray
    PRNGKey     : TypeAlias                 = None
    JaxDevice   : TypeAlias                 = None

#####################################################################################################
#! CONFIGURATION AND RESULT DATACLASSES
#####################################################################################################

@dataclass
class EvaluationResult:
    """Result of an evaluation operation."""
    
    values          : Array                                 # Computed values array
    has_stats       : bool            = field(init=True)    # Whether statistics are computed
    mean            : Optional[float] = None                # Mean of values
    std             : Optional[float] = None                # Standard deviation of values
    min_val         : Optional[float] = None                # Minimum value
    max_val         : Optional[float] = None                # Maximum value
    n_samples       : int             = 0                   # Number of samples evaluated
    
    variance        : Optional[float] = None                # Variance of values
    error_of_mean   : Optional[float] = None                # Error of the mean    
    
    backend_used    : str             = 'unknown'           # Backend used for evaluation
    metadata        : Dict[str, Any]  = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived statistics."""
        
        self.n_samples              = len(self.values)        

        if not self.has_stats:
            return
        
        if self.mean is None:       self.mean       = np.mean(self.values)
        if self.std is None:        self.std        = np.std(self.values)
        if self.min_val is None:    self.min_val    = np.min(self.values)
        if self.max_val is None:    self.max_val    = np.max(self.values)
        
        if self.variance is None:   
            self.variance   = self.std ** 2
        
        if self.error_of_mean is None:
            self.error_of_mean      = self.std / np.sqrt(max(self.n_samples - 1, 1))
        
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        if self.has_stats:
            return {
                'values_shape'  : self.values.shape,
                'n_samples'     : self.n_samples,
                'mean'          : self.mean,
                'std'           : self.std,
                'min'           : self.min_val,
                'max'           : self.max_val,
                'backend'       : self.backend_used,
            }
        return {
            'values_shape'  : self.values.shape,
            'n_samples'     : self.n_samples,
            'backend'       : self.backend_used,
        }
    
    def stats(self):
        ''' Calculate statistics if not already present '''
        if not self.has_stats:
            self.mean           = np.mean(self.values)
            self.std            = np.std(self.values)
            self.min_val        = np.min(self.values)
            self.max_val        = np.max(self.values)
            self.variance       = self.std ** 2
            self.error_of_mean  = self.std / np.sqrt(max(self.n_samples - 1, 1))
            self.has_stats      = True
        return (self.mean, self.std, self.min_val, self.max_val)
    
    def __str__(self) -> str:
        ''' As a single line summary '''
        return ','.join([f"{k}={v}" for k, v in self.summary().items()])
    
    def __repr__(self) -> str:
        return self.__str__()

#####################################################################################################
#! DATA STRUCTURES
#####################################################################################################

@dataclass
class NQSLoss(EvaluationResult):
    """Statistics about local energy computations."""
    
    @property
    def local_loss(self)                -> Array: return self.values
    @local_loss.setter
    def local_loss(self, v: Array):     self.values = v
    
    @property
    def mean_loss(self)                 -> float: return self.mean
    @mean_loss.setter
    def mean_loss(self, v: float):      self.mean = v
    
    @property
    def std_loss(self)                  -> float: return self.std
    @std_loss.setter
    def std_loss(self, v: float):       self.std = v
    
    @property
    def min_loss(self)                  -> float: return self.min_val
    @min_loss.setter
    def min_loss(self, v: float):       self.min_val = v

    @property
    def max_loss(self)                  -> float: return self.max_val
    @max_loss.setter
    def max_loss(self, v: float):       self.max_val = v
    
    variance                            : float = field(init=False)     # Variance of energy
    error_of_mean                       : float = field(init=False)     # Error of the mean

    def summary(self) -> Dict[str, float]:
        """Get a summary of loss statistics."""
        if not self.has_stats:
            return {
                'name'      : 'Local Loss',
                'n_samples' : self.n_samples,
            }
        return {
            'name'          : 'Local Loss',
            'mean'          : self.mean_loss,
            'std'           : self.std_loss,
            'variance'      : self.variance,
            'min'           : self.min_loss,
            'max'           : self.max_loss,
            'error_of_mean' : self.error_of_mean,
            'n_samples'     : self.n_samples,
        }
        
    def __str__(self) -> str:
        ''' As a single line summary '''
        return ','.join([f"{k}={v}" for k, v in self.summary().items()])
    
    def __repr__(self) -> str:
        return self.__str__()

@dataclass
class NQSObservable(EvaluationResult):
    """
    Result from observable evaluation. This computes the expectation value
    of an observable O with respect to the NQS. It means that we evaluate
    local values O(s) on sampled states s from |psi|^2 distribution and 
    average them over samples.
    
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
    
    observable_name : str           = ""  # Name of the observable
    observable_num  : int           = 1   # Number of observables evaluated
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the observable evaluation."""
        return {
            'name'                  : self.observable_name,
            'expectation'           : self.mean,
            'std'                   : self.std,
            'min'                   : self.min_val,
            'max'                   : self.max_val,
            'variance'              : self.variance,
            'error_of_mean'         : self.error_of_mean,
            'n_samples'             : self.n_samples,
        }

    def stats(self):
        ''' Calculate statistics if not already present '''
        if not self.has_stats:
            if self.observable_num == 1:
                self.mean           = np.mean(self.values)
                self.std            = np.std(self.values)
                self.min_val        = np.min(self.values)
                self.max_val        = np.max(self.values)
                self.variance       = self.std ** 2
                self.error_of_mean  = self.std / np.sqrt(max(self.n_samples - 1, 1))
                self.has_stats      = True
            else:
                self.mean           = [np.mean(v) for v in self.values]
                self.std            = [np.std(v) for v in self.values]
                self.min_val        = [np.min(v) for v in self.values]
                self.max_val        = [np.max(v) for v in self.values]
                self.variance       = [s ** 2 for s in self.std]
                self.error_of_mean  = [s / np.sqrt(max(self.n_samples - 1, 1)) for s in self.std]
                self.has_stats      = True
                
        return (self.mean, self.std, self.min_val, self.max_val)

#####################################################################################################
#! COMPUTE LOCAL ENERGY CLASS
#####################################################################################################

class NQSEvalEngine:
    """
    High-level interface for NQS energy and observable computations.
    
    This class provides methods for:
    - Compute loss functions, e.g.,:
        - Computing local energies 
            E_loc(s) = <s|H|psi>/<s|psi>
    - Evaluating observables, e.g.,  
        - Computing local observables 
            <O> = <psi|O|psi>
    - Getting statistics
    - Evaluating the NQS ansatz, e.g.,
        - log|psi(s)| for Wavefunction ansatz psi
    Features:
        - Efficient batch processing
        - Backend abstraction (JAX, NumPy)
    
    Example:
        >>> computer        = NQSEvalEngine(nqs, backend='auto', batch_size=32)
        >>> energy_stats    = computer.compute_local_energy(states, ham_action_func)
        >>> print(f"E = {energy_stats.mean_energy:.6f} ± {energy_stats.std_energy:.6f}")
    """
    
    def __init__(self, 
                nqs             : 'NQS', 
                batch_size      : Optional[int] = None, 
                evaluate_stats  : bool          = True,
                **kwargs):
        """
        Initialize ComputeLocalEnergy.
        
        Parameters:
            nqs: NQS
                The NQS instance to compute with. This will provide the ansatz function
                and parameters.
            batch_size: 
                Optional batch size for evaluation. Otherwise, batch size from NQS will be used.
            evaluate_stats: bool
                Whether to compute and return statistics
        """
        if nqs is None:
            raise ValueError("NQS instance must be provided to NQSEvalEngine")
    
        self.nqs                = nqs
        
        if batch_size is not None:
            self.set_batch_size(batch_size)
        else:
            self.batch_size     = nqs.batch_size
            
        # Info about whether to compute statistics    
        self.evaluate_stats     = evaluate_stats
        
        # Cache for results
        self._cached_results    = {}
    
    #####################################################################################################
    #! ANSATZ EVALUATION
    #####################################################################################################
    
    def ansatz(self,
            states          : Optional[Array]               = None,
            *,
            params          : Optional[Any]                 = None,
            batch_size      : Optional[int]                 = None,
            num_samples     : Optional[int]                 = None,
            num_chains      : Optional[int]                 = None,
            return_stats    : Optional[bool]                = None,
            ansatz_func     : Optional['AnsatzFunctionType']= None,
            return_values   : bool                          = False
            ) -> EvaluationResult:
        """
        Evaluate the NQS ansatz log|psi(s)| on states.
        This function takes the ansatz function from the NQS instance, it's
        parameters, and evaluates it on the provided states.
        
        Parameters:
        -----------
            states: 
                Array of shape (N, shape) to evaluate the ansatz on. If None, samples will be drawn.
                N is the number of samples and shape is the configuration shape.
            params: 
                Network parameters. If None, uses current NQS parameters
            num_samples: 
                Number of samples to draw if states is None
            num_chains: 
                Number of Markov chains to use when sampling if states is None
            batch_size: 
                Optional batch size override
            return_stats: 
                Whether to compute and return statistics
            return_values: 
                Whether to return raw values instead of EvaluationResult

        Returns:
            EvaluationResult with ansatz values and statistics
        """
        if params is not None:
            self.nqs.set_params(params)
            params = self.nqs.get_params()
        
        if return_stats is None:    return_stats    = self.evaluate_stats
        if ansatz_func is None:     ansatz_func     = self.nqs.__call__
            
        # Set batch size if override provided
        if batch_size is not None or self.batch_size != batch_size:
            self.set_batch_size(batch_size)

        if states is None or len(states) == 0:
            (_, _), (states, ansatze), (_) = self.nqs.sample(num_samples=num_samples, num_chains=num_chains)  
            
            if return_values:
                return ansatze
            
            self._cached_results['ansatz'] = ansatze
            return EvaluationResult(values=ansatze, has_stats=return_stats, backend_used=self.nqs.backend_str)
        
        # Evaluate using NQS
        vals = ansatz_func(states)
        if return_values:
            return vals
        return EvaluationResult(values=vals, has_stats=return_stats, backend_used=self.nqs.backend_str)
    
    #####################################################################################################
    #! LOCAL ENERGY COMPUTATION
    #####################################################################################################
    
    def loss(self,
            states          : Optional[Array]                       = None,
            ansatze         : Optional[Array]                       = None,
            *,
            action_func     : Optional['CallableFunctionType']      = None, # can be replaced if we want to evaluate different model
            params          : Optional[Any]                         = None,
            probabilities   : Optional[Array]                       = None,
            batch_size      : Optional[int]                         = None,
            num_samples     : Optional[int]                         = None,
            num_chains      : Optional[int]                         = None,
            return_stats    : Optional[bool]                        = None,
            return_values   : bool                                  = False
            ) -> NQSLoss:
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
        if params is None:          params          = self.nqs.get_params()
        if return_stats is None:    return_stats    = self.evaluate_stats
        if action_func is None:     action_func     = self.nqs.loss_function
        
        # Set batch size if override provided
        if batch_size is not None or self.batch_size != batch_size:
            self.set_batch_size(batch_size)
        
        try:
            output, m, std = self.nqs.apply(
                                functions       = action_func,
                                states_and_psi  = (states, ansatze),
                                probabilities   = probabilities,
                                batch_size      = self.batch_size,
                                parameters      = params,
                                num_samples     = num_samples,
                                num_chains      = num_chains,
                                return_values   = True,
                            )
                                    
            # Convert to EnergyStatistics
            local_losses                            = np.array(output)
            self._cached_results['losses/val']      = local_losses
            self._cached_results['losses/mean']     = m
            self._cached_results['losses/std']      = std
            
            if return_values:
                return local_losses
            return  NQSLoss(values = local_losses, has_stats = return_stats, backend_used = self.nqs.backend_str,)
        
        except Exception as e:
            raise RuntimeError(f"Error computing loss: {e}")
    
    #####################################################################################################
    #! OBSERVABLE EVALUATION
    #####################################################################################################
    
    def observable(self,
            states          : Optional['Array']                                             = None,
            ansatze         : Optional['Array']                                             = None,
            functions       : Union['CallableFunctionType', List['CallableFunctionType']]   = None,
            names           : Optional[Union[str, List[str]]]                               = None,
            *,
            params          : Optional[Any]                                                 = None,
            probabilities   : Optional[Array]                                               = None,
            batch_size      : Optional[int]                                                 = None,
            num_samples     : Optional[int]                                                 = None,
            num_chains      : Optional[int]                                                 = None,
            return_stats    : Optional[bool]                                                = None,
            return_values   : bool                                                          = False,
            log_progress    : bool                                                          = False,
            args            : Optional[tuple]                                               = None
        ) -> Union[Array, NQSObservable]:
        """
        Evaluate observables O(s) on sampled states s from |psi|^mu distribution.
        This computes local observable values O(s) for each sampled state s,
        and averages them to get expectation values.
        
        Parameters:
            states: 
                Array of configurations shape (N, ...)
            functions: 
                Single function or list of functions to evaluate observables
            names: 
                Optional name or list of names for observables
            params: 
                Network parameters. If None, uses current NQS parameters
            probabilities: 
                Optional probability weights for importance sampling
            batch_size: 
                Optional batch size override
            args:
                Optional tuple of additional arguments to pass to the observable function.
                These are broadcast (not vmapped) across all sampled states.
                Useful for computing correlation functions <O_i O_j> where (i, j) are
                passed as runtime arguments, avoiding N² recompilations.
        Returns:
            NQSObservable or array of NQSObservable with observable values and statistics
        """
        
        if params is None:          params          = self.nqs.get_params()
        if return_stats is None:    return_stats    = self.evaluate_stats
        if functions is None or isinstance(functions, (list, tuple)) and len(functions) == 0:
            raise ValueError("At least one observable function must be provided")
        
        # Set batch size if override provided
        if batch_size is not None or self.batch_size != batch_size:
            self.set_batch_size(batch_size)
            
        single_function = False
        if not isinstance(functions, (list, tuple)):
            functions       = [functions]
            single_function = True
            
        try:
            output = self.nqs.apply(
                        functions       = functions,
                        states_and_psi  = (states, ansatze),
                        probabilities   = probabilities,
                        batch_size      = self.batch_size,
                        parameters      = params,
                        num_samples     = num_samples,
                        num_chains      = num_chains,
                        return_values   = True,
                        log_progress    = log_progress,
                        args            = args
                    )
                            
            if not single_function:
                values      = [np.array(v[0]) for v in output]  # Extract values from output
                means       = [v[1] for v in output]
                stds        = [v[2] for v in output]
            else:
                values      = [np.array(output[0])]
                means       = [output[1]]
                stds        = [output[2]]
                            
            # Convert to NQSObservable
            observables = []
            for idx, vals in enumerate(values):
                name        = names[idx] if names is not None and isinstance(names, (list, tuple)) else (names if isinstance(names, str) else f"O_{idx}")
                obs_result  = NQSObservable(values = np.array(vals), has_stats = return_stats, backend_used = self.nqs.backend_str, observable_name = name)
                observables.append(obs_result)
            
            self._cached_results['observables/values']  = values
            self._cached_results['observables/means']   = means
            self._cached_results['observables/stds']    = stds
            
            if return_values:
                return values
            
            return observables[0] if single_function else observables
        
        except Exception as e:
            raise RuntimeError(f"Error computing observables: {e}")
        
    #####################################################################################################
    #! CONFIGURATION MANAGEMENT
    #####################################################################################################
    
    def set_batch_size(self, batch_size: Optional[int]):
        """Set batch size for evaluation."""
        self.batch_size     = batch_size
        self.nqs.batch_size = batch_size
            
    #####################################################################################################
    #! UTILITIES
    #####################################################################################################
    
    def clear_cache(self):
        """Clear cached computation results."""
        self._cached_results.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the compute engine configuration."""
        return {
            'backend'           : self.nqs.backend_str,
            'batch_size'        : self.batch_size,
            'cached_results'    : len(self._cached_results),
        }

# ---------------------------------------------------------------------------------------------------
#! END OF FILE
# ---------------------------------------------------------------------------------------------------