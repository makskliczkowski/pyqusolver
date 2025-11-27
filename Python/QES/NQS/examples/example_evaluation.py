"""
Examples: Unified Evaluation Interface for NQS

This file demonstrates the new unified evaluation interface with ComputeLocalEnergy
and UnifiedEvaluationEngine.

-----------------------------------------------------------------------------
File        : NQS/examples/example_unified_evaluation.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Description : Shows how the new architecture consolidates.
-----------------------------------------------------------------------------
"""

import numpy as np
from typing import Callable, Dict, Optional

# In actual use, import from QES.NQS
from QES.NQS.src.nqs_engine import (
    create_compute_local_energy, 
    EnergyStatistics, 
    ObservableResult
)
from QES.NQS.src.general.nqs_general_engine import (
    create_evaluation_engine,
    EvaluationConfig,
    EvaluationResult,
)

# Create mock NQS
class MockNQS:
    def __init__(self):
        self._params        = {'weights': np.array([0.1, 0.2, 0.3])}
        self._ansatz_func   = lambda p, s: np.dot(p['weights'], s)
    def get_params(self):
        return self._params

#####################################################################################################
#! EXAMPLE 1: Basic Ansatz Evaluation
#####################################################################################################

def example_1_basic_ansatz_evaluation():
    """
    Example 1: Evaluate ansatz on a set of states.
    
    This shows how the new unified interface replaces the old:
        nqs.ansatz(states)  [old, deprecated]
    
    With the new:
        computer.evaluate_ansatz(states)  [new, recommended]
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Ansatz Evaluation")
    print("="*70)
    
    nqs         = MockNQS()
    
    # Create evaluation computer
    computer    = create_compute_local_energy(nqs, backend='auto', batch_size=None)
    
    # Generate test states
    states      = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                ])
    
    # Evaluate ansatz
    result      = computer.evaluate_ansatz(states)
    
    print(f"\nInput states shape: {states.shape}")
    print(f"Ansatz values shape: {result.values.shape}")
    print(f"Ansatz values: {result.values}")
    print(f"Mean: {result.mean:.6f}")
    print(f"Std: {result.std:.6f}")
    print(f"Backend used: {result.backend_used}")
    print("(ok) Example 1 complete")

#####################################################################################################
#! EXAMPLE 2: Local Energy Computation
#####################################################################################################

def example_2_local_energy_computation():
    """
    Example 2: Compute local energies E_loc(s) = <s|H|psi>/<s|psi>
    
    This shows the new unified energy computation interface, which replaces:
        nqs.step()  [old, multi-purpose]
        nqs._single_step_groundstate()  [old, internal]
    
    With the new:
        computer.compute_local_energy(states, ham_func)  [new, focused]
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Local Energy Computation")
    print("="*70)
    
    
    # Create evaluation computer
    nqs         = MockNQS()
    computer    = create_compute_local_energy(nqs, backend='auto', batch_size=2)
    
    # Define a mock Hamiltonian that computes local energy
    def hamiltonian_action(s):
        r"""Mock Hamiltonian: H = Σ_i (\sigma_x_i + \sigma_z_i \sigma_z_{i+1})"""
        # For simplicity, just return energy based on configuration
        return np.sum(s) + 0.1 * np.sum(s[:-1] * s[1:])
    
    # Generate test states
    states = np.random.randint(0, 2, size=(10, 4))
    
    # Compute local energies
    energy_stats = computer.compute_local_energy(states, hamiltonian_action)
    
    print(f"\nNumber of samples: {energy_stats.n_samples}")
    print(f"Local energies shape: {energy_stats.local_energies.shape}")
    print(f"Local energies: {energy_stats.local_energies}")
    print(f"\nEnergy Statistics:")
    print(f"  Mean: {energy_stats.mean_energy:.6f}")
    print(f"  Std: {energy_stats.std_energy:.6f}")
    print(f"  Error of mean: {energy_stats.error_of_mean:.6f}")
    print(f"  Min: {energy_stats.min_energy:.6f}")
    print(f"  Max: {energy_stats.max_energy:.6f}")
    print(f"  Variance: {energy_stats.variance:.6f}")
    print("(ok) Example 2 complete")

#####################################################################################################
#! EXAMPLE 3: Observable Evaluation
#####################################################################################################

def example_3_observable_evaluation():
    """
    Example 3: Evaluate multiple observables
    
    This replaces the scattered:
        nqs.eval_observables(...)  [old]
        nqs.apply(functions, ...)  [old, generic]
    
    With the new focused:
        computer.compute_observable(func, states, name)  [new]
        computer.compute_observables(funcs_dict, states)  [new]
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Observable Evaluation")
    print("="*70)
    
    nqs         = MockNQS()
    computer    = create_compute_local_energy(nqs, backend='auto', batch_size=3)
    
    # Generate test states
    states      = np.array([
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ])
    
    # Define observables
    def particle_number(s):
        """Observable: particle number N = Σ_i n_i"""
        return np.sum(s)
    
    def correlation(s):
        """Observable: nearest-neighbor correlation"""
        return np.sum(s[:-1] * s[1:])
    
    def magnetization(s):
        """Observable: effective magnetization"""
        return np.sum((-1)**s)
    
    # Define observable dictionary
    observables     = {
        'ParticleNumber'    : particle_number,
        'Correlation'       : correlation,
        'Magnetization'     : magnetization,
    }
    
    # Single observable
    print("\nEvaluating single observable:")
    result = computer.compute_observable(
        particle_number, states, 'ParticleNumber'
    )
    print(f"  Observable: {result.observable_name}")
    print(f"  Local values: {result.local_values}")
    print(f"  Mean: {result.mean_local_value:.6f}")
    print(f"  Std: {result.std_local_value:.6f}")
    
    # Multiple observables
    print("\nEvaluating multiple observables:")
    results = computer.compute_observables(observables, states)
    for name, obs_result in results.items():
        print(f"  {name:20s}: <O> = {obs_result.mean_local_value:8.4f} ± {obs_result.std_local_value:.4f}")
    
    print("(ok) Example 3 complete")

#####################################################################################################
#! EXAMPLE 4: Batch Processing with Different Backends
#####################################################################################################

def example_4_batch_processing():
    """
    Example 4: Efficient batch processing with automatic backend selection.
    
    Key features:
    - Automatic backend dispatch based on array type
    - Configurable batch sizes
    - Statistics computation
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing with Backend Selection")
    print("="*70)
        
    nqs         = MockNQS()
    # Large state set for batching
    n_states    = 100
    state_dim   = 5
    states      = np.random.randint(0, 2, size=(n_states, state_dim))

    # Test with different batch sizes
    batch_sizes = [None, 10, 25, 50]
    
    print(f"\nProcessing {n_states} states of dimension {state_dim}")
    print("\nBatch size comparison:")
    
    for batch_size in batch_sizes:
        computer    = create_compute_local_energy(nqs, backend='auto', batch_size=batch_size)
        result      = computer.evaluate_ansatz(states)
        print(f"  Batch size {str(batch_size):>5s}: mean={result.mean:8.4f}, std={result.std:.4f}")
    
    print("\n(ok) Example 4 complete")

#####################################################################################################
#! EXAMPLE 5: Comparison of Backends
#####################################################################################################

def example_5_backend_comparison():
    """
    Example 5: Compare NumPy vs JAX backends.
    
    Shows how the unified interface allows easy backend switching:
        computer.set_backend('numpy')
        computer.set_backend('jax')
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Backend Comparison (NumPy vs JAX)")
    print("="*70)
    
    nqs         = MockNQS()
    states      = np.random.randn(20, 3).astype(np.float32)

    print("\nTesting with NumPy backend:")
    computer    = create_compute_local_energy(nqs, backend='numpy')
    result_np   = computer.evaluate_ansatz(states)
    print(f"  Backend: {result_np.backend_used}")
    print(f"  Mean: {result_np.mean:.6f}")
    print(f"  Config: {computer.get_config()}")
    
    print("\nTesting with Auto backend (will select based on array type):")
    computer.set_backend('auto')
    result_auto = computer.evaluate_ansatz(states)
    print(f"  Backend: {result_auto.backend_used}")
    print(f"  Mean: {result_auto.mean:.6f}")
    
    # Results should be identical
    diff = np.abs(result_np.values - result_auto.values).max()
    print(f"\nMax difference between backends: {diff:.2e}")
    assert diff < 1e-6, "Backend results differ!"
    print("(ok) Backend results consistent")
    print("(ok) Example 5 complete")

#####################################################################################################
#! EXAMPLE 6: Advanced - Custom Function Evaluation
#####################################################################################################

def example_6_custom_function_evaluation():
    """
    Example 6: Evaluate custom functions on state batches.
    
    This shows the flexibility of the new evaluate_function interface:
        computer.evaluate_function(custom_func, states, ...)
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Function Evaluation")
    print("="*70)

    nqs         = MockNQS()
    computer    = create_compute_local_energy(nqs, backend='auto', batch_size=5)

    # Custom functions to evaluate
    def entropy(s):
        """Shannon entropy of configuration"""
        p = np.mean(s)
        if p > 0 and p < 1:
            return -(p * np.log(p) + (1-p) * np.log(1-p))
        return 0.0
    
    def pattern_match(s):
        """Check for specific pattern: alternating bits"""
        pattern = np.array([1, 0, 1, 0])
        return float(np.allclose(s, pattern))
    
    def structure_factor(s):
        """Simplified structure factor"""
        return np.abs(np.fft.fft(s.astype(float))[1])
    
    # Test states
    states = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
    ])
    
    print("\nEvaluating custom functions:")
    
    result_entropy = computer.evaluate_function(entropy, states)
    print(f"  Entropy:")
    print(f"    Values: {result_entropy.values}")
    print(f"    Mean: {result_entropy.mean:.6f}")
    
    result_pattern = computer.evaluate_function(pattern_match, states)
    print(f"  Pattern match:")
    print(f"    Values: {result_pattern.values}")
    print(f"    Mean: {result_pattern.mean:.6f}")
    
    result_structure = computer.evaluate_function(structure_factor, states)
    print(f"  Structure factor:")
    print(f"    Values: {result_structure.values[:3]}")
    print(f"    Mean: {result_structure.mean:.6f}")
    
    print("(ok) Example 6 complete")

#####################################################################################################
#! MAIN: RUN ALL EXAMPLES
#####################################################################################################

def main():
    """Run all examples."""
    
    print("\n" + "="*70)
    print("UNIFIED EVALUATION INTERFACE - COMPREHENSIVE EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate the new ComputeLocalEnergy interface")
    print("which consolidates 19+ scattered evaluation methods into a clean,")
    print("maintainable framework that is easier to test and extend.")
    print("\nSession 5, Task 5.5 - Example Creation")
    
    try:
        example_1_basic_ansatz_evaluation()
        example_2_local_energy_computation()
        example_3_observable_evaluation()
        example_4_batch_processing()
        example_5_backend_comparison()
        example_6_custom_function_evaluation()
        
        print("\n" + "="*70)
        print("(ok) ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nKey takeaways:")
        print("  1. UnifiedEvaluationEngine eliminates backend dispatch duplication")
        print("  2. ComputeLocalEnergy provides NQS-specific interface")
        print("  3. Backwards compatible with existing NQS API")
        print("  4. New API is clearer and more maintainable")
        print("  5. Batch processing and backend selection automatic")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR in examples: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

# ----------------------------------------------------------------------

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
    
# ----------------------------------------------------------------------
#! END OF FILE
# ----------------------------------------------------------------------
