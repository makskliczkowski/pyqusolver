#!/usr/bin/env python3
"""
Example: Autoregressive Neural Networks for Quantum States
===========================================================

This example demonstrates how to use Autoregressive (AR) neural networks
with the Neural Quantum State solver.

Key Features:
1. Autoregressive wavefunction factorization
2. Comparison with RBM and CNN architectures
3. Parameter efficiency analysis
4. Sequential generation of quantum states

The autoregressive ansatz factorizes the wavefunction as:

    psi(s₁, s₂, ..., sₙ) = p(s₁) times  p(s₂|s₁) times  p(s₃|s₁,s₂) times  ... 

where each conditional probability is computed by a neural network.

Author: Development Team
Date: November 1, 2025
"""

import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

# Add parent directory to path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from QES.NQS.src.nqs_network_integration import NetworkFactory, NetworkSelector
    from QES.Algebra.hilbert import HilbertSpace
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


# =============================================================================
#! Utility Functions
# =============================================================================

def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{title}:")


def print_result(item: str, success: bool, message: str = ""):
    """Print formatted result."""
    status = "(ok)" if success else "(error)"
    msg_part = f" {message}" if message else ""
    print(f"  {status} {item:<45}{msg_part}")


def print_network_comparison(networks: dict, test_states: np.ndarray):
    """Print detailed comparison of networks."""
    print_subsection("Network Comparison")
    
    results = []
    for net_type, network in networks.items():
        try:
            params = network.get_params()
            output = network(test_states)
            
            results.append({
                'type': net_type.upper(),
                'params': len(params),
                'output_shape': output.shape,
                'mean': np.mean(output),
                'std': np.std(output),
            })
        except Exception as e:
            print(f"  Error evaluating {net_type}: {e}")
    
    if results:
        # Print table
        print(f"\n  {'Network':<15} {'Parameters':<15} {'Output Shape':<15} {'Mean':<12} {'Std':<12}")
        print("  " + "-"*70)
        for r in results:
            print(f"  {r['type']:<15} {r['params']:<15} {str(r['output_shape']):<15} "
                  f"{r['mean']:>11.6f}  {r['std']:>11.6f}")


def generate_test_states(n_sites: int, num_states: int) -> np.ndarray:
    """Generate test quantum states."""
    states = np.zeros((num_states, n_sites), dtype=np.float32)
    
    if num_states >= 1:
        states[0, :] = 1                    # All ones
    if num_states >= 2:
        states[1, ::2] = 1                  # Checkerboard
    if num_states >= 3:
        states[2, :n_sites//2] = 1          # Half filled
    if num_states >= 4:
        np.random.seed(42)
        for i in range(3, num_states):
            states[i] = np.random.randint(0, 2, n_sites, dtype=np.float32)
    
    return states


# =============================================================================
#! Main Example - Autoregressive Networks
# =============================================================================

def main():
    """Run autoregressive network example."""
    
    print_section("AUTOREGRESSIVE NEURAL NETWORKS FOR QUANTUM STATES", width=80)
    print("\nKey Concepts:")
    print("  • Factorization: psi(s) = ∏ᵢ p(sᵢ | s₁...sᵢ₋₁)")
    print("  • Conditioning: Each factor depends on previous qubits")
    print("  • Efficiency: Parameter count ≈ N times  (hidden_layers)")
    print("  • Applications: Large systems, state generation, density estimation")
    
    # =========================================================================
    # STEP 1: System Setup
    # =========================================================================
    print_section("STEP 1: System Setup")
    
    n_sites = 8
    try:
        hilbert = HilbertSpace(n_sites)
        n_visible = 2**hilbert.Ns
        print_result("Hilbert Space", True, f"{n_visible} states (2^{n_sites})")
    except Exception as e:
        print_result("Hilbert Space", False, str(e))
        return
    
    # Generate test states
    try:
        test_states = generate_test_states(n_sites, num_states=6)
        print_result("Test States", True, f"shape={test_states.shape}")
    except Exception as e:
        print_result("Test States", False, str(e))
        return
    
    # =========================================================================
    # STEP 2: Understanding Autoregressive Architecture
    # =========================================================================
    print_section("STEP 2: Autoregressive Architecture Overview")
    
    print_subsection("How Autoregressive Networks Work")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │ Autoregressive Factorization                            │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │ Qubit 1:  log p(s₁) ← [constant or bias]              │
  │ Qubit 2:  log p(s₂|s₁) ← Network([s₁])                │
  │ Qubit 3:  log p(s₃|s₁,s₂) ← Network([s₁,s₂])          │
  │ Qubit 4:  log p(s₄|s₁,s₂,s₃) ← Network([s₁,s₂,s₃])    │
  │ ...                                                     │
  │                                                         │
  │ Total log probability:                                 │
  │ log psi(s) = Σᵢ log p(sᵢ|s₁,...,sᵢ₋₁)                    │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
    """)
    
    print_subsection("Computational Flow")
    print("""
  Input: [s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈]
  
  Forward Pass (Sequential):
  ├─ Position 1: Input=[], Output=log p(s₁)
  ├─ Position 2: Input=[s₁], Output=log p(s₂|s₁)
  ├─ Position 3: Input=[s₁,s₂], Output=log p(s₃|s₁,s₂)
  ├─ Position 4: Input=[s₁,s₂,s₃], Output=log p(s₄|s₁,s₂,s₃)
  └─ ...
  
  Total: log psi(s) = Σ outputs
    """)
    
    # =========================================================================
    # STEP 3: Create Networks Comparison
    # =========================================================================
    print_section("STEP 3: Network Architecture Comparison")
    
    networks = {}
    
    # Get available networks
    try:
        available = NetworkFactory.list_available()
        print_result("Available Networks", True, str(available))
    except Exception as e:
        print_result("Available Networks", False, str(e))
        return
    
    # Create networks
    print_subsection("Creating Networks")
    
    for net_type in available:
        try:
            net = NetworkFactory.create(net_type, hilbert)
            networks[net_type] = net
            params = net.get_params()
            print_result(f"{net_type.upper()} Network", True, f"{len(params)} parameters")
        except Exception as e:
            print_result(f"{net_type.upper()} Network", False, str(e))
    
    # =========================================================================
    # STEP 4: Autoregressive-Specific Analysis
    # =========================================================================
    if 'ar' in networks:
        print_section("STEP 4: Autoregressive-Specific Analysis")
        
        ar_network = networks['ar']
        
        print_subsection("Autoregressive Architecture Details")
        try:
            info = ar_network.get_info()
            print(f"  Network Type: {info['name']}")
            print(f"  Number of Qubits: {info['n_qubits']}")
            print(f"  Hidden Layers: {info['hidden_layers']}")
            print(f"  Data Type: {info['dtype']}")
            print(f"  Parameter Type: {info['param_dtype']}")
        except Exception as e:
            print(f"  Info retrieval error: {e}")
        
        print_subsection("Sequential Evaluation")
        print("""
  Autoregressive models evaluate sequentially:
  ┌─ Start with empty context []
  ├─ Predict qubit 1, append to context
  ├─ Predict qubit 2 given context [s₁], append
  ├─ Predict qubit 3 given context [s₁,s₂], append
  └─ Continue until all qubits evaluated
  
  This sequential nature:
  (ok) Allows tractable generation of samples
  (ok) Enables exact computation of likelihoods
  ✗ Makes inference slower than parallel models (RBM, CNN)
        """)
        
        print_subsection("Parameter Efficiency")
        ar_params = len(ar_network.get_params())
        print(f"\n  Autoregressive Parameters: {ar_params}")
        print(f"  Scale: O(N_qubits times  hidden_size²)")
        print(f"\n  For N={n_sites} qubits with hidden_layers=(32,32):")
        print(f"    ├─ Qubit 1 network: 0 -> 32 -> 1")
        print(f"    ├─ Qubit 2 network: 1 -> 32 -> 1")
        print(f"    ├─ Qubit 3 network: 2 -> 32 -> 1")
        print(f"    ├─ ...")
        print(f"    └─ Qubit {n_sites} network: {n_sites-1} -> 32 -> 1")
    
    # =========================================================================
    # STEP 5: Network Comparison and Evaluation
    # =========================================================================
    print_section("STEP 5: Network Evaluation and Comparison")
    
    if networks:
        print_network_comparison(networks, test_states)
    
    # =========================================================================
    # STEP 6: Recommendations
    # =========================================================================
    print_section("STEP 6: When to Use Each Network")
    
    print_subsection("Network Selection Guide")
    
    print("""
  RBM (Restricted Boltzmann Machine):
  ├─ Best for: Small to medium systems (< 50 qubits)
  ├─ Advantages: Expressive, well-studied, parallel evaluation
  ├─ Disadvantages: O(N_visible times  N_hidden) parameters
  └─ Use: When you need expressivity and speed
  
  CNN (Convolutional Neural Network):
  ├─ Best for: Lattice systems with spatial structure
  ├─ Advantages: Exploits locality, parameter efficient
  ├─ Disadvantages: Requires spatial lattice, medium speeds
  └─ Use: When system has translation symmetry
  
  Autoregressive (AR):
  ├─ Best for: Large systems, parameter efficiency
  ├─ Advantages: Sequential factorization, flexible, scalable
  ├─ Disadvantages: Sequential evaluation slower, order-dependent
  └─ Use: When system size is large (> 50 qubits) or parameter efficiency critical
    """)
    
    print_subsection("System Size Recommendations")
    size_examples = [
        (8, 'general'),
        (16, 'lattice_2d'),
        (50, 'large_system'),
        (100, 'large_system'),
    ]
    
    for size, structure in size_examples:
        try:
            recs = NetworkSelector.get_recommendations(size, structure)
            print_result(f"{size}-qubit {structure}", True, str(recs))
        except Exception as e:
            print_result(f"{size}-qubit {structure}", False, str(e))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY")
    
    print("\n(ok) Key Takeaways:")
    print("  1. Autoregressive models factorize: psi = ∏ᵢ p(sᵢ|s₁...sᵢ₋₁)")
    print("  2. Each conditional probability is computed by a neural network")
    print("  3. Sequential evaluation allows exact likelihood computation")
    print("  4. Parameter efficient: scales as O(N times  hidden_layers)")
    print("  5. Best for large systems and parameter-constrained scenarios")
    
    print("\n(ok) Comparison Summary:")
    print("  ├─ RBM: Expressive, parallel, for small-medium systems")
    print("  ├─ CNN: Structured, efficient, for lattice systems")
    print("  └─ AR:  Scalable, flexible, for large systems")
    
    if networks:
        print(f"\n(ok) Networks created: {len(networks)}")
        print("  Available networks:", ", ".join(networks.keys()))
    
    print("\n" + "="*80)
    print("  Resources:")
    print("  • NETWORK_ARCHITECTURE_GUIDE.md - General network reference")
    print("  • AUTOREGRESSIVE_GUIDE.md - Detailed AR documentation")
    print("  • network_integration.py - NetworkFactory implementation")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
