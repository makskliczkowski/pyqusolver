#!/usr/bin/env python3
"""
Example: Using Multiple Neural Network Architectures with NQS
==============================================================

This example demonstrates how to use different neural network architectures
(RBM, CNN) with the Neural Quantum State solver, and compare their properties.

Networks covered:
1. Restricted Boltzmann Machine (RBM) - General purpose
2. Convolutional Neural Network (CNN) - Lattice systems

This example uses a structured approach with dedicated preparation functions
for each component (network, backend, sampler, etc.).

--------------------------------------------------------------
File        : examples/example_multiple_networks.py
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
date        : 2025-11-01
--------------------------------------------------------------
"""

import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

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
#! Configuration and Parameter Classes
# =============================================================================

@dataclass
class NetworkConfig:
    """Configuration for network preparation."""
    network_type        : str               # 'rbm' or 'cnn'
    n_sites             : int               # Number of qubits/sites
    n_hidden            : int               = 16
    features            : List[int]         = field(default_factory=lambda: [16, 32])  # For CNN
    kernel_sizes        : List[int]         = field(default_factory=lambda: [3, 3])    # For CNN
    strides             : List[int]         = field(default_factory=lambda: [1, 1])    # For CNN
    reshape_dims        : Optional[tuple]   = None        # For CNN
    seed                : int               = 42
    dtype               : Any               = np.float32

@dataclass
class SimulationConfig:
    """Configuration for the entire simulation."""
    n_sites             : int               = 8
    batch_size          : int               = 4
    num_test_states     : int               = 3
    network_types       : List[str]         = field(default_factory=lambda: ['rbm', 'cnn'])
    seed                : int               = 42

# =============================================================================
#! Utility Functions
# =============================================================================

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

# =============================================================================
#! Example Functions
# =============================================================================

def test_rbm_network():
    
    """Example 1: Using RBM (Restricted Boltzmann Machine)."""
    print_section("EXAMPLE 1: RBM Network")
    
    try:
        # Create Hilbert space (8 qubits)
        ns                  = 8
        hilbert             = HilbertSpace(ns=ns)
        n_visible           = ns
        print(f"Hilbert space: {n_visible} states")
        
        # Create RBM network
        print("\nRBM Configuration:")
        num_hidden          = 16
        print(f"  - Visible units: {n_visible}")
        print(f"  - Hidden units: {num_hidden}")
        
        try:
            from QES.general_python.ml.net_impl.networks.net_rbm import RBM
            
            # input_shape must be a tuple of (n_visible,)
            input_shape     = (n_visible,)
            rbm             = RBM(input_shape=input_shape, n_hidden=num_hidden)
            print(f"RBM network created")
            
            # Get network info
            params          = rbm.get_params()
            print(f"\n(ok) Network parameters: {len(params)} parameters")
            
            # Test evaluation
            test_states     = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                                        [0, 1, 0, 1, 0, 1, 0, 1],
                                        [1, 1, 0, 0, 1, 1, 0, 0]], dtype=np.float32)
            
            print(f"\nTest evaluation on {len(test_states)} states:")
            result  = rbm(test_states)
            print(f"  - Output shape: {result.shape}")
            print(f"  - First 3 values: {result[:3]}")
            
            return rbm, True
            
        except ImportError:
            print("(warning) RBM not available (JAX required)")
            return None, False
            
    except Exception as e:
        print(f"(error) Error: {e}")
        import traceback
        traceback.print_exc()
        return None, False

# -----------------------------------------------------------------------------
#! Example Functions Continued
# -----------------------------------------------------------------------------

def test_cnn_network():
    """Example 2: Using CNN (Convolutional Neural Network)."""
    print_section("EXAMPLE 2: CNN Network")
    
    try:
        # Create Hilbert space (1D chain: 8 sites for CNN)
        ns              = 8
        hilbert         = HilbertSpace(ns=ns)
        n_visible       = ns
        print(f"(ok) Hilbert space: {n_visible} states (8 sites)")
        
        # Create CNN network
        print("\nCNN Configuration:")
        features        = [16, 32]
        kernel_sizes    = [3, 3]
        strides         = [1, 1]
        
        print(f"  - Features: {features}")
        print(f"  - Kernel sizes: {kernel_sizes}")
        print(f"  - Strides: {strides}")
        
        try:
            from QES.general_python.ml.net_impl.networks.net_cnn import CNN
            
            # input_shape must be a tuple, reshape_dims specifies spatial layout (8,) for 1D
            input_shape     = (n_visible,)
            reshape_dims    = (1, n_visible)        # 1D chain
            
            cnn = CNN(
                input_shape     =   input_shape,
                reshape_dims    =   reshape_dims,
                features        =   features,
                kernel_sizes    =   kernel_sizes,
                strides         =   strides
            )
            print(f"(ok) CNN network created")
            
            # Get network info
            params              = cnn.get_params()
            print(f"\n(ok) Network parameters: {len(params)} parameters")
            
            # Test evaluation
            test_states         = np.zeros((3, ns), dtype=np.float32)
            test_states[0, :]   = 1  # All ones
            test_states[1, ::2] = 1  # Checkerboard
            test_states[2, :4]  = 1  # Half filled

            print(f"\nTest evaluation on {len(test_states)} states:")
            result = cnn(test_states)
            print(f"  - Output shape: {result.shape}")
            print(f"  - First 3 values: {result[:3]}")
            
            return cnn, True
            
        except ImportError:
            print("(warning) CNN not available (JAX required)")
            return None, False
            
    except Exception as e:
        print(f"(error) Error: {e}")
        import traceback
        traceback.print_exc()
        return None, False

# -----------------------------------------------------------------------------

def test_network_factory():
    """Example 3: Using NetworkFactory for easy network creation."""
    print_section("EXAMPLE 3: Network Factory")
    
    try:
        # List available networks
        available = NetworkFactory.list_available()
        print(f"Available networks: {available}")
        
        if not available:
            print("(warning) No networks available (JAX required)")
            return False
        
        # Get recommendations
        print("\nNetwork recommendations:")
        print(f"  - 8-qubit system:         {NetworkSelector.get_recommendations(8, 'general')}")
        print(f"  - 16-qubit 2D lattice:    {NetworkSelector.get_recommendations(16, 'lattice_2d')}")
        print(f"  - 100-qubit system:       {NetworkSelector.get_recommendations(100, 'large_system')}")
        
        # Create networks using factory
        hilbert = HilbertSpace(8)
        print(f"\nCreating networks with factory (Hilbert space: {2**hilbert.Ns} states):")
        
        networks = {}
        for net_type in available:
            try:
                net = NetworkFactory.create(net_type, hilbert)
                networks[net_type] = net
                print(f"  (ok) {net_type.upper()}: {net.__class__.__name__}")
            except Exception as e:
                print(f"  (error) {net_type.upper()}: {e}")
        
        # Create NQS with each network
        print(f"\nCreating NQS solvers:")
        solvers = {}
        
        for net_type, net in networks.items():
            try:
                params = net.get_params()
                solvers[net_type] = net
                print(f"  (ok) {net_type.upper()}: Network with {len(params)} parameters")
            except Exception as e:
                print(f"  (error) {net_type.upper()}: {e}")
        
        return len(solvers) > 0
        
    except Exception as e:
        print(f"(error) Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------

def test_unified_evaluation():
    """Example 4: Using unified evaluation engine with different networks."""
    print_section("EXAMPLE 4: Unified Evaluation Engine with Different Networks")
    
    try:
        hilbert     = HilbertSpace(8)
        available   = NetworkFactory.list_available()

        if not available:
            print("(warning) No networks available")
            return False
        
        # Test states
        test_states = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        print(f"Test states shape: {test_states.shape}")
        print(f"\nEvaluation results for each network:")
        
        for net_type in available:
            try:
                net = NetworkFactory.create(net_type, hilbert)
                
                # Use unified evaluation interface (network is callable)
                result = net(test_states)
                
                print(f"\n{net_type.upper()}:")
                print(f"  - Output type: {type(result)}")
                print(f"  - Shape: {result.shape}")
                print(f"  - Mean: {np.mean(result):.6f}")
                print(f"  - Std: {np.std(result):.6f}")
                
            except Exception as e:
                print(f"\n{net_type.upper()}: (error) {e}")
        
        return True
        
    except Exception as e:
        print(f"(error) Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------
#! Main Execution
# -----------------------------------------------------------------------------

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  MULTIPLE NEURAL NETWORK ARCHITECTURES WITH NQS")
    print("="*70)
    print("\nThis example demonstrates:")
    print("  1. Using RBM networks")
    print("  2. Using CNN networks")
    print("  3. Using NetworkFactory for easy creation")
    print("  4. Unified evaluation engine with different networks")
    
    results = []
    
    # Run examples
    nqs_rbm, rbm_ok = test_rbm_network()
    results.append(("RBM Network", rbm_ok))
    
    nqs_cnn, cnn_ok = test_cnn_network()
    results.append(("CNN Network", cnn_ok))
    
    factory_ok = test_network_factory()
    results.append(("Network Factory", factory_ok))
    
    eval_ok = test_unified_evaluation()
    results.append(("Unified Evaluation", eval_ok))
    
    # Summary
    print_section("SUMMARY")
    
    print("\nResults:")
    passed = 0
    for name, ok in results:
        status = "(ok) PASS" if ok else "(warning) SKIP/FAIL"
        print(f"  {name:.<50} {status}")
        if ok:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} examples completed successfully")
    
    if passed == len(results):
        print("\n(weee) All examples completed successfully!")
    else:
        print("\n(warning) Some examples skipped (JAX dependencies may not be available)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()

# =============================================================================
#! End of File
# =============================================================================