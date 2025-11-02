#!/usr/bin/env python3
"""
Example: Using Multiple Neural Network Architectures with NQS
==============================================================

This example demonstrates how to use different neural network architectures
(RBM, CNN) with the Neural Quantum State solver using a structured approach.

Networks covered:
1. Restricted Boltzmann Machine (RBM) - General purpose
2. Convolutional Neural Network (CNN) - Lattice systems

This example uses preparation functions for clean code organization.

Author: Development Team
Date: November 1, 2025
"""

import numpy as np
import sys
import os
import math
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple

# Add parent directory to path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from QES.NQS.src.network_integration import NetworkFactory, NetworkSelector
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
    network_type: str                               # 'rbm' or 'cnn'
    n_sites: int                                    # Number of qubits/sites
    n_hidden: int                   = 16            # For RBM
    features: List[int]             = field(default_factory=lambda: [16, 32])  # For CNN
    kernel_sizes: List[int]         = field(default_factory=lambda: [3, 3])    # For CNN
    strides: List[int]              = field(default_factory=lambda: [1, 1])    # For CNN
    reshape_dims: Optional[Tuple]   = None          # For CNN (computed if None)
    seed: int                       = 42
    dtype: Any                      = np.float32


@dataclass
class SimulationConfig:
    """Configuration for the entire simulation."""
    n_sites: int                = 8
    batch_size: int             = 4
    num_test_states: int        = 3
    network_types: List[str]    = field(default_factory=lambda: ['rbm', 'cnn'])
    seed: int                   = 42


# =============================================================================
#! Utility Functions
# =============================================================================

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{title}:")


def print_result(item: str, success: bool, message: str = ""):
    """Print formatted result."""
    status = "✓" if success else "❌"
    print(f"  {status} {item:<40} {message}")


# =============================================================================
#! Preparation Functions
# =============================================================================

def prepare_hilbert_space(n_sites: int) -> Tuple[HilbertSpace, int]:
    """
    Prepare the Hilbert space.
    
    Parameters:
        n_sites (int): Number of qubits/sites
        
    Returns:
        Tuple[HilbertSpace, int]: Hilbert space and number of visible units
    """
    hilbert = HilbertSpace(n_sites)
    n_visible = 2**hilbert.Ns
    return hilbert, n_visible


def prepare_network_config(config: NetworkConfig, n_visible: int) -> NetworkConfig:
    """
    Prepare and validate network configuration.
    
    Parameters:
        config (NetworkConfig): Network configuration
        n_visible (int): Number of visible units
        
    Returns:
        NetworkConfig: Updated configuration
    """
    input_shape = (n_visible,)
    
    if config.network_type == 'cnn' and config.reshape_dims is None:
        # For CNN, compute reshape_dims if not provided
        side = int(math.sqrt(n_visible))
        if side * side == n_visible:
            config.reshape_dims = (side, side)
        else:
            # If not square, use 1D
            config.reshape_dims = (n_visible,)
    
    return config


def prepare_rbm_network(config: NetworkConfig, n_visible: int) -> Any:
    """
    Prepare RBM network.
    
    Parameters:
        config (NetworkConfig): Network configuration
        n_visible (int): Number of visible units
        
    Returns:
        Any: RBM network instance
    """
    try:
        from QES.general_python.ml.net_impl.networks.net_rbm import RBM
        
        input_shape = (n_visible,)
        rbm = RBM(
            input_shape=input_shape,
            n_hidden=config.n_hidden,
            dtype=config.dtype,
            param_dtype=config.dtype,
            seed=config.seed,
            visible_bias=True,
            bias=True
        )
        return rbm
        
    except ImportError:
        raise ImportError("RBM not available. JAX is required.")
    except Exception as e:
        raise RuntimeError(f"Failed to create RBM network: {e}")


def prepare_cnn_network(config: NetworkConfig, n_visible: int) -> Any:
    """
    Prepare CNN network.
    
    Parameters:
        config (NetworkConfig): Network configuration
        n_visible (int): Number of visible units
        
    Returns:
        Any: CNN network instance
    """
    try:
        from QES.general_python.ml.net_impl.networks.net_cnn import CNN
        from QES.general_python.ml.net_impl.activation_functions import elu_jnp
        
        # Ensure reshape_dims is set
        if config.reshape_dims is None:
            side = int(math.sqrt(n_visible))
            if side * side == n_visible:
                config.reshape_dims = (side, side)
            else:
                config.reshape_dims = (n_visible,)
        
        input_shape = (n_visible,)
        cnn = CNN(
            input_shape=input_shape,
            reshape_dims=config.reshape_dims,
            features=config.features,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            activations=[elu_jnp] * len(config.features),
            dtype=config.dtype,
            param_dtype=config.dtype,
            final_activation=elu_jnp,
            seed=config.seed,
            output_shape=(1,)
        )
        return cnn
        
    except ImportError:
        raise ImportError("CNN not available. JAX is required.")
    except Exception as e:
        raise RuntimeError(f"Failed to create CNN network: {e}")


def prepare_network(network_type: str, config: NetworkConfig, n_visible: int) -> Any:
    """
    Prepare neural network based on type.
    
    Parameters:
        network_type (str): Type of network ('rbm' or 'cnn')
        config (NetworkConfig): Network configuration
        n_visible (int): Number of visible units
        
    Returns:
        Any: Network instance
    """
    if network_type == 'rbm':
        return prepare_rbm_network(config, n_visible)
    elif network_type == 'cnn':
        return prepare_cnn_network(config, n_visible)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def generate_test_states(n_sites: int, num_states: int, network_type: str = 'rbm') -> np.ndarray:
    """
    Generate test states for network evaluation.
    
    Parameters:
        n_sites (int): Number of sites
        num_states (int): Number of test states
        network_type (str): Type of network (affects state generation)
        
    Returns:
        np.ndarray: Test states
    """
    states = np.zeros((num_states, n_sites), dtype=np.float32)
    
    if num_states >= 1:
        states[0, :] = 1  # All ones
    if num_states >= 2:
        states[1, ::2] = 1  # Checkerboard
    if num_states >= 3:
        states[2, :n_sites//2] = 1  # Half filled
    
    return states


# =============================================================================
#! Main Example Functions
# =============================================================================

def example_1_rbm():
    """Example 1: Create and evaluate RBM network."""
    print_section("EXAMPLE 1: RBM Network")
    
    try:
        # Prepare Hilbert space
        print_subsection("Step 1: Prepare Hilbert Space")
        hilbert, n_visible = prepare_hilbert_space(n_sites=8)
        print_result("Hilbert space", True, f"{n_visible} states")
        
        # Prepare network configuration
        print_subsection("Step 2: Prepare Network Configuration")
        config = NetworkConfig(
            network_type='rbm',
            n_sites=8,
            n_hidden=16
        )
        config = prepare_network_config(config, n_visible)
        print_result("Network config", True, f"hidden_units={config.n_hidden}")
        
        # Create network
        print_subsection("Step 3: Create Network")
        network = prepare_rbm_network(config, n_visible)
        params = network.get_params()
        print_result("RBM network", True, f"{len(params)} parameters")
        
        # Generate test states
        print_subsection("Step 4: Generate Test States")
        test_states = generate_test_states(n_sites=8, num_states=3)
        print_result("Test states", True, f"shape={test_states.shape}")
        
        # Evaluate network
        print_subsection("Step 5: Evaluate Network")
        result = network(test_states)
        print_result("Network evaluation", True, f"shape={result.shape}")
        print(f"    First 3 outputs: {result[:3]}")
        
        return True
        
    except Exception as e:
        print_result("RBM example", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def example_2_cnn():
    """Example 2: Create and evaluate CNN network."""
    print_section("EXAMPLE 2: CNN Network")
    
    try:
        # Prepare Hilbert space
        print_subsection("Step 1: Prepare Hilbert Space")
        hilbert, n_visible = prepare_hilbert_space(n_sites=8)
        print_result("Hilbert space", True, f"{n_visible} states (1D chain)")
        
        # Prepare network configuration
        print_subsection("Step 2: Prepare Network Configuration")
        config = NetworkConfig(
            network_type='cnn',
            n_sites=8,
            features=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1]
        )
        config = prepare_network_config(config, n_visible)
        print_result("Network config", True, f"reshape_dims={config.reshape_dims}")
        
        # Create network
        print_subsection("Step 3: Create Network")
        network = prepare_cnn_network(config, n_visible)
        params = network.get_params()
        print_result("CNN network", True, f"{len(params)} parameters")
        
        # Generate test states
        print_subsection("Step 4: Generate Test States")
        test_states = generate_test_states(n_sites=8, num_states=3)
        print_result("Test states", True, f"shape={test_states.shape}")
        
        # Evaluate network
        print_subsection("Step 5: Evaluate Network")
        result = network(test_states)
        print_result("Network evaluation", True, f"shape={result.shape}")
        print(f"    First 3 outputs: {result[:3]}")
        
        return True
        
    except Exception as e:
        print_result("CNN example", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def example_3_factory():
    """Example 3: Using NetworkFactory for easy creation."""
    print_section("EXAMPLE 3: Network Factory")
    
    try:
        # List available networks
        print_subsection("Step 1: List Available Networks")
        available = NetworkFactory.list_available()
        print_result("Available networks", True, str(available))
        
        if not available:
            print("⚠  No networks available (JAX required)")
            return False
        
        # Get recommendations
        print_subsection("Step 2: Get Network Recommendations")
        recommendations = {
            'small_system': NetworkSelector.get_recommendations(8, 'general'),
            'medium_system': NetworkSelector.get_recommendations(16, 'lattice_2d'),
            'large_system': NetworkSelector.get_recommendations(100, 'large_system'),
        }
        for sys_type, nets in recommendations.items():
            print_result(f"Recommended for {sys_type}", True, str(nets))
        
        # Create networks using factory
        print_subsection("Step 3: Create Networks with Factory")
        hilbert, n_visible = prepare_hilbert_space(n_sites=8)
        networks_created = {}
        
        for net_type in available:
            try:
                net = NetworkFactory.create(net_type, hilbert)
                networks_created[net_type] = net
                print_result(f"Created {net_type.upper()}", True, net.__class__.__name__)
            except Exception as e:
                print_result(f"Created {net_type.upper()}", False, str(e))
        
        return len(networks_created) > 0
        
    except Exception as e:
        print_result("Factory example", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def example_4_comparison():
    """Example 4: Compare networks using unified interface."""
    print_section("EXAMPLE 4: Network Comparison")
    
    try:
        # Prepare
        print_subsection("Step 1: Prepare Systems")
        hilbert, n_visible = prepare_hilbert_space(n_sites=8)
        available = NetworkFactory.list_available()
        print_result("Available networks", True, str(available))
        
        # Generate test states
        print_subsection("Step 2: Generate Test States")
        test_states = generate_test_states(n_sites=8, num_states=3)
        print_result("Test states", True, f"shape={test_states.shape}")
        
        # Compare networks
        print_subsection("Step 3: Compare Network Evaluations")
        print()
        
        for net_type in available:
            try:
                net = NetworkFactory.create(net_type, hilbert)
                result = net(test_states)
                
                print(f"  {net_type.upper()}:")
                print(f"    - Output shape: {result.shape}")
                print(f"    - Mean: {np.mean(result):.6f}")
                print(f"    - Std:  {np.std(result):.6f}")
                print(f"    - Min:  {np.min(result):.6f}")
                print(f"    - Max:  {np.max(result):.6f}")
                
            except Exception as e:
                print(f"  {net_type.upper()}: ❌ {e}")
        
        return True
        
    except Exception as e:
        print_result("Comparison example", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def example_5_configuration():
    """Example 5: Using configuration classes."""
    print_section("EXAMPLE 5: Configuration Classes")
    
    try:
        # Create simulation config
        print_subsection("Step 1: Create Simulation Configuration")
        sim_config = SimulationConfig(
            n_sites=8,
            batch_size=4,
            num_test_states=3,
            network_types=['rbm', 'cnn'],
            seed=42
        )
        print(f"  Simulation Config:")
        print(f"    - Sites: {sim_config.n_sites}")
        print(f"    - Batch size: {sim_config.batch_size}")
        print(f"    - Test states: {sim_config.num_test_states}")
        print(f"    - Networks: {sim_config.network_types}")
        
        # Prepare Hilbert space
        print_subsection("Step 2: Prepare Hilbert Space")
        hilbert, n_visible = prepare_hilbert_space(sim_config.n_sites)
        print_result("Hilbert space", True, f"{n_visible} states")
        
        # Create networks from config
        print_subsection("Step 3: Create Networks from Configuration")
        networks = {}
        
        for net_type in sim_config.network_types:
            try:
                config = NetworkConfig(
                    network_type=net_type,
                    n_sites=sim_config.n_sites,
                    seed=sim_config.seed
                )
                config = prepare_network_config(config, n_visible)
                
                network = prepare_network(net_type, config, n_visible)
                networks[net_type] = network
                print_result(f"Network {net_type}", True, f"{len(network.get_params())} params")
                
            except Exception as e:
                print_result(f"Network {net_type}", False, str(e))
        
        # Test all networks
        print_subsection("Step 4: Test All Networks")
        test_states = generate_test_states(
            n_sites=sim_config.n_sites,
            num_states=sim_config.num_test_states
        )
        
        for net_type, network in networks.items():
            try:
                result = network(test_states)
                print_result(
                    f"Evaluate {net_type}",
                    True,
                    f"output shape={result.shape}, mean={np.mean(result):.4f}"
                )
            except Exception as e:
                print_result(f"Evaluate {net_type}", False, str(e))
        
        return True
        
    except Exception as e:
        print_result("Configuration example", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
#! Main
# =============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  MULTIPLE NEURAL NETWORK ARCHITECTURES WITH NQS")
    print("="*70)
    print("\nThis example demonstrates using a structured approach with")
    print("dedicated preparation functions for network creation and evaluation.")
    
    results = []
    
    # Run examples
    results.append(("RBM Network", example_1_rbm()))
    results.append(("CNN Network", example_2_cnn()))
    results.append(("NetworkFactory", example_3_factory()))
    results.append(("Network Comparison", example_4_comparison()))
    results.append(("Configuration Classes", example_5_configuration()))
    
    # Summary
    print_section("SUMMARY")
    
    print("\nResults:")
    passed = 0
    for name, success in results:
        status = "✓ PASS" if success else "⚠ FAIL"
        print(f"  {name:.<50} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} examples completed successfully")
    
    if passed == len(results):
        print("\n🎉 All examples completed successfully!")
    else:
        print("\n⚠ Some examples failed (check error messages above)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
