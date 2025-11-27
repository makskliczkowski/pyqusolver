#!/usr/bin/env python3
"""
Example: Network Architecture Showcase
=======================================

This single, comprehensive example demonstrates the key concepts for using
neural network architectures with the Neural Quantum State solver.

Key Features:
1. RBM (Restricted Boltzmann Machine)   - General purpose networks
2. CNN (Convolutional Neural Network)   - Lattice systems
3. NetworkFactory                       - Unified network creation
4. Configuration Classes                - Clean parameter management

This example uses a structured approach with preparation functions,
following the best practices for network initialization and evaluation.

------------------------------------------------------------------------------
File            : NQS/examples/example_network_showcase.py
Author          : Maksymilian Kliczkowski
Date            : 2025-11-01
License         : MIT
------------------------------------------------------------------------------
"""

import numpy as np
import sys
import os
import math
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple

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
#! Configuration Classes
# =============================================================================

@dataclass
class NetworkConfig:
    """Configuration for network preparation."""
    network_type: str                                                           # 'rbm', 'cnn', 'ar', etc.
    n_sites: int                                                                # Number of qubits/sites
    n_hidden: int                   = 16                                        # For RBM
    features: List[int]             = field(default_factory=lambda: [16, 32])   # For CNN
    kernel_sizes: List[int]         = field(default_factory=lambda: [3, 3])     # For CNN
    strides: List[int]              = field(default_factory=lambda: [1, 1])     # For CNN
    reshape_dims: Optional[Tuple]   = None                                      # For CNN (computed if None)
    seed: int                       = 42
    dtype: Any                      = np.float32


@dataclass
class SimulationConfig:
    """Configuration for the entire simulation."""
    n_sites: int                    = 8
    batch_size: int                 = 4
    num_test_states: int            = 5
    network_types: List[str]        = field(default_factory=lambda: ['rbm', 'cnn'])
    seed: int                       = 42


# =============================================================================
#! Utility Functions
# =============================================================================

def print_section(title: str, width: int = 70):
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

def print_network_info(net_type: str, network: Any, test_states: np.ndarray):
    """Print detailed information about a network."""
    try:
        params = network.get_params()
        result = network(test_states)
        
        print(f"\n  {net_type.upper()} Network:")
        print(f"    ├─ Parameters:    {len(params)}")
        print(f"    ├─ Output Shape:  {result.shape}")
        print(f"    ├─ Mean Output:   {np.mean(result):.6f}")
        print(f"    ├─ Std Output:    {np.std(result):.6f}")
        print(f"    ├─ Min/Max:       {np.min(result):.6f} / {np.max(result):.6f}")
        print(f"    └─ Class:         {network.__class__.__name__}")
        return True
    except Exception as e:
        print(f"  {net_type.upper()}: Error - {e}")
        return False

# =============================================================================
#! Preparation Functions
# =============================================================================

def prepare_hilbert_space(n_sites: int) -> Tuple[HilbertSpace, int]:
    """Prepare the Hilbert space."""
    hilbert     = HilbertSpace(n_sites)
    n_visible   = 2**hilbert.Ns
    return hilbert, n_visible

def prepare_network_config(config: NetworkConfig, n_visible: int) -> NetworkConfig:
    """Prepare and validate network configuration."""
    if config.network_type == 'cnn' and config.reshape_dims is None:
        side = int(math.sqrt(n_visible))
        if side * side == n_visible:
            config.reshape_dims = (side, side)
        else:
            config.reshape_dims = (n_visible,)
    return config

def prepare_rbm_network(config: NetworkConfig, n_visible: int) -> Any:
    """Prepare RBM network."""
    try:
        from QES.general_python.ml.net_impl.networks.net_rbm import RBM
        input_shape = (n_visible,)
        rbm = RBM(
            input_shape     = input_shape,
            n_hidden        = config.n_hidden,
            dtype           = config.dtype,
            param_dtype     = config.dtype,
            seed            = config.seed,
            visible_bias    = True,
            bias            = True
        )
        return rbm
    except ImportError:
        raise ImportError("RBM not available. JAX is required.")
    except Exception as e:
        raise RuntimeError(f"Failed to create RBM network: {e}")

def prepare_cnn_network(config: NetworkConfig, n_visible: int) -> Any:
    """Prepare CNN network."""
    try:
        from QES.general_python.ml.net_impl.networks.net_cnn import CNN
        from QES.general_python.ml.net_impl.activation_functions import elu_jnp
        
        if config.reshape_dims is None:
            side = int(math.sqrt(n_visible))
            if side * side == n_visible:
                config.reshape_dims = (side, side)
            else:
                config.reshape_dims = (n_visible,)
        
        input_shape = (n_visible,)
        cnn = CNN(
            input_shape         = input_shape,
            reshape_dims        = config.reshape_dims,
            features            = config.features,
            kernel_sizes        = config.kernel_sizes,
            strides             = config.strides,
            activations         = [elu_jnp] * len(config.features),
            dtype               = config.dtype,
            param_dtype         = config.dtype,
            final_activation    = elu_jnp,
            seed                = config.seed,
            output_shape        = (1,)
        )
        return cnn
    except ImportError:
        raise ImportError("CNN not available. JAX is required.")
    except Exception as e:
        raise RuntimeError(f"Failed to create CNN network: {e}")

def prepare_network(network_type: str, config: NetworkConfig, n_visible: int) -> Any:
    """Prepare neural network based on type."""
    if network_type == 'rbm':
        return prepare_rbm_network(config, n_visible)
    elif network_type == 'cnn':
        return prepare_cnn_network(config, n_visible)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

def generate_test_states(n_sites: int, num_states: int) -> np.ndarray:
    """Generate diverse test states for network evaluation."""
    states = np.zeros((num_states, n_sites), dtype=np.float32)
    
    if num_states >= 1:
        states[0, :] = 1                    # All ones
    if num_states >= 2:
        states[1, ::2] = 1                  # Checkerboard
    if num_states >= 3:
        states[2, :n_sites//2] = 1          # Half filled
    if num_states >= 4:
        # Random states
        for i in range(3, num_states):
            states[i] = np.random.randint(0, 2, n_sites, dtype=np.float32)
    
    return states

# =============================================================================
#! Main Example - Comprehensive Network Showcase
# =============================================================================

def main():
    """Run comprehensive network showcase."""
    
    print_section("NEURAL NETWORK ARCHITECTURE SHOWCASE", width=80)
    print("\nThis example demonstrates:")
    print("  • Creating different neural network architectures (RBM, CNN)")
    print("  • Using configuration classes for parameter management")
    print("  • Using NetworkFactory for unified network creation")
    print("  • Evaluating and comparing network outputs")
    print("  • Best practices for structured network code")
    
    # =========================================================================
    # STEP 1: Setup and Preparation
    # =========================================================================
    print_section("STEP 1: Setup and Preparation")
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        n_sites         = 8,
        batch_size      = 4,
        num_test_states = 5,
        network_types   = ['rbm', 'cnn'],
        seed            = 42
    )
    print_result("Simulation Config", True, f"{sim_config.n_sites} sites, {len(sim_config.network_types)} networks")
    
    # Prepare Hilbert space
    try:
        hilbert, n_visible = prepare_hilbert_space(sim_config.n_sites)
        print_result("Hilbert Space", True, f"{n_visible} states (2^{sim_config.n_sites})")
    except Exception as e:
        print_result("Hilbert Space", False, str(e))
        return
    
    # =========================================================================
    # STEP 2: Generate Test States
    # =========================================================================
    print_section("STEP 2: Generate Test States")
    
    try:
        test_states = generate_test_states(n_sites=sim_config.n_sites, num_states=sim_config.num_test_states)
        print_result("Test States", True, f"{test_states.shape} - diverse quantum states")
        print(f"\n  State examples:")
        for i, state in enumerate(test_states[:3]):
            print(f"    State {i+1}: {state.astype(int)}")
    except Exception as e:
        print_result("Test States", False, str(e))
        return
    
    # =========================================================================
    # STEP 3: Direct Network Creation (RBM and CNN)
    # =========================================================================
    print_section("STEP 3: Direct Network Creation")
    
    networks_created = {}
    
    # RBM Network
    print_subsection("Creating RBM Network")
    try:
        rbm_config = NetworkConfig(
            network_type    =   'rbm',
            n_sites         =   sim_config.n_sites,
            n_hidden        =   16
        )
        rbm_config              = prepare_network_config(rbm_config, n_visible)
        rbm_network             = prepare_rbm_network(rbm_config, n_visible)
        networks_created['rbm'] = rbm_network
        print_network_info('rbm', rbm_network, test_states)
        print_result("RBM Creation", True, "Success")
    except Exception as e:
        print_result("RBM Creation", False, str(e))
        traceback.print_exc()
    
    # CNN Network
    print_subsection("Creating CNN Network")
    try:
        cnn_config = NetworkConfig(
            network_type    =   'cnn',
            n_sites         =   sim_config.n_sites,
            features        =   [16, 32],
            kernel_sizes    =   [3, 3],
            strides         =   [1, 1]
        )
        cnn_config              = prepare_network_config(cnn_config, n_visible)
        cnn_network             = prepare_cnn_network(cnn_config, n_visible)
        networks_created['cnn'] = cnn_network
        print_network_info('cnn', cnn_network, test_states)
        print_result("CNN Creation", True, "Success")
    except Exception as e:
        print_result("CNN Creation", False, str(e))
        traceback.print_exc()
    
    # =========================================================================
    # STEP 4: Using NetworkFactory for Easy Creation
    # =========================================================================
    print_section("STEP 4: Using NetworkFactory for Easy Creation")
    
    try:
        available = NetworkFactory.list_available()
        print_result("Available Networks", True, str(available))
        
        if available:
            print_subsection("Creating with NetworkFactory")
            factory_networks = {}
            for net_type in available:
                try:
                    net = NetworkFactory.create(net_type, hilbert)
                    factory_networks[net_type] = net
                    print_result(f"Factory Create {net_type.upper()}", True, net.__class__.__name__)
                except Exception as e:
                    print_result(f"Factory Create {net_type.upper()}", False, str(e))
    except Exception as e:
        print_result("NetworkFactory", False, str(e))
    
    # =========================================================================
    # STEP 5: Getting Recommendations
    # =========================================================================
    print_section("STEP 5: Network Recommendations")
    
    print_subsection("Smart Network Selection")
    try:
        small_rec   = NetworkSelector.get_recommendations(8, 'general')
        medium_rec  = NetworkSelector.get_recommendations(16, 'lattice_2d')
        large_rec   = NetworkSelector.get_recommendations(100, 'large_system')

        print_result("Small system (8 qubits)", True, str(small_rec))
        print_result("Medium system (16 qubits, 2D)", True, str(medium_rec))
        print_result("Large system (100 qubits)", True, str(large_rec))
    except Exception as e:
        print_result("Recommendations", False, str(e))
    
    # =========================================================================
    # STEP 6: Network Comparison
    # =========================================================================
    print_section("STEP 6: Network Comparison")
    
    if networks_created:
        print_subsection("Evaluating All Created Networks")
        
        for net_type, network in networks_created.items():
            try:
                print_network_info(net_type, network, test_states)
            except Exception as e:
                print(f"  Error evaluating {net_type}: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY")
    
    print("\n(ok) Key Takeaways:")
    print("  1. Configuration classes provide clean parameter management")
    print("  2. Preparation functions make code modular and reusable")
    print("  3. NetworkFactory provides unified interface for different networks")
    print("  4. All networks share same evaluation interface")
    print("  5. Type hints and docstrings improve code maintainability")
    
    print(f"\n(ok) Networks created: {len(networks_created)}")
    if networks_created:
        print("  Available networks:", ", ".join(networks_created.keys()))
    
    print("\n" + "="*80)
    print("  For more information, see:")
    print("  • NETWORK_ARCHITECTURE_GUIDE.md - Comprehensive network reference")
    print("  • network_integration.py - NetworkFactory and NetworkSelector implementation")
    print("="*80 + "\n")

# =============================================================================

if __name__ == '__main__':
    main()

# =============================================================================
#! End of File
# =============================================================================