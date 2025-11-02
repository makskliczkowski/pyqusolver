"""
Neural Network Architectures for NQS
=====================================

This module provides utilities and examples for using different neural network
architectures with the Neural Quantum State (NQS) solver.

Available Networks:
-------------------
1. RBM (Restricted Boltzmann Machine)
   - Best for: General quantum systems, flexibility
   - Advantages: Expressive, well-studied, fast
   - Disadvantages: Scales as O(N_v * N_h) parameters
   - Use when: You need a general-purpose network

2. CNN (Convolutional Neural Network)
   - Best for: Lattice systems with local structure
   - Advantages: Exploits spatial locality, fewer parameters
   - Disadvantages: Requires spatial lattice structure
   - Use when: Your system has translational symmetry

3. Autoregressive (AR) Network
   - Best for: Large systems, parameter efficiency
   - Advantages: Sequential factorization, flexible, parameter-efficient
   - Disadvantages: Slower evaluation due to sequential generation
   - Use when: You need maximum parameter efficiency for large systems

Author: Development Team
Date: November 1, 2025
"""

import numpy as np
from typing import Optional, Tuple, Union, Any, List, Dict
from dataclasses import dataclass

# Quantum system imports
try:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.hamil import Hamiltonian
except ImportError:
    HilbertSpace = None
    Hamiltonian = None

# Network imports
try:
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
    HAS_RBM = True
except ImportError:
    HAS_RBM = False
    RBM = None

try:
    from QES.general_python.ml.net_impl.networks.net_cnn import CNN
    HAS_CNN = True
except ImportError:
    HAS_CNN = False
    CNN = None

try:
    from QES.general_python.ml.net_impl.networks.net_autoregressive import Autoregressive
    HAS_AR = True
except ImportError:
    HAS_AR = False
    Autoregressive = None


@dataclass
class NetworkInfo:
    """Information about a network architecture."""
    name: str
    description: str
    param_count: int
    suitable_for: List[str]
    scaling: str
    properties: Dict[str, Any]


class NetworkFactory:
    """Factory for creating different network architectures."""
    
    AVAILABLE_NETWORKS = {
        'rbm': {
            'name': 'RBM',
            'class': RBM if HAS_RBM else None,
            'description': 'Restricted Boltzmann Machine',
            'requires': ['hilbert_space'],
            'optional': ['num_hidden', 'use_bias', 'visible_bias']
        },
        'cnn': {
            'name': 'CNN',
            'class': CNN if HAS_CNN else None,
            'description': 'Convolutional Neural Network',
            'requires': ['hilbert_space'],
            'optional': ['features', 'kernel_sizes', 'strides', 'activations']
        },
        'ar': {
            'name': 'Autoregressive',
            'class': Autoregressive if HAS_AR else None,
            'description': 'Autoregressive Neural Network',
            'requires': ['hilbert_space'],
            'optional': ['hidden_layers', 'activation', 'output_activation']
        },
    }
    
    @staticmethod
    def list_available() -> List[str]:
        """List all available network types."""
        available = []
        for key, info in NetworkFactory.AVAILABLE_NETWORKS.items():
            if info['class'] is not None:
                available.append(key)
        return available
    
    @staticmethod
    def create(network_type: str, hilbert_space: HilbertSpace, **kwargs) -> Any:
        """Create a network instance."""
        if network_type not in NetworkFactory.AVAILABLE_NETWORKS:
            raise ValueError(f"Unknown network type: {network_type}")
        
        info = NetworkFactory.AVAILABLE_NETWORKS[network_type]
        
        if info['class'] is None:
            raise ValueError(f"Network type '{network_type}' is not installed")
        
        network_class = info['class']
        n_visible = 2**hilbert_space.Ns
        input_shape = (n_visible,)
        
        try:
            if network_type == 'rbm':
                return network_class(input_shape=input_shape, **kwargs)
            elif network_type == 'cnn':
                # For CNN, need reshape_dims. Default to sqrt(n_visible) x sqrt(n_visible)
                import math
                side = int(math.sqrt(n_visible))
                if side * side != n_visible:
                    # If not square, try 1D
                    reshape_dims = (n_visible,)
                else:
                    reshape_dims = (side, side)
                return network_class(input_shape=input_shape, reshape_dims=reshape_dims, **kwargs)
            elif network_type == 'ar':
                # For autoregressive, use n_visible directly
                return network_class(input_shape=(n_visible,), **kwargs)
            else:
                raise ValueError(f"Unknown network type: {network_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to create {network_type} network: {e}")
    
    @staticmethod
    def get_info(network_type: str) -> Dict[str, Any]:
        """Get information about a network type."""
        if network_type not in NetworkFactory.AVAILABLE_NETWORKS:
            raise ValueError(f"Unknown network type: {network_type}")
        return NetworkFactory.AVAILABLE_NETWORKS[network_type]


class NetworkSelector:
    """Helper for selecting appropriate networks for different scenarios."""
    
    RECOMMENDATIONS = {
        'small_system': {'size_range': (1, 10), 'recommended': ['rbm']},
        'medium_system': {'size_range': (10, 50), 'recommended': ['rbm', 'cnn']},
        'large_system': {'size_range': (50, 1000), 'recommended': ['ar', 'cnn']},
        'very_large_system': {'size_range': (1000, 10000), 'recommended': ['ar']},
        'lattice_1d': {'recommended': ['cnn', 'rbm', 'ar']},
        'lattice_2d': {'recommended': ['cnn', 'rbm']},
        'arbitrary_topology': {'recommended': ['rbm', 'ar']},
    }
    
    @staticmethod
    def get_recommendations(system_size: int, system_structure: str = 'general') -> List[str]:
        """Get network recommendations for a system."""
        if system_structure in NetworkSelector.RECOMMENDATIONS:
            return NetworkSelector.RECOMMENDATIONS[system_structure]['recommended']
        
        for size_key in ['very_large_system', 'large_system', 'medium_system', 'small_system']:
            size_range = NetworkSelector.RECOMMENDATIONS[size_key]['size_range']
            if size_range[0] <= system_size <= size_range[1]:
                return NetworkSelector.RECOMMENDATIONS[size_key]['recommended']
        
        return NetworkFactory.list_available()
