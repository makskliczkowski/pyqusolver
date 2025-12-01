"""
Neural Network Architectures for NQS (Integration Wrapper)
========================================================

This module serves as the NQS-specific interface for creating neural networks.
It delegates the actual creation to the core QES library's robust factory.

Available Networks:
-------------------
1. RBM (Restricted Boltzmann Machine)
2. CNN (Convolutional Neural Network)
3. AR (Autoregressive Network)

-------------------------------------------------------------------------------
File        : NQS/src/nqs_network_integration.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-01
-------------------------------------------------------------------------------
"""

from typing import Any, List, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# Import the robust smart-factory from general_python
try:
    if TYPE_CHECKING:
        from QES.general_python.ml.networks import GeneralNet
        
    from QES.general_python.ml.networks import choose_network, Networks
except ImportError as e:
    raise ImportError(f"Could not import core QES network factory. Ensure QES is installed correctly.\nOriginal error: {e}")

# ----------------------------------
#  Helpers for the user
# ----------------------------------

@dataclass
class NetworkInfo:
    """Metadata about available architectures."""
    name        : str
    description : str
    best_for    : str
    arguments   : Dict[str, Any] = None

# ----------------------------------
# The Factory Wrapper
# ----------------------------------

class NetworkFactory:
    """
    NQS-specific factory for creating Neural Quantum States.
    Wraps QES.general_python.ml.networks.choose_network.
    """
    
    # Metadata for documentation/UI purposes
    _INFO = {
        'rbm': NetworkInfo(
            "RBM", 
            "Restricted Boltzmann Machine", 
            "General purpose, non-local correlations. Good starting point for many systems.",
            arguments = {
                "input_shape"       : "Shape of the input layer (e.g., `(n_spins,)`)",
                "alpha"             : "Hidden unit density (float, e.g., 2.0)",
                "use_visible_bias"  : "Whether to use a bias on the visible layer (bool, default: True)",
                "use_hidden_bias"   : "Whether to use a bias on the hidden layer (bool, default: True)",
                "dtype"             : "Data type for weights ('float32', 'complex64', etc.)",
            }
        ),
        'cnn': NetworkInfo(
            "CNN", 
            "Convolutional Neural Network", 
            "Lattice systems with translational symmetry. Good for local correlations.",
            arguments = {
                "input_shape"       : "Shape of the 1D input (e.g., `(n_spins,)`)",
                "reshape_dims"      : "Dimensions to reshape for convolution (e.g., `(8, 8)` for a 64-spin system)",
                "features"          : "List of channel counts for each conv layer (e.g., `[8, 16]`)",
                "kernel_sizes"      : "List of kernel sizes for each conv layer (e.g., `[3, 3]`)",
                "activations"       : "Activation function(s) for conv layers (e.g., 'relu', ['relu', 'tanh'])",
                "output_shape"      : "Shape of the final output (e.g., `(1,)` for log-amplitude)",
                "dtype"             : "Data type for weights ('float32', 'complex64', etc.)",
            }
        ),
        'ar': NetworkInfo(
            "Autoregressive", 
            "Autoregressive Dense/RNN", 
            "Large systems requiring exact sampling. It is useful when the sampler needs to be exact.",
            arguments = {
                "input_shape"       : "Shape of the input layer (e.g., `(n_spins,)`)",
                "depth"             : "Number of layers in the autoregressive model (int)",
                "num_hidden"        : "Number of hidden units in each layer (int)",
                "rnn_type"          : "Type of recurrent cell if using RNN backend (e.g., 'lstm', 'gru')",
                "activations"       : "Activation function(s) for layers (e.g., 'relu')",
                "dtype"             : "Data type for weights ('float32', 'complex64', etc.)",
            }
        ),
    }

    @staticmethod
    def create(network_type: str, input_shape: Tuple[int, ...], dtype: str = 'complex128', backend: str = 'jax', **kwargs) -> 'GeneralNet':
        """
        Creates a network instance using the core QES factory.
        
        Args:
            network_type (str): 
                'rbm', 'cnn', 'ar', 'simple'
            input_shape (Tuple[int, ...]): 
                Shape of the input layer
            dtype (str): 
                Data type for the network weights
            backend (str): 
                Backend to use ('jax', 'tensorflow', etc.)    
        
            **kwargs: 
                Arguments passed to the network constructor 
                (e.g. alpha, kernel_size)
        
        Returns:
            A GeneralNet compatible instance (usually FlaxInterface).
            
        Examples:
        ---------
            >>> # Create a real-valued RBM
            >>> rbm_net = NetworkFactory.create(
            ...     network_type    =   'rbm',
            ...     input_shape     =   (100,),
            ...     alpha           =   2.0
            ... )
            
            >>> # Create a complex-valued CNN for a 10x10 lattice
            >>> cnn_net = NetworkFactory.create(
            ...     network_type    =   'cnn',
            ...     input_shape     =   (100,),
            ...     reshape_dims    =   (10, 10),
            ...     features        =   [8, 16],
            ...     kernel_sizes    =   [3, 3],
            ...     activations     =   ['relu', 'relu'],
            ...     output_shape    =   (1,),
            ...     dtype           =   'complex64'
            ... )
        """
        # Delegate to the robust implementation in general_python
        return choose_network(network_type, input_shape=input_shape, dtype=dtype, backend=backend, **kwargs)

    @staticmethod
    def list_available() -> List[str]:
        """List all available network types."""
        return list(NetworkFactory._INFO.keys())

    @staticmethod
    def get_info(network_type: str) -> Dict[str, str]:
        """Get details about a specific network."""
        key = network_type.lower()
        
        if key in NetworkFactory._INFO:
            info = NetworkFactory._INFO[key]
            return {
                "name"          : info.name,
                "description"   : info.description,
                "best_for"      : info.best_for,
                "arguments"     : info.arguments or {}
            }
        return {"error": "Unknown network type"}

# ----------------------------------
#! End of File
# ----------------------------------