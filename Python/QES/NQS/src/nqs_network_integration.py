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
File        : NQS/src/network_integration.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-01
-------------------------------------------------------------------------------
"""

from typing import Any, List, Dict
from dataclasses import dataclass

# Import the robust smart-factory from general_python
try:
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
            "General purpose, non-local correlations"
        ),
        'cnn': NetworkInfo(
            "CNN", 
            "Convolutional Neural Network", 
            "Lattice systems with translational symmetry"
        ),
        'ar': NetworkInfo(
            "Autoregressive", 
            "Autoregressive Dense/RNN", 
            "Large systems requiring exact sampling"
        ),
    }

    @staticmethod
    def create(network_type: str, **kwargs) -> Any:
        """
        Creates a network instance using the core QES factory.
        
        Args:
            network_type (str): 'rbm', 'cnn', 'ar', 'simple'
            **kwargs: Arguments passed to the network constructor 
                      (e.g. input_shape, alpha, kernel_size)
        
        Returns:
            A GeneralNet compatible instance (usually FlaxInterface).
        """
        # Delegate to the robust implementation in general_python
        return choose_network(network_type, **kwargs)

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
                "best_for"      : info.best_for
            }
        return {"error": "Unknown network type"}

# ----------------------------------
#! End of File
# ----------------------------------