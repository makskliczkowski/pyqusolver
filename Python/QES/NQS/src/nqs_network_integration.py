"""
Neural Network Architectures for NQS (Integration Wrapper)
========================================================

This module serves as the NQS-specific interface for creating neural networks.
It delegates the actual creation to the core QES library's robust factory.

Available Networks:
-------------------
1. RBM (Restricted Boltzmann Machine)
2. CNN (Convolutional Neural Network)
3. ResNet (Deep Residual Network)
4. AR (Autoregressive Network)

-------------------------------------------------------------------------------
File        : NQS/src/nqs_network_integration.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-01
-------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

# Import the robust smart-factory from general_python
try:
    if TYPE_CHECKING:
        from QES.general_python.ml.networks import GeneralNet

    from QES.general_python.ml.net_impl.activation_functions import list_activations
    from QES.general_python.ml.networks import Networks, choose_network
except ImportError as e:
    raise ImportError(
        f"Could not import core QES network factory. Ensure QES is installed correctly.\nOriginal error: {e}"
    )

# ----------------------------------
#  Helpers for the user
# ----------------------------------


@dataclass
class NetworkInfo:
    """Metadata about available architectures."""

    name: str
    description: str
    best_for: str
    arguments: Dict[str, Any] = None


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
        "rbm": NetworkInfo(
            "RBM",
            "Restricted Boltzmann Machine",
            "General purpose, non-local correlations. Good starting point for many systems.",
            arguments={
                "input_shape": "Shape of the input layer (e.g., `(n_spins,)`)",
                "alpha": "Hidden unit density (float, e.g., 2.0)",
                "use_visible_bias": "Whether to use a bias on the visible layer (bool, default: True)",
                "use_hidden_bias": "Whether to use a bias on the hidden layer (bool, default: True)",
                "dtype": "Data type for weights ('float32', 'complex64', etc.)",
            },
        ),
        "cnn": NetworkInfo(
            "CNN",
            "Convolutional Neural Network",
            "Lattice systems with translational symmetry. Good for local correlations.",
            arguments={
                "input_shape": "Shape of the 1D input (e.g., `(n_spins,)`)",
                "reshape_dims": "Dimensions to reshape for convolution (e.g., `(8, 8)` for a 64-spin system)",
                "features": "List of channel counts for each conv layer (e.g., `[8, 16]`)",
                "kernel_sizes": "List of kernel sizes for each conv layer (e.g., `[3, 3]`)",
                "activations": "Activation function(s) for conv layers (e.g., 'relu', ['relu', 'tanh'])",
                "output_shape": "Shape of the final output (e.g., `(1,)` for log-amplitude)",
                "dtype": "Data type for weights ('float32', 'complex64', etc.)",
            },
        ),
        "ar": NetworkInfo(
            "Autoregressive",
            "Autoregressive Dense/RNN",
            "Large systems requiring exact sampling. It is useful when the sampler needs to be exact.",
            arguments={
                "input_shape": "Shape of the input layer (e.g., `(n_spins,)`)",
                "depth": "Number of layers in the autoregressive model (int)",
                "num_hidden": "Number of hidden units in each layer (int)",
                "rnn_type": "Type of recurrent cell if using RNN backend (e.g., 'lstm', 'gru')",
                "activations": "Activation function(s) for layers (e.g., 'relu')",
                "dtype": "Data type for weights ('float32', 'complex64', etc.)",
            },
        ),
        "resnet": NetworkInfo(
            "ResNet",
            "Deep Residual Network",
            "State-of-the-Art for 2D topological phases (Kitaev, frustrated magnets). Uses periodic convolutions and residual connections.",
            arguments={
                "input_shape": "Shape of the 1D input (e.g., `(n_spins,)`)",
                "reshape_dims": "Lattice dimensions for reshaping (e.g., `(8, 8)` for a 64-site system)",
                "features": "Number of feature channels / network width (int, default: 32)",
                "depth": "Number of residual blocks (int, default: 4)",
                "kernel_size": "Spatial kernel size (int or tuple, default: 3 -> (3,3) for 2D)",
                "dtype": "Data type for weights ('float32', 'complex128', etc.)",
            },
        ),
        "pp": NetworkInfo(
            "PairProduct",
            "Pair Product Ansatz",
            "Captures pairwise correlations via Pfaffian. Effective for fermions and frustrated spins.",
            arguments={
                "use_rbm": "Whether to augment with an RBM component (bool, default: True)",
                "input_shape": "Shape of the 1D input (e.g., `(n_spins,)`)",
                "init_scale": "Initialization scale for F matrix (float, default: 0.01)",
                "dtype": "Data type for weights ('float32', 'complex128', etc.)",
            },
        ),
        "activations": list_activations("jax"),
    }

    @staticmethod
    def create(
        network_type: str,
        input_shape: Tuple[int, ...],
        dtype: str = "complex128",
        backend: str = "jax",
        **kwargs,
    ) -> "GeneralNet":
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
                For 'cnn':
                - reshape_dims (Tuple[int, ...]) : The spatial dimensions to reshape the 1D input into (e.g., (8, 8)).
                - features (Sequence[int]) : Number of output channels for each convolutional layer.
                - kernel_sizes (Sequence[Union[int, Tuple]]) : Size of the kernel for each conv layer.
                - strides (Sequence[Union[int, Tuple]]) : Stride for each conv layer. Defaults to 1.
                - output_shape (Tuple[int, ...]) : Shape of the final output. Default: (1,),
                - activations (Union[str, Sequence[Union[str, Callable]]]) : Activation function(s) for each conv layer.
                - periodic (bool) : Whether to use periodic boundary conditions. Default-: True.
                - sum_pooling (bool) : Whether to sum pool the final output over spatial dimensions. Default: True.
                For 'rbm':
                - alpha (float) : Hidden unit density (n_hidden / n_visible).
                - use_visible_bias (bool) : Whether to use a bias on the visible layer. Default: True.
                - use_hidden_bias (bool) : Whether to use a bias on the hidden layer. Default: True.
                For 'ar':
                - depth (int) : Number of layers in the autoregressive model.
                - num_hidden (int) : Number of hidden units in each layer.
                - rnn_type (str) : Type of recurrent cell if using RNN backend ('lstm', 'gru').
                - activations (Union[str, Sequence[Union[str, Callable]]]) : Activation function(s) for each layer.
                For 'resnet':
                - reshape_dims (Tuple[int, ...]) : The spatial dimensions to reshape the 1D input into (e.g., (8, 8)).
                - features (int) : Number of feature channels / network width. Default: 32.
                - depth (int) : Number of residual blocks. Default: 4.
                - kernel_size (Union[int, Tuple]) : Spatial kernel size. Default: 3 (becomes (3,3) for 2D).
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
        return choose_network(
            network_type, input_shape=input_shape, dtype=dtype, backend=backend, **kwargs
        )

    @staticmethod
    def list_available() -> List[str]:
        """List all available network types."""
        return list(NetworkFactory._INFO.keys())

    @staticmethod
    def list_activations() -> List[str]:
        """List all available activation functions."""
        return NetworkFactory._INFO["activations"]

    @staticmethod
    def get_info(network_type: str) -> Dict[str, str]:
        """Get details about a specific network."""
        key = network_type.lower()

        if key in NetworkFactory._INFO:
            info = NetworkFactory._INFO[key]
            return {
                "name": info.name,
                "description": info.description,
                "best_for": info.best_for,
                "arguments": info.arguments or {},
            }
        return {"error": "Unknown network type"}

    @staticmethod
    def net_help():
        rbm = """
Restricted Boltzmann Machine (RBM) for NQS.

The RBM is a single-layer dense network. It connects all visible spins
to a layer of hidden units. It is the standard baseline for NQS.

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    # 1. Define RBM Parameters
    # ------------------------
    # An alpha (density) of 2 means: n_hidden = 2 * n_visible
    rbm_params = {
        'input_shape': (100,),       # 100 spins
        'alpha': 2,                  # Density of hidden units
        'use_bias': True,
        'dtype': 'complex128'        # Essential for quantum phases
    }
    
    # 2. Create the Network
    # ---------------------
    # 'rbm' key triggers the RBM class factory
    net = choose_network('rbm', **rbm_params)
    
    # 3. Initialize & Run
    # -------------------
    # Initialize with a random key (handled internally or explicitly)
    # params = net.init(jax.random.PRNGKey(0))
    # log_psi = net(params, sample_configuration)
"""
        cnn = """
Convolutional Neural Network (CNN) for NQS.

A deep architecture that respects the locality of physical interactions.
Essential for 2D frustrated systems (like Kitaev or J1-J2 models) where
local correlations are complex.

Features:
- Periodic Boundary Conditions (Torus geometry).
- Sum Pooling: Ensures energy is extensive (scales with N).
- Complex Weights: Captures the sign structure of the wavefunction.

Usage
-----
    from QES.general_python.ml.networks import choose_network
    import jax.numpy as jnp
    
    # 1. Define Lattice Geometry
    # --------------------------
    # For a 10x10 Lattice (100 spins)
    L = 10
    n_sites = L * L
    
    # 2. Define CNN Parameters
    # ------------------------
    cnn_params = {
        'input_shape':  (n_sites,),
        'reshape_dims': (L, L),          # Reshape 1D input to 2D grid
        'features':     (16, 32, 64),    # Deep network with increasing channels
        'kernel_sizes': ((3,3), (3,3), (3,3)),
        'activations':  ['lncosh'] * 3,  # Holomorphic activation
        'periodic':     True,            # Wrap edges (Torus)
        'sum_pooling':  True,            # Sum output over all spatial sites
        'dtype':        'complex128'
    }
    
    # 3. Create the Network
    # ---------------------
    net = choose_network('cnn', **cnn_params)
    
    # 4. Debug/Check
    # --------------
    print(f"Total Parameters: {net.nparams}")
    # > Total Parameters: ~25k (Complex)
"""
        ar = """
Autoregressive Network (AR) for NQS.
An autoregressive architecture that allows for exact sampling.
Useful for large systems where MCMC sampling is challenging.
Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    # 1. Define AR Parameters
    # -----------------------
    ar_params = {
        'input_shape': (100,),       # 100 spins
        'depth': 3,                  # Number of layers
        'num_hidden': 128,           # Hidden units per layer
        'rnn_type': 'lstm',          # RNN cell type
        'activations': 'relu',       # Activation function
        'dtype': 'complex128'        # Essential for quantum phases
    }
    
    # 2. Create the Network
    # ---------------------
    net = choose_network('ar', **ar_params)
    
    # 3. Initialize & Run
    # -------------------
    # Initialize with a random key (handled internally or explicitly)
    # params = net.init(jax.random.PRNGKey(0))
    # log_psi = net(params, sample_configuration)
"""
        res = """
Deep Residual Network (ResNet) for NQS.

State-of-the-Art architecture for 2D topological phases such as the Kitaev
Spin Liquid and frustrated magnets. Uses periodic convolutions and residual
connections to learn deep representations.

Features:
- Residual Connections: Enables very deep networks without vanishing gradients.
- Periodic Boundary Conditions: Respects the torus geometry of the lattice.
- Sum Pooling: Ensures extensive scaling with system size.
- Complex Weights: Captures the non-trivial sign structure of quantum states.

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    # 1. Define Lattice Geometry
    # --------------------------
    # For an 8x8 Lattice (64 sites)
    L = 8
    n_sites = L * L
    
    # 2. Define ResNet Parameters
    # ---------------------------
    resnet_params = {
        'input_shape':  (n_sites,),
        'reshape_dims': (L, L),          # Lattice dimensions
        'features':     32,              # Hidden channel width
        'depth':        4,               # Number of residual blocks
        'kernel_size':  3,               # Kernel size (becomes (3,3) for 2D)
        'dtype':        'complex128'     # Essential for quantum phases
    }
    
    # 3. Create the Network
    # ---------------------
    # Use 'resnet' key to create a ResNet instance
    net = choose_network('resnet', **resnet_params)
    
    # 4. Debug/Check
    # --------------
    print(net)
    # > ComplexResNet(reshape=(8, 8), features=32, depth=4, ...)
"""
        pp = """
Pair Product (PP) Ansatz for NQS.

Captures pairwise correlations in the wavefunction using a Pfaffian structure.
Particularly effective for fermionic systems and frustrated spins.

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    # 1. Define PP Parameters
    # -----------------------
    pp_params = {
        'input_shape'   : (64,),        # 64 sites
        'init_scale'    : 0.01,         # Small initialization
        'dtype'         : 'complex128'  # Essential for quantum phases
    }
    
    # 2. Create the Network
    # ---------------------
    net = choose_network('pp', **pp_params)
"""
        return {
            "rbm": rbm,
            "cnn": cnn,
            "ar": ar,
            "resnet": res,
            "pp": pp,
        }


# ----------------------------------
#! End of File
# ----------------------------------
