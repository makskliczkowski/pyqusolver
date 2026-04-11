"""
Public NQS-facing network factory and documentation metadata.

This module provides the `NetworkFactory` class, which serves as a centralized factory for creating neural 
quantum state (NQS) network instances based on user-friendly specifications. It abstracts away the details of individual network 
implementations and allows users to request architectures by name with relevant hyperparameters.

The factory also includes a registry of available network types, along with metadata describing their best use cases and expected arguments.

---------------
file        : QES/NQS/src/network/factory.py
author      : Maksymilian Kliczkowski
---------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from QES.NQS.ansatze                                        import EquivariantGCNN, resolve_ansatz_request
    from QES.general_python.ml.net_impl.activation_functions    import list_activations
    from QES.general_python.ml.networks                         import choose_network

    from .representation                                        import ModelRepresentationInfo, apply_nqs_representation_overrides, canonical_network_request_family
except ImportError as e:
    raise ImportError("Required QES modules are not available.") from e

@dataclass
class NetworkInfo:
    """Metadata about available architectures."""
    name        : str
    description : str
    best_for    : str
    arguments   : Dict[str, Any] = None


class NetworkFactory:
    """
    NQS-specific factory for creating neural quantum states.
    """

    _INFO = {
        "rbm": NetworkInfo(
            "RBM",
            "Restricted Boltzmann Machine [R1] Carleo & Troyer 2017",
            "General purpose baseline. alpha=1->2->4 for increasing accuracy. Complex weights for frustrated/topological.",
            {
                "input_shape"       : "Shape of the input layer (e.g., `(n_spins,)`)",
                "alpha"             : "Hidden unit density (float, e.g., 2.0)",
                "use_visible_bias"  : "Whether to use a bias on the visible layer (bool, default: True)",
                "use_hidden_bias"   : "Whether to use a bias on the hidden layer (bool, default: True)",
                "dtype"             : "Data type for weights ('float32', 'complex64', etc.)",
            },
        ),
        "cnn": NetworkInfo(
            "CNN",
            "Convolutional Neural Network [R2] Choo, Neupert & Carleo 2019",
            "Frustrated 2D lattice systems (J1-J2). 6 layers, descending channels, complex-ReLU/log_cosh.",
            {
                "input_shape"       : "Shape of the 1D input (e.g., `(n_spins,)`)",
                "reshape_dims"      : "Dimensions to reshape for convolution (e.g., `(8, 8)` for a 64-spin system)",
                "features"          : "List of channel counts for each conv layer (e.g., `[8, 16]`)",
                "kernel_sizes"      : "List of kernel sizes for each conv layer (e.g., `[3, 3]`)",
                "activations"       : "Activation function(s) for conv layers (e.g., 'relu', ['relu', 'tanh'])",
                "output_shape"      : "Shape of the final output (e.g., `(1,)` for log-amplitude)",
                "dtype"             : "Data type for weights ('float32', 'complex64', etc.)",
            },
        ),
        "ar": NetworkInfo(
            "Autoregressive",
            "Autoregressive NAQS / RNN [R3] Sharir 2020, [R4] Hibat-Allah 2020",
            "Large systems requiring exact (uncorrelated) sampling. Staged training: Adam->SGD+momentum.",
            {
                "input_shape"       : "Shape of the input layer (e.g., `(n_spins,)`)",
                "depth"             : "Number of layers in the autoregressive model (int)",
                "num_hidden"        : "Number of hidden units in each layer (int)",
                "rnn_type"          : "Type of recurrent cell if using RNN backend (e.g., 'lstm', 'gru')",
                "activations"       : "Activation function(s) for layers (e.g., 'relu')",
                "dtype"             : "Data type for weights ('float32', 'complex64', etc.)",
            },
        ),
        "resnet": NetworkInfo(
            "ResNet",
            "Deep Residual Network [R5] Roth, Szabo & MacDonald 2023",
            "SOTA for 2D frustrated magnets. LayerSR + Lanczos post-processing. Periodic convolutions, residual connections.",
            {
                "input_shape"       : "Shape of the 1D input (e.g., `(n_spins,)`)",
                "reshape_dims"      : "Lattice dimensions for reshaping (e.g., `(8, 8)` for a 64-site system)",
                "features"          : "Number of feature channels / network width (int, default: 32)",
                "depth"             : "Number of residual blocks (int, default: 4)",
                "kernel_size"       : "Spatial kernel size (int or tuple, default: 3 -> (3,3) for 2D)",
                "dtype"             : "Data type for weights ('float32', 'complex128', etc.)",
            },
        ),
        "pp": NetworkInfo(
            "PairProduct",
            "Pair Product / Pfaffian Ansatz",
            "O(Ns^2) F-matrix for pairwise correlations. Natural for fermionic / RVB states.",
            {
                "use_rbm"           : "Whether to augment with an RBM component (bool, default: True)",
                "input_shape"       : "Shape of the 1D input (e.g., `(n_spins,)`)",
                "init_scale"        : "Initialization scale for F matrix (float, default: 0.01)",
                "dtype"             : "Data type for weights ('float32', 'complex128', etc.)",
            },
        ),
        "eqgcnn": NetworkInfo(
            "EquivariantGCNN",
            "Group-Equivariant CNN [R5] Roth et al. 2023, arXiv:2505.23728",
            "Systems with space-group symmetries. L=2 layers sufficient for fidelity >0.99999 with ED. LayerSR for deep nets.",
            {
                "input_shape"       : "Shape of the input (e.g., `(n_sites,)`)",
                "symmetry_perms"    : "Permutation table (|G|, Ns) from generate_space_group_perms()",
                "channels"          : "Channel widths per layer (e.g., `(8, 8)` for 2 layers)",
                "split_complex"     : "Use real-valued backbone with split complex output (bool)",
                "dtype"             : "Data type for weights ('complex128', etc.)",
            },
        ),
        "approx_symmetric": NetworkInfo(
            "ApproxSymmetric",
            "Approximately symmetric Chi -> Omega -> Sigma ansatz",
            "Geometry-aware combo architecture: non-invariant site block, plaquette-invariant Wilson map, and invariant dual-plaquette block.",
            {
                "chi_channels"      : "Non-invariant block channels (Table S2 style: `(1,2,4)`).",
                "omega_channels"    : "Invariant block channels (Table S2 style: `(4,4,4)`).",
                "chi_kernel_size"   : "Site-neighborhood kernel size for Chi (default: `3`).",
                "omega_kernel_size" : "Dual-plaquette kernel size for Omega (default: `15` for high-accuracy GPU runs).",
                "nib_act"           : "Non-invariant activation (default: `'c_sigmoid'`).",
                "ib_act"            : "Invariant activation (default: `'c_elu'`).",
                "lattice_type"      : "Geometry source (e.g., `'honeycomb'`).",
                "lattice_shape"     : "Unit-cell lattice size `(Lx, Ly)` for geometry construction.",
                "dtype"             : "Data type for weights ('complex64', 'complex128', etc.).",
            },
        ),
        "activations": list_activations("jax"),
    }

    @staticmethod
    def create(
        network_type        : Any,
        input_shape         : Tuple[int, ...],
        dtype               : str = "complex128",
        backend             : str = "jax",
        param_dtype         : Any = None,
        representation_info : Optional[ModelRepresentationInfo] = None,
        **kwargs,
    ):
        """
        Create a network instance using the general QES factory plus NQS overrides.
        This method resolves the requested network type and applies any necessary overrides based on the provided representation info.
        It then delegates to the general `choose_network` function to construct the actual network instance.
        
        Parameters
        ----------
        network_type : str
            The type of network to create (e.g., 'rbm', 'cnn', 'ar', 'resnet', 'pp', 'eqgcnn', 'approx_symmetric').
        input_shape : Tuple[int, ...]
            The shape of the input layer (e.g., `(n_spins,)`).
        dtype : str, optional
            The data type for the network weights (e.g., 'complex128'), by default "complex128".
        backend : str, optional
            The computational backend to use (e.g., 'jax', 'torch'), by default "jax".
        param_dtype : Any, optional
            The data type for the network parameters, if different from `dtype`. If None, defaults to `dtype`.
        representation_info : Optional[ModelRepresentationInfo], optional
            Additional information about the model's state representation that may influence network construction, by default None.
        **kwargs
            Additional keyword arguments specific to the requested network type.
            
        Returns
        -------
        An instance of the requested network architecture, initialized with the provided parameters.
        """
        
        resolved_kwargs                 = apply_nqs_representation_overrides(network_type, representation_info, kwargs)
        resolved_type, resolved_kwargs  = resolve_ansatz_request(network_type, resolved_kwargs)
        family                          = canonical_network_request_family(resolved_type)

        if family == "eqgcnn" and "symmetry_perms" not in resolved_kwargs:
            lattice = resolved_kwargs.pop("lattice", None)
            if lattice is not None:
                channels = resolved_kwargs.pop("channels", (8, 8))
                point_group = resolved_kwargs.pop("point_group", "full")
                return EquivariantGCNN.from_lattice(
                    lattice,
                    channels=channels,
                    point_group=point_group,
                    dtype=dtype,
                    backend=backend,
                    **resolved_kwargs,
                )
            raise ValueError(
                "eqgcnn construction requires either `symmetry_perms` or a lattice object "
                "so symmetry permutations can be derived automatically."
            )

        return choose_network(
            resolved_type,
            input_shape=input_shape,
            dtype=dtype,
            param_dtype=param_dtype,
            backend=backend,
            **resolved_kwargs,
        )

    @staticmethod
    def list_available() -> List[str]:
        return list(NetworkFactory._INFO.keys())

    @staticmethod
    def list_activations() -> List[str]:
        return NetworkFactory._INFO["activations"]

    @staticmethod
    def get_info(network_type: str) -> Dict[str, Any]:
        key = network_type.lower().replace("-", "_")
        if key == "gcnn":
            key = "eqgcnn"
        if key in ("approxsym", "asym"):
            key = "approx_symmetric"
        if key in NetworkFactory._INFO:
            info = NetworkFactory._INFO[key]
            return {
                "name"          : info.name,
                "description"   : info.description,
                "best_for"      : info.best_for,
                "arguments"     : info.arguments or {},
            }
        return {"error": "Unknown network type"}

    @staticmethod
    def net_help():
        rbm = """
Restricted Boltzmann Machine (RBM) for NQS [R1].

Usage
-----
    from QES.NQS.src.network import NetworkFactory
    net = NetworkFactory.create(network_type='rbm', input_shape=(100,), alpha=2, dtype='complex128')
"""
        cnn = """
Convolutional Neural Network (CNN) for NQS [R2].

Usage
-----
    from QES.NQS.src.network import NetworkFactory
    net = NetworkFactory.create(network_type='cnn', input_shape=(100,), reshape_dims=(10, 10), features=(12, 10, 8, 6, 4, 2), kernel_sizes=((3, 3),) * 6, activations=['lncosh'] * 6, periodic=True, sum_pooling=True, dtype='complex128')
"""
        ar = """
Autoregressive Network (AR / NAQS / RNN) for NQS [R3, R4].

Usage
-----
    from QES.NQS.src.network import NetworkFactory
    net = NetworkFactory.create(network_type='ar', input_shape=(100,), depth=3, num_hidden=128, rnn_type='lstm', activations='relu', dtype='complex128')
"""
        res = """
Deep Residual Network (ResNet) for NQS [R5].

Usage
-----
    from QES.NQS.src.network import NetworkFactory
    net = NetworkFactory.create(network_type='resnet', input_shape=(64,), reshape_dims=(8, 8), features=32, depth=4, kernel_size=3, dtype='complex128')
"""
        pp = """
Pair Product (PP) Ansatz for NQS.

Usage
-----
    from QES.NQS.src.network import NetworkFactory
    net = NetworkFactory.create(network_type='pp', input_shape=(64,), init_scale=0.01, dtype='complex128')
"""
        eqgcnn = """
Group-Equivariant CNN (EquivariantGCNN) for NQS.

Usage
-----
    from QES.NQS.src.network import NetworkFactory
    net = NetworkFactory.create(network_type='eqgcnn', input_shape=(Ns,), lattice=lattice, channels=(8, 8), point_group='full', dtype='complex128')
"""
        return {
            "rbm": rbm,
            "cnn": cnn,
            "ar": ar,
            "resnet": res,
            "pp": pp,
            "eqgcnn": eqgcnn,
        }


__all__ = ["NetworkFactory", "NetworkInfo"]
