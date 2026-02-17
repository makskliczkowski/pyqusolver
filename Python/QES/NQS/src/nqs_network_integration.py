"""
Neural Network Architectures for NQS (Integration Wrapper)
========================================================

This module serves as the NQS-specific interface for creating neural networks.
It delegates the actual creation to the core QES library's robust factory.

Available Networks:
-------------------
1. RBM      (Restricted Boltzmann Machine)
2. CNN      (Convolutional Neural Network)
3. ResNet   (Deep Residual Network)
4. AR       (Autoregressive Network)
5. PP       (Pair Product / Pfaffian Ansatz)
6. RBMPP    (RBM + Pair Product Hybrid)
7. GCNN     (Graph Convolutional Neural Network)
8. EqGCNN   (Group-Equivariant CNN — arXiv:2505.23728)

-------------------------------------------------------------------------------
File        : NQS/src/nqs_network_integration.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-01
Version     : 2.2.0
-------------------------------------------------------------------------------
"""

import  math
from    dataclasses import dataclass, field
from    typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import  numpy as _np

# Import the robust smart-factory from general_python
try:
    if TYPE_CHECKING:
        from QES.general_python.ml.networks import GeneralNet

    from QES.general_python.ml.net_impl.activation_functions import list_activations
    from QES.general_python.ml.networks import Networks, choose_network
except ImportError as e:
    raise ImportError(f"Could not import core QES network factory. Ensure QES is installed correctly.\nOriginal error: {e}")

# Re-export EquivariantGCNN and symmetry utilities from their canonical locations
# so existing code using ``from nqs_network_integration import ...`` still works.
try:
    from QES.general_python.ml.net_impl.networks.net_gcnn   import EquivariantGCNN  # noqa: F401
    from QES.general_python.lattices.tools.lattice_symmetry import (                # noqa: F401
        generate_translation_perms,
        generate_point_group_perms_square,
        generate_space_group_perms,
        compute_cayley_table,
    )
    _HAS_JAX_EQGCNN = True
except ImportError:
    _HAS_JAX_EQGCNN = False

# ===========================================================================
#! Network Parameter Estimator
# ===========================================================================

@dataclass
class SOTANetConfig:
    r"""
    Hyperparameter configuration for NQS ansätze.
    
    Produced by ``estimate_network_params()`` based on system size, lattice
    geometry, and model physics.  Defaults are drawn from the following
    profiled sources:

    [R1] Carleo & Troyer, Science 355 602 (2017) — RBM baseline + SR.
    [R2] Choo, Neupert & Carleo, PRB 100 125124 (2019) — Complex CNN, J1-J2.
    [R3] Sharir et al., PRL 124 020503 (2020) — Autoregressive NAQS (PixelCNN).
    [R4] Hibat-Allah et al., PRR 2 023358 (2020) — RNN wavefunctions.
    [R5] Roth, Szabó & MacDonald, PRB 108 054410 (2023) — Deep GCNN + LayerSR.
    [R6] Chen & Heyl, Nat. Phys. (2024) — MinSR for deep NQS.
    [R7] Pfau, Spencer, Matthews & Foulkes, PRR 2 033429 (2020) — FermiNet.
    [R8] Sprague & Czischek, arXiv:2306.03921 (2024) — Patched Transformers.

    Architecture families and their scaling:

    - **RBM** [R1]: Single hidden layer, $\alpha = M/N$ hidden-unit density.
      Performance improves systematically with $\alpha$ (1 -> 2 -> 4).
      Complex weights essential for frustrated / topological systems.
      SR diagonal regularization schedule:
      $\lambda(p) = \max(\lambda_0 b^p, \lambda_{\min})$ with
      $(\lambda_0, b, \lambda_{\min}) = (100, 0.9, 10^{-4})$ [R1].
    - **CNN** [R2]: Deep convolutional with complex-ReLU / $\log\cosh$.
      6 layers, channels 12->10->8->6->4->2, ~3838 complex params for J1-J2.
      Sign-structure initialization critical for frustrated phases.
    - **ResNet/GCNN** [R5]: Deep residual + group-equivariant layers.
      LayerSR avoids full Jacobian storage; sample budget
      1024 -> 4096 -> 16384 as depth grows.  4–16 layers with 6–12
      group-indexed feature maps.
    - **AR/MADE** [R3]: Masked-conv (PixelCNN) blocks; 3x3 filters,
      32 channels, 10–40 blocks. Exact autoregressive sampling;
      staged Adam -> SGD+momentum; batch 100 -> 1000 -> larger.
    - **RNN** [R4]: GRU cells, $d_h$ memory units; batch 500;
      schedules $(\eta^{-1}+0.1t)^{-1}$ or $\eta(1+t/5000)^{-1}$.
    - **MinSR** [R6]: Enables deep nets up to 64 layers / $10^6$ params;
      cost linear in number of parameters.
    - **PairProduct / Pfaffian**: $O(N_s^2)$ F-matrix, natural for
      fermionic / RVB states.  Hybridised RBM+PP gives best of both.

    Every field can be overridden via ``**kwargs`` in
    ``estimate_network_params``; see its docstring for per-architecture
    keyword arguments.
    """
    net_type        : str
    input_shape     : Tuple[int, ...]
    reshape_dims    : Optional[Tuple[int, ...]]     = None
    alpha           : Optional[float]               = None   # RBM hidden density
    
    features        : Optional[Any]                 = None   # CNN/ResNet channels  
    depth           : Optional[int]                 = None   # CNN/ResNet depth
    kernel_size     : Optional[Any]                 = None   # CNN/ResNet kernel
    
    ar_hidden       : Optional[Tuple[int, ...]]     = None   # AR hidden dims
    phase_hidden    : Optional[Tuple[int, ...]]     = None   # AR phase dims
    
    activations     : Optional[Any]                 = None
    periodic        : bool                          = True
    dtype           : str                           = "complex128"
    split_complex   : bool                          = False
    extra_kwargs    : Dict[str, Any]                = field(default_factory=dict)
    description     : str                           = ""
    estimated_params: int                           = 0
    
    def to_factory_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs suitable for NetworkFactory.create()."""
        kw = {
            "input_shape"   : self.input_shape,
            "dtype"         : self.dtype,
        }
        if self.net_type == "rbm":
            if self.alpha is not None:
                kw["alpha"]         = self.alpha
                
        elif self.net_type == "cnn":
            if self.reshape_dims:
                kw["reshape_dims"]  = self.reshape_dims
            if self.features:
                kw["features"]      = self.features
            if self.kernel_size:
                kw["kernel_sizes"]  = self.kernel_size
            if self.activations:
                kw["activations"]   = self.activations
            kw["periodic"]          = self.periodic
            if self.split_complex:
                kw["split_complex"] = True
                
        elif self.net_type == "resnet":
            if self.reshape_dims:
                kw["reshape_dims"]  = self.reshape_dims
            if self.features:
                kw["features"]      = self.features
            if self.depth:
                kw["depth"]         = self.depth
            if self.kernel_size:
                kw["kernel_size"]   = self.kernel_size
            kw["periodic_boundary"] = self.periodic
            
        elif self.net_type == "ar":
            if self.ar_hidden:
                kw["ar_hidden"]     = self.ar_hidden
            if self.phase_hidden:
                kw["phase_hidden"]  = self.phase_hidden
                
        elif self.net_type in ("pp", "rbmpp"):
            if self.alpha:
                kw["alpha"]         = self.alpha
            kw["use_rbm"]           = (self.net_type == "rbmpp")
            
        kw.update(self.extra_kwargs)
        
        return kw

def estimate_network_params(
    net_type        : str,
    num_sites       : int,
    lattice_dims    : Optional[Tuple[int, ...]]     = None,
    lattice_type    : str                           = "honeycomb",
    model_type      : str                           = "kitaev",
    *,
    target_accuracy : str                           = "high",
    dtype           : str                           = "complex128",
    max_params      : Optional[int]                 = None,
    **kwargs
) -> SOTANetConfig:
    r"""
    Estimate network hyperparameters for a given system.

    Returns a ``SOTANetConfig`` whose defaults are drawn from the profiled
    sources listed in the class docstring.  Every architecture-specific
    default can be overridden via ``**kwargs``.

    Parameters
    ----------
    net_type : str
        Architecture: ``'rbm'``, ``'cnn'``, ``'resnet'``, ``'ar'``,
        ``'pp'``, ``'rbmpp'``, ``'eqgcnn'`` / ``'gcnn'``.
    num_sites : int
        Total number of lattice sites $N_s$.
    lattice_dims : tuple of int, optional
        ``(Lx, Ly)`` or ``(Lx, Ly, Lz)`` for spatial reshaping.
        Inferred from *num_sites* when omitted.
    lattice_type : str
        ``'honeycomb'``, ``'square'``, ``'chain'``.
    model_type : str
        ``'kitaev'``, ``'heisenberg'``, ``'j1j2'``, ``'tfi'``.
    target_accuracy : str
        ``'fast'`` (x 0.5), ``'medium'`` (x 1), ``'high'`` (x 2) —
        multiplicative tier applied to base hyperparameters.
    dtype : str
        Parameter dtype; auto-promoted to ``'complex128'`` for frustrated
        models.
    max_params : int, optional
        Hard cap on total parameter count (RBM only at present).

    **kwargs — User Overrides (per architecture)

    Any keyword below, when supplied, *replaces* the estimated default.
    Tier scaling is suppressed for explicitly overridden values.

    **RBM** (hidden-unit density):
        ``alpha``            : float — $\alpha = M / N$; base values
                               kitaev 4, heisenberg 2, j1j2 4, tfi 1 [R1].

    **CNN** (convolutional layers):
        ``n_layers``         : int   — number of conv layers (default 2–4).
        ``base_feat``        : int   — base channel count (default 12 for
                               frustrated, 8 otherwise) [R2].
        ``activation``       : str   — per-layer activation; default
                               ``'log_cosh'`` for holomorphicity.

    **ResNet** (residual blocks):
        ``base_features``    : int   — channel width (default 32 Kitaev,
                               16 Heis., 8 TFI) [R5].
        ``base_depth``       : int   — number of residual blocks
                               (default 4 frustrated, 2 simple).
        ``kernel_size``      : int   — spatial kernel (default
                               $\min(3, \min(L_x, L_y))$).

    **AR / MADE** (autoregressive):
        ``base_hidden``      : int   — hidden-layer width
                               (default $\max(32, 2 N_s)$) [R3, R4].
        ``n_layers``         : int   — depth (default 2–4).

    **PairProduct** (``'pp'`` / ``'rbmpp'``):
        ``alpha``            : float — RBM density for hybrid (default
                               $2 \times$ tier).

    **EqGCNN / GCNN** (group-equivariant):
        ``n_layers``         : int   — equivariant conv layers (default 2).
        ``base_channels``    : int   — channels per layer (default 8
                               Kitaev / J1-J2, 4 TFI) [R5].

    Returns
    -------
    SOTANetConfig
        Complete configuration with description string.

    Examples
    --------
    >>> # Use defaults
    >>> cfg = estimate_network_params('rbm', 18, model_type='kitaev')
    >>> # Override alpha
    >>> cfg = estimate_network_params('rbm', 18, model_type='kitaev', alpha=6.0)
    >>> # Override ResNet depth
    >>> cfg = estimate_network_params('resnet', 64, lattice_dims=(8,8),
    ...                              model_type='j1j2', base_depth=6)
    """
    net_type = net_type.lower()
    
    # Determine spatial dimensions for CNN/ResNet reshaping
    if lattice_dims is not None:
        
        reshape_dims = tuple(lattice_dims)
        if lattice_type == "honeycomb":
            # honeycomb: 2 sites per unit cell
            reshape_dims = (lattice_dims[0], lattice_dims[1] * 2) if len(lattice_dims) == 2 else lattice_dims
    else:
        
        # Infer from num_sites
        sqrt_n = int(math.sqrt(num_sites))
        
        if sqrt_n * sqrt_n == num_sites:
            reshape_dims = (sqrt_n, sqrt_n)
        else:
            # Try common factorizations
            for factor in range(int(math.sqrt(num_sites)), 0, -1):
                if num_sites % factor == 0:
                    reshape_dims = (factor, num_sites // factor)
                    break
            else:
                reshape_dims = (num_sites, 1)
    
    # Accuracy tiers
    tier_mult       = {"fast": 0.5, "medium": 1.0, "high": 2.0}.get(target_accuracy, 1.0)
    
    # Frustrated/topological model needs complex dtype
    is_frustrated   = model_type in ("kitaev", "j1j2", "heisenberg")
    
    if is_frustrated and "complex" not in dtype:
        dtype = "complex128"
    
    # ================================================================
    #  RBM  [R1] Carleo & Troyer 2017
    # ================================================================
    if net_type == "rbm":
        # Hidden-unit density alpha = M/N.  Performance improves
        # systematically with alpha (1 -> 2 -> 4) [R1 Fig. 2].
        # Frustrated / topological models need higher alpha.
        # SR regularisation: l(p) = max(l_t b^r, l_min),
        #   (l_t, b, l_min) = (100, 0.9, 1e-4) [R1].
        base_alpha  = {
            "kitaev"        : 4.0,      # frustrated -> alpha >= 4
            "heisenberg"    : 2.0,
            "j1j2"          : 4.0,
            "tfi"           : 1.0,
        }.get(model_type, 2.0)
        
        user_alpha = kwargs.get("alpha", None)
        if user_alpha is not None:
            base_alpha  = user_alpha
        
        # Scale with system size for better convergence on large lattices
        if num_sites > 50 and "alpha" not in kwargs:
            base_alpha = max(base_alpha, 2.0 + num_sites / 50.0)
        
        alpha           = base_alpha * (tier_mult if "alpha" not in kwargs else 1.0)
        n_hidden        = int(alpha * num_sites)
        # W(N,M) + b_hidden(M) + a_visible(N)
        n_params        = n_hidden * num_sites + n_hidden + num_sites
        
        if max_params and n_params > max_params:
            alpha       = max_params / (num_sites * num_sites + num_sites + 1)
            n_hidden    = int(alpha * num_sites)
            n_params    = n_hidden * num_sites + n_hidden + num_sites
        
        return SOTANetConfig(
            net_type    =   "rbm", 
            input_shape =   (num_sites,), 
            alpha       =   alpha,
            dtype       =   dtype, estimated_params=n_params,
            description =   (
                f"RBM [R1] with a={alpha:.1f} ({n_hidden} hidden units, "
                f"{n_params} params). Complex weights for {model_type}. "
                f"Override: alpha=<float>."
            ),
        )
    
    # ================================================================
    #  CNN  [R2] Choo, Neupert & Carleo 2019
    # ================================================================
    elif net_type == "cnn":
        # [R2]: 6 conv layers, channels 12->10->8->6->4->2, complex-ReLU /
        # log_cosh, ~3838 complex params for J1-J2 on 6x6.
        # Sign-structure initialisation is critical for frustrated phases.
        # We use a descending-channel pattern inspired by [R2], scaled by
        # base_feat and tier_mult.
        
        if model_type in ("kitaev", "j1j2"):
            # Frustrated: deeper network, base 12 channels [R2]
            n_layers    = max(2, min(6, int(3 * tier_mult)))
            base_feat   = 12
        else:
            n_layers    = max(1, min(4, int(2 * tier_mult)))
            base_feat   = 8
        
        n_layers    = kwargs.get('n_layers', n_layers)       # user override
        base_feat   = kwargs.get('base_feat', base_feat)     # user override
        
        # Descending channel schedule (à la R2: 12->10->8->6->4->2)
        features = tuple(
            max(2, int(base_feat * tier_mult - 2 * i))
            for i in range(n_layers)
        )
        
        # Kernel size: match interaction range (3x3 captures NN + NNN)
        ndim = len(reshape_dims)
        if ndim == 2:
            k            = min(3, min(reshape_dims))
            kernel_sizes = tuple(((k, k),) * n_layers)
        else:
            k            = min(3, reshape_dims[0])
            kernel_sizes = tuple(((k,),) * n_layers)
        
        # Activation: log_cosh for holomorphicity in frustrated systems [R2]
        activations = [kwargs.get('activation', "log_cosh")] * n_layers
        
        return SOTANetConfig(
            net_type="cnn", input_shape=(num_sites,), reshape_dims=reshape_dims,
            features=features, kernel_size=kernel_sizes, activations=activations,
            periodic=True, dtype=dtype, split_complex=False,
            description=(
                f"CNN [R2] with {n_layers} layers, features={features}, "
                f"kernel={kernel_sizes[0] if kernel_sizes else 'N/A'}, periodic BCs. "
                f"Override: n_layers, base_feat, activation."
            ),
        )
    
    # ================================================================
    #  ResNet  [R5] Roth, Szabó & MacDonald 2023
    # ================================================================
    
    elif net_type == "resnet":
        # [R5]: GCNN with L layers of 6 group-indexed feature maps;
        # residual GCNN with 12 residual blocks; avoids amp/phase
        # splitting.  LayerSR for deep nets. Sample budget:
        # training 1024 -> 4096, deep residual 4096 -> 16384,
        # observables up to 2^18.
        # We pick features=32, depth=4 as baseline for frustrated
        # phases, matching the [R5] recipe.
        base_features = {
            "kitaev"        : 32,
            "j1j2"          : 32,
            "heisenberg"    : 16,
            "tfi"           : 8,
        }.get(model_type, 16)
        base_features   = kwargs.get('base_features', base_features)    # user override
        features        = int(base_features * (tier_mult if 'base_features' not in kwargs else 1.0))
        
        # Depth: 4–12 residual blocks for frustrated, 2–4 for simpler [R5]
        base_depth      = kwargs.get("base_depth", 4 if is_frustrated else 2)
        depth           = max(2, int(base_depth * (tier_mult if 'base_depth' not in kwargs else 1.0)))

        # Kernel: 3x3 for 2D (captures NN interactions on honeycomb/square)
        kernel_size     = kwargs.get('kernel_size', min(3, min(reshape_dims)))
        
        return SOTANetConfig(
            net_type="resnet", input_shape=(num_sites,), reshape_dims=reshape_dims,
            features=features, depth=depth, kernel_size=kernel_size,
            periodic=True, dtype=dtype,
            description=(
                f"ResNet [R5] with {depth} residual blocks, {features} channels, "
                f"kernel={kernel_size}, periodic BCs. "
                f"SOTA for frustrated / topological {model_type}. "
                f"Override: base_features, base_depth, kernel_size."
            ),
        )
    
    # ================================================================
    #  Autoregressive (AR / MADE)  [R3] Sharir 2020, [R4] Hibat-Allah 2020
    # ================================================================
    elif net_type == "ar":
        # [R3] NAQS: masked-conv (PixelCNN), 3x3 filters, 32 channels,
        #   10-40 blocks; staged Adam->SGD+momentum;
        #   batch 100 -> 1000 -> larger.
        # [R4] RNN: GRU cells, d_h memory units, batch=500;
        #   schedules (eta^{-1} + 0.1t)^{-1} or eta(1 + t/5000)^{-1}.
        # Common: exact autoregressive sampling (no MCMC).
        #
        # We target MADE-style dense AR here; hidden ≈ 2-4x Ns.
        # Separate phase network is critical for frustrated phases.
        base_hidden = kwargs.get('base_hidden', max(32, num_sites * 2))   # user override
        n_layers    = kwargs.get('n_layers', max(2, min(4, int(2 * tier_mult))))  # user override
        
        if 'base_hidden' not in kwargs:
            ar_hidden   = tuple(int(base_hidden * tier_mult) for _ in range(n_layers))
        else:
            ar_hidden   = tuple(int(base_hidden) for _ in range(n_layers))
            
        # Phase network: crucial for frustrated / sign-structured phases [R3, R4]
        _tmult = tier_mult if 'base_hidden' not in kwargs else 1.0
        if is_frustrated:
            phase_hidden = tuple(int(base_hidden * _tmult * 0.75) for _ in range(n_layers))
        else:
            phase_hidden = tuple(int(base_hidden * _tmult * 0.5) for _ in range(max(1, n_layers - 1)))
        
        return SOTANetConfig(
            net_type="ar", input_shape=(num_sites,),
            ar_hidden=ar_hidden, phase_hidden=phase_hidden,
            dtype=dtype,
            description=(
                f"MADE autoregressive [R3,R4] with amplitude={ar_hidden}, "
                f"phase={phase_hidden}. Exact sampling (no MCMC). "
                f"Override: base_hidden, n_layers."
            ),
        )
    
    # ================================================================
    #  PairProduct / RBM+PP
    # ================================================================
    elif net_type in ("pp", "rbmpp"):
        alpha       = kwargs.get('alpha', 2.0 * tier_mult)  # user override
        n_params    = 4 * num_sites * num_sites # F matrix
        
        if net_type == "rbmpp":
            n_params += int(alpha * num_sites * num_sites)  # RBM part
        
        return SOTANetConfig(
            net_type=net_type, input_shape=(num_sites,), alpha=alpha,
            dtype=dtype, estimated_params=n_params,
            extra_kwargs={"use_rbm": net_type == "rbmpp", "init_scale": 0.01},
            description=(
                f"{'RBM+' if net_type == 'rbmpp' else ''}PairProduct (Pfaffian), "
                f"$O(N_s^2)$ F-matrix ({n_params} params). "
                f"Override: alpha."
            ),
        )
    
    # ================================================================
    #  Equivariant GCNN  [R5] Roth et al. 2023, arXiv:2505.23728
    # ================================================================
    elif net_type in ("eqgcnn", "gcnn"):
        # [R5]: GCNN with L layers of group-indexed features; residual
        # variant with 12 blocks; avoids amplitude/phase splitting.
        # LayerSR stores only layer Jacobians; sample budget
        # 1024 -> 4096 -> 16384 (deep).
        # arXiv:2505.23728: EquivariantGCNN on QDM/Kitaev/J1-J2.
        # Parameter scaling: c_1·Ns + sum  c_i·c_{i+1}·|G| + sum  c_i
        #   p4m on LxL: |G| = 8L^2; translations only: |G| = L^2.
        #
        # Key findings:
        #   - L=2 layers: fidelity > 0.99999 with ED at L=8
        #   - L=2 sufficient for L=12-32 (benchmarked vs QMC)
        #   - QGT rank saturates -> more layers don't help
        #   - Complex SELU activation outperforms ReLU
        
        n_layers        = kwargs.get('n_layers', max(2, min(3, int(2 * tier_mult))))  # user override
        base_channels   = {
            "kitaev"        : 8,
            "j1j2"          : 8,
            "heisenberg"    : 8,
            "tfi"           : 4,
        }.get(model_type, 8)
        base_channels   = kwargs.get('base_channels', base_channels)    # user override
        if 'base_channels' not in kwargs:
            channels    = tuple(int(base_channels * tier_mult) for _ in range(n_layers))
        else:
            channels    = tuple(int(base_channels) for _ in range(n_layers))
            
        # Estimate group size for parameter count
        if lattice_dims and len(lattice_dims) >= 2:
            Lx, Ly      = lattice_dims[0], lattice_dims[1]
        else:
            Lx = Ly     = int(math.sqrt(num_sites))

        if lattice_type == "square":
            group_size  = 8 * Lx * Ly           # p4m
        elif lattice_type == "honeycomb":
            group_size  = Lx * Ly               # translations only
        else:
            group_size  = Lx * Ly               # translations
        
        n_params = channels[0] * num_sites      # embedding
        for i in range(n_layers - 1):
            n_params += channels[i] * channels[i + 1] * group_size
        n_params += sum(channels)               # biases
        
        return SOTANetConfig(
            net_type        =   "eqgcnn", 
            input_shape     =   (num_sites,),
            features        =   channels, 
            depth           =   n_layers, 
            dtype           =   dtype,
            estimated_params=   n_params,
            extra_kwargs    =   {
                "channels"      : channels, "group_size": group_size,
                "lattice_type"  : lattice_type,
                "note"          : (
                    "Create via EquivariantGCNN.from_lattice(lattice, channels=...) "
                    "or EquivariantGCNN(input_shape, symmetry_perms=...)"
                ),
            },
            description=(
                f"EqGCNN [R5, arXiv:2505.23728] with {n_layers} layers, "
                f"channels={channels}, |G|≈{group_size}. "
                f"Exact {lattice_type} space-group equivariance. "
                f"Override: n_layers, base_channels."
            ),
        )
    
    else:
        raise ValueError(
            f"Unknown network type '{net_type}'. "
            f"Supported: rbm, cnn, resnet, ar, pp, rbmpp, eqgcnn/gcnn."
        )

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
        "rbm": NetworkInfo(
            "RBM",
            "Restricted Boltzmann Machine [R1] Carleo & Troyer 2017",
            "General purpose baseline. alpha=1->2->4 for increasing accuracy. Complex weights for frustrated/topological.",
            arguments={
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
            arguments={
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
            arguments={
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
            "Deep Residual Network [R5] Roth, Szabó & MacDonald 2023",
            "SOTA for 2D frustrated magnets. LayerSR + Lanczos post-processing. Periodic convolutions, residual connections.",
            arguments={
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
            arguments={
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
            arguments={
                "input_shape"       : "Shape of the input (e.g., `(n_sites,)`)",
                "symmetry_perms"    : "Permutation table (|G|, Ns) from generate_space_group_perms()",
                "channels"          : "Channel widths per layer (e.g., `(8, 8)` for 2 layers)",
                "split_complex"     : "Use real-valued backbone with split complex output (bool)",
                "dtype"             : "Data type for weights ('complex128', etc.)",
            },
        ),
        "activations": list_activations("jax"),
    }

    @staticmethod
    def create(
        network_type: str,
        input_shape     : Tuple[int, ...],
        dtype           : str = "complex128",
        backend         : str = "jax",
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
Restricted Boltzmann Machine (RBM) for NQS [R1].

Carleo & Troyer, Science 355 602 (2017).

Single hidden layer; hidden-unit density alpha = M/N.
Performance improves systematically with alpha (1 -> 2 -> 4).
Complex weights essential for frustrated / topological systems.
SR regularisation: l(p) = max(l_t b^r, l_min),
  (l_t, b, l_min) = (100, 0.9, 1e-4).

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    rbm_params = {
        'input_shape': (100,),
        'alpha': 2,             # alpha = 1, 2, 4 are typical
        'use_bias': True,
        'dtype': 'complex128'
    }
    net = choose_network('rbm', **rbm_params)
"""
        cnn = """
Convolutional Neural Network (CNN) for NQS [R2].

Choo, Neupert & Carleo, PRB 100 125124 (2019).

6 conv layers, channels 12->10->8->6->4->2, complex-ReLU / log_cosh,
~3838 complex params for J1-J2.  Sign-structure initialisation
critical for frustrated phases.

Features:
- Periodic Boundary Conditions (Torus geometry).
- Sum Pooling: Ensures energy is extensive.
- Complex Weights: Captures the sign structure.

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    L = 10
    n_sites = L * L
    cnn_params = {
        'input_shape':  (n_sites,),
        'reshape_dims': (L, L),
        'features':     (12, 10, 8, 6, 4, 2),  # descending à la [R2]
        'kernel_sizes': ((3,3),) * 6,
        'activations':  ['lncosh'] * 6,
        'periodic':     True,
        'sum_pooling':  True,
        'dtype':        'complex128'
    }
    net = choose_network('cnn', **cnn_params)
"""
        ar = """
Autoregressive Network (AR / NAQS / RNN) for NQS [R3, R4].

Sharir et al., PRL 124 020503 (2020) — masked-conv NAQS.
Hibat-Allah et al., PRR 2 023358 (2020) — GRU-based RNN.

Exact autoregressive sampling (no MCMC).  Staged training:
Adam lr~1e-3 -> SGD+momentum; batch 100 -> 1000 -> larger [R3].
RNN variant: batch=500, schedules (eta^{-1}+0.1t)^{-1} [R4].

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    ar_params = {
        'input_shape': (100,),
        'depth': 3,
        'num_hidden': 128,
        'rnn_type': 'lstm',
        'activations': 'relu',
        'dtype': 'complex128'
    }
    net = choose_network('ar', **ar_params)
"""
        res = """
Deep Residual Network (ResNet) for NQS [R5].

Roth, Szabó & MacDonald, PRB 108 054410 (2023).

SOTA for 2D frustrated magnets.  Residual connections + periodic
convolutions.  LayerSR avoids full Jacobian storage.  Sample budget:
1024 -> 4096 -> 16384 as depth grows; observables up to 2^18.
Avoids amplitude/phase splitting.

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    L = 8
    n_sites = L * L
    resnet_params = {
        'input_shape':  (n_sites,),
        'reshape_dims': (L, L),
        'features':     32,          # 32 channels [R5]
        'depth':        4,           # 4 residual blocks
        'kernel_size':  3,
        'dtype':        'complex128'
    }
    net = choose_network('resnet', **resnet_params)
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
        eqgcnn = """
Group-Equivariant CNN (EquivariantGCNN) for NQS [R5, arXiv:2505.23728].

Roth, Szabó & MacDonald, PRB 108 054410 (2023) — deep GCNN + LayerSR.
arXiv:2505.23728 — EquivariantGCNN on QDM/Kitaev/J1-J2.

Convolutions over the symmetry group G (not the lattice graph),
enforcing exact G-invariance.  L=2 layers achieve fidelity > 0.99999
with ED at L=8; QGT rank saturates above L=2.
Complex SELU activation outperforms ReLU.

Usage
-----
    from QES.general_python.ml.net_impl.networks.net_gcnn import EquivariantGCNN

    # From a lattice object (recommended)
    net = EquivariantGCNN.from_lattice(
        lattice,
        channels    = (8, 8),
        point_group = 'full',
        dtype       = 'complex128',
    )

    # From explicit permutations
    from QES.general_python.lattices.tools.lattice_symmetry import generate_space_group_perms
    perms = generate_space_group_perms(Lx, Ly, sites_per_cell=1, point_group='full')
    net = EquivariantGCNN(input_shape=(Ns,), symmetry_perms=perms, channels=(8, 8))

    # With SOTA parameter estimation
    from QES.NQS.src.nqs_network_integration import estimate_network_params
    config = estimate_network_params('eqgcnn', num_sites=64, lattice_type='square')
"""
        return {
            "rbm"       : rbm,
            "cnn"       : cnn,
            "ar"        : ar,
            "resnet"    : res,
            "pp"        : pp,
            "eqgcnn"    : eqgcnn,
        }

# ----------------------------------
#! End of File
# ----------------------------------
