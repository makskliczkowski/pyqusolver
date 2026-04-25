"""
Preset and heuristic network estimation for QES.NQS.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

@dataclass
class SOTANetConfig:
    r"""
    Hyperparameter configuration for NQS ansatze.
    """

    net_type            : str
    input_shape         : Tuple[int, ...]
    reshape_dims        : Optional[Tuple[int, ...]] = None
    alpha               : Optional[float] = None
    features            : Optional[Any] = None
    depth               : Optional[int] = None
    kernel_size         : Optional[Any] = None
    ar_hidden           : Optional[Tuple[int, ...]] = None
    phase_hidden        : Optional[Tuple[int, ...]] = None
    activations         : Optional[Any] = None
    periodic            : bool = True
    dtype               : str = "complex128"
    split_complex       : bool = False
    extra_kwargs        : Dict[str, Any] = field(default_factory=dict)
    description         : str = ""
    estimated_params    : int = 0

    def to_factory_kwargs(self) -> Dict[str, Any]:
        kw = {"input_shape": self.input_shape, "dtype": self.dtype}

        if self.net_type == "rbm":
            if self.alpha is not None:
                kw["alpha"] = self.alpha
        elif self.net_type == "cnn":
            if self.reshape_dims:
                kw["reshape_dims"] = self.reshape_dims
            if self.features:
                kw["features"] = self.features
            if self.kernel_size:
                kw["kernel_sizes"] = self.kernel_size
            if self.activations:
                kw["activations"] = self.activations
            kw["periodic"] = self.periodic
            if self.split_complex:
                kw["split_complex"] = True
        elif self.net_type == "resnet":
            if self.reshape_dims:
                kw["reshape_dims"] = self.reshape_dims
            if self.features:
                kw["features"] = self.features
            if self.depth:
                kw["depth"] = self.depth
            if self.kernel_size:
                kw["kernel_size"] = self.kernel_size
            kw["periodic_boundary"] = self.periodic
        elif self.net_type == "ar":
            if self.ar_hidden:
                kw["ar_hidden"] = self.ar_hidden
            if self.phase_hidden:
                kw["phase_hidden"] = self.phase_hidden
        elif self.net_type in ("pp", "rbmpp"):
            if self.alpha:
                kw["alpha"] = self.alpha
            kw["use_rbm"] = self.net_type == "rbmpp"
        elif self.net_type == "approx_symmetric" and self.features:
            kw["chi_channels"] = tuple(self.features)

        kw.update(self.extra_kwargs)
        return kw


def _resolve_reshape_dims(num_sites: int, lattice_dims: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Resolve a generic spatial layout without model-specific assumptions."""
    if lattice_dims is not None:
        raw_dims = tuple(int(d) for d in lattice_dims if d is not None and int(d) > 0)
        spatial_dims = tuple(int(d) for d in raw_dims if int(d) > 1)
        if spatial_dims:
            if math.prod(spatial_dims) == num_sites:
                if len(spatial_dims) == 1:
                    return (spatial_dims[0], 1)
                if len(spatial_dims) == 2:
                    return spatial_dims
                return (spatial_dims[0], int(math.prod(spatial_dims[1:])))

    sqrt_n = int(math.sqrt(num_sites))
    if sqrt_n * sqrt_n == num_sites:
        return (sqrt_n, sqrt_n)

    for factor in range(int(math.sqrt(num_sites)), 0, -1):
        if num_sites % factor == 0:
            return (factor, num_sites // factor)
    return (num_sites, 1)

def estimate_network_params(
    net_type: str,
    num_sites: int,
    lattice_dims: Optional[Tuple[int, ...]] = None,
    lattice_type: str = "honeycomb",
    *,
    target_accuracy: str = "medium",
    dtype: str = "complex128",
    max_params: Optional[int] = None,
    **kwargs,
) -> SOTANetConfig:
    r"""
    Estimate network hyperparameters for a given system.
    """
    net_type = net_type.lower().replace("-", "_")

    reshape_dims = _resolve_reshape_dims(num_sites, lattice_dims)
    tier_mult = {"fast": 0.75, "medium": 1.0, "high": 1.35}.get(target_accuracy, 1.0)

    if net_type == "rbm":
        base_alpha = kwargs.get("alpha", max(1.0, min(4.0, 1.0 + num_sites / 24.0)))
        alpha       = base_alpha * (tier_mult if "alpha" not in kwargs else 1.0)
        n_hidden    = int(alpha * num_sites)
        n_params    = n_hidden * num_sites + n_hidden + num_sites
        if max_params and n_params > max_params:
            alpha       = max_params / (num_sites * num_sites + num_sites + 1)
            n_hidden    = int(alpha * num_sites)
            n_params    = n_hidden * num_sites + n_hidden + num_sites
        return SOTANetConfig(
            net_type="rbm",
            input_shape=(num_sites,),
            alpha=alpha,
            dtype=dtype,
            estimated_params=n_params,
            description=(
                f"RBM [R1] with a={alpha:.1f} ({n_hidden} hidden units, "
                f"{n_params} params). "
                f"Override: alpha=<float>."
            ),
        )

    if net_type == "cnn":
        default_layers = 1 if num_sites <= 8 else 2 if num_sites <= 48 else 3
        n_layers    = int(kwargs.get("n_layers", max(1, min(4, round(default_layers * tier_mult)))))
        base_feat   = int(kwargs.get("base_feat", max(4, min(16, 2 * int(math.sqrt(num_sites)) + 2))))
        feat0       = max(4, int(round(base_feat * (tier_mult if "base_feat" not in kwargs else 1.0))))
        features    = tuple(feat0 for _ in range(n_layers))
        kernel_shape = tuple(max(1, min(3, int(dim))) for dim in reshape_dims)
        kernel_sizes = tuple(kernel_shape for _ in range(n_layers))
        activations = [kwargs.get("activation", "log_cosh")] * n_layers
        return SOTANetConfig(
            net_type="cnn",
            input_shape=(num_sites,),
            reshape_dims=reshape_dims,
            features=features,
            kernel_size=kernel_sizes,
            activations=activations,
            periodic=True,
            dtype=dtype,
            split_complex=False,
            description=(
                f"CNN [R2] with {n_layers} layers, features={features}, "
                f"kernel={kernel_sizes[0] if kernel_sizes else 'N/A'}, periodic BCs. "
                f"Override: n_layers, base_feat, activation."
            ),
        )

    if net_type == "resnet":
        base_features = int(kwargs.get("base_features", max(8, min(32, 2 * num_sites))))
        features = int(base_features * (tier_mult if "base_features" not in kwargs else 1.0))
        base_depth = kwargs.get("base_depth", 2 if num_sites <= 16 else 3)
        depth = max(2, int(base_depth * (tier_mult if "base_depth" not in kwargs else 1.0)))
        kernel_size = kwargs.get("kernel_size", tuple(max(1, min(3, int(dim))) for dim in reshape_dims))
        return SOTANetConfig(
            net_type="resnet",
            input_shape=(num_sites,),
            reshape_dims=reshape_dims,
            features=features,
            depth=depth,
            kernel_size=kernel_size,
            periodic=True,
            dtype=dtype,
            description=(
                f"ResNet [R5] with {depth} residual blocks, {features} channels, "
                f"kernel={kernel_size}, periodic BCs. "
                f"Override: base_features, base_depth, kernel_size."
            ),
        )

    if net_type == "ar":
        base_hidden = kwargs.get("base_hidden", max(32, num_sites * 2))
        n_layers = kwargs.get("n_layers", max(2, min(4, int(2 * tier_mult))))
        if "base_hidden" not in kwargs:
            ar_hidden = tuple(int(base_hidden * tier_mult) for _ in range(n_layers))
        else:
            ar_hidden = tuple(int(base_hidden) for _ in range(n_layers))
        phase_hidden = kwargs.get("phase_hidden", ar_hidden)
        return SOTANetConfig(
            net_type="ar",
            input_shape=(num_sites,),
            ar_hidden=ar_hidden,
            phase_hidden=phase_hidden,
            dtype=dtype,
            description=(
                f"MADE autoregressive [R3,R4] with amplitude={ar_hidden}, "
                f"phase={phase_hidden}. Exact sampling (no MCMC). "
                f"Override: base_hidden, n_layers."
            ),
        )

    if net_type in ("pp", "rbmpp"):
        alpha = kwargs.get("alpha", 2.0 * tier_mult)
        n_params = 4 * num_sites * num_sites
        if net_type == "rbmpp":
            n_params += int(alpha * num_sites * num_sites)
        return SOTANetConfig(
            net_type=net_type,
            input_shape=(num_sites,),
            alpha=alpha,
            dtype=dtype,
            estimated_params=n_params,
            extra_kwargs={"use_rbm": net_type == "rbmpp", "init_scale": 0.01},
            description=(
                f"{'RBM+' if net_type == 'rbmpp' else ''}PairProduct (Pfaffian), "
                f"O(Ns^2) F-matrix ({n_params} params). "
                f"Override: alpha."
            ),
        )

    if net_type == "approx_symmetric":
        if "chi_channels" in kwargs and kwargs["chi_channels"] is not None:
            chi_channels = tuple(int(v) for v in kwargs["chi_channels"])
        elif "chi_features" in kwargs and kwargs["chi_features"] is not None:
            feats = tuple(int(v) for v in kwargs["chi_features"])
            chi_channels = (1,) + feats if (not feats or feats[0] != 1) else feats
        else:
            chi_channels = (1, 2, 2) if target_accuracy == "fast" else (1, 2, 4)

        if "omega_channels" in kwargs and kwargs["omega_channels"] is not None:
            omega_channels = tuple(int(v) for v in kwargs["omega_channels"])
        else:
            omega_channels = (2, 2, 2) if target_accuracy == "fast" else (4, 4, 4)

        chi_kernel_size = int(kwargs.get("chi_kernel_size", 3))
        default_omega_kernel = 15 if target_accuracy == "high" else 11
        if target_accuracy == "fast":
            default_omega_kernel = 7
        if reshape_dims is not None and len(reshape_dims) >= 2:
            size_hint = int(max(1, min(reshape_dims[0], reshape_dims[1])))
            default_omega_kernel = min(default_omega_kernel, 2 * size_hint - 1)

        omega_kernel_size = int(kwargs.get("omega_kernel_size", max(3, default_omega_kernel)))
        if omega_kernel_size % 2 == 0:
            omega_kernel_size += 1

        nib_act = kwargs.get("nib_act", kwargs.get("chi_act", "c_sigmoid"))
        ib_act = kwargs.get("ib_act", "c_elu")
        readout_act = kwargs.get("readout_act", None)

        n_params = 0
        for cin, cout in zip(chi_channels[:-1], chi_channels[1:]):
            n_params += cout * cin * max(1, chi_kernel_size) + cout
        for cin, cout in zip(omega_channels[:-1], omega_channels[1:]):
            n_params += cout * cin * max(1, omega_kernel_size) + cout

        lattice_shape = None
        if lattice_dims is not None and len(lattice_dims) >= 2:
            lattice_shape = (int(lattice_dims[0]), int(lattice_dims[1]))

        extra_kwargs = {
            "architecture": "combo",
            "chi_channels": chi_channels,
            "omega_channels": omega_channels,
            "chi_kernel_size": chi_kernel_size,
            "omega_kernel_size": omega_kernel_size,
            "nib_act": nib_act,
            "ib_act": ib_act,
            "readout_act": readout_act,
            "lattice_type": lattice_type,
            "lattice_shape": lattice_shape,
            "bc": kwargs.get("bc", "pbc"),
            "ib_init_std": kwargs.get("ib_init_std", 0.02),
            "nib_identity_init": kwargs.get("nib_identity_init", True),
            "wilson_rescale": kwargs.get("wilson_rescale", 10 ** 1.5),
            "pool_mode": kwargs.get("pool_mode", "sum"),
            "wilson_separate_complex": kwargs.get("wilson_separate_complex", True),
        }

        return SOTANetConfig(
            net_type="approx_symmetric",
            input_shape=(num_sites,),
            features=chi_channels,
            depth=len(chi_channels) + len(omega_channels) - 2,
            dtype=dtype,
            estimated_params=n_params,
            extra_kwargs=extra_kwargs,
            description=(
                f"ApproxSymmetric(combo) with NIB channels={chi_channels}, "
                f"IB channels={omega_channels}, kernels=({chi_kernel_size},{omega_kernel_size}), "
                f"activations=({nib_act},{ib_act})."
            ),
        )

    if net_type == "eqgcnn":
        n_layers = kwargs.get("n_layers", max(2, min(3, int(2 * tier_mult))))
        base_channels = kwargs.get("base_channels", 8)
        if "base_channels" not in kwargs:
            channels = tuple(int(base_channels * tier_mult) for _ in range(n_layers))
        else:
            channels = tuple(int(base_channels) for _ in range(n_layers))

        if lattice_dims and len(lattice_dims) >= 2:
            lx, ly = lattice_dims[0], lattice_dims[1]
        else:
            lx = ly = int(math.sqrt(num_sites))

        if lattice_type == "square":
            group_size = 8 * lx * ly
        else:
            group_size = lx * ly

        n_params = channels[0] * num_sites
        for i in range(n_layers - 1):
            n_params += channels[i] * channels[i + 1] * group_size
        n_params += sum(channels)

        return SOTANetConfig(
            net_type="eqgcnn",
            input_shape=(num_sites,),
            features=channels,
            depth=n_layers,
            dtype=dtype,
            estimated_params=n_params,
            extra_kwargs={
                "channels": channels,
                "group_size": group_size,
                "lattice_type": lattice_type,
                "note": (
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

    raise ValueError(
        f"Unknown network type '{net_type}'. "
        f"Supported: rbm, cnn, resnet, ar, pp, rbmpp, eqgcnn, approx_symmetric."
    )


__all__ = ["SOTANetConfig", "estimate_network_params"]

# ------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------
