"""
QES.NQS.src.network.representation
==================================

Representation resolution and NQS-side input overrides for network creation.
This module translates model/Hilbert metadata to the input conventions expected 
by the NQS-facing ansatz wrappers and generic backbones.

It provides:
    1. Metadata normalization (ModelRepresentationInfo).
    2. State value resolution (spin magnitudes vs unit binary).
    3. Array conversion between conventions (spin <-> binary).
    4. Hamiltonian callback binding (local energy wrapping).
    5. Network-specific input adapters (automatic PM1 mapping).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    _JAX_AVAILABLE = False

try:
    from QES.general_python.ml.networks import Networks
except ImportError:
    raise ImportError("QES.general_python.ml.networks module is required for network family resolution.")

# ------------------------------------------------------------------------------
#! REPRESENTATION RESOLUTION
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelRepresentationInfo:
    """
    Normalized description of the model-side state representation.
    """
    basis_type              : str   # e.g. "real", "complex", "qubit", etc.
    local_space_type        : str   # e.g. "spin-1/2", "spin-1", "fermion", etc.
    vector_encoding         : str   # e.g. "Length-Ns array with entries in {0,1}." or "Length-Ns array with entries in {-S,...,S}."
    integer_encoding        : str   # e.g. "Binary computational basis." or "Integer encoding of spin states."
    sampler_representation  : str   # e.g. "binary_01", "spin_pm", "occupation_binary", etc.
    network_representation  : str   # e.g. "spin_binary_native", "occupation_binary_native", etc., for internal network input conventions.

    @property
    def is_binary(self) -> bool:
        return self.sampler_representation in {"binary_01", "occupation_binary"}

    @property
    def is_spin(self) -> bool:
        return self.local_space_type.startswith("spin-")

# ------------------------------------------------------------------------------
#! UTILITIES
# ------------------------------------------------------------------------------

def _normalize_key(value: Any, default: str = "") -> str:
    """ Normalize string keys for consistent comparison. """
    if value is None:
        return default
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")

def _is_jax_array(x: Any) -> bool:
    """ Check if an object is a JAX array. """
    if not _JAX_AVAILABLE:
        return False
    module = type(x).__module__
    return isinstance(x, jax.Array) or module.startswith("jax")

def _array_backend(x: Any):
    """ Return the appropriate array backend (numpy or jnp). """
    return jnp if _is_jax_array(x) else np

def _same_convention(
    source_representation   : Any,
    source_mode_repr        : float,
    target_representation   : Any,
    target_mode_repr        : float,
) -> bool:
    """ Check if two conventions are effectively identical. """
    return _normalize_key(source_representation) == _normalize_key(target_representation) and np.isclose(float(source_mode_repr), float(target_mode_repr))

# ------------------------------------------------------------------------------
#! RESOLUTION LOGIC
# ------------------------------------------------------------------------------

def resolve_spin_mode_repr(local_space_type: str) -> Optional[float]:
    """
    Return the physical spin magnitude for signed spin representations.
    """
    local_key = _normalize_key(local_space_type)
    if local_key in {"spin-1/2", "spin-half"}:
        return 0.5
    if local_key in {"spin-1", "spin-one"}:
        return 1.0
    return None

def resolve_representation_value(
    representation      : Any,
    *,
    local_space_type    : Optional[str] = None,
    fallback            : float         = 1.0,
) -> float:
    """
    Return the explicit value used for the "up"/occupied state in a convention.

    Binary computational bases always use unit-valued occupations externally,
    even for spin-1/2 models. Physical spin magnitudes such as +/-0.5 only
    apply to explicit signed-spin conventions.
    """
    key = _normalize_key(representation)
    if key in {"binary-01", "occupation-binary"}:
        return 1.0
    if key == "spin-pm":
        return float(resolve_spin_mode_repr(local_space_type or "") or fallback)
    return float(fallback)

def _resolve_sampler_representation(local_space_type: str, local_representation: str, vector_encoding: str) -> str:
    """ Pick the default sampler representation based on model physics. """
    local_key   = _normalize_key(local_space_type)
    local_repr  = _normalize_key(local_representation)
    vector_key  = _normalize_key(vector_encoding)

    if local_key in {"spin-1/2", "spin-half"}:
        return "spin_pm"
    if local_repr:
        return local_repr
    if "{0,1}" in vector_key:
        return "binary_01"
    return "dense_local"

# ------------------------------------------------------------------------------
#! CONVERSION & BINDING
# ------------------------------------------------------------------------------

def convert_state_array_representation(
    states                  : Any,
    *,
    source_representation   : Any,
    source_mode_repr        : float,
    target_representation   : Any,
    target_mode_repr        : float,
):
    """
    Convert array-valued states between supported external NQS conventions.

    Supported conversions:
        - signed spin ``spin_pm`` <-> binary occupation ``binary_01``
        - rescaling between different binary/occupation values
        - rescaling between different signed spin magnitudes
    """
    if _same_convention(source_representation, source_mode_repr, target_representation, target_mode_repr):
        return states

    src_key     = _normalize_key(source_representation)
    dst_key     = _normalize_key(target_representation)
    xp          = _array_backend(states)
    arr         = xp.asarray(states)
    dtype       = getattr(arr, "dtype", None)

    src_val     = float(source_mode_repr)
    dst_val     = float(target_mode_repr)

    # Identical family (binary->binary or spin->spin) but different scale
    if src_key == dst_key:
        if src_key in {"binary-01", "occupation-binary"} and src_val != 0.0:
            return arr * (dst_val / src_val)
        if src_key == "spin-pm":
            target = xp.asarray(dst_val, dtype=dtype)
            return xp.where(arr > 0, target, -target)
        return arr

    # Binary -> Spin
    if src_key in {"binary-01", "occupation-binary"} and dst_key == "spin-pm":
        src_scale   = src_val if src_val != 0.0 else 1.0
        target      = xp.asarray(dst_val, dtype=dtype)
        return (2.0 * (arr / src_scale) - 1.0) * target

    # Spin -> Binary
    if src_key == "spin-pm" and dst_key in {"binary-01", "occupation-binary"}:
        target      = xp.asarray(dst_val, dtype=dtype)
        return (arr > 0).astype(dtype) * target

    # Binary rescale (fallback)
    if src_key in {"binary-01", "occupation-binary"} and dst_key in {"binary-01", "occupation-binary"} and src_val != 0.0:
        return arr * (dst_val / src_val)

    return arr

def bind_local_energy_state_convention(
    local_energy_fn     : Any,
    *,
    state_convention    : Dict[str, Any],
    local_convention    : Dict[str, Any],
):
    """
    Wrap a Hamiltonian local-energy callback so inputs and generated states use
    the active NQS convention while the operator kernels keep their local one.
    """
    if local_energy_fn is None or not callable(local_energy_fn):
        return local_energy_fn

    s_repr  = state_convention.get("representation")
    l_repr  = local_convention.get("representation")
    s_mode  = float(state_convention.get("mode_repr", 1.0))
    l_mode  = float(local_convention.get("mode_repr", 1.0))

    if _same_convention(s_repr, s_mode, l_repr, l_mode):
        return local_energy_fn

    def _wrapped(state):
        local_state = convert_state_array_representation(
            state,
            source_representation   =   s_repr,
            source_mode_repr        =   s_mode,
            target_representation   =   l_repr,
            target_mode_repr        =   l_mode,
        )
        new_states, coeffs = local_energy_fn(local_state)
        public_states = convert_state_array_representation(
            new_states,
            source_representation   =   l_repr,
            source_mode_repr        =   l_mode,
            target_representation   =   s_repr,
            target_mode_repr        =   s_mode,
        )
        return public_states, coeffs

    try:
        _wrapped.__name__ = getattr(local_energy_fn, "__name__", type(local_energy_fn).__name__)
    except Exception:
        pass
    return _wrapped

# ------------------------------------------------------------------
#! NETWORK ADAPTERS
# ------------------------------------------------------------------

def canonical_network_request_family(network_type: Any) -> str:
    """ Normalize user-facing network identifiers to a canonical family name. """
    if isinstance(network_type, (str, Networks)):
        key = str(network_type).strip().lower().replace("-", "_").replace(" ", "_")
    elif isinstance(network_type, type):
        key = getattr(network_type, "__name__", "").strip().lower().replace("-", "_").replace(" ", "_")
    else:
        return ""

    aliases = {
        "res"                       : "resnet",
        "complexar"                 : "ar",
        "autoregressive"            : "ar",
        "pairproduct"               : "pp",
        "ansatzapproxsymmetric"     : "approx_symmetric",
        "equivariantgcnn"           : "eqgcnn",
    }
    return aliases.get(key, key)

def resolve_model_representation(model: Any, hilbert: Optional[Any] = None) -> ModelRepresentationInfo:
    """ Resolve model-driven state representation from Hamiltonian/Hilbert metadata. """
    basis_type = "real"
    if model is not None:
        if hasattr(model, "get_transformation_state") and callable(model.get_transformation_state):
            try:
                basis_state = model.get_transformation_state()
                basis_type  = _normalize_key(basis_state.get("current_basis"), default=basis_type)
            except Exception:
                pass
        elif hasattr(model, "get_basis_type") and callable(model.get_basis_type):
            try:
                basis_type = _normalize_key(model.get_basis_type(), default=basis_type)
            except Exception:
                pass

    if hilbert is None and model is not None:
        hilbert = getattr(model, "hilbert", None)
        if hilbert is None:
            hilbert = getattr(model, "_hilbert_space", None)

    l_type                  = "spin-1/2"
    v_enc                   = "Length-Ns array with entries in {0,1}."
    i_enc                   = "Binary computational basis."
    local_representation    = ""
    if hilbert is not None:
        l_space = getattr(hilbert, "local_space", None)
        if l_space is not None and hasattr(l_space, "typ"):
            l_type = _normalize_key(getattr(l_space.typ, "value", l_type), l_type)

        conv = None
        if hasattr(hilbert, "state_convention"):
            try:
                conv = hilbert.state_convention
            except Exception:
                conv = None

        if conv:
            l_type  = _normalize_key(conv.get("local_space_type", conv.get("name")), l_type)
            local_representation = _normalize_key(
                conv.get("vector_representation", conv.get("representation")),
                "",
            )
            v_enc   = conv.get("vector_encoding", v_enc)
            i_enc   = conv.get("integer_encoding", i_enc)

    s_repr = _resolve_sampler_representation(l_type, local_representation, v_enc)
    if s_repr == "binary_01" and basis_type == "real" and l_type.startswith("spin-"):
        n_repr = "spin_binary_native"
    elif s_repr == "occupation_binary":
        n_repr = "occupation_binary_native"
    else:
        n_repr = s_repr

    return ModelRepresentationInfo(
        basis_type              =   basis_type,
        local_space_type        =   l_type,
        vector_encoding         =   v_enc,
        integer_encoding        =   i_enc,
        sampler_representation  =   s_repr,
        network_representation  =   n_repr,
    )

def apply_nqs_representation_overrides(
    network_type    : Any,
    representation  : Optional[ModelRepresentationInfo],
    kwargs          : Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply NQS-side basis/state-convention overrides to otherwise generic networks.
    Useful for automatically mapping binary inputs to PM1 spins inside the network.
    """
    res_kwargs = dict(kwargs or {})
    if representation is None:
        return res_kwargs

    family = canonical_network_request_family(network_type)
    if not family:
        return res_kwargs

    s_repr      = res_kwargs.get("state_representation", representation.sampler_representation)
    s_key       = _normalize_key(s_repr)
    l_key       = _normalize_key(representation.local_space_type)
    is_spin_m   = l_key.startswith("spin-")
    is_bin_s    = s_key in {"binary-01", "occupation-binary"}
    is_spin_s   = s_key == "spin-pm"
    in_val      = resolve_representation_value(s_repr, local_space_type=representation.local_space_type, fallback=1.0)
    if family == "rbm" and is_spin_m and is_bin_s:
        res_kwargs.setdefault("input_is_spin", is_spin_s)
        res_kwargs.setdefault("input_value", in_val)
        res_kwargs.setdefault("input_activation", True)
    elif family in {"cnn", "mlp", "gcnn", "resnet"} and is_spin_m and s_key == "binary-01":
        res_kwargs.setdefault("input_is_spin", is_spin_s)
        res_kwargs.setdefault("input_value", in_val)
        res_kwargs.setdefault("transform_input", True)
    elif family in {"pp", "mps"}:
        res_kwargs.setdefault("input_is_spin", is_spin_s)
        res_kwargs.setdefault("input_value", in_val)

    return res_kwargs

# ------------------------------------------------------------------
#! NQS INSTANCE HELPERS
# ------------------------------------------------------------------

def resolve_nqs_state_defaults(nqs: Any, fallback_mode_repr: float = 0.5) -> Tuple[bool, float]:
    """
    Resolve the effective state convention (spin flag and mode magnitude) 
    attached to an NQS instance.
    """
    # Try explicit convention attribute
    conv = getattr(nqs, "state_convention", None)
    if isinstance(conv, dict):
        return bool(conv.get("spin", False)), float(conv.get("mode_repr", fallback_mode_repr))

    sampler = getattr(nqs, "sampler", None)
    rep_inf = getattr(nqs, "_representation_info", None)

    # Resolve representation string
    s_repr = getattr(nqs, "_state_representation", None)
    if s_repr is None:
        s_repr = getattr(sampler, "state_representation", None)
    if s_repr is None and rep_inf is not None:
        s_repr = getattr(rep_inf, "sampler_representation", None)

    s_key = _normalize_key(s_repr)

    # Resolve spin flag
    s_spin = getattr(nqs, "_spin_repr", None)
    if s_spin is None:
        s_spin = getattr(sampler, "_spin_repr", None)
    if s_spin is None:
        s_spin = s_key == "spin-pm"

    # Resolve mode magnitude
    s_mode = getattr(nqs, "_mode_repr", None)
    if s_mode is None:
        s_mode = getattr(sampler, "_mode_repr", None)
    if s_mode is None:
        s_mode = resolve_representation_value(
            s_repr,
            local_space_type    =   getattr(rep_inf, "local_space_type", None) if rep_inf else None,
            fallback            =   float(fallback_mode_repr),
        )

    return bool(s_spin), float(s_mode)


__all__ = [
    "ModelRepresentationInfo",
    "apply_nqs_representation_overrides",
    "bind_local_energy_state_convention",
    "canonical_network_request_family",
    "convert_state_array_representation",
    "resolve_model_representation",
    "resolve_nqs_state_defaults",
    "resolve_representation_value",
    "resolve_spin_mode_repr",
]

# ------------------------------------------------------------------
