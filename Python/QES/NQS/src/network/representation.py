"""
Representation resolution and NQS-side input overrides for network creation.

This module owns the translation from model/Hilbert metadata to the input
conventions expected by the NQS-facing ansatz wrappers and generic backbones.
"""

from dataclasses import dataclass
from functools import partial
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

from QES.general_python.ml.net_impl.utils.net_state_repr_jax import map_state_to_pm1
from QES.general_python.ml.networks import Networks


@dataclass(frozen=True)
class ModelRepresentationInfo:
    """
    Normalized description of the model-side state representation.
    """

    basis_type              : str
    local_space_type        : str
    vector_encoding         : str
    integer_encoding        : str
    sampler_representation  : str
    network_representation  : str

    @property
    def is_binary(self) -> bool:
        return self.sampler_representation in {"binary_01", "occupation_binary"}

    @property
    def is_spin(self) -> bool:
        return self.local_space_type.startswith("spin-")

def _normalize_key(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")

def _resolve_sampler_representation(local_space_type: str, vector_encoding: str) -> str:
    local_key = _normalize_key(local_space_type)
    vector_key = _normalize_key(vector_encoding)

    if local_key in {"spin-1/2", "spin-half"}:
        return "spin_pm"
    if local_key in {"spinless-fermions", "abelian-anyons"}:
        return "occupation_binary"
    if local_key in {"spin-1", "bosons"}:
        return "dense_local"
    if "{0,1}" in vector_key:
        return "binary_01"
    return "dense_local"

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
    representation: Any,
    *,
    local_space_type: Optional[str] = None,
    fallback: float = 1.0,
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

def _is_jax_array(x: Any) -> bool:
    if not _JAX_AVAILABLE:
        return False
    module = type(x).__module__
    return isinstance(x, jax.Array) or module.startswith("jax")

def _array_backend(x: Any):
    return jnp if _is_jax_array(x) else np

def _same_convention(
    source_representation: Any,
    source_mode_repr: float,
    target_representation: Any,
    target_mode_repr: float,
) -> bool:
    return (
        _normalize_key(source_representation) == _normalize_key(target_representation)
        and np.isclose(float(source_mode_repr), float(target_mode_repr))
    )

def convert_state_array_representation(
    states: Any,
    *,
    source_representation: Any,
    source_mode_repr: float,
    target_representation: Any,
    target_mode_repr: float,
):
    """
    Convert array-valued states between supported external NQS conventions.

    The supported conversions are intentionally narrow:
    - signed spin ``spin_pm`` <-> binary occupation ``binary_01``
    - binary rescaling between different occupied-state values
    - signed spin rescaling between different physical magnitudes
    Dense local encodings are returned unchanged.
    """
    if _same_convention(
        source_representation,
        source_mode_repr,
        target_representation,
        target_mode_repr,
    ):
        return states

    source_key = _normalize_key(source_representation)
    target_key = _normalize_key(target_representation)
    xp = _array_backend(states)
    arr = xp.asarray(states)
    dtype = getattr(arr, "dtype", None)

    source_val = float(source_mode_repr)
    target_val = float(target_mode_repr)

    if source_key == target_key:
        if source_key in {"binary-01", "occupation-binary"} and source_val != 0.0:
            return arr * (target_val / source_val)
        if source_key == "spin-pm":
            target = xp.asarray(target_val, dtype=dtype)
            return xp.where(arr > 0, target, -target)
        return arr

    if source_key in {"binary-01", "occupation-binary"} and target_key == "spin-pm":
        source_scale = source_val if source_val != 0.0 else 1.0
        target = xp.asarray(target_val, dtype=dtype)
        return (2.0 * (arr / source_scale) - 1.0) * target

    if source_key == "spin-pm" and target_key in {"binary-01", "occupation-binary"}:
        target = xp.asarray(target_val, dtype=dtype)
        return (arr > 0).astype(dtype) * target

    if (
        source_key in {"binary-01", "occupation-binary"}
        and target_key in {"binary-01", "occupation-binary"}
        and source_val != 0.0
    ):
        return arr * (target_val / source_val)

    return arr

def bind_local_energy_state_convention(
    local_energy_fn: Any,
    *,
    state_convention: Dict[str, Any],
    local_convention: Dict[str, Any],
):
    """
    Wrap a Hamiltonian local-energy callback so inputs and generated states use
    the active NQS convention while the operator kernels keep their local one.
    """
    if local_energy_fn is None or not callable(local_energy_fn):
        return local_energy_fn

    state_repr = state_convention.get("representation")
    local_repr = local_convention.get("representation")
    state_mode = float(state_convention.get("mode_repr", 1.0))
    local_mode = float(local_convention.get("mode_repr", 1.0))

    if _same_convention(state_repr, state_mode, local_repr, local_mode):
        return local_energy_fn

    def _wrapped(state):
        local_state = convert_state_array_representation(
            state,
            source_representation=state_repr,
            source_mode_repr=state_mode,
            target_representation=local_repr,
            target_mode_repr=local_mode,
        )
        new_states, coeffs = local_energy_fn(local_state)
        public_states = convert_state_array_representation(
            new_states,
            source_representation=local_repr,
            source_mode_repr=local_mode,
            target_representation=state_repr,
            target_mode_repr=state_mode,
        )
        return public_states, coeffs

    try:
        _wrapped.__name__ = getattr(local_energy_fn, "__name__", type(local_energy_fn).__name__)
    except Exception:
        pass
    return _wrapped

# ------------------------------------------------------------------
# Public API for model-driven representation resolution and NQS-side overrides based on model/Hilbert metadata
# ------------------------------------------------------------------

def canonical_network_request_family(network_type: Any) -> str:
    """
    Normalize user-facing network/ansatz identifiers to one family name.
    """
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
    """
    Resolve model-driven state representation from Hamiltonian/Hilbert metadata.
    """
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

    local_space_type    = "spin-1/2"
    vector_encoding     = "Length-Ns array with entries in {0,1}."
    integer_encoding    = "Binary computational basis."

    if hilbert is not None:
        local_space = getattr(hilbert, "local_space", None)
        if local_space is not None and hasattr(local_space, "typ"):
            local_space_type = _normalize_key(getattr(local_space.typ, "value", local_space_type), local_space_type)

        convention = None
        if hasattr(hilbert, "state_convention"):
            try:
                convention = hilbert.state_convention
            except Exception:
                convention = None

        if convention:
            local_space_type = _normalize_key(convention.get("name"), local_space_type)
            vector_encoding = convention.get("vector_encoding", vector_encoding)
            integer_encoding = convention.get("integer_encoding", integer_encoding)

    sampler_representation = _resolve_sampler_representation(local_space_type, vector_encoding)
    if sampler_representation == "binary_01" and basis_type == "real" and local_space_type.startswith("spin-"):
        network_representation = "spin_binary_native"
    elif sampler_representation == "occupation_binary":
        network_representation = "occupation_binary_native"
    else:
        network_representation = sampler_representation

    return ModelRepresentationInfo(
        basis_type=basis_type,
        local_space_type=local_space_type,
        vector_encoding=vector_encoding,
        integer_encoding=integer_encoding,
        sampler_representation=sampler_representation,
        network_representation=network_representation,
    )


def apply_nqs_representation_overrides(
    network_type: Any,
    representation: Optional[ModelRepresentationInfo],
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply NQS-side basis/state-convention kwargs to otherwise generic networks.
    """
    resolved_kwargs = dict(kwargs or {})
    if representation is None:
        return resolved_kwargs

    family = canonical_network_request_family(network_type)
    if not family:
        return resolved_kwargs

    sampler_representation = resolved_kwargs.get("state_representation", representation.sampler_representation)
    sampler_key = _normalize_key(sampler_representation)
    local_key = _normalize_key(representation.local_space_type)
    is_spin_model = local_key.startswith("spin-")
    is_binary_state = sampler_key in {"binary-01", "occupation-binary"}
    is_spin_state = sampler_key == "spin-pm"
    input_value = resolve_representation_value(
        sampler_representation,
        local_space_type=representation.local_space_type,
        fallback=1.0,
    )

    if family == "rbm" and is_spin_model and is_binary_state:
        resolved_kwargs.setdefault(
            "input_activation",
            partial(map_state_to_pm1, input_is_spin=is_spin_state, input_value=input_value),
        )
    elif family in {"cnn", "mlp", "gcnn", "resnet"} and is_spin_model and sampler_key == "binary-01":
        resolved_kwargs.setdefault(
            "input_adapter",
            partial(map_state_to_pm1, input_is_spin=is_spin_state, input_value=input_value),
        )
    elif family in {"pp", "mps"}:
        resolved_kwargs.setdefault("input_is_spin", is_spin_state)
        resolved_kwargs.setdefault("input_value", input_value)

    return resolved_kwargs

def resolve_nqs_state_defaults(nqs: Any, fallback_mode_repr: float = 0.5) -> Tuple[bool, float]:
    """
    Resolve the effective state convention attached to an NQS instance.
    """
    convention = getattr(nqs, "state_convention", None)
    if isinstance(convention, dict):
        return bool(convention.get("spin", False)), float(convention.get("mode_repr", fallback_mode_repr))

    sampler = getattr(nqs, "sampler", None)
    representation_info = getattr(nqs, "_representation_info", None)

    sampler_representation = getattr(nqs, "_state_representation", None)
    if sampler_representation is None:
        sampler_representation = getattr(sampler, "state_representation", None)
    if sampler_representation is None and representation_info is not None:
        sampler_representation = getattr(representation_info, "sampler_representation", None)

    sampler_key = _normalize_key(sampler_representation)
    sampler_spin = getattr(nqs, "_spin_repr", None)
    if sampler_spin is None:
        sampler_spin = getattr(sampler, "_spin_repr", None)
    if sampler_spin is None:
        sampler_spin = sampler_key == "spin-pm"

    sampler_mode_repr = getattr(nqs, "_mode_repr", None)
    if sampler_mode_repr is None:
        sampler_mode_repr = getattr(sampler, "_mode_repr", None)
    if sampler_mode_repr is None:
        sampler_mode_repr = resolve_representation_value(
            sampler_representation,
            local_space_type=(
                getattr(representation_info, "local_space_type", None)
                if representation_info is not None
                else None
            ),
            fallback=float(fallback_mode_repr),
        )

    return bool(sampler_spin), float(sampler_mode_repr)


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
