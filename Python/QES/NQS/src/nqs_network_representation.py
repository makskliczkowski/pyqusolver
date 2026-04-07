"""
Representation resolution and NQS-side input overrides for network creation.

This module owns the translation from model/Hilbert metadata to the input
conventions expected by the NQS-facing ansatz wrappers and generic backbones.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

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
        return "binary_01"
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
                basis_type = _normalize_key(basis_state.get("current_basis"), default=basis_type)
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

    sampler_key = _normalize_key(representation.sampler_representation)
    local_key = _normalize_key(representation.local_space_type)
    spin_value = float(resolve_spin_mode_repr(representation.local_space_type) or 1.0)
    is_spin_model = local_key.startswith("spin-")
    is_binary_state = sampler_key in {"binary-01", "occupation-binary"}
    is_spin_state = sampler_key == "spin-pm"
    input_value = spin_value if is_spin_state else 1.0

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
        if sampler_key in {"binary-01", "occupation-binary"}:
            sampler_mode_repr = 1.0
        else:
            sampler_mode_repr = float(fallback_mode_repr)

    return bool(sampler_spin), float(sampler_mode_repr)


__all__ = [
    "ModelRepresentationInfo",
    "apply_nqs_representation_overrides",
    "canonical_network_request_family",
    "resolve_model_representation",
    "resolve_nqs_state_defaults",
    "resolve_spin_mode_repr",
]

# ------------------------------------------------------------------