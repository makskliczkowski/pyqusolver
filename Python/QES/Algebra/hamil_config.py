"""
Hamiltonian configuration and registry utilities.

This module mirrors the modular structure provided for operators and Hilbert
spaces.  Users can register Hamiltonian builders together with descriptive
metadata and instantiate them from lightweight dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Tuple, Union

try:
    from QES.Algebra.hilbert import HilbertConfig, HilbertSpace
except ImportError as e:
    raise ImportError(
        "Could not import HilbertSpace or HilbertConfig from QES.Algebra.hilbert"
    ) from e

if TYPE_CHECKING:
    from .hamil import Hamiltonian

# ---------------------------------------------------------------------------
#! Registry data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HamiltonianSpec:
    """
    Declarative description of a Hamiltonian builder.
    Parameters
    ----------
    key:
        Unique string key identifying the Hamiltonian type.
    builder:
        Callable that constructs the Hamiltonian instance.
    description:
        One-liner description of the Hamiltonian.
    tags:
        Optional tags for categorization.
    default_kwargs:
        Default keyword arguments for the builder.
    """

    key: str
    builder: Callable[["HamiltonianConfig", Dict[str, Any]], "Hamiltonian"]
    description: str
    tags: Tuple[str, ...] = field(default_factory=tuple)
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "tags": self.tags,
            "default_kwargs": dict(self.default_kwargs),
        }


# ---------------------------------------------------------------------------


class HamiltonianRegistry:
    """
    Registry of available Hamiltonian constructions.
    This class maintains a mapping from string keys to Hamiltonian builders
    along with descriptive metadata.  Users can register new Hamiltonian types
    and instantiate them from configuration dataclasses.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._registry: Dict[str, HamiltonianSpec] = {}

    # ------------------
    #! registration
    # ------------------

    def register(
        self,
        key: str,
        builder: Callable[["HamiltonianConfig", Dict[str, Any]], "Hamiltonian"],
        *,
        description: str,
        tags: Tuple[str, ...] = (),
        default_kwargs: Optional[Mapping[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a new Hamiltonian builder.

        Parameters
        ----------
        key:
            Unique string key identifying the Hamiltonian type.
        builder:
            Callable that constructs the Hamiltonian instance.
        description:
            One-liner description of the Hamiltonian.
        tags:
            Optional tags for categorization.
        default_kwargs:
            Default keyword arguments for the builder.
        overwrite:
            Whether to overwrite an existing registration.
        """

        if not overwrite and key in self._registry:
            raise KeyError(f"Hamiltonian '{key}' already registered.")

        spec = HamiltonianSpec(
            key=key,
            builder=builder,
            description=description,
            tags=tuple(tags),
            default_kwargs=default_kwargs or {},
        )
        self._registry[key] = spec

    # ------------------
    #! accessors
    # ------------------

    def get(self, key: str) -> HamiltonianSpec:
        try:
            return self._registry[key]
        except KeyError as exc:
            raise KeyError(f"Hamiltonian '{key}' is not registered.") from exc

    def available(self) -> Tuple[str, ...]:
        return tuple(self._registry.keys())

    def describe(self, key: str) -> Dict[str, Any]:
        return self.get(key).to_dict()

    # ------------------
    #! instantiation
    # ------------------

    def instantiate(self, config: "HamiltonianConfig", **overrides: Any) -> "Hamiltonian":
        spec = self.get(config.kind)
        params: Dict[str, Any] = dict(spec.default_kwargs)
        params.update(config.parameters)
        params.update(overrides)
        return spec.builder(config, params)


# Singleton registry instance
HAMILTONIAN_REGISTRY = HamiltonianRegistry()

# ---------------------------------------------------------------------------
#! Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HamiltonianConfig:
    """
    Declarative configuration for a Hamiltonian instance.

    Parameters
    ----------
    kind:
        String key registered in :data:`HAMILTONIAN_REGISTRY`.
    hilbert:
        Either a ready-made :class:`HilbertSpace` or a :class:`HilbertConfig`
        blueprint that will be materialised on demand.
    parameters:
        Free-form keyword arguments forwarded to the Hamiltonian builder.
    metadata:
        Optional metadata (ignored by the builder, useful for user tooling).
    """

    kind: str
    hilbert: Optional[Union[HilbertSpace, HilbertConfig]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_override(self, **updates: Any) -> "HamiltonianConfig":
        """
        Return a new config with selected fields replaced.
        """
        return replace(self, **updates)

    def resolve_hilbert(self) -> Optional[HilbertSpace]:
        """
        Materialise the Hilbert space if a blueprint was provided.
        """
        if self.hilbert is None:
            return None
        if isinstance(self.hilbert, HilbertSpace):
            return self.hilbert
        if isinstance(self.hilbert, HilbertConfig):
            return HilbertSpace.from_config(self.hilbert)
        raise TypeError(f"Unsupported hilbert specification: {type(self.hilbert)!r}")

    def to_builder_kwargs(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combine configuration parameters with optional overrides.
        """
        params = dict(self.parameters)
        if extra:
            params.update(extra)
        return params


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def register_hamiltonian(
    key: str,
    *,
    builder: Callable[[HamiltonianConfig, Dict[str, Any]], "Hamiltonian"],
    description: str,
    tags: Tuple[str, ...] = (),
    default_kwargs: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """
    Register a Hamiltonian builder with the global registry.
    Parameters
    ----------
    key:
        Unique string key identifying the Hamiltonian type.
    builder:
        Callable that constructs the Hamiltonian instance.
    description:
        One-liner description of the Hamiltonian.
    tags:
        Optional tags for categorization.
    default_kwargs:
        Default keyword arguments for the builder.
    overwrite:
        Whether to overwrite an existing registration.
    """
    HAMILTONIAN_REGISTRY.register(
        key,
        builder=builder,
        description=description,
        tags=tags,
        default_kwargs=default_kwargs,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------
