"""
Operator catalog
================

This module centralises the description and registration of onsite operators.
Operator implementations register themselves with the global ``OPERATOR_CATALOG``,
which can then be queried by Hilbert space builders or user code that wants
structured metadata (description, algebra, sign convention) together with the
callable kernels.

The catalog keeps operators grouped by the underlying local Hilbert space type.
Keys are normalised so both ``LocalSpaceTypes`` enums and plain strings can be
used interchangeably.

Typical usage
-------------

>>> from QES.Algebra.Operator.catalog import OPERATOR_CATALOG
>>> from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
>>> OPERATOR_CATALOG.names_for(LocalSpaceTypes.SPIN_1_2)
['sigma_x', 'sigma_y', 'sigma_z', ...]
>>> sx = OPERATOR_CATALOG.instantiate(LocalSpaceTypes.SPIN_1_2, "sigma_x")
>>> sx.description
'Pauli-X flip'

----------------------------------------------------
File        : QES/Algebra/Operator/catalog.py
Description : Operator catalog implementation
Author      : Maksymilian Kliczkowski
Date        : 2025-10-30
----------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorSpec:
    """
    Declarative description of an onsite operator.

    Attributes
    ----------
    key:
        Unique identifier within the local space family (e.g. ``"sigma_x"``).
    factory:
        Callable that returns a ``LocalOpKernels`` instance. It receives ``**kwargs``
        provided to :meth:`OperatorCatalog.instantiate`/``build_local_operator_map``.
    description:
        Short human readable explanation of what the operator does.
    algebra:
        Text snippet describing the algebraic relations (commutators, etc.).
    sign_convention:
        Explanation of the sign/statistics convention used by the kernels.
    tags:
        Optional collection of helper tags (e.g. ``("spin", "pauli")``) to make
        filtering simpler.
    default_kwargs:
        Keyword arguments forwarded to ``factory`` unless explicitly overridden.
    """

    key: str  # Unique operator identifier
    factory: Callable[..., Any]  # Factory producing operator kernels
    description: str  # Short human readable explanation
    algebra: str  # Text snippet describing the algebraic relations
    sign_convention: str  # Explanation of the sign/statistics convention
    tags: Tuple[str, ...] = field(default_factory=tuple)  # Optional tags
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)  # Optional default kwargs


# ---------------------------------------------------------------------------
#! Catalog implementation
# ---------------------------------------------------------------------------


class OperatorCatalog:
    """
    Registry that keeps track of onsite operator specifications.
    The catalog allows registering and querying operator metadata and
    kernel factories in a structured manner.

    Operators are grouped by the identifier of the local space they belong to.
    The identifier can either be a ``LocalSpaceTypes`` enum member, or an
    arbitrary string (case insensitive).
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, OperatorSpec]] = {}

    # -----------------------------
    #! Utilities
    # -----------------------------

    @staticmethod
    def _normalise_space_key(local_space: Any) -> str:
        """
        Convert a LocalSpaceTypes enum or string into a consistent registry key.
        """
        if hasattr(local_space, "name") and hasattr(local_space, "value"):
            enum_name = f"{local_space.__class__.__name__}.{local_space.name}"
            return enum_name.lower()

        if isinstance(local_space, str):
            return local_space.strip().lower()

        return str(local_space).strip().lower()

    def _space_bucket(self, local_space: Any) -> Dict[str, OperatorSpec]:
        """
        Retrieve or create the registry bucket for a specific local space.

        Parameters
        ----------
        local_space:
            ``LocalSpaceTypes`` enum member or string identifying the local space.
        """
        key = self._normalise_space_key(local_space)
        if key not in self._registry:
            self._registry[key] = {}
        return self._registry[key]

    # -----------------------------
    #! Registration API
    # -----------------------------

    def register(self, local_space: Any, spec: OperatorSpec, *, overwrite: bool = False) -> None:
        """
        Register a new operator specification.

        Parameters
        ----------
        local_space:
            ``LocalSpaceTypes`` enum member or string identifying the local space.
        spec:
            Operator specification with metadata and kernel factory.
        overwrite:
            If ``True`` an existing registration with the same key is replaced.
            Otherwise a ``ValueError`` is raised on duplicates.
        """
        bucket = self._space_bucket(local_space)
        if not overwrite and spec.key in bucket:
            raise ValueError(f"Operator '{spec.key}' already registered for {local_space!r}.")
        bucket[spec.key] = spec

    # -----------------------------
    #! Query helpers
    # -----------------------------

    def specs_for(self, local_space: Any) -> Iterable[OperatorSpec]:
        """
        Iterate over all operator specs registered for ``local_space``.
        """
        return self._space_bucket(local_space).values()

    def names_for(self, local_space: Any) -> Iterable[str]:
        """
        Iterate over registered operator identifiers for ``local_space``.
        """
        yield from self._space_bucket(local_space).keys()

    # -----------------------------
    #! Instantiation helpers
    # -----------------------------

    def instantiate(self, local_space: Any, key: str, **kwargs: Any):
        """
        Materialise the kernels and metadata for a specific operator.

        Returns
        -------
        LocalOperator
            Object describing the operator together with its kernels.
        """
        bucket = self._space_bucket(local_space)
        try:
            spec = bucket[key]
        except KeyError as exc:
            raise KeyError(
                f"Operator '{key}' not registered for local space {local_space!r}."
            ) from exc

        merged_kwargs = dict(spec.default_kwargs)
        merged_kwargs.update(kwargs)

        # Local import to avoid circular dependency during module initialisation.
        from QES.Algebra.Hilbert.hilbert_local import LocalOperator, LocalOpKernels

        kernels = spec.factory(**merged_kwargs)

        # Verify correct return type
        if not isinstance(kernels, LocalOpKernels):
            raise TypeError(
                f"Factory for operator '{key}' returned {type(kernels)!r}, expected LocalOpKernels."
            )

        return LocalOperator(
            key=spec.key,
            kernels=kernels,
            description=spec.description,
            algebra=spec.algebra,
            sign_convention=spec.sign_convention,
            tags=spec.tags,
            parameters=merged_kwargs,
        )

    def build_local_operator_map(self, local_space: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Instantiate every operator registered for ``local_space``.

        Keyword arguments are forwarded to each factory (on top of the defaults
        declared in the specification). The return value is a mapping from operator
        key to :class:`LocalOperator`.
        """
        result: Dict[str, Any] = {}
        for spec in self.specs_for(local_space):
            result[spec.key] = self.instantiate(local_space, spec.key, **kwargs)
        return result


# ---------------------------------------------------------------------------
#! Global catalog instance
# ---------------------------------------------------------------------------

OPERATOR_CATALOG = OperatorCatalog()


def register_local_operator(
    local_space: Any,
    *,
    key: str,
    factory: Callable[..., Any],
    description: str,
    algebra: str,
    sign_convention: str,
    tags: Tuple[str, ...] = (),
    default_kwargs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Convenience wrapper around :meth:`OperatorCatalog.register`.
    Parameters
    ----------
    local_space:
        ``LocalSpaceTypes`` enum member or string identifying the local space.
    key:
        Unique operator identifier within the local space family.
    factory:
        Callable that returns a ``LocalOpKernels`` instance. It receives ``**kwargs``
        provided to :meth:`OperatorCatalog.instantiate`/``build_local_operator_map``.
    description:
        Short human readable explanation of what the operator does.
    algebra:
        Text snippet describing the algebraic relations (commutators, etc.).
    sign_convention:
        Explanation of the sign/statistics convention used by the kernels.
    tags:
        Optional collection of helper tags (e.g. ``("spin", "pauli")``) to make
        filtering simpler.
    default_kwargs:
        Keyword arguments forwarded to ``factory`` unless explicitly overridden.
    overwrite:
        If ``True`` an existing registration with the same key is replaced.
        Otherwise a ``ValueError`` is raised on duplicates.
    """

    OPERATOR_CATALOG.register(
        local_space,
        OperatorSpec(
            key=key,
            factory=factory,
            description=description,
            algebra=algebra,
            sign_convention=sign_convention,
            tags=tuple(tags),
            default_kwargs=default_kwargs or {},
        ),
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------

__all__ = [
    "OPERATOR_CATALOG",
    "OperatorCatalog",
    "OperatorSpec",
    "register_local_operator",
]

# ---------------------------------------------------------------------------
#! End of file
# ---------------------------------------------------------------------------
