"""
Local Hilbert-space helpers.

This module defines light-weight containers that describe local Hilbert spaces
and their onsite operators.  They are used both by the operator catalog and by
high-level Hilbert space builders.
"""

import numpy as np
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

####################################################################################################

# kernels always return (states_array, coeffs_array)
# int:   (state:int, ns:int,    sites:np.ndarray[int32], *extra) -> (np.ndarray[int64], np.ndarray[float/complex])
# numpy: (state:np.ndarray,     sites:np.ndarray[int32], *extra) -> (np.ndarray[...,],  np.ndarray[...])
# jax:   (state:jnp.ndarray,    sites:jnp.ndarray[int32],*extra) -> (jnp.ndarray[...,], jnp.ndarray[...])
@dataclass(frozen=True)
class LocalOpKernels:
    fun_int             : Callable
    fun_np              : Optional[Callable] = None
    fun_jax             : Optional[Callable] = None
    # how many site indices the op needs at call-time (0=global/all-or-preset, 1=local, 2=correlation)
    site_parity         : int = 1
    # whether applying the op can change the state (helps sparse assembly sizing)
    modifies_state      : bool = True
    # default extra args bound at operator creation (can be overridden)
    default_extra_args  : Tuple = ()


@dataclass(frozen=True)
class LocalOperator:
    """
    Rich description of an onsite operator together with its kernels.
    """

    key             : str
    kernels         : LocalOpKernels
    description     : str
    algebra         : str
    sign_convention : str
    tags            : Tuple[str, ...] = field(default_factory=tuple)
    parameters      : Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """
        Return a compact human readable summary string.
        """
        return f"{self.key}: {self.description} ({self.algebra})"


class LocalSpaceTypes(Enum):
    SPIN_1_2            = "spin-1/2"
    SPIN_1              = "spin-1"
    SPINLESS_FERMIONS   = "spinless-fermions"
    ANYON_ABELIAN       = "abelian-anyons"
    BOSONS              = "bosons"


def _ensure_operator_modules_for(local_type: LocalSpaceTypes) -> None:
    """
    Lazily import operator modules to guarantee that catalog registrations
    are executed before we try to instantiate kernels.
    """
    try:
        if local_type in (LocalSpaceTypes.SPIN_1_2, LocalSpaceTypes.SPIN_1):
            import QES.Algebra.Operator.operators_spin                  # noqa: F401
        elif local_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            import QES.Algebra.Operator.operators_spinless_fermions     # noqa: F401,E402
        elif local_type == LocalSpaceTypes.ANYON_ABELIAN:
            import QES.Algebra.Operator.operators_anyon                 # noqa: F401,E402
        # elif local_type == LocalSpaceTypes.BOSONS:
            # import QES.Algebra.Operator.operators_bosons  # noqa: F401,E402
    except ImportError:
        # Optional module - if it is genuinely missing the catalog will
        # simply expose whatever was registered elsewhere.
        pass

@dataclass(frozen=True)
class LocalSpace:
    name                : str
    typ                 : LocalSpaceTypes
    local_dim           : int   # e.g. 2S+1 for spins, cutoff+1 for bosons
    sign_rule           : int   # +1 for bosons/spins, −1 for fermions
    cutoff              : int   # 0 unless a truncation is imposed
    max_occupation      : int   # 1 for hard-core species, cutoff for bosons
    onsite_operators    : Dict[str, LocalOperator] = field(default_factory=dict)

    def has_op(self, key: str) -> bool:
        return key in self.onsite_operators

    def get_op(self, key: str) -> LocalOperator:
        if not self.has_op(key):
            raise KeyError(f"Operator '{key}' not found.")
        return self.onsite_operators[key]

    def get_kernels(self, key: str) -> LocalOpKernels:
        """
        Convenience wrapper returning only the kernels.
        """
        return self.get_op(key).kernels

    def list_operator_keys(self) -> Tuple[str, ...]:
        return tuple(self.onsite_operators.keys())

    def describe_operator(self, key: str) -> str:
        op = self.get_op(key)
        tags = ", ".join(op.tags) if op.tags else "-"
        return (
            f"{op.key}: {op.description}\n"
            f"  Algebra         : {op.algebra}\n"
            f"  Sign convention : {op.sign_convention}\n"
            f"  Tags            : {tags}"
        )

    def with_catalog_operators(self, **kwargs) -> "LocalSpace":
        """
        Populate onsite operators using the global operator catalog for this space.
        """
        from QES.Algebra.Operator.catalog import OPERATOR_CATALOG

        _ensure_operator_modules_for(self.typ)

        ops = OPERATOR_CATALOG.build_local_operator_map(self.typ, **kwargs)
        return replace(self, onsite_operators=ops)

    def __str__(self) -> str:
        return (
            f"LocalSpace(name={self.name}, local_dim={self.local_dim}, "
            f"sign_rule={self.sign_rule}, cutoff={self.cutoff}, "
            f"max_occupation={self.max_occupation}, "
            f"operators={self.list_operator_keys()})"
        )

    def __repr__(self) -> str:
        return f"LocalSpace(name={self.name},nhl={self.local_dim},occ_max={self.max_occupation})"

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def default_spin_half(**kwargs):
        # Spin-1/2: dim=2, no cutoff, sign +1
        base = LocalSpace(
            name                =   LocalSpaceTypes.SPIN_1_2.value,
            local_dim           =   2,
            sign_rule           =   +1,
            cutoff              =   0,
            max_occupation      =   1,
            typ                 =   LocalSpaceTypes.SPIN_1_2,
            onsite_operators    =   {},
        )
        return base.with_catalog_operators(**kwargs)

    @staticmethod
    def default_fermion_spinless(**kwargs):
        # Spinless fermion: dim=2, Pauli exclusion, fermionic sign −1
        base = LocalSpace(
            name                =   LocalSpaceTypes.SPINLESS_FERMIONS.value,
            local_dim           =   2,
            sign_rule           =   -1,
            cutoff              =   0,
            max_occupation      =   1,
            typ                 =   LocalSpaceTypes.SPINLESS_FERMIONS,
            onsite_operators    =   {}
        )
        return base.with_catalog_operators(**kwargs)

    @staticmethod
    def default_boson(cutoff: int = 4, **kwargs):
        # Boson with cutoff (dim = cutoff+1)
        base = LocalSpace(
            name                =   LocalSpaceTypes.BOSONS.value,
            local_dim           =   cutoff + 1,
            sign_rule           =   +1,
            cutoff              =   cutoff,
            max_occupation      =   cutoff,
            typ                 =   LocalSpaceTypes.BOSONS,
            onsite_operators    =   {}
        )
        return base.with_catalog_operators(**kwargs)

    @staticmethod
    def default_abelian_anyon(statistics_angle: float, **kwargs):
        """
        Default hard-core abelian anyon space (dim=2) with exchange angle ``statistics_angle``.
        ``statistics_angle = pi`` reproduces fermions, ``0`` gives hard-core bosons.
        """
        base = LocalSpace(
            name                =   LocalSpaceTypes.ANYON_ABELIAN.value,
            local_dim           =   2,
            sign_rule           =   +1,  # explicit phases handled in kernels
            cutoff              =   0,
            max_occupation      =   1,
            typ                 =   LocalSpaceTypes.ANYON_ABELIAN,
            onsite_operators    =   {}
        )
        return base.with_catalog_operators(statistics_angle=statistics_angle, **kwargs)

    @staticmethod
    def default(**kwargs):
        return LocalSpace.default_spin_half(**kwargs)

#####################################################################################################

class StateTypes(Enum):
    INTEGER = "integer"
    VECTOR  = "vector"

    def lower(self):
        return self.value.lower()

    def upper(self):
        return self.value.upper()

    def __str__(self):
        return self.value

#####################################################################################################
