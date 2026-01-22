"""
Declarative configuration helpers for constructing Hilbert spaces.

The :class:`HilbertConfig` dataclass provides a convenient way to package
all inputs required to create a :class:`~QES.Algebra.hilbert.HilbertSpace`
instance.  This allows higher-level modules, tests, or notebooks to assemble
re-usable blueprints that can be instantiated with small overrides (e.g.
changing the backend or toggling mapping generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from QES.general_python.lattices.lattice import Lattice, LatticeDirection

from .globals import GlobalSymmetry
from .Hilbert.hilbert_local import LocalSpace, StateTypes
from .Operator.operator import SymmetryGenerators


@dataclass(frozen=True)
class SymmetrySpec:
    """
    Lightweight container describing a symmetry generator and its sector.
    """

    generator: SymmetryGenerators
    sector: Union[int, float, complex]

    def as_tuple(self) -> Tuple[SymmetryGenerators, Union[int, float, complex]]:
        return self.generator, self.sector


StateFilter = Callable[[int], bool]


@dataclass(frozen=True)
class HilbertConfig:
    """
    Declarative description of a Hilbert space construction recipe.

    Parameters mirror the arguments of :class:`~QES.Algebra.hilbert.HilbertSpace`
    and can be further tailored via :meth:`with_override`.
    """

    ns: Optional[int] = None
    lattice: Optional[Lattice] = None
    nh: Optional[int] = None
    is_manybody: bool = True
    part_conserv: Optional[bool] = True
    local_space: Optional[LocalSpace] = None
    symmetry_generators: Tuple[SymmetrySpec, ...] = ()
    global_symmetries: Tuple[GlobalSymmetry, ...] = ()
    gen_mapping: bool = False
    state_type: StateTypes = StateTypes.INTEGER
    backend: str = "default"
    dtype: Optional[np.dtype] = np.float64
    boundary_flux: Optional[Mapping[Union[str, LatticeDirection], float]] = None
    state_filter: Optional[StateFilter] = None
    threadnum: int = 1
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def with_override(self, **updates: Any) -> "HilbertConfig":
        """
        Return a new config instance with selected fields replaced.
        """
        return replace(self, **updates)

    def sym_tuple(self) -> Tuple[Tuple[SymmetryGenerators, Union[int, float, complex]], ...]:
        """
        Return symmetry specifications as simple tuples.
        """
        return tuple(spec.as_tuple() for spec in self.symmetry_generators)

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Materialise the configuration as a kwargs dictionary suitable for
        passing to :class:`~QES.Algebra.hilbert.HilbertSpace`.
        """
        kwargs: Dict[str, Any] = {
            "ns": self.ns,
            "lattice": self.lattice,
            "nh": self.nh,
            "is_manybody": self.is_manybody,
            "part_conserv": self.part_conserv,
            "local_space": self.local_space,
            "sym_gen": list(self.sym_tuple()),
            "global_syms": list(self.global_symmetries),
            "gen_mapping": self.gen_mapping,
            "state_type": self.state_type,
            "backend": self.backend,
            "dtype": self.dtype,
            "boundary_flux": self.boundary_flux,
            "state_filter": self.state_filter,
            "threadnum": self.threadnum,
        }
        # Merge extra kwargs, ensuring we do not overwrite explicit keys with None.
        for key, value in self.extra_kwargs.items():
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = value
        return kwargs
