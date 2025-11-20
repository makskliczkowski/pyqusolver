"""
Model factory wrapping the QES Heisenberg–Kitaev Hamiltonian.

This module keeps all lattice / impurity / coupling plumbing in one place so both
the NQS and ED stacks can request *identically parameterized* Hamiltonians.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
    from QES.general_python.lattices.honeycomb import HoneycombLattice
    from QES.Algebra.hilbert import HilbertSpace


def _import_qes_components():
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
    from QES.general_python.lattices.honeycomb import HoneycombLattice
    from QES.Algebra.hilbert import HilbertSpace

    return HeisenbergKitaev, HoneycombLattice, HilbertSpace


@dataclass
class ModelConfig:
    """Typed container for all model knobs."""

    lx: int = 3
    ly: int = 2
    lz: int = 1
    dim: int = 2
    bc: str = "pbc"
    use_forward: bool = True
    dtype: str = "float64"

    # Couplings
    K: Dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
    )
    J: Optional[float] = None
    delta: float = 1.0
    Gamma: Dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    hx: Optional[float] = None
    hz: Optional[float] = None

    # Disorder / impurities
    impurities: Sequence[Tuple[int, float]] = field(default_factory=list)

    def lattice_kwargs(self) -> Dict[str, int]:
        return {"dim": self.dim, "lx": self.lx, "ly": self.ly, "lz": self.lz, "bc": self.bc}

    def dtype_obj(self):
        return getattr(np, self.dtype)

    def summary(self) -> Dict[str, object]:
        data = asdict(self)
        data["dtype"] = self.dtype
        return data


class KitaevModelBuilder:
    """
    Wraps lattice creation and Hamiltonian instantiation.

    Allows a single entry point for every workflow component.
    """

    def __init__(self, lattice_cls=None):
        self._lattice_cls = lattice_cls

    def build_lattice(self, config: ModelConfig) -> "HoneycombLattice":
        _, honeycomb_cls, _ = _import_qes_components()
        cls = self._lattice_cls or honeycomb_cls
        return cls(**config.lattice_kwargs())

    def build_hilbert_space(self, lattice: "HoneycombLattice") -> "HilbertSpace":
        *_, hilbert_cls = _import_qes_components()
        return hilbert_cls(lattice=lattice)

    def build_hamiltonian(
        self,
        config: ModelConfig,
        *,
        hilbert_space: Optional[HilbertSpace] = None,
    ) -> "HeisenbergKitaev":
        HeisenbergKitaev, _, _ = _import_qes_components()
        lattice = self.build_lattice(config)
        hilbert = hilbert_space or self.build_hilbert_space(lattice)
        dtype = config.dtype_obj()
        K = tuple(config.K.get(axis, 0.0) for axis in ("x", "y", "z"))
        Gamma = tuple(config.Gamma.get(axis, 0.0) if config.Gamma.get(axis) is not None else None for axis in ("x", "y", "z"))
        return HeisenbergKitaev(
            hilbert_space=hilbert,
            lattice=lattice,
            K=K,
            J=config.J,
            dlt=config.delta,
            Gamma=Gamma,
            hx=config.hx,
            hz=config.hz,
            impurities=list(config.impurities),
            dtype=dtype,
            use_forward=config.use_forward,
        )

    def with_impurities(self, config: ModelConfig, impurities: Iterable[Tuple[int, float]]) -> ModelConfig:
        updated = asdict(config)
        updated["impurities"] = list(impurities)
        return ModelConfig(**updated)
