"""
Base utilities for defining NeuralAnsatz implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..types import NeuralAnsatz, NQSTrainingConfig

try:
    from QES.NQS.nqs import NQS
except ImportError:  # pragma: no cover - optional dependency at scaffold stage
    NQS = Any  # type: ignore


@dataclass
class BaseQESAnsatz(NeuralAnsatz):
    """
    Thin wrapper holding the information needed to instantiate QES.NQS.
    """

    name: str
    net_spec: Any
    net_kwargs: Dict[str, Any] = field(default_factory=dict)
    sampler_override: Optional[Any] = None

    def create_solver(self, hamiltonian, hilbert_space, config: NQSTrainingConfig):
        sampler = self.sampler_override or config.sampler
        return NQS(
            net=self.net_spec,
            sampler=sampler,
            model=hamiltonian,
            batch_size=config.batch_size,
            hilbert=hilbert_space,
            backend=config.backend,
            seed=config.seed,
            **self.net_kwargs,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name, "net_spec": self.net_spec, **self.net_kwargs}
