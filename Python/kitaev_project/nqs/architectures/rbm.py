"""
Restricted Boltzmann Machine ansatz.
"""
from dataclasses import dataclass, field
from typing import Dict

from .base import BaseQESAnsatz


@dataclass
class RBMAnsatz(BaseQESAnsatz):
    name: str = "rbm"
    net_spec: str = "rbm"
    net_kwargs: Dict[str, float] = field(default_factory=lambda: {"hidden_density": 2.0})

