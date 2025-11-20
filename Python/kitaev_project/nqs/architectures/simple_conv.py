"""
Lightweight convolutional ansatz.
"""
from dataclasses import dataclass, field
from typing import Dict

from .base import BaseQESAnsatz


@dataclass
class SimpleConvAnsatz(BaseQESAnsatz):
    name: str = "simple_conv"
    net_spec: str = "simple_conv"
    net_kwargs: Dict[str, int] = field(default_factory=lambda: {"n_filters": 16, "depth": 2})

