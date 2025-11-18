"""
Autoregressive neural ansatz definition.
"""
from dataclasses import dataclass

from .base import BaseQESAnsatz


@dataclass
class AutoregressiveAnsatz(BaseQESAnsatz):
    name: str = "autoregressive"
    net_spec: str = "autoregressive"

