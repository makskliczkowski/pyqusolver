"""
Autoregressive neural ansatz definition.
"""
from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseQESAnsatz


@dataclass
class AutoregressiveAnsatz(BaseQESAnsatz):
    """
    Configuration for an Autoregressive ansatz.

    Autoregressive models build the probability of a sequence one element at a time,
    making them powerful for generating samples exactly.

    Attributes:
        name (str): Unique identifier for the ansatz ('autoregressive').
        net_spec (str): Network key for the factory ('ar' or 'autoregressive').
        net_kwargs (Dict): Keyword arguments for the autoregressive network constructor,
                           such as `depth` and `num_hidden`.

    Example:
    --------
        >>> from QES.pyqusolver.nqs import NQSSolver
        >>> from QES.pyqusolver.models import KitaevModel
        >>> from QES.pyqusolver.nqs.architectures import AutoregressiveAnsatz
        >>>
        >>> # Define the model
        >>> model = KitaevModel(Lx=3, Ly=3)
        >>>
        >>> # Define the Autoregressive ansatz
        >>> ar_ansatz = AutoregressiveAnsatz(
        ...     net_kwargs={'depth': 2, 'num_hidden': 64}
        ... )
        >>>
        >>> # Initialize the solver
        >>> solver = NQSSolver(model=model, ansatz=ar_ansatz)
        >>>
        >>> print(solver.nqs.net)
    """
    name: str = "autoregressive"
    net_spec: str = "autoregressive"
    net_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "depth": 2,
        "num_hidden": 32,
    })
