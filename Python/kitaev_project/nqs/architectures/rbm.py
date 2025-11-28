"""
Restricted Boltzmann Machine ansatz.
"""
from dataclasses import dataclass, field
from typing import Dict

from .base import BaseQESAnsatz


@dataclass
class RBMAnsatz(BaseQESAnsatz):
    """
    Configuration for a Restricted Boltzmann Machine (RBM) ansatz.

    Attributes:
        name (str): Unique identifier for the ansatz ('rbm').
        net_spec (str): Network key for the factory ('rbm').
        net_kwargs (Dict): Dictionary of keyword arguments passed to the RBM network constructor.
                           For example, `{'alpha': 2.0}` sets the hidden unit density.

    Example:
    --------
        >>> from QES.pyqusolver.nqs import NQSSolver
        >>> from QES.pyqusolver.models import KitaevModel
        >>> from QES.pyqusolver.nqs.architectures import RBMAnsatz
        >>>
        >>> # Define the model
        >>> model = KitaevModel(Lx=2, Ly=2)
        >>>
        >>> # Define the RBM ansatz with a hidden unit density of 1.5
        >>> rbm_ansatz = RBMAnsatz(net_kwargs={'alpha': 1.5})
        >>>
        >>> # Initialize the solver
        >>> solver = NQSSolver(model=model, ansatz=rbm_ansatz)
        >>>
        >>> # The solver will now use an RBM with n_hidden = 1.5 * n_visible
        >>> print(solver.nqs.net)
    """
    name: str = "rbm"
    net_spec: str = "rbm"
    net_kwargs: Dict[str, float] = field(default_factory=lambda: {"alpha": 2.0})

