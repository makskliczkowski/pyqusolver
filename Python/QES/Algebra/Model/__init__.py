"""
QES Model Module
================

This module provides implementations of various quantum many-body models.

Submodules:
-----------
- Interacting: Models with particle interactions
- Noninteracting: Free particle and non-interacting models

Classes:
--------
Various quantum model implementations including:
- Spin models (Heisenberg, Ising, XY, etc.)
- Fermionic models (Hubbard, t-J, etc.)
- Bosonic models

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from . import Interacting as intr

from . import Noninteracting as nintr


__all__ = ["intr", "nintr", "choose_model"]


def choose_model(model_name: str, **kwargs):

    """
    Factory function to choose a quantum model by name.
    
    Args:
        model_name (str):
            Type of model (e.g. "heisenberg_kitaev", "syk2", "aubry_andre", "xxz")
        **kwargs:
            Parameters for the model constructor (lattice, ns, etc.).
    Returns:
        Hamiltonian: An instance of the desired quantum model.
    """

    # Try interacting/spin models
    try:
        # Spin is under Interacting
        return intr.Spin.choose_model(model_name, **kwargs)
    except (ValueError, AttributeError):
        pass

    # Try non-interacting models
    try:
        return nintr.choose_model(model_name, **kwargs)
    except (ValueError, AttributeError):
        pass
    
    # Try dummy
    if model_name.lower() in ["dummy", "dummy_hamiltonian"]:
        from .dummy import DummyHamiltonian
        return DummyHamiltonian(**kwargs)
    raise ValueError(f"Unknown model '{model_name}'.")