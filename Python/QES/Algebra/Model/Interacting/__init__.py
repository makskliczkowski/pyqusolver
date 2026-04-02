"""Interacting quantum model definitions.

Modules:
--------
- Spin: Spin models (Heisenberg, Ising, XXZ, etc.)
- Fermionic: Interacting many-body fermion models

-------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
License         : MIT
Version         : 1.0
-------------------------------------------------------------------------
"""

from __future__ import annotations
from typing     import TYPE_CHECKING

__all__: list[str] = ["Spin", "Fermionic", "choose_model"]

import importlib

if TYPE_CHECKING:
    from .Spin      import Spin
    from .Fermionic import Fermionic

def __getattr__(name: str):
    ''' Lazy import of submodules. '''
    if name == "Spin":
        module = importlib.import_module(".Spin", __name__)
        globals()[name] = module
        return module
    if name == "Fermionic":
        module = importlib.import_module(".Fermionic", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)

def choose_model(model_name: str, **kwargs):
    """
    Pick an interacting model across available families.
    Parameters    
    ----------
    model_name : str
        Name of the model to choose. Should be unique across all families.
    **kwargs : dict
        Additional parameters to pass to the model constructor.
    Returns
    -------
    Hamiltonian
        An instance of the chosen model.
    Raises
    ------
    ValueError
        If the model name is not recognized in any family.
    Notes
    -----
    The function first tries to find the model in the Spin family, then in the Fermionic
    """
    # Spin family
    try:
        return __getattr__("Spin").choose_model(model_name, **kwargs)
    except ValueError as exc:
        if "Unknown spin model" not in str(exc):
            raise
    except AttributeError:
        pass

    # Fermionic family
    try:
        return __getattr__("Fermionic").choose_model(model_name, **kwargs)
    except ValueError as exc:
        if "Unknown interacting fermionic model" not in str(exc):
            raise
    except AttributeError:
        pass

    raise ValueError(f"Unknown interacting model '{model_name}'.")

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------