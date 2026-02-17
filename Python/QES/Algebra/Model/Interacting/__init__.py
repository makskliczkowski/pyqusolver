"""Interacting quantum model definitions.

Modules:
--------
- Spin: Spin models (Heisenberg, Ising, XXZ, etc.)
- Fermionic: Interacting many-body fermion models

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

__all__: list[str] = ["Spin", "Fermionic", "choose_model"]

import importlib

def __getattr__(name: str):
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