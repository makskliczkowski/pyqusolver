"""Interacting quantum model definitions.

Modules:
--------
- Spin: Spin models (Heisenberg, Ising, XXZ, etc.)

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

__all__: list[str] = ["Spin"]

import importlib

def __getattr__(name: str):
    if name == "Spin":
        module = importlib.import_module(".Spin", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
