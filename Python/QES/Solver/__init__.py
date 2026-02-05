"""QES solver package.

Provides abstract solver interfaces and concrete algorithm families.

Correctness notes
-----------------
* ``Solver`` subclasses should expose deterministic ``init``/``train`` behavior
  for fixed random seeds.
* Solver outputs should record metadata needed for reproducibility (settings,
  acceptance rates, or convergence diagnostics depending on solver type).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

MODULE_DESCRIPTION = "Core solver interfaces and Monte Carlo-based solvers."

_LAZY_IMPORTS = {
    "Solver": (".solver", "Solver"),
    "MonteCarlo": (".MonteCarlo", None),
}

if TYPE_CHECKING:
    from .solver import Solver
    from . import MonteCarlo

__all__ = ["Solver", "MonteCarlo"]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return module if attr_name is None else getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_IMPORTS.keys()) + __all__))
