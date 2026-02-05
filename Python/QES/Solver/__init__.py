"""
QES Solver Module
=================

Simulation engines and optimization backends.

This module provides the abstract interfaces and concrete implementations for
simulating quantum systems. It primarily houses the Monte Carlo engines used
by NQS.

Entry Points
------------
- :class:`MonteCarloSolver`: Base class for MC-based simulations.
- :class:`QES.Solver.MonteCarlo.sampler.Sampler`: Abstract interface for sampling.
- :class:`QES.Solver.MonteCarlo.vmc.VMCSampler`: Metropolis-Hastings sampler.

Flow
----
::

    Sampler (MCMC)
        |
        v
    MonteCarloSolver (Optimization Loop)
        |
        v
    Physics Results

Submodules
----------
- ``MonteCarlo``: Samplers and VMC logic.
- ``solver``: Abstract base classes.

Invariants
----------
- **Compatibility**: Solvers expect a `Hamiltonian` and `HilbertSpace` that match
  (e.g., same number of sites, compatible basis).
- **Interface**: Solvers usually implement a `run()` or `train()` method that
  returns a results object (e.g., `McsReturn` or `NQSTrainStats`).
"""

import importlib
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Lazy Import Configuration
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    "MonteCarlo": (".MonteCarlo", None),
    "Solver": (".solver", "Solver"),
}

if TYPE_CHECKING:
    from . import MonteCarlo
    from .solver import Solver


def __getattr__(name: str) -> Any:
    """Lazily import submodules and classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        if attr_name:
            return getattr(module, attr_name)
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


__all__ = [
    "Solver",
    "MonteCarlo",
]

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = "Core solver interfaces and Monte Carlo-based solvers."
