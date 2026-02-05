"""
Monte Carlo Solver Module
=========================

This module provides Monte Carlo methods for quantum many-body systems.

Modules:
--------
- montecarlo: Core Monte Carlo algorithms
- parallel: Parallel Monte Carlo implementations
- sampler: Sampling algorithms for quantum states
- vmc: Variational Monte Carlo sampler

Invariants
----------
- **Sampler Reset**: If the number of chains or the system size changes,
  `sampler.reset()` must be called to resize internal buffers.
- **Batched Inputs**: Samplers operate on batches of states (shape `(n_chains, n_sites)`).
- **Backend**: Samplers respect the global backend setting (NumPy vs JAX).

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

import importlib
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Lazy Import Configuration
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # montecarlo
    "MonteCarloSolver": (".montecarlo", "MonteCarloSolver"),
    "McsTrain": (".montecarlo", "McsTrain"),
    "McsReturn": (".montecarlo", "McsReturn"),
    # sampler
    "Sampler": (".sampler", "Sampler"),
    "SamplerErrors": (".sampler", "SamplerErrors"),
    "SamplerType": (".sampler", "SamplerType"),
    "get_sampler": (".sampler", "get_sampler"),
    "UpdateRule": (".sampler", "UpdateRule"),
    "get_update_function": (".sampler", "get_update_function"),
    # parallel
    "ParallelTempering": (".parallel", "ParallelTempering"),
    "BetaSpacing": (".parallel", "BetaSpacing"),
    # vmc
    "VMCSampler": (".vmc", "VMCSampler"),
}

_SUBMODULES = {
    "montecarlo": "Core Monte Carlo algorithms",
    "parallel": "Parallel Monte Carlo implementations",
    "sampler": "Sampling algorithms for quantum states",
    "vmc": "Variational Monte Carlo sampler",
}

if TYPE_CHECKING:
    from .montecarlo import McsReturn, McsTrain, MonteCarloSolver
    from .parallel import BetaSpacing, ParallelTempering
    from .sampler import (
        Sampler,
        SamplerErrors,
        SamplerType,
        UpdateRule,
        get_sampler,
        get_update_function,
    )
    from .vmc import VMCSampler


def __getattr__(name: str) -> Any:
    """Lazily import submodules and classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    # Allow importing submodules directly (e.g. QES.Solver.MonteCarlo.vmc)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", package=__name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()) + list(_SUBMODULES.keys()))


__all__ = [
    # montecarlo
    "MonteCarloSolver",
    "McsTrain",
    "McsReturn",
    # sampler
    "Sampler",
    "SamplerErrors",
    "SamplerType",
    "get_sampler",
    "UpdateRule",
    "get_update_function",
    # parallel
    "ParallelTempering",
    "BetaSpacing",
    # vmc
    "VMCSampler",
    # submodules
    "montecarlo",
    "parallel",
    "sampler",
    "vmc",
]
