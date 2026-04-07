"""Monte Carlo solvers and sampling utilities.

This package contains Monte Carlo building blocks used by variational and
finite-temperature workflows.

Invariants and data-shape notes
-------------------------------
* Sampler state containers are backend array objects with leading batch/chain
  axes where applicable.
* Estimator outputs are real or complex scalars/vectors depending on the
  measured observable.
* Import paths under ``QES.Solver.MonteCarlo`` are preserved for compatibility.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_IMPORTS = {
    "MonteCarloSolver"      : (".montecarlo", "MonteCarloSolver"),
    "McsTrain"              : (".montecarlo", "McsTrain"),
    "McsReturn"             : (".montecarlo", "McsReturn"),
    "Sampler"               : (".sampler", "Sampler"),
    "SolverInitState"       : (".sampler", "SolverInitState"),
    "get_sampler"           : (".sampler", "get_sampler"),
    # Diagnostics
    "autocorr_func_1d"      : (".diagnostics", "autocorr_func_1d"),
    "compute_autocorr_time" : (".diagnostics", "compute_autocorr_time"),
    "compute_ess"           : (".diagnostics", "compute_ess"),
    "compute_rhat"          : (".diagnostics", "compute_rhat"),
    "jackknife_estimate"    : (".diagnostics", "jackknife_estimate"),
    "bootstrap_estimate"    : (".diagnostics", "bootstrap_estimate"),
}

if TYPE_CHECKING:
    from .montecarlo import McsReturn, McsTrain, MonteCarloSolver
    from .sampler import Sampler, SolverInitState, get_sampler


__all__ = [
    "MonteCarloSolver",
    "McsTrain",
    "McsReturn",
    "Sampler",
    "SolverInitState",
    "get_sampler",
    "autocorr_func_1d",
    "compute_autocorr_time",
    "compute_ess",
    "compute_rhat",
    "jackknife_estimate",
    "bootstrap_estimate",
    "parallel",
    "montecarlo",
    "sampler",
    "vmc",
    "diagnostics",
    "updates",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return module if attr_name is None else getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_IMPORTS.keys()) + __all__))
