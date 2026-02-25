"""
pydqmc Module.
State-of-the-art Determinant Quantum Monte Carlo scheme for QES.
"""

from __future__ import annotations
import importlib

__all__ = [
    # Models
    "DQMCModel",
    "HubbardDQMCModel",
    "choose_dqmc_model",
    # Sampler
    "DQMCSampler",
    "calculate_green_stable",
    "sherman_morrison_update",
    "propagate_green",
    # Solver
    "DQMCSolver",
]

_EXPORT_MAP = {
    # dqmc_model
    "DQMCModel": (".dqmc_model", "DQMCModel"),
    "HubbardDQMCModel": (".dqmc_model", "HubbardDQMCModel"),
    "choose_dqmc_model": (".dqmc_model", "choose_dqmc_model"),
    # dqmc_sampler
    "DQMCSampler": (".dqmc_sampler", "DQMCSampler"),
    "calculate_green_stable": (".dqmc_sampler", "calculate_green_stable"),
    "sherman_morrison_update": (".dqmc_sampler", "sherman_morrison_update"),
    "propagate_green": (".dqmc_sampler", "propagate_green"),
    # dqmc_solver
    "DQMCSolver": (".dqmc_solver", "DQMCSolver"),
}

def __getattr__(name: str):
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)
