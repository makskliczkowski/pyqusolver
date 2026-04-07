"""
Determinant Quantum Monte Carlo tools for QES.

Recommended entrypoint:

    `run_dqmc(model, beta=..., M=...)`

This module also exposes the lower-level model, sampler, stabilization, and
HS-building blocks used by the solver.
"""

from __future__ import annotations

import importlib

__all__ = [
    # High-level solver path
    "DQMCSolver",
    "run_dqmc",
    # Models and adapters
    "DQMCModel",
    "HubbardDQMCModel",
    "choose_dqmc_model",
    "DQMCModelAdapter",
    "HubbardAdapter",
    "SpinlessDensityDensityAdapter",
    "choose_dqmc_adapter",
    # Sampler and stabilization
    "DQMCSampler",
    "calculate_green_stable",
    "calculate_green_stable_numpy",
    "localized_diagonal_update",
    "localized_diagonal_update_ratio",
    "sherman_morrison_update",
    "propagate_green",
    "stack_product",
    "stack_product_numpy",
    "green_residual",
    "green_residual_numpy",
    "green_residual_from_stack",
    "green_residual_from_stack_numpy",
    # HS layer
    "HSTransformation",
    "MagneticHubbardHS",
    "ChargeHubbardHS",
    "GaussianHubbardHS",
    "CompactInterpolatingHubbardHS",
    "BondDensityDifferenceHS",
    "choose_hs_transformation",
    # Measurements
    "measure_equal_time",
    "measure_time_displaced",
]

_EXPORT_MAP = {
    "DQMCSolver"                     : (".dqmc_solver", "DQMCSolver"),
    "run_dqmc"                       : (".dqmc_solver", "run_dqmc"),
    "DQMCModel"                      : (".dqmc_model", "DQMCModel"),
    "HubbardDQMCModel"               : (".dqmc_model", "HubbardDQMCModel"),
    "choose_dqmc_model"              : (".dqmc_model", "choose_dqmc_model"),
    "DQMCModelAdapter"               : (".dqmc_adapter", "DQMCModelAdapter"),
    "HubbardAdapter"                 : (".dqmc_adapter", "HubbardAdapter"),
    "SpinlessDensityDensityAdapter"  : (".dqmc_adapter", "SpinlessDensityDensityAdapter"),
    "choose_dqmc_adapter"            : (".dqmc_adapter", "choose_dqmc_adapter"),
    "DQMCSampler"                    : (".dqmc_sampler", "DQMCSampler"),
    "calculate_green_stable"         : (".dqmc_sampler", "calculate_green_stable"),
    "calculate_green_stable_numpy"   : (".stabilization", "calculate_green_stable_numpy"),
    "localized_diagonal_update"      : (".stabilization", "localized_diagonal_update"),
    "localized_diagonal_update_ratio": (".stabilization", "localized_diagonal_update_ratio"),
    "sherman_morrison_update"        : (".dqmc_sampler", "sherman_morrison_update"),
    "propagate_green"                : (".dqmc_sampler", "propagate_green"),
    "stack_product"                  : (".stabilization", "stack_product"),
    "stack_product_numpy"            : (".stabilization", "stack_product_numpy"),
    "green_residual"                 : (".stabilization", "green_residual"),
    "green_residual_numpy"           : (".stabilization", "green_residual_numpy"),
    "green_residual_from_stack"      : (".stabilization", "green_residual_from_stack"),
    "green_residual_from_stack_numpy": (".stabilization", "green_residual_from_stack_numpy"),
    "HSTransformation"               : (".hs", "HSTransformation"),
    "MagneticHubbardHS"              : (".hs", "MagneticHubbardHS"),
    "ChargeHubbardHS"                : (".hs", "ChargeHubbardHS"),
    "GaussianHubbardHS"              : (".hs", "GaussianHubbardHS"),
    "CompactInterpolatingHubbardHS"  : (".hs", "CompactInterpolatingHubbardHS"),
    "BondDensityDifferenceHS"        : (".hs", "BondDensityDifferenceHS"),
    "choose_hs_transformation"       : (".hs", "choose_hs_transformation"),
    "measure_equal_time"             : (".measurements", "measure_equal_time"),
    "measure_time_displaced"         : (".measurements", "measure_time_displaced"),
}


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
