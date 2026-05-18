"""
Internal NQS helpers.

This package hosts the maintained backend modules used by ``QES.NQS.nqs``:
precision policy, spectral result containers, exact helpers, and entropy
utilities. Most users should import from ``QES.NQS`` or ``QES.NQS.nqs``.
"""

import importlib

_LAZY_EXPORTS = {
    "NQSPrecisionPolicy"                : (".nqs_precision", "NQSPrecisionPolicy"),
    "resolve_precision_policy"          : (".nqs_precision", "resolve_precision_policy"),
    "cast_for_precision"                : (".nqs_precision", "cast_for_precision"),
    # Driver callbacks/loggers
    "AbstractLog"                       : (".nqs_driver", "AbstractLog"),
    "RuntimeLog"                        : (".nqs_driver", "RuntimeLog"),
    "JsonLog"                           : (".nqs_driver", "JsonLog"),
    "StopTraining"                      : (".nqs_driver", "StopTraining"),
    "InvalidLossStopping"               : (".nqs_driver", "InvalidLossStopping"),
    "ConvergenceStopping"               : (".nqs_driver", "ConvergenceStopping"),
    "TimeoutStopping"                   : (".nqs_driver", "TimeoutStopping"),
    # Spectral results
    "NQSCorrelatorResult"               : (".nqs_spectral", "NQSCorrelatorResult"),
    "NQSSpectralMapResult"              : (".nqs_spectral", "NQSSpectralMapResult"),
    "NQSSpectralResult"                 : (".nqs_spectral", "NQSSpectralResult"),
    # TDVP record
    "NQSTDVPRecord"                     : (".nqs_spectral", "NQSTDVPRecord"),
    "load_exact_impl"                   : (".nqs_exact", "load_exact_impl"),
    # Entropy computations
    "compute_ed_entanglement_entropy"   : (".nqs_entropy", "compute_ed_entanglement_entropy"),
    "compute_renyi_entropy"             : (".nqs_entropy", "compute_renyi_entropy"),
    "compute_renyi_entropies"           : (".nqs_entropy", "compute_renyi_entropies"),
    "compute_entropy_sweep"             : (".nqs_entropy", "compute_entropy_sweep"),
    "bipartition_cuts"                  : (".nqs_entropy", "bipartition_cuts"),
}

def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr_name  = _LAZY_EXPORTS[name]
    module                  = importlib.import_module(module_path, package=__name__)
    value                   = getattr(module, attr_name)
    globals()[name]         = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    # NQS precision policy
    "NQSPrecisionPolicy",
    "resolve_precision_policy",
    "cast_for_precision",
    "AbstractLog",
    "RuntimeLog",
    "JsonLog",
    "StopTraining",
    "InvalidLossStopping",
    "ConvergenceStopping",
    "TimeoutStopping",
    # NQS spectral results
    "NQSCorrelatorResult",
    "NQSSpectralMapResult",
    "NQSSpectralResult",
    # TDVP record
    "NQSTDVPRecord",
    # Exact loading
    "load_exact_impl",
    # Entropy computations
    "compute_ed_entanglement_entropy",
    "compute_renyi_entropy",
    "compute_renyi_entropies",
    "compute_entropy_sweep",
    "bipartition_cuts"
]

# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------
