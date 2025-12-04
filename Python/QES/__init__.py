"""
QES package initialization
=========================

Quantum EigenSolver (QES): Comprehensive framework for quantum eigenvalue problem solving.

This is the top-level package for Quantum EigenSolver (QES).
It provides unified access to all submodules, global singletons, and core functionality.

Usage
-----
Import QES and its submodules:

    import QES
    from QES import Algebra, NQS, Hamiltonian
    
    log     = QES.get_logger()
    backend = QES.get_backend_manager()
    
----------------------------------------------------------
Author          : Maks Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.10.2025
Description     : Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving.
----------------------------------------------------------
"""

__version__         = "0.1.0"
__author__          = "Maksymilian Kliczkowski"
__email__           = "maksymilian.kliczkowski@pwr.edu.pl"
__license__         = "CC-BY-4.0"
__description__     = "Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving"

__all__ = [
    # RNG / backend helpers (public stable API surface)
    "qes_reseed",
    "qes_next_key",
    "qes_split_keys",
    "qes_seed_scope",
    # Discovery utilities
    "list_modules",
    "describe_module",
    # --- Convenience API exports (lazy) ---
    # Core
    "HilbertSpace",
    "Hamiltonian",
    "Operator",
    # Solvers
    "NQS",
    "MonteCarloSolver",
    "Sampler",
    # Networks
    "RBM",
    "CNN",
    "ResNet",
    "Autoregressive",
    "SimpleNet",
    "choose_network",
    # Global accessor re-exports
    "get_logger",
    "get_backend_manager",
    "get_numpy_rng",
    "reseed_all",
    "next_jax_key",
    "split_jax_keys",
    # Meta
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
]

####################################################################################################

import importlib
from contextlib import contextmanager
from typing import Optional, Dict, Any

# Centralized globals (lazy singletons)
from .qes_globals import (
    get_logger,
    get_backend_manager,
    get_numpy_rng,
    reseed_all,
    next_jax_key        as _next_jax_key,
    split_jax_keys      as _split_jax_keys,
)

# Lightweight registry utilities
from .registry import list_modules, describe_module

####################################################################################################

def qes_reseed(seed: int) -> None:
    """Reseed the global backend manager RNGs.

    Thin wrapper over :func:`reseed_all` for backward compatibility.
    """
    reseed_all(seed)

def qes_next_key() -> Any:
    """Return a fresh JAX PRNG subkey (if JAX backend active)."""
    return _next_jax_key()

def qes_split_keys(n: int) -> Any:
    """Split the current JAX PRNG key into ``n`` subkeys (if JAX active)."""
    return _split_jax_keys(n)

@contextmanager
def qes_seed_scope(seed: int, *, touch_numpy_global: bool = False, touch_python_random: bool = False):
    """Context manager to temporarily set a deterministic seed across backends.

    Delegates to the backend manager's ``seed_scope``.
    """
    backend_mgr = get_backend_manager()
    with backend_mgr.seed_scope(seed, touch_numpy_global=touch_numpy_global, touch_python_random=touch_python_random) as suite:
        yield suite

# ----------------------------------------------------------------------------
# Lazy access to top-level subpackages and common classes (keeps `import QES` light)
# ----------------------------------------------------------------------------

# Top-level packages accessible as `QES.Submodule`
_SUBMODULES: Dict[str, str] = {
    'Algebra'           : 'QES.Algebra',
    'NQS'               : 'QES.NQS',
    'Solver'            : 'QES.Solver',
    'general_python'    : 'QES.general_python',
}

# Specific classes/functions accessible as `from QES import Object` or `QES.Object`
_API_EXPORTS: Dict[str, str] = {
    # Core from QES.Algebra
    'HilbertSpace'      : 'QES.Algebra.hilbert',
    'Hamiltonian'       : 'QES.Algebra.hamil',
    'Operator'          : 'QES.Algebra.Operator.operator',
    # Core from QES.NQS
    'NQS'               : 'QES.NQS.nqs',
    # Core from QES.Solver
    'MonteCarloSolver'  : 'QES.Solver.MonteCarlo.montecarlo',
    'Sampler'           : 'QES.Solver.MonteCarlo.sampler',
    # Networks from QES.general_python.ml
    'RBM'               : 'QES.general_python.ml.net_impl.networks.net_rbm',
    'CNN'               : 'QES.general_python.ml.net_impl.networks.net_cnn',
    'ResNet'            : 'QES.general_python.ml.net_impl.networks.net_res',
    'Autoregressive'    : 'QES.general_python.ml.net_impl.networks.net_autoregressive',
    'SimpleNet'         : 'QES.general_python.ml.net_impl.net_simple',
    'choose_network'    : 'QES.general_python.ml.networks',
}

# Deeper submodules accessible as `QES.alias`, e.g., `QES.gp_ml`
_LAZY_MODULES: Dict[str, str] = {
    # general_python modules
    'gp_ml'                     : 'QES.general_python.ml',
    'gp_algebra'                : 'QES.general_python.algebra',
    'gp_common'                 : 'QES.general_python.common',
    'gp_lattices'               : 'QES.general_python.lattices',
    'gp_maths'                  : 'QES.general_python.maths',
    'gp_physics'                : 'QES.general_python.physics',
    # ML submodules
    'gp_activation_functions'   : 'QES.general_python.ml.net_impl.activation_functions',
    'gp_interface_net_flax'     : 'QES.general_python.ml.net_impl.interface_net_flax',
    'gp_net_general'            : 'QES.general_python.ml.net_impl.net_general',
    'gp_networks'               : 'QES.general_python.ml.networks',
    'gp_schedulers'             : 'QES.general_python.ml.schedulers',
    # Common utilities
    'gp_flog'                   : 'QES.general_python.common.flog',
    'gp_timer'                  : 'QES.general_python.common.timer',
    'gp_binary'                 : 'QES.general_python.common.binary',
    'gp_directories'            : 'QES.general_python.common.directories',
}

def __getattr__(name: str) -> Any:  # PEP 562
    if name in _SUBMODULES:
        return importlib.import_module(_SUBMODULES[name])
    if name in _API_EXPORTS:
        mod = importlib.import_module(_API_EXPORTS[name])
        return getattr(mod, name)
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    raise AttributeError(f"module 'QES' has no attribute {name!r}")

# -------------------------------------------------------------------------------------------------
#! End of QES package initialization
# -------------------------------------------------------------------------------------------------