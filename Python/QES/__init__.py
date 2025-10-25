
"""
QES package initialization
=========================

File    : QES/__init__.py
Author  : Maks Kliczkowski
Email   : maxgrom97@gmail.com
Date    : 01.10.25
File    : QES/__init__.py

This is the top-level package for Quantum EigenSolver (QES).
It provides unified access to all submodules, global singletons, and core functionality.

Usage
-----
Import QES and its submodules:

    import QES
    from QES import qes_globals, Algebra, NQS, Solver
    log = qes_globals.get_logger()
    backend = qes_globals.get_backend_manager()
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
    # Convenience API exports (lazy):
    "HilbertSpace",
    "Hamiltonian",
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
from typing import Optional, Dict


import importlib
from contextlib import contextmanager
from typing import Optional, Dict

# Centralized globals (lazy singletons)
from .qes_globals import (
    get_logger,
    get_backend_manager,
    get_numpy_rng,
    reseed_all,
    next_jax_key as _next_jax_key,
    split_jax_keys as _split_jax_keys,
)

# Lightweight registry utilities
from .registry import list_modules, describe_module

####################################################################################################

def qes_reseed(seed: int):
    """Reseed the global backend manager RNGs.

    Thin wrapper over :func:`reseed_all` for backward compatibility.
    """
    reseed_all(seed)

def qes_next_key():
    """Return a fresh JAX PRNG subkey (if JAX backend active)."""
    return _next_jax_key()

def qes_split_keys(n: int):
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

_SUBMODULES: Dict[str, str] = {
    'Algebra'           : 'QES.Algebra',
    'NQS'               : 'QES.NQS',
    'Solver'            : 'QES.Solver',
    'general_python'    : 'QES.general_python',
}

_API_EXPORTS: Dict[str, str] = {
    'HilbertSpace': 'QES.Algebra.hilbert',
    'Hamiltonian': 'QES.Algebra.hamil',
}

def __getattr__(name: str):  # PEP 562
    if name in _SUBMODULES:
        return importlib.import_module(_SUBMODULES[name])
    if name in _API_EXPORTS:
        mod = importlib.import_module(_API_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'QES' has no attribute {name!r}")

# -------------------------------------------------------------------------------------------------
#! End of QES package initialization
# -------------------------------------------------------------------------------------------------