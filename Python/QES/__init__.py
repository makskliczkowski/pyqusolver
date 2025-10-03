"""
Quantum EigenSolver Package
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Date        : 2025-02-01
file        : QES/__init__.py
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

from contextlib import contextmanager

# Centralized globals (lazy singletons)
from .qes_globals import (
    get_logger,
    get_backend_manager,
    get_numpy_rng,
    reseed_all,
    next_jax_key as _next_jax_key,
    split_jax_keys as _split_jax_keys,
)

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

# -------------------------------------------------------------------------------------------------
#! End of QES package initialization
# -------------------------------------------------------------------------------------------------