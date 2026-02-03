"""
Centralized global singletons for the QES package.

Motivation
==========
Historically several modules (e.g. `general_python.algebra.utils`, test helpers,
and sub-packages) attempted to (re)initialize logging, backend managers, and
random number generation on import. This led to duplicated initialization,
especially for the NumPy / JAX backend manager and the global logger, and in
some cases race conditions or double printing of startup banners.

This module provides a SINGLE authoritative place where these shared objects
are created exactly once per Python process. All other code should import
the accessors defined here instead of constructing new instances.

Provided Singletons
-------------------
- Global logger        : via `get_logger()` (returns the flog.Logger instance)
- Backend manager      : via `get_backend_manager()`
- RNG conveniences     : via `get_numpy_rng()` / `next_jax_key()` / `split_jax_keys()`

Usage Pattern
-------------
    from QES.qes_globals import get_logger, get_backend_manager, get_numpy_rng, next_jax_key

    log                 = get_logger()
    backend_mgr         = get_backend_manager()
    xp                  = backend_mgr.np
    rng                 = get_numpy_rng()

    with backend_mgr.seed_scope(123):
        ... deterministic code ...

Design Notes
------------
We lazily import heavy modules (like algebra.utils) so importing just the
top-level `QES` package remains lightweight. The first call to
`get_backend_manager()` triggers the underlying backend initialization in
`general_python.algebra.utils`.

!IMPORTANT: Do NOT perform side effects at module import other than creating
!lightweight sentinels; heavy initialization is deferred until first access.
"""

from    __future__ import annotations

import  threading
from    typing import Any, List, Union

try:
    from numpy.random import Generator
except ImportError:
    Generator   = Any

# Thread-local storage for singletons
_LOCK           = threading.Lock()

# Internal storage for singletons
_LOGGER         : Any = None
_BACKEND_MGR    : Any = None

# ----------------------------------------------------------------
#! Global logger accessor
# ----------------------------------------------------------------

def get_logger(**kwargs) -> Any:
    """
    Return the process-global logger instance.

    Parameters
    ----------
    **kwargs : dict
        Optional keyword arguments forwarded to `get_global_logger` the first
        time the logger is created.

    Returns
    -------
    QES.general_python.common.flog.Logger
        The global logger instance.
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    with _LOCK:
        if _LOGGER is None:
            from QES.general_python.common.flog import get_global_logger

            _LOGGER = get_global_logger(**kwargs)
    return _LOGGER

# ----------------------------------------------------------------
#! Global backend manager accessor
# ----------------------------------------------------------------

def get_backend_manager() -> Any:
    """
    Return the global backend manager (lazy import).

    The first call imports `general_python.algebra.utils` which performs its
    guarded initialization. Subsequent calls are cheap.

    Returns
    -------
    QES.general_python.algebra.utils.BackendManager
        The global backend manager handling NumPy/JAX dispatch and RNGs.
    """
    global _BACKEND_MGR
    if _BACKEND_MGR is not None:
        return _BACKEND_MGR
    with _LOCK:
        if _BACKEND_MGR is None:
            from QES.general_python.algebra.utils import backend_mgr

            _BACKEND_MGR = backend_mgr
    return _BACKEND_MGR

# ----------------------------------------------------------------
#! RNG accessors
# ----------------------------------------------------------------

def get_numpy_rng() -> Generator:
    """
    Return the NumPy Generator instance from the backend manager.

    This generator should be used for all NumPy-based random number generation
    to ensure reproducibility when reseeding via the backend manager.

    Returns
    -------
    numpy.random.Generator
        The global NumPy random generator.
    """
    mgr = get_backend_manager()
    return mgr.default_rng

def reseed_all(seed: int) -> Any:
    """
    Reseed backend manager (NumPy / JAX) and python's random state.

    Parameters
    ----------
    seed : int
        The integer seed to set.

    Returns
    -------
    QES.general_python.algebra.utils.BackendManager
        The backend manager object, for chaining.
    """
    mgr = get_backend_manager()
    return mgr.reseed(seed)

# ----------------------------------------------------------------

def next_jax_key() -> Any:
    """
    Get a fresh JAX subkey from the backend manager (if JAX active).

    Returns
    -------
    jax.random.PRNGKey or None
        A new JAX key if JAX is the active backend, otherwise None.
    """
    mgr = get_backend_manager()
    return mgr.next_key()

def split_jax_keys(n: int) -> Union[Any, List[None]]:
    """
    Split the current JAX key into ``n`` subkeys (if JAX active).

    Parameters
    ----------
    n : int
        Number of keys to generate.

    Returns
    -------
    jax.random.PRNGKeyArray or List[None]
        An array of n keys if JAX is active, otherwise a list of Nones.
    """
    mgr = get_backend_manager()
    return mgr.split_keys(n)

# ----------------------------------------------------------------

__all__ = [
    "get_logger",
    "get_backend_manager",
    "get_numpy_rng",
    "reseed_all",
    "next_jax_key",
    "split_jax_keys",
]

# ----------------------------------------------------------------
#! End of QES global singletons
# ----------------------------------------------------------------
