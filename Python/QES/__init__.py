"""
Top-level QES package.

This module exposes the maintained package entrypoints, global RNG/backend
helpers, module discovery utilities, and lazy imports for the main QES
subpackages.

Typical usage
-------------
    import QES
    from QES import Algebra

    log = QES.get_logger()
    backend = QES.get_backend_manager()
    modules = QES.list_modules()

----------------------------------------------------------
Author          : Maks Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.10.2025
Modified        : 08.03.2026
Description     : Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving.
----------------------------------------------------------
"""

from __future__ import annotations

__version__         = "1.3.0"
__author__          = "Maksymilian Kliczkowski"
__email__           = "maksymilian.kliczkowski@pwr.edu.pl"
__license__         = "CC-BY-4.0"
__description__     = (
    "Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving"
)

import  importlib as _importlib
import  os as _os
import  warnings as _warnings
from    contextlib import contextmanager as _contextmanager
from    typing import TYPE_CHECKING
import  typing as _t


def _apply_master_backend_env_defaults() -> None:
    """Normalize global backend env flags before importing backend-aware modules."""
    jax_dont_use    = _os.environ.get("PY_JAX_DONT_USE", "0") in ("1", "true", "True")
    force_cpu       = (
                        _os.environ.get("CUDA_VISIBLE_DEVICES")     == ""
                        or _os.environ.get("PY_JAX_CPU_ONLY", "0")  in ("1", "true", "True")
                        or _os.environ.get("PY_FORCE_CPU", "0")     in ("1", "true", "True")
                    )

    if jax_dont_use:
        _os.environ.setdefault("PY_BACKEND",            "np")
        _os.environ.setdefault("QES_BACKEND",           "numpy")
        _os.environ.setdefault("CUDA_VISIBLE_DEVICES",  "")
        _os.environ.setdefault("JAX_PLATFORM_NAME",     "cpu")
        _os.environ.setdefault("JAX_PLATFORMS",         "cpu")
    elif force_cpu:
        # If user explicitly requested no GPUs or if CPU-only is requested,
        # we force JAX to CPU to avoid CUDA initialization errors.
        _os.environ.setdefault("JAX_PLATFORMS",         "cpu")
        _os.environ.setdefault("JAX_PLATFORM_NAME",     "cpu")


_apply_master_backend_env_defaults()

# -----------------------------------------------------------------------------
# Global singletons

# Centralized globals (lazy singletons)
from .qes_globals       import get_backend_manager, get_logger, get_numpy_rng, reseed_all
from .qes_globals       import next_jax_key as _next_jax_key
from .qes_globals       import split_jax_keys as _split_jax_keys

# -----------------------------------------------------------------------------
# Registry helpers

# Lightweight registry utilities
from .registry import describe_module, list_modules

# -----------------------------------------------------------------------------
# Stable compatibility wrappers


def qes_reseed(seed: int) -> None:
    """Reseed the global backend manager RNGs.

    Parameters
    ----------
    seed : int
        Deterministic seed applied through the global backend manager.

    Returns
    -------
    None
        Thin wrapper over :func:`reseed_all` for backward compatibility.
    """
    reseed_all(seed)

def qes_next_key() -> _t.Any:
    """Return a fresh JAX PRNG subkey.

    Returns
    -------
    Any
        Backend-specific PRNG key object when the JAX path is active.
    """
    return _next_jax_key()

def qes_split_keys(n: int) -> _t.Any:
    """Split the current JAX PRNG key into ``n`` subkeys.

    Parameters
    ----------
    n : int
        Number of subkeys to generate.

    Returns
    -------
    Any
        Backend-specific key container with ``n`` fresh subkeys.
    """
    return _split_jax_keys(n)

@_contextmanager
def qes_seed_scope(
    seed: int, *, touch_numpy_global: bool = False, touch_python_random: bool = False
):
    """Temporarily install a deterministic seed across supported backends.

    Parameters
    ----------
    seed : int
        Seed applied for the scope lifetime.
    touch_numpy_global : bool, optional
        If True, also reseed the process-global NumPy RNG.
    touch_python_random : bool, optional
        If True, also reseed the standard-library ``random`` module.

    Yields
    ------
    Any
        Backend-manager seed-suite payload returned by ``seed_scope``.
    """
    backend_mgr = get_backend_manager()
    with backend_mgr.seed_scope(
        seed, touch_numpy_global=touch_numpy_global, touch_python_random=touch_python_random
    ) as suite:
        yield suite


# ----------------------------------------------------------------------------
# Lazy Import Configuration
# ----------------------------------------------------------------------------

# Mapping of attribute names to (module_relative_path, attribute_name_in_module)
# If attribute_name_in_module is None, the module itself is imported.
_STABLE_LAZY_IMPORTS = {
    # Session Management
    'QESSession'        : ('.session',          'QESSession'),
    'run'               : ('.session',          'run'),

    # Top-level packages
    "Algebra"                   : (".Algebra", None),
    "NQS"                       : (".NQS", None),
    "Solver"                    : (".Solver", None),
    # Core classes from QES.Algebra
    "HilbertSpace"              : (".Algebra.hilbert", "HilbertSpace"),
    "Hamiltonian"               : (".Algebra.hamil", "Hamiltonian"),
    "Operator"                  : (".Algebra.Operator.operator", "Operator"),
    # Core classes from QES.Solver
    "MonteCarloSolver"          : (".Solver.MonteCarlo.montecarlo", "MonteCarloSolver"),
    "Sampler"                   : (".Solver.MonteCarlo.sampler", "Sampler"),
    # Networks from QES.general_python.ml
    "RBM"                       : (".general_python.ml.net_impl.networks.net_rbm", "RBM"),
    "CNN"                       : (".general_python.ml.net_impl.networks.net_cnn", "CNN"),
    "ResNet"                    : (".general_python.ml.net_impl.networks.net_res", "ResNet"),
    "Autoregressive"            : (".general_python.ml.net_impl.networks.net_autoregressive", "Autoregressive"),
    "SimpleNet"                 : (".general_python.ml.net_impl.net_simple", "SimpleNet"),
    "choose_network"            : (".general_python.ml.networks", "choose_network"),
    # Common Utilities (Lazy & Flattened)
    "log_memory_status"         : (".general_python.common", "log_memory_status"),
    "check_memory_for_operation": (".general_python.common", "check_memory_for_operation"),
    "Timer"                     : (".general_python.common", "Timer"),
}

_LEGACY_LAZY_IMPORTS = {
    # general_python compatibility aliases
    "gp_ml"                     : (".general_python.ml", None),
    "gp_algebra"                : (".general_python.algebra", None),
    "gp_common"                 : (".general_python.common", None),
    "gp_lattices"               : (".general_python.lattices", None),
    "gp_maths"                  : (".general_python.maths", None),
    "gp_physics"                : (".general_python.physics", None),
    # Deeper aliases
    "gp_networks"               : (".general_python.ml.networks", None),
    "gp_flog"                   : (".general_python.common.flog", None),
    # Historical compatibility alias
    "NQS_Model"                 : (".NQS.nqs", "NQS"),
}

_LAZY_IMPORTS           = {**_STABLE_LAZY_IMPORTS, **_LEGACY_LAZY_IMPORTS}
_LEGACY_IMPORT_NAMES    = frozenset(_LEGACY_LAZY_IMPORTS)

# Cache for lazily loaded modules/attributes
_LAZY_CACHE             = {}

if TYPE_CHECKING:
    from .session import QESSession, run

    from . import Algebra
    from . import NQS
    from . import Solver
    from . import general_python

    # Core
    from .Algebra.hilbert           import HilbertSpace
    from .Algebra.Operator.operator import Operator
    from .general_python            import algebra  as gp_algebra
    from .general_python            import common   as gp_common
    from .general_python            import lattices as gp_lattices
    from .general_python            import maths    as gp_maths

    # Aliases
    from .general_python import ml as gp_ml
    from .general_python import physics as gp_physics

    # Utilities
    from .general_python.common.memory import check_memory_for_operation, log_memory_status
    from .general_python.common.timer import Timer
    from .general_python.ml.net_impl.net_simple import SimpleNet
    from .general_python.ml.net_impl.networks.net_autoregressive import Autoregressive
    from .general_python.ml.net_impl.networks.net_cnn import CNN

    # Networks
    from .general_python.ml.net_impl.networks.net_rbm import RBM
    from .general_python.ml.net_impl.networks.net_res import ResNet
    from .general_python.ml.networks import choose_network

    # Solver
    from .Solver.MonteCarlo.montecarlo import MonteCarloSolver
    from .Solver.MonteCarlo.sampler import Sampler


def _lazy_import(name: str):
    """Resolve a lazily exported module or attribute.

    Parameters
    ----------
    name : str
        Public attribute name requested from the top-level ``QES`` package.

    Returns
    -------
    Any
        Imported module object or attribute value.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name]

    try:
        if name in _LEGACY_IMPORT_NAMES:
            _warnings.warn(
                f"QES.{name} is a legacy compatibility alias and may be removed from the top-level API in a future cleanup. "
                "Import from the concrete submodule instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        # Import the module
        module = _importlib.import_module(module_path, package=__name__)

        # If attr_name is None, we want the module itself
        if attr_name is None:
            result = module
        else:
            result = getattr(module, attr_name)

        _LAZY_CACHE[name] = result
        return result
    except ImportError as e:
        raise ImportError(f"Failed to import lazy module '{name}' from '{module_path}': {e}") from e

def __getattr__(name: str) -> _t.Any:  # PEP 562
    """Resolve lazy top-level attributes on demand."""
    return _lazy_import(name)

def __dir__():
    """Return the stable top-level package attribute list."""
    return sorted(set(__all__))

def next_jax_key() -> _t.Any:
    """Return a fresh JAX PRNG subkey.

    Deprecated top-level compatibility alias for :func:`qes_next_key`.
    """
    _warnings.warn(
        "QES.next_jax_key is deprecated. Use QES.qes_next_key instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return qes_next_key()

def split_jax_keys(n: int) -> _t.Any:
    """Split the current JAX PRNG key into ``n`` subkeys.

    Deprecated top-level compatibility alias for :func:`qes_split_keys`.
    """
    _warnings.warn(
        "QES.split_jax_keys is deprecated. Use QES.qes_split_keys instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return qes_split_keys(n)

__all__ = [
    # Session
    "QESSession",
    "run",
    # RNG / backend helpers (public stable API surface)
    "qes_reseed",
    "qes_next_key",
    "qes_split_keys",
    "qes_seed_scope",
    # Discovery utilities
    "list_modules",
    "describe_module",
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
] + list(_LAZY_IMPORTS.keys())

# -----------------------------------------------------------------------------
# End of package initialization
