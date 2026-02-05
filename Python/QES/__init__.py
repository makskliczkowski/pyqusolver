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

Architecture & Invariants
-------------------------
1.  **Backend Agnosticism**: QES supports both NumPy (CPU) and JAX (GPU/TPU) backends.
    The active backend is managed globally via `QES.get_backend_manager()`.
    All heavy numerical operations delegate to the active backend.
2.  **Global State**: Random number generators and logging are managed globally.
    Use `QES.qes_reseed(seed)` to ensure reproducibility across all components.
3.  **Lazy Loading**: Submodules are lazily loaded to minimize startup time.
    Accessing `QES.NQS` or `QES.Solver` triggers their import.

Modules
-------
-   **Algebra**: Core quantum mechanics (Hilbert spaces, Operators, Hamiltonians).
-   **NQS**: Neural Quantum States and Variational Monte Carlo.
-   **Solver**: Simulation engines (VMC, ED).
-   **general_python**: Shared utilities and neural network implementations.

----------------------------------------------------------
Author          : Maks Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.10.2025
Description     : Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving.
----------------------------------------------------------
"""

__version__ = "0.1.0"
__author__ = "Maksymilian Kliczkowski"
__email__ = "maksymilian.kliczkowski@pwr.edu.pl"
__license__ = "CC-BY-4.0"
__description__ = (
    "Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving"
)

import importlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional

# Centralized globals (lazy singletons)
from .qes_globals import (
    get_backend_manager,
    get_logger,
    get_numpy_rng,
    reseed_all,
)
from .qes_globals import (
    next_jax_key as _next_jax_key,
)
from .qes_globals import (
    split_jax_keys as _split_jax_keys,
)

# Lightweight registry utilities
from .registry import describe_module, list_modules

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
def qes_seed_scope(
    seed: int, *, touch_numpy_global: bool = False, touch_python_random: bool = False
):
    """Context manager to temporarily set a deterministic seed across backends.

    Delegates to the backend manager's ``seed_scope``.
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
_LAZY_IMPORTS = {
    # Session Management
    'QESSession'        : ('.session',          'QESSession'),
    'run'               : ('.session',          'run'),

    # Top-level packages
    "Algebra": (".Algebra", None),
    "NQS": (".NQS", None),
    "Solver": (".Solver", None),
    # Core classes from QES.Algebra
    "HilbertSpace": (".Algebra.hilbert", "HilbertSpace"),
    "Hamiltonian": (".Algebra.hamil", "Hamiltonian"),
    "Operator": (".Algebra.Operator.operator", "Operator"),
    # Core classes from QES.NQS
    "NQS_Model": (".NQS.nqs", "NQS"),
    # Core classes from QES.Solver
    "MonteCarloSolver": (".Solver.MonteCarlo.montecarlo", "MonteCarloSolver"),
    "Sampler": (".Solver.MonteCarlo.sampler", "Sampler"),
    # Networks from QES.general_python.ml
    "RBM": (".general_python.ml.net_impl.networks.net_rbm", "RBM"),
    "CNN": (".general_python.ml.net_impl.networks.net_cnn", "CNN"),
    "ResNet": (".general_python.ml.net_impl.networks.net_res", "ResNet"),
    "Autoregressive": (".general_python.ml.net_impl.networks.net_autoregressive", "Autoregressive"),
    "SimpleNet": (".general_python.ml.net_impl.net_simple", "SimpleNet"),
    "choose_network": (".general_python.ml.networks", "choose_network"),
    # Common Utilities (Lazy & Flattened)
    "log_memory_status": (".general_python.common", "log_memory_status"),
    "check_memory_for_operation": (".general_python.common", "check_memory_for_operation"),
    "Timer": (".general_python.common", "Timer"),
    # Aliases for general_python modules (convenience)
    "gp_ml": (".general_python.ml", None),
    "gp_algebra": (".general_python.algebra", None),
    "gp_common": (".general_python.common", None),
    "gp_lattices": (".general_python.lattices", None),
    "gp_maths": (".general_python.maths", None),
    "gp_physics": (".general_python.physics", None),
    # Deeper aliases
    "gp_networks": (".general_python.ml.networks", None),
    "gp_flog": (".general_python.common.flog", None),
}

# Cache for lazily loaded modules/attributes
_LAZY_CACHE = {}

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
    """
    Lazily import a module or attribute based on _LAZY_IMPORTS configuration.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name]

    try:
        # Import the module
        module = importlib.import_module(module_path, package=__name__)

        # If attr_name is None, we want the module itself
        if attr_name is None:
            result = module
        else:
            result = getattr(module, attr_name)

        _LAZY_CACHE[name] = result
        return result
    except ImportError as e:
        raise ImportError(f"Failed to import lazy module '{name}' from '{module_path}': {e}") from e


def __getattr__(name: str) -> Any:  # PEP 562
    return _lazy_import(name)


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


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

# -------------------------------------------------------------------------------------------------
#! End of QES package initialization
# -------------------------------------------------------------------------------------------------
