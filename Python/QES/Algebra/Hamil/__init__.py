"""
Hamiltonian Module
==================

This module contains Hamiltonian construction and energy calculation utilities.

Modules:
--------
- hamil_energy: Energy calculation methods for Hamiltonians
- hamil_energy_helper: Helper functions for energy calculations
- hamil_energy_jax: JAX-based energy calculation implementations
- hamil_jit_methods: JIT-compiled methods for Hamiltonian operations
- hamil_types: Type definitions and utilities for Hamiltonians

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hamil_energy import (
        local_energy_np_wrap,
        process_mod_nosites,
        process_mod_sites,
        process_nmod_nosites,
        process_nmod_sites,
    )
    from .hamil_energy_helper import (
        default_operator,
        default_operator_njit,
        flatten_operator_terms,
        unpack_operator_terms,
    )
    from .hamil_jit_methods import energy_width, gap_ratio, mean_level_spacing
    from .hamil_types import (
        DummyVector,
        Hamiltonians,
        check_dense,
        check_noninteracting,
    )

# ---------------------------------------------------------------------------
# Lazy loading infrastructure
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # hamil_energy
    "process_mod_sites": (".hamil_energy", "process_mod_sites"),
    "process_mod_nosites": (".hamil_energy", "process_mod_nosites"),
    "process_nmod_sites": (".hamil_energy", "process_nmod_sites"),
    "process_nmod_nosites": (".hamil_energy", "process_nmod_nosites"),
    "local_energy_np_wrap": (".hamil_energy", "local_energy_np_wrap"),
    # hamil_energy_helper
    "default_operator": (".hamil_energy_helper", "default_operator"),
    "default_operator_njit": (".hamil_energy_helper", "default_operator_njit"),
    "flatten_operator_terms": (".hamil_energy_helper", "flatten_operator_terms"),
    "unpack_operator_terms": (".hamil_energy_helper", "unpack_operator_terms"),
    # hamil_jit_methods
    "mean_level_spacing": (".hamil_jit_methods", "mean_level_spacing"),
    "energy_width": (".hamil_jit_methods", "energy_width"),
    "gap_ratio": (".hamil_jit_methods", "gap_ratio"),
    # hamil_types
    "check_noninteracting": (".hamil_types", "check_noninteracting"),
    "check_dense": (".hamil_types", "check_dense"),
    "Hamiltonians": (".hamil_types", "Hamiltonians"),
    "DummyVector": (".hamil_types", "DummyVector"),
}

# Submodule descriptions for help()
_SUBMODULES = {
    "hamil_energy": "Energy calculation methods for Hamiltonians (local energy)",
    "hamil_energy_helper": "Helper functions for flattening/unpacking terms",
    "hamil_jit_methods": "JIT-compiled analysis (level spacing, energy width)",
    "hamil_types": "Hamiltonian enums and type checks",
}

def __getattr__(name: str) -> Any:
    """Lazily import functions/classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


def help(verbose: bool = True) -> None:
    """
    Print help information about the Hamiltonian module.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         QES Hamiltonian Module                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Construction and energy evaluation for quantum Hamiltonians.                ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    print("Available Submodules:")
    print("-" * 60)
    for name, desc in _SUBMODULES.items():
        print(f"  {name:32s} : {desc}")

    if verbose:
        print("\nKey Functions:")
        print("-" * 60)
        print("  - hamil_energy.local_energy_np_wrap  : Local energy estimator")
        print("  - hamil_jit_methods.mean_level_spacing : Spectral statistics")
        print("  - hamil_types.Hamiltonians           : Supported model types")


__all__ = [
    # Base
    "help",
    # hamil_energy
    "process_mod_sites",
    "process_mod_nosites",
    "process_nmod_sites",
    "process_nmod_nosites",
    "local_energy_np_wrap",
    # hamil_energy_helper
    "default_operator",
    "default_operator_njit",
    "flatten_operator_terms",
    "unpack_operator_terms",
    # hamil_jit_methods
    "mean_level_spacing",
    "energy_width",
    "gap_ratio",
    # hamil_types
    "check_noninteracting",
    "check_dense",
    "Hamiltonians",
    "DummyVector",
]
