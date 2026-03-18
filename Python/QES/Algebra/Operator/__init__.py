"""
Operator Module
===============

Operator classes and concrete operators (spin, fermions) with matrix builders.

This module provides the core `Operator` class and a collection of concrete operator implementations
for various physical systems (spin-1/2, spin-1, fermions, etc.).

Usage
-----
**Via Hamiltonian (recommended - automatic operator selection)**::

    >>> from QES.Algebra.Model.Interacting.Spin import HeisenbergXXZ
    >>> hamil = HeisenbergXXZ(lattice=lattice, J=1.0, delta=0.5)
    >>>
    >>> # Lazy-loaded operators based on Hilbert space type
    >>> ops = hamil.operators
    >>>
    >>> # Create local operators
    >>> sig_z = ops.sig_z(lattice=lattice, type_act='local')
    >>> sig_x = ops.sig_x(lattice=lattice, type_act='local')

**Direct import for spin operators**::

    >>> from QES.Algebra.Operator import operators_spin
    >>> from QES.Algebra.Operator.impl.operators_spin import sig_z
    >>>
    >>> # Create spin-z operator for all sites
    >>> Sz_total = operators_spin.sig_z(lattice=lattice, type_act='global')

-----------------------------------------------------------
File    : QES/Algebra/Operator/__init__.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
-----------------------------------------------------------
"""

import importlib
from typing import TYPE_CHECKING, Any

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = (
    "Operator classes and concrete operators (spin, fermions) with matrix builders."
)

# ---------------------------------------------------------------------------
# Lazy loading infrastructure
# ---------------------------------------------------------------------------

# Only import lightweight base classes at module load time
from .operator import Operator, SymmetryGenerators

if TYPE_CHECKING:
    from . import catalog, matrix, phase_utils, special_operator
    from .impl import (
        jax,
        operators_anyon,
        operators_hardcore,
        operators_spin,
        operators_spin_1,
        operators_spinless_fermions,
    )
    from .operator import Operator, SymmetryGenerators

# Available submodules (loaded on demand)
_SUBMODULES = {
    "operator": "Base Operator class and SymmetryGenerators",
    # concrete operator submodules
    "operators_spin": "Spin-1/2 operators (Pauli matrices)",
    "operators_spin_1": "Spin-1 operators (S=1 matrices)",
    "operators_spinless_fermions": "Spinless fermion operators (c, c†, n)",
    "operators_hardcore": "Hardcore boson operators",
    "operators_anyon": "Anyonic operators",
    # JAX backends (in jax/ subfolder)
    "jax": "JAX-accelerated operators subpackage",
    # helpers
    "catalog": "Operator catalog and registry",
    "phase_utils": "Fermionic parity and sign utilities",
    "special_operator": "Special/custom operator support",
    "matrix": "Matrix representation utilities",
}

# Map public names to (module_path, attribute_name)
# If attribute_name is None, return the module itself.
_LAZY_IMPORTS = {
    "operator": (".operator", None),
    "operators_spin": (".impl.operators_spin", None),
    "operators_spin_1": (".impl.operators_spin_1", None),
    "operators_spinless_fermions": (".impl.operators_spinless_fermions", None),
    "operators_hardcore": (".impl.operators_hardcore", None),
    "operators_anyon": (".impl.operators_anyon", None),
    "jax": (".impl.jax", None),
    "phase_utils": (".phase_utils", None),
    "catalog": (".catalog", None),
    "special_operator": (".special_operator", None),
    "matrix": (".matrix", None),
}

# Operator types available in each submodule (for help/introspection)
_OPERATOR_TYPES = {
    "operators_spin": [
        "sig_x",
        "sig_y",
        "sig_z",  # Pauli matrices
        "sig_plus",
        "sig_minus",  # Raising/lowering
        "make_sigma_mixed",  # Mixed correlators (e.g., sigma_x_i sigma_y_j)
    ],
    "operators_spin_1": [
        "s1_x",
        "s1_y",
        "s1_z",  # Spin-1 matrices
        "s1_plus",
        "s1_minus",  # Raising/lowering (S_+, S_-)
        "s1_squared",  # S^2 operator
        "s1_xy",
        "s1_yx",
        "s1_xz",  # Two-site correlators
        "s1_zx",
        "s1_yz",
        "s1_zy",  # Two-site correlators
        "s1_quadrupole",  # Quadrupole operators (Q_zz, etc.)
    ],
    "operators_spinless_fermions": [
        "c",
        "cdag",  # Annihilation/creation
        "n_op",  # Number operator
        "hopping",  # Hopping term
    ],
    "operators_hardcore": [
        "b",
        "b_dag",
        "n_hc",  # Hardcore boson operators
    ],
    "phase_utils": [
        "bit_popcount",  # Count set bits
        "bit_popcount_mask",  # Masked bit count
        "fermionic_parity_int",  # Fermionic sign for integer state
        "fermionic_parity_array",  # Fermionic sign for array state
    ],
}

__all__ = [
    # Base classes (always available)
    "Operator",
    "SymmetryGenerators",
    # Helper functions
    "help",
    "list_operators",
    "list_submodules",
    # Submodule names (for explicit import)
    "operator",
    "operators_spin",
    "operators_spin_1",
    "operators_spinless_fermions",
    "operators_hardcore",
    "operators_anyon",
    # JAX subpackage
    "jax",
    "phase_utils",
    "catalog",
    "special_operator",
    "matrix",
]


def __getattr__(name: str) -> Any:
    """Lazily import submodules."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        if attr_name:
            return getattr(module, attr_name)
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


def help(verbose: bool = True) -> None:
    """
    Print help information about the Operator module.

    Parameters
    ----------
    verbose : bool
        If True, print detailed information including all operators.
        If False, print only submodule summary.

    Examples
    --------
    >>> from QES.Algebra import Operator
    >>> Operator.help()
    """
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         QES Operator Module                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quantum mechanical operators for spin, fermion, and other systems.          ║
║                                                                              ║
║  RECOMMENDED: Use model.operators for automatic operator selection:          ║
║      >>> ops = model.operators                                               ║
║      >>> Sz = ops.sig_z(lattice=lattice, type_act='local')                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )

    print("Available Submodules:")
    print("-" * 60)
    for name, desc in _SUBMODULES.items():
        print(f"  {name:32s} : {desc}")

    if verbose:
        print("\nOperator Types by Submodule:")
        print("-" * 60)
        for module, ops in _OPERATOR_TYPES.items():
            print(f"\n  {module}:")
            for op in ops:
                print(f"    - {op}")

        print("\n" + "=" * 60)
        print("Quick Start:")
        print("-" * 60)
        print("""
# Via Model (e.g., Hamiltonian) (recommended):
>>> ops = model.operators
>>> Sz = ops.sig_z(lattice=lattice, type_act='local')

# Direct import:
>>> from QES.Algebra.Operator.impl.operators_spin import sig_z
>>> Sz = sig_z(lattice=lattice, type_act='global')

# Operator types:
>>> type_act='local'       # Single-site operator
>>> type_act='global'      # Sum over all sites  
>>> type_act='correlation' # Two-point correlator
""")


def list_submodules() -> dict:
    """
    Return dictionary of available submodules and their descriptions.

    Returns
    -------
    dict
        Mapping of submodule names to descriptions.

    Examples
    --------
    >>> from QES.Algebra import Operator
    >>> Operator.list_submodules()
    {'operators_spin': 'Spin-1/2 operators (Pauli matrices)', ...}
    """
    return _SUBMODULES.copy()


def list_operators(submodule: str = None) -> dict:
    """
    List available operators, optionally filtered by submodule.

    Parameters
    ----------
    submodule : str, optional
        If provided, only list operators from this submodule.
        If None, list all operators from all submodules.

    Returns
    -------
    dict
        Mapping of submodule names to lists of operator names.

    Examples
    --------
    >>> from QES.Algebra import Operator
    >>> Operator.list_operators('operators_spin')
    ['sig_x', 'sig_y', 'sig_z', 'sig_plus', 'sig_minus', 'make_sigma_mixed']
    """
    if submodule is not None:
        if submodule in _OPERATOR_TYPES:
            return {submodule: _OPERATOR_TYPES[submodule]}
        else:
            return {}
    return _OPERATOR_TYPES.copy()


# ---------------------------------------------------------------------------
#! EOF
# ---------------------------------------------------------------------------
