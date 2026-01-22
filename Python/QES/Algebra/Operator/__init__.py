r"""
QES Operator Module
===================

This module provides quantum mechanical operators for various particle types,
including spin systems, spinless fermions, hardcore bosons, and anyons.

**Lazy Loading**: Submodules are loaded on-demand to minimize import time.
Use ``hamil.operators`` for automatic operator selection based on Hilbert space,
or import specific modules directly when needed.

Submodules
----------
operator : Base classes
    Core ``Operator`` class and ``SymmetryGenerators`` for building quantum operators.

operators_spin : Spin-1/2 operators
    Pauli matrices (sigma_x, sigma_y, sigma_z), raising/lowering (sigma_+, sigma_-), and spin correlators.
    Supports local, global, and correlation operator types.

operators_spin_1 : Spin-1 operators
    Spin-1 matrices (S_x, S_y, S_z), raising/lowering (S_+, S_-), and correlators.
    3-dimensional local Hilbert space with states |+1⟩, |0⟩, |-1⟩.
    Includes quadrupole operators for higher-order observables.

operators_spinless_fermions : Fermionic operators
    Creation (c†), annihilation (c), number (n), and hopping operators.
    Handles fermionic sign (Jordan-Wigner) automatically.

operators_hardcore : Hardcore boson operators
    Bosonic operators with hard-core constraint (max 1 per site).

operators_anyon : Anyonic operators
    Operators for systems with fractional statistics.

jax/ : JAX-accelerated operators subpackage
    GPU-accelerated versions using JAX. Contains:
    - jax.operators_spin                : Spin-1/2 operators
    - jax.operators_spin_1              : Spin-1 operators
    - jax.operators_spinless_fermions   : Spinless fermion operators

catalog : Operator catalog
    Registry of available operators with metadata.

phase_utils : Phase utilities
    Fermionic parity, bit counting, and sign calculation utilities.

Examples
--------
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
    >>>
    >>> # Apply to states
    >>> new_state = sig_z.matvec(psi, site=0, hilbert=hamil.hilbert_space)

**Direct import for spin operators**::

    >>> from QES.Algebra.Operator.impl.operators_spin import sig_x, sig_y, sig_z
    >>> from QES.Algebra.Operator.impl.operators_spin import sig_plus, sig_minus
    >>>
    >>> # Create spin-z operator for all sites
    >>> Sz_total = sig_z(lattice=lattice, type_act='global')
    >>>
    >>> # Create correlation operator for sites i,j
    >>> Sz_Sz = sig_z(lattice=lattice, type_act='correlation')

**Direct import for fermionic operators**::

    >>> from QES.Algebra.Operator.impl.operators_spinless_fermions import cdag, c, n_op
    >>>
    >>> # Number operator
    >>> n = n_op(ns=10, sites=[0, 1, 2])
    >>>
    >>> # Hopping: c†_i c_j
    >>> hopping = cdag(ns=10, sites=[0]) @ c(ns=10, sites=[1])

**Operator types**::

    >>> # Local: acts on single site, returns modified state
    >>> op_local = sig_z(lattice=lattice, type_act='local')
    >>> result = op_local.matvec(psi, site=3, hilbert=hilbert)
    >>>
    >>> # Global: sum over all sites (e.g., total magnetization)
    >>> op_global = sig_z(lattice=lattice, type_act='global')
    >>> Sz_total = op_global.matrix  # Full matrix representation
    >>>
    >>> # Correlation: two-point correlator <sigma__i sigma__j>
    >>> op_corr = sig_z(lattice=lattice, type_act='correlation')
    >>> corr_ij = op_corr.expectation(psi, i=0, j=5)

**Building custom operators**::

    >>> from QES.Algebra.Operator import Operator
    >>>
    >>> # Compose operators
    >>> Sx_Sy = sig_x(lattice) @ sig_y(lattice)  # Product
    >>> H_field = 0.5 * sig_z(lattice)  # Scalar multiplication

See Also
--------
QES.Algebra.Operator.operator                       : Base Operator class documentation
QES.Algebra.Operator.impl.operators_spin                 : Spin operator documentation
QES.Algebra.Operator.impl.operators_spinless_fermions    : Fermion operator documentation

-----------------------------------------------------------
File    : QES/Algebra/Operator/__init__.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
-----------------------------------------------------------
"""

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = (
    "Operator classes and concrete operators (spin, fermions) with matrix builders."
)

# ---------------------------------------------------------------------------
# Lazy loading infrastructure
# ---------------------------------------------------------------------------

# Only import lightweight base classes at module load time
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
    "jax.operators_spin": "JAX-accelerated spin-1/2 operators",
    "jax.operators_spin_1": "JAX-accelerated spin-1 operators",
    "jax.operators_spinless_fermions": "JAX-accelerated fermion operators",
    # helpers
    "catalog": "Operator catalog and registry",
    "phase_utils": "Fermionic parity and sign utilities",
    "special_operator": "Special/custom operator support",
    "matrix": "Matrix representation utilities",
}

# Operator types available in each submodule
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
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         QES Operator Module                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quantum mechanical operators for spin, fermion, and other systems.          ║
║                                                                              ║
║  RECOMMENDED: Use model.operators for automatic operator selection:          ║
║      >>> ops = model.operators                                               ║
║      >>> Sz = ops.sig_z(lattice=lattice, type_act='local')                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

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
