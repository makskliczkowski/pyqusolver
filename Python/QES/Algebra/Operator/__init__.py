"""
QES Operator Module
===================

This module provides quantum mechanical operators for various particle types.

Modules:
--------
- operator: Base operator classes and utilities
- operator_matrix: Matrix representations of operators
- operators_spin: Spin operators (Pauli matrices, etc.)
- operators_spin_jax: JAX implementations of spin operators
- operators_spinless_fermions: Fermionic operators for spinless particles
- operators_spinless_fermions_jax: JAX implementations of fermionic operators

Classes:
--------
- Operator: Base class for quantum operators
- SymmetryGenerators: Symmetry operation generators

File    : QES/Algebra/Operator/__init__.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = "Operator classes and concrete operators (spin, fermions) with matrix builders."

try:
    from .catalog import OPERATOR_CATALOG, register_local_operator
    from .operator import Operator, SymmetryGenerators, operator_from_local
    # from .operator_matrix import *
    from .operators_spin import *
    from .operators_spinless_fermions import *
    from .operators_anyon import *
    from .phase_utils import (
        bit_popcount,
        bit_popcount_mask,
        fermionic_parity_int,
        fermionic_parity_array,
    )

    __all__ = [
        'OPERATOR_CATALOG',
        'Operator',
        'SymmetryGenerators',
        'operator_from_local',
        'register_local_operator',
        # Operator matrix builders
        'sig_plus',
        'sig_minus',
        # Phase utilities and functions
        'bit_popcount',
        'bit_popcount_mask',
        'fermionic_parity_int',
        'fermionic_parity_array',
    ]
except ImportError:
    __all__ = []

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
