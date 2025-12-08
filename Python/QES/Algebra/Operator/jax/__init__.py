r"""
QES Operator JAX Module
=======================

This subpackage contains JAX-accelerated implementations of quantum operators.
All functions are JIT-compiled for high performance on GPUs/TPUs.

Submodules
----------
operators_spin : JAX-accelerated spin-1/2 operators
    Pauli matrices (sigma_x, sigma_y, sigma_z) and raising/lowering operators
    with JAX JIT compilation for efficient computation.

operators_spin_1 : JAX-accelerated spin-1 operators
    Spin-1 matrices (S_x, S_y, S_z) and ladder operators with 3-dimensional
    local Hilbert space.

operators_spinless_fermions : JAX-accelerated spinless fermion operators
    Creation/annihilation operators with Jordan-Wigner transformation,
    including momentum-space versions.

Usage
-----
These modules are typically imported automatically by the parent operator modules
when JAX is available. For direct access::

    from QES.Algebra.Operator.jax.operators_spin import sigma_x_jnp, sigma_z_jnp
    from QES.Algebra.Operator.jax.operators_spin_1 import spin1_z_jnp
    from QES.Algebra.Operator.jax.operators_spinless_fermions import c_jnp, c_dag_jnp

Requirements
------------
- JAX must be installed (``pip install jax jaxlib``)
- GPU support requires appropriate JAX GPU installation

Notes
-----
If JAX is not available, all operator functions will be set to ``None``.

--------------------------------------------------------------
File        : QES/Algebra/Operator/jax/__init__.py
Author      : Maksymilian Kliczkowski
Date        : December 2025
--------------------------------------------------------------
"""

from QES.general_python.common.binary import JAX_AVAILABLE

# Lazy loading - only import when JAX is available
if JAX_AVAILABLE:
    # Spin-1/2 operators
    from .operators_spin import (
        # Matrices
        _SIG_0_jnp, _SIG_X_jnp, _SIG_Y_jnp, _SIG_Z_jnp, _SIG_P_jnp, _SIG_M_jnp,
        # Integer state operators
        sigma_x_int_jnp, sigma_y_int_jnp, sigma_z_int_jnp,
        sigma_plus_int_jnp, sigma_minus_int_jnp,
        sigma_pm_int_jnp, sigma_mp_int_jnp,
        sigma_k_int_jnp, sigma_z_total_int_jnp,
        # Array state operators
        sigma_x_jnp, sigma_x_inv_jnp,
        sigma_y_jnp, sigma_y_real_jnp, sigma_y_inv_jnp,
        sigma_z_jnp, sigma_z_inv_jnp,
        sigma_plus_jnp, sigma_minus_jnp,
        sigma_pm_jnp, sigma_mp_jnp,
        sigma_k_jnp, sigma_k_inv_jnp,
        sigma_z_total_jnp,
    )
    
    # Spin-1 operators
    from .operators_spin_1 import (
        # Matrices
        _S1_X_jnp, _S1_Y_jnp, _S1_Z_jnp, _S1_PLUS_jnp, _S1_MINUS_jnp, _S1_Z2_jnp,
        # Integer state operators
        spin1_z_int_jnp, spin1_plus_int_jnp, spin1_minus_int_jnp,
        spin1_x_int_jnp, spin1_y_int_jnp,
        spin1_pm_int_jnp, spin1_mp_int_jnp, spin1_zz_int_jnp,
        spin1_squared_int_jnp, spin1_z_total_int_jnp, spin1_z2_int_jnp,
        # Array state operators
        spin1_z_jnp, spin1_z_inv_jnp,
        spin1_plus_jnp, spin1_minus_jnp,
        spin1_z_total_jnp,
    )
    
    # Spinless fermion operators
    from .operators_spinless_fermions import (
        # Parity functions
        f_parity_int_jnp, f_parity_np_jnp,
        # Integer state operators
        c_int_jnp, c_dag_int_jnp,
        c_k_int_jnp, c_k_dag_int_jnp,
        n_int_jax,
        # Array state operators
        c_jnp, c_dag_jnp,
        c_k_jnp, c_k_dag_jnp,
        n_jax,
    )

__all__ = [
    'JAX_AVAILABLE',
    # Submodules
    'operators_spin',
    'operators_spin_1', 
    'operators_spinless_fermions',
]

# Add exports only if JAX is available
if JAX_AVAILABLE:
    __all__.extend([
        # Spin-1/2
        '_SIG_0_jnp', '_SIG_X_jnp', '_SIG_Y_jnp', '_SIG_Z_jnp', '_SIG_P_jnp', '_SIG_M_jnp',
        'sigma_x_int_jnp', 'sigma_y_int_jnp', 'sigma_z_int_jnp',
        'sigma_plus_int_jnp', 'sigma_minus_int_jnp',
        'sigma_pm_int_jnp', 'sigma_mp_int_jnp',
        'sigma_k_int_jnp', 'sigma_z_total_int_jnp',
        'sigma_x_jnp', 'sigma_x_inv_jnp',
        'sigma_y_jnp', 'sigma_y_real_jnp', 'sigma_y_inv_jnp',
        'sigma_z_jnp', 'sigma_z_inv_jnp',
        'sigma_plus_jnp', 'sigma_minus_jnp',
        'sigma_pm_jnp', 'sigma_mp_jnp',
        'sigma_k_jnp', 'sigma_k_inv_jnp',
        'sigma_z_total_jnp',
        # Spin-1
        '_S1_X_jnp', '_S1_Y_jnp', '_S1_Z_jnp', '_S1_PLUS_jnp', '_S1_MINUS_jnp', '_S1_Z2_jnp',
        'spin1_z_int_jnp', 'spin1_plus_int_jnp', 'spin1_minus_int_jnp',
        'spin1_x_int_jnp', 'spin1_y_int_jnp',
        'spin1_pm_int_jnp', 'spin1_mp_int_jnp', 'spin1_zz_int_jnp',
        'spin1_squared_int_jnp', 'spin1_z_total_int_jnp', 'spin1_z2_int_jnp',
        'spin1_z_jnp', 'spin1_z_inv_jnp',
        'spin1_plus_jnp', 'spin1_minus_jnp',
        'spin1_z_total_jnp',
        # Fermions
        'f_parity_int_jnp', 'f_parity_np_jnp',
        'c_int_jnp', 'c_dag_int_jnp',
        'c_k_int_jnp', 'c_k_dag_int_jnp',
        'n_int_jax',
        'c_jnp', 'c_dag_jnp',
        'c_k_jnp', 'c_k_dag_jnp',
        'n_jax',
    ])
