"""
file        : Algebra/Operator/operators_spin.py

This module implements spin operators for quantum systems.
It includes functions for sigma_x, sigma_y, sigma_z, sigma_plus (raising),
sigma_minus (lowering), their products, and a Fourier-transformed sigma_k operator.
The implementation is based on the provided C++ code and uses a general Operator class.

Author      : Maksymilian Kliczkowski, WUST, Poland
Date        : February 2025
Version     : 1.0

"""

import os 
import numpy as np
import numba
from typing import List, Union, Optional, Callable, Tuple

################################################################################

try:
    # main imports
    from QES.Algebra.Operator.operator import Operator, OperatorTypeActing, create_operator, ensure_operator_output_shape_numba
    from QES.Algebra.Operator.catalog import register_local_operator
    from QES.Algebra.Hilbert.hilbert_local import LocalOpKernels, LocalSpaceTypes
except ImportError as e:
    raise ImportError("Failed to import required modules. Ensure that the QES package is correctly installed.") from e

################################################################################

try:
    import QES.general_python.common.binary as _binary
    from QES.general_python.lattices.lattice import Lattice
    from QES.general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, DEFAULT_NP_CPX_TYPE
    from QES.general_python.common.binary import BACKEND_REPR as _SPIN, BACKEND_DEF_SPIN
    from QES.general_python.common.binary import flip, check
except ImportError as e:
    raise ImportError("Failed to import required QES general Python modules.") from e

################################################################################
JAX_AVAILABLE = os.getenv("PY_JAX_AVAILABLE", "0") == "1"
################################################################################

if JAX_AVAILABLE:
    import jax
    # import Algebra.Operator.operators_spin_jax as jaxpy
    import jax.numpy as jnp
    # sigma x
    from QES.Algebra.Operator.operators_spin_jax import sigma_x_int_jnp, sigma_x_jnp, sigma_x_inv_jnp
    # sigma y
    from QES.Algebra.Operator.operators_spin_jax import sigma_y_int_jnp, sigma_y_jnp, sigma_y_real_jnp, sigma_y_inv_jnp
    # sigma z
    from QES.Algebra.Operator.operators_spin_jax import sigma_z_int_jnp, sigma_z_jnp, sigma_z_inv_jnp
    # sigma plus
    from QES.Algebra.Operator.operators_spin_jax import sigma_plus_int_jnp, sigma_plus_jnp
    # sigma minus
    from QES.Algebra.Operator.operators_spin_jax import sigma_minus_int_jnp, sigma_minus_jnp
    # sigma pm
    from QES.Algebra.Operator.operators_spin_jax import sigma_pm_int_jnp, sigma_pm_jnp
    # sigma mp
    from QES.Algebra.Operator.operators_spin_jax import sigma_mp_int_jnp, sigma_mp_jnp
    # sigma k
    from QES.Algebra.Operator.operators_spin_jax import sigma_k_int_jnp, sigma_k_jnp, sigma_k_inv_jnp
    # sigma z total
    from QES.Algebra.Operator.operators_spin_jax import sigma_z_total_int_jnp, sigma_z_total_jnp
else:
    print("JAX is not available. JAX-based implementations of spin operators will not be accessible.")
    sigma_x_int_jnp         = sigma_x_jnp       = None
    sigma_x_jnp             = None
    sigma_x_inv_jnp         = None
    sigma_y_int_jnp         = sigma_y_jnp       = None
    sigma_y_real_jnp        = None
    sigma_y_inv_jnp         = None
    sigma_z_int_jnp         = sigma_z_jnp       = None
    sigma_z_inv_jnp         = None
    sigma_plus_int_jnp      = sigma_plus_jnp    = None
    sigma_minus_int_jnp     = sigma_minus_jnp   = None
    sigma_pm_int_jnp        = sigma_pm_jnp      = None
    sigma_mp_int_jnp        = sigma_mp_jnp      = None
    sigma_k_int_jnp         = sigma_k_jnp       = None
    sigma_k_inv_jnp         = None
    sigma_z_total_int_jnp   = sigma_z_total_jnp = None
    
#! Standard Pauli matrices
################################################################################

# Define the Pauli matrices for reference
_SIG_0 = np.array([[1, 0],
                [0, 1]], dtype=float)
_SIG_X = np.array([[0, 1],
                [1, 0]], dtype=float)
_SIG_Y = np.array([[0, -1j],
                [1j, 0]], dtype=complex)
_SIG_Z = np.array([[1,  0],
                [0, -1]], dtype=float)
_SIG_P = np.array([[0, 1],
                [0, 0]], dtype=float)
_SIG_M = np.array([[0, 0],
                [1, 0]], dtype=float)

# -----------------------------------------------------------------------------
#! Sigma-X (\sigma _x) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_x_int_np(state, ns, sites, spin: bool = BACKEND_DEF_SPIN, spin_value=_SPIN):
    r"""
    Apply the Pauli-X (\sigma _x) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    coeff           = 1.0
    for site in sites:
        pos     = ns - 1 - site
        state   = _binary.flip_int(state, pos)
        coeff   *= spin_value
    out_state[0]    = state
    out_coeff[0]    = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

def sigma_x_int(state  : int,
            ns          : int,
            sites       : Union[List[int], None],
            spin_value  : float     = _SPIN):
    r"""
    Apply the Pauli-X (\sigma _x) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    Parameters
    ----------
    state : int
        The state to apply the operator to.
    ns : int
        The number of spins in the system.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    """
    return sigma_x_int_np(state, ns, sites, BACKEND_DEF_SPIN, spin_value)

@numba.njit
def sigma_x_np(state    : np.ndarray,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    r"""
    Apply the Pauli-X (\sigma _x) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator.
    """
    coeff   = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out     = state.copy()
    for site in sites:
        out     = _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
        coeff  *= spin_value
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

def sigma_x(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool      = BACKEND_DEF_SPIN,
            spin_value  : float     = _SPIN):
    r"""
    Apply the Pauli-X (\sigma _x) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    Parameters
    ----------
    state : int or np.ndarray
        The state to apply the operator to.
    ns : int
        The number of spins in the system. 
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    int or np.ndarray
        The state after applying the operator.   
    """
    if sites is None:
        sites = list(range(ns))
    
    if isinstance(state, (int, np.integer)):
        return sigma_x_int_np(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return sigma_x_np(state, sites, spin, spin_value)
    return sigma_x_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Y (\sigma _y) operator
# -----------------------------------------------------------------------------

@numba.njit(cache=True)
def sigma_y_int_np_real(state, 
                        ns, 
                        sites,
                        spin        : bool  = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN,
                        ):
    r"""
    \sigma _y on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    coeff           = 1.0 + 0j
    for site in sites:
        pos     = ns - 1 - site
        bit     = _binary.check_int(state, pos)
        coeff   *= (2 * bit - 1.0) * 1.0j * spin_value
        state   = _binary.flip_int(state, pos)
        
    # Create output arrays
    out_state[0] = state
    out_coeff[0] = coeff.real
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_y_int_np(state, ns, sites, spin: bool = BACKEND_DEF_SPIN, spin_value=_SPIN):
    r"""
    \sigma _y on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_CPX_TYPE)
    coeff           = 1.0 + 0j
    for site in sites:
        pos     = ns - 1 - site
        bit     = _binary.check_int(state, pos)
        coeff   *= (2 * bit - 1.0) * 1.0j * spin_value
        state   = _binary.flip_int(state, pos)
        
    # Create output arrays
    out_state[0] = state
    out_coeff[0] = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

def sigma_y_int(state       : int,
                ns          : int,
                sites       : Union[List[int], None],
                spin_value  : float = _SPIN):
    r"""
    \sigma _y on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    Parameters
    ----------
    state : int
        The state to apply the operator to.
    ns : int
        The number of spins in the system.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    int
        The state after applying the operator.    
    """
    if not isinstance(state, (int, np.integer)):
        return sigma_y_int_jnp(state, ns, sites, spin_value)
    if len(sites) % 2 == 1:
        return sigma_y_int_np(state, ns, sites, spin_value)
    return sigma_y_int_np_real(state, ns, sites, spin_value)

@numba.njit
def sigma_y_np_real(state       : np.ndarray,
                    sites       : Union[List[int], None],
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
    r"""
    \sigma _y on a NumPy array state.
    For each site, use the given site as index.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator    
    """
    coeff = 1.0 + 0j
    out   = state.copy()
    for site in sites:
        # For NumPy arrays, we use the site index directly.
        factor  =  (2.0 * _binary.check_arr_np(state, site) - 1.0) * 1.0j * spin_value
        coeff  *=  factor
        out     =  _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
    return ensure_operator_output_shape_numba(out, coeff.real)
    # return out, coeff.real

@numba.njit
def sigma_y_np(state        : np.ndarray,
                sites       : Union[List[int], None],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
    r"""
    \sigma _y on a NumPy array state.
    For each site, use the given site as index.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator    
    """
    coeff   = np.ones(1, dtype=DEFAULT_NP_CPX_TYPE)
    out     = state.copy()
    for site in sites:
        bit     =   _binary.check_arr_np(state, site)
        factor  =   (2.0 * bit - 1.0) * 1.0j * spin_value
        coeff  *=   factor
        out     =   _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
    return ensure_operator_output_shape_numba(out, coeff)

def sigma_y(state,
            ns              : int,
            sites           : Union[List[int], None],
            spin            : bool  = BACKEND_DEF_SPIN,
            spin_value      : float = _SPIN):
    r"""
    Dispatch for \sigma _y.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_y_int_np(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return sigma_y_np(state, sites, spin, spin_value)
    else:
        return sigma_y_jnp(state, sites, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Z (\sum _z) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_z_int_np(state     : int,
                ns           : int,
                sites        : Union[List[int], None],
                spin         : bool = BACKEND_DEF_SPIN,
                spin_value   : bool = _SPIN):
    r"""
    \sum _z on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
    Parameters
    ----------
    state : int
        The state to apply the operator to.
    ns : int
        The number of spins in the system.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    int
        The state after applying the operator.
    float
        The coefficient after applying the operator.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]    = state
    coeff           = 1.0
    for site in sites:
        pos         =  ns - 1 - site
        bit         =  _binary.check_int(state, pos)
        coeff      *=  (2.0 * bit - 1.0) * spin_value
    out_coeff[0]    = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

def sigma_z_int(state       : int,
                ns          : int,
                sites       : Union[List[int], None],
                spin_value  : float = _SPIN):
    r"""
    \sum _z on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
    """
    if not isinstance(state, (int, np.integer)):
        return sigma_z_int_jnp(state, ns, sites, spin_value)
    return sigma_z_int_np(state, ns, sites, spin_value)

@numba.njit
def sigma_z_np(state        : np.ndarray,
                sites       : Union[List[int], None],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
    r"""
    \sum _z on a NumPy array state.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator.
    float
        The coefficient after applying the operator.
    """
    coeff   = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    for site in sites:
        bit     = _binary.check_arr_np(state, site)
        coeff  *= (2 * bit - 1.0) * spin_value
    return ensure_operator_output_shape_numba(state, coeff)
    # return state, coeff

def sigma_z(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    r"""
    Dispatch for \sum _z.
    Parameters
    ----------
    state : int or np.ndarray
        The state to apply the operator to.
    ns : int
        The number of spins in the system.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    int or np.ndarray
        The state after applying the operator.
    float
        The coefficient after applying the operator.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer, jnp.integer)):
        return sigma_z_int_np(state, ns, sites, spin, spin_value)
    elif isinstance(state, np.ndarray):
        return sigma_z_np(state, sites, spin, spin_value)
    return sigma_z_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Z total (\sum _z total) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_z_total_int_np(state        : int,
                          ns          : int,
                          sites       : Union[List[int], None],
                          spin        : bool  = BACKEND_DEF_SPIN,
                          spin_value  : float = _SPIN):
    r"""
    \sum _z total on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
    """
    if sites is None:
        sites_list = []
        for i in range(ns):
            sites_list.append(i)
        sites = sites_list
    else:
        # Ensure sites is a list
        sites = list(sites)
    values = 0.0
    for site in sites:
        values += sigma_z_int_np(state, ns, [site], spin, spin_value)[1][0]
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_state[0]    = state
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_coeff[0]    = values
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_z_total_np(state      : np.ndarray,
                    sites       : Union[List[int], None],
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
    r"""
    \sum _z total on a NumPy array state.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator.
    float
        The coefficient after applying the operator.
    """
    if sites is None:
        sites = list(range(state.shape[0]))
    coeff   = np.zeros(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    for site in sites:
        bit     = _binary.check_arr_np(state, site)
        coeff  += (2 * bit - 1.0) * spin_value
    return ensure_operator_output_shape_numba(state, coeff)

# -----------------------------------------------------------------------------
#! Sigma-Plus (\sigma ^+) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_plus_int_np(state         : int,
                    ns              : int,
                    sites           : List[int],
                    spin            : bool  = BACKEND_DEF_SPIN,
                    spin_value      : float = _SPIN):
    r"""
    \sigma ^+ |state>  on an *integer* spin string.
    Returns (len -1 array new_state, len -1 array coeff).  Zero coeff -> annihilation.
    """
    new_state = state
    coeff     = 1.0
    for site in sites:
        pos = ns - 1 - site
        if _binary.check_int(new_state, pos):
            coeff = 0.0
            break
        new_state = _binary.flip_int(new_state, pos)
        coeff *= spin_value

    out_state            = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]         = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_plus_np(state         : np.ndarray,
                sites           : List[int],
                spin            : bool  = BACKEND_DEF_SPIN,
                spin_value      : float = _SPIN):
    r"""
    \sigma ^+ |state> on a NumPy array representation (0/1 occupation).
    """
    coeff = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out   = state.copy()
    for site in sites:
        bit     = _binary.check_arr_np(out, site)
        if bit: # annihilation
            coeff *= 0.0
            break
        out     = _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
        coeff  *= spin_value
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

def sigma_plus(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_plus_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_plus_np(state, sites, spin, spin_value)
    return sigma_plus_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Minus (\sigma ^-) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_minus_int_np(state        : int,
                    ns              : int,
                    sites           : List[int],
                    spin            : bool  = BACKEND_DEF_SPIN,
                    spin_value      : float = _SPIN):
    r"""
    Applies the spin lowering (\sigma ^ -) operator to the specified sites of a quantum spin state represented as an integer.

    Parameters:
        state (int):
            The integer representation of the quantum spin state.
        ns (int):
            The total number of sites (spins) in the system.
        sites (List[int]):
            List of site indices (0-based, leftmost is 0) where the \sigma ^ - operator is applied.
        spin (bool, optional):
            Indicates if the system uses spin representation. Defaults to BACKEND_DEF_SPIN.
        spin_value (float, optional):
            The value to multiply the coefficient by when lowering a spin. Defaults to _SPIN.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: A NumPy array containing the new state(s) as integer(s) after applying \sigma ^ -.
            - out_coeff: A NumPy array containing the corresponding coefficient(s) for the new state(s).

    Notes:
        - If any of the specified sites is already in the |↓> (spin-down, 0) state, the coefficient is set to 0 and the state is not changed.
        - The function assumes that the spin state is encoded in binary, with each bit representing a site.
    """

    new_state = state
    coeff     = 1.0
    for site in sites:
        pos = ns - 1 - site
        if _binary.check_int(new_state, pos) == 0: # trying to lower |↓> -> 0
            coeff = 0.0
            break
        new_state = _binary.flip_int(new_state, pos)  # 1 -> 0
        coeff    *= spin_value

    out_state            = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]         = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_minus_np(state        : np.ndarray,
                sites           : List[int],
                spin            : bool  = BACKEND_DEF_SPIN,
                spin_value      : float = _SPIN):
    r"""
    Applies the sigma-minus (lowering) operator to the specified sites of a quantum state.

    This function operates on a numpy array representing the quantum state, applying the lowering operator
    to each site in the provided list. If the occupation at a site is 0, the operation annihilates the state
    and returns a coefficient of 0. Otherwise, the occupation is flipped from 1 to 0, and the coefficient is
    multiplied by the given spin value.

    Args:
        state (np.ndarray):
            The quantum state represented as a numpy array.
        sites (List[int]):
            List of site indices where the sigma-minus operator is applied.
        spin (bool, optional):
            Indicates whether to use spinful or spinless operations. Defaults to BACKEND_DEF_SPIN.
        spin_value (float, optional):
            The value to multiply the coefficient by for each successful lowering operation. Defaults to _SPIN.

    Returns:
        Tuple[np.ndarray, float]: The updated quantum state and the resulting coefficient after applying the operator.
    """

    coeff = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out   = state.copy()
    for site in sites:
        bit     = _binary.check_arr_np(out, site)
        if bit == 0:
            coeff *= 0.0
            break
        out     = _binary.flip_array_np_spin(out, site) if spin else _binary.flip_array_np_nspin(out, site)
        coeff  *= spin_value
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

def sigma_minus(state,
                ns          : int,
                sites       : Union[List[int], None],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
    r"""
    Dispatch for \sigma ^-.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_minus_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_minus_np(state, sites, spin, spin_value)
    return sigma_minus_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma_pm (\sigma ^+ then \sigma ^-) operator and Sigma_mp (\sigma ^- then \sigma ^+) operator
# -----------------------------------------------------------------------------

@numba.njit
def _sigma_pm_int_core(state        : int,
                        ns          : int,
                        sites       : List[int],
                        start_up    : bool,
                        spin        : bool  = BACKEND_DEF_SPIN,
                        spin_val    : float = _SPIN):
    r"""
    Core for \sum _pm / \sum _mp alternating flips on integer states.
    start_up = True  -> even  = \sigma ^+ , odd = \sigma ^ -
    start_up = False -> even  = \sigma ^ - , odd = \sigma ^+
    """
    new_state   = state
    coeff       = 1.0
    for i, site in enumerate(sites):
        pos     = ns - 1 - site
        bit     = _binary.check_int(new_state, pos)
        need_up = (i % 2 == 0) == start_up

        if need_up: # \sigma ^+
            if bit == 1:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        else: # \sigma ^ -
            if bit == 0:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        coeff *= spin_val
    return new_state, coeff

@numba.njit
def sigma_pm_int_np(state       : int,
                    ns          : int,
                    sites       : List[int],
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_val    : float = _SPIN):
    r"""
    Applies the spin raising or lowering operator (\sigma ^+ or \sigma ^ -) to a given quantum state represented as an integer.

    Parameters:
        state (int):
            The integer representation of the quantum state to which the operator is applied.
        ns (int):
            The number of sites (spins) in the system.
        sites (List[int]):
            The list of site indices where the operator acts.
        spin_val (float, optional):
            The value of the spin (default is _SPIN).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: A NumPy array containing the new state(s) as integer(s) after applying the operator.
            - out_coeff: A NumPy array containing the corresponding coefficient(s) for each new state.
    """
    new_state, coeff     = _sigma_pm_int_core(state, ns, sites, True, spin, spin_val)
    out_state            = np.empty((1,1), dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0,0]       = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def _sigma_pm_np_core(state     : np.ndarray,
                    sites       : List[int],
                    start_up    : bool,
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_val    : float = _SPIN):
    r"""
    Applies a sequence of spin raising or lowering operators (\sigma ^+ or \sigma ^ -) to specified sites of a quantum state array.

    Parameters:
        state (np.ndarray):
            The quantum state represented as a NumPy array of bits (0 for spin-down, 1 for spin-up).
        sites (List[int]):
            List of site indices where the spin operators are to be applied.
        start_up (bool):
            If True, the sequence starts with a raising operator (\sigma ^+); if False, starts with lowering (\sigma ^ -).
        spin_val (float):
            The coefficient to multiply for each successful spin flip.

    Returns:
        Tuple[np.ndarray, float]: 
            - The modified state array after applying the operators (may be unchanged if operation is invalid).
            - The resulting coefficient (0.0 if the operation is not allowed by spin selection rules, otherwise the product of spin_val for each flip).
    """
    coeff = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out   = state.copy()
    for i, site in enumerate(sites):
        bit      = _binary.check_arr_np(out, site)
        need_up  = (i % 2 == 0) == start_up
        if need_up:
            if bit == 1:
                coeff *= 0.0
                break
        else:
            if bit == 0:
                coeff *= 0.0
                break
        out     = _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
        coeff  *= spin_val
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

@numba.njit
def sigma_pm_np(state       : np.ndarray,
                sites       : List[int],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_val    : float = _SPIN):
    r"""\sigma ^+ \sigma ^ - alternating on NumPy array (start \sigma ^+)."""
    return _sigma_pm_np_core(state, sites, True, spin, spin_val)

def sigma_pm(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_pm_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_pm_np(state, sites, spin, spin_value)
    return sigma_pm_jnp(state, ns, sites, spin, spin_value)

@numba.njit
def sigma_mp_int_np(state      : int,
                    ns         : int,
                    sites      : List[int],
                    spin       : bool  = BACKEND_DEF_SPIN,
                    spin_val   : float = _SPIN):
    r"""Alternating \sigma ^ - \sigma ^+ starting with \sigma ^ - on even index (integer state)."""
    new_state, coeff     = _sigma_pm_int_core(state, ns, sites, False, spin, spin_val)
    out_state            = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]         = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_mp_np(state       : np.ndarray,
                sites       : List[int],
                spin        : bool  = BACKEND_DEF_SPIN,  
                spin_val    : float = _SPIN):
    r"""\sigma ^ - \sigma ^+ alternating on NumPy array (start \sigma ^ -)."""
    return _sigma_pm_np_core(state, sites, False, spin, spin_val)

def sigma_mp(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_mp_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_mp_np(state, sites, spin, spin_value)
    return sigma_mp_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-K operator (Fourier-transformed spin operator)
# -----------------------------------------------------------------------------

@numba.njit
def sigma_k_int_np(state    : int,
                ns          : int,
                sites       : List[int],
                k           : float,
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
    r"""
    Applies the momentum-space spin operator (\sum _k) to a given quantum state represented as an integer.

    This function computes the action of the \sum _k operator (sum over sites of \sum _z * exp(i k r)) on a basis state
    encoded as an integer, for a specified set of sites and momentum k. The result is returned as arrays of output
    states and their corresponding coefficients.

    Args:
        state (int):
            Integer representation of the quantum basis state.
        ns (int):
            Total number of sites in the system.
        sites (List[int]):
            List of site indices on which to apply the operator.
        k (float):
            Momentum value (in radians) for the Fourier transform.
        spin (bool, optional):
            If True, use spin-1/2 convention for \sum _z; otherwise, use occupation number. Defaults to BACKEND_DEF_SPIN.
        spin_value (float, optional):
            Value to scale the spin operator. Defaults to _SPIN.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: Array of output state(s) (as integers).
            - out_coeff: Array of corresponding coefficients (complex numbers).

    Notes:
        - The function assumes that the binary representation of `state` encodes the occupation/spin configuration.
        - The normalization is by sqrt(len(sites)), or 1.0 if `sites` is empty.
    """

    accum = 0.0 + 0.0j
    for i in sites:
        pos        = ns - 1 - i
        bit        = _binary.check_int(state, pos)
        sigma_z_i  = (2.0 * bit - 1.0) * spin_value
        accum     += sigma_z_i * (np.cos(k * i) + 1j * np.sin(k * i))
    norm          = np.sqrt(len(sites)) if sites else 1.0
    out_state     = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff     = np.empty(1, dtype=DEFAULT_NP_CPX_TYPE)
    out_state[0]  = state
    out_coeff[0]  = accum / norm
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_k_np(state    : np.ndarray,
            sites       : List[int],
            k           : float,
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    r"""
    \sigma _k |state>  for a NumPy spin/occupation array (same formula as above).
    """
    accum = np.zeros(1, dtype=DEFAULT_NP_CPX_TYPE)
    for i in sites:
        bit        = _binary.check_arr_np(state, i)
        sigma_z_i  = (2.0 * bit - 1.0) * spin_value
        accum     += sigma_z_i * np.exp(1j * k * i)
    norm = np.sqrt(len(sites)) if len(sites) > 0 else 1.0
    return ensure_operator_output_shape_numba(state, accum / norm)
    # return state, accum / norm

def sigma_k(state,
            ns          : int,
            sites       : Union[List[int], None],
            k           : float,
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_k_int_np(int(state), ns, sites, k, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_k_np(state, sites, k, spin, spin_value)
    return sigma_k_jnp(state, ns, sites, k, spin, spin_value)

################################################################################
#! Factory Functions: Wrap the elementary functions in Operator objects.
################################################################################

# -----------------------------------------------------------------------------
#! Factory function for sigma-x (\sigma _x)
# -----------------------------------------------------------------------------

def sig_x(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory function for \sigma _x.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator. 
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    Operator
        The \sigma _x operator.    
    """
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_x_int_np,
        op_func_np  = sigma_x_np,
        op_func_jnp = sigma_x_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sx",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-y (\sigma _y)
# -----------------------------------------------------------------------------

def sig_y( lattice     : Optional[Lattice]     = None,
            ns         : Optional[int]         = None,
            type_act   : OperatorTypeActing    = OperatorTypeActing.Global,
            sites      : Optional[List[int]]   = None,
            spin       : bool                  = BACKEND_DEF_SPIN,
            spin_value : float                 = _SPIN) -> Operator:
    r"""
    Factory function for \sigma _y.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator.
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    Operator
        The \sigma _y operator.
    """
    is_real = type_act == OperatorTypeActing.Correlation or (sites is not None and isinstance(sites, list) and len(sites) % 2 == 0)
    
    int_fun = sigma_y_int_np if not is_real else sigma_y_int_np_real
    np_fun  = sigma_y_np
    jnp_fun = sigma_y_jnp
    
    return create_operator(
        type_act    = type_act,
        op_func_int = int_fun,
        op_func_np  = np_fun,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sy",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma_z (\sum _z)
# -----------------------------------------------------------------------------

def sig_z(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory function for \sigma _x.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator.
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    Operator
        The \sigma _x operator.
    """
    
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_z_int_np,
        op_func_np  = sigma_z_np,
        op_func_jnp = sigma_z_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sz",
        modifies    = False
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-plus (\sigma ^+)
# -----------------------------------------------------------------------------

def sig_p(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the spin-raising operator \sigma ^+.
    """
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_plus_int_np,
        op_func_np  = sigma_plus_np,
        op_func_jnp = sigma_plus_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),          # create_operator injects `spin`
        name        = "Sp",
        modifies    = True                    # \sigma ^+ flips bits
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-minus (\sigma ^ -)
# -----------------------------------------------------------------------------

def sig_m(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the spin-lowering operator \sigma ^ -.
    """

    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_minus_int_np,
        op_func_np  = sigma_minus_np,
        op_func_jnp = sigma_minus_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sm",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-pm (\sigma ^+ then \sigma ^ -)
# -----------------------------------------------------------------------------

def sig_pm( lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the alternating operator: even-indexed sites \sigma ^+, odd-indexed \sigma ^ -.
    """

    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_pm_int_np,
        op_func_np  = sigma_pm_np,
        op_func_jnp = sigma_pm_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Spm",
        modifies    = True
    )
    
# -----------------------------------------------------------------------------
#! Factory function for sigma-mp (\sigma ^ - then \sigma ^+)
# -----------------------------------------------------------------------------

def sig_mp( lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the alternating operator: even-indexed sites \sigma ^ -, odd-indexed \sigma ^+.
    """

    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_mp_int_np,
        op_func_np  = sigma_mp_np,
        op_func_jnp = sigma_mp_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Smp",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-k (\sigma _k)
# -----------------------------------------------------------------------------

def sig_k(  k           : float,
            lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the momentum-space operator  

        \sigma _k = (1/\sqrtN)\,\sum_{i\in\text{sites}} \sum _z(i)\,e^{\,ik i}.
    """

    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_k_int_np,
        op_func_np  = sigma_k_np,
        op_func_jnp = sigma_k_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (k, spin, spin_value),
        name        = f"Sk(k={k:.3g})",
        modifies    = False # \sigma _k leaves the state unchanged
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-total (\sum _total)
# -----------------------------------------------------------------------------

def sig_z_total( lattice     : Optional[Lattice]     = None,
                ns          : Optional[int]         = None,
                type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
                sites       : Optional[List[int]]   = None,
                spin        : bool                  = BACKEND_DEF_SPIN,
                spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the total spin operator in the z-direction.
    This operator is the sum of all \sum _z operators acting on the specified sites.
    """
    
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_z_total_int_np,
        op_func_np  = sigma_z_total_np,
        op_func_jnp = sigma_z_total_jnp,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sz_total",
        modifies    = False # \sum _total leaves the state unchanged
    )

# -----------------------------------------------------------------------------
#! Most common mixtures
# -----------------------------------------------------------------------------

#? Int

def create_sigma_mixed_int(op1_fun: Callable, op2_fun: Callable):
    
    @numba.njit
    def sigma_mixed_int_np(state, ns, sites, spin=True, spin_value=0.5):
        if sites is None or len(sites) < 2:
            s_dumb, c_dumb = op1_fun(state, ns, sites, spin, spin_value)
            return s_dumb, c_dumb.astype(np.complex128)

        # Apply Op 1
        s_arr1, c_arr1  = op1_fun(state, ns, [sites[0]], spin, spin_value)
        coeff1          = c_arr1[0]
        
        if coeff1 == 0.0 or (coeff1 == 0.0 + 0.0j):
            #! Force cast to complex so this branch matches the other branch
            #! even if op1 is purely real (like Sigma X or Z)
            return s_arr1, c_arr1.astype(np.complex128) * 0.0
            
        s_int           = s_arr1[0]
        s_arr2, c_arr2  = op2_fun(s_int, ns, [sites[1]], spin, spin_value)
        
        # Explicitly cast both to complex before multiplying to be 100% safe
        return s_arr2, c_arr1.astype(np.complex128) * c_arr2.astype(np.complex128)
    
    return sigma_mixed_int_np

def create_sigma_mixed_np(op1_fun: Callable, op2_fun: Callable):
    @numba.njit(cache=True)
    def sigma_mixed_np(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        if sites is None or len(sites) < 2:
            raise ValueError("sigma_mixed_np requires at least two sites.")
        s_arr1, c_arr1 = op1_fun(state, [sites[0]], spin, spin_value)
        coeff1 = c_arr1[0] if c_arr1.shape[0] == 1 else c_arr1
        if coeff1 == 0.0 or (coeff1 == 0.0 + 0.0j):
            return s_arr1, c_arr1 * 0.0
        s_arr2, c_arr2 = op2_fun(s_arr1, [sites[1]], spin, spin_value)
        return s_arr2, c_arr1 * c_arr2
    return sigma_mixed_np

def create_sigma_mixed_jnp(op1_fun: Callable, op2_fun: Callable):
    if not JAX_AVAILABLE:
        def _stub(*args, **kwargs):
            raise RuntimeError("JAX not available for sigma mixed operator.")
        return _stub

    @jax.jit
    def sigma_mixed_jnp(state, ns, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        if (sites is None) or (len(sites) < 2):
            raise ValueError("sigma_mixed_jnp requires at least two sites.")
        s_arr1, c_arr1  = op1_fun(state, ns, [sites[0]], spin, spin_value)
        coeff1          = c_arr1[0]
        def _apply_second(s_arr1, c_arr1):
            s_int           = s_arr1[0]
            s_arr2, c_arr2  = op2_fun(s_int, ns, [sites[1]], spin, spin_value)
            return s_arr2, c_arr1 * c_arr2
        # Branch: if first coeff zero skip second op
        return jax.lax.cond(
            (coeff1 == 0.0) | (coeff1 == 0.0 + 0.0j),
            lambda _: (s_arr1, c_arr1 * 0.0),
            lambda _: _apply_second(s_arr1, c_arr1),
            operand = None
        )
    return sigma_mixed_jnp

sigma_xy_mixed_int_np   = create_sigma_mixed_int(sigma_x_int_np, sigma_y_int_np)
sigma_yx_mixed_int_np   = create_sigma_mixed_int(sigma_y_int_np, sigma_x_int_np)
sigma_yz_mixed_int_np   = create_sigma_mixed_int(sigma_y_int_np, sigma_z_int_np)
sigma_zx_mixed_int_np   = create_sigma_mixed_int(sigma_z_int_np, sigma_x_int_np)
sigma_xz_mixed_int_np   = create_sigma_mixed_int(sigma_x_int_np, sigma_z_int_np)
sigma_zy_mixed_int_np   = create_sigma_mixed_int(sigma_z_int_np, sigma_y_int_np)

#? NP
sigma_xy_mixed_np       = create_sigma_mixed_np(sigma_x_np, sigma_y_np)
sigma_yx_mixed_np       = create_sigma_mixed_np(sigma_y_np, sigma_x_np)
sigma_yz_mixed_np       = create_sigma_mixed_np(sigma_y_np, sigma_z_np)
sigma_zx_mixed_np       = create_sigma_mixed_np(sigma_z_np, sigma_x_np)
sigma_xz_mixed_np       = create_sigma_mixed_np(sigma_x_np, sigma_z_np)
sigma_zy_mixed_np       = create_sigma_mixed_np(sigma_z_np, sigma_y_np)

#? JNP
if JAX_AVAILABLE:
    sigma_xy_mixed_jnp  = create_sigma_mixed_jnp(sigma_x_jnp, sigma_y_jnp)
    sigma_yx_mixed_jnp  = create_sigma_mixed_jnp(sigma_y_jnp, sigma_x_jnp)
    sigma_yz_mixed_jnp  = create_sigma_mixed_jnp(sigma_y_jnp, sigma_z_jnp)
    sigma_zx_mixed_jnp  = create_sigma_mixed_jnp(sigma_z_jnp, sigma_x_jnp)
    sigma_xz_mixed_jnp  = create_sigma_mixed_jnp(sigma_x_jnp, sigma_z_jnp)
    sigma_zy_mixed_jnp  = create_sigma_mixed_jnp(sigma_z_jnp, sigma_y_jnp)
else:
    sigma_xy_mixed_jnp  = None
    sigma_yx_mixed_jnp  = None
    sigma_yz_mixed_jnp  = None
    sigma_zx_mixed_jnp  = None
    sigma_xz_mixed_jnp  = None
    sigma_zy_mixed_jnp  = None

def make_sigma_mixed(name, lattice=None):
    
    if name == 'xy':
        int_func = sigma_xy_mixed_int_np
        np_func  = sigma_xy_mixed_np
        jnp_func = sigma_xy_mixed_jnp if JAX_AVAILABLE else None
    elif name == 'yx':
        int_func = sigma_yx_mixed_int_np
        np_func  = sigma_yx_mixed_np
        jnp_func = sigma_yx_mixed_jnp if JAX_AVAILABLE else None
    elif name == 'yz':
        int_func = sigma_yz_mixed_int_np
        np_func  = sigma_yz_mixed_np
        jnp_func = sigma_yz_mixed_jnp if JAX_AVAILABLE else None
    elif name == 'zx':
        int_func = sigma_zx_mixed_int_np
        np_func  = sigma_zx_mixed_np
        jnp_func = sigma_zx_mixed_jnp if JAX_AVAILABLE else None
    elif name == 'xz':
        int_func = sigma_xz_mixed_int_np
        np_func  = sigma_xz_mixed_np
        jnp_func = sigma_xz_mixed_jnp if JAX_AVAILABLE else None
    elif name == 'zy':
        int_func = sigma_zy_mixed_int_np
        np_func  = sigma_zy_mixed_np
        jnp_func = sigma_zy_mixed_jnp if JAX_AVAILABLE else None
    else:
        raise ValueError(f"Unknown mixed sigma operator name: {name}")
    
    return create_operator(
        type_act    = OperatorTypeActing.Correlation,
        op_func_int = int_func,
        op_func_np  = np_func,
        op_func_jnp = jnp_func,
        lattice     = lattice,
        name        = name,
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Finalize
# -----------------------------------------------------------------------------

# Aliases for legacy/test compatibility
sig_plus    = sig_p
sig_minus   = sig_m

# -----------------------------------------------------------------------------

###############################################################################
#! Registration with the operator catalog
###############################################################################

def _register_catalog_entries():
    """
    Register standard spin-1/2 onsite operators with the global catalog.
    """

    def _sigma_x_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         =   sigma_x_int,
            fun_np          =   sigma_x_np,
            fun_jax         =   sigma_x_jnp if JAX_AVAILABLE else None,
            site_parity     =   1,
            modifies_state  =   True,
        )

    def _sigma_y_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         =   sigma_y_int_np, # Use the NumPy version for integer inputs
            fun_np          =   sigma_y_np,
            fun_jax         =   sigma_y_jnp if JAX_AVAILABLE else None,
            site_parity     =   1,
            modifies_state  =   True,
        )

    def _sigma_z_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         =   sigma_z_int,
            fun_np          =   sigma_z_np,
            fun_jax         =   sigma_z_jnp if JAX_AVAILABLE else None,
            site_parity     =   1,
            modifies_state  =   False,
        )

    def _sigma_plus_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         =   sigma_plus_int_np,
            fun_np          =   sigma_plus_np,
            fun_jax         =   sigma_plus_jnp if JAX_AVAILABLE else None,
            site_parity     =   1,
            modifies_state  =   True,
        )

    def _sigma_minus_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         =   sigma_minus_int_np,
            fun_np          =   sigma_minus_np,
            fun_jax         =   sigma_minus_jnp if JAX_AVAILABLE else None,
            site_parity     =   1,
            modifies_state  =   True,
        )

    # -------------------------------------------------------------
    # Register the operators
    # -------------------------------------------------------------
    
    register_local_operator(
        LocalSpaceTypes.SPIN_1_2,
        key                 =   "sigma_x",
        factory             =   _sigma_x_factory,
        description         =   r"Pauli \sigma_x flip acting on spin-1/2 basis states.",
        algebra             =   r"\sigma_x^2 = 1,   {\sigma_x, \sum _y} = 0,   [\sigma_x, \sum _z] = 2i \sum _y",
        sign_convention     =   r"No non-trivial sign; operates locally in the \sum _z eigenbasis.",
        tags                =   ("spin", "pauli"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1_2,
        key                 =   "sigma_y",
        factory             =   _sigma_y_factory,
        description         =   r"Pauli \sigma_y rotation acting on spin-1/2 basis states.",
        algebra             =   r"\sigma_y^2 = 1,   {\sigma_y, \sigma_x} = 0,   [\sigma_y, \sigma_z] = -2i \sigma_x",
        sign_convention     =   r"Standard Pauli matrix in \sigma_z basis; introduces +/- i phases.",
        tags                =   ("spin", "pauli"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1_2,
        key                 =   "sigma_z",
        factory             =   _sigma_z_factory,
        description         =   r"Pauli \sigma_z operator (magnetisation in the computational basis).",
        algebra             =   r"\sigma_z^2 = 1,   [\sigma_z, \sigma_+/- ] = +/- 2 \sigma_+/- ",
        sign_convention     =   r"Diagonal in \sigma_z basis; no additional sign factors.",
        tags                =   ("spin", "pauli"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1_2,
        key                 =   "sigma_plus",
        factory             =   _sigma_plus_factory,
        description         =   r"Spin raising operator \sigma ^+ = (\sigma _x + i \sigma _y)/2.",
        algebra             =   r"\sigma ^+\sigma ^ - + \sigma ^ -\sigma ^+ = 1,   [\sigma _z, \sigma ^+] = 2\sigma ^+",
        sign_convention     =   r"Acts locally without JW strings; raises \sigma _z eigenvalue by one.",
        tags                =   ("spin", "raising"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1_2,
        key                 =   "sigma_minus",
        factory             =   _sigma_minus_factory,
        description         =   r"Spin lowering operator \sigma ^ - = (\sigma _x - i \sigma _y)/2.",
        algebra             =   r"\sigma ^+\sigma ^ - + \sigma ^ -\sigma ^+ = 1,   [\sigma _z, \sigma ^ -] = -2\sigma ^ -",
        sign_convention     =   r"Acts locally without JW strings; lowers \sigma _z eigenvalue by one.",
        tags                =   ("spin", "lowering"),
    )

# -----------------------------------------------------------------------------
#! Register the catalog entries upon module import
# -----------------------------------------------------------------------------

_register_catalog_entries()

# -----------------------------------------------------------------------------
#! EOF 
# -----------------------------------------------------------------------------