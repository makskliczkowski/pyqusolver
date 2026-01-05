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

import  os 
import  numba
import  numpy   as np
from    typing  import Dict, List, Union, Optional, Callable
from    enum    import IntEnum

################################################################################

try:
    # main imports
    from QES.Algebra.Operator.operator          import Operator, OperatorTypeActing, create_operator, ensure_operator_output_shape_numba
    from QES.Algebra.Operator.catalog           import register_local_operator
    from QES.Algebra.Operator.special_operator  import CUSTOM_OP_BASE
    from QES.Algebra.Hilbert.hilbert_local      import LocalOpKernels, LocalSpaceTypes
except ImportError as e:
    raise ImportError("Failed to import required modules. Ensure that the QES package is correctly installed.") from e

################################################################################

try:
    import QES.general_python.common.binary     as _binary
    from QES.general_python.lattices.lattice    import Lattice
    from QES.general_python.algebra.utils       import DEFAULT_NP_FLOAT_TYPE, DEFAULT_NP_CPX_TYPE
    from QES.general_python.common.binary       import BACKEND_REPR as _SPIN, BACKEND_DEF_SPIN
except ImportError as e:
    raise ImportError("Failed to import required QES general Python modules.") from e

################################################################################
JAX_AVAILABLE = os.getenv("PY_JAX_AVAILABLE", "0") == "1"
################################################################################

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    # sigma x
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_x_int_jnp, sigma_x_jnp, sigma_x_inv_jnp
    # sigma y
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_y_int_jnp, sigma_y_jnp, sigma_y_real_jnp, sigma_y_inv_jnp
    # sigma z
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_z_int_jnp, sigma_z_jnp, sigma_z_inv_jnp
    # sigma plus
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_plus_int_jnp, sigma_plus_jnp
    # sigma minus
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_minus_int_jnp, sigma_minus_jnp
    # sigma pm
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_pm_int_jnp, sigma_pm_jnp
    # sigma mp
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_mp_int_jnp, sigma_mp_jnp
    # sigma k
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_k_int_jnp, sigma_k_jnp, sigma_k_inv_jnp
    # sigma z total
    from QES.Algebra.Operator.impl.jax.operators_spin import sigma_z_total_int_jnp, sigma_z_total_jnp
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

@numba.njit(inline='always')
def _sigma_x_core(state, ns, sites, spin_value=_SPIN):
    new_state   = state
    coeff       = 1.0
    
    for site in sites:
        pos         = ns - 1 - site
        new_state   = _binary.flip_int(new_state, pos)
        coeff      *= spin_value
        
    return new_state, coeff

@numba.njit(inline='always')
def sigma_x_int_np(state, ns, sites, spin: bool = BACKEND_DEF_SPIN, spin_value=_SPIN):
    r"""
    Apply the Pauli-X (\sigma _x) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    """
    s, c = _sigma_x_core(state, ns, sites, spin_value)
    return (s,), (c,)

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

@numba.njit(inline='always')
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

@numba.njit(inline='always')
def _sigma_y_core(state, ns, sites, spin_value=_SPIN):
    new_state   = state
    coeff       = 1.0 + 0.0j
    
    for site in sites:
        pos         = ns - 1 - site
        bit         = _binary.check_int(new_state, pos)
        # Pauli Y logic: 1j * (1 if 0, -1 if 1)
        factor      = 1j * (1.0 - 2.0 * bit) * spin_value
        coeff      *= factor
        new_state   = _binary.flip_int(new_state, pos)
    return new_state, coeff

@numba.njit(inline='always')
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
    s, c            = _sigma_y_core(state, ns, sites, spin_value)
    return (s,), (c.real,)

@numba.njit(inline='always')
def sigma_y_int_np(state, ns, sites, spin: bool = BACKEND_DEF_SPIN, spin_value=_SPIN):
    r"""
    \sigma _y on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    """
    s, c = _sigma_y_core(state, ns, sites, spin_value)
    return (s,), (c,)

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

@numba.njit(inline='always')
def _sigma_z_core(state, ns, sites, spin_value=_SPIN):
    coeff = 1.0
    for site in sites:
        pos     = ns - 1 - site
        bit     = _binary.check_int(state, pos)
        coeff  *= (1.0 - 2.0 * bit) * spin_value
    return state, coeff

@numba.njit(inline='always')
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
    s, c = _sigma_z_core(state, ns, sites, spin_value)
    return (s,), (c,)

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

@numba.njit(inline='always')
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

@numba.njit(inline='always')
def sigma_z_total_int_np(state      : int,
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
    return (state,), (values,)

@numba.njit(inline='always')
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

@numba.njit(inline='always')
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

    return (new_state,), (coeff,)

@numba.njit(inline='always')
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

@numba.njit(inline='always')
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

    return (new_state,), (coeff,)

@numba.njit(inline='always')
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

@numba.njit(inline='always')
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
    new_state   = state
    coeff       = 1.0
    for i, site in enumerate(sites):
        pos     = ns - 1 - site
        bit     = _binary.check_int(new_state, pos)
        need_up = (i % 2 == 0)  # even index -> \sigma ^+

        if need_up: # \sigma ^+
            if bit == 1:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        else:       # \sigma ^ -
            if bit == 0:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        coeff *= spin_val
    return (new_state,), (coeff,)

@numba.njit(inline='always')
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

@numba.njit(inline='always')
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

@numba.njit(inline='always')
def sigma_mp_int_np(state      : int,
                    ns         : int,
                    sites      : List[int],
                    spin       : bool  = BACKEND_DEF_SPIN,
                    spin_val   : float = _SPIN):
    r"""Alternating \sigma ^ - \sigma ^+ starting with \sigma ^ - on even index (integer state)."""
    new_state   = state
    coeff       = 1.0
    for i, site in enumerate(sites):
        pos     = ns - 1 - site
        bit     = _binary.check_int(new_state, pos)
        need_up = (i % 2 != 0)  # even index -> \sigma ^-

        if need_up: # \sigma ^+
            if bit == 1:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        else:       # \sigma ^ -
            if bit == 0:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        coeff *= spin_val
    return (new_state,), (coeff,)

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
    return (state,), (accum / norm,)

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

#? Lookup codes for NUMBA, as it's idiotically not hashable.

class SpinLookupCodes(IntEnum):
    # globals
    sig_x_global:   int = 0
    sig_y_global:   int = 1
    sig_z_global:   int = 2
    sig_p_global:   int = 3
    sig_m_global:   int = 4
    sig_xy_global:  int = 5
    sig_yx_global:  int = 6
    sig_xz_global:  int = 7
    sig_zx_global:  int = 8
    sig_yz_global:  int = 9
    sig_zy_global:  int = 10
    
    # local - accept single site
    sig_x_local:    int = 11
    sig_y_local:    int = 12
    sig_z_local:    int = 13
    sig_p_local:    int = 14
    sig_m_local:    int = 15
    
    # correlation
    sig_x_corr:     int = 20
    sig_y_corr:     int = 21
    sig_z_corr:     int = 22
    sig_p_corr:     int = 23
    sig_m_corr:     int = 24
    sig_xy_corr:    int = 25
    sig_yx_corr:    int = 26
    sig_xz_corr:    int = 27
    sig_zx_corr:    int = 28
    sig_yz_corr:    int = 29
    sig_zy_corr:    int = 30
    sig_pm_corr:    int = 31
    sig_mp_corr:    int = 32
    
    @staticmethod
    def is_custom_code(code: int) -> bool:
        """Check if a code represents a custom operator."""
        return code > 32 or code < 0
    
    @staticmethod
    def to_dict() -> Dict[str, int]:
        return {
            'Sx'            : SpinLookupCodes.sig_x_global,
            'Sy'            : SpinLookupCodes.sig_y_global,
            'Sz'            : SpinLookupCodes.sig_z_global,
            'Sp'            : SpinLookupCodes.sig_p_global,
            'Sm'            : SpinLookupCodes.sig_m_global,
            'SxSy'          : SpinLookupCodes.sig_xy_global,
            'SySx'          : SpinLookupCodes.sig_yx_global,
            'SxSz'          : SpinLookupCodes.sig_xz_global,
            'SzSx'          : SpinLookupCodes.sig_zx_global,
            'SySz'          : SpinLookupCodes.sig_yz_global,
            'SzSy'          : SpinLookupCodes.sig_zy_global,
            # local
            'Sx/L'          : SpinLookupCodes.sig_x_local,
            'Sy/L'          : SpinLookupCodes.sig_y_local,
            'Sz/L'          : SpinLookupCodes.sig_z_local,
            'Sp/L'          : SpinLookupCodes.sig_p_local,
            'Sm/L'          : SpinLookupCodes.sig_m_local,
            # corr
            'Sx/C'          : SpinLookupCodes.sig_x_corr,
            'Sy/C'          : SpinLookupCodes.sig_y_corr,
            'Sz/C'          : SpinLookupCodes.sig_z_corr,
            'Sp/C'          : SpinLookupCodes.sig_p_corr,
            'Sm/C'          : SpinLookupCodes.sig_m_corr,
            'SxSy/C'        : SpinLookupCodes.sig_xy_corr,
            'SySx/C'        : SpinLookupCodes.sig_yx_corr,
            'SxSz/C'        : SpinLookupCodes.sig_xz_corr,
            'SzSx/C'        : SpinLookupCodes.sig_zx_corr,
            'SySz/C'        : SpinLookupCodes.sig_yz_corr,
            'SzSy/C'        : SpinLookupCodes.sig_zy_corr,
            'Spm/C'         : SpinLookupCodes.sig_pm_corr,
            'Smp/C'         : SpinLookupCodes.sig_mp_corr,
        }

SPIN_LOOKUP_CODES = SpinLookupCodes

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
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code = SPIN_LOOKUP_CODES.sig_x_global
    elif OperatorTypeActing.is_type_local(type_act):
        code = SPIN_LOOKUP_CODES.sig_x_local
    else:
        code = SPIN_LOOKUP_CODES.sig_x_corr
    
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
        modifies    = True,
        code        = code
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
    
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code = SPIN_LOOKUP_CODES.sig_y_global
    elif OperatorTypeActing.is_type_local(type_act):
        code = SPIN_LOOKUP_CODES.sig_y_local
    else:
        code = SPIN_LOOKUP_CODES.sig_y_corr
    
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
        modifies    = True,
        code        = code
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
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code = SPIN_LOOKUP_CODES.sig_z_global
    elif OperatorTypeActing.is_type_local(type_act):
        code = SPIN_LOOKUP_CODES.sig_z_local
    else:
        code = SPIN_LOOKUP_CODES.sig_z_corr
        
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
        modifies    = False,
        code        = code
    )

def sig_p(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the spin-raising operator \sigma ^+.
    """
    
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code = SPIN_LOOKUP_CODES.sig_p_global
    elif OperatorTypeActing.is_type_local(type_act):
        code = SPIN_LOOKUP_CODES.sig_p_local
    else:
        code = SPIN_LOOKUP_CODES.sig_p_corr
    
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
        modifies    = True,                          # \sigma ^+ flips bits
        code        = code
    )

def sig_m(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the spin-lowering operator \sigma ^ -.
    """

    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code = SPIN_LOOKUP_CODES.sig_m_global
    elif OperatorTypeActing.is_type_local(type_act):
        code = SPIN_LOOKUP_CODES.sig_m_local
    else:
        code = SPIN_LOOKUP_CODES.sig_m_corr

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
        modifies    = True,
        code        = code
    )

def sig_pm( lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the alternating operator: even-indexed sites \sigma ^+, odd-indexed \sigma ^ -.
    """

    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code = SPIN_LOOKUP_CODES.sig_pm_corr
    else:
        code = SPIN_LOOKUP_CODES.sig_pm_corr

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
        modifies    = True,
        code        = code
    )

def sig_mp( lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the alternating operator: even-indexed sites \sigma ^ -, odd-indexed \sigma ^+.
    """

    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code = SPIN_LOOKUP_CODES.sig_mp_corr
    else:
        code = SPIN_LOOKUP_CODES.sig_mp_corr

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
        modifies    = True,
        code        = code
    )

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
#! Pauli Operators
# -----------------------------------------------------------------------------

@numba.njit(nogil=True)
def _apply_pauli_sequence_kernel(state, ns, sites, codes, spin_val):
    """
    Applies a sequence of Pauli operators to an integer state.
    codes: 0=X, 1=Y, 2=Z
    """
    coeff = 1.0 + 0.0j
    curr  = state
    n     = len(codes)
    
    for i in range(n):
        c = codes[i]
        s = sites[i]
        
        if c == 0: # X
            curr, cf = _sigma_x_core(curr, ns, (s,), spin_val)
            coeff   *= cf
        elif c == 1: # Y
            curr, cf = _sigma_y_core(curr, ns, (s,), spin_val)
            coeff   *= cf
        elif c == 2: # Z
            curr, cf = _sigma_z_core(curr, ns, (s,), spin_val)
            coeff   *= cf
    
    return (curr,), (coeff,)

@numba.njit(nogil=True)
def _apply_pauli_sequence_kernel_np(state, sites, codes, spin_val):
    """
    Applies a sequence of Pauli operators to an array of state vectors.
    codes: 0=X, 1=Y, 2=Z
    """
    coeff = 1.0 + 0.0j
    curr  = state.copy()
    n     = len(codes)
    for i in range(n):
        c = codes[i]
        s = sites[i]
        
        if c == 0: # X
            (curr,), (cf,) = sigma_x_np(curr, sites=(s,), spin_value=spin_val)
            coeff   *= cf
        elif c == 1: # Y
            (curr,), (cf,) = sigma_y_np(curr, sites=(s,), spin_value=spin_val)
            coeff   *= cf
        elif c == 2: # Z
            (curr,), (cf,) = sigma_z_np(curr, sites=(s,), spin_value=spin_val)
            coeff   *= cf
    return (curr,), (coeff,)

def pauli_string(op_codes, op_sites, ns: int, return_op: bool = False, type_act: Optional[Union[str, OperatorTypeActing]] = None, code: Optional[int] = None) -> Union[Callable, Operator]:
    """
    Apply a general string of Pauli operators O_n ... O_1 to vecs.
    
    Requires that each O_i is one of {X, Y, Z}.
    
    Parameters
    ----------
    op_codes : list[str or int]
        List of operator codes ('X', 'Y', 'Z') or their integer mappings.
    op_sites : list[int]
        List of site indices where the operators act.
    ns : int
        Number of sites in the Hilbert space.
    return_op : bool, optional
        If True, returns an Operator object.
    type_act : str or OperatorTypeActing, optional
        The type of acting for the operator. Defaults to Global if sites provided, else Correlation.
    """
    
    # map the operator codes to Pauli application kernel
    op_codes_map    = { 'X': 0, 'Y': 1, 'Z': 2 }
    op_codes_out    = []
    for c in op_codes:
        if isinstance(c, str):
            code_upper = c.upper()
            if code_upper in op_codes_map:
                op_codes_out.append(op_codes_map[code_upper])
            else:
                raise ValueError(f"Invalid Pauli operator code: {c}. Must be one of 'X', 'Y', 'Z'.")
        elif isinstance(c, int):
            if c in (0, 1, 2):
                op_codes_out.append(c)
            else:
                raise ValueError(f"Invalid Pauli operator integer code: {c}. Must be 0 (X), 1 (Y), or 2 (Z).")
        else:
            raise TypeError(f"Operator code must be str or int. Got: {type(c)}")
    
    # Prepare JIT arguments
    op_codes_arr = np.array(op_codes_out, dtype=np.int32)
    op_sites_arr = np.array(op_sites, dtype=np.int32) if op_sites is not None else np.array([], dtype=np.int32)
    spin_val     = _SPIN
    
    if not return_op:
        return _apply_pauli_sequence_kernel, (ns, op_sites_arr, op_codes_arr, spin_val)
    else:
        # Determine acting type
        if type_act is None:
            type_act = OperatorTypeActing.Global if op_sites is not None else OperatorTypeActing.Correlation
        elif isinstance(type_act, str):
            type_act = OperatorTypeActing.from_string(type_act)

        return create_operator(
            type_act    = type_act,
            op_func_int = _apply_pauli_sequence_kernel,
            op_func_np  = _apply_pauli_sequence_kernel_np,
            op_func_jnp = None, #!TODO JAX
            ns          = ns,
            name        = 'Pauli:' + ','.join(f'S_{c}' for c in op_codes),
            modifies    = True,
            sites       = op_sites if OperatorTypeActing.is_type_global(type_act) else None,
            extra_args  = (op_codes_arr, spin_val),
            code        = code,
        )

# -----------------------------------------------------------------------------
#! Most common mixtures
# -----------------------------------------------------------------------------

#? Int

def create_sigma_mixed_int(op1_fun: Callable, op2_fun: Callable):
    r'''
    Create a mixed sigma operator function for integer states.
    Applies two spin operators in sequence on an integer-represented quantum state.
    
    Parameters
    ----------
    op1_fun : Callable
        The first spin operator function to apply.
    op2_fun : Callable
        The second spin operator function to apply.
    Returns
    -------
    Callable
        A function that applies the mixed sigma operators on integer states.
    '''
    
    @numba.njit(inline='always')
    def sigma_mixed_int_np(state, ns, sites, spin=True, spin_value=0.5):
        
        if sites is None or len(sites) < 2:
            s_dumb, c_dumb = op1_fun(state, ns, sites, spin, spin_value)
            return s_dumb, (complex(c_dumb[0]),)

        # Apply Op 1
        s_arr1, c_arr1  = op1_fun(state, ns, [sites[0]], spin, spin_value)
        coeff1          = c_arr1[0]
        
        if coeff1 == 0.0 or (coeff1 == 0.0 + 0.0j):
            #! Force cast to complex so this branch matches the other branch
            #! even if op1 is purely real (like Sigma X or Z)
            return s_arr1, (0.0j,)
            
        s_int           = s_arr1[0]
        s_arr2, c_arr2  = op2_fun(s_int, ns, [sites[1]], spin, spin_value)
        
        # Explicitly cast both to complex before multiplying to be 100% safe
        return s_arr2, (complex(coeff1) * complex(c_arr2[0]),)
    
    return sigma_mixed_int_np

def create_sigma_mixed_int_core(op1_fun: Callable, op2_fun: Callable):
    r'''
    Create a mixed sigma operator function for integer states.
    Applies two spin operators in sequence on an integer-represented quantum state.
    
    Parameters
    ----------
    op1_fun : Callable
        The first spin operator function to apply.
    op2_fun : Callable
        The second spin operator function to apply.
    Returns
    -------
    Callable
        A function that applies the mixed sigma operators on integer states.
    '''
    
    @numba.njit(inline='always')
    def sigma_mixed_int_core(state, ns, sites):
        
        if sites is None or len(sites) < 2:
            s_dumb, c_dumb = op1_fun(state, ns, sites)
            return s_dumb, np.complex128(c_dumb)

        # Apply Op 1
        s_arr1, c_arr1  = op1_fun(state, ns, [sites[0]])
        
        if c_arr1 == 0.0 or (c_arr1 == 0.0 + 0.0j):
            #! Force cast to complex so this branch matches the other branch
            #! even if op1 is purely real (like Sigma X or Z)
            return s_arr1, np.complex128(c_arr1) * 0.0
            
        s_int           = s_arr1
        s_arr2, c_arr2  = op2_fun(s_int, ns, [sites[1]])
        # Explicitly cast both to complex before multiplying to be 100% safe
        return s_arr2, np.complex128(c_arr1) * np.complex128(c_arr2)
    
    return sigma_mixed_int_core

def create_sigma_mixed_np(op1_fun: Callable, op2_fun: Callable):
    @numba.njit
    def sigma_mixed_np(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        if sites is None or len(sites) < 2:
            s_dumb, c_dumb = op1_fun(state, sites, spin, spin_value)
            return s_dumb, c_dumb * 0.0
        
        s_arr1, c_arr1  = op1_fun(state, [sites[0]], spin, spin_value)
        coeff1          = c_arr1[0] if c_arr1.shape[0] == 1 else c_arr1
        if coeff1 == 0.0 or (coeff1 == 0.0 + 0.0j):
            return s_arr1, c_arr1 * 0.0
        s_arr2, c_arr2  = op2_fun(s_arr1, [sites[1]], spin, spin_value)
        return s_arr2, c_arr1 * c_arr2
    return sigma_mixed_np

def create_sigma_mixed_jnp(op1_fun: Callable, op2_fun: Callable):
    if not JAX_AVAILABLE:
        def _stub(*args, **kwargs):
            raise RuntimeError("JAX not available for sigma mixed operator.")
        return _stub

    @jax.jit
    def sigma_mixed_jnp(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        if (sites is None) or (len(sites) < 2):
            raise ValueError("sigma_mixed_jnp requires at least two sites.")
        
        s_arr1, c_arr1  = op1_fun(state, [sites[0]], spin, spin_value)
        coeff1          = c_arr1[0]
        def _apply_second(s_arr1, c_arr1):
            s_int           = s_arr1[0]
            s_arr2, c_arr2  = op2_fun(s_int, [sites[1]], spin, spin_value)
            return s_arr2, c_arr1 * c_arr2
        
        # Define the target complex type (usually complex128 for float64 systems)
        # You can also use c_arr1.dtype if it's already complex, but explicit is safer here.
        target_dtype = jnp.complex128

        # Branch: if first coeff zero skip second op
        return jax.lax.cond(
            (coeff1 == 0.0) | (coeff1 == 0.0 + 0.0j),
            # True Branch: Return zero, explicitly cast to complex
            lambda _: (s_arr1, (c_arr1 * 0.0).astype(target_dtype)),
            # False Branch: Apply second op, explicitly cast result to complex
            lambda _: (lambda res: (res[0], res[1].astype(target_dtype)))(_apply_second(s_arr1, c_arr1)),
            operand = None
        )
        
    return sigma_mixed_jnp

sigma_xy_mixed_int_np   = create_sigma_mixed_int(sigma_x_int_np, sigma_y_int_np)
sigma_yx_mixed_int_np   = create_sigma_mixed_int(sigma_y_int_np, sigma_x_int_np)
sigma_yz_mixed_int_np   = create_sigma_mixed_int(sigma_y_int_np, sigma_z_int_np)
sigma_zx_mixed_int_np   = create_sigma_mixed_int(sigma_z_int_np, sigma_x_int_np)
sigma_xz_mixed_int_np   = create_sigma_mixed_int(sigma_x_int_np, sigma_z_int_np)
sigma_zy_mixed_int_np   = create_sigma_mixed_int(sigma_z_int_np, sigma_y_int_np)

sigma_xy_mixed_int_core = create_sigma_mixed_int_core(_sigma_x_core, _sigma_y_core)
sigma_yx_mixed_int_core = create_sigma_mixed_int_core(_sigma_y_core, _sigma_x_core)
sigma_yz_mixed_int_core = create_sigma_mixed_int_core(_sigma_y_core, _sigma_z_core)
sigma_zx_mixed_int_core = create_sigma_mixed_int_core(_sigma_z_core, _sigma_x_core)
sigma_xz_mixed_int_core = create_sigma_mixed_int_core(_sigma_x_core, _sigma_z_core)
sigma_zy_mixed_int_core = create_sigma_mixed_int_core(_sigma_z_core, _sigma_y_core)

#? NP
sigma_xy_mixed_np       = create_sigma_mixed_np(sigma_x_np, sigma_y_np)
sigma_yx_mixed_np       = create_sigma_mixed_np(sigma_y_np, sigma_x_np)
sigma_yz_mixed_np       = create_sigma_mixed_np(sigma_y_np, sigma_z_np)
sigma_zx_mixed_np       = create_sigma_mixed_np(sigma_z_np, sigma_x_np)
sigma_xz_mixed_np       = create_sigma_mixed_np(sigma_x_np, sigma_z_np)
sigma_zy_mixed_np       = create_sigma_mixed_np(sigma_z_np, sigma_y_np)

#? JNP
sigma_xy_mixed_jnp      = create_sigma_mixed_jnp(sigma_x_jnp, sigma_y_jnp)
sigma_yx_mixed_jnp      = create_sigma_mixed_jnp(sigma_y_jnp, sigma_x_jnp)
sigma_yz_mixed_jnp      = create_sigma_mixed_jnp(sigma_y_jnp, sigma_z_jnp)
sigma_zx_mixed_jnp      = create_sigma_mixed_jnp(sigma_z_jnp, sigma_x_jnp)
sigma_xz_mixed_jnp      = create_sigma_mixed_jnp(sigma_x_jnp, sigma_z_jnp)
sigma_zy_mixed_jnp      = create_sigma_mixed_jnp(sigma_z_jnp, sigma_y_jnp)

def make_sigma_mixed(name, lattice=None, ns=None, type_act='global', sites=None, code: Optional[int] = None) -> Operator:
    r''' Factory for mixed sigma operators (e.g., \sigma _x \sigma _y). '''
    
    if sites is not None and len(sites) != 2:
        raise ValueError("Mixed sigma operators require exactly two sites.")
    
    # Map name to Pauli characters
    name_map = {
        'xy': ['X', 'Y'], 'yx': ['Y', 'X'],
        'yz': ['Y', 'Z'], 'zy': ['Z', 'Y'],
        'zx': ['Z', 'X'], 'xz': ['X', 'Z']
    }
    
    if name not in name_map:
        raise ValueError(f"Unknown mixed sigma operator name: {name}")
    
    pauli_ops = name_map[name]
    ns_val    = ns or (lattice.ns if lattice else None)
    
    if OperatorTypeActing.is_type_local(type_act):
        raise ValueError("Mixed sigma operators do not support local acting type.")
    
    # Determine type of acting and default code
    if sites is not None:
        type_act_val = OperatorTypeActing.Global
        if code is None:
            code_attr   = f"sig_{name}_global"
            code        = getattr(SPIN_LOOKUP_CODES, code_attr, None)
    else:
        type_act_val = OperatorTypeActing.Correlation
        if code is None:
            code_attr   = f"sig_{name}_corr"
            code        = getattr(SPIN_LOOKUP_CODES, code_attr, None)
        
    return pauli_string(pauli_ops, sites, ns_val, return_op=True, type_act=type_act_val, code=code)

def sig_xy(lattice=None, ns=None, type_act='global', sites=None, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN) -> Operator:
    r''' Factory for the \sigma _x \sigma _y operator. '''
    return make_sigma_mixed('xy', lattice, ns, type_act=type_act, sites=sites)
def sig_yx(lattice=None, ns=None, type_act='global', sites=None, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN) -> Operator:
    r''' Factory for the \sigma _y \sigma _x operator. '''
    return make_sigma_mixed('yx', lattice, ns, type_act=type_act, sites=sites)
def sig_yz(lattice=None, ns=None, type_act='global', sites=None, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN) -> Operator:
    r''' Factory for the \sigma _y \sigma _z operator. '''
    return make_sigma_mixed('yz', lattice, ns, type_act=type_act, sites=sites)
def sig_zy(lattice=None, ns=None, type_act='global', sites=None, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN) -> Operator:
    r''' Factory for the \sigma _z \sigma _y operator. '''
    return make_sigma_mixed('zy', lattice, ns, type_act=type_act, sites=sites)
def sig_zx(lattice=None, ns=None, type_act='global', sites=None, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN) -> Operator:
    r''' Factory for the \sigma _z \sigma _x operator. '''
    return make_sigma_mixed('zx', lattice, ns, type_act=type_act, sites=sites)
def sig_xz(lattice=None, ns=None, type_act='global', sites=None, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN) -> Operator:
    r''' Factory for the \sigma _x \sigma _z operator. '''
    return make_sigma_mixed('xz', lattice, ns, type_act=type_act, sites=sites)

# -----------------------------------------------------------------------------
#! N-SITE CORRELATOR FACTORY (for arbitrary multi-site operators)
# -----------------------------------------------------------------------------

def make_nsite_correlator(pauli_ops: List[str], ns: int, sites: Optional[List[int]] = None, lattice: Optional[Lattice] = None, name: Optional[str] = None,) -> Operator:
    r"""
    Factory for creating arbitrary n-site spin correlators.
    
    Creates a numba-compatible operator that applies a sequence of Pauli operators
    at specified sites. For example, to create a 3-site correlator \sigma _x \otimes \sigma _y \otimes \sigma _z
    acting on sites [0, 1, 2], you would call:
    
    Parameters
    ----------
    pauli_ops : List[str]
        List of Pauli operator names: 'x', 'y', 'z', 'p' (plus), 'm' (minus).
        The order corresponds to the order of sites.
    ns : int
        Number of sites in the system.
    sites : List[int], optional
        Sites where each operator acts. Must have same length as pauli_ops.
        If None, uses sites [0, 1, 2, ...] up to len(pauli_ops).
    lattice : Lattice, optional
        The lattice object (can provide ns if not given directly).
    name : str, optional
        Custom name for the operator. If None, auto-generated from operators.
        
    Returns
    -------
    Operator
        A numba-compatible operator for the n-site correlator.
        
    Examples
    --------
    >>> # Create \sigma_x \otimes \sigma_y \otimes \sigma_z on sites 0, 1, 2
    >>> xyz_op = make_nsite_correlator(['x', 'y', 'z'], ns=4, sites=[0, 1, 2])
    >>> hamil.add(xyz_op, multiplier=1.0, sites=[0, 1, 2])
    
    >>> # Create 4-site correlator \sigma_z \otimes \sigma_z \otimes \sigma_z \otimes \sigma_z
    >>> zzzz_op = make_nsite_correlator(['z', 'z', 'z', 'z'], ns=8, sites=[0, 2, 4, 6])
    >>> hamil.add(zzzz_op, multiplier=0.5, sites=[0, 2, 4, 6])
    
    Notes
    -----
    This operator is registered as a custom operator in the Hamiltonian's
    instruction registry. It will be slower than predefined operators but
    maintains full numba JIT compatibility.
    """
    if ns is None:
        if lattice is not None:
            ns = lattice.ns
        else:
            raise ValueError("Either 'ns' or 'lattice' must be provided.")
    
    n_ops = len(pauli_ops)
    
    if sites is None:
        sites = list(range(n_ops))
    
    if len(sites) != n_ops:
        raise ValueError(f"Number of sites ({len(sites)}) must match number of operators ({n_ops}).")
    
    # Map operator names to core functions
    op_map = {
        'x': _sigma_x_core,
        'y': _sigma_y_core,
        'z': _sigma_z_core,
        'p': sigma_plus_int_np,
        'm': sigma_minus_int_np,
    }
    
    # Validate operator names
    for op in pauli_ops:
        if op.lower() not in op_map:
            raise ValueError(f"Unknown Pauli operator: '{op}'. Use 'x', 'y', 'z', 'p', or 'm'.")
    
    # Generate name if not provided
    if name is None:
        name = 'S' + ''.join(pauli_ops)
    
    # Create the JIT-compiled function for this specific operator composition
    # Convert operator names to integer codes for numba compatibility
    OP_X, OP_Y, OP_Z, OP_P, OP_M    = 0, 1, 2, 3, 4
    op_name_to_code                 = {'x': OP_X, 'y': OP_Y, 'z': OP_Z, 'p': OP_P, 'm': OP_M}
    
    # Create numpy arrays for the operator codes and sites (numba-friendly)
    op_codes_arr                    = np.array([op_name_to_code[op.lower()] for op in pauli_ops], dtype=np.int32)
    sites_arr                       = np.array(sites, dtype=np.int32)
    n_ops_val                       = n_ops
    
    # Create the numba-compatible integer function
    @numba.njit
    def nsite_correlator_int(state: int, ns_val: int, sites_tuple: tuple):
        """
        Apply n-site correlator: product of Pauli operators at specified sites.
        Returns (new_state, coefficient).
        """
        current_state = state
        current_coeff = 1.0 + 0.0j
        
        for i in range(n_ops_val):
            op_code = op_codes_arr[i]
            # Use provided sites if available, otherwise use stored sites
            if i < len(sites_tuple):
                site = sites_tuple[i]
            else:
                site = sites_arr[i]
            site_tuple = (site,)
            
            if op_code == 0:  # x
                new_s, new_c = _sigma_x_core(current_state, ns_val, sites=site_tuple)
            elif op_code == 1:  # y
                new_s, new_c = _sigma_y_core(current_state, ns_val, sites=site_tuple)
            elif op_code == 2:  # z
                new_s, new_c = _sigma_z_core(current_state, ns_val, sites=site_tuple)
            elif op_code == 3:  # p (plus)
                s_arr, c_arr = sigma_plus_int_np(current_state, ns_val, sites=site_tuple)
                new_s, new_c = s_arr[0], c_arr[0]
            elif op_code == 4:  # m (minus)
                s_arr, c_arr = sigma_minus_int_np(current_state, ns_val, sites=site_tuple)
                new_s, new_c = s_arr[0], c_arr[0]
            else:
                # Should not happen
                new_s, new_c = current_state, 1.0 + 0.0j
            
            current_state = new_s
            current_coeff = current_coeff * new_c
            
            # Early exit if coefficient becomes zero
            if abs(current_coeff) < 1e-15:
                return state, 0.0 + 0.0j
        
        return current_state, current_coeff
    
    # Determine if the operator is diagonal (all z operators)
    is_diagonal                     = all(op.lower() == 'z' for op in pauli_ops)
    
    # Create the Operator object
    return create_operator(
        type_act    = OperatorTypeActing.Correlation,
        op_func_int = nsite_correlator_int,
        op_func_np  = None,  # No numpy version for custom correlators (use int version)
        op_func_jnp = None,  # No JAX version for custom correlators
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        name        = name,
        modifies    = not is_diagonal,
        code        = None,  # Will be assigned a custom code when added to Hamiltonian
    )

def sig_xyz(ns: int, sites: Optional[List[int]] = None, lattice: Optional[Lattice] = None) -> Operator:
    r"""Factory for the 3-site \sigma_x \otimes \sigma_y \otimes \sigma_z correlator."""
    return make_nsite_correlator(['x', 'y', 'z'], ns=ns, sites=sites, lattice=lattice, name='SxSySz')

# -----------------------------------------------------------------------------
#? COMPOSITION FUNCTION BASED ON THE CODES!
# -----------------------------------------------------------------------------

def sigma_composition_integer(is_complex: bool, only_apply: bool = False) -> Callable:
    r'''
    Creates a sigma operator composition kernel based on a list of operator codes. 
    It supports both real and complex coefficients.
    
    Parameters
    ----------
    is_complex : bool
        Whether the operator coefficients are complex.
    only_apply : bool, optional
        If True, only applies the operators without combining coefficients.
    '''
    dtype           = np.complex128 if is_complex else np.float64
    
    @numba.njit(nogil=True) # compile separately for real/complex
    def sigma_operator_composition_single_op(state: int, code: int, site1: int, site2: int, ns: int):
        ''' Applies a single operator based on the code. '''
        
        current_s   = state
        current_c   = dtype(1.0 + 0.0j)
        is_diagonal = False
        sites_1     = (site1,)
        sites_2     = (site1, site2)
        
        #! 1) GLOBAL OPERATORS
        if code <= 10:
            raise ValueError("Global operators not allowed in composition function.")
        
        #! 2) LOCAL OPERATORS
        if code == SPIN_LOOKUP_CODES.sig_x_local:
            s_arr, c_arr    = _sigma_x_core(state, ns, sites=sites_1)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_y_local:
            s_arr, c_arr    = _sigma_y_core(state, ns, sites=sites_1)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_z_local:
            s_arr, c_arr    = _sigma_z_core(state, ns, sites=sites_1)
            is_diagonal     = True
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_p_local:
            s_arr, c_arr    = sigma_plus_int_np(state, ns, sites=sites_1)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        elif code == SPIN_LOOKUP_CODES.sig_m_local:
            s_arr, c_arr    = sigma_minus_int_np(state, ns, sites=sites_1)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        #! 3) CORRELATION OPERATORS
        elif code == SPIN_LOOKUP_CODES.sig_x_corr:
            s_arr, c_arr    = _sigma_x_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_y_corr:
            s_arr, c_arr    = _sigma_y_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_z_corr:
            s_arr, c_arr    = _sigma_z_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
            is_diagonal     = True
        elif code == SPIN_LOOKUP_CODES.sig_p_corr:
            s_arr, c_arr    = sigma_plus_int_np(state, ns, sites=sites_2)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        elif code == SPIN_LOOKUP_CODES.sig_m_corr:
            s_arr, c_arr    = sigma_minus_int_np(state, ns, sites=sites_2)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        elif code == SPIN_LOOKUP_CODES.sig_xy_corr:
            s_arr, c_arr    = sigma_xy_mixed_int_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_yx_corr:
            s_arr, c_arr    = sigma_yx_mixed_int_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_xz_corr:
            s_arr, c_arr    = sigma_xz_mixed_int_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_zx_corr:
            s_arr, c_arr    = sigma_zx_mixed_int_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_yz_corr:
            s_arr, c_arr    = sigma_yz_mixed_int_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_zy_corr:
            s_arr, c_arr    = sigma_zy_mixed_int_core(state, ns, sites=sites_2)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == SPIN_LOOKUP_CODES.sig_pm_corr:
            s_arr, c_arr    = sigma_pm_int_np(state, ns, sites=sites_2)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        elif code == SPIN_LOOKUP_CODES.sig_mp_corr:
            s_arr, c_arr    = sigma_mp_int_np(state, ns, sites=sites_2)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        else:
            pass
        
        return current_s, current_c, is_diagonal
    
    if only_apply:
        return sigma_operator_composition_single_op
    
    @numba.njit(nogil=True) # compile separately for real/complex
    def sigma_operator_composition_int(state: int, nops: int, codes, sites, coeffs, ns: int, out_states, out_vals):
        ''' Creates a combined operator based on this list of codes. '''
        ptr         = 0
        diag_sum    = 0.0

        for k in range(nops):
            code                                = codes[k]
            coeff                               = coeffs[k]
            s1                                  = sites[k, 0]
            s2                                  = sites[k, 1]
            current_s, current_c, is_diagonal   = sigma_operator_composition_single_op(state, code, s1, s2, ns)
            
            # Multiply by coefficient
            final_val = current_c * coeff
            if abs(final_val) < 1e-15:  continue
            
            # Accumulate results
            if is_diagonal:
                diag_sum       += final_val
            else:
                out_states[ptr] = current_s
                out_vals[ptr]   = final_val
                ptr            += 1
        
        # Save Diagonal
        if abs(diag_sum) > 1e-15:
            out_states[ptr] = state
            out_vals[ptr]   = diag_sum
            ptr            += 1
        
        return out_states[:ptr], out_vals[:ptr]
    
    return sigma_operator_composition_int

def sigma_composition_with_custom(is_complex: bool, custom_op_funcs: Dict[int, Callable], custom_op_arity: Dict[int, int]):
    r"""
    Creates a sigma operator composition kernel that supports both predefined and custom operators.
    
    This function creates a hybrid composition kernel that:
    1. Uses fast predefined lookups for standard operators (codes < 1000)
    2. Falls back to custom operator functions for codes >= 1000
    
    Parameters
    ----------
    is_complex : bool
        Whether to use complex coefficients.
    custom_op_funcs : Dict[int, Callable]
        Dictionary mapping custom operator codes to their numba-compiled functions.
        Each function should have signature: (state: int, ns: int, sites: tuple) -> (new_state, coeff)
    custom_op_arity : Dict[int, int]
        Dictionary mapping custom operator codes to their arity (number of sites).
        
    Returns
    -------
    Callable
        A numba-compiled composition function.
        
    Notes
    -----
    Custom operators support arbitrary arity (not limited to 3 sites).
    The sites tuple passed to custom operators will have length equal to the operator's arity.
    
    Example
    -------
    >>> # Register a 3-site correlator: \sigma_x \otimes \sigma_y \otimes \sigma_z
    >>> @numba.njit
    ... def sig_xyz_int(state, ns, sites):
    ...     s1, c1 = _sigma_x_core(state, ns, sites=(sites[0],))
    ...     s2, c2 = _sigma_y_core(s1, ns, sites=(sites[1],))
    ...     s3, c3 = _sigma_z_core(s2, ns, sites=(sites[2],))
    ...     return s3, c1 * c2 * c3
    >>> 
    >>> custom_funcs = {1000: sig_xyz_int}
    >>> custom_arity = {1000: 3}
    >>> comp_func = sigma_composition_with_custom(True, custom_funcs, custom_arity)
    """
    
    dtype       = np.complex128 if is_complex else np.float64
    n_custom    = len(custom_op_funcs)
    
    if n_custom == 0:
        # No custom operators - just use the standard composition
        return sigma_composition_integer(is_complex)
    
    # Create arrays of custom operator info for numba
    from numba.typed import List as NList
    custom_codes_arr    = np.array(sorted(custom_op_funcs.keys()), dtype=np.int64)
    custom_arity_arr    = np.array([custom_op_arity.get(code, 1) for code in custom_codes_arr], dtype=np.int32)
    
    # Create typed list of custom functions
    custom_funcs_list   = NList()
    for code in custom_codes_arr:
        custom_funcs_list.append(custom_op_funcs[code])
    
    @numba.njit(nogil=True)
    def find_custom_idx(code, codes_arr):
        """Linear search for custom operator index."""
        for i in range(len(codes_arr)):
            if codes_arr[i] == code:
                return i
        return -1
    
    apply_predefined_operator = sigma_composition_integer(is_complex, only_apply=True)
    
    @numba.njit(nogil=True)
    def sigma_operator_composition_with_custom_int(state: int, nops: int, codes, sites, coeffs, ns: int, out_states, out_vals):
        """
        Composition function that handles both predefined and custom operators.
        Supports arbitrary arity for custom operators.
        """
        ptr         = 0
        diag_sum    = 0.0

        for k in range(nops):
            code        = codes[k]
            coeff       = coeffs[k]
            current_s   = state
            current_c   = 1.0 + 0.0j
            is_diagonal = False
            handled     = False
            custom_app  = code >= CUSTOM_OP_BASE
            
            # Check for custom operators first (codes >= CUSTOM_OP_BASE)
            if custom_app:
                idx = find_custom_idx(code, custom_codes_arr)
                if idx >= 0:
                    custom_func = custom_funcs_list[idx]
                    arity       = custom_arity_arr[idx]
                    
                    # Build sites tuple dynamically based on arity
                    # Numba requires static tuple construction, so we handle common cases
                    if arity == 1:
                        sites_tuple = (sites[k, 0],)
                    elif arity == 2:
                        sites_tuple = (sites[k, 0], sites[k, 1])
                    elif arity == 3:
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2])
                    elif arity == 4:
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2], sites[k, 3])
                    elif arity == 5:
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2], sites[k, 3], sites[k, 4])
                    elif arity == 6:
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2], sites[k, 3], sites[k, 4], sites[k, 5])
                    elif arity == 7:
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2], sites[k, 3], sites[k, 4], sites[k, 5], sites[k, 6])
                    elif arity == 8:
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2], sites[k, 3], sites[k, 4], sites[k, 5], sites[k, 6], sites[k, 7])
                    else:
                        # For very high arity, fall back to max supported
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2], sites[k, 3], sites[k, 4], sites[k, 5], sites[k, 6], sites[k, 7])
                    
                    s_arr, c_arr    = custom_func(state, ns, sites_tuple)
                    current_s       = s_arr
                    current_c       = dtype(c_arr)
                    handled         = True
            else:
                # Try predefined operators
                current_s, current_c, is_diagonal   = apply_predefined_operator(state, code, sites[k, 0], sites[k, 1], coeff, ns)
                handled                             = True
                
            if not handled:
                continue  # Unknown operator
            
            # Multiply by coefficient
            final_val = current_c * coeff
            if abs(final_val) < 1e-15:
                continue
            
            # Accumulate results
            if is_diagonal:
                diag_sum       += final_val
            else:
                out_states[ptr] = current_s
                out_vals[ptr]   = final_val
                ptr            += 1
        
        # Save diagonal contribution
        if abs(diag_sum) > 1e-15:
            out_states[ptr]     = state
            out_vals[ptr]       = diag_sum
            ptr                += 1
        
        return out_states[:ptr], out_vals[:ptr]
    
    return sigma_operator_composition_with_custom_int

# -----------------------------------------------------------------------------
#! Plaquette Operators
# -----------------------------------------------------------------------------

def spin_plaquette(plaquette: np.ndarray, lattice: 'Lattice', *, bond_to_op = None, return_op: bool = False):
    """
    Apply the plaquette (flux) operator W_p to vecs using a single optimized kernel.
    This avoids intermediate projections when using symmetry-reduced bases.

    Parameters
    ----------
    plaquette : list[int]
        site indices in CCW order.
    out : ndarray, shape (Nh, ns)
        Workspace, overwritten.
    bond_to_op : dict, optional
        Mapping from bond types to spin operators.
        Default is {0: sig_z, 1: sig_y, 2: sig_x}.
    return_op : bool, optional
        If True, returns the JIT function and arguments instead of applying.
    """
    
    # Default bond_to_op if None
    if bond_to_op is None:
        bond_to_op      = { 0: 'Z', 1: 'Y', 2: 'X' }

    # Pre-calculate the sequence of operations
    P                   = len(plaquette)
    
    op_codes            = []
    op_sites            = []
    op_bonds            = []

    for k in range(P):
        site    = plaquette[k]
        nxt     = plaquette[(k + 1) % P]

        # Get the bond type from current site to next site
        bond    = lattice.bond_type(site, nxt)

        if bond < 0:
            raise RuntimeError(f"Invalid plaquette bond from site {site} to {nxt}: bond_type={bond}")

        # Apply operator corresponding to this bond type
        op      = bond_to_op[bond]
        
        op_codes.append(op)
        op_sites.append(site)
        op_bonds.append(missing)

    return pauli_string(op_codes, op_sites, lattice.ns, return_op=return_op), op_codes

def spin_plaquettes(list_plaquettes : list, lattice: 'Lattice', *, bond_to_op = None, return_op: bool = False):
    """
    Apply a product of multiple plaquette operators W_{p1} W_{p2} ... to vecs 
    using a single optimized kernel without intermediate projections.

    Parameters
    ----------
    vecs : ndarray
        Input state vectors.
    list_plaquettes : list of list[int]
        List of plaquettes (each is a list of sites). 
        The operators are applied in the order they appear in the list.
    out : ndarray
        Workspace.
    """
    
    if bond_to_op is None:
        bond_to_op  = {0: 'Z', 1: 'Y', 2: 'X'}

    op_codes        = []
    op_sites        = []

    # Process each plaquette in the list
    for plaquette in list_plaquettes:
        P = len(plaquette)
        for k in range(P):
            site    = plaquette[k]
            nxt     = plaquette[(k + 1) % P]

            # Get the bond type from current site to next site
            bond    = lattice.bond_type(site, nxt)

            if bond < 0:
                raise RuntimeError(f"Invalid plaquette bond from site {site} to {nxt}: bond_type={bond}")

            # Apply operator corresponding to this bond type
            op      = bond_to_op[bond]
            
            op_codes.append(op)
            op_sites.append(site)
            
    return pauli_string(op_codes, op_sites, lattice.ns, return_op=return_op)

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

_register_catalog_entries()

# -----------------------------------------------------------------------------
#! EOF 
# -----------------------------------------------------------------------------