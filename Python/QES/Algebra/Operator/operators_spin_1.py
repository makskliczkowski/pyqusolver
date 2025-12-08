"""
This module implements spin-1 operators for quantum systems.
It includes functions for S_x, S_y, S_z, S_plus (raising), S_minus (lowering),
and their products for spin-1 (S=1) systems.

Spin-1 systems have local dimension 3 with states |+1⟩, |0⟩, |-1⟩.
The representation uses integer encoding where each site uses 2 bits (trits):
    00 (0) -> |+1⟩  (m = +1)
    01 (1) -> | 0⟩  (m =  0)  
    10 (2) -> |-1⟩  (m = -1)
    11 (3) -> invalid (not used)

Spin-1 Matrices (in |+1⟩, |0⟩, |-1⟩ basis):
    
    S_x = 1/sqrt2 * |0  1  0|      S_y = 1/sqrt2 *  |0  -i  0|      S_z =   |1  0  0|
                    |1  0  1|                       |i   0 -i|              |0  0  0|
                    |0  1  0|                       |0   i  0|              |0  0 -1|
    
    S_+ = sqrt2 *   |0  1  0|        S_- = sqrt2 *  |0  0  0|
                    |0  0  1|                       |1  0  0|
                    |0  0  0|                       |0  1  0|
--------------------------------------------------------------
File        : QES/Algebra/Operator/operators_spin_1.py
Description : Spin-1 operator implementations
Author      : Maksymilian Kliczkowski
Date        : December 2025
--------------------------------------------------------------
"""

import  os 
import  numba
import  numpy           as np
from    dataclasses     import dataclass
from    typing          import Dict, List, Union, Optional, Callable, Tuple
from    enum            import IntEnum

################################################################################

try:
    from QES.Algebra.Operator.operator          import Operator, OperatorTypeActing, create_operator, ensure_operator_output_shape_numba
    from QES.Algebra.Operator.catalog           import register_local_operator
    from QES.Algebra.Operator.special_operator  import CUSTOM_OP_BASE
    from QES.Algebra.Hilbert.hilbert_local      import LocalOpKernels, LocalSpaceTypes
except ImportError as e:
    raise ImportError("Failed to import required modules. Ensure that the QES package is correctly installed.") from e

################################################################################

try:
    from QES.general_python.lattices.lattice    import Lattice
except ImportError as e:
    raise ImportError("Failed to import required QES general Python modules.") from e

################################################################################

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    jnp             = None
    jax             = None
    JAX_AVAILABLE   = False

################################################################################
#! JAX-accelerated operators (imported from jax/operators_spin_1)
################################################################################

if JAX_AVAILABLE:
    try:
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_z_int_jnp, spin1_z_jnp, spin1_z_inv_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_plus_int_jnp, spin1_plus_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_minus_int_jnp, spin1_minus_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_x_int_jnp, spin1_y_int_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_pm_int_jnp, spin1_mp_int_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_zz_int_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_squared_int_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_z_total_int_jnp, spin1_z_total_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import spin1_z2_int_jnp
        # JAX matrices
        from QES.Algebra.Operator.jax.operators_spin_1 import _S1_X_jnp, _S1_Y_jnp, _S1_Z_jnp
        from QES.Algebra.Operator.jax.operators_spin_1 import _S1_PLUS_jnp, _S1_MINUS_jnp, _S1_Z2_jnp
    except ImportError:
        # JAX module not available, operators will be None
        pass

################################################################################
#! Constants and Standard Spin-1 Matrices
################################################################################

# Spin value for S=1
_SPIN_1             = 1.0
_SQRT2              = np.sqrt(2.0)
_SQRT2_INV          = 1.0 / np.sqrt(2.0)

# Define the spin-1 matrices (3x3) in basis |+1⟩, |0⟩, |-1⟩
_S1_IDENTITY        = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=float)

_S1_X               = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=float) * _SQRT2_INV

_S1_Y               = np.array([[0, -1j, 0],
                                [1j, 0, -1j],
                                [0, 1j, 0]], dtype=complex) * _SQRT2_INV
_S1_Z               = np.array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 0, -1]], dtype=float)

_S1_PLUS            = np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=float) * _SQRT2

_S1_MINUS           = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=float) * _SQRT2

# S_z^2 for quadrupolar interactions
_S1_Z2              = np.array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 0, 1]], dtype=float)

################################################################################
#! Bit manipulation for spin-1 (using 2 bits per site)
################################################################################

# Spin-1 states encoded as 2 bits per site:
# State encoding: 0 = |+1⟩, 1 = |0⟩, 2 = |-1⟩
# Position of site i in the integer state: bits at positions 2*(ns-1-i) and 2*(ns-1-i)+1

@numba.njit(inline='always')
def _get_spin1_state(state: int, ns: int, site: int) -> int:
    """
    Get the spin-1 state (0, 1, or 2) at a given site.
    
    Parameters
    ----------
    state : int
        The full system state as an integer.
    ns : int
        Number of sites.
    site : int
        Site index (0 to ns-1).
        
    Returns
    -------
    int
        Local state: 0 = |+1⟩, 1 = |0⟩, 2 = |-1⟩
    """
    bit_pos = 2 * (ns - 1 - site)
    return (state >> bit_pos) & 0b11

@numba.njit(inline='always')
def _set_spin1_state(state: int, ns: int, site: int, local_state: int) -> int:
    """
    Set the spin-1 state at a given site.
    
    Parameters
    ----------
    state : int
        The full system state as an integer.
    ns : int
        Number of sites.
    site : int
        Site index (0 to ns-1).
    local_state : int
        New local state: 0 = |+1⟩, 1 = |0⟩, 2 = |-1⟩
        
    Returns
    -------
    int
        Updated state.
        
    Example
    -------
    >>> new_state = _set_spin1_state(state, ns, site, local_state)
    """
    bit_pos     = 2 * (ns - 1 - site)
    # Clear the 2 bits at this position
    # Set the new value
    mask        = ~(0b11 << bit_pos)
    state       = state & mask
    state       = state | (local_state << bit_pos)
    return state

@numba.njit(inline='always')
def _spin1_magnetization(local_state: int) -> float:
    """Get the m value for a local spin-1 state."""
    # 0 -> +1, 1 -> 0, 2 -> -1
    return 1.0 - float(local_state)

################################################################################
#! S_z operator (diagonal)
################################################################################

@numba.njit(inline='always')
def _spin1_z_core(state: int, ns: int, sites: tuple) -> Tuple[int, float]:
    """
    Core implementation of S_z for spin-1.
    S_z |m⟩ = m |m⟩
    """
    coeff = 1.0
    for site in sites:
        local_s = _get_spin1_state(state, ns, site)
        if local_s == 3:  # Invalid state
            return state, 0.0
        m = _spin1_magnetization(local_s)
        coeff *= m
    return state, coeff

@numba.njit(inline='always')
def spin1_z_int_np(state: int, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
    """
    Apply S_z operator to an integer state for spin-1 system.
    
    Parameters
    ----------
    state : int
        The state as an integer.
    ns : int
        Number of sites.
    sites : list or tuple
        Sites to apply the operator to.
    spin_value : float
        Spin value (default 1.0).
        
    Returns
    -------
    tuple
        (states_array, coefficients_array)
    """
    if isinstance(sites, list):
        sites = tuple(sites)
    s, c = _spin1_z_core(state, ns, sites)
    return (np.array([s], dtype=np.int64), np.array([c * spin_value], dtype=np.float64))

def spin1_z_int(state: int, ns: int, sites: Union[List[int], None], spin_value: float = _SPIN_1):
    """Dispatch for S_z on integer state."""
    if sites is None:
        sites = list(range(ns))
    return spin1_z_int_np(state, ns, sites, spin_value)

@numba.njit(inline='always')
def spin1_z_np(state: np.ndarray, sites: Union[List[int], tuple], spin: bool = True, spin_value: float = _SPIN_1):
    """
    Apply S_z to a NumPy array state (wavefunction representation).
    This is for future compatibility with array-based states.
    """
    # For array states, S_z is diagonal
    coeff = np.ones(1, dtype=np.float64)
    # Note: This would need proper implementation for array states
    return ensure_operator_output_shape_numba(state, coeff)

################################################################################
#! S_z^2 operator (diagonal, for quadrupolar terms)
################################################################################

@numba.njit(inline='always')
def _spin1_z2_core(state: int, ns: int, sites: tuple) -> Tuple[int, float]:
    """
    Core implementation of S_z^2 for spin-1.
    S_z^2 |m⟩ = m^2 |m⟩
    """
    coeff = 1.0
    for site in sites:
        local_s = _get_spin1_state(state, ns, site)
        if local_s == 3:  # Invalid state
            return state, 0.0
        m = _spin1_magnetization(local_s)
        coeff *= m * m
    return state, coeff

@numba.njit(inline='always')
def spin1_z2_int_np(state: int, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
    """Apply S_z^2 operator to an integer state."""
    if isinstance(sites, list):
        sites = tuple(sites)
    s, c = _spin1_z2_core(state, ns, sites)
    return (np.array([s], dtype=np.int64), np.array([c], dtype=np.float64))

################################################################################
#! S_+ (raising) operator
################################################################################

@numba.njit(inline='always')
def _spin1_plus_core(state: int, ns: int, site: int) -> Tuple[int, float]:
    """
    Core implementation of S_+ for spin-1 at a single site.
    S_+ |m⟩ = sqrt(S(S+1) - m(m+1)) |m+1⟩
    
    For S=1:
        S_+ |+1⟩ = 0
        S_+ | 0⟩ = sqrt2 |+1⟩
        S_+ |-1⟩ = sqrt2 | 0⟩
    """
    local_s = _get_spin1_state(state, ns, site)
    
    if local_s == 0:  # |+1⟩ -> cannot raise further
        return state, 0.0
    elif local_s == 1:  # |0⟩ -> |+1⟩
        new_state = _set_spin1_state(state, ns, site, 0)
        return new_state, _SQRT2
    elif local_s == 2:  # |-1⟩ -> |0⟩
        new_state = _set_spin1_state(state, ns, site, 1)
        return new_state, _SQRT2
    else:  # Invalid state
        return state, 0.0

@numba.njit(inline='always')
def spin1_plus_int_np(state: int, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
    """
    Apply S_+ operator to an integer state for spin-1 system.
    For multiple sites, applies sequentially.
    """
    if isinstance(sites, list):
        sites = tuple(sites)
    
    current_state = state
    coeff = 1.0
    
    for site in sites:
        current_state, c = _spin1_plus_core(current_state, ns, site)
        coeff *= c
        if abs(coeff) < 1e-15:
            return (np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64))
    
    return (np.array([current_state], dtype=np.int64), np.array([coeff], dtype=np.float64))

def spin1_plus_int(state: int, ns: int, sites: Union[List[int], None], spin_value: float = _SPIN_1):
    """Dispatch for S_+ on integer state."""
    if sites is None:
        sites = list(range(ns))
    return spin1_plus_int_np(state, ns, sites, spin_value)

################################################################################
#! S_- (lowering) operator
################################################################################

@numba.njit(inline='always')
def _spin1_minus_core(state: int, ns: int, site: int) -> Tuple[int, float]:
    """
    Core implementation of S_- for spin-1 at a single site.
    S_- |m⟩ = sqrt(S(S+1) - m(m-1)) |m-1⟩
    
    For S=1:
        S_- |+1⟩ = sqrt2 | 0⟩
        S_- | 0⟩ = sqrt2 |-1⟩
        S_- |-1⟩ = 0
    """
    local_s = _get_spin1_state(state, ns, site)
    
    if local_s == 0:  # |+1⟩ -> |0⟩
        new_state = _set_spin1_state(state, ns, site, 1)
        return new_state, _SQRT2
    elif local_s == 1:  # |0⟩ -> |-1⟩
        new_state = _set_spin1_state(state, ns, site, 2)
        return new_state, _SQRT2
    elif local_s == 2:  # |-1⟩ -> cannot lower further
        return state, 0.0
    else:  # Invalid state
        return state, 0.0

@numba.njit(inline='always')
def spin1_minus_int_np(state: int, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
    """
    Apply S_- operator to an integer state for spin-1 system.
    """
    if isinstance(sites, list):
        sites = tuple(sites)
    
    current_state = state
    coeff = 1.0
    
    for site in sites:
        current_state, c = _spin1_minus_core(current_state, ns, site)
        coeff *= c
        if abs(coeff) < 1e-15:
            return (np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64))
    
    return (np.array([current_state], dtype=np.int64), np.array([coeff], dtype=np.float64))

def spin1_minus_int(state: int, ns: int, sites: Union[List[int], None], spin_value: float = _SPIN_1):
    """Dispatch for S_- on integer state."""
    if sites is None:
        sites = list(range(ns))
    return spin1_minus_int_np(state, ns, sites, spin_value)

################################################################################
#! S_x operator
################################################################################

@numba.njit(inline='always')
def _spin1_x_single_site(state: int, ns: int, site: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply S_x to a single site for spin-1.
    S_x = (S_+ + S_-) / 2
    
    S_x |+1⟩ = (1/sqrt2) | 0⟩
    S_x | 0⟩ = (1/sqrt2) (|+1⟩ + |-1⟩)
    S_x |-1⟩ = (1/sqrt2) | 0⟩
    """
    local_s = _get_spin1_state(state, ns, site)
    
    if local_s == 0:  # |+1⟩ -> |0⟩
        new_state = _set_spin1_state(state, ns, site, 1)
        return np.array([new_state], dtype=np.int64), np.array([_SQRT2_INV], dtype=np.float64)
    
    elif local_s == 1:  # |0⟩ -> |+1⟩ and |-1⟩
        state_plus = _set_spin1_state(state, ns, site, 0)
        state_minus = _set_spin1_state(state, ns, site, 2)
        return (np.array([state_plus, state_minus], dtype=np.int64), 
                np.array([_SQRT2_INV, _SQRT2_INV], dtype=np.float64))
    
    elif local_s == 2:  # |-1⟩ -> |0⟩
        new_state = _set_spin1_state(state, ns, site, 1)
        return np.array([new_state], dtype=np.int64), np.array([_SQRT2_INV], dtype=np.float64)
    
    else:  # Invalid
        return np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64)

@numba.njit(inline='always')
def _spin1_x_core(state: int, ns: int, sites: tuple) -> Tuple[int, float]:
    """
    Core S_x for instruction code system (single output approximation).
    For Hamiltonian composition, we need a simpler interface.
    This applies S_+ + S_- / 2 but only returns the first result.
    
    NOTE: This is an approximation for the composition system.
    For exact treatment, use the matrix representation.
    """
    # For single site, S_x can produce 1 or 2 output states
    # The composition system expects single output, so we use S_+ contribution only
    # This is a limitation - for exact spin-1, use matrix representation
    if len(sites) == 1:
        site = sites[0]
        local_s = _get_spin1_state(state, ns, site)
        
        if local_s == 0:  # |+1⟩ -> |0⟩ (from S_-)
            new_state = _set_spin1_state(state, ns, site, 1)
            return new_state, _SQRT2_INV
        elif local_s == 1:  # |0⟩ -> average behavior
            # This is approximate - in reality produces two states
            new_state = _set_spin1_state(state, ns, site, 0)
            return new_state, _SQRT2_INV
        elif local_s == 2:  # |-1⟩ -> |0⟩ (from S_+)
            new_state = _set_spin1_state(state, ns, site, 1)
            return new_state, _SQRT2_INV
    
    return state, 0.0

def spin1_x_int(state: int, ns: int, sites: Union[List[int], None], spin_value: float = _SPIN_1):
    """
    Apply S_x to an integer state.
    Returns arrays of output states and coefficients.
    """
    if sites is None:
        sites = list(range(ns))
    
    # Start with the initial state
    current_states = np.array([state], dtype=np.int64)
    current_coeffs = np.array([1.0], dtype=np.float64)
    
    # Apply S_x at each site sequentially
    for site in sites:
        new_states = []
        new_coeffs = []
        
        for i in range(len(current_states)):
            s = current_states[i]
            c = current_coeffs[i]
            out_s, out_c = _spin1_x_single_site(s, ns, site)
            
            for j in range(len(out_s)):
                if abs(out_c[j]) > 1e-15:
                    new_states.append(out_s[j])
                    new_coeffs.append(c * out_c[j])
        
        if len(new_states) == 0:
            return (np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64))
        
        current_states = np.array(new_states, dtype=np.int64)
        current_coeffs = np.array(new_coeffs, dtype=np.float64)
    
    return (current_states, current_coeffs * spin_value)

################################################################################
#! S_y operator
################################################################################

@numba.njit(inline='always')
def _spin1_y_single_site(state: int, ns: int, site: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply S_y to a single site for spin-1.
    S_y = (S_+ - S_-) / (2i) = -i(S_+ - S_-) / 2
    
    S_y |+1⟩ = -i (1/sqrt2) | 0⟩
    S_y | 0⟩ = -i (1/sqrt2) (|+1⟩ - |-1⟩) 
    S_y |-1⟩ = +i (1/sqrt2) | 0⟩
    """
    local_s = _get_spin1_state(state, ns, site)
    
    if local_s == 0:  # |+1⟩ -> |0⟩ with -i/sqrt2
        new_state = _set_spin1_state(state, ns, site, 1)
        return np.array([new_state], dtype=np.int64), np.array([-1j * _SQRT2_INV], dtype=np.complex128)
    
    elif local_s == 1:  # |0⟩ -> |+1⟩ and |-1⟩
        state_plus = _set_spin1_state(state, ns, site, 0)
        state_minus = _set_spin1_state(state, ns, site, 2)
        # S_y |0⟩ = -i/sqrt2 |+1⟩ + i/sqrt2 |-1⟩
        return (np.array([state_plus, state_minus], dtype=np.int64), 
                np.array([-1j * _SQRT2_INV, 1j * _SQRT2_INV], dtype=np.complex128))
    
    elif local_s == 2:  # |-1⟩ -> |0⟩ with +i/sqrt2
        new_state = _set_spin1_state(state, ns, site, 1)
        return np.array([new_state], dtype=np.int64), np.array([1j * _SQRT2_INV], dtype=np.complex128)
    
    else:  # Invalid
        return np.array([state], dtype=np.int64), np.array([0.0 + 0.0j], dtype=np.complex128)

@numba.njit(inline='always')
def _spin1_y_core(state: int, ns: int, sites: tuple) -> Tuple[int, complex]:
    """Core S_y for instruction code system (single output approximation)."""
    if len(sites) == 1:
        site = sites[0]
        local_s = _get_spin1_state(state, ns, site)
        
        if local_s == 0:  # |+1⟩ -> |0⟩
            new_state = _set_spin1_state(state, ns, site, 1)
            return new_state, -1j * _SQRT2_INV
        elif local_s == 1:  # |0⟩ -> approximate
            new_state = _set_spin1_state(state, ns, site, 0)
            return new_state, -1j * _SQRT2_INV
        elif local_s == 2:  # |-1⟩ -> |0⟩
            new_state = _set_spin1_state(state, ns, site, 1)
            return new_state, 1j * _SQRT2_INV
    
    return state, 0.0 + 0.0j

def spin1_y_int(state: int, ns: int, sites: Union[List[int], None], spin_value: float = _SPIN_1):
    """Apply S_y to an integer state."""
    if sites is None:
        sites = list(range(ns))
    
    current_states = np.array([state], dtype=np.int64)
    current_coeffs = np.array([1.0 + 0.0j], dtype=np.complex128)
    
    for site in sites:
        new_states = []
        new_coeffs = []
        
        for i in range(len(current_states)):
            s = current_states[i]
            c = current_coeffs[i]
            out_s, out_c = _spin1_y_single_site(s, ns, site)
            
            for j in range(len(out_s)):
                if abs(out_c[j]) > 1e-15:
                    new_states.append(out_s[j])
                    new_coeffs.append(c * out_c[j])
        
        if len(new_states) == 0:
            return (np.array([state], dtype=np.int64), np.array([0.0 + 0.0j], dtype=np.complex128))
        
        current_states = np.array(new_states, dtype=np.int64)
        current_coeffs = np.array(new_coeffs, dtype=np.complex128)
    
    return (current_states, current_coeffs * spin_value)

################################################################################
#! S_+ S_- and S_- S_+ correlators
################################################################################

@numba.njit(inline='always')
def spin1_pm_int_np(state: int, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
    """S_+ at site1, then S_- at site2."""
    if len(sites) < 2:
        return (np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64))
    
    site1, site2 = sites[0], sites[1]
    
    # Apply S_+ at site1
    s1, c1 = _spin1_plus_core(state, ns, site1)
    if abs(c1) < 1e-15:
        return (np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64))
    
    # Apply S_- at site2
    s2, c2 = _spin1_minus_core(s1, ns, site2)
    
    return (np.array([s2], dtype=np.int64), np.array([c1 * c2], dtype=np.float64))

@numba.njit(inline='always')
def spin1_mp_int_np(state: int, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
    """S_- at site1, then S_+ at site2."""
    if len(sites) < 2:
        return (np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64))
    
    site1, site2 = sites[0], sites[1]
    
    # Apply S_- at site1
    s1, c1 = _spin1_minus_core(state, ns, site1)
    if abs(c1) < 1e-15:
        return (np.array([state], dtype=np.int64), np.array([0.0], dtype=np.float64))
    
    # Apply S_+ at site2
    s2, c2 = _spin1_plus_core(s1, ns, site2)
    
    return (np.array([s2], dtype=np.int64), np.array([c1 * c2], dtype=np.float64))

################################################################################
#! Lookup codes for spin-1 operators
################################################################################

class Spin1LookupCodes(IntEnum):
    """Instruction codes for spin-1 operators."""
    
    # Global operators (0-10)
    s1_x_global:    int = 100
    s1_y_global:    int = 101
    s1_z_global:    int = 102
    s1_p_global:    int = 103
    s1_m_global:    int = 104
    s1_z2_global:   int = 105  # S_z^2 for quadrupolar
    
    # Local operators (single site)
    s1_x_local:     int = 111
    s1_y_local:     int = 112
    s1_z_local:     int = 113
    s1_p_local:     int = 114
    s1_m_local:     int = 115
    s1_z2_local:    int = 116
    
    # Correlation operators (two-site)
    s1_x_corr:      int = 120
    s1_y_corr:      int = 121
    s1_z_corr:      int = 122
    s1_p_corr:      int = 123
    s1_m_corr:      int = 124
    s1_pm_corr:     int = 125
    s1_mp_corr:     int = 126
    s1_z2_corr:     int = 127
    
    @staticmethod
    def is_custom_code(code: int) -> bool:
        """Check if a code represents a custom operator."""
        return code >= CUSTOM_OP_BASE or code < 100
    
    @staticmethod
    def to_dict() -> Dict[str, int]:
        return {
            # Global
            'S1x'           : Spin1LookupCodes.s1_x_global,
            'S1y'           : Spin1LookupCodes.s1_y_global,
            'S1z'           : Spin1LookupCodes.s1_z_global,
            'S1p'           : Spin1LookupCodes.s1_p_global,
            'S1m'           : Spin1LookupCodes.s1_m_global,
            'S1z2'          : Spin1LookupCodes.s1_z2_global,
            # Local
            'S1x/L'         : Spin1LookupCodes.s1_x_local,
            'S1y/L'         : Spin1LookupCodes.s1_y_local,
            'S1z/L'         : Spin1LookupCodes.s1_z_local,
            'S1p/L'         : Spin1LookupCodes.s1_p_local,
            'S1m/L'         : Spin1LookupCodes.s1_m_local,
            'S1z2/L'        : Spin1LookupCodes.s1_z2_local,
            # Correlation
            'S1x/C'         : Spin1LookupCodes.s1_x_corr,
            'S1y/C'         : Spin1LookupCodes.s1_y_corr,
            'S1z/C'         : Spin1LookupCodes.s1_z_corr,
            'S1p/C'         : Spin1LookupCodes.s1_p_corr,
            'S1m/C'         : Spin1LookupCodes.s1_m_corr,
            'S1pm/C'        : Spin1LookupCodes.s1_pm_corr,
            'S1mp/C'        : Spin1LookupCodes.s1_mp_corr,
            'S1z2/C'        : Spin1LookupCodes.s1_z2_corr,
        }

SPIN1_LOOKUP_CODES = Spin1LookupCodes

################################################################################
#! Factory functions for creating Operator objects
################################################################################

def s1_x(lattice    : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
        sites       : Optional[List[int]]   = None,
        spin_value  : float                 = _SPIN_1) -> Operator:
    r"""
    Factory function for spin-1 S_x operator.
    
    Parameters
    ----------
    lattice : Lattice, optional
        Lattice object for geometry.
    ns : int, optional
        Number of sites.
    type_act : OperatorTypeActing
        Operator locality (Global, Local, Correlation).
    sites : list, optional
        Sites to act on.
    spin_value : float
        Spin value scaling.
        
    Returns
    -------
    Operator
        The S_x operator for spin-1.
    """
    if ns is None:
        ns = lattice.ns if lattice is not None else 1
    
    if type_act == OperatorTypeActing.Global:
        code = Spin1LookupCodes.s1_x_global
        name = 'S1x'
    elif type_act == OperatorTypeActing.Local:
        code = Spin1LookupCodes.s1_x_local
        name = 'S1x/L'
    else:
        code = Spin1LookupCodes.s1_x_corr
        name = 'S1x/C'
    
    return create_operator(
        name        = name,
        fun_int     = spin1_x_int,
        fun_np      = None,
        fun_jax     = None,
        ns          = ns,
        lattice     = lattice,
        type_act    = type_act,
        sites       = sites,
        modifies    = True,
        matrix      = _S1_X,
        code        = code,
    )

def s1_y(lattice: Optional[Lattice] = None,
         ns: Optional[int] = None,
         type_act: OperatorTypeActing = OperatorTypeActing.Global,
         sites: Optional[List[int]] = None,
         spin_value: float = _SPIN_1) -> Operator:
    r"""Factory function for spin-1 S_y operator."""
    if ns is None:
        ns = lattice.ns if lattice is not None else 1
    
    if type_act == OperatorTypeActing.Global:
        code = Spin1LookupCodes.s1_y_global
        name = 'S1y'
    elif type_act == OperatorTypeActing.Local:
        code = Spin1LookupCodes.s1_y_local
        name = 'S1y/L'
    else:
        code = Spin1LookupCodes.s1_y_corr
        name = 'S1y/C'
    
    return create_operator(
        name        = name,
        fun_int     = spin1_y_int,
        fun_np      = None,
        fun_jax     = None,
        ns          = ns,
        lattice     = lattice,
        type_act    = type_act,
        sites       = sites,
        modifies    = True,
        matrix      = _S1_Y,
        code        = code,
    )

def s1_z(lattice: Optional[Lattice] = None,
         ns: Optional[int] = None,
         type_act: OperatorTypeActing = OperatorTypeActing.Global,
         sites: Optional[List[int]] = None,
         spin_value: float = _SPIN_1) -> Operator:
    r"""Factory function for spin-1 S_z operator."""
    if ns is None:
        ns = lattice.ns if lattice is not None else 1
    
    if type_act == OperatorTypeActing.Global:
        code = Spin1LookupCodes.s1_z_global
        name = 'S1z'
    elif type_act == OperatorTypeActing.Local:
        code = Spin1LookupCodes.s1_z_local
        name = 'S1z/L'
    else:
        code = Spin1LookupCodes.s1_z_corr
        name = 'S1z/C'
    
    return create_operator(
        name        = name,
        fun_int     = spin1_z_int,
        fun_np      = spin1_z_np,
        fun_jax     = None,
        ns          = ns,
        lattice     = lattice,
        type_act    = type_act,
        sites       = sites,
        modifies    = False,
        matrix      = _S1_Z,
        code        = code,
    )

def s1_z2(lattice: Optional[Lattice] = None,
          ns: Optional[int] = None,
          type_act: OperatorTypeActing = OperatorTypeActing.Global,
          sites: Optional[List[int]] = None,
          spin_value: float = _SPIN_1) -> Operator:
    r"""Factory function for spin-1 S_z^2 operator (quadrupolar)."""
    if ns is None:
        ns = lattice.ns if lattice is not None else 1
    
    if type_act == OperatorTypeActing.Global:
        code = Spin1LookupCodes.s1_z2_global
        name = 'S1z2'
    elif type_act == OperatorTypeActing.Local:
        code = Spin1LookupCodes.s1_z2_local
        name = 'S1z2/L'
    else:
        code = Spin1LookupCodes.s1_z2_corr
        name = 'S1z2/C'
    
    return create_operator(
        name        = name,
        fun_int     = lambda s, ns, sites, sv=spin_value: spin1_z2_int_np(s, ns, sites, sv),
        fun_np      = None,
        fun_jax     = None,
        ns          = ns,
        lattice     = lattice,
        type_act    = type_act,
        sites       = sites,
        modifies    = False,
        matrix      = _S1_Z2,
        code        = code,
    )

def s1_plus(lattice: Optional[Lattice] = None,
            ns: Optional[int] = None,
            type_act: OperatorTypeActing = OperatorTypeActing.Global,
            sites: Optional[List[int]] = None,
            spin_value: float = _SPIN_1) -> Operator:
    r"""Factory function for spin-1 S_+ (raising) operator."""
    if ns is None:
        ns = lattice.ns if lattice is not None else 1
    
    if type_act == OperatorTypeActing.Global:
        code = Spin1LookupCodes.s1_p_global
        name = 'S1p'
    elif type_act == OperatorTypeActing.Local:
        code = Spin1LookupCodes.s1_p_local
        name = 'S1p/L'
    else:
        code = Spin1LookupCodes.s1_p_corr
        name = 'S1p/C'
    
    return create_operator(
        name        = name,
        fun_int     = spin1_plus_int,
        fun_np      = None,
        fun_jax     = None,
        ns          = ns,
        lattice     = lattice,
        type_act    = type_act,
        sites       = sites,
        modifies    = True,
        matrix      = _S1_PLUS,
        code        = code,
    )

def s1_minus(lattice: Optional[Lattice] = None,
             ns: Optional[int] = None,
             type_act: OperatorTypeActing = OperatorTypeActing.Global,
             sites: Optional[List[int]] = None,
             spin_value: float = _SPIN_1) -> Operator:
    r"""Factory function for spin-1 S_- (lowering) operator."""
    if ns is None:
        ns = lattice.ns if lattice is not None else 1
    
    if type_act == OperatorTypeActing.Global:
        code = Spin1LookupCodes.s1_m_global
        name = 'S1m'
    elif type_act == OperatorTypeActing.Local:
        code = Spin1LookupCodes.s1_m_local
        name = 'S1m/L'
    else:
        code = Spin1LookupCodes.s1_m_corr
        name = 'S1m/C'
    
    return create_operator(
        name        = name,
        fun_int     = spin1_minus_int,
        fun_np      = None,
        fun_jax     = None,
        ns          = ns,
        lattice     = lattice,
        type_act    = type_act,
        sites       = sites,
        modifies    = True,
        matrix      = _S1_MINUS,
        code        = code,
    )

# Aliases
s1_p = s1_plus
s1_m = s1_minus

################################################################################
#! Composition function for spin-1 operators
################################################################################

def spin1_composition_integer(is_complex: bool, only_apply: bool = False) -> Callable:
    """
    Creates a spin-1 operator composition kernel based on instruction codes.
    
    Parameters
    ----------
    is_complex : bool
        Whether to use complex coefficients.
    only_apply : bool
        If True, only returns the single-operator application function.
        
    Returns
    -------
    Callable
        JIT-compiled composition function.
    """
    dtype = np.complex128 if is_complex else np.float64
    
    @numba.njit(nogil=True)
    def spin1_operator_composition_single_op(state: int, code: int, site1: int, site2: int, ns: int):
        """Applies a single spin-1 operator based on the code."""
        
        current_s = state
        current_c = dtype(1.0 + 0.0j)
        is_diagonal = False
        sites_1 = (site1,)
        sites_2 = (site1, site2)
        
        # Local operators
        if code   == Spin1LookupCodes.s1_x_local:
            s_arr, c_arr    = _spin1_x_core(state, ns, sites_1)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == Spin1LookupCodes.s1_y_local:
            s_arr, c_arr    = _spin1_y_core(state, ns, sites_1)
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == Spin1LookupCodes.s1_z_local:
            s_arr, c_arr    = _spin1_z_core(state, ns, sites_1)
            is_diagonal     = True
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == Spin1LookupCodes.s1_z2_local:
            s_arr, c_arr    = _spin1_z2_core(state, ns, sites_1)
            is_diagonal     = True
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == Spin1LookupCodes.s1_p_local:
            s_arr, c_arr    = spin1_plus_int_np(state, ns, sites_1)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        elif code == Spin1LookupCodes.s1_m_local:
            s_arr, c_arr    = spin1_minus_int_np(state, ns, sites_1)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        
        # Correlation operators
        elif code == Spin1LookupCodes.s1_z_corr:
            s_arr, c_arr    = _spin1_z_core(state, ns, sites_2)
            is_diagonal     = True
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == Spin1LookupCodes.s1_z2_corr:
            s_arr, c_arr    = _spin1_z2_core(state, ns, sites_2)
            is_diagonal     = True
            current_s       = s_arr
            current_c       = dtype(c_arr)
        elif code == Spin1LookupCodes.s1_pm_corr:
            s_arr, c_arr    = spin1_pm_int_np(state, ns, sites_2)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        elif code == Spin1LookupCodes.s1_mp_corr:
            s_arr, c_arr    = spin1_mp_int_np(state, ns, sites_2)
            current_s       = s_arr[0]
            current_c       = dtype(c_arr[0])
        else:
            pass
        
        return current_s, current_c, is_diagonal
    
    if only_apply:
        return spin1_operator_composition_single_op
    
    @numba.njit(nogil=True)
    def spin1_operator_composition_int(state: int, nops: int, codes, sites, coeffs, ns: int, out_states, out_vals):
        """Apply a sequence of spin-1 operators to a basis state."""
        ptr         = 0
        diag_sum    = 0.0 + 0.0j if is_complex else 0.0
        
        for k in range(nops):
            code                                = codes[k]
            coeff                               = coeffs[k]
            s1                                  = sites[k, 0]
            s2                                  = sites[k, 1] if sites.shape[1] > 1 else -1
            
            current_s, current_c, is_diagonal   = spin1_operator_composition_single_op(state, code, s1, s2, ns)
            final_val                           = current_c * coeff
            if abs(final_val) < 1e-15:
                continue
            
            if is_diagonal:
                diag_sum += final_val
            else:
                out_states[ptr] = current_s
                out_vals[ptr]   = final_val
                ptr            += 1
        
        if abs(diag_sum) > 1e-15:
            out_states[ptr] = state
            out_vals[ptr]   = diag_sum
            ptr            += 1
        
        return out_states[:ptr], out_vals[:ptr]
    
    return spin1_operator_composition_int

################################################################################
#! Registration with the operator catalog
################################################################################

def _register_spin1_catalog_entries():
    """Register spin-1 operators with the global catalog."""
    
    def _s1_x_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         = spin1_x_int,
            fun_np          = None,
            fun_jax         = None,
            site_parity     = 1,
            modifies_state  = True,
        )
    
    def _s1_y_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         = spin1_y_int,
            fun_np          = None,
            fun_jax         = None,
            site_parity     = 1,
            modifies_state  = True,
        )
    
    def _s1_z_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         = spin1_z_int,
            fun_np          = spin1_z_np,
            fun_jax         = None,
            site_parity     = 1,
            modifies_state  = False,
        )
    
    def _s1_z2_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         = lambda s, ns, sites: spin1_z2_int_np(s, ns, sites),
            fun_np          = None,
            fun_jax         = None,
            site_parity     = 1,
            modifies_state  = False,
        )
    
    def _s1_plus_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         = spin1_plus_int,
            fun_np          = None,
            fun_jax         = None,
            site_parity     = 1,
            modifies_state  = True,
        )
    
    def _s1_minus_factory() -> LocalOpKernels:
        return LocalOpKernels(
            fun_int         = spin1_minus_int,
            fun_np          = None,
            fun_jax         = None,
            site_parity     = 1,
            modifies_state  = True,
        )
    
    # Register operators
    register_local_operator(
        LocalSpaceTypes.SPIN_1,
        key             = "s1_x",
        factory         = _s1_x_factory,
        description     = r"Spin-1 S_x operator.",
        algebra         = r"[S_x, S_y] = i S_z, S^2 = S(S+1) = 2",
        sign_convention = r"Standard spin-1 matrices in S_z basis.",
        tags            = ("spin-1", "pauli"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1,
        key             = "s1_y",
        factory         = _s1_y_factory,
        description     = r"Spin-1 S_y operator.",
        algebra         = r"[S_y, S_z] = i S_x, S^2 = S(S+1) = 2",
        sign_convention = r"Standard spin-1 matrices in S_z basis.",
        tags            = ("spin-1", "pauli"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1,
        key             = "s1_z",
        factory         = _s1_z_factory,
        description     = r"Spin-1 S_z operator (diagonal).",
        algebra         = r"[S_z, S_+] = S_+, [S_z, S_-] = -S_-",
        sign_convention = r"Eigenvalues +1, 0, -1 in computational basis.",
        tags            = ("spin-1", "pauli", "diagonal"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1,
        key             = "s1_z2",
        factory         = _s1_z2_factory,
        description     = r"Spin-1 S_z^2 operator for quadrupolar interactions.",
        algebra         = r"S_z^2 |m⟩ = m^2 |m⟩",
        sign_convention = r"Eigenvalues 1, 0, 1 for |+1⟩, |0⟩, |-1⟩.",
        tags            = ("spin-1", "quadrupolar", "diagonal"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1,
        key             = "s1_plus",
        factory         = _s1_plus_factory,
        description     = r"Spin-1 S_+ raising operator.",
        algebra         = r"S_+ |m⟩ = sqrt(2 - m(m+1)) |m+1⟩",
        sign_convention = r"Standard ladder operator.",
        tags            = ("spin-1", "raising"),
    )
    register_local_operator(
        LocalSpaceTypes.SPIN_1,
        key             = "s1_minus",
        factory         = _s1_minus_factory,
        description     = r"Spin-1 S_- lowering operator.",
        algebra         = r"S_- |m⟩ = sqrt(2 - m(m-1)) |m-1⟩",
        sign_convention = r"Standard ladder operator.",
        tags            = ("spin-1", "lowering"),
    )

# Register upon import
_register_spin1_catalog_entries()

################################################################################
#! EOF
################################################################################
