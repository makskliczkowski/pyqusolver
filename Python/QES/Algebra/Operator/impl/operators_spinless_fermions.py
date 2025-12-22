r"""
Spinless Fermion Operators
==========================

This module implements creation, annihilation, and number operators for spinless fermions,
as well as their momentum-space versions. It provides efficient routines for acting on
Fock basis states represented as integers or NumPy arrays.

Overview
--------

For a chain of :math:`N_s` spinless fermionic modes, sites are labeled from the left,
:math:`i = 0, 1, \ldots, N_s-1`. The on-site occupation number

.. math::

    n_i = c_i^\dagger c_i \in \{0, 1\}

defines the computational Fock basis

.. math::

    |\mathbf{n}\rangle = |n_0\, n_1\, \ldots\, n_{N_s-1}\rangle.

Integer Encoding
----------------

A basis vector is encoded as a non-negative integer

.. math::

    s(\mathbf{n}) = \sum_{i=0}^{N_s-1} n_i\,2^{N_s-1-i},

so that bit :math:`N_s-1-i` of :math:`s` stores :math:`n_i`.
The Hamming weight (``popcount``) of :math:`s` equals :math:`\sum_i n_i`.

Canonical Anticommutation Relations
-----------------------------------

Creation and annihilation operators obey

.. math::

    \{c_i, c_j\} = 0, \qquad
    \{c_i^\dagger, c_j^\dagger\} = 0, \qquad
    \{c_i, c_j^\dagger\} = \delta_{ij}.

Jordan-Wigner Phase (Fermionic Sign)
------------------------------------

Because fermions anticommute, the operator must “pass through''
all modes to its left before acting. Define the parity operator up to (but not including) site :math:`i`:

.. math::

    \mathcal{P}_i = (-1)^{\sum_{j=0}^{i-1} n_j}

With the integer representation this is

.. math::

    \mathcal{P}_i(s) = (-1)^{\mathrm{popcount}\left(s\,\&\,\left[(1\ll(N_s-i))-1\right]\right)}

Action on a Basis State
-----------------------

.. math::

    c_i^\dagger\,|\mathbf{n}\rangle &= \mathcal{P}_i(\mathbf{n})\,(1-n_i)\,|n_0\ldots 1_i\ldots n_{N_s-1}\rangle \\
    c_i\,|\mathbf{n}\rangle &= \mathcal{P}_i(\mathbf{n})\,n_i\,|n_0\ldots 0_i\ldots n_{N_s-1}\rangle

If the occupation constraint :math:`(1-n_i)` or :math:`n_i` vanishes, the state is annihilated.

Momentum-Space Operator
-----------------------

For any subset :math:`S \subseteq \{0, \ldots, N_s-1\}` we define

.. math::

    c_k = \frac{1}{\sqrt{|S|}} \sum_{i\in S} e^{-ik i}\,c_i, \qquad k\in[0,2\pi).

Acting on a basis state gives a superposition with amplitudes

.. math::

    \langle\mathbf{n}|\,c_k\,|\mathbf{n}\rangle =
    \frac{1}{\sqrt{|S|}} \sum_{i\in S} \mathcal{P}_i(\mathbf{n})\,n_i\,e^{-ik i}
    
----------------------------------------------------------------
File        : Algebra/Operator/operators_spinless_fermions.py
Author      : Maksymilian Kliczkowski
Date        : 2025-10-25
License     : MIT
----------------------------------------------------------------
"""

import  numpy as np
import  numba
from    typing import List, Union, Dict, Optional, Callable, TYPE_CHECKING
from    enum import IntEnum

################################################################################

try:
    from QES.Algebra.Operator.operator                  import Operator, OperatorTypeActing, create_operator, ensure_operator_output_shape_numba
    from QES.Algebra.Operator.catalog                   import register_local_operator
    from QES.Algebra.Hilbert.hilbert_local              import LocalOpKernels, LocalSpaceTypes
    from QES.Algebra.Operator.phase_utils               import bit_popcount, fermionic_parity_int, fermionic_parity_array
    from QES.Algebra.Operator.special_operator          import CUSTOM_OP_BASE
    from QES.Algebra.Operator.impl.operators_hardcore   import (
                                                            hardcore_create_int,
                                                            hardcore_create_np,
                                                            hardcore_annihilate_int,
                                                            hardcore_annihilate_np,
                                                            hardcore_number_int,
                                                            hardcore_number_np,
                                                        )
except ImportError:
    raise ImportError("QES.Algebra.Operator and QES.Algebra.Hilbert modules are required for spinless fermion operators.")

################################################################################

from QES.general_python.lattices.lattice            import Lattice
from QES.general_python.common.binary               import BACKEND_REPR as JAX_AVAILABLE, check_int, flip_int

if JAX_AVAILABLE:
    from QES.Algebra.Operator.jax.operators_spinless_fermions import c_dag_jnp, c_jnp, c_k_jnp, c_k_dag_jnp, n_jax, n_int_jax
else:
    c_dag_jnp = c_jnp = c_k_jnp = c_k_dag_jnp = n_jax = n_jax_int = None

# Default data types for integer and float arrays
_DEFAULT_INT    = np.int64
_DEFAULT_FLOAT  = np.float64
_bit            = check_int
_flip           = flip_int

################################################################################

class FermionLookupCodes(IntEnum):
    r"""
    Integer codes for fermion operators used in JIT-compiled composition functions.
    
    These codes map operator types to integers for efficient dispatching in numba.
    """
    
    # Global operators (act on all sites)
    c_dag_global        : int = 0
    c_ann_global        : int = 1
    n_global            : int = 2
    c_dag_c_global      : int = 3     # hopping term c^dag _i c_j
    c_c_dag_global      : int = 4     # c_i c^dag _j
    n_n_global          : int = 5     # density-density n_i n_j
    c_dag_c_dag_global  : int = 6     # pairing c^dag _i c^dag _j
    c_c_global          : int = 7     # pairing c_i c_j
    c_dag_n_global      : int = 8     # c^dag _i n_j
    c_n_global          : int = 9     # c_i n_j
    n_c_dag_global      : int = 10    # n_i c^dag _j
    n_c_global          : int = 11    # n_i c_j
    
    # Local operators (single site)
    c_dag_local         : int = 12
    c_ann_local         : int = 13
    n_local             : int = 14
    
    # Correlation operators (two sites)
    c_dag_c_corr        : int = 20    # hopping c^dag _i c_j
    c_c_dag_corr        : int = 21    # c_i c^dag _j
    n_n_corr            : int = 22    # density-density n_i n_j
    c_dag_c_dag_corr    : int = 23    # pairing c^dag _i c^dag _j
    c_c_corr            : int = 24    # pairing c_i c_j
    c_dag_n_corr        : int = 25    # c^dag _i n_j
    c_n_corr            : int = 26    # c_i n_j
    n_c_dag_corr        : int = 27    # n_i c^dag _j
    n_c_corr            : int = 28    # n_i c_j
    
    @staticmethod
    def is_custom_code(code: int) -> bool:
        """Check if a code represents a custom operator."""
        return code >= 29 or code < 0
    
    @staticmethod
    def to_dict() -> Dict[str, int]:
        """Return dictionary mapping operator names to codes."""
        return {
            # Global operators
            'c_dag'         : FermionLookupCodes.c_dag_global,
            'c'             : FermionLookupCodes.c_ann_global,
            'c_ann'         : FermionLookupCodes.c_ann_global,
            'n'             : FermionLookupCodes.n_global,
            # other global operators
            'c_dag_c'       : FermionLookupCodes.c_dag_c_global,
            'c_c_dag'       : FermionLookupCodes.c_c_dag_global,
            'n_n'           : FermionLookupCodes.n_n_global,
            'c_dag_n'       : FermionLookupCodes.c_dag_n_global,
            'c_n'           : FermionLookupCodes.c_n_global,
            'n_c_dag'       : FermionLookupCodes.n_c_dag_global,
            'n_c'           : FermionLookupCodes.n_c_global,
            # Local operators
            'c_dag/L'       : FermionLookupCodes.c_dag_local,
            'c/L'           : FermionLookupCodes.c_ann_local,
            'c_ann/L'       : FermionLookupCodes.c_ann_local,
            'n/L'           : FermionLookupCodes.n_local,
            # Correlation operators
            'c_dag_c/C'     : FermionLookupCodes.c_dag_c_corr,
            'c_c_dag/C'     : FermionLookupCodes.c_c_dag_corr,
            'n_n/C'         : FermionLookupCodes.n_n_corr,
            'c_dag_c_dag/C' : FermionLookupCodes.c_dag_c_dag_corr,
            'c_c/C'         : FermionLookupCodes.c_c_corr,
            'c_dag_n/C'     : FermionLookupCodes.c_dag_n_corr,
            'c_n/C'         : FermionLookupCodes.c_n_corr,
            'n_c_dag/C'     : FermionLookupCodes.n_c_dag_corr,
            'n_c/C'         : FermionLookupCodes.n_c_corr,
        }

# Alias for backward compatibility
FERMION_LOOKUP_CODES = FermionLookupCodes

###############################################################################
#! Creation / annihilation on *integer* occupation number
###############################################################################

@numba.njit
def c_dag_int_np(state: int, ns: int, sites: List[int], prefactor: float = 1.0):
    r"""
    Applies the creation operator (c\dag) for spinless fermions on a given site of an integer-represented Fock state.
    Parameters
    ----------
    state : int
        The integer representation of the Fock state.
    ns : int
        The total number of sites.
    sites : List[int]
        The list of site indices (0-based, leftmost site is 0).
    prefactor : float, optional
        A prefactor to multiply the resulting coefficient (default is 1.0).
    Returns
    -------
    out_state : numpy.ndarray
        Array of length 1 containing the new state(s) as integer(s) after applying the creation operator.
    out_coeff : numpy.ndarray
        Array of length 1 containing the corresponding coefficient(s) after applying the creation operator.
    Notes
    -----
    - If the site is already occupied, the result is zero (fermionic exclusion principle).
    - The sign is determined by the fermionic parity up to the given site.
    - Helper functions `_bit`, `_flip`, and `fermionic_parity_int` are used for bit manipulation and parity calculation.
    
    Example
    -------
    >>> state = 0b000000
    >>> ns = 6
    >>> site = 2
    >>> prefactor = 1.0
    >>> out_state, out_coeff = c_dag_int_np(state, ns, site, prefactor)
    >>> print(out_state)  # Output: [0b000100]
    >>> print(out_coeff)  # Output: [1.0]
    """
    
    # position of the site in the integer representation
    
    new_state = state
    coeff_val = 1.0
    
    for site in sites:
        pos = ns - 1 - site

        # occupancy check on the *current* state
        if _bit(new_state, pos):
            coeff_val = 0.0
            new_state = state         # revert to input
            break

        sign        = fermionic_parity_int(new_state, ns, site)
        new_state   = _flip(new_state, pos)
        coeff_val  *= sign * prefactor
    
    return (new_state,), (coeff_val,) # no allocation for numba

@numba.njit
def c_int_np(state: int, ns: int, sites: int, prefactor: float = 1.0):
    """
    Applies the annihilation operator (c) for spinless fermions on a given site of a basis state represented as an integer.
    Parameters
    ----------
    state : int
        The integer representation of the basis state.
    ns : int
        The total number of sites in the system.
    site : int
        The site index (0-based, leftmost site is 0) where the operator acts.
    prefactor : float, optional
        A prefactor to multiply the resulting coefficient (default is 1.0).
    Returns
    -------
    out_state : numpy.ndarray
        Array of shape (1,) containing the new basis state(s) as integer(s) after applying the operator.
    out_coeff : numpy.ndarray
        Array of shape (1,) containing the corresponding coefficient(s) after applying the operator.
    Notes
    -----
    - If the site is unoccupied (bit is 0), the output coefficient is 0 and the state is unchanged.
    - The function accounts for the fermionic sign (parity) when applying the operator.
    - Helper functions `_bit`, `_flip`, and `fermionic_parity_int` are assumed to be defined elsewhere in the module.
    """

    # position of the site in the integer representation
    
    new_state = state
    coeff_val = 1.0
    for site in sites:
        pos = ns - 1 - site
        # occupancy check on the *current* state
        if _bit(state, pos) == 0:
            coeff_val = 0.0
            new_state = state
            break
        sign        = fermionic_parity_int(new_state, ns, site)
        new_state   = _flip(new_state, pos)
        coeff_val  *= sign * prefactor
    
    return (new_state,), (coeff_val,) # no allocation for numba

###############################################################################
#!  Creation / annihilation on *NumPy* occupation array
###############################################################################

@numba.njit
def c_dag_np(state: np.ndarray, sites: np.ndarray, prefactor  : float = 1.0):
    r"""
    Applies the fermionic creation operator (c\dag) at a given site to a spinless fermion state.

    Parameters
    ----------
    state : np.ndarray
        The occupation number representation of the fermionic state (1D array of 0s and 1s).
    sites : np.ndarray
        The site indices at which to apply the creation operator.
    prefactor : float, optional
        A multiplicative prefactor to apply to the resulting amplitude (default is 1.0).

    Returns
    -------
    tuple
        A tuple (new_state, amplitude) where:
            - new_state (np.ndarray): The updated state after applying the creation operator.
            - amplitude (float): The resulting amplitude, including the sign from fermionic parity and the prefactor.
        If the site is already occupied, returns (state, 0.0).

    Notes
    -----
    This function modifies the input state in-place. The sign is determined by the fermionic parity up to the given site.
    """
    sign    = 1.0
    coeff   = np.ones(1, dtype=_DEFAULT_FLOAT)
    out     = state.copy()
    for site in sites:
        if out[site] > 0: # already occupied
            coeff *= 0.0
            break
        sign       *= fermionic_parity_array(out, site)
        out[site]   = 1
    n_sites  = sites.shape[0]
    coeff   *= sign * prefactor**n_sites
    return ensure_operator_output_shape_numba(out, coeff)

@numba.njit
def c_np(state: np.ndarray, sites: np.ndarray, prefactor   : float = 1.0):
    r"""
    Applies the annihilation operator (c) to a spinless fermion state at a given site.

    Parameters
    ----------
    state : np.ndarray
        The occupation number representation of the fermionic state (1 for occupied, 0 for unoccupied).
    sites : np.ndarray
        The sites indices at which to apply the annihilation operator.
    prefactor : float, optional
        A multiplicative prefactor to apply to the result (default is 1.0).

    Returns
    -------
    tuple
        A tuple containing:
            - The updated state as a numpy array.
            - The resulting coefficient (float), which is zero if the site is unoccupied,
                or the product of the sign from the parity function and the prefactor otherwise.

    Notes
    -----
    The function uses the Jordan-Wigner transformation convention, where the sign is determined
    by the parity of occupied sites to the left of the target site.
    """
    
    coeff   = np.ones(1, dtype=_DEFAULT_FLOAT)
    out     = state.copy()
    for site in sites:
        if out[site] == 0:
            coeff *= 0.0
            break
        sign        = fermionic_parity_array(out, site)
        out[site] = 0
    n_sites  = sites.shape[0]
    coeff   *= sign * prefactor**n_sites
    return ensure_operator_output_shape_numba(out, coeff)
    # return state, sign * prefactor**n_sites

###############################################################################
#!  Momentum-space fermionic operator  c_k and c_k\dag
#      c_k = (1/\sqrtN) \sum _i  e^{-ik i} c_i
###############################################################################

@numba.njit
def c_k_int_np(state: int, ns: int, sites: List[int], k: float, prefactor: float = 1.0):
    r"""
    Applies the momentum-space annihilation operator c_k to a given spinless fermion state.

    This function constructs the action of the annihilation operator in momentum space
    on a given basis state, for a set of site indices, and returns the resulting states
    and their coefficients.

    Args:
        state (int):
            The integer representation of the input basis state.
        ns (int):
            The total number of sites in the system.
        sites (List[int]):
            List of site indices where the operator acts.
        k (float):
            The momentum value (in radians) for the operator.
        prefactor (float, optional):
            A prefactor to multiply the operator by. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: Array of resulting basis states after applying the operator at each site.
            - out_coeff: Array of corresponding coefficients for each resulting state.

    Notes:
        - The coefficients are normalized by sqrt(len(sites)) if sites is not empty, otherwise by 1.0.
        - The function assumes the existence of `c_int_np`, which applies the site-local annihilation operator.
    """
    # count non-zero bits in the state
    non_zero        = bit_popcount(state, ns)
    if non_zero == 0:
        return np.empty(0, dtype=_DEFAULT_INT), np.empty(0, dtype=_DEFAULT_FLOAT)
    out_state       = np.empty(non_zero, dtype=_DEFAULT_INT)
    out_coeff       = np.empty(non_zero, dtype=_DEFAULT_FLOAT)
    index           = 0
    for i in sites:
        res_state, coeff    = c_int_np(state, ns, i, prefactor)
        if coeff == 0.0:
            continue
        out_state[index]    = res_state
        out_coeff[index]    = coeff * np.exp(-1j * k * i)
        index              += 1
    return out_state, out_coeff / np.sqrt(non_zero)

@numba.njit
def c_k_np(state: np.ndarray, sites: np.ndarray, k: float, prefactor: float = 1.0):
    ns               = state.shape[0]
    # number of occupied sites => upper bound on output length
    non_zero         = 0
    for j in range(ns):
        non_zero    += 1 if state[j] else 0

    if non_zero == 0:
        return (np.empty((0, ns), dtype=state.dtype),
                np.empty(0,        dtype=_DEFAULT_FLOAT))

    out_state        = np.empty((non_zero, ns), dtype=state.dtype)
    out_coeff        = np.empty(non_zero,        dtype=_DEFAULT_FLOAT)

    index            = 0
    for i in sites:
        if state[i] == 0:                         # empty -> no contribution
            continue

        tmp_state     = state.copy()
        tmp_state, c  = c_np(tmp_state, i, prefactor) # local c_i
        if c == 0.0:
            continue

        out_state[index, :] = tmp_state
        out_coeff[index]    = c * np.exp(-1j * k * i)
        index              += 1
    return ensure_operator_output_shape_numba(out_state[:index], out_coeff[:index] / np.sqrt(np.max(index, 1)))    

@numba.njit
def c_k_dag_int_np(state      : int,
                   ns         : int,
                   sites      : List[int],
                   k          : float,
                   prefactor  : float = 1.0):
    r"""
    Applies the momentum-space fermionic creation operator c\dag_k to a given basis state.

    This function constructs the action of the creation operator in momentum space
    on a spinless fermion basis state represented as an integer bitstring. It sums
    over all possible site indices, applying the real-space creation operator and
    weighting each term by the appropriate phase factor exp(1j * k * i).

    Args:
        state (int):
            The input basis state represented as an integer bitstring.
        ns (int):
            The total number of sites in the system.
        sites (List[int]): 
            List of site indices to consider for the creation operator.
        k (float):
            The momentum value for the creation operator.
        prefactor (float, optional):
            An overall prefactor to multiply the coefficients. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - out_state: Array of resulting basis states (as integers) after applying the operator.
            - out_coeff: Array of corresponding coefficients (complex floats), normalized by sqrt(number of nonzero terms).

    Notes:
        - If the input state is fully occupied, returns empty arrays.
        - The coefficients are normalized by the square root of the number of nonzero terms.
        - Requires the helper functions `bit_popcount` and `c_dag_int_np`, as well as the constants `_DEFAULT_INT` and `_DEFAULT_FLOAT`.
    """
    # number of empty sites
    occ_bits        = bit_popcount(state, ns)
    non_zero        = ns - occ_bits
    if non_zero == 0:
        return (np.empty(0, dtype=_DEFAULT_INT),
                np.empty(0, dtype=_DEFAULT_FLOAT))

    out_state       = np.empty(non_zero, dtype=_DEFAULT_INT)
    out_coeff       = np.empty(non_zero, dtype=_DEFAULT_FLOAT)

    index           = 0
    for i in sites:
        res_state, coeff = c_dag_int_np(state, ns, i, prefactor)
        if coeff == 0.0:
            continue
        out_state[index]  = res_state
        out_coeff[index]  = coeff * np.exp(1j * k * i)
        index            += 1

    return (out_state[:index],
            out_coeff[:index] / np.sqrt(max(index, 1)))

@numba.njit
def c_k_dag_np(state       : np.ndarray,
               sites       : List[int],
               k           : float,
               prefactor   : float = 1.0):
    r"""
    Apply momentum-space creation operator c_k\dag to a NumPy occupation array.
    """
    ns               = state.shape[0]
    # empty sites count -> upper bound
    non_zero         = ns
    for j in range(ns):
        non_zero    -= 1 if state[j] else 0

    if non_zero == 0:
        return (np.empty((0, ns), dtype=state.dtype),
                np.empty(0,        dtype=_DEFAULT_FLOAT))

    out_state        = np.empty((non_zero, ns), dtype=state.dtype)
    out_coeff        = np.empty(non_zero,        dtype=_DEFAULT_FLOAT)

    index            = 0
    for i in sites:
        if state[i] == 1:                         # already occupied
            continue

        tmp_state     = state.copy()
        tmp_state, c  = c_dag_np(tmp_state, i, prefactor)  # local c_i\dag
        if c == 0.0:
            continue

        out_state[index, :] = tmp_state
        out_coeff[index]    = c * np.exp(1j * k * i)
        index              += 1
    return ensure_operator_output_shape_numba(out_state[:index],
            out_coeff[:index] / np.sqrt(np.max(index, 1)))
    # return (out_state[:index],
            # out_coeff[:index] / np.sqrt(max(index, 1)))

###############################################################################
#! Number of fermions
###############################################################################

@numba.njit
def n_int_np(state: int, ns: int, sites: List[int], prefactor: float = 1.0):
    r"""
    Applies the number operator n_i for spinless fermions on a given site of an integer-represented Fock state.
    Parameters
    ----------
    state : int
        The integer representation of the Fock state.
    ns : int
        The total number of sites.
    sites : List[int]
        The list of site indices (0-based, leftmost site is 0).
    prefactor : float, optional
        A prefactor to multiply the resulting coefficient (default is 1.0).
    Returns
    -------
    out_state : numpy.ndarray
        Array of length 1 containing the new state(s) as integer(s) after applying the number operator.
    out_coeff : numpy.ndarray
        Array of length 1 containing the corresponding coefficient(s) after applying the number operator.
    Notes
    -----
    - If the site is unoccupied, the result is zero.
    - Helper function `_bit` is used for bit manipulation.
    Example
    -------
    >>> state = 0b000100
    >>> ns = 6
    >>> site = 2
    >>> prefactor = 1.0
    >>> out_state, out_coeff = n_int_np(state, ns, site, prefactor)
    >>> print(out_state)  # Output: [0b000100]
    >>> print(out_coeff)  # Output: [1.0]
    """
    
    coeff_val = 1.0
    for site in sites:
        pos = ns - 1 - site
        if _bit(state, pos) == 0:   # site unoccupied -> result = 0
            coeff_val = 0.0
            break
        
    n_sites        = len(sites)
    return (state, ), (coeff_val * prefactor**n_sites) # no allocation for numba

@numba.njit
def n_np(state: np.ndarray, sites: np.ndarray, prefactor: float = 1.0):
    r"""
    Applies the number operator n_i to a spinless fermion state at given sites.
    """
    
    coeff_val   = np.ones(1, dtype=_DEFAULT_FLOAT)
    out         = state.copy()
    for site in sites:
        if out[site] == 0:        # unoccupied -> zero immediately
            coeff_val = np.zeros(1, dtype=_DEFAULT_FLOAT)
            break

    n_sites = sites.shape[0]
    return ensure_operator_output_shape_numba(out, coeff_val * prefactor**n_sites)

###############################################################################
#! Public dispatch helpers  (match your \sigma -operator API)
###############################################################################

def _c_dag(state: Union[int, np.ndarray], ns: int, sites: int, prefactor: float = 1.0):
    """Creation operator dispatcher."""
    if isinstance(state, (int, np.integer)):
        return c_dag_int_np(int(state), ns, sites, prefactor)
    if isinstance(state, np.ndarray):
        return c_dag_np(state, sites, prefactor)
    return c_dag_jnp(state, ns, sites, prefactor)

def _c(state: Union[int, np.ndarray], ns: int, sites: int, prefactor: float = 1.0):
    """Annihilation operator dispatcher."""
    if isinstance(state, (int, np.integer)):
        return c_int_np(int(state), ns, sites, prefactor)
    if isinstance(state, np.ndarray):
        return c_np(state, sites, prefactor)
    return c_jnp(state, ns, sites, prefactor)

def _c_k(state        : Union[int, np.ndarray],
        ns           : int,
        sites        : Optional[List[int]],
        k            : float,
        prefactor    : float = 1.0):
    """Momentum-space annihilation operator dispatcher."""
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return c_k_int_np(int(state), ns, sites, k, prefactor)
    if isinstance(state, np.ndarray):
        return c_k_np(state, sites, k, prefactor)
    return c_k_jnp(state, ns, sites, k, prefactor)

def _c_k_dag(state     : Union[int, np.ndarray],
            ns        : int,
            sites     : Optional[List[int]],
            k         : float,
            prefactor : float = 1.0):
    """Momentum-space creation operator dispatcher."""
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return c_k_dag_int_np(int(state), ns, sites, k, prefactor)
    if isinstance(state, np.ndarray):
        return c_k_dag_np(state, sites, k, prefactor)
    return c_k_dag_jnp(state, ns, sites, k, prefactor)

def _n(state: Union[int, np.ndarray], ns: int, sites: Optional[List[int]], prefactor: float = 1.0):
    r"""Number operator dispatcher."""
    
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return n_int_np(int(state), ns, sites, prefactor)
    if isinstance(state, np.ndarray):
        return n_np(state, sites, prefactor)
    raise TypeError(f"Unsupported type for state: {type(state)}")

##############################################################################
#! Factory for the operator
##############################################################################

def c(  lattice     : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    r"""
    Factory for the fermionic annihilation operator c_i.
    """
    
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code        = FermionLookupCodes.c_ann_global
    elif OperatorTypeActing.is_type_local(type_act):
        code        = FermionLookupCodes.c_ann_local
    else:
        code        = FermionLookupCodes.c_c_corr
    
    return create_operator(
        type_act    = type_act,
        op_func_int = c_int_np,            # integer kernel dispatcher already wraps
        op_func_np  = c_np,
        op_func_jnp = c_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (prefactor,),
        name        = "c",
        modifies    = True,
        code        = code
    )

def cdag( lattice   : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    r"""
    Factory for the fermionic creation operator c_i\dag.
    """
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code        = FermionLookupCodes.c_dag_global
    elif OperatorTypeActing.is_type_local(type_act):
        code        = FermionLookupCodes.c_dag_local
    else:
        code        = FermionLookupCodes.c_dag_c_dag_corr
    
    return create_operator(
        type_act    = type_act,
        op_func_int = c_dag_int_np,
        op_func_np  = c_dag_np,
        op_func_jnp = c_dag_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (prefactor,),
        name        = "cdag",
        modifies    = True,
        code        = code
    )

def ck( k           : float,
        lattice     : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    r"""
    Factory for the momentum-space annihilation operator
        c_k = \frac{1}{\sqrt{|S|}} \sum_{i\in S} e^{-ik i}\,c_i .
    """
    #TOOD: check sites for global
    
    return create_operator(
        type_act    = type_act,
        op_func_int = c_k_int_np,
        op_func_np  = c_k_np,
        op_func_jnp = c_k_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (k, prefactor),
        name        = f"ck(k={k:.3g})",
        modifies    = True
    )

def ckdag(  k         : float,
            lattice   : Optional[Lattice]     = None,
            ns        : Optional[int]         = None,
            type_act  : OperatorTypeActing    = OperatorTypeActing.Global,
            sites     : Optional[List[int]]   = None,
            prefactor : float                 = 1.0) -> Operator:
    r"""
    Factory for the momentum-space creation operator
        c_k^{\dagger} = \frac{1}{\sqrt{|S|}} \sum_{i\in S} e^{+ik i}\,c_i^{\dagger}.
    """
    #TOOD: check sites for global
    
    return create_operator(
        type_act    = type_act,
        op_func_int = c_k_dag_int_np,
        op_func_np  = c_k_dag_np,
        op_func_jnp = c_k_dag_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (k, prefactor),
        name        = f"ckdag(k={k:.3g})",
        modifies    = True
    )

def n( lattice      : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    """
    Factory for the fermionic number operator n_i.
    """
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:
        code        = FermionLookupCodes.n_global
    elif OperatorTypeActing.is_type_local(type_act):
        code        = FermionLookupCodes.n_local
    else:
        code        = FermionLookupCodes.n_n_corr
    
    return create_operator(
        type_act    = type_act,
        op_func_int = n_int_np,
        op_func_np  = n_np,
        op_func_jnp = n_jax if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (prefactor,),
        name        = "n",
        modifies    = False
    )

# -----------------------------------------------------------------------------
#! Most common mixtures
# -----------------------------------------------------------------------------

def create_mixed_int(op1_fun, op2_fun):
    """
    Create a mixed integer operator from two operator functions.
    """
    
    @numba.njit(inline='always')
    def mixed_int(state, ns, sites, prefactor=1.0):
        
        # dumbo
        if sites is None or len(sites) < 2:
            s_dumb, c_dumb = op1_fun(state, ns, sites, prefactor)
            return s_dumb, c_dumb

        # Apply Op 1
        s_arr1, c_arr1  = op1_fun(state, ns, [sites[0]], prefactor)
        coeff1          = c_arr1[0]
        
        if coeff1 == 0.0 or (coeff1 == 0.0 + 0.0j):
            return s_arr1, c_arr1 * 0.0
            
        s_int           = s_arr1[0]
        s_arr2, c_arr2  = op2_fun(s_int, ns, [sites[1]], prefactor)
        
        # Explicitly cast both to complex before multiplying to be 100% safe
        return s_arr2, c_arr1 * c_arr2
    return mixed_int

def create_mixed_np(op1_fun, op2_fun):
    """
    Create a mixed numpy operator from two operator functions.
    """
    
    @numba.njit(inline='always')
    def mixed_np(state, sites, prefactor=1.0):
        
        # dumbo
        if sites is None or len(sites) < 2:
            s_dumb, c_dumb = op1_fun(state, sites, prefactor)
            return s_dumb, c_dumb

        # Apply Op 1
        s_arr1, c_arr1  = op1_fun(state, [sites[0]], prefactor)
        coeff1          = c_arr1[0]
        
        if coeff1 == 0.0 or (coeff1 == 0.0 + 0.0j):
            return s_arr1, c_arr1 * 0.0
            
        s_np            = s_arr1[0, :]
        s_arr2, c_arr2  = op2_fun(s_np, [sites[1]], prefactor)
        
        # Explicitly cast both to complex before multiplying to be 100% safe
        return s_arr2, c_arr1 * c_arr2
    return mixed_np

def create_mixed_jnp(op1_fun, op2_fun):
    """
    Create a mixed jax operator from two operator functions.
    """
    
    if not JAX_AVAILABLE:
        def _stub(*args, **kwargs):
            raise ImportError("JAX is not available.")
        return _stub

    import jax
    
    @jax.jit
    def mixed_jnp(state, ns, sites, prefactor=1.0):
        # dumbo
        if sites is None or len(sites) < 2:
            s_dumb, c_dumb = op1_fun(state, ns, sites, prefactor)
            return s_dumb, c_dumb

        # Apply Op 1
        s_arr1, c_arr1  = op1_fun(state, ns, [sites[0]], prefactor)
        coeff1          = c_arr1[0]
        
        if coeff1 == 0.0 or (coeff1 == 0.0 + 0.0j):
            return s_arr1, c_arr1 * 0.0
            
        s_jnp           = s_arr1[0]
        s_arr2, c_arr2  = op2_fun(s_jnp, ns, [sites[1]], prefactor)
        
        # Explicitly cast both to complex before multiplying to be 100% safe
        return s_arr2, c_arr1 * c_arr2
    return mixed_jnp

c_dag_c         = create_mixed_int(c_dag_int_np, c_int_np)
c_c_dag         = create_mixed_int(c_int_np, c_dag_int_np)
c_dag_n         = create_mixed_int(c_dag_int_np, n_int_np)
n_c_dag         = create_mixed_int(n_int_np, c_dag_int_np)
c_n             = create_mixed_int(c_int_np, n_int_np)
n_c             = create_mixed_int(n_int_np, c_int_np)

#? NP
c_dag_c_np      = create_mixed_np(c_dag_np, c_np)
c_c_dag_np      = create_mixed_np(c_np, c_dag_np)
c_dag_n_np      = create_mixed_np(c_dag_np, n_np)
n_c_dag_np      = create_mixed_np(n_np, c_dag_np)
c_n_np          = create_mixed_np(c_np, n_np)
n_c_np          = create_mixed_np(n_np, c_np)

#? JNP
c_dag_c_jnp     = create_mixed_jnp(c_dag_jnp, c_jnp)
c_c_dag_jnp     = create_mixed_jnp(c_jnp, c_dag_jnp)
c_dag_n_jnp     = create_mixed_jnp(c_dag_jnp, n_jax)
n_c_dag_jnp     = create_mixed_jnp(n_jax, c_dag_jnp)
c_n_jnp         = create_mixed_jnp(c_jnp, n_jax)
n_c_jnp         = create_mixed_jnp(n_jax, c_jnp)

def make_fermionic_mixed(name, lattice=None, ns=None, type_act='global', sites=None) -> Operator:
    """
    Factory for common fermionic mixed operators.
    """

    if sites is not None and len(sites) != 2:
        raise ValueError("Mixed fermionic operators require exactly two sites.")
    
    lookup = {
        "c_dag_c"   : (c_dag_c, c_dag_c_np, c_dag_c_jnp),
        "c_c_dag"   : (c_c_dag, c_c_dag_np, c_c_dag_jnp),
        "c_dag_n"   : (c_dag_n, c_dag_n_np, c_dag_n_jnp),
        "n_c_dag"   : (n_c_dag, n_c_dag_np, n_c_dag_jnp),
        "c_n"       : (c_n, c_n_np, c_n_jnp),
        "n_c"       : (n_c, n_c_np, n_c_jnp),
    }
    
    if name not in lookup:
        raise ValueError(f"Unknown mixed fermionic operator: {name}")
    
    op_func_int, op_func_np, op_func_jnp = lookup[name]
    return create_operator(
        type_act    = OperatorTypeActing(type_act),
        op_func_int = op_func_int,
        op_func_np  = op_func_np,
        op_func_jnp = op_func_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (),
        name        = name,
        modifies    = True,
        code        = FermionLookupCodes[name] if name in FermionLookupCodes.__members__ else None
    )


################################################################################
#! Fermion Composition Function (for Hamiltonian matrix building)
################################################################################

def fermion_composition_integer(is_complex: bool = True, only_apply: bool = False) -> Callable:
    """
    Creates a JIT-compiled fermion operator composition kernel.
    
    This function returns a numba-compiled composition function that evaluates
    the action of multiple fermion operators on a basis state. It's used for
    efficient Hamiltonian matrix construction.
    
    Parameters
    ----------
    is_complex : bool
        Whether to use complex coefficients (default True).
    only_apply : bool, optional
        If True, returns only the single-operator apply function for reuse.
        
    Returns
    -------
    Callable
        A numba-compiled composition function with signature:
        (state, nops, codes, sites, coeffs, ns, states_out, vals_out) -> (states, vals)
        
        If only_apply=True, returns:
        (state, code, site1, site2, coeff, ns) -> (new_state, coeff, is_diagonal)
    
    Example
    -------
    >>> comp_func = fermion_composition_integer(is_complex=True)
    >>> states, vals = comp_func(state, n_ops, codes, sites, coeffs, ns, states_buf, vals_buf)
    """
    dtype = np.complex128 if is_complex else np.float64
    
    @numba.njit(nogil=True)
    def fermion_operator_composition_single_op(state: int, code: int, site1: int, site2: int, ns: int):
        """Applies a single fermion operator based on the code."""
        
        current_s   = state
        current_c   = dtype(1.0 + 0.0j)
        is_diagonal = False
        
        # Extract sites
        s1          = site1
        s2          = site2
        
        #! Skip global operators (not supported in this composition)
        if code <= 10:
            raise ValueError("Global operators not allowed in composition function.")
        
        #! LOCAL OPERATORS
        if code == FermionLookupCodes.c_dag_local:
            # Creation operator at single site
            pos = ns - 1 - s1
            if _bit(current_s, pos):
                current_c   = dtype(0.0)
            else:
                sign        = fermionic_parity_int(current_s, ns, s1)
                current_s   = _flip(current_s, pos)
                current_c   = dtype(sign)
                
        elif code == FermionLookupCodes.c_ann_local:
            # Annihilation operator at single site
            pos = ns - 1 - s1
            if not _bit(current_s, pos):
                current_c   = dtype(0.0)
            else:
                sign        = fermionic_parity_int(current_s, ns, s1)
                current_s   = _flip(current_s, pos)
                current_c   = dtype(sign)
                
        elif code == FermionLookupCodes.n_local:
            # Number operator (diagonal)
            pos             = ns - 1 - s1
            is_diagonal     = True
            current_c       = dtype(_bit(current_s, pos))
            
        #! CORRELATION OPERATORS (two-site)
        elif code == FermionLookupCodes.c_dag_c_corr:
            # Hopping term: c^dag _i c_j
            # First apply c_j (annihilation at s2)
            pos2            = ns - 1 - s2
            if not _bit(current_s, pos2):
                current_c   = dtype(0.0)
            else:
                sign2       = fermionic_parity_int(current_s, ns, s2)
                current_s   = _flip(current_s, pos2)
                # Then apply c^dag _i (creation at s1)
                pos1        = ns - 1 - s1
                if _bit(current_s, pos1):
                    current_c   = dtype(0.0)
                else:
                    sign1       = fermionic_parity_int(current_s, ns, s1)
                    current_s   = _flip(current_s, pos1)
                    current_c   = dtype(sign1 * sign2)
                    
        elif code == FermionLookupCodes.c_c_dag_corr:
            # c_i c^dag _j
            # First apply c^dag _j (creation at s2)
            pos2            = ns - 1 - s2
            if _bit(current_s, pos2):
                current_c   = dtype(0.0)
            else:
                sign2       = fermionic_parity_int(current_s, ns, s2)
                current_s   = _flip(current_s, pos2)
                # Then apply c_i (annihilation at s1)
                pos1        = ns - 1 - s1
                if not _bit(current_s, pos1):
                    current_c   = dtype(0.0)
                else:
                    sign1       = fermionic_parity_int(current_s, ns, s1)
                    current_s   = _flip(current_s, pos1)
                    current_c   = dtype(sign1 * sign2)
                    
        elif code == FermionLookupCodes.n_n_corr:
            # Density-density: n_i n_j (diagonal)
            is_diagonal         = True
            pos1                = ns - 1 - s1
            pos2                = ns - 1 - s2
            current_c           = dtype(_bit(current_s, pos1) * _bit(current_s, pos2))
            
        elif code == FermionLookupCodes.c_dag_c_dag_corr:
            # Pairing: c^dag _i c^dag _j
            # First apply c^dag _j
            pos2 = ns - 1 - s2
            if _bit(current_s, pos2):
                current_c   = dtype(0.0)
            else:
                sign2       = fermionic_parity_int(current_s, ns, s2)
                current_s   = _flip(current_s, pos2)
                # Then apply c^dag _i
                pos1        = ns - 1 - s1
                if _bit(current_s, pos1):
                    current_c   = dtype(0.0)
                else:
                    sign1       = fermionic_parity_int(current_s, ns, s1)
                    current_s   = _flip(current_s, pos1)
                    current_c   = dtype(sign1 * sign2)
                    
        elif code == FermionLookupCodes.c_c_corr:
            # Pairing: c_i c_j
            # First apply c_j
            pos2 = ns - 1 - s2
            if not _bit(current_s, pos2):
                current_c = dtype(0.0)
            else:
                sign2 = fermionic_parity_int(current_s, ns, s2)
                current_s = _flip(current_s, pos2)
                # Then apply c_i
                pos1 = ns - 1 - s1
                if not _bit(current_s, pos1):
                    current_c = dtype(0.0)
                else:
                    sign1 = fermionic_parity_int(current_s, ns, s1)
                    current_s = _flip(current_s, pos1)
                    current_c = dtype(sign1 * sign2)
        
        elif code == FermionLookupCodes.c_n_corr:
            # c_i n_j
            pos2            = ns - 1 - s2
            if not _bit(current_s, pos2):
                current_c   = dtype(0.0)
            else:
                sign2       = fermionic_parity_int(current_s, ns, s2)
                current_s   = _flip(current_s, pos2)
                # Then apply c_i
                pos1        = ns - 1 - s1
                if not _bit(current_s, pos1):
                    current_c   = dtype(0.0)
                else:
                    sign1       = fermionic_parity_int(current_s, ns, s1)
                    current_s   = _flip(current_s, pos1)
                    current_c   = dtype(sign1 * sign2)
            
        elif code == FermionLookupCodes.n_c_corr:
            # n_i c_j
            pos1            = ns - 1 - s1
            if not _bit(current_s, pos1):
                current_c   = dtype(0.0)
            else:
                sign1       = fermionic_parity_int(current_s, ns, s1)
                current_s   = _flip(current_s, pos1)
                # Then apply c_j
                pos2        = ns - 1 - s2
                if not _bit(current_s, pos2):
                    current_c   = dtype(0.0)
                else:
                    sign2       = fermionic_parity_int(current_s, ns, s2)
                    current_s   = _flip(current_s, pos2)
                    current_c   = dtype(sign1 * sign2)
                    
        elif code == FermionLookupCodes.n_c_dag_corr:
            # n_i c^dag _j
            pos1            = ns - 1 - s1
            if not _bit(current_s, pos1):
                current_c   = dtype(0.0)
            else:
                sign1       = fermionic_parity_int(current_s, ns, s1)
                current_s   = _flip(current_s, pos1)
                # Then apply c^dag _j
                pos2        = ns - 1 - s2
                if _bit(current_s, pos2):
                    current_c   = dtype(0.0)
                else:
                    sign2       = fermionic_parity_int(current_s, ns, s2)
                    current_s   = _flip(current_s, pos2)
                    current_c   = dtype(sign1 * sign2)
                    
        elif code == FermionLookupCodes.c_dag_n_corr:
            # c^dag _i n_j
            pos2            = ns - 1 - s2
            if not _bit(current_s, pos2):
                current_c   = dtype(0.0)
            else:
                sign2       = fermionic_parity_int(current_s, ns, s2)
                current_s   = _flip(current_s, pos2)
                # Then apply c^dag _i
                pos1        = ns - 1 - s1
                if _bit(current_s, pos1):
                    current_c   = dtype(0.0)
                else:
                    sign1       = fermionic_parity_int(current_s, ns, s1)
                    current_s   = _flip(current_s, pos1)
                    current_c   = dtype(sign1 * sign2)        
            
        else:
            # Unknown operator code
            pass
        
        return current_s, current_c, is_diagonal
    
    if only_apply:
        return fermion_operator_composition_single_op
    
    @numba.njit(nogil=True)
    def fermion_operator_composition_int(state: int, nops: int, codes, sites, coeffs, ns: int, out_states, out_vals):
        """Apply a sequence of fermion operators to a basis state."""
        ptr         = 0
        diag_sum    = 0.0 + 0.0j if is_complex else 0.0

        for k in range(nops):
            code        = codes[k]
            coeff       = coeffs[k]
            s1          = sites[k, 0]
            s2          = sites[k, 1] if sites.shape[1] > 1 else -1
            
            current_s, current_c, is_diagonal = fermion_operator_composition_single_op(state, code, s1, s2, ns)
            
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
            out_states[ptr] = state
            out_vals[ptr]   = diag_sum
            ptr            += 1
        
        return out_states[:ptr], out_vals[:ptr]
    
    return fermion_operator_composition_int

def fermion_composition_with_custom(is_complex: bool, custom_op_funcs: Dict[int, Callable], custom_op_arity: Dict[int, int]) -> Callable:
    """
    Creates a fermion composition kernel that supports both predefined and custom operators.
    
    This function creates a hybrid composition kernel that:
    1. Uses fast predefined lookups for standard fermion operators (codes < 1000)
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
    Custom operators support arbitrary arity (up to 8 sites).
    The sites tuple passed to custom operators will have length equal to the operator's arity.
    
    Example
    -------
    >>> # Register a 3-site custom fermion operator
    >>> @numba.njit
    ... def custom_op_int(state, ns, sites):
    ...     # Custom logic here
    ...     return new_state, coeff
    >>> 
    >>> custom_funcs = {1000: custom_op_int}
    >>> custom_arity = {1000: 3}
    >>> comp_func = fermion_composition_with_custom(True, custom_funcs, custom_arity)
    """
    dtype       = np.complex128 if is_complex else np.float64
    n_custom    = len(custom_op_funcs)
    
    if n_custom == 0:
        return fermion_composition_integer(is_complex)
    
    # Create arrays for custom operators
    from numba.typed import List as NList
    custom_codes_arr    = np.array(sorted(custom_op_funcs.keys()), dtype=np.int64)
    custom_arity_arr    = np.array([custom_op_arity.get(code, 1) for code in custom_codes_arr], dtype=np.int32)
    
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
    
    # Get the predefined operator apply function from fermion_composition_integer
    apply_predefined_fermion_op = fermion_composition_integer(is_complex, only_apply=True)
    
    @numba.njit(nogil=True)
    def fermion_composition_with_custom_int(state: int, nops: int, codes, sites, coeffs, ns: int, out_states, out_vals):
        """
        Composition function for fermions with custom operator support.
        Supports arbitrary arity for custom operators (up to 8 sites).
        """
        ptr         = 0
        diag_sum    = 0.0 + 0.0j if is_complex else 0.0

        for k in range(nops):
            code        = codes[k]
            coeff       = coeffs[k]
            
            current_s   = state
            current_c   = dtype(1.0)
            is_diagonal = False
            handled     = False
            
            # Check for custom operators first (codes >= CUSTOM_OP_BASE)
            if code >= CUSTOM_OP_BASE:
                idx = find_custom_idx(code, custom_codes_arr)
                if idx >= 0:
                    custom_func = custom_funcs_list[idx]
                    arity       = custom_arity_arr[idx]
                    
                    # Build sites tuple based on arity (up to 8 sites)
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
                    else:
                        sites_tuple = (sites[k, 0], sites[k, 1], sites[k, 2], sites[k, 3], sites[k, 4], sites[k, 5], sites[k, 6], sites[k, 7])
                    
                    s_arr, c_arr    = custom_func(state, ns, sites_tuple)
                    current_s       = s_arr
                    current_c       = dtype(c_arr)
                    handled         = True
            else:
                # Try predefined operators
                current_s, current_c, is_diagonal   = apply_predefined_fermion_op(state, code, sites[k, 0], sites[k, 1], ns)
                handled                             = True
            
            if not handled:
                continue
            
            # Multiply by coefficient
            final_val = current_c * coeff
            if abs(final_val) < 1e-15:
                continue
            
            # Accumulate results
            if is_diagonal:
                diag_sum += final_val
            else:
                out_states[ptr] = current_s
                out_vals[ptr]   = final_val
                ptr += 1
        
        # Save diagonal contribution
        if abs(diag_sum) > 1e-15:
            out_states[ptr] = state
            out_vals[ptr]   = diag_sum
            ptr += 1
        
        return out_states[:ptr], out_vals[:ptr]
    
    return fermion_composition_with_custom_int

################################################################################
# Registration with the operator catalog
################################################################################

def _register_catalog_entries():
    """
    Register canonical spinless fermion operators with the global catalog.
    """

    def _creation_factory(statistics_angle: float = np.pi) -> LocalOpKernels:
        r''' Creation operator c\dag factory '''
        def _int_kernel(state, ns, sites):
            out_state, out_coeff    = hardcore_create_int(state, ns, sites, statistics_angle)
            coeff                   = np.real_if_close(out_coeff).astype(_DEFAULT_FLOAT, copy=False)
            return out_state, coeff

        def _np_kernel(state, sites):
            out_state, out_coeff    = hardcore_create_np(state, sites, statistics_angle)
            coeff                   = np.real_if_close(out_coeff).astype(_DEFAULT_FLOAT, copy=False)
            return out_state, coeff

        if JAX_AVAILABLE:
            def _jax_kernel(state, sites):
                return c_dag_jnp(state, sites, pref=1.0)
        else:
            _jax_kernel = None

        return LocalOpKernels(
            fun_int         =   _int_kernel,
            fun_np          =   _np_kernel,
            fun_jax         =   _jax_kernel,
            site_parity     =   1,
            modifies_state  =   True,
        )

    def _annihilation_factory(statistics_angle: float = np.pi) -> LocalOpKernels:
        r''' Annihilation operator c_i factory '''
        
        def _int_kernel(state, ns, sites):
            out_state, out_coeff    = hardcore_annihilate_int(state, ns, sites, statistics_angle)
            coeff                   = np.real_if_close(out_coeff).astype(_DEFAULT_FLOAT, copy=False)
            return out_state, coeff

        def _np_kernel(state, sites):
            out_state, out_coeff    = hardcore_annihilate_np(state, sites, statistics_angle)
            coeff                   = np.real_if_close(out_coeff).astype(_DEFAULT_FLOAT, copy=False)
            return out_state, coeff

        if JAX_AVAILABLE:
            def _jax_kernel(state, sites):
                return c_jnp(state, sites, pref=1.0)
        else:
            _jax_kernel = None

        return LocalOpKernels(
            fun_int         =   _int_kernel,
            fun_np          =   _np_kernel,
            fun_jax         =   _jax_kernel,
            site_parity     =   1,
            modifies_state  =   True,
        )

    def _number_factory() -> LocalOpKernels:
        r''' Number operator n_i = c\dag_i c_i factory '''
        
        def _int_kernel(state, ns, sites):
            out_state, out_coeff = hardcore_number_int(state, ns, sites)
            return out_state, out_coeff.astype(_DEFAULT_FLOAT, copy=False)

        def _np_kernel(state, sites):
            out_state, out_coeff = hardcore_number_np(state, sites)
            return out_state, out_coeff.astype(_DEFAULT_FLOAT, copy=False)

        if JAX_AVAILABLE:
            def _jax_kernel(state, sites):
                return n_jax(state, sites, pref=1.0)
        else:
            _jax_kernel = None

        return LocalOpKernels(
            fun_int         =   _int_kernel,
            fun_np          =   _np_kernel,
            fun_jax         =   _jax_kernel,
            site_parity     =   1,
            modifies_state  =   False,
        )

    # --------------------------------------------------------------------------
    #! Register operators in the global catalog
    # --------------------------------------------------------------------------

    register_local_operator(
        LocalSpaceTypes.SPINLESS_FERMIONS,
        key             =   "cdag",
        factory         =   _creation_factory,
        description     =   r"Fermionic creation operator c\dag_i acting on the computational basis.",
        algebra         =   r"{c_i, c_j\dag} = \delta _{ij}",
        sign_convention =   "Jordan-Wigner string counting occupied sites to the left of i.",
        tags            =   ("fermion", "creation"),
    )

    register_local_operator(
        LocalSpaceTypes.SPINLESS_FERMIONS,
        key             =   "c",
        factory         =   _annihilation_factory,
        description     =   r"Fermionic annihilation operator c_i.",
        algebra         =   r"{c_i, c_j\dag} = \delta _{ij}",
        sign_convention =   "Jordan-Wigner string counting occupied sites to the left of i.",
        tags            =   ("fermion", "annihilation"),
    )

    register_local_operator(
        LocalSpaceTypes.SPINLESS_FERMIONS,
        key             =   "n",
        factory         =   _number_factory,
        description     =   r"Onsite fermion number operator n_i = c\dag_i c_i.",
        algebra         =   r"[n_i, c_j\dag] = \delta _{ij} c_j\dag,  [n_i, c_j] = -\delta _{ij} c_j",
        sign_convention =   "No additional phase; diagonal in occupation basis.",
        tags            =   ("fermion", "number"),
    )

# Execute registration upon module import
_register_catalog_entries()

##############################################################################
#! End of file
##############################################################################
