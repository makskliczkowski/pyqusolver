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
"""

import numpy as np
import numba
from typing import List, Union, Optional, Callable

################################################################################
from QES.Algebra.Operator.operator import (
    Operator, OperatorTypeActing, SymmetryGenerators, create_operator,
    ensure_operator_output_shape_numba
)
from QES.Algebra.Operator.catalog import register_local_operator
from QES.Algebra.Hilbert.hilbert_local import LocalOpKernels, LocalSpaceTypes
from QES.Algebra.Operator.operators_hardcore import (
    hardcore_create_int,
    hardcore_create_np,
    hardcore_annihilate_int,
    hardcore_annihilate_np,
    hardcore_number_int,
    hardcore_number_np,
)
################################################################################

from QES.general_python.common.tests import GeneralAlgebraicTest
from QES.Algebra.Operator.phase_utils import (
    bit_popcount,
    fermionic_parity_int,
    fermionic_parity_array,
)

from QES.general_python.lattices.lattice import Lattice
from QES.general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, DEFAULT_NP_CPX_TYPE
from QES.general_python.common.binary import BACKEND_REPR as _SPIN, BACKEND_DEF_SPIN, JAX_AVAILABLE
import QES.general_python.common.binary as _binary

if JAX_AVAILABLE:
    from QES.Algebra.Operator.operators_spinless_fermions_jax import c_dag_jnp, c_jnp, c_k_jnp, c_k_dag_jnp, n_jax, n_int_jax
else:
    c_dag_jnp = c_jnp = c_k_jnp = c_k_dag_jnp = n_jax = n_jax_int = None

_DEFAULT_INT    = DEFAULT_NP_INT_TYPE
_DEFAULT_FLOAT  = DEFAULT_NP_FLOAT_TYPE
_bit            = _binary.check_int
_flip           = _binary.flip_int

###############################################################################
#! Creation / annihilation on *integer* occupation number
###############################################################################

@numba.njit
def c_dag_int_np(state      : int,
                ns          : int,
                sites       : List[int],
                prefactor   : float = 1.0):
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
        
    out_state       = np.empty(1, dtype=_DEFAULT_INT)
    out_coeff       = np.empty(1, dtype=_DEFAULT_FLOAT)
    out_state[0]    = new_state
    out_coeff[0]    = coeff_val 
    return out_state, out_coeff

@numba.njit
def c_int_np(state       : int,
             ns          : int,
             sites       : int,
             prefactor   : float = 1.0):
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
    
    out_state       = np.empty(1, dtype=_DEFAULT_INT)
    out_coeff       = np.empty(1, dtype=_DEFAULT_FLOAT)
    out_state[0]    = new_state
    out_coeff[0]    = coeff_val
    return out_state, out_coeff

###############################################################################
#!  Creation / annihilation on *NumPy* occupation array
###############################################################################

@numba.njit
def c_dag_np(state      : np.ndarray,
             sites      : np.ndarray,
             prefactor  : float = 1.0):
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
    # return state, sign * prefactor**n_sites

@numba.njit
def c_np(state       : np.ndarray,
         sites       : np.ndarray,
         prefactor   : float = 1.0):
    r"""
    Applies the annihilation operator (c) to a spinless fermion state at a given site.

    Parameters
    ----------
    state : np.ndarray
        The occupation number representation of the fermionic state (1 for occupied, 0 for unoccupied).
    sites : int
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
def c_k_int_np(state      : int,
               ns         : int,
               sites      : List[int],
               k          : float,
               prefactor  : float = 1.0):
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
def c_k_np(state       : np.ndarray,
           sites       : List[int],
           k           : float,
           prefactor   : float = 1.0):
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
    return ensure_operator_output_shape_numba(out_state[:index],
            out_coeff[:index] / np.sqrt(np.max(index, 1)))    
    # return (out_state[:index],
    #         out_coeff[:index] / np.sqrt(max(index, 1)))

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
def n_int_np(state     : int,
             ns        : int,
             sites     : List[int],
             prefactor : float = 1.0):

    coeff_val = 1.0
    for site in sites:
        pos = ns - 1 - site
        if _bit(state, pos) == 0:   # site unoccupied -> result = 0
            coeff_val = 0.0
            break

    n_sites        = len(sites)
    out_state      = np.empty(1, dtype=_DEFAULT_INT)
    out_coeff      = np.empty(1, dtype=_DEFAULT_FLOAT)
    out_state[0]   = state
    out_coeff[0]   = coeff_val * prefactor**n_sites
    return out_state, out_coeff

@numba.njit
def n_np(state      : np.ndarray,
         sites      : np.ndarray,
         prefactor  : float = 1.0):
    coeff_val   = np.ones(1, dtype=_DEFAULT_FLOAT)
    out         = state.copy()
    for site in sites:
        if out[site] == 0:        # unoccupied -> zero immediately
            coeff_val = np.zeros(1, dtype=_DEFAULT_FLOAT)
            break

    n_sites = sites.shape[0]
    return ensure_operator_output_shape_numba(out, coeff_val * prefactor**n_sites)
    # return state, coeff_val * prefactor**n_sites

###############################################################################
#! Public dispatch helpers  (match your \sigma -operator API)
###############################################################################

def c_dag(state       : Union[int, np.ndarray],
          ns          : int,
          sites       : int,
          prefactor   : float = 1.0):
    """Creation operator dispatcher."""
    if isinstance(state, (int, np.integer)):
        return c_dag_int_np(int(state), ns, sites, prefactor)
    if isinstance(state, np.ndarray):
        return c_dag_np(state, sites, prefactor)
    return c_dag_jnp(state, ns, sites, prefactor)

def c(state          : Union[int, np.ndarray],
      ns             : int,
      sites          : int,
      prefactor      : float = 1.0):
    """Annihilation operator dispatcher."""
    if isinstance(state, (int, np.integer)):
        return c_int_np(int(state), ns, sites, prefactor)
    if isinstance(state, np.ndarray):
        return c_np(state, sites, prefactor)
    return c_jnp(state, ns, sites, prefactor)

def c_k(state        : Union[int, np.ndarray],
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

def c_k_dag(state     : Union[int, np.ndarray],
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

def n(  state           : Union[int, np.ndarray],
        ns              : int,
        sites           : Optional[List[int]],
        prefactor       : float = 1.0):
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

def c( lattice  : Optional[Lattice]     = None,
    ns          : Optional[int]         = None,
    type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
    sites       : Optional[List[int]]   = None,
    prefactor   : float                 = 1.0) -> Operator:
    r"""
    Factory for the fermionic annihilation operator c_i.
    """
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
        modifies    = True
    )

def cdag( lattice   : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    r"""
    Factory for the fermionic creation operator c_i\dag.
    """
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
        modifies    = True
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

# ------------------------------------------------------------------------------
#! Number operator factory
# ------------------------------------------------------------------------------

def n( lattice      : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    """
    Factory for the fermionic number operator n_i.
    """
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
        key             =   "c_dag",
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
