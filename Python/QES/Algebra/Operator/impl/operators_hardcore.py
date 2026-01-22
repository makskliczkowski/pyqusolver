"""
Generic hard-core particle operators with configurable exchange statistics.

The helpers in this module implement creation, annihilation and number operators
that act on Fock states encoded either as integers (bit-strings) or as NumPy
occupation arrays.  The statistics are captured through the ``statistics_angle``
parameter:

* ``statistics_angle = np.pi`` reproduces fermions (Jordan-Wigner sign).
* ``statistics_angle = 0`` yields hard-core bosons.
* Other values describe abelian anyons (e.g. ``np.pi/2`` for semions).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from QES.Algebra.Operator import sign as op_sign
from QES.Algebra.Operator.operator import ensure_operator_output_shape_numba
from QES.general_python.algebra.utils import (
    DEFAULT_NP_CPX_TYPE,
    DEFAULT_NP_FLOAT_TYPE,
    DEFAULT_NP_INT_TYPE,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ensure_tuple_sites(sites: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(s) for s in sites))


def _zero_int_response(state: int):
    out_state = np.array([state], dtype=DEFAULT_NP_INT_TYPE)
    out_coeff = np.array([0.0], dtype=DEFAULT_NP_CPX_TYPE)
    return out_state, out_coeff


def _zero_np_response(state: np.ndarray):
    out_state = state.copy()
    out_coeff = np.array([0.0], dtype=DEFAULT_NP_FLOAT_TYPE)
    return ensure_operator_output_shape_numba(out_state, out_coeff)


# ---------------------------------------------------------------------------
# Integer basis
# ---------------------------------------------------------------------------


def hardcore_create_int(
    state: int,
    ns: int,
    sites: Sequence[int],
    statistics_angle: float,
):
    """
    Create hard-core particles on ``sites`` for an integer-encoded state.
    """
    sites = _ensure_tuple_sites(sites)
    coeff = 1.0 + 0j
    new_state = state
    for site in sites:
        pos = ns - 1 - site
        if ((new_state >> pos) & 1) != 0:
            return _zero_int_response(state)
        phase = op_sign.fractional_statistics_phase(
            op_sign.count_left_int(new_state, ns, site),
            statistics_angle,
        )
        coeff *= phase
        new_state ^= 1 << pos

    out_state = np.array([new_state], dtype=DEFAULT_NP_INT_TYPE)
    out_coeff = np.array([coeff], dtype=DEFAULT_NP_CPX_TYPE)
    return out_state, out_coeff


def hardcore_annihilate_int(
    state: int,
    ns: int,
    sites: Sequence[int],
    statistics_angle: float,
):
    """
    Annihilate hard-core particles on ``sites`` for an integer-encoded state.
    """
    sites = _ensure_tuple_sites(sites)
    coeff = 1.0 + 0j
    new_state = state
    for site in sites:
        pos = ns - 1 - site
        if ((new_state >> pos) & 1) == 0:
            return _zero_int_response(state)
        phase = op_sign.fractional_statistics_phase(
            op_sign.count_left_int(new_state, ns, site),
            -statistics_angle,
        )
        coeff *= phase
        new_state ^= 1 << pos

    out_state = np.array([new_state], dtype=DEFAULT_NP_INT_TYPE)
    out_coeff = np.array([coeff], dtype=DEFAULT_NP_CPX_TYPE)
    return out_state, out_coeff


def hardcore_number_int(state: int, ns: int, sites: Sequence[int]):
    """
    Number operator; returns coefficients equal to the occupancy of ``sites``.
    """
    sites = _ensure_tuple_sites(sites)
    occ = 0.0
    for site in sites:
        occ += float(op_sign.occupation_number_from_bits(state, ns, site))
    out_state = np.array([state], dtype=DEFAULT_NP_INT_TYPE)
    out_coeff = np.array([occ], dtype=DEFAULT_NP_FLOAT_TYPE)
    return out_state, out_coeff


# ---------------------------------------------------------------------------
# NumPy occupation arrays
# ---------------------------------------------------------------------------


def hardcore_create_np(state: np.ndarray, sites: Sequence[int], statistics_angle: float):
    """
    Create hard-core particles on ``sites`` for a numpy occupation array.
    """
    sites = _ensure_tuple_sites(sites)
    coeff = 1.0 + 0j
    new_state = state.copy()
    for site in sites:
        if new_state[site] > 0:
            return _zero_np_response(state)
        phase = op_sign.fractional_statistics_phase(
            op_sign.count_left_array(new_state, site),
            statistics_angle,
        )
        coeff *= phase
        new_state[site] = 1
    out_coeff = np.array([coeff], dtype=DEFAULT_NP_CPX_TYPE)
    return ensure_operator_output_shape_numba(new_state, out_coeff)


def hardcore_annihilate_np(state: np.ndarray, sites: Sequence[int], statistics_angle: float):
    """
    Annihilate hard-core particles on ``sites`` for a numpy occupation array.
    """
    sites = _ensure_tuple_sites(sites)
    coeff = 1.0 + 0j
    new_state = state.copy()
    for site in sites:
        if new_state[site] == 0:
            return _zero_np_response(state)
        phase = op_sign.fractional_statistics_phase(
            op_sign.count_left_array(new_state, site),
            -statistics_angle,
        )
        coeff *= phase
        new_state[site] = 0
    out_coeff = np.array([coeff], dtype=DEFAULT_NP_CPX_TYPE)
    return ensure_operator_output_shape_numba(new_state, out_coeff)


def hardcore_number_np(state: np.ndarray, sites: Sequence[int]):
    """
    Number operator for numpy occupation arrays.
    """
    sites = _ensure_tuple_sites(sites)
    occ = float(np.sum(state[list(sites)]))
    out_coeff = np.array([occ], dtype=DEFAULT_NP_FLOAT_TYPE)
    return ensure_operator_output_shape_numba(state.copy(), out_coeff)


__all__ = [
    "hardcore_annihilate_int",
    "hardcore_annihilate_np",
    "hardcore_create_int",
    "hardcore_create_np",
    "hardcore_number_int",
    "hardcore_number_np",
]
