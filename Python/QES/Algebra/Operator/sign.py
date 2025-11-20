"""
Sign and statistics utilities for onsite operators.

The helper functions here standardise how sign factors are computed for
different particle statistics (fermions, anyons, etc.) across the operator
modules.  They encode the convention used throughout QES:

* Sites are labelled ``0`` (left-most) to ``Ns-1`` (right-most).
* Integer-encoded basis states follow the big-endian convention
  ``s = sum_{i=0}^{Ns-1} n_i 2^{Ns-1-i}``.
* The phase acquired when moving past occupied sites depends on how many
  particles sit to the *left* (smaller site index) of the site being acted on.
"""

from __future__ import annotations

import cmath
import numpy as np
from typing import Iterable


def _count_occupied_to_left_bits(state: int, ns: int, site: int) -> int:
    """
    Return the number of occupied sites strictly to the left of ``site`` for
    an integer-encoded occupation state.
    """
    if site <= 0:
        return 0
    if site >= ns:
        return state.bit_count()
    full_mask = (1 << ns) - 1
    right_mask = (1 << (ns - site)) - 1
    mask = full_mask ^ right_mask
    return (state & mask).bit_count()


def _count_occupied_to_left_array(state: np.ndarray, site: int) -> int:
    """
    Return the number of occupied sites strictly to the left of ``site`` for a
    numpy occupation array.
    """
    if site <= 0:
        return 0
    return int(np.count_nonzero(state[:site]))


def count_left_int(state: int, ns: int, site: int) -> int:
    """
    Public helper that exposes :func:`_count_occupied_to_left_bits`.
    """
    return _count_occupied_to_left_bits(state, ns, site)


def count_left_array(state: np.ndarray, site: int) -> int:
    """
    Public helper that exposes :func:`_count_occupied_to_left_array`.
    """
    return _count_occupied_to_left_array(state, site)


def jordan_wigner_parity_int(state: int, ns: int, site: int) -> float:
    """
    Fermionic sign (-1)^{# occupied to the left} for integer-encoded states.
    """
    parity = _count_occupied_to_left_bits(state, ns, site) & 1
    return -1.0 if parity else 1.0


def jordan_wigner_parity_array(state: np.ndarray, site: int) -> float:
    """
    Fermionic sign (-1)^{# occupied to the left} for numpy occupation arrays.
    """
    parity = _count_occupied_to_left_array(state, site) & 1
    return -1.0 if parity else 1.0


def fractional_statistics_phase(count_left: int, statistics_angle: float):
    """
    Phase accumulated when moving past ``count_left`` particles with exchange
    angle ``statistics_angle``.
    """
    return cmath.exp(1j * statistics_angle * count_left)


def anyon_phase_int(state: int, ns: int, site: int, statistics_angle: float):
    """
    Exchange phase for hard-core abelian anyons on integer-encoded states.
    """
    count_left = _count_occupied_to_left_bits(state, ns, site)
    return fractional_statistics_phase(count_left, statistics_angle)


def anyon_phase_array(state: np.ndarray, site: int, statistics_angle: float):
    """
    Exchange phase for hard-core abelian anyons on numpy occupation arrays.
    """
    count_left = _count_occupied_to_left_array(state, site)
    return fractional_statistics_phase(count_left, statistics_angle)


def occupation_number_from_bits(state: int, ns: int, site: int) -> int:
    """
    Return the occupation number (0/1) of ``site`` for an integer-encoded state.
    """
    pos = ns - 1 - site
    return (state >> pos) & 1


def ensure_sorted_sites(sites: Iterable[int]) -> np.ndarray:
    """
    Return a sorted numpy int32 array of site indices.
    """
    return np.array(sorted(int(s) for s in sites), dtype=np.int32)


__all__ = [
    "count_left_array",
    "count_left_int",
    "anyon_phase_array",
    "anyon_phase_int",
    "ensure_sorted_sites",
    "fractional_statistics_phase",
    "jordan_wigner_parity_array",
    "jordan_wigner_parity_int",
    "occupation_number_from_bits",
]
