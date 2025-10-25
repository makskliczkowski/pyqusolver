"""
Shared utilities for handling fermionic signs and occupation counting.

The helper functions defined here are used across multiple operator modules to
ensure consistent and JIT-friendly implementations of Jordan-Wigner phases,
parity checks, and popcount operations.

File        : QES/Algebra/Operator/phase_utils.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
"""

from __future__ import annotations
import numpy as np
import numba

__all__ = [
    "bit_popcount_mask",
    "bit_popcount",
    "fermionic_parity_int",
    "fermionic_parity_array",
]

@numba.njit(inline="always")
def bit_popcount_mask(x: int, mask_bits: int) -> int:
    """
    Return the Hamming weight of ``x & mask_bits``.
    """
    v = x & mask_bits
    count = 0
    while v:
        v &= v - 1
        count += 1
    return count


@numba.njit(inline="always")
def bit_popcount(x: int, ns: int) -> int:
    """
    Return the number of occupied sites in the first ``ns`` bits of ``x``.
    """
    mask = (1 << ns) - 1
    return bit_popcount_mask(x, mask)


@numba.njit
def fermionic_parity_int(state: int, ns: int, site: int) -> float:
    """
    Jordan-Wigner parity for integer-encoded fermionic states.

    Parameters
    ----------
    state:
        Integer encoding of the Fock state.
    ns:
        Number of sites.
    site:
        Site index (0-based) at which the operator acts.
    """
    shift = ns - site
    mask_bits = (1 << shift) - 1
    parity = bit_popcount_mask(state, mask_bits) & 1
    return -1.0 if parity else 1.0


@numba.njit
def fermionic_parity_array(state: np.ndarray, site: int) -> float:
    """
    Jordan-Wigner parity for array-encoded fermionic states.
    """
    parity = 0
    for i in range(site):
        parity ^= int(state[i] > 0)
    return -1.0 if parity else 1.0

