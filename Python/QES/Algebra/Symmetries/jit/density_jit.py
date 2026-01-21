""" """

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numba
import numpy as np

try:
    # public physics imports

    from .symmetry_container_jit import _NUMBA_OVERHEAD_SIZE, apply_group_element_fast
except Exception:
    raise RuntimeError("Couldn't import symmetries correctly...")

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace

# -----------------------------
# Hash-map cache for columns v_b
# -----------------------------


@numba.njit(cache=True)
def _hash64(x: np.int64) -> np.int64:
    # Knuth-ish multiplicative hash
    return np.int64(x * np.int64(11400714819323198485))


# -----------------------------


@numba.njit(cache=True)
def _cache_find_or_insert(keys, used, vals, b_key, dimA):
    """
    Open-addressing linear-probe map:
        b_key -> row index in vals.
    Returns:
        (row_index, inserted_new: bool)
    """
    K = keys.shape[0]
    h = _hash64(b_key)
    idx = np.int64(h % K)

    while True:
        if used[idx] == 0:
            used[idx] = 1
            keys[idx] = b_key

            return idx, True
        elif keys[idx] == b_key:
            return idx, False

        else:
            idx += 1
            if idx == K:
                idx = 0


# -----------------------------


@numba.njit(cache=True, fastmath=True)
def _flush_cache_into_rhoA(rhoA, used, vals):
    """
    rhoA += sum_{active rows i} vals[i] vals[i]^†
    then clears used/vals.
    """
    K = vals.shape[0]
    dimA = vals.shape[1]

    for i in range(K):
        if used[i] == 0:
            continue

        # v = vals[i, :]
        # rhoA[p,q] += v[p] * conj(v[q])
        for p in range(dimA):
            vp = vals[i, p]
            if vp != 0.0j:
                for q in range(dimA):
                    rhoA[p, q] += vp * np.conj(vals[i, q])

    # clear
    for i in range(K):
        used[i] = 0
        for p in range(dimA):
            vals[i, p] = 0.0j


# ---------------------------------------------
# rhoA from reduced-basis vector c_red
# ---------------------------------------------


@numba.njit(cache=True, fastmath=True)
def _rho_symmetries(
    c_red: np.ndarray,  # (n_rep,) complex128 - states
    basis_args: Tuple,  # (rep_list, rep_norm)
    ns: np.int64,
    local_dim: np.int64,
    va: np.int64,
    cg_args: Tuple,  # (n_group, ...)
    tb_args: Tuple,  # (tables...)
    cache_size: np.int64,  # number of b-columns to cache
    flush_load_factor: np.float64 = 0.5,  # e.g. 0.5
) -> np.ndarray:
    """
    Build reduced density matrix rhoA of subsystem A (first va sites)
    directly from symmetry-reduced coefficients.

    Parameters
    ----------
    c_red               : np.ndarray
        Symmetry-reduced state vector (complex128) of shape (n_rep,).
    basis_args          : Tuple
        Tuple containing (rep_list, rep_norm):
            rep_list : np.ndarray
                List of representative states (int64) of shape (n_rep,).
            rep_norm : np.ndarray
                Normalization factors (float64) of shape (n_rep,).
    ns                  : np.int64
        Number of sites in the full system.
    local_dim           : np.int64
        Local Hilbert space dimension per site.
    va                  : np.int64
        Number of sites in subsystem A.
    cg_args             : Tuple

    """

    rep_list, rep_norm = basis_args
    dimA = np.int64(local_dim**va)
    rhoA = np.zeros((dimA, dimA), dtype=np.complex128)

    # Unpack cg
    n_group, _, _, _, _, chi = cg_args

    # cache structures
    K = int(cache_size)
    keys = np.empty(K, dtype=np.int64)
    used = np.zeros(K, dtype=np.uint8)
    vals = np.zeros((K, dimA), dtype=np.complex128)

    filled = 0
    limit = int(flush_load_factor * K)
    inv_sqrt_G = 1.0 / np.sqrt(n_group)
    if limit < 1:
        limit = 1

    # Loop over reps and group elements, spray amplitudes into columns
    nh_red = rep_list.shape[0]
    for r in range(nh_red):
        Nk = rep_norm[r]

        if Nk <= 0.0:
            continue

        ck = c_red[r]
        if ck == 0.0j:
            continue

        rstate = np.int64(rep_list[r])
        invNk = 1.0 / Nk

        for g in range(n_group):
            s, ph = apply_group_element_fast(rstate, ns, np.int64(g), cg_args, tb_args)
            amp = ck * np.conj(chi[g]) * ph * invNk * inv_sqrt_G

            if amp == 0.0j:
                continue

            a = np.int64(s % dimA)  # index in subsystem A, this is given by taking mod
            b = np.int64(s // dimA)  # index in subsystem B, this is given by integer division
            slot, inserted = _cache_find_or_insert(keys, used, vals, b, dimA)

            if inserted:
                filled += 1

                if filled >= limit:
                    _flush_cache_into_rhoA(rhoA, used, vals)
                    filled = 0

            vals[slot, a] += amp
    _flush_cache_into_rhoA(rhoA, used, vals)
    return rhoA


@numba.njit(cache=True, fastmath=True)
def _rho_symmetries_mask(
    c_red: np.ndarray,  # (n_rep,) complex128 - states
    basis_args: Tuple,  # (rep_list, rep_norm)
    ns: np.int64,
    local_dim: np.int64,
    sites_a: np.ndarray,  # indices of sites in A
    sites_b: np.ndarray,  # indices of sites in B
    cg_args: Tuple,  # (n_group, ...)
    tb_args: Tuple,  # (tables...)
    cache_size: np.int64,  # number of b-columns to cache
    flush_load_factor: np.float64 = 0.5,  # e.g. 0.5
) -> np.ndarray:
    """
    Build reduced density matrix rhoA of subsystem A (explicitly given sites)
    directly from symmetry-reduced coefficients.

    It is assumed that sites_a and sites_b form a partition of the full system.
    Parameters
    ----------
    c_red               : np.ndarray
        Symmetry-reduced state vector (complex128) of shape (n_rep,).
    basis_args          : Tuple
        Tuple containing (rep_list, rep_norm):
            rep_list : np.ndarray
                List of representative states (int64) of shape (n_rep,).
            rep_norm : np.ndarray
                Normalization factors (float64) of shape (n_rep,).
    ns                  : np.int64
        Number of sites in the full system.
    local_dim           : np.int64
        Local Hilbert space dimension per site.
    sites_a             : np.ndarray
        Indices of sites in subsystem A.
    sites_b             : np.ndarray
        Indices of sites in subsystem B.
    cg_args             : Tuple
    tb_args             : Tuple
    cache_size          : np.int64
        Number of b-columns to cache.
    flush_load_factor   : np.float64
        Load factor to trigger cache flush.
    Returns
    -------
    rhoA                : np.ndarray
        Reduced density matrix of subsystem A (complex128) of shape (dimA, dimA),
        where dimA = local_dim ** len(sites_a).
    """

    rep_list, rep_norm = basis_args
    va = sites_a.shape[0]
    dimA = np.int64(local_dim**va)
    rhoA = np.zeros((dimA, dimA), dtype=np.complex128)

    # Unpack cg
    n_group, _, _, _, _, chi = cg_args

    # cache structures
    K = int(cache_size)
    keys = np.empty(K, dtype=np.int64)
    used = np.zeros(K, dtype=np.uint8)
    vals = np.zeros((K, dimA), dtype=np.complex128)

    filled = 0
    limit = int(flush_load_factor * K)
    inv_sqrt_G = 1.0 / np.sqrt(n_group)
    if limit < 1:
        limit = 1

    # Loop over reps and group elements, spray amplitudes into columns
    nh_red = rep_list.shape[0]
    for r in range(nh_red):
        Nk = rep_norm[r]
        if Nk <= 0.0:
            continue
        ck = c_red[r]
        if ck == 0.0j:
            continue

        rstate = np.int64(rep_list[r])
        invNk = 1.0 / Nk

        for g in range(n_group):
            s, ph = apply_group_element_fast(rstate, ns, np.int64(g), cg_args, tb_args)
            amp = ck * np.conj(chi[g]) * ph * invNk * inv_sqrt_G
            if amp == 0.0j:
                continue

            # Extract index a and b from s
            a = np.int64(0)
            b = np.int64(0)
            if local_dim == 2:
                for k in range(va):
                    if (s >> sites_a[k]) & 1:
                        a |= np.int64(1) << k
                for k in range(sites_b.shape[0]):
                    if (s >> sites_b[k]) & 1:
                        b |= np.int64(1) << k
            else:
                p_a = np.int64(1)
                for k in range(va):
                    val = (s // (local_dim ** sites_a[k])) % local_dim
                    a += val * p_a
                    p_a *= local_dim
                p_b = np.int64(1)
                for k in range(sites_b.shape[0]):
                    val = (s // (local_dim ** sites_b[k])) % local_dim
                    b += val * p_b
                    p_b *= local_dim

            slot, inserted = _cache_find_or_insert(keys, used, vals, b, dimA)
            if inserted:
                filled += 1
                if filled >= limit:
                    _flush_cache_into_rhoA(rhoA, used, vals)
                    filled = 0
            vals[slot, a] += amp

    # final flush
    _flush_cache_into_rhoA(rhoA, used, vals)
    return rhoA


# ---------------------------------------------
#! Public function
# ---------------------------------------------


def rho_symmetries(
    state,
    va: Union[int, np.ndarray, List[int]],
    hilbert: "HilbertSpace",
    cache_size=256,
    full_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Symmetric density of states rhoA for subsystem A,
    from symmetry-reduced state vector.
    ---------------------------------------------
    Parameters:
    state       : np.ndarray
        Symmetry-reduced state vector (complex128).
    va          : int or array-like
        Number of sites in subsystem A (assumed first va sites)
        OR explicit site indices of subsystem A.
    hilbert     : HilbertSpace
        Hilbert space with symmetries.
    """
    local_dim = hilbert.local_space.local_dim
    ns = hilbert.lattice.ns

    # determine if we use mask or contiguous
    is_contiguous = isinstance(va, (int, np.integer))

    if not is_contiguous:
        sites_a = np.sort(np.asarray(va, dtype=np.int64))
        va_size = sites_a.shape[0]
        # check if it's actually contiguous from 0
        if va_size > 0 and np.all(sites_a == np.arange(va_size, dtype=np.int64)):
            is_contiguous = True
            va = va_size
        else:
            # complement
            mask = np.ones(ns, dtype=np.bool_)
            mask[sites_a] = False
            sites_b = np.arange(ns, dtype=np.int64)[mask]
    else:
        va_size = va

    if hilbert.nhfull <= _NUMBA_OVERHEAD_SIZE or not hilbert.has_sym:
        from QES.general_python.physics.density_matrix import rho_numba_mask, rho_numpy

        if hilbert.has_sym:
            final_state = hilbert.expand_state(state)
        else:
            final_state = state

        if is_contiguous:
            dimA = local_dim**va
            dimB = hilbert.nhfull // dimA
            return rho_numpy(final_state, dimA, dimB)
        else:
            order = tuple(sites_a) + tuple(sites_b)
            psi_reshaped = rho_numba_mask(final_state, order, va_size)
            return psi_reshaped @ psi_reshaped.conj().T

    container = hilbert.sym_container
    if container is None:
        raise RuntimeError(
            "The SymmetryContainer is necessary for density matrices with symmetries..."
        )
    cg = container._compiled_group
    tb = container._tables
    cd = container._compact_data

    if is_contiguous:
        rhoA = _rho_symmetries(
            c_red=np.asarray(state, dtype=np.complex128),
            basis_args=(
                np.asarray(cd.representative_list, dtype=np.int64),
                np.asarray(cd.normalization, dtype=np.float64),
            ),
            ns=np.int64(container.ns),
            local_dim=np.int64(local_dim),
            va=np.int64(va),
            cg_args=cg.args,
            tb_args=tb.args,
            cache_size=np.int64(cache_size),
        )
    else:
        rhoA = _rho_symmetries_mask(
            c_red=np.asarray(state, dtype=np.complex128),
            basis_args=(
                np.asarray(cd.representative_list, dtype=np.int64),
                np.asarray(cd.normalization, dtype=np.float64),
            ),
            ns=np.int64(container.ns),
            local_dim=np.int64(local_dim),
            sites_a=sites_a,
            sites_b=sites_b,
            cg_args=cg.args,
            tb_args=tb.args,
            cache_size=np.int64(cache_size),
        )
    return rhoA


# ---------------------------------------------
# End of file
# ---------------------------------------------
