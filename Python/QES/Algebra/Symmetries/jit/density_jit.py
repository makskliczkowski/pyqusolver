""" 
This module implements the reduced density matrix rhoA for subsystem A
directly from symmetry-reduced state vectors, using JIT compilation for performance.
This is highly memory-efficient as it avoids expanding the full state vector.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numba
import numpy as np

try:
    from .symmetry_container_jit                    import _NUMBA_OVERHEAD_SIZE, apply_group_element_fast
    from ....general_python.physics.density_matrix  import rho_spectrum, mask_subsystem
except Exception:
    # Fallback for imports if needed, but in standard repo it should work
    pass

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace

# -----------------------------------------------------------------------------
#! Internal JIT helpers
# -----------------------------------------------------------------------------

@numba.njit(cache=True)
def _hash64(x: np.int64) -> np.int64:
    # Knuth-ish multiplicative hash
    return np.int64(x * np.int64(11400714819323198485))

@numba.njit(cache=True)
def _cache_find_or_insert(keys, used, vals, b_key, dimA):
    """
    Open-addressing linear-probe map: b_key -> row index in vals.
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
            idx = (idx + 1) % K

@numba.njit(cache=True, fastmath=True)
def _flush_cache_into_rhoA(rhoA, used, vals):
    """
    rhoA += sum_{active rows i} vals[i] vals[i]^†
    """
    K       = vals.shape[0]
    dimA    = vals.shape[1]

    for i in range(K):
        if used[i] == 0:
            continue

        # rhoA += v v.H
        for p in range(dimA):
            vp = vals[i, p]
            if vp != 0.0j:
                for q in range(dimA):
                    rhoA[p, q] += vp * np.conj(vals[i, q])
                    
    # clear cache
    used[:] = 0
    vals[:] = 0.0j

# -----------------------------------------------------------------------------
#! Core Symmetry-Reduced RDM Kernels
# -----------------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def _rho_symmetries_kernel(
    c_red               : np.ndarray,
    rep_list            : np.ndarray,
    rep_norm            : np.ndarray,
    ns                  : int,
    local_dim           : int,
    sites_a             : np.ndarray,
    sites_b             : np.ndarray,
    cg_args             : Tuple,
    tb_args             : Tuple,
    cache_size          : int,
    flush_load_factor   : float = 0.5,
    is_contiguous       : bool  = True
) -> np.ndarray:
    """
    Unified kernel for symmetry-reduced RDM.
    """
    va                  = sites_a.shape[0]
    dimA                = np.int64(local_dim**va)
    rhoA                = np.zeros((dimA, dimA), dtype=np.complex128)

    # Unpack group info
    n_group             = cg_args[0]
    chi                 = cg_args[5]

    # cache structures
    K                   = int(cache_size)
    keys                = np.empty(K, dtype=np.int64)
    used                = np.zeros(K, dtype=np.uint8)
    vals                = np.zeros((K, dimA), dtype=np.complex128)

    filled              = 0
    limit               = int(flush_load_factor * K)
    inv_sqrt_G          = 1.0 / np.sqrt(n_group)
    
    # Pre-calculate powers for arbitrary local_dim
    powers = np.zeros(ns, dtype=np.int64)
    for i in range(ns):
        powers[i] = local_dim**i

    # Loop over reps and group elements
    n_rep = rep_list.shape[0]
    for r in range(n_rep):
        Nk = rep_norm[r]
        ck = c_red[r]
        if Nk <= 0.0 or ck == 0.0j:
            continue

        rstate  = np.int64(rep_list[r])
        invNk   = 1.0 / Nk

        for g in range(n_group):
            s, ph   = apply_group_element_fast(rstate, ns, np.int64(g), cg_args, tb_args)
            amp     = ck * np.conj(chi[g]) * ph * invNk * inv_sqrt_G

            if amp == 0.0j:
                continue

            # Extract indices a and b
            if is_contiguous:
                a = np.int64(s % dimA)
                b = np.int64(s // dimA)
            else:
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
                    pa = np.int64(1)
                    for k in range(va):
                        val = (s // powers[sites_a[k]]) % local_dim
                        a += val * pa
                        pa *= local_dim
                    pb = np.int64(1)
                    for k in range(sites_b.shape[0]):
                        val = (s // powers[sites_b[k]]) % local_dim
                        b += val * pb
                        pb *= local_dim

            slot, inserted = _cache_find_or_insert(keys, used, vals, b, dimA)
            if inserted:
                filled += 1
                if filled >= limit:
                    _flush_cache_into_rhoA(rhoA, used, vals)
                    filled = 0
            vals[slot, a] += amp

    _flush_cache_into_rhoA(rhoA, used, vals)
    return rhoA

# -----------------------------------------------------------------------------
#! Public high-level function
# -----------------------------------------------------------------------------

def rho_symmetries(
    state       : np.ndarray,
    va          : Union[int, np.ndarray, List[int]],
    hilbert     : "HilbertSpace",
    cache_size  : int = 512,
    contiguous  : bool = False
) -> np.ndarray:
    """
    Compute reduced density matrix rhoA for a symmetry-reduced state.

    Parameters
    ----------
    state : np.ndarray
        Symmetry-reduced state vector.
    va : int or array-like
        Subsystem specification (contiguous size or explicit indices).
    hilbert : HilbertSpace
        Hilbert space object containing symmetry info.
    cache_size : int
        Size of the hash-map cache for b-columns.
    contiguous : bool
        If va is int, treat as contiguous first `va` sites.

    Returns
    -------
    np.ndarray
        Reduced density matrix of subsystem A.
    """
    local_dim       = hilbert.local_space.local_dim
    ns              = hilbert.lattice.ns

    # Normalize va input
    (size_a, size_b), order = mask_subsystem(va, ns, local_dim, contiguous)
    sites_a         = np.array(order[:size_a], dtype=np.int64)
    sites_b         = np.array(order[size_a:], dtype=np.int64)
    
    # Check if we should fallback to full expansion (for tiny systems or no symmetry)
    if not hilbert.has_sym:
        from ....general_python.physics.density_matrix import rho
        return rho(state, va, ns, local_dim, contiguous)

    container       = hilbert.sym_container
    if container is None:
        raise RuntimeError("SymmetryContainer is required for symmetry-reduced RDMs.")

    cg = container._compiled_group
    tb = container._tables
    cd = container._compact_data

    # Call the JIT kernel
    return _rho_symmetries_kernel(
        c_red=np.asarray(state, dtype=np.complex128),
        rep_list=np.asarray(cd.representative_list, dtype=np.int64),
        rep_norm=np.asarray(cd.normalization, dtype=np.float64),
        ns=ns,
        local_dim=local_dim,
        sites_a=sites_a,
        sites_b=sites_b,
        cg_args=cg.args,
        tb_args=tb.args,
        cache_size=cache_size,
        is_contiguous=contiguous and isinstance(va, (int, np.integer))
    )

# -----------------------------------------------------------------------------
#! End of file
# -----------------------------------------------------------------------------
