'''

'''

import  numba
import  numpy   as np
from    typing  import TYPE_CHECKING

try:
    from .symmetry_container_jit                    import apply_group_element_compiled
except Exception as e:
    raise RuntimeError(f"Couldn't import symmetries correctly...")

if TYPE_CHECKING:
    from QES.Algebra.hilbert                        import HilbertSpace
    from QES.Algebra.Symmetries.symmetry_container  import SymmetryContainer

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
    K       = keys.shape[0]
    h       = _hash64(b_key)
    idx     = np.int64(h % K)

    while True:
        if used[idx] == 0:
            used[idx]   = 1
            keys[idx]   = b_key
            
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
    K    = vals.shape[0]
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
    c_red               : np.ndarray,   # (n_rep,) complex128 - states
    rep_list            : np.ndarray,   # (n_rep,) int64
    rep_norm            : np.ndarray,   # (n_rep,) float64

    ns                  : np.int64,
    local_dim           : np.int64,
    va                  : np.int64,

    # compiled group
    n_group             : np.int64,
    n_ops               : np.ndarray,
    op_code             : np.ndarray,
    arg0                : np.ndarray,
    arg1                : np.ndarray,
    chi                 : np.ndarray,

    # tables
    trans_perm          : np.ndarray,
    trans_cross_mask    : np.ndarray,
    refl_perm           : np.ndarray,
    inv_perm            : np.ndarray,
    parity_axis         : np.ndarray,
    boundary_phase      : np.ndarray,

    cache_size          : np.int64,     # number of b-columns to cache
    flush_load_factor   : np.float64    = 0.5 # e.g. 0.5
) -> np.ndarray:
    """
    Build reduced density matrix rhoA of subsystem A (first va sites)
    directly from symmetry-reduced coefficients.

    Returns: rhoA (dimA, dimA) complex128
    """
    dimA            = local_dim ** va
    rhoA            = np.zeros((dimA, dimA), dtype=np.complex128)

    # cache structures
    K               = int(cache_size)
    keys            = np.empty(K, dtype=np.int64)
    used            = np.zeros(K, dtype=np.uint8)
    vals            = np.zeros((K, dimA), dtype=np.complex128)

    filled          = 0
    limit           = int(flush_load_factor * K)
    if limit < 1:   limit = 1

    # Loop over reps and group elements, spray amplitudes into columns
    n_rep = rep_list.shape[0]
    for r_idx in range(n_rep):

        nr = rep_norm[r_idx]
        if nr <= 0.0:
            continue

        coeff = c_red[r_idx] / np.sqrt(nr)
        if coeff == 0.0j:
            continue

        rstate = np.int64(rep_list[r_idx])

        for g in range(n_group):

            # apply group element to repstate
            s, ph = apply_group_element_compiled(
                rstate, ns, np.int64(g),
                n_ops, op_code, arg0, arg1,
                trans_perm, trans_cross_mask,
                refl_perm, inv_perm,
                parity_axis, boundary_phase
            )

            # amplitude contribution: coeff * conj(chi[g]) * phase
            amp = coeff * np.conj(chi[g]) * ph
            if amp == 0.0j:
                continue

            # General splitting for any local_dim
            # s = a + b * dimA
            a = np.int64(s % dimA)
            b = np.int64(s // dimA)

            row, inserted = _cache_find_or_insert(keys, used, vals, b, dimA)
            if inserted:
                filled += 1
                # flush if too full
                if filled >= limit:
                    _flush_cache_into_rhoA(rhoA, used, vals)
                    filled = 0

            vals[row, a] += amp

    # final flush
    _flush_cache_into_rhoA(rhoA, used, vals)

    return rhoA

# ---------------------------------------------

def rho_symmetries(state, va, hilbert: 'HilbertSpace', cache_size=256):
    ''' 
    Symmetric density of states
    '''
    local_dim = hilbert.local_space.local_dim
    
    if not hilbert.has_sym:
        from QES.general_python.physics.density_matrix import rho_numpy
        dimA = local_dim ** va
        dimB = hilbert.nh // dimA
        return rho_numpy(state, dimA, dimB)
    
    container               = hilbert.sym_container
    if container is None:   raise RuntimeError("The SymmetryContainer is necessary for density matrices with symmetries...")
    cg                      = container._compiled_group
    tb                      = container._tables
    cd                      = container._compact_data

    rhoA                    = _rho_symmetries(
                                c_red               = np.asarray(state, dtype=np.complex128),
                                rep_list            = np.asarray(cd.representative_list, dtype=np.int64),
                                rep_norm            = np.asarray(cd.normalization, dtype=np.float64),
                                ns                  = np.int64(container.ns),
                                local_dim           = np.int64(local_dim),
                                va                  = np.int64(va),

                                n_group             = np.int64(cg.n_group),
                                n_ops               = cg.n_ops,
                                op_code             = cg.op_code,
                                arg0                = cg.arg0,
                                arg1                = cg.arg1,
                                chi                 = cg.chi,

                                trans_perm          = tb.trans_perm,
                                trans_cross_mask    = tb.trans_cross_mask,
                                refl_perm           = tb.refl_perm,
                                inv_perm            = tb.inv_perm,
                                parity_axis         = tb.parity_axis,
                                boundary_phase      = tb.boundary_phase,

                                cache_size          = np.int64(cache_size)
                            )
    return rhoA