'''
Methods for handling global symmetries in symmetry containers using JIT compilation.

---------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-01
License         : MIT
---------------------------
'''

import  numba
import  numpy   as np
from    typing  import List, Tuple, TypeAlias

# --------------------------------------------------------------------------
#! APPLY FUNCTION
# -------------------------------------------------------------------------

try:
    from QES.Algebra.globals                import violates_global_syms
    from QES.Algebra.Symmetries.base        import OP_IDENTITY, OP_TRANSLATION, OP_REFLECTION, OP_PARITY, OP_INVERSION
    from QES.Algebra.Symmetries.translation import _apply_translation_prim
    from QES.Algebra.Symmetries.parity      import _apply_parity_prim
    from QES.Algebra.Symmetries.reflection  import _apply_perm_prim
    from QES.Algebra.Symmetries.inversion   import _apply_inversion_permutation
    # other
except ImportError:
    raise ImportError("Could not import required modules for symmetry application.")

#############################################################################
#! Constants
#############################################################################

_STATE_TYPE             : TypeAlias = np.int64
_STATE_TYPE_NB          : TypeAlias = numba.int64
_REPR_MAP_DTYPE         : TypeAlias = np.uint32
_REPR_MAP_DTYPE_NB      : TypeAlias = numba.uint32
_PHASE_IDX_DTYPE        : TypeAlias = np.uint8
_PHASE_IDX_DTYPE_NB     : TypeAlias = numba.uint8

_INVALID_REPR_IDX_NB    : TypeAlias = numba.uint32(0xFFFFFFFF)  # Max uint32 for numba  (sufficient for up to ~4 billion representatives)
_INVALID_PHASE_IDX_NB   : TypeAlias = numba.uint8(0xFF)         # Max uint8 for numba   (sufficient for up to 255 distinct phases)

_INT_HUGE               = np.int64(0x7FFFFFFFFFFFFFFF)          # THE HUGEST!!!
_INVALID_REPR_IDX       = np.iinfo(_REPR_MAP_DTYPE).max         # ~4 billion, marks state not in sector
_INVALID_PHASE_IDX      = np.iinfo(_PHASE_IDX_DTYPE).max        # 255, marks invalid phase index
_SYM_NORM_THRESHOLD     = 1e-7
_NUMBA_OVERHEAD_SIZE    = 2**24                                 # remove threads for small arrays to check

# -----------------------
#! APPLY GROUP ELEMENT FUNCTION 
# -----------------------

@numba.njit(fastmath=True)
def apply_group_element_compiled(state      : np.int64,     # input state
                                ns          : np.int64,     # number of sites
                                g           : np.int64,     # group element index
                                # ---
                                n_ops       : np.ndarray,   # number of primitive ops per group element
                                op_code     : np.ndarray,   # primitive op codes
                                arg0        : np.ndarray,   # first argument per op
                                arg1        : np.ndarray,   # second argument per op
                                # ---
                                trans_perm  : np.ndarray,   # (n_trans, ns)         int64       -> translation permutations
                                trans_cross : np.ndarray,   # (n_trans, ns)         uint8/bool  -> translation crossing masks
                                refl_perm   : np.ndarray,   # (n_refl, ns)          int64       -> reflection permutations
                                inv_perm    : np.ndarray,   # (n_inv, ns)           int64       -> inversion permutations
                                parity_axis : np.ndarray,   # (n_parity,)           uint8       -> (0=x,1=y,2=z) -> parity axes
                                bound_phase : np.ndarray    # (directions, ns + 1)  complex128  -> apply boundary phase fluxes
                            ) -> tuple[np.int64, np.complex128]:
    ''' 
    Apply compiled group element to integer state 
    '''
    
    cur     = state                         # current state 
    phase   = np.complex128(1.0 + 0.0j)     # initialize phase
    m       = n_ops[g]                      # number of primitive operations in group element g

    for j in range(m):
        code = op_code[g, j]                # operation code -> which primitive operation to apply
        a0   = arg0[g, j]

        if code == OP_TRANSLATION:
            t_idx   = a0                    # translation index from precomputed table
            t_power = arg1[g, j]
            for _ in range(t_power):
                cur, occ = _apply_translation_prim(cur, ns, trans_perm[t_idx], trans_cross[t_idx])
                phase   *= bound_phase[t_idx, occ]

        elif code == OP_REFLECTION:
            r_idx       = a0
            cur         = _apply_perm_prim(cur, ns, refl_perm[r_idx])
            phase      *= np.complex128(1.0 + 0.0j)

        elif code == OP_PARITY:
            p_idx       = a0
            cur, ph     = _apply_parity_prim(cur, ns, parity_axis[p_idx])
            phase      *= ph

        elif code == OP_INVERSION:
            inv_idx     = a0
            cur         = _apply_inversion_permutation(cur, ns, 2, inv_perm[inv_idx])
            phase      *= np.complex128(1.0 + 0.0j)

        else:
            pass

    return cur, phase

@numba.njit(fastmath=True)
def apply_group_element_fast(state, ns, g, cg_args, tb_args):
    r"""
    Wrapper that accepts tuples from CompiledGroup.args and SymOpTables.args
    to make passing data between JIT functions more generic and cleaner.
    """
    
    # Unpack cg_args: (n_group, n_ops, op_code, arg0, arg1, chi)
    n_ops, op_code, arg0, arg1 = cg_args[1], cg_args[2], cg_args[3], cg_args[4]
    
    # Unpack tb_args: (trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
    trans_perm, trans_cross, refl_perm, inv_perm, parity_axis, bound_phase = tb_args
    
    return apply_group_element_compiled(state, ns, g, n_ops, op_code, arg0, arg1,
        trans_perm, trans_cross, refl_perm, inv_perm, parity_axis, bound_phase)

# -----------------------

@numba.njit(fastmath=True)
def _compute_normalization_compiled(
    state               : np.int64,
    ns                  : np.int64,
    n_group             : np.int64,
    cg_args             : Tuple,
    tb_args             : Tuple
) -> np.float64:
    """
    Compute normalization factor for a representative state using compiled group.
    """
    chi  = cg_args[5]
    proj = np.complex128(0.0 + 0.0j)

    for g in range(n_group):
        new_state, phase = apply_group_element_fast(state, ns, g, cg_args, tb_args)

        if new_state == state:
            proj += np.conj(chi[g]) * phase

    val = np.abs(proj)
    if val < _SYM_NORM_THRESHOLD:
        return np.float64(0.0)

    return np.sqrt(val)

@numba.njit(parallel=True, fastmath=True)
def compute_normalization(
    reps                : np.ndarray,
    out_norm            : np.ndarray,
    ns                  : np.int64,
    n_group             : np.int64,
    cg_args             : Tuple,
    tb_args             : Tuple,
):
    for i in numba.prange(reps.shape[0]):
        out_norm[i] = _compute_normalization_compiled(reps[i], ns, n_group, cg_args, tb_args)

# -----------------------

@numba.njit
def fill_representatives(repr_map: np.ndarray, out: np.ndarray):
    k = 0
    for i in range(repr_map.shape[0]):
        if repr_map[i] != _INVALID_REPR_IDX and repr_map[i] == i:
            out[k]  = i
            k      += 1

# -----------------------

@numba.njit(parallel=True, fastmath=True)
def scan_chunk_find_representatives(
    start_state     : np.int64,
    end_state       : np.int64,
    ns              : np.int64,
    n_group         : np.int64,
    cg_args         : Tuple,
    tb_args         : Tuple,
    # global constraints
    global_op_codes : np.ndarray,
    global_op_vals  : np.ndarray,
    # phase lookup
    g_to_pidx       : np.ndarray,
    # outputs
    repr_map        : np.ndarray,
    repr_chunker    : np.ndarray,
    phase_idx       : np.ndarray
    ):
    
    chi = cg_args[5]

    for state in numba.prange(start_state, end_state):
        if violates_global_syms(state, global_op_codes, global_op_vals, ns):
            continue

        min_state   = _INT_HUGE
        min_g       = -1
        norm        = 0.0

        for g in range(n_group):
            new_state, phase = apply_group_element_fast(state, ns, g, cg_args, tb_args)

            if new_state < min_state:
                min_state = new_state
                min_g     = g
                
            if new_state == state:
                norm += np.conj(chi[g]) * phase
            
        if np.abs(norm) > _SYM_NORM_THRESHOLD:
            if min_state == state and min_g != _INVALID_REPR_IDX:
                repr_chunker[state - start_state] = True

            repr_map[state]     = min_state 
            phase_idx[state]    = g_to_pidx[min_g]

# ---------------------------

@numba.njit(parallel=True, fastmath=True)
def expand_to_full_state_jit(
    vec_red             : np.ndarray,
    vec_full            : np.ndarray,
    rep_list            : np.ndarray,
    rep_norm            : np.ndarray,
    ns                  : np.int64,
    cg_args             : Tuple,
    tb_args             : Tuple
) -> None:
    """
    JIT-compiled state expansion: reduced basis -> full basis.
    Supports both single vectors (1D) and multiple vectors (2D batch).
    |ψ> = sum_k (ck / (Nk * sqrt(|G|))) sum_g chi^*(g) U(g) |rk>
    """
    n_group     = cg_args[0]
    chi         = cg_args[5]
    n_rep       = rep_list.shape[0]
    inv_sqrt_G  = 1.0 / np.sqrt(float(n_group))
    
    is_batch    = (vec_red.ndim == 2)
    n_batch     = vec_red.shape[1] if is_batch else 1

    for r_idx in numba.prange(n_rep):
        Nk      = rep_norm[r_idx]
        if Nk <= 0.0: continue
            
        rstate  = rep_list[r_idx]
        invNk   = 1.0 / Nk
        
        # Check if any element in the batch is non-zero for this representative
        any_nonzero = False
        if is_batch:
            for b in range(n_batch):
                if vec_red[r_idx, b] != 0.0j:
                    any_nonzero = True
                    break
        else:
            if vec_red[r_idx] != 0.0j:
                any_nonzero = True
        
        if not any_nonzero:
            continue

        for g in range(n_group):
            s, ph   = apply_group_element_fast(rstate, ns, np.int64(g), cg_args, tb_args)
            # character contribution
            w_g     = np.conj(chi[g]) * ph * invNk * inv_sqrt_G
            
            if is_batch:
                for b in range(n_batch):
                    vec_full[s, b] += vec_red[r_idx, b] * w_g
            else:
                vec_full[s] += vec_red[r_idx] * w_g

# ---------------------------
#! End of symmetry_container_jit.py
# ---------------------------