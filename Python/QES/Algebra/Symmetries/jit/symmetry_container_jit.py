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
    from QES.Algebra.globals                import GlobalSymmetries, violates_global_syms
    from QES.Algebra.Symmetries.base        import CompiledGroup, SymmetryApplicationCodes, SymOpTables, _popcount64
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

@numba.njit(fastmath=True)
def apply_group_element_compiled(state      : np.int64,     # input state
                                ns          : np.int64,     # number of sites
                                g           : np.int64,     # group element index
                                n_ops       : np.ndarray,   # number of primitive ops per group element
                                op_code     : np.ndarray,   # primitive op codes
                                arg0        : np.ndarray,   # first argument per op
                                arg1        : np.ndarray,   # second argument per op
                                # tables:
                                trans_perm  : np.ndarray,   # (n_trans, ns)         int64       -> translation permutations
                                trans_cross : np.ndarray,   # (n_trans, ns)         uint8/bool  -> translation crossing masks
                                refl_perm   : np.ndarray,   # (n_refl, ns)          int64       -> reflection permutations
                                inv_perm    : np.ndarray,   # (n_inv, ns)           int64       -> inversion permutations
                                parity_axis : np.ndarray,   # (n_parity,)           uint8       -> (0=x,1=y,2=z) -> parity axes
                                bound_phase : np.ndarray    # (directions, ns + 1)  complex128  -> apply boundary phase fluxes
                            ) -> tuple[np.int64, np.complex128]:
    ''' 
    Apply compiled group element to integer state 
    This function applies a group element 'g', which consists of a sequence of primitive symmetry operations,
    to an integer-represented quantum state. It uses precomputed tables for efficiency.
    
    Parameters
    ----------
    state : np.int64
        The input quantum state represented as an integer (bitstring).
    ns : np.int64
        The number of sites in the system.
    g : np.int64
        The index of the group element to apply.
    n_ops : np.ndarray
        Array containing the number of primitive operations for each group element.
    op_code : np.ndarray
        Array containing the operation codes for each primitive operation.
    arg0 : np.ndarray
        First argument for each primitive operation.
    arg1 : np.ndarray
        Second argument for each primitive
        operation.
    trans_perm : np.ndarray
        Precomputed translation permutations
    trans_cross : np.ndarray
        Precomputed translation crossing masks
    refl_perm : np.ndarray
        Precomputed reflection permutations
    parity_axis : np.ndarray
        Precomputed parity axes
    Returns
    -------
    new_state : np.int64
        The transformed quantum state after applying the group element.
    phase : np.complex128
        The accumulated phase factor resulting from the symmetry operations.
    Notes
    -----
    This function relies on primitive operation functions (_apply_translation_prim, _apply_perm_prim, _apply_parity_prim)
    '''
    
    cur     = state                         # current state 
    phase   = np.complex128(1.0 + 0.0j)     # initialize phase
    m       = n_ops[g]                      # number of primitive operations in group element g

    for j in range(m):
        code = op_code[g, j]                # operation code -> which primitive operation to apply
        a0   = arg0[g, j]
        # a1 = arg1[g, j]                   # reserved, use later
        # a2 = arg1[g, j]                   # reserved, use later

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
            #TODO: Add different base...
            cur         = _apply_inversion_permutation(cur, ns, 2, inv_perm[inv_idx])
            phase      *= np.complex128(1.0 + 0.0j)

        else:
            # identity / unknown => do nothing
            pass

    return cur, phase

@numba.njit(fastmath=True)
def _compute_normalization_compiled(
    state               : np.int64,
    ns                  : np.int64,
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
    boundary_phase      : np.ndarray,) -> np.float64:
    """
    Compute normalization factor for a representative state using compiled group.
    This function calculates the normalization factor for a given quantum state
    by summing over the contributions from all group elements in the symmetry group.
    
    Parameters
    ----------
    state : np.int64
        The input quantum state represented as an integer (bitstring).
    ns : np.int64
        The number of sites in the system.
    n_group : np.int64
        The number of group elements in the symmetry group.
    n_ops : np.ndarray 
        Array containing the number of primitive operations for each group element.
    op_code : np.ndarray
        Array containing the operation codes for each primitive operation.
    arg0 : np.ndarray
        First argument for each primitive operation.
    arg1 : np.ndarray
        Second argument for each primitive operation.
    chi : np.ndarray
        Character of each group element in the chosen irrep/sector.
    trans_perm : np.ndarray
        Precomputed translation permutations    
    trans_cross_mask : np.ndarray
        Precomputed translation crossing masks
    refl_perm : np.ndarray
        Precomputed reflection permutations
    inv_perm : np.ndarray
        Precomputed inversion permutations
    parity_axis : np.ndarray
        Precomputed parity axes
    boundary_phase : np.ndarray
        Precomputed boundary phase factors
    Returns
    -------
    normalization : np.float64
        The computed normalization factor for the input state.
    Notes
    -----
    The normalization is computed as the square root of the absolute value of the sum over group elements,
    weighted by the character of each group element.
    """

    proj = np.complex128(0.0 + 0.0j)

    for g in range(n_group):
        new_state, phase = apply_group_element_compiled(
            state, ns, g,
            n_ops,
            op_code,
            arg0,
            arg1,
            trans_perm,
            trans_cross_mask,
            refl_perm,
            inv_perm,
            parity_axis,
            boundary_phase,
        )

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
    # ---
    ns                  : np.int64,
    n_group             : np.int64,
    n_ops               : np.ndarray,
    op_code             : np.ndarray,
    arg0                : np.ndarray,
    arg1                : np.ndarray,
    chi                 : np.ndarray,
    trans_perm          : np.ndarray,
    trans_cross_mask    : np.ndarray,
    refl_perm           : np.ndarray,
    inv_perm            : np.ndarray,
    parity_axis         : np.ndarray,
    boundary_phase      : np.ndarray,
):
    for i in numba.prange(reps.shape[0]):
        out_norm[i] = _compute_normalization_compiled(reps[i], ns, n_group, n_ops, op_code, arg0, arg1, chi,
            trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)

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

    # compiled group
    n_group         : np.int64,
    n_ops           : np.ndarray,
    op_code         : np.ndarray,
    arg0            : np.ndarray,
    arg1            : np.ndarray,
    chi             : np.ndarray,

    # operator tables
    trans_perm      : np.ndarray,
    trans_cross     : np.ndarray,
    refl_perm       : np.ndarray,
    inv_perm        : np.ndarray,
    parity_axis     : np.ndarray,
    bound_phase     : np.ndarray,

    # global constraints
    global_op_codes : np.ndarray,
    global_op_vals  : np.ndarray,

    # phase lookup
    g_to_pidx       : np.ndarray,

    # outputs (preallocated!)
    repr_map        : np.ndarray,
    repr_chunker    : np.ndarray,
    phase_idx       : np.ndarray
    ):
    r''' 
    Generate representative map and phase index for a chunk of states
    Here, we fill only the portion of the representative map and phase index
    corresponding to states in the range [start_state, end_state).
    
    The basis is constructed later, checking the normalization after this map is built already.
    
    Parameters
    ----------
    start_state : np.int64
        The starting integer state (inclusive) for this chunk.
    end_state : np.int64
        The ending integer state (exclusive) for this chunk.
    ns : np.int64
        The number of sites in the system.
    n_group : np.int64
        The number of group elements in the symmetry group.
    n_ops : np.ndarray
        Array containing the number of primitive operations for each group element.
    op_code : np.ndarray
        Array containing the operation codes for each primitive operation.
    arg0 : np.ndarray
        First argument for each primitive operation.
    arg1 : np.ndarray
        Second argument for each primitive operation.
    # ---
    trans_perm : np.ndarray
        Precomputed translation permutations
    trans_cross : np.ndarray
        Precomputed translation crossing masks
    refl_perm : np.ndarray
        Precomputed reflection permutations
    inv_perm : np.ndarray
        Precomputed inversion permutations
    parity_axis : np.ndarray
        Precomputed parity axes
    bound_phase : np.ndarray
        Precomputed boundary phase factors
    # ---
    global_op_codes : np.ndarray
        Codes for global symmetry operations to check.
    global_op_vals : np.ndarray
        Values for global symmetry operations to check.
        
    # ---
    g_to_pidx : np.ndarray
        Mapping from group element index to phase index.
    repr_map : np.ndarray
        Output array to store the representative state for each input state.
    phase_idx : np.ndarray
        Output array to store the phase index for each input state.
    Returns
    -------
    '''

    for state in numba.prange(start_state, end_state):
    # for state in range(start_state, end_state):

        if violates_global_syms(state, global_op_codes, global_op_vals, ns):
            continue

        min_state   = _INT_HUGE
        min_g       = -1
        norm        =  0.0

        for g in range(n_group):
            new_state, phase = apply_group_element_compiled(
                state, ns, g,
                n_ops, op_code, arg0, arg1,
                trans_perm, trans_cross,
                refl_perm, inv_perm,
                parity_axis, bound_phase
            )

            if new_state < min_state:
                min_state = new_state
                min_g     = g
                
            if new_state == state:
                norm += np.conj(chi[g]) * phase
            
        is_correct = np.abs(norm) > _SYM_NORM_THRESHOLD
        if is_correct:
            
            if min_state == state and min_g != _INVALID_REPR_IDX:
                repr_chunker[state - start_state] = True

            repr_map[state]     = min_state 
            phase_idx[state]    = g_to_pidx[min_g]

# ---------------------------
#! End of symmetry_container_jit.py
# ---------------------------