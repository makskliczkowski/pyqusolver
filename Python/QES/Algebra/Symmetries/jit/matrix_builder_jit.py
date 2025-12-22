'''
JIT-compiled matrix builder applying an operator in a projected compact basis.

This module provides a Numba JIT-compiled function to efficiently apply an operator
to a set of input vectors in a reduced basis using symmetry projections.
The implementation leverages parallel processing and thread-local buffers for performance.

This is a twin module to QES/Algebra/Hilbert/matrix_builder.py but adapted for
the projected compact basis using symmetry groups.

------------------------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-05
Copyright       : (c) 2025-2026 Quantum EigenSolver
License         : MIT
------------------------------------------------------------------------------------------
'''

from    __future__      import annotations
import  numpy           as np
import  scipy.sparse    as sp
from    typing          import Callable, Optional, Union, Tuple, TYPE_CHECKING

try:
    import numba
    from numba.typed    import List
    NUMBA_AVAILABLE     = True
except ImportError:
    NUMBA_AVAILABLE     = False
    raise ImportError("Numba is required for JIT-compiled matrix building.")

# ------------------------------------------------------------------------------------------

try:
    from QES.Algebra.Symmetries.jit.symmetry_container_jit import apply_group_element_compiled, _INVALID_REPR_IDX_NB
except ImportError as e:
    raise ImportError("QES.Algebra.Symmetries.symmetry_container module is required for matrix building: " + str(e))

# ------------------------------------------------------------------------------------------

@numba.njit(fastmath=True, parallel=True, nogil=True)
def _apply_op_batch_projected_compact_jit(
        vecs_in                 : np.ndarray,          # (nh_in, n_batch)
        vecs_out                : np.ndarray,          # (nh_out, n_batch)
        op_func                 : Callable,            # (state, *args) -> (new_states, values)
        args                    : Tuple,

        # Input basis
        representative_list_in  : np.ndarray,           # int64[nh_in]
        normalization_in        : np.ndarray,           # float64[nh_in]

        # Output basis mapping
        repr_map_out            : np.ndarray,           # uint32[nh_full]
        normalization_out       : np.ndarray,           # float64[nh_out]
        representative_list_out : np.ndarray,           # int64[nh_out]

        # compiled symmetry group apply (Input sector characters)
        ns                      : np.int64,
        n_group                 : np.int64,
        n_ops                   : np.ndarray,
        # ... operator encoding ...                     -> needs to match apply_group_element_compiled
        op_code                 : np.ndarray,
        arg0                    : np.ndarray,
        arg1                    : np.ndarray,
        chi_in                  : np.ndarray,           # complex128[n_group]  (character per group element for INPUT sector)
        chi_out                 : np.ndarray,           # complex128[n_group]  (character per group element for OUTPUT sector)

        # symmetry data
        trans_perm              : np.ndarray,
        trans_cross_mask        : np.ndarray,
        refl_perm               : np.ndarray,
        inv_perm                : np.ndarray,
        parity_axis             : np.ndarray,
        boundary_phase          : np.ndarray,
        # ... this list may grow in future ...

        *,
        chunk_size              : int = 4,
        thread_buffers          : Optional[np.ndarray] = None
    ) -> None:
    '''
    Apply operator in projected compact basis using symmetry group. This function
    is necessary when the operator does not commute with the symmetry group. This means
    that we need to expand the ket side using the projector (orbit sum), apply the operator,
    and then project the bra side back to the representative basis.
    
    Supports applying operator between different symmetry sectors (hilbert_in -> hilbert_out).
    '''

    nh_in, n_batch      = vecs_in.shape
    nh_out              = vecs_out.shape[0]
    n_threads           = numba.get_num_threads()

    chunk_size          = min(chunk_size, n_batch)
    bufs                = thread_buffers
    
    # Buffer needs to match OUTPUT shape
    if bufs is None or bufs.shape[0] < n_threads or bufs.shape[1] != nh_out or bufs.shape[2] < chunk_size:
        bufs            = np.zeros((n_threads, nh_out, chunk_size), dtype=vecs_out.dtype)

    # Precompute normalization factor 1/|G|
    # matrix element = 1/(|G| * N_j * N_k) * sum_{g,h} ...
    inv_n_group         = 1.0 / float(n_group)

    for b_start in range(0, n_batch, chunk_size):
        b_end           = min(b_start + chunk_size, n_batch)
        actual_w        = b_end - b_start
        bufs[:n_threads, :, :actual_w].fill(0.0)

        # Loop over reduced basis representatives (INPUT)
        for k in numba.prange(nh_in):
            tid         = numba.get_thread_id()
            rep         = np.int64(representative_list_in[k])
            norm_k      = normalization_in[k]
            if norm_k == 0.0:
                continue

            # Expand the ket using the projector (orbit sum)
            # |k> ~ (1/norm_k) * Σ_g chi_in[g] * g|rep>
            for g in range(n_group):
                s, ph_g = apply_group_element_compiled(rep, ns, np.int64(g), n_ops, op_code, arg0, arg1,
                    trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)

                # ket weight
                # |k> = 1/N_k * sum ...
                # We use conj(chi) because P = sum chi* U
                w_g                 = np.conj(chi_in[g]) * ph_g / norm_k 
                new_states, values  = op_func(s, *args)

                for i in range(len(new_states)):
                    new_state = new_states[i]
                    val       = values[i]
                    if abs(val) < 1e-15:
                        continue
                    
                    # Output projection: sum_h chi_out^*(h) U(h) |new_state>
                    for h in range(n_group):
                        s_out, ph_h = apply_group_element_compiled(new_state, ns, np.int64(h), n_ops, op_code, arg0, arg1,
                                            trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
                        
                        # Check if s_out is a representative
                        idx = repr_map_out[s_out]
                        if idx == _INVALID_REPR_IDX_NB:
                            continue
                            
                        # Double check if it is indeed the representative (it should be if idx is valid and maps to itself)
                        # We use representative_list_out to be sure
                        if representative_list_out[idx] != s_out:
                            continue
                        
                        norm_new  = normalization_out[idx]
                        if norm_new == 0.0:
                            continue
                        
                        # Weight for this term
                        # 1/(|G| * N_j) * chi_out^*(h) * ph_h
                        w_h     = np.conj(chi_out[h]) * ph_h * inv_n_group / norm_new
                        factor  = val * w_g * w_h

                        for b in range(actual_w):
                            bufs[tid, idx, b] += factor * vecs_in[k, b_start + b]

        # Reduction
        for t in range(n_threads):
            for b in range(actual_w):
                vecs_out[:, b_start + b] += bufs[t, :, b]

@numba.njit(fastmath=True, parallel=True, nogil=True)
def _apply_fourier_batch_projected_compact_jit(
        vecs_in                 : np.ndarray,          # (nh_in, n_batch)
        vecs_out                : np.ndarray,          # (nh_out, n_batch)
        phases                  : np.ndarray,          # (ns,) complex128
        op_func                 : Callable,            # (state, site_idx) -> (new_states, values)

        # Input basis
        representative_list_in  : np.ndarray,           # int64[nh_in]
        normalization_in        : np.ndarray,           # float64[nh_in]

        # Output basis mapping
        repr_map_out            : np.ndarray,           # uint32[nh_full]
        normalization_out       : np.ndarray,           # float64[nh_out]
        representative_list_out : np.ndarray,           # int64[nh_out]

        # compiled symmetry group apply (Input sector characters)
        ns                      : np.int64,
        n_group                 : np.int64,
        n_ops                   : np.ndarray,
        # ... operator encoding ...                     -> needs to match apply_group_element_compiled
        op_code                 : np.ndarray,
        arg0                    : np.ndarray,
        arg1                    : np.ndarray,
        chi_in                  : np.ndarray,           # complex128[n_group]
        chi_out                 : np.ndarray,           # complex128[n_group]

        # symmetry data
        trans_perm              : np.ndarray,
        trans_cross_mask        : np.ndarray,
        refl_perm               : np.ndarray,
        inv_perm                : np.ndarray,
        parity_axis             : np.ndarray,
        boundary_phase          : np.ndarray,

        *,
        chunk_size              : int = 4,
        thread_buffers          : Optional[np.ndarray] = None
    ) -> None:
    '''
    Apply Fourier operator in projected compact basis using symmetry group.
    Sum over sites and project.
    '''

    nh_in, n_batch      = vecs_in.shape
    nh_out              = vecs_out.shape[0]
    n_sites             = len(phases)
    n_threads           = numba.get_num_threads()

    chunk_size          = min(chunk_size, n_batch)
    bufs                = thread_buffers
    
    if bufs is None or bufs.shape[0] < n_threads or bufs.shape[1] != nh_out or bufs.shape[2] < chunk_size:
        bufs            = np.zeros((n_threads, nh_out, chunk_size), dtype=vecs_out.dtype)

    inv_n_group         = 1.0 / float(n_group)

    for b_start in range(0, n_batch, chunk_size):
        b_end           = min(b_start + chunk_size, n_batch)
        actual_w        = b_end - b_start
        bufs[:n_threads, :, :actual_w].fill(0.0)

        for k in numba.prange(nh_in):
            tid         = numba.get_thread_id()
            rep         = np.int64(representative_list_in[k])
            norm_k      = normalization_in[k]
            if norm_k == 0.0:
                continue

            for g in range(n_group):
                s, ph_g = apply_group_element_compiled(rep, ns, np.int64(g), n_ops, op_code, arg0, arg1,
                    trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)

                w_g = np.conj(chi_in[g]) * ph_g / norm_k 

                # Sum over sites for Fourier
                for site_idx in range(n_sites):
                    c_site = phases[site_idx]
                    
                    new_states, values = op_func(s, site_idx)

                    for i in range(len(new_states)):
                        new_state = new_states[i]
                        val       = values[i]
                        
                        fourier_val = val * c_site
                        if abs(fourier_val) < 1e-15:
                            continue
                        
                        for h in range(n_group):
                            s_out, ph_h = apply_group_element_compiled(new_state, ns, np.int64(h), n_ops, op_code, arg0, arg1,
                                                trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
                            
                            idx = repr_map_out[s_out]
                            if idx == _INVALID_REPR_IDX_NB:
                                continue
                            
                            if representative_list_out[idx] != s_out:
                                continue
                            
                            norm_new = normalization_out[idx]
                            if norm_new == 0.0:
                                continue
                            
                            w_h     = np.conj(chi_out[h]) * ph_h * inv_n_group / norm_new
                            factor  = fourier_val * w_g * w_h

                            for b in range(actual_w):
                                bufs[tid, idx, b] += factor * vecs_in[k, b_start + b]

        for t in range(n_threads):
            for b in range(actual_w):
                vecs_out[:, b_start + b] += bufs[t, :, b]

@numba.njit(nogil=True)
def _build_sparse_projected_jit(
        rows                    : np.ndarray,
        cols                    : np.ndarray,
        data                    : np.ndarray,
        data_idx                : int,
        
        op_func                 : Callable,
        
        representative_list_in  : np.ndarray,
        normalization_in        : np.ndarray,
        
        repr_map_out            : np.ndarray,
        normalization_out       : np.ndarray,
        representative_list_out : np.ndarray,
        
        ns                      : np.int64,
        n_group                 : np.int64,
        n_ops                   : np.ndarray,
        op_code                 : np.ndarray,
        arg0                    : np.ndarray,
        arg1                    : np.ndarray,
        chi_in                  : np.ndarray,
        chi_out                 : np.ndarray,
        
        trans_perm              : np.ndarray,
        trans_cross_mask        : np.ndarray,
        refl_perm               : np.ndarray,
        inv_perm                : np.ndarray,
        parity_axis             : np.ndarray,
        boundary_phase          : np.ndarray
    ):
    '''
    Sparse matrix builder for projected operators.
    '''
    
    nh_in = len(representative_list_in)
    inv_n_group = 1.0 / float(n_group)
    
    for k in range(nh_in):
        rep = np.int64(representative_list_in[k])
        norm_k = normalization_in[k]
        if norm_k == 0.0:
            continue
            
        for g in range(n_group):
            s, ph_g = apply_group_element_compiled(rep, ns, np.int64(g), n_ops, op_code, arg0, arg1,
                trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
            
            w_g = np.conj(chi_in[g]) * ph_g / norm_k
            
            new_states, values = op_func(s)
            
            for i in range(len(new_states)):
                new_state = new_states[i]
                val = values[i]
                if np.abs(val) < 1e-15:
                    continue
                
                for h in range(n_group):
                    s_out, ph_h = apply_group_element_compiled(new_state, ns, np.int64(h), n_ops, op_code, arg0, arg1,
                        trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
                    
                    idx = repr_map_out[s_out]
                    if idx == _INVALID_REPR_IDX_NB:
                        continue
                    
                    if representative_list_out[idx] != s_out:
                        continue
                        
                    norm_new = normalization_out[idx]
                    if norm_new == 0.0:
                        continue
                        
                    w_h = np.conj(chi_out[h]) * ph_h * inv_n_group / norm_new
                    factor = val * w_g * w_h
                    
                    if np.abs(factor) < 1e-15:
                        continue
                        
                    rows[data_idx] = idx
                    cols[data_idx] = k
                    data[data_idx] = factor
                    data_idx += 1
                    
    return data_idx

@numba.njit(nogil=True)
def _build_dense_projected_jit(
        matrix                  : np.ndarray,
        
        op_func                 : Callable,
        
        representative_list_in  : np.ndarray,
        normalization_in        : np.ndarray,
        
        repr_map_out            : np.ndarray,
        normalization_out       : np.ndarray,
        representative_list_out : np.ndarray,
        
        ns                      : np.int64,
        n_group                 : np.int64,
        n_ops                   : np.ndarray,
        op_code                 : np.ndarray,
        arg0                    : np.ndarray,
        arg1                    : np.ndarray,
        chi_in                  : np.ndarray,
        chi_out                 : np.ndarray,
        
        trans_perm              : np.ndarray,
        trans_cross_mask        : np.ndarray,
        refl_perm               : np.ndarray,
        inv_perm                : np.ndarray,
        parity_axis             : np.ndarray,
        boundary_phase          : np.ndarray
    ):
    '''
    Dense matrix builder for projected operators.
    '''
    
    nh_in = len(representative_list_in)
    inv_n_group = 1.0 / float(n_group)
    
    for k in range(nh_in):
        rep = np.int64(representative_list_in[k])
        norm_k = normalization_in[k]
        if norm_k == 0.0:
            continue
            
        for g in range(n_group):
            s, ph_g = apply_group_element_compiled(rep, ns, np.int64(g), n_ops, op_code, arg0, arg1,
                trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
            
            w_g = np.conj(chi_in[g]) * ph_g / norm_k
            
            new_states, values = op_func(s)
            
            for i in range(len(new_states)):
                new_state = new_states[i]
                val = values[i]
                if np.abs(val) < 1e-15:
                    continue
                
                for h in range(n_group):
                    s_out, ph_h = apply_group_element_compiled(new_state, ns, np.int64(h), n_ops, op_code, arg0, arg1,
                        trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
                    
                    idx = repr_map_out[s_out]
                    if idx == _INVALID_REPR_IDX_NB:
                        continue
                    
                    if representative_list_out[idx] != s_out:
                        continue
                        
                    norm_new = normalization_out[idx]
                    if norm_new == 0.0:
                        continue
                        
                    w_h = np.conj(chi_out[h]) * ph_h * inv_n_group / norm_new
                    factor = val * w_g * w_h
                    
                    if np.abs(factor) < 1e-15:
                        continue
                        
                    matrix[idx, k] += factor
