
import numpy as np
import numba
from .operators_spin import _sigma_x_core, _sigma_y_core, _sigma_z_core

@numba.njit(parallel=True, fastmath=True)
def batch_spin_correlation_kernel(
    eigenvectors,       # (nh, n_states) complex128
    basis_states,       # (nh,) int64 or -1 if not available
    lookup_map,         # (size_space,) int32 or -1 if not using direct lookup
    ns,
    site_i,             # fixed site i
    op_codes_list,      # (n_ops, 2) int32
    spin_val,
    use_direct_lookup   # bool
):
    """
    Computes <psi | S_i^a S_j^b | psi> for a batch of states.
    Returns: (n_ops, ns, n_states) complex128 array.
    """
    nh, n_batch = eigenvectors.shape
    n_ops = op_codes_list.shape[0]
    
    results = np.zeros((n_ops, ns, n_batch), dtype=np.complex128)
    
    # Pre-compute loop constants
    # We loop over:
    # 1. Batch of states (parallel)
    # 2. Site j
    # 3. Operators (xx, xy...)
    # 4. Basis states (contraction)
    
    for b in numba.prange(n_batch):
        vec = eigenvectors[:, b]
        
        for row in range(nh):
            val_row = vec[row]
            if val_row == 0.0j:
                continue
            
            # Basis state
            if basis_states.ndim == 1:
                state_row = basis_states[row]
            else:
                # Fallback if basis_states is not passed (e.g. standard sorted)
                state_row = row 
            
            for j in range(ns):
                for k in range(n_ops):
                    code1 = op_codes_list[k, 0]
                    code2 = op_codes_list[k, 1]
                    
                    # Apply first operator at site i
                    curr_state = state_row
                    curr_coeff = 1.0 + 0.0j
                    
                    # X=0, Y=1, Z=2
                    if code1 == 0:   # X
                        curr_state, cf = _sigma_x_core(curr_state, ns, (site_i,), spin_val)
                        curr_coeff *= cf
                    elif code1 == 1: # Y
                        curr_state, cf = _sigma_y_core(curr_state, ns, (site_i,), spin_val)
                        curr_coeff *= cf
                    elif code1 == 2: # Z
                        curr_state, cf = _sigma_z_core(curr_state, ns, (site_i,), spin_val)
                        curr_coeff *= cf
                    
                    # Op 2 (at site j)
                    if code2 == 0:   # X
                        curr_state, cf = _sigma_x_core(curr_state, ns, (j,), spin_val)
                        curr_coeff *= cf
                    elif code2 == 1: # Y
                        curr_state, cf = _sigma_y_core(curr_state, ns, (j,), spin_val)
                        curr_coeff *= cf
                    elif code2 == 2: # Z
                        curr_state, cf = _sigma_z_core(curr_state, ns, (j,), spin_val)
                        curr_coeff *= cf
                    
                    # Find column index
                    col = -1
                    if use_direct_lookup:
                        col = lookup_map[curr_state]
                    else:
                        # Binary search
                        # Assume basis_states is sorted
                        col = np.searchsorted(basis_states, curr_state)
                        if col >= nh or basis_states[col] != curr_state:
                            col = -1
                    
                    if col != -1:
                        # <psi | O | psi> term
                        # term = conj(vec[col]) * coeff * vec[row]
                        # Correct logic:
                        # O |n> = c |m>  => <m|O|n> = c
                        # <psi|O|psi> = sum_{nm} conj(v_m) <m|O|n> v_n
                        #             = sum_{nm} conj(v_m) c delta(m, target) v_n
                        #             = conj(v_target) * c * v_n
                        
                        term = np.conjugate(vec[col]) * val_row * curr_coeff
                        results[k, j, b] += term

    return results
