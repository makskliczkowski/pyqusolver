"""
Matrix construction supporting:
- Binary search for O(log Nh) representative lookup
- Sector-changing operators
- Symmetry rotation matrices
- Memory-efficient implementation
- Correct symmetry eigenvalue conjugation matching C++ implementation
    - see QES C++ HilbertSpace::generateMat for reference

--------------------------------------------------------------------------------------------
file        : QES/Algebra/Hilbert/matrix_builder.py
description : Optimized matrix construction for symmetry-reduced Hilbert spaces
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
date        : 2025-10-29
version     : 2.2.0
--------------------------------------------------------------------------------------------
"""

from    __future__      import annotations
import  numpy           as np
import  scipy.sparse    as sp
from    typing          import Callable, Optional, Union, Tuple, TYPE_CHECKING

try:
    import numba
    from numba.typed import List
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    raise ImportError("Numba is required for JIT-compiled matrix building.")

# ------------------------------------------------------------------------------------------

try:
    from QES.Algebra.Symmetries.symmetry_container import _binary_search_representative_list, _INVALID_REPR_IDX_NB, _NUMBA_OVERHEAD_SIZE
except ImportError as e:
    raise ImportError("QES.Algebra.Symmetries.symmetry_container module is required for matrix building: " + str(e))

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace

# ------------------------------------------------------------------------------------------
#! Helper functions
# ------------------------------------------------------------------------------------------

def _determine_matrix_dtype(dtype: np.dtype, hilbert_spaces, operator_func=None, ns=None) -> np.dtype:
    """
    Determine the appropriate dtype for matrix storage based on symmetries and operator output.
    
    Parameters
    ----------
    dtype : np.dtype
        Requested dtype
    hilbert_spaces : HilbertSpace or list of HilbertSpace
        The Hilbert space(s) involved
    operator_func : Callable, optional
        Operator function to test for complex output
    ns : int, optional
        Number of sites for operator test
        
    Returns
    -------
    np.dtype
        The dtype to use for matrix storage
    """
    if isinstance(hilbert_spaces, list):
        h_spaces    = hilbert_spaces
    else:
        h_spaces    = [hilbert_spaces]
    
    is_real_dtype           = np.issubdtype(dtype, np.floating)
    alloc_dtype             = dtype
    
    # Check for complex symmetries
    has_complex_sym         = any(getattr(hs, 'has_complex_symmetries', False) for hs in h_spaces)
    
    # Check repr_phase arrays
    repr_phase_is_complex   = False
    for hs in h_spaces:
        # Corrected: access from HilbertSpace's compact_data
        compact_data = getattr(hs, 'compact_data', None)
        if compact_data is not None:
            if np.iscomplexobj(compact_data.phase_table):
                repr_phase_is_complex = True
                break
    
    # Check Hilbert space dtypes
    hilbert_dtype_is_complex = False
    for hs in h_spaces:
        hd = getattr(hs, '_dtype', None)
        if hd is not None:
            try:
                if np.issubdtype(np.dtype(hd), np.complexfloating):
                    hilbert_dtype_is_complex = True
                    break
            except:
                if isinstance(hd, (complex, np.complexfloating)):
                    hilbert_dtype_is_complex = True
                    break
    
    # Check if any Hilbert space has symmetries (forces complex for proper normalization)
    has_symmetries = any(getattr(hs, 'representative_list', None) is not None for hs in h_spaces)
    
    # Check operator output
    operator_returns_complex = False
    if operator_func is not None and ns is not None:
        try:
            # Need a dummy state to call the operator_func
            # Using 0 is fine for most spin-1/2 operators.
            test_state = 0
            # operator_func expects (state, ns)
            _, test_values = operator_func(test_state, ns) # Corrected call, pass ns
            if len(test_values) > 0 and np.iscomplexobj(test_values[0]):
                operator_returns_complex = True
        except Exception as e:
            # Handle cases where operator_func might require other args or fail for test_state=0
            # print(f"Warning: Could not determine operator_func's output type for complex check: {e}")
            pass
    
    # Force complex if needed
    if not is_real_dtype or has_complex_sym or repr_phase_is_complex or hilbert_dtype_is_complex or has_symmetries or operator_returns_complex:
        alloc_dtype = np.complex128
    
    return alloc_dtype

# -----------------------------------------------------------------------------------------------
#! JITTED FUNCTIONS
# -----------------------------------------------------------------------------------------------

@numba.njit(nogil=True)
def _build_sparse_same_sector_compact_jit(
                representative_list : np.ndarray,   # int64[n_repr]         - representative state values
                normalization       : np.ndarray,   # float64[n_repr]       - normalization per repr
                repr_map            : np.ndarray,   # uint32[nh_full]       - state -> repr index
                phase_idx           : np.ndarray,   # uint8[nh_full]        - state -> phase table index
                phase_table         : np.ndarray,   # complex128[n_phases]  - distinct phases
                operator_func       : Callable,
                ns                  : int,          # Add ns argument
                rows                : np.ndarray,   # int64[max_nnz]        - preallocated row indices
                cols                : np.ndarray,   # int64[max_nnz]        - preallocated column indices
                data                : np.ndarray,   # complex128[max_nnz]   - preallocated data array
                data_idx            : int           # current data index
            ):
    """
    JIT-compiled sparse matrix builder using compact O(1) symmetry lookups.
    
    This is the preferred path when CompactSymmetryData is available.
    All lookups are O(1) array accesses - no binary search needed.
    
    Memory efficient: uses uint32 + uint8 per state (~5 bytes vs ~24+ bytes).
    """
    
    nh          = len(representative_list)
    invalid_idx = _INVALID_REPR_IDX_NB
    
    for k in range(nh):
        state               = representative_list[k]
        norm_k              = normalization[k]
        new_states, values  = operator_func(state)
        for i in range(len(new_states)):
            new_state = new_states[i]
            value     = values[i]
            
            if np.abs(value) < 1e-14:
                continue

            idx = repr_map[new_state]
            if idx == invalid_idx:
                continue
            
            pidx        = phase_idx[new_state]
            phase       = phase_table[pidx]
            norm_idx    = normalization[idx]
            sym_factor  = np.conj(phase) * norm_idx / norm_k

            matrix_elem = value * sym_factor
            if np.abs(matrix_elem) < 1e-14:
                continue

            rows[data_idx]       = idx
            cols[data_idx]       = k
            data[data_idx]       = matrix_elem
            data_idx            += 1
    
    return data_idx

@numba.njit(nogil=True)
def _build_dense_same_sector_compact_jit(
                representative_list : np.ndarray,
                normalization       : np.ndarray,
                repr_map            : np.ndarray,
                phase_idx           : np.ndarray,
                phase_table         : np.ndarray,
                operator_func       : Callable,
                matrix              : np.ndarray
            ):
    """
    JIT-compiled dense matrix builder using compact O(1) symmetry lookups.
    """
    nh          = len(representative_list)
    invalid_idx = _INVALID_REPR_IDX_NB
    
    for k in range(nh):
        state               = representative_list[k]
        norm_k              = normalization[k]
        new_states, values  = operator_func(state)

        for i in range(len(new_states)):
            new_state = new_states[i]
            value     = values[i]
            
            if np.abs(value) < 1e-14:
                continue

            idx = repr_map[new_state]
            if idx == invalid_idx:
                continue
            
            pidx        = phase_idx[new_state]
            phase       = phase_table[pidx]
            norm_idx    = normalization[idx]
            sym_factor  = np.conj(phase) * norm_idx / norm_k

            matrix[idx, k] += value * sym_factor

############################################################################################
#! SAME SECTOR MATRIX BUILDING
############################################################################################

def _build_same_sector(hilbert_space    : HilbertSpace,
                    operator_func       : Callable,
                    sparse              : bool,
                    max_local_changes   : int,
                    dtype               : np.dtype,
                    check_sym           : bool
                ) -> Union[sp.csr_matrix, np.ndarray]:
    """Build matrix for operator within same sector."""
    
    # Determine appropriate dtype
    nh                      = hilbert_space.dim
    ns                      = hilbert_space.ns
    has_sym                 = hilbert_space.has_sym
    alloc_dtype             = _determine_matrix_dtype(dtype, hilbert_space, operator_func, ns)
    
    if not sparse:
        # Dense matrix
        matrix              = np.zeros((nh, nh), dtype=alloc_dtype)
        if not has_sym:
            return _build_no_hilbert(operator_func, nh, ns, sparse=False, max_local_changes=max_local_changes, dtype=alloc_dtype)
        else:
            compact_data    = hilbert_space.compact_symmetry_data
            if compact_data is not None:
                # TODO: Implement dense projected builder if needed
                if check_sym:
                    # Fallback to sparse for projected build for now, or implement dense projected
                    raise NotImplementedError("Dense matrix building with check_sym=True not yet implemented.")
                
                _build_dense_same_sector_compact_jit(
                    compact_data.representative_list,
                    compact_data.normalization,
                    compact_data.repr_map,
                    compact_data.phase_idx,
                    compact_data.phase_table,
                    operator_func, matrix   # matrix passed by reference, we fill it in place
                )
            else:
                raise ValueError("Compact symmetry data missing for symmetric Hilbert space in dense build.")
        return matrix
    
    # Sparse matrix
    # If using projected builder, we might have more elements per row than max_local_changes
    # because of group sum. Scale estimate.
    sym_factor = 1
    if check_sym and hilbert_space._sym_container and hilbert_space._sym_container.compiled_group:
        sym_factor = hilbert_space._sym_container.compiled_group.n_group
        
    max_nnz                 = nh * max_local_changes * (ns) * sym_factor    # estimate
    rows                    = np.zeros(max_nnz, dtype=np.int64)     # preallocate
    cols                    = np.zeros(max_nnz, dtype=np.int64)     # preallocate
    data                    = np.zeros(max_nnz, dtype=alloc_dtype)

    if not has_sym:
        return _build_no_hilbert(operator_func, nh, ns, sparse, max_local_changes, alloc_dtype)
    else:
        # Use compact O(1) lookup
        compact_data        = hilbert_space.compact_symmetry_data
        
        if compact_data is not None:
            if check_sym:
                
                try:
                    from QES.Algebra.Symmetries.jit.matrix_builder_jit import _build_sparse_projected_jit, _build_dense_projected_jit
                except ImportError:
                    raise ImportError("Projected matrix builder requires QES.Algebra.Symmetries.jit.matrix_builder_jit module.")
                
                # Use projected builder
                sym_c = hilbert_space._sym_container
                cg = sym_c.compiled_group
                tb = sym_c.tables
                
                if sparse:
                    data_idx = _build_sparse_projected_jit(
                        rows, cols, data, 0,
                        operator_func,
                        compact_data.representative_list,
                        compact_data.normalization,
                        compact_data.repr_map,
                        compact_data.normalization,
                        compact_data.representative_list,
                        np.int64(ns),
                        cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, cg.chi,
                        tb.trans_perm, tb.trans_cross_mask, tb.refl_perm, tb.inv_perm, tb.parity_axis, tb.boundary_phase
                    )
                else:
                    _build_dense_projected_jit(
                        matrix,
                        operator_func,
                        compact_data.representative_list,
                        compact_data.normalization,
                        compact_data.repr_map,
                        compact_data.normalization,
                        compact_data.representative_list,
                        np.int64(ns),
                        cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, cg.chi,
                        tb.trans_perm, tb.trans_cross_mask, tb.refl_perm, tb.inv_perm, tb.parity_axis, tb.boundary_phase
                    )
                    return matrix # Dense case returns here
            else:
                if not sparse:
                    _build_dense_same_sector_compact_jit(
                        compact_data.representative_list,
                        compact_data.normalization,
                        compact_data.repr_map,
                        compact_data.phase_idx,
                        compact_data.phase_table,
                        operator_func, matrix
                    )
                    return matrix

                # Fast path: O(1) lookups using compact structure (~5 bytes/state)
                data_idx = _build_sparse_same_sector_compact_jit(
                    compact_data.representative_list,
                    compact_data.normalization,
                    compact_data.repr_map,
                    compact_data.phase_idx,
                    compact_data.phase_table,
                    operator_func, ns, rows, cols, data, 0 # initial data_idx
                )
        else:
             raise ValueError("Compact symmetry data missing for symmetric Hilbert space in sparse build.")

    return sp.csr_matrix((data[:data_idx], (rows[:data_idx], cols[:data_idx])),shape=(nh, nh), dtype=alloc_dtype)

def _build_sector_change(hilbert_in         : object,
                        hilbert_out         : object,
                        operator_func       : Callable,
                        sparse              : bool,
                        max_local_changes   : int,
                        dtype               : Union[np.dtype, str],
                        check_sym           : bool
                    ) -> Union[sp.csr_matrix, np.ndarray]:
    """Build rectangular matrix for sector-changing operator."""
    nh_in                       = hilbert_in.dim
    nh_out                      = hilbert_out.dim
    ns                          = hilbert_in.ns
    
    representative_list_in      = hilbert_in.representative_list if hasattr(hilbert_in, 'representative_list') else np.arange(nh_in, dtype=np.int64)
    representative_list_out     = hilbert_out.representative_list if hasattr(hilbert_out, 'representative_list') else np.arange(nh_out, dtype=np.int64)
    
    norm_in                     = hilbert_in.normalization if hasattr(hilbert_in, 'normalization') else None
    norm_out                    = hilbert_out.normalization if hasattr(hilbert_out, 'normalization') else None
    
    # Determine appropriate dtype
    alloc_dtype                 = _determine_matrix_dtype(dtype, [hilbert_in, hilbert_out], operator_func, ns)
    
    # Check for compact symmetry data (preferred O(1) path)
    compact_data_out            = getattr(hilbert_out,  'compact_symmetry_data', None)
    compact_data_in             = getattr(hilbert_in,   'compact_symmetry_data', None) # Needed for projected build
    use_compact                 = compact_data_out is not None
    
    if not use_compact:
        raise ValueError("Compact symmetry data missing for output Hilbert space in sector change build.")
    
    if not sparse:
        # Dense matrix
        matrix                  = np.zeros((nh_out, nh_in), dtype=alloc_dtype)
        
        if check_sym and compact_data_in and compact_data_out:
            try:
                from QES.Algebra.Symmetries.jit.matrix_builder_jit import _build_dense_projected_jit
            except ImportError:
                raise ImportError("Projected matrix builder requires QES.Algebra.Symmetries.jit.matrix_builder_jit module.")
            
            sym_c_in    = hilbert_in._sym_container
            cg_in       = sym_c_in.compiled_group
            cg_out      = hilbert_out._sym_container.compiled_group
            tb          = sym_c_in.tables
            
            _build_dense_projected_jit(
                matrix,
                operator_func,
                compact_data_in.representative_list,
                compact_data_in.normalization,
                compact_data_out.repr_map,
                compact_data_out.normalization,
                compact_data_out.representative_list,
                np.int64(ns),
                cg_in.n_group, cg_in.n_ops, cg_in.op_code, cg_in.arg0, cg_in.arg1,
                cg_in.chi, cg_out.chi,
                tb.trans_perm, tb.trans_cross_mask, tb.refl_perm, tb.inv_perm, tb.parity_axis, tb.boundary_phase
            )
            return matrix

        for idx_col in range(nh_in):
            state_col           = int(hilbert_in[idx_col])
            norm_col            = norm_in[idx_col] if norm_in is not None else 1.0
            new_states, values  = operator_func(state_col)

            for new_state, value in zip(new_states, values):
                if abs(value) < 1e-14:
                    continue

                # O(1) compact lookup
                idx_row     = compact_data_out.repr_map[int(new_state)]
                if idx_row == 0xFFFFFFFF:  # _INVALID_REPR_IDX
                    continue
                pidx        = compact_data_out.phase_idx[int(new_state)]
                phase       = compact_data_out.phase_table[pidx]
                norm_rep    = compact_data_out.normalization[idx_row]
                sym_factor  = np.conj(phase) * norm_rep / norm_col

                matrix[idx_row, idx_col] += value * sym_factor
        return matrix
    
    # Sparse matrix
    sym_factor = 1
    if check_sym and hilbert_in._sym_container and hilbert_in._sym_container.compiled_group:
        sym_factor = hilbert_in._sym_container.compiled_group.n_group

    max_nnz         = nh_in * max_local_changes * ns * sym_factor
    rows            = np.zeros(max_nnz, dtype=np.int64)
    cols            = np.zeros(max_nnz, dtype=np.int64)
    data            = np.zeros(max_nnz, dtype=alloc_dtype)
    data_idx        = 0
    
    if check_sym and compact_data_in and compact_data_out:
        
        try:
            from QES.Algebra.Symmetries.jit.matrix_builder_jit import _build_sparse_projected_jit
        except ImportError:
            raise ImportError("Projected sparse matrix builder requires QES.Algebra.Symmetries.jit.matrix_builder_jit module.")
        
        # Projected build
        sym_c_in    = hilbert_in._sym_container
        cg_in       = sym_c_in.compiled_group
        cg_out      = hilbert_out._sym_container.compiled_group
        tb          = sym_c_in.tables
        data_idx    = _build_sparse_projected_jit(
            rows, cols, data, 0,
            operator_func,
            compact_data_in.representative_list,
            compact_data_in.normalization,
            compact_data_out.repr_map,
            compact_data_out.normalization,
            compact_data_out.representative_list,
            np.int64(ns),
            cg_in.n_group, cg_in.n_ops, cg_in.op_code, cg_in.arg0, cg_in.arg1,
            cg_in.chi, cg_out.chi, # chi_in, chi_out
            tb.trans_perm, tb.trans_cross_mask, tb.refl_perm, tb.inv_perm, tb.parity_axis, tb.boundary_phase
        )

    else:
        # Standard fast path
        # Iterate over input basis states
        for idx_col in range(nh_in):
            state_col           = int(hilbert_in[idx_col])
            norm_col            = norm_in[idx_col] if norm_in is not None else 1.0
            new_states, values  = operator_func(state_col)

            for new_state, value in zip(new_states, values):
                if abs(value) < 1e-14:
                    continue

                # O(1) compact lookup
                idx_row     = compact_data_out.repr_map[int(new_state)]
                if idx_row == 0xFFFFFFFF:  # _INVALID_REPR_IDX
                    continue
                pidx        = compact_data_out.phase_idx[int(new_state)]
                phase       = compact_data_out.phase_table[pidx]
                norm_rep    = compact_data_out.normalization[idx_row]
                sym_factor  = np.conj(phase) * norm_rep / norm_col

                matrix_elem = value * sym_factor
                if abs(matrix_elem) > 1e-14:
                    rows[data_idx]       = int(idx_row)
                    cols[data_idx]       = int(idx_col)
                    data[data_idx]       = matrix_elem
                    data_idx            += 1

    return sp.csr_matrix((data[:data_idx], (rows[:data_idx], cols[:data_idx])), shape=(nh_out, nh_in), dtype=alloc_dtype)

############################################################################################
#! WITHOUT HILBERT SPACE SUPPORT
############################################################################################

@numba.njit(nogil=True) # nogil for thread safety -> allows multi-threading if needed, stands for "no global interpreter lock"
def _fill_dense_kernel(matrix, operator_func, nh):
    """
    Fills a dense matrix entirely within Numba.
    """
    
    for row in range(nh):
        # Call the operator directly inside the loop
        out_states, out_vals    = operator_func(row)
        n_items                 = len(out_states)
        for k in range(n_items):
            col                 = out_states[k]
            val                 = out_vals[k]
            
            # Bounds check and threshold check
            if (0 <= col < nh) and (np.abs(val) >= 1e-14):
                matrix[row, col] += val

@numba.njit(nogil=True)
def _fill_sparse_kernel(
    start_row,                  # Where to start/resume
    nh,                         # Total rows
    rows, cols, vals,           # The buffers
    current_ptr,                # Where to write in buffer
    max_len,                    # Buffer limit
    operator_func               # The JIT-compiled operator
):
    """
    Iterates rows and fills buffers. 
    Returns: (next_row_to_process, new_buffer_pointer, is_finished)
    """
    ptr = current_ptr
    
    for row in range(start_row, nh):
        out_states, out_vals    = operator_func(row)
        n_items                 = len(out_states)
        
        # Check if we have space for this entire batch of connections
        # (Conservative check: if this row fits, we process it. If not, we pause).
        if ptr + n_items >= max_len:
            return row, ptr, False # Signal: Paused, buffer full
            
        for k in range(n_items):
            col = out_states[k]
            val = out_vals[k]
            
            if (0 <= col < nh) and (np.abs(val) >= 1e-14):
                rows[ptr] = row
                cols[ptr] = col
                vals[ptr] = val
                ptr += 1
                
    return nh, ptr, True # Signal: Finished all rows

def _build_no_hilbert(operator_func: Callable, nh: int, ns: int, sparse: bool, max_local_changes: int, dtype: np.dtype):
    ''' 
    This function builds the operator matrix without Hilbert space support.
    It supports both sparse and dense formats.
    
    Compilation takes 0.3s - 0.5s depending on operator complexity     
    '''
    
    # Determine precision
    alloc_dtype     = dtype

    # Dense path
    if not sparse:        
        matrix = np.zeros((nh, nh), dtype=alloc_dtype)
        _fill_dense_kernel(matrix, operator_func, nh)   # Note: operator_func MUST be jitted for this to work efficiently
        return matrix
        
    # Initial Allocation
    # Estimate size: nh * max_changes * safety_factor
    # We use a chunk-based approach to be memory safe.
    estimated_nnz   = int(nh * max(2, max_local_changes) * 1.2)     # Initial estimate
    buffer_size     = max(1024, estimated_nnz)                      # Minimum size to prevent excessive resizing on tiny systems
    
    rows            = np.empty(buffer_size, dtype=np.int64)
    cols            = np.empty(buffer_size, dtype=np.int64)
    vals            = np.empty(buffer_size, dtype=alloc_dtype)
    
    curr_row        = 0
    curr_ptr        = 0
    finished        = False
    
    # The "Pump" Loop
    while not finished:
        # Call the kernel. It runs until it finishes OR runs out of space.
        next_row, next_ptr, finished = _fill_sparse_kernel(curr_row, nh, rows, cols, vals, curr_ptr, buffer_size, operator_func)
        
        curr_row    = next_row
        curr_ptr    = next_ptr
        
        if not finished:
            # Strategy: Double the buffer size (Exponential growth)
            # This amortizes the cost of copying to O(1) per element
            new_size = buffer_size * 2
            
            # Resize checks
            try:
                rows.resize(new_size, refcheck=False)
                cols.resize(new_size, refcheck=False)
                vals.resize(new_size, refcheck=False)
            except MemoryError:
                raise MemoryError(f"Failed to allocate sparse buffer of size {new_size}")
                
            buffer_size = new_size
            
    # Shrink to fit exact number of elements
    return sp.csr_matrix((vals[:curr_ptr], (rows[:curr_ptr], cols[:curr_ptr])), shape=(nh, nh))

# ------------------------------------------------------------------------------------------
#! GENERAL MATRIX BUILDER INTERFACE
# ------------------------------------------------------------------------------------------

def build_operator_matrix(
        operator_func           : Callable,
        *,
        hilbert_space           : Optional[HilbertSpace]    = None,
        hilbert_space_out       : Optional[HilbertSpace]    = None,
        sparse                  : bool                      = True,
        max_local_changes       : int                       = 2,
        dtype                   : np.dtype                  = np.float64,
        ns                      : Optional[int]             = None,
        nh                      : Optional[int]             = None,
        check_sym               : bool                      = False
    ) -> Union[sp.csr_matrix, np.ndarray]:
    """
    Build operator matrix with support:
    - for sector-changing operators.
    - same sector operators.
    - sparse and dense formats.
    - binary search for representative lookup.
    
    This version supports:
    - Same sector: 
        - Square matrices (nh x nh)
    - Different sectors: 
        - Rectangular matrices (nh_out x nh_in)
    - Binary search for O(log Nh) lookup (requires sorted representative_list - it is sorted by construction in HilbertSpace)
        - Efficient representative lookup
    - Memory-efficient construction
    
    Parameters
    ----------
        operator_func: 
            (generally) Numba function (state) -> (new_states, values)
        hilbert_space : Optional[HilbertSpace]
            Input Hilbert space (column space). If None, nh and ns must be provided for no-symmetry case.
        hilbert_space_out: 
            Output Hilbert space (row space), None for same sector
        sparse: 
            Use sparse format
        max_local_changes: 
            Estimated non-zeros per row
        dtype: 
            Matrix data type
        ns: 
            Number of sites (for operator_func). If hilbert_space is provided, defaults to hilbert_space.ns
        nh: 
            Hilbert space dimension. If hilbert_space is provided, defaults to hilbert_space.dim
        check_sym:
            If True, use strict projector logic for symmetric matrix building.
            This expands states to the full symmetry orbit, applies operator, and projects back.
            Necessary if operator does not strictly commute with symmetries (e.g. flux insertion)
            or if using sophisticated symmetry breaking terms.

    Returns:
    ---------
    
        Operator matrix (sparse CSR or dense array)
        
    Example:
    --------
    
    >>> # Same sector (square matrix)
    >>> H = build_operator_matrix(hilbert, hamiltonian_op)
    >>> # Sector-changing (rectangular matrix)
    >>> # e.g., creation operator changing particle number
    >>> c_dag = build_operator_matrix(
    >>> hilbert_n, c_dag_op, hilbert_space_out=hilbert_n_plus_1
    >>> )
    >>> # Without Hilbert space (no symmetries)
    >>> H_simple = build_operator_matrix(operator_func, nh=dim, ns=num_sites)
    """
    
    if hilbert_space is None or hilbert_space.nh == hilbert_space.nhfull:
        
        # This is invalid if we are in the 'full space' or 'no space' case.
        if hilbert_space_out is not None:
            raise ValueError("hilbert_space_out not supported when operating in the full Hilbert space (i.e., when hilbert_space is None or nh == nhfull).")

        # If hilbert_space is None, we must ensure nh and ns are explicitly provided.
        if hilbert_space is None:
            if nh is None or ns is None:
                raise ValueError("If hilbert_space is None, nh (Hilbert space size) and ns (number of spins) must be provided.")
        
        # If we reach here, we are in the full space case (nh == nhfull or hilbert_space is None) 
        # and we have all necessary parameters (nh, ns).
        return _build_no_hilbert(operator_func, nh or (hilbert_space.dim if hilbert_space is not None else None), ns or (hilbert_space.ns if hilbert_space is not None else None), sparse, max_local_changes, dtype)
    
    # Use provided or default values
    nh = nh or hilbert_space.dim
    ns = ns or hilbert_space.ns
    
    if hilbert_space_out is None:
        return _build_same_sector(hilbert_space, operator_func, sparse, max_local_changes, dtype, check_sym)
    else:
        return _build_sector_change(hilbert_space, hilbert_space_out, operator_func, sparse, max_local_changes, dtype, check_sym)

# ------------------------------------------------------------------------------------------
#! Matrix-Free Operator Application (optimized!)
# ------------------------------------------------------------------------------------------

def ensure_thread_buffer(nh: int, chunk_size: int, dtype=np.complex128, thread_buffers: np.ndarray | None = None):
    """
    Allocate or validate a buffer of shape:
        (n_threads, nh, chunk_size)
        
    for thread-safe operator application.
    
    This buffer is used in multi-threaded operator applications to avoid write conflicts.
    """
    n_threads = numba.get_num_threads()

    if (thread_buffers is None or thread_buffers.ndim != 3 or thread_buffers.shape[0] != n_threads or thread_buffers.shape[1] != nh or thread_buffers.shape[2] != chunk_size):
        return np.zeros((n_threads, nh, chunk_size), dtype=dtype)
    return thread_buffers

# a) No symmetry support

@numba.njit(boundscheck=False)
def _apply_op_batch_seq_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        op_func             : Callable,
        args                : Tuple,
    ) -> None:
    ''' Sequential batch operator application. No symmetry support. '''
    nh, n_batch = vecs_in.shape
    for k in range(nh):
        state               = np.int64(k)
        new_states, values  = op_func(state, *args)
        n_new               = len(new_states)
        for i in range(n_new):
            target_idx      = new_states[i]
            if not (0 <= target_idx < nh):
                continue
            val             = values[i]
            # if abs(val) < 1e-15: continue
            for b in range(n_batch):
                vecs_out[target_idx, b] += val * vecs_in[k, b]

@numba.njit(parallel=True, nogil=True)
def _apply_op_batch_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        op_func             : Callable,
        args                : Tuple,
        thread_buffers      : np.ndarray,
        # basis -> leave basis for other kernels
        chunk_size          : int = 6,
    ) -> None:
    '''
    Jitted batch operator application with optional basis support.
    This function works for no-symmetry Hilbert spaces.
    
    Notes
    ----------
    - We process the input vectors in chunks to manage memory usage. The chunk size can be adjusted.
    - Each thread has its own buffer to avoid write conflicts.
    - The operator function `op_func` should be jitted and have the signature:
        (state: int, *args) -> (new_states: np.ndarray, values: np.ndarray)
    - The function writes results directly into `vecs_out` -> no return value.
    
    Parameters
    ----------
    vecs_in : np.ndarray
        Input vectors (shape: nh x n_batch)
    vecs_out : np.ndarray
        Output vectors (shape: nh x n_batch)
    op_func : Callable
        Jitted operator function: (state, *args) -> (new_states, values)
    args : Tuple
        Additional arguments for op_func
    thread_buffers : np.ndarray
        Buffers for thread-safe writes. It already has shape (n_threads, nh, chunk_size). Need to be preallocated.
    chunk_size : int
        Size of chunks for batch processing
    
    Returns
    -------
    None (results are written to vecs_out)
    
    Example
    -------
    >>> _apply_op_batch_jit(vecs_in, vecs_out, op_func, args, thread_buffers, chunk_size=4)
    '''
    
    nh, n_batch = vecs_in.shape
    
    # Fast path for small Hilbert spaces to avoid parallel overhead
    if nh <= _NUMBA_OVERHEAD_SIZE:
        for k in range(nh):
            state               = np.int64(k)
            new_states, values  = op_func(state, *args)
            for i in range(len(new_states)):
                target_idx      = new_states[i]
                val             = values[i]
                # if abs(val) < 1e-15: continue
                if 0 <= target_idx < nh:
                    for b in range(n_batch):
                        vecs_out[target_idx, b] += val * vecs_in[k, b]
        return 

    n_threads           = thread_buffers.shape[0]
    
    # We process the batch in small chunks to prevent VRAM/RAM explosion.
    # Chunk size of 1 is safest for memory. 
    # If one has massive RAM, you can increase this to 2 or 4.
    chunk_size          = min(chunk_size, n_batch)
    bufs                = thread_buffers

    # Loop over the batch in chunks
    for b_start in range(0, n_batch, chunk_size):
        b_end           = min(b_start + chunk_size, n_batch)
        actual_width    = b_end - b_start
        
        # Reset buffers for this chunk (crucial!)
        bufs[:, :, :actual_width].fill(0.0)

        # Parallel Loop over Hilbert Space
        for k in numba.prange(nh):
            tid         = numba.get_thread_id()
            state       = np.int64(k)
        
            # This returns new states and values (independent of batch index)
            new_states, values          = op_func(state, *args)
            
            # [Map & Write], assume it returns many states
            for i in range(len(new_states)):
                new_state               = new_states[i]
                val                     = values[i]

                if abs(val) < 1e-15:    continue
                
                target_idx              = -1
                target_idx              = new_state

                if 0 <= target_idx < nh:
                    # Apply to the current batch slice
                    # thread_buffers[tid, row, batch_col]
                    # vecs_in[k, global_batch_col]
                    
                    for b_local in range(actual_width):
                        bufs[tid, target_idx, b_local] += val * vecs_in[k, b_start + b_local]
        
        # [Reduction Step]
        # Sum thread buffers into the final output for this chunk
        for t in range(n_threads):
            for b_local in range(actual_width):
                # We can vectorize the sum over NH if memory bandwidth allows, 
                # but looping column-wise is cache-friendly for the reduction 
                # if vecs_out is column-major (F-contiguous) or row-major (C-contiguous).
                # Assuming standard C-contiguous (row-major), strict loops are fine.
                vecs_out[:, b_start + b_local] += bufs[t, :, b_local]

# b) No symmetry with Fourier phases

@numba.njit
def _apply_fourier_batch_seq_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        phases              : np.ndarray,
        op_func             : Callable,
    ) -> None:
    ''' Sequential Fourier batch operator application. No symmetry support. '''
    nh, n_batch = vecs_in.shape
    n_sites     = len(phases)
    for k in range(nh):
        state = k
        for site_idx in range(n_sites):
            c_site              = phases[site_idx]
            new_states, values  = op_func(state, site_idx)
            n_new               = len(new_states)
            for j in range(n_new):
                target_idx      = new_states[j]
                if not (0 <= target_idx < nh): continue
                val             = values[j]
                factor          = val * c_site
                # if abs(factor) < 1e-15: continue
                for b in range(n_batch):
                    vecs_out[target_idx, b] += factor * vecs_in[k, b]

# @numba.njit(parallel=False, fastmath=True)
@numba.njit(parallel=True)
def _apply_fourier_batch_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        phases              : np.ndarray,
        op_func             : Callable,
        thread_buffers      : np.ndarray,
        chunk_size          : int = 4
    ) -> None:
    r'''
    Jitted batch operator application with Fourier phases.
    This function works for no-symmetry Hilbert spaces with Fourier phase factors.
    
    Notes
    ----------
    - We process the input vectors in chunks to manage memory usage. The chunk size can be adjusted.
    - Each thread has its own buffer to avoid write conflicts.
    - The operator function `op_func` should be jitted and have the signature:
        (state: int, site_idx: int) -> (new_states: np.ndarray, values: np.ndarray)
    - The function writes results directly into `vecs_out` -> no return value.
    
    Parameters
    ----------
    vecs_in : np.ndarray
        Input vectors (shape: nh x n_batch)
    vecs_out : np.ndarray
        Output vectors (shape: nh x n_batch)
    phases : np.ndarray
        Fourier phase factors for each site (shape: n_sites)
    op_func : Callable
        Jitted operator function: (state, site_idx) -> (new_states, values)
    thread_buffers : np.ndarray
        Buffers for thread-safe writes. It already has shape (n_threads, nh, chunk_size). Need to be preallocated.
    chunk_size : int
        Size of chunks for batch processing
    Returns
    -------
    None (results are written to vecs_out)
    '''
    
    if vecs_in.shape[0] <= _NUMBA_OVERHEAD_SIZE:
        nh, n_batch     = vecs_in.shape
        n_sites         = len(phases)
        for k in range(nh):
            state       = k
            for site_idx in range(n_sites):
                c_site              = phases[site_idx]
                new_states, values  = op_func(state, site_idx)
                for j in range(len(new_states)):
                    new_state       = new_states[j]
                    val             = values[j]
                    factor          = val * c_site
                    # if abs(factor) < 1e-15: continue
                    target_idx      = new_state
                    if 0 <= target_idx < nh:
                        for b in range(n_batch):
                            vecs_out[target_idx, b] += factor * vecs_in[k, b]
        return
    
    nh, n_batch     = vecs_in.shape
    n_sites         = len(phases)
    n_threads       = thread_buffers.shape[0]

    # Buffer Management (Same logic as matvec)
    bufs            = thread_buffers 
        
    # Main Chunked Loop
    for b_start in range(0, n_batch, chunk_size):
        b_end           = min(b_start + chunk_size, n_batch)
        actual_width    = b_end - b_start
        
        # Reset active buffer area
        bufs[:, :, :actual_width].fill(0.0)

        # PARALLEL LOOP over Hilbert Space
        for k in numba.prange(nh):
            # State Decoding
            tid     = numba.get_thread_id()
            state   = k
            
            # Site Loop (Summation for Fourier Transform)
            for site_idx in range(n_sites):
                c_site              = phases[site_idx]
                new_states, values  = op_func(state, site_idx)
                
                for j in range(len(new_states)):
                    new_state       = new_states[j]
                    val             = values[j]
                    
                    # Compute Factor: Operator Value * Fourier Phase
                    factor          = val * c_site 
                    # if abs(factor) < 1e-15: continue

                    # Find Target Index
                    target_idx      = new_state

                    # Safe Write to Thread-Local Buffer
                    if 0 <= target_idx < nh:
                        for b in range(actual_width):
                            bufs[tid, target_idx, b] += factor * vecs_in[k, b_start + b]

        # Reduction Step
        # Sum all thread buffers into the final output for this chunk
        for t in range(n_threads):
            for b in range(actual_width):
                vecs_out[:, b_start + b] += bufs[t, :, b]

# c) Symmetry support with CompactSymmetryData - fast path when matvec preserves sectors

@numba.njit
def _apply_op_batch_compact_seq_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        op_func             : Callable,
        args                : Tuple,
        basis_args          : Tuple,        # (rep_list, norm, repr_map, p_idx, p_table)
    ) -> None:
    ''' Sequential batch operator application using CompactSymmetryData for O(1) lookups. '''
    nh, n_batch = vecs_in.shape
    representative_list, normalization, repr_map, phase_idx, phase_table = basis_args
    for k in range(nh):
        state               = representative_list[k]
        norm_k              = normalization[k]
        inv_norm_k          = 1.0 / norm_k if norm_k > 0.0 else 0.0
        new_states, values  = op_func(state, *args)
        n_new               = len(new_states)
        for i in range(n_new):
            new_state   = new_states[i]
            idx         = repr_map[new_state]
            if idx == _INVALID_REPR_IDX_NB: 
                continue
            val         = values[i]
            if abs(val) < 1e-15:
                continue
            pidx        = phase_idx[new_state]
            phase       = phase_table[pidx]
            norm_new    = normalization[idx]
            factor      = val * np.conj(phase) * norm_new * inv_norm_k
            for b in range(n_batch):
                vecs_out[idx, b] += factor * vecs_in[k, b]

@numba.njit(parallel=True)
def _apply_op_batch_compact_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        op_func             : Callable,
        args                : Tuple,
        basis_args          : Tuple,        # (rep_list, norm, repr_map, p_idx, p_table)
        thread_buffers      : np.ndarray,
        chunk_size          : int = 6,
    ) -> None:
    '''
    Jitted batch operator application using CompactSymmetryData for O(1) lookups.
    '''
    
    nh, n_batch = vecs_in.shape
    representative_list, normalization, repr_map, phase_idx, phase_table = basis_args
    
    # Fast path for small Hilbert spaces to avoid parallel overhead
    if nh <= _NUMBA_OVERHEAD_SIZE:
        for k in range(nh):
            state               = representative_list[k]
            norm_k              = normalization[k]
            new_states, values  = op_func(state, *args)
            for i in range(len(new_states)):
                new_state   = new_states[i]
                val         = values[i]
                if abs(val) < 1e-15:    
                    continue
                idx         = repr_map[new_state]
                if idx == _INVALID_REPR_IDX_NB: 
                    continue
                pidx        = phase_idx[new_state]
                phase       = phase_table[pidx]
                norm_new    = normalization[idx]
                sym_factor  = np.conj(phase) * (norm_new / norm_k)
                factor      = val * sym_factor
                for b in range(n_batch):
                    vecs_out[idx, b] += factor * vecs_in[k, b]
        return

    n_threads           = thread_buffers.shape[0]
    chunk_size          = min(chunk_size, n_batch)
    bufs                = thread_buffers

    # Loop over the batch in chunks
    for b_start in range(0, n_batch, chunk_size):
        b_end           = min(b_start + chunk_size, n_batch)
        actual_width    = b_end - b_start
        
        # Reset buffers
        bufs[:n_threads, :, :actual_width].fill(0.0)

        # Parallel Loop
        for k in numba.prange(nh):
            tid                 = numba.get_thread_id()
            state               = representative_list[k]    # Get representative state
            norm_k              = normalization[k]          # Normalization factor for this state
            
            new_states, values  = op_func(state, *args)     # Apply operator -> new states and values
            
            for i in range(len(new_states)):
                new_state   = new_states[i]
                val         = values[i]

                if abs(val) < 1e-15: continue
                
                # Compact O(1) lookup
                idx         = repr_map[new_state]           # Get index in reduced basis -> maps to index of representative_list
                if idx == _INVALID_REPR_IDX_NB:
                    continue
                
                pidx        = phase_idx[new_state]
                phase       = phase_table[pidx]
                norm_new    = normalization[idx]
                
                sym_factor  = np.conj(phase) * (norm_new / norm_k)
                
                # Vectorized write to thread buffer
                for b in range(actual_width):
                    bufs[tid, idx, b] += val * sym_factor * vecs_in[k, b_start + b]
        
        # Reduction
        for t in range(n_threads):
            for b in range(actual_width):
                vecs_out[:, b_start + b] += bufs[t, :, b]

# d) Symmetry support with Fourier phases and CompactSymmetryData - fast path when matvec preserves sectors

@numba.njit(fastmath=True)
def _apply_fourier_batch_compact_seq_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        phases              : np.ndarray,
        op_func             : Callable,
        basis_args          : Tuple,
    ) -> None:
    ''' Sequential Fourier batch Fourier transform using CompactSymmetryData for O(1) lookups. '''
    nh, n_batch         = vecs_in.shape
    representative_list, normalization, repr_map, phase_idx, phase_table = basis_args
    n_sites             = len(phases)
    for k in range(nh):
        state               = representative_list[k]
        norm_k              = normalization[k]
        inv_norm_k          = 1.0 / norm_k if norm_k > 0.0 else 0.0
        for site_idx in range(n_sites):
            c_site              = phases[site_idx]
            new_states, values  = op_func(state, site_idx)
            n_new               = len(new_states)
            for j in range(n_new):
                new_state       = new_states[j]
                idx             = repr_map[new_state]
                if idx == _INVALID_REPR_IDX_NB:
                    continue
                val             = values[j]
                factor          = val * c_site 
                if abs(factor) < 1e-15:
                    continue
                pidx        = phase_idx[new_state]
                phase       = phase_table[pidx]
                norm_new    = normalization[idx]
                total_factor= factor * np.conj(phase) * norm_new * inv_norm_k
                for b in range(n_batch):
                    vecs_out[idx, b] += total_factor * vecs_in[k, b]

@numba.njit(parallel=True)
def _apply_fourier_batch_compact_jit(
        vecs_in             : np.ndarray,
        vecs_out            : np.ndarray,
        phases              : np.ndarray,
        op_func             : Callable,
        basis_args          : Tuple,
        thread_buffers      : np.ndarray,
        chunk_size          : int = 6
    ) -> None:
    '''
    Jitted batch Fourier transform using CompactSymmetryData for O(1) lookups.
    '''
    
    nh, n_batch         = vecs_in.shape
    representative_list, normalization, repr_map, phase_idx, phase_table = basis_args

    if nh <= _NUMBA_OVERHEAD_SIZE:
        for k in range(nh):
            state               = representative_list[k]
            norm_k              = normalization[k]
            n_sites             = len(phases)
            for site_idx in range(n_sites):
                c_site              = phases[site_idx]
                new_states, values  = op_func(state, site_idx)
                for j in range(len(new_states)):
                    new_state       = new_states[j]
                    val             = values[j]
                    factor          = val * c_site 
                    if abs(factor) < 1e-15: continue
                    idx             = repr_map[new_state]
                    if idx == _INVALID_REPR_IDX_NB:
                        continue
                    pidx        = phase_idx[new_state]
                    phase       = phase_table[pidx]
                    norm_new    = normalization[idx]
                    sym_factor  = np.conj(phase) * (norm_new / norm_k)
                    total_factor= factor * sym_factor
                    for b in range(n_batch):
                        vecs_out[idx, b] += total_factor * vecs_in[k, b]
        return
    
    n_sites         = len(phases)
    n_threads       = thread_buffers.shape[0]
    chunk_size      = min(chunk_size, n_batch)
    bufs            = thread_buffers
    
    for b_start in range(0, n_batch, chunk_size):
        b_end           = min(b_start + chunk_size, n_batch)
        actual_width    = b_end - b_start
        
        bufs[:n_threads, :, :actual_width].fill(0.0)

        for k in numba.prange(nh):
            tid         = numba.get_thread_id()
            
            state       = representative_list[k]
            norm_k      = normalization[k]

            for site_idx in range(n_sites):
                c_site              = phases[site_idx]
                new_states, values  = op_func(state, site_idx)
                
                for j in range(len(new_states)):
                    new_state       = new_states[j]
                    val             = values[j]
                    
                    factor          = val * c_site 
                    if abs(factor) < 1e-15: continue

                    # Compact O(1) lookup
                    idx = repr_map[new_state]
                    if idx == _INVALID_REPR_IDX_NB:
                        continue
                    
                    pidx        = phase_idx[new_state]
                    phase       = phase_table[pidx]
                    norm_new    = normalization[idx]
                    
                    sym_factor  = np.conj(phase) * (norm_new / norm_k)

                    # Vectorized write
                    for b in range(actual_width):
                        bufs[tid, idx, b] += (factor * sym_factor) * vecs_in[k, b_start + b]

        # Reduction
        for t in range(n_threads):
            for b in range(actual_width):
                vecs_out[:, b_start + b] += bufs[t, :, b]

# ------

def canonicalize_args(args):
    """
    Convert Python scalars to fixed-width numpy scalars so Numba doesn't
    re-specialize on int32 vs int64 vs Python int etc.
    """
    if args is None:
        return ()

    if isinstance(args, tuple):
        src = args
    else:
        src = (args,)

    out = []
    for a in src:
        # ints
        if isinstance(a, (int, np.integer)):
            out.append(np.int64(a))
            continue
        # floats
        if isinstance(a, (float, np.floating)):
            out.append(np.float64(a))
            continue
        # complex
        if isinstance(a, (complex, np.complexfloating)):
            out.append(np.complex128(a))
            continue
        # leave objects/arrays as-is (Numba will type them if supported)
        out.append(a)

    return tuple(out)

# -------------------------------------------------------------------------------
#! Symmetry rotation matrix construction
# -------------------------------------------------------------------------------

def get_symmetry_rotation_matrix(hilbert_space: HilbertSpace, dtype=np.complex128, *args, to_dense=False) -> Union[sp.csr_matrix, np.ndarray]:
    """
    Generate symmetry rotation matrix that expands symmetry-reduced basis to full Hilbert space.
    
    This creates a sparse matrix U such that:
        |full_state> = U |reduced_state>
    
    where |reduced_state> is in the symmetry-reduced basis and |full_state> is in
    the full Hilbert space. The matrix U has shape (nh_full x nh_reduced).
    
    For each representative state |r>, the symmetry operations generate an orbit
    of states {G|r>}. The rotation matrix properly normalizes these states:
    
        U_{i,k} = conj(phase) / (norm_k * sqrt(|symmetry_group|))
    
    where phase is the symmetry eigenvalue for the transformation G that takes
    representative k to state i.
    
    Parameters:
    -----------
    hilbert_space: HilbertSpace
        HilbertSpace with symmetries and representative_list
    dtype: np.dtype 
        Data type for matrix elements (typically complex128)
    ---
    to_dense: bool    
        If True, return a dense array instead of a sparse matrix

    Returns:
        Sparse CSR matrix U (nh_full x nh_reduced) or dense array if to_dense=True.
        
    Example:
        # Create Hilbert space with translation symmetry
        hs = HilbertSpace(
            lattice=lattice,
            sym_gen=[(SymmetryGenerators.Translation_x, 0)],
            gen_mapping=True  # Enable precomputed representative mappings for fast matrix building
        )
        
        # Get rotation matrix
        U = get_symmetry_rotation_matrix(hs)
        
        # Expand reduced state to full Hilbert space
        psi_reduced = np.array([...])  # State in reduced basis
        psi_full = U @ psi_reduced     # State in full basis
    """
    if not hasattr(hilbert_space, '_sym_group') or not hilbert_space._sym_group:
        raise ValueError("Hilbert space has no symmetries. Rotation matrix not applicable.")
    
    nh_reduced          = hilbert_space.dim
    representative_list = hilbert_space.representative_list
    normalization       = hilbert_space.normalization if hasattr(hilbert_space, 'normalization') else np.ones(nh_reduced, dtype=np.float64)
    sym_group           = hilbert_space.sym_group

    if not sym_group:
        raise ValueError("Symmetry group is empty.")
    
    sym_size            = len(sym_group)
    inv_sqrt_sym_size   = 1.0 / np.sqrt(sym_size)
    
    # Handle global symmetries
    fMap = None
    if hasattr(hilbert_space, 'get_full_map'):
        try:
            fMap = hilbert_space.get_full_map()
        except:
            fMap = None
    
    if fMap is not None and len(fMap) > 0:
        max_dim     = len(fMap)
        find_index  = lambda idx: _binary_search_representative_list(fMap, idx)
    else:
        max_dim     = hilbert_space.full
        find_index  = lambda idx: idx

    rows = []
    cols = []
    data = []
    
    if to_dense:
        matrix = np.zeros((max_dim, nh_reduced), dtype=dtype)
            
        for k in range(nh_reduced):
            rep_state   = int(representative_list[k])
            norm_k      = normalization[k]
            for G in sym_group:
                new_state, val  = G(rep_state)
                idx             = find_index(int(new_state))
                if idx < max_dim:
                    # matrix_elem = val * inv_sqrt_sym_size / norm_k
                    matrix_elem = np.conj(val) * inv_sqrt_sym_size / norm_k
                    if abs(matrix_elem) > 1e-14:
                        matrix[idx, k] = matrix_elem
        return matrix
    
    # ----
    #! Sparse matrix assembly
    # ----
    
    for k in range(nh_reduced):
        rep_state   = int(representative_list[k])
        norm_k      = normalization[k]
        for G in sym_group:
            new_state, val  = G(rep_state)
            idx             = find_index(int(new_state))
            if idx < max_dim:
                # matrix_elem = val * inv_sqrt_sym_size / norm_k
                matrix_elem = np.conj(val) * inv_sqrt_sym_size / norm_k
                if abs(matrix_elem) > 1e-14:
                    rows.append(idx)
                    cols.append(k)
                    data.append(matrix_elem)
    
    return sp.csr_matrix(
        (np.array(data, dtype=dtype), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(max_dim, nh_reduced),
        dtype=dtype
    )

# -------------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------------