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
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Callable, Optional, Union, Tuple, TYPE_CHECKING

try:
    import numba
    from numba.typed import List
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# ------------------------------------------------------------------------------------------

try:
    from QES.general_python.algebra.utils import DEFAULT_NP_INT_TYPE
except ImportError:
    raise ImportError("QES.general_python.algebra.utils module is required for matrix building.")

try:
    from QES.Algebra.Symmetries.symmetry_container import _binary_search_representative_list
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
        h_spaces = hilbert_spaces
    else:
        h_spaces = [hilbert_spaces]
    
    is_real_dtype = np.issubdtype(dtype, np.floating)
    alloc_dtype = dtype
    
    # Check for complex symmetries
    has_complex_sym = any(getattr(hs, 'has_complex_symmetries', False) for hs in h_spaces)
    
    # Check repr_phase arrays
    repr_phase_is_complex = False
    for hs in h_spaces:
        repr_phase = getattr(hs, 'repr_phase', None)
        if repr_phase is not None and np.iscomplexobj(repr_phase):
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
            for hs in h_spaces:
                if hs.dim > 0:
                    test_state = int(hs[0])
                    _, test_values = operator_func(test_state)
                    if len(test_values) > 0 and np.iscomplexobj(test_values[0]):
                        operator_returns_complex = True
                        break
        except:
            pass
    
    # Force complex if needed
    if not is_real_dtype or has_complex_sym or repr_phase_is_complex or hilbert_dtype_is_complex or has_symmetries or operator_returns_complex:
        alloc_dtype = np.complex128
    
    return alloc_dtype

# -----------------------------------------------------------------------------------------------
#! JITTED FUNCTIONS
# -----------------------------------------------------------------------------------------------

@numba.njit(cache=True, nogil=True)
def _build_sparse_same_sector_jit(
                representative_list : np.ndarray,
                normalization       : np.ndarray,
                repr_idx            : np.ndarray,
                repr_phase          : np.ndarray,
                operator_func       : Callable,
                rows                : np.ndarray,
                cols                : np.ndarray,
                data                : np.ndarray,
                data_idx            : int
            ):
    """
    Jitted builder for same-sector operators using precomputed arrays.
    """
    nh = len(representative_list)
    for k in range(nh):
        state               = representative_list[k]
        norm_k              = normalization[k]
        new_states, values  = operator_func(state)

        for i in range(len(new_states)):
            new_state = new_states[i]
            value     = values[i]
            
            if np.abs(value) < 1e-14:
                continue

            # Use precomputed lookup
            idx = repr_idx[new_state]
            
            # Check if state is in this sector (idx == -1 means not in sector)
            if idx < 0:
                continue
                
            phase       = repr_phase[new_state]
            norm_idx    = normalization[idx]
            sym_factor  = np.conj(phase) * norm_idx / norm_k

            matrix_elem = value * sym_factor
            if np.abs(matrix_elem) > 1e-14:
                rows[data_idx]  = idx
                cols[data_idx]  = k
                data[data_idx]  = matrix_elem
                data_idx       += 1

    return data_idx

@numba.njit(cache=True, nogil=True)
def _build_sparse_same_sector_no_symmetry_jit(
                basis               : Optional[np.ndarray],
                operator_func       : Callable,
                rows                : np.ndarray,
                cols                : np.ndarray,
                data                : np.ndarray,
                data_idx            : int,
                nh                  : int
            ):
    """
    Jitted builder for no-symmetry case.
    """
    for k in range(nh):
        if basis is not None:
            state = basis[k]
        else:
            state = k
            
        new_states, values = operator_func(state)

        for i in range(len(new_states)):
            new_state = new_states[i]
            value     = values[i]
            
            if np.abs(value) < 1e-14:
                continue

            # For no-symmetry, the index is the state itself if basis is None
            # If basis is present, we need to find the index of new_state in basis.
            # Assuming basis is sorted, we can use binary search.
            # If basis is None, idx = new_state.
            
            if basis is not None:
                # Binary search
                idx = _binary_search_representative_list(basis, new_state)
            else:
                idx = new_state
            
            if idx < 0 or idx >= nh:
                continue
            
            rows[data_idx] = k
            cols[data_idx] = idx
            data[data_idx] = value
            data_idx      += 1            
    return data_idx

def _build_sparse_same_sector_py(
                hilbert_space       : HilbertSpace,
                representative_list : np.ndarray,
                normalization       : np.ndarray,
                operator_func       : Callable[[int], Tuple[np.ndarray, np.ndarray]],
                rows                : np.ndarray,
                cols                : np.ndarray,
                data                : np.ndarray,
                data_idx            : int
            ):
    """
    Pure-Python fallback builder for same-sector operators that uses
    HilbertSpace.find_representative(...) to obtain the representative index
    and the correct normalization/phase factor. This mirrors the C++
    findRep(baseIdx, nB) semantics and ensures the matrix element is set as
    val * sym_factor where sym_factor includes normalization ratio and
    conjugation when required.
    """
    nh              = len(representative_list)
    use_precomputed = hasattr(hilbert_space, 'repr_idx') and hilbert_space.repr_idx is not None

    for k in range(nh):
        state   = int(representative_list[k])
        norm_k  = normalization[k] if normalization is not None else 1.0

        new_states, values = operator_func(state)

        # support Python sequences and numpy arrays
        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue

            if use_precomputed:
                idx             = int(hilbert_space.repr_idx[int(new_state)])
                # Check if state is in this sector (idx == -1 means not in sector)
                if idx < 0:
                    continue
                # repr_phase stores: conj(phase_to_rep) for non-rep states, 1.0 for rep states
                # Matrix element formula: conj(phase) x norm_idx / norm_k
                phase           = hilbert_space.repr_phase[int(new_state)]
                norm_idx        = normalization[idx] if normalization is not None else 1.0
                sym_factor      = np.conj(phase) * norm_idx / norm_k
            else:
                rep, sym_factor = hilbert_space._sym_container.find_representative(int(new_state), norm_k)
                # Binary search returns -1 if state not in this sector
                idx             = _binary_search_representative_list(representative_list, rep)
                if idx < 0:
                    continue  # State not in this symmetry sector, skip

            # idx is valid (>= 0), add matrix element
            matrix_elem = value * sym_factor
            if abs(matrix_elem) > 1e-14:
                # C++ does H(idx, k) where idx is output state, k is input state
                # This matches operator convention: H|k⟩ gives contributions to state |idx⟩
                rows[data_idx]  = int(idx)  # Row = output state
                cols[data_idx]  = int(k)    # Col = input state  
                data[data_idx]  = matrix_elem
                data_idx       += 1

    return data_idx

def _build_sparse_same_sector_no_symmetry_py(
                hilbert_space       : HilbertSpace,
                operator_func       : Callable[[int], Tuple[np.ndarray, np.ndarray]],
                rows                : np.ndarray,
                cols                : np.ndarray,
                data                : np.ndarray,
                data_idx            : int,
            ):
    """
    Optimized Python builder for the no-symmetry case where the representative_list is
    a contiguous identity representative_list (state index == representative index).

    This avoids calling HilbertSpace.find_representative and is the fast
    fallback for systems without symmetries.
    """
    
    # iterate over reduced basis indices without allocating a representative_list array
    nh          = hilbert_space.dim
    for k in range(nh):
        # obtain the full-state integer for reduced-basis index k
        state               = int(hilbert_space[k]) # get mapped state[k]
        new_states, values  = operator_func(state)

        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue

            idx             = int(new_state)
            
            # if the operator returned an index outside the reduced space skip
            if idx < 0 or idx >= nh:
                continue
            
            matrix_elem     = value

            if abs(matrix_elem) > 1e-14:
                rows[data_idx] = int(k)
                cols[data_idx] = int(idx)
                data[data_idx] = matrix_elem
                data_idx      += 1            
    return data_idx

def _build_dense_same_sector_py(
                hilbert_space       : HilbertSpace,
                normalization       : np.ndarray,
                operator_func       : Callable[[int], Tuple[np.ndarray, np.ndarray]],
                matrix              : np.ndarray,
            ):
    """
    Build dense matrix for operator within same sector.
    """
    # Get the number of basis states
    nh                      = hilbert_space.dim
    representative_list     = hilbert_space.representative_list if hasattr(hilbert_space, 'representative_list') else np.arange(nh, dtype=np.int64)
    use_precomputed         = hasattr(hilbert_space, 'repr_idx') and hilbert_space.repr_idx is not None

    # Iterate over reduced basis indices
    for k in range(nh):
        
        # Obtain the full-state integer for reduced-basis index k
        state               = int(hilbert_space[k])
        norm_k              = normalization[k] if normalization is not None else 1.0
        new_states, values  = operator_func(state)

        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue

            if use_precomputed:
                idx             = int(hilbert_space.repr_idx[int(new_state)])
                phase           = hilbert_space.repr_phase[int(new_state)]
                norm_rep        = normalization[idx] if normalization is not None else 1.0
                sym_factor      = phase * norm_k / norm_rep
            else:
                rep, sym_factor = hilbert_space.find_repr(int(new_state), norm_k)
                idx             = _binary_search_representative_list(representative_list, rep)

            if idx >= 0:
                # accumulate (multiple contributions may map to same repr)
                matrix[k, idx] += value * sym_factor

    return matrix

# ------------------------------------------------------------------------------------------

def _build_same_sector(hilbert_space    : HilbertSpace,
                    operator_func       : Callable,
                    sparse              : bool,
                    max_local_changes   : int,
                    dtype               : np.dtype
                ) -> Union[sp.csr_matrix, np.ndarray]:
    """Build matrix for operator within same sector."""
    
    nh                      = hilbert_space.dim
    ns                      = hilbert_space.ns
    representative_list     = getattr(hilbert_space, 'representative_list', None)
    normalization           = getattr(hilbert_space, 'normalization', None)
    
    # Determine appropriate dtype
    alloc_dtype             = _determine_matrix_dtype(dtype, hilbert_space, operator_func, ns)
    
    if not sparse:
        # Dense matrix
        matrix              = np.zeros((nh, nh), dtype=alloc_dtype)
        return _build_dense_same_sector_py(hilbert_space, normalization, operator_func, ns, matrix)
    
    # Sparse matrix
    max_nnz                 = nh * max_local_changes * (ns) # estimate
    rows                    = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    cols                    = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    data                    = np.zeros(max_nnz, dtype=alloc_dtype)

    # If there are no symmetries present (no representative_list/repr arrays and no
    # symmetry group), use a simple optimized builder that iterates the
    # HilbertSpace directly (no representative_list allocation). Otherwise use the
    # Python assembly that uses HilbertSpace.find_representative
    has_symmetry            = representative_list is not None

    if not has_symmetry:
        # Optimized no-symmetry builder
        # Note: 'basis' here refers to the constrained basis (e.g. particle conservation)
        # which is distinct from 'representative_list' used for symmetry reduction.
        basis = getattr(hilbert_space, 'basis', None)
        # If basis is not an array (e.g. None or list), convert or handle
        if basis is not None and not isinstance(basis, np.ndarray):
             basis = np.array(basis)
             
        data_idx = _build_sparse_same_sector_no_symmetry_jit(basis, operator_func, rows, cols, data, 0, nh)
    else:
        # Check if we have precomputed arrays for JIT
        # 'representative_list' here implies symmetry reduction is active.
        # If 'repr_idx' (lookup table) is available, we use the fast JIT path.
        # Otherwise, we must calculate representatives on the fly (slow Python path).
        repr_idx = getattr(hilbert_space, 'repr_idx', None)
        repr_phase = getattr(hilbert_space, 'repr_phase', None)
        
        if repr_idx is not None and repr_phase is not None:
             data_idx = _build_sparse_same_sector_jit(
                 representative_list, normalization, repr_idx, repr_phase,
                 operator_func, rows, cols, data, 0
             )
        else:
            # Python assembly that uses HilbertSpace.find_representative
            # This is necessary when the lookup table is too large to store.
            data_idx    = _build_sparse_same_sector_py(hilbert_space, representative_list, normalization, operator_func, rows, cols, data, 0)

    return sp.csr_matrix((data[:data_idx], (rows[:data_idx], cols[:data_idx])),shape=(nh, nh), dtype=alloc_dtype)

def _build_sector_change(hilbert_in         : object,
                        hilbert_out         : object,
                        operator_func       : Callable,
                        sparse              : bool,
                        max_local_changes   : int,
                        dtype               : Union[np.dtype, str],
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
    
    if not sparse:
        # Dense matrix
        matrix                  = np.zeros((nh_out, nh_in), dtype=alloc_dtype)
        use_precomputed         = hasattr(hilbert_out, 'repr_idx') and hilbert_out.repr_idx is not None
        
        for idx_col in range(nh_in):
            state_col           = int(hilbert_in[idx_col])
            norm_col            = norm_in[idx_col] if norm_in is not None else 1.0
            new_states, values  = operator_func(state_col)

            for new_state, value in zip(new_states, values):
                if abs(value) < 1e-14:
                    continue

                if use_precomputed:
                    rep             = int(hilbert_out.repr_idx[int(new_state)])
                    phase           = hilbert_out.repr_phase[int(new_state)]
                    norm_rep        = norm_out[rep] if norm_out is not None else 1.0
                    sym_factor      = phase * norm_col / norm_rep
                else:
                    rep, sym_factor = hilbert_out.find_representative(int(new_state), norm_col)

                idx_row = _binary_search_representative_list(representative_list_out, rep)
                if idx_row >= 0:
                    matrix[idx_row, idx_col] += value * sym_factor
        return matrix
    
    # Sparse matrix
    max_nnz         = nh_in * max_local_changes * ns
    rows            = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    cols            = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    data            = np.zeros(max_nnz, dtype=alloc_dtype)

    # Python assembly that uses HilbertSpace.find_representative
    use_precomputed = hasattr(hilbert_out, 'repr_idx') and hilbert_out.repr_idx is not None
    data_idx        = 0
    
    # Iterate over input basis states
    for idx_col in range(nh_in):
        state_col           = int(hilbert_in[idx_col])
        norm_col            = norm_in[idx_col] if norm_in is not None else 1.0
        new_states, values  = operator_func(state_col)

        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue

            if use_precomputed:
                rep             = int(hilbert_out.repr_idx[int(new_state)])
                phase           = hilbert_out.repr_phase[int(new_state)]
                norm_rep        = norm_out[rep] if norm_out is not None else 1.0
                sym_factor      = phase * norm_col / norm_rep
            else:
                rep, sym_factor = hilbert_out.find_representative(int(new_state), norm_col)

            idx_row = _binary_search_representative_list(representative_list_out, rep)
            if idx_row >= 0:
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

# ------------------------------------------------------------------------------------------
# 1. DENSE KERNEL (Fastest for small systems)
# ------------------------------------------------------------------------------------------

# @numba.njit(nogil=True) # nogil for thread safety -> allows multi-threading if needed, stands for "no global interpreter lock"
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

# ------------------------------------------------------------------------------------------
# 2. SPARSE KERNEL (The "Pause and Resume" Engine)
# ------------------------------------------------------------------------------------------

# @numba.njit(nogil=True)
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

    # Determine precision
    alloc_dtype = dtype
    
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
        if nh is None or ns is None:
            raise ValueError("If hilbert_space is None, nh and ns must be provided")
        if hilbert_space_out is not None:
            raise ValueError("hilbert_space_out not supported without hilbert_space")
        return _build_no_hilbert(operator_func, nh, ns, sparse, max_local_changes, dtype)
    
    # Use provided or default values
    nh = nh or hilbert_space.dim
    ns = ns or hilbert_space.ns
    
    if hilbert_space_out is None:
        return _build_same_sector(hilbert_space, operator_func, sparse, max_local_changes, dtype)
    else:
        return _build_sector_change(hilbert_space, hilbert_space_out, operator_func, sparse, max_local_changes, dtype)

# -------------------------------------------------------------------------------
#! Matrix-Free Operator Application (optimized!)
# -------------------------------------------------------------------------------

# @numba.njit(fastmath=True, parallel=False) # Serial is often faster here to avoid race conditions on write
def _apply_op_batch_jit(
    vecs_in             : np.ndarray,
    vecs_out            : np.ndarray,
    op_func             : Callable,
    op_args             : Tuple,
    basis               : Optional[np.ndarray],
    representative_list : Optional[np.ndarray],
    normalization       : Optional[np.ndarray],
    repr_idx            : Optional[np.ndarray],
    repr_phase          : Optional[np.ndarray]) -> None:
    
    nh, n_batch     = vecs_in.shape
    has_symmetry    = representative_list is not None
    has_basis       = basis is not None

    # Loop over the Hilbert space (Rows)
    for k in range(nh):
        
        # 1. Identify the state integer (Do this ONCE per row)
        if has_symmetry:
            state   = representative_list[k]
            norm_k  = normalization[k]
        elif has_basis:
            state   = basis[k]
        else:
            state   = k
            
        # 2. Compute Operator Action (Do this ONCE per row)
        new_states, values = op_func(state, *op_args)
        
        # 3. Map back to indices and apply to the WHOLE BATCH
        for i in range(len(new_states)):
            new_state   = new_states[i]
            val         = values[i]
            
            if abs(val) < 1e-15:
                continue
            
            target_idx  = -1
            sym_factor  = 1.0 + 0j
            
            # Find target index
            if has_symmetry:
                if repr_idx is not None:
                    target_idx = repr_idx[new_state]
                    if target_idx >= 0:
                        phase       = repr_phase[new_state]
                        norm_idx    = normalization[target_idx]
                        sym_factor  = np.conj(phase) * (norm_idx / norm_k)
            elif has_basis:
                # Binary Search
                found_idx = np.searchsorted(basis, new_state)
                if found_idx < nh and basis[found_idx] == new_state:
                    target_idx = found_idx
            else:
                target_idx = new_state

            # 4. Vectorized Update (SIMD)
            # Instead of looping over 'b', we update the whole row slice.
            # This removes the Python-loop overhead for the batch dimension.
            if 0 <= target_idx < nh:
                # vecs_in[k, :] is shape (n_batch,)
                # vecs_out[target_idx, :] is shape (n_batch,)
                if has_symmetry:
                    # Complex multiplication with symmetry factor
                    factor                      = val * sym_factor
                    vecs_out[target_idx, :]    += factor * vecs_in[k, :]
                else:
                    vecs_out[target_idx, :]    += val * vecs_in[k, :]

# @numba.njit(parallel=False, fastmath=True)
def _apply_fourier_batch_jit(
    vecs_in             : np.ndarray,
    vecs_out            : np.ndarray,
    phases              : np.ndarray,
    op_func             : Callable,
    base_args           : Tuple, 
    basis               : Optional[np.ndarray],
    representative_list : Optional[np.ndarray],
    normalization       : Optional[np.ndarray],
    repr_idx            : Optional[np.ndarray],
    repr_phase          : Optional[np.ndarray]) -> None:
    
    nh, n_batch     = vecs_in.shape
    n_sites         = len(phases)
    
    has_symmetry    = representative_list is not None
    has_basis       = basis is not None

    # Loop Hilbert Space
    for k in range(nh):
        # Optimization: Skip empty input rows if using iterative solvers on sparse states
        # if np.abs(vecs_in[k, 0]) < 1e-15: continue 

        # Decode State (Once per basis state)
        if has_symmetry:
            state   = representative_list[k]
            norm_k  = normalization[k]
        elif has_basis:
            state   = basis[k]
        else:
            state   = k

        # Loop Sites (Accumulate operator terms)
        for site_idx in range(n_sites):
            c_site              = phases[site_idx]
            new_states, values  = op_func(state, site_idx, *base_args)
            
            for j in range(len(new_states)):
                new_state       = new_states[j]
                val             = values[j]
                
                # Combined factor
                factor          = val * c_site 
                if abs(factor) < 1e-15: continue

                # Find Target
                target_idx      = -1
                sym_factor      = 1.0 + 0j

                if has_symmetry:
                    if repr_idx is not None:
                        target_idx = repr_idx[new_state]
                        if target_idx >= 0:
                            ph          = repr_phase[new_state]
                            norm_new    = normalization[target_idx]
                            sym_factor  = np.conj(ph) * (norm_new / norm_k)
                elif has_basis:
                    # Binary Search
                    target_idx = np.searchsorted(basis, new_state)
                    if target_idx >= nh or basis[target_idx] != new_state:
                        target_idx = -1
                else:
                    target_idx = new_state

                # Vectorized Batch Update
                if 0 <= target_idx < nh:
                    if has_symmetry:
                        # vecs_in[k] is a contiguous view of the batch row
                        vecs_out[target_idx] += (factor * sym_factor) * vecs_in[k]
                    else:
                        vecs_out[target_idx] += factor * vecs_in[k]

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