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
version     : 2.0.0
--------------------------------------------------------------------------------------------
"""
import numpy as np
import scipy.sparse as sp
import numba
from typing import Callable, Optional, Union, Tuple
from QES.general_python.algebra.utils import DEFAULT_NP_INT_TYPE

# --- JIT accelerated builders that use precomputed repr arrays ---------------------------------
@numba.njit(cache=True)
def _build_sparse_same_sector_jit(
    mapping,
    normalization,
    repr_idx,
    repr_phase,
    operator_func,
    ns,
    rows,
    cols,
    data,
    data_idx
):
    nh = len(mapping)
    for k in range(nh):
        state = mapping[k]
        norm_k = normalization[k] if normalization is not None else 1.0

        new_states, values = operator_func(state, ns)
        for i in range(len(new_states)):
            new_state = new_states[i]
            value = values[i]
            if abs(value) < 1e-14:
                continue

            idx_rep = repr_idx[new_state]
            if idx_rep >= 0:
                norm_rep = normalization[idx_rep] if normalization is not None else 1.0
                phase = repr_phase[new_state]
                # same-sector: factor = norm_rep / norm_k * conj(phase)
                matrix_elem = value * (norm_rep / norm_k) * np.conj(phase)
                if abs(matrix_elem) > 1e-14:
                    rows[data_idx] = k
                    cols[data_idx] = idx_rep
                    data[data_idx] = matrix_elem
                    data_idx += 1
    return data_idx


@numba.njit(cache=True)
def _build_sparse_sector_change_jit(
    mapping_in,
    mapping_out,
    norm_in,
    norm_out,
    repr_idx_out,
    repr_phase_out,
    operator_func,
    ns,
    rows,
    cols,
    data,
    data_idx
):
    nh_in = len(mapping_in)
    for idx_col in range(nh_in):
        state_col = mapping_in[idx_col]
        norm_col = norm_in[idx_col] if norm_in is not None else 1.0

        new_states, values = operator_func(state_col, ns)
        for i in range(len(new_states)):
            new_state = new_states[i]
            value = values[i]
            if abs(value) < 1e-14:
                continue

            idx_row = repr_idx_out[new_state]
            if idx_row >= 0:
                norm_row = norm_out[idx_row] if norm_out is not None else 1.0
                phase = repr_phase_out[new_state]
                # sector-change: C++ does op(newIdxA, idxB) += valB * conj(symValA)
                # where findRep returned norm_row/norm_col * conj(phase), so
                # conjugating gives norm_row/norm_col * phase
                matrix_elem = value * (norm_row / norm_col) * phase
                if abs(matrix_elem) > 1e-14:
                    rows[data_idx] = idx_row
                    cols[data_idx] = idx_col
                    data[data_idx] = matrix_elem
                    data_idx += 1
    return data_idx

# -----------------------------------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def _binary_search_mapping(mapping, state):
    """
    Binary search to find index of state in sorted mapping array.
    Memory efficient - O(1) space, O(log Nh) time.
    
    Args:
        mapping: Sorted array of representative states
        state: State to find
        
    Returns:
        Index in mapping if found, -1 otherwise
    """
    left = 0
    right = len(mapping) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if mapping[mid] == state:
            return mid
        elif mapping[mid] < state:
            left = mid + 1
        else:
            right = mid - 1
    
    return numba.int64(-1)

@numba.njit(cache=True, fastmath=True)
def _build_dense_same_sector(
    mapping,
    normalization,
    operator_func,
    ns,
    matrix,
    use_binary_search
):
    """
    Build dense matrix for operators within the same sector.
    
    Args:
        mapping: Representative states array
        normalization: Normalization factors
        operator_func: Numba function
        ns: Number of sites
        matrix: Preallocated dense matrix (nh x nh)
        use_binary_search: Use binary search for lookup
    """
    nh = len(mapping)
    
    for k in range(nh):
        state = mapping[k]
        norm_k = normalization[k] if normalization is not None else 1.0
        
        new_states, values = operator_func(state, ns)
        
        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue
            
            if use_binary_search:
                idx = _binary_search_mapping(mapping, new_state)
            else:
                idx = numba.int64(-1)
                for j in range(nh):
                    if mapping[j] == new_state:
                        idx = j
                        break
            
            if idx >= 0:
                norm_new = normalization[idx] if normalization is not None else 1.0
                matrix[k, idx] += value * norm_k / norm_new

def _build_sparse_same_sector_py(
    hilbert_space,
    mapping,
    normalization,
    operator_func,
    ns,
    rows,
    cols,
    data,
    data_idx
):
    """
    Pure-Python fallback builder for same-sector operators that uses
    HilbertSpace.find_representative(...) to obtain the representative index
    and the correct normalization/phase factor. This mirrors the C++
    findRep(baseIdx, nB) semantics and ensures the matrix element is set as
    val * sym_factor where sym_factor includes normalization ratio and
    conjugation when required.
    """
    nh = len(mapping)

    for k in range(nh):
        state = int(mapping[k])
        norm_k = normalization[k] if normalization is not None else 1.0

        new_states, values = operator_func(state, ns)

        # support Python sequences and numpy arrays
        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue

            # find representative in the same (output) Hilbert space
            idx_rep, sym_factor = hilbert_space.find_representative(int(new_state), norm_k)

            # sym_factor already encodes normalization[idx]/norm_k * conj(sym_eig)
            if idx_rep is None or sym_factor == 0:
                continue

            matrix_elem = value * sym_factor
            if abs(matrix_elem) > 1e-14:
                rows[data_idx] = int(k)
                cols[data_idx] = int(idx_rep)
                data[data_idx] = matrix_elem
                data_idx += 1

    return data_idx

def _build_sparse_same_sector_no_symmetry_py(
    hilbert_space,
    normalization,
    operator_func,
    ns,
    rows,
    cols,
    data,
    data_idx,
    use_binary_search=True,
):
    """
    Optimized Python builder for the no-symmetry case where the mapping is
    a contiguous identity mapping (state index == representative index).

    This avoids calling HilbertSpace.find_representative and is the fast
    fallback for systems without symmetries.
    """
    # iterate over reduced basis indices without allocating a mapping array
    nh = hilbert_space.dim

    for k in range(nh):
        # obtain the full-state integer for reduced-basis index k
        state = int(hilbert_space[k])
        norm_k = normalization[k] if normalization is not None else 1.0

        new_states, values = operator_func(state, ns)

        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue

            idx = int(new_state)
            # if the operator returned an index outside the reduced space skip
            if idx < 0 or idx >= nh:
                continue

            norm_new = normalization[idx] if normalization is not None else 1.0
            matrix_elem = value * (norm_k / norm_new)

            if abs(matrix_elem) > 1e-14:
                rows[data_idx] = int(k)
                cols[data_idx] = int(idx)
                data[data_idx] = matrix_elem
                data_idx += 1

    return data_idx

def build_operator_matrix(
    hilbert_space,
    operator_func: Callable,
    hilbert_space_out = None,
    sparse: bool = True,
    max_local_changes: int = 2,
    dtype = np.float64,
    use_binary_search: bool = True
) -> Union[sp.csr_matrix, np.ndarray]:
    """
    Build operator matrix with support for sector-changing operators.
    
    This version supports:
    - Same sector: Square matrices (nh x nh)
    - Different sectors: Rectangular matrices (nh_out x nh_in)
    - Binary search for O(log Nh) lookup (requires sorted mapping)
    - Memory-efficient construction
    
    Args:
        hilbert_space: Input Hilbert space (column space)
        operator_func: Numba function (state, ns) -> (new_states, values)
        hilbert_space_out: Output Hilbert space (row space), None for same sector
        sparse: Use sparse format
        max_local_changes: Estimated non-zeros per row
        dtype: Matrix data type
        use_binary_search: Use O(log Nh) binary search (mapping must be sorted)
        
    Returns:
        Operator matrix (sparse CSR or dense array)
        
    Example:
        # Same sector (square matrix)
        H = build_operator_matrix(hilbert, hamiltonian_op)
        
        # Sector-changing (rectangular matrix)
        # e.g., creation operator changing particle number
        c_dag = build_operator_matrix(
            hilbert_n, c_dag_op, hilbert_space_out=hilbert_n_plus_1
        )
    """
    if hilbert_space_out is None:
        return _build_same_sector(
            hilbert_space, operator_func, sparse, max_local_changes, dtype, use_binary_search
        )
    else:
        return _build_sector_change(
            hilbert_space, hilbert_space_out, operator_func, sparse, max_local_changes, dtype, use_binary_search
        )

def _build_same_sector(
    hilbert_space,
    operator_func,
    sparse,
    max_local_changes,
    dtype,
    use_binary_search
):
    """Build matrix for operator within same sector."""
    nh = hilbert_space.dim
    ns = hilbert_space.ns
    
    # If the Hilbert space has an explicit mapping (symmetry-reduced), use it.
    # Otherwise avoid constructing a mapping array for the no-symmetry case
    # — we'll iterate the `hilbert_space` directly in the ultrafast path.
    has_mapping     = hasattr(hilbert_space, 'mapping') and hilbert_space.mapping is not None and len(hilbert_space.mapping) > 0
    mapping         = hilbert_space.mapping if has_mapping else None
    normalization   = hilbert_space.normalization if has_mapping and hasattr(hilbert_space, 'normalization') else None

    if sparse:
        max_nnz     = nh * max_local_changes * ns
        rows        = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
        cols        = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
        alloc_dtype = dtype
        try:
            is_real_dtype = np.issubdtype(np.dtype(dtype), np.floating)
        except Exception:
            is_real_dtype = False
            
        # prefer complex storage only if the symmetry eigenvalues can be complex
        has_complex_sym = False
        try:
            has_complex_sym = bool(getattr(hilbert_space, 'has_complex_symmetries', False))
        except Exception:
            has_complex_sym = False
        # If repr_phase exists and contains complex entries, force complex dtype
        repr_phase = getattr(hilbert_space, 'repr_phase', None)
        repr_phase_is_complex = False
        try:
            if repr_phase is not None:
                repr_phase_is_complex = np.iscomplexobj(repr_phase)
        except Exception:
            repr_phase_is_complex = False

        # Also force complex if Hilbert space dtype is complex (avoid data loss)
        hilbert_dtype_is_complex = False
        try:
            hd = getattr(hilbert_space, '_dtype', None)
            if hd is not None:
                try:
                    hilbert_dtype_is_complex = np.issubdtype(np.dtype(hd), np.complexfloating)
                except Exception:
                    # fallback: check instance
                    hilbert_dtype_is_complex = isinstance(hd, complex) or isinstance(hd, np.complexfloating)
        except Exception:
            hilbert_dtype_is_complex = False

        if is_real_dtype and (has_complex_sym or repr_phase_is_complex or hilbert_dtype_is_complex):
            alloc_dtype = np.complex128
        data        = np.zeros(max_nnz, dtype=alloc_dtype)

        # If there are no symmetries present (no mapping/repr arrays and no
        # symmetry group), use a simple optimized builder that iterates the
        # HilbertSpace directly (no mapping allocation). Otherwise use the
        # JIT or Hilbert-aware assembly paths.
        has_symmetry = (mapping is not None) or (getattr(hilbert_space, 'repr_idx', None) is not None) or (getattr(hilbert_space, '_sym_group', None) is not None)

        if not has_symmetry:
            # Optimized no-symmetry pure-Python builder — iterate hilbert_space directly
            data_idx = _build_sparse_same_sector_no_symmetry_py(
                hilbert_space, normalization, operator_func, ns,
                rows, cols, data, 0, use_binary_search
            )
        else:
            # Fast JIT path: use repr_idx/repr_phase arrays if present and operator_func is a numba function
            use_jit = False
            if getattr(hilbert_space, 'repr_idx', None) is not None and getattr(hilbert_space, 'repr_phase', None) is not None:
                # operator_func may be a numba-compiled function (has attribute 'py_func' or is a cpu dispatcher)
                try:
                    if hasattr(operator_func, 'signatures') or hasattr(operator_func, 'py_func'):
                        use_jit = True
                except Exception:
                    use_jit = False

            if use_jit:
                # Ensure we pass an explicit integer mapping array to the jitted builder
                repr_idx = hilbert_space.repr_idx
                repr_phase = hilbert_space.repr_phase
                mapping_for_jit = mapping if mapping is not None else np.arange(nh, dtype=np.int64)
                data_idx = _build_sparse_same_sector_jit(
                    mapping_for_jit, normalization, repr_idx, repr_phase,
                    operator_func, ns, rows, cols, data, 0
                )
            else:
                # fallback: Python Hilbert-aware assembly
                data_idx = _build_sparse_same_sector_py(
                    hilbert_space, mapping, normalization, operator_func, ns,
                    rows, cols, data, 0
                )

        return sp.csr_matrix(
            (data[:data_idx], (rows[:data_idx], cols[:data_idx])),
            shape=(nh, nh), dtype=alloc_dtype
        )
    else:
        # Dense path: create a temporary explicit mapping if necessary since
        # the dense numba builder expects an array-like mapping. Dense builds
        # are uncommon for large Hilbert spaces; constructing this small
        # array here is acceptable.
        if mapping is None:
            mapping = np.arange(nh, dtype=np.int64)
        matrix = np.zeros((nh, nh), dtype=dtype)
        _build_dense_same_sector(
            mapping, normalization, operator_func, ns, matrix, use_binary_search
        )
        return matrix

def _build_sector_change(hilbert_in         : object,
                        hilbert_out         : object,
                        operator_func       : Callable,
                        sparse              : bool,
                        max_local_changes   : int,
                        dtype               : Union[np.dtype, str],
                        use_binary_search   : bool
                    ):
    """Build rectangular matrix for sector-changing operator."""
    nh_in               = hilbert_in.dim
    nh_out              = hilbert_out.dim
    ns                  = hilbert_in.ns
    
    mapping_in          = hilbert_in.mapping if hasattr(hilbert_in, 'mapping') else np.arange(nh_in, dtype=np.int64)
    mapping_out         = hilbert_out.mapping if hasattr(hilbert_out, 'mapping') else np.arange(nh_out, dtype=np.int64)
    
    norm_in             = hilbert_in.normalization if hasattr(hilbert_in, 'normalization') else None
    norm_out            = hilbert_out.normalization if hasattr(hilbert_out, 'normalization') else None
    
    if not sparse:
        raise NotImplementedError("Dense sector-changing matrices not yet implemented. Use sparse=True.")
    
    max_nnz             = nh_in * max_local_changes * ns
    rows                = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    cols                = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    # allocate complex storage if either Hilbert space has symmetries
    alloc_dtype         = dtype
    try:
        is_real_dtype   = np.issubdtype(np.dtype(dtype), np.floating)
    except Exception:
        is_real_dtype   = False
        
    #! Decide whether complex storage is required:
    # - if either Hilbert reports complex symmetry eigenvalues
    # - or repr_phase arrays contain complex entries
    # - or the Hilbert space dtype is complex
    repr_phase_out              = getattr(hilbert_out, 'repr_phase', None)
    repr_phase_out_is_complex   = False
    try:
        if repr_phase_out is not None:
            repr_phase_out_is_complex   = np.iscomplexobj(repr_phase_out)
    except Exception:
        repr_phase_out_is_complex       = False

    has_complex_sym_in          = bool(getattr(hilbert_in, 'has_complex_symmetries', False)) if getattr(hilbert_in, 'has_complex_symmetries', None) is not None else False
    has_complex_sym_out         = bool(getattr(hilbert_out, 'has_complex_symmetries', False)) if getattr(hilbert_out, 'has_complex_symmetries', None) is not None else False

    hilbert_in_dtype_complex    = False
    hilbert_out_dtype_complex   = False
    try:
        hd_in = getattr(hilbert_in, '_dtype', None)
        if hd_in is not None:
            hilbert_in_dtype_complex    = np.issubdtype(np.dtype(hd_in), np.complexfloating)
    except Exception:
        hilbert_in_dtype_complex        = False
    try:
        hd_out = getattr(hilbert_out, '_dtype', None)
        if hd_out is not None:
            hilbert_out_dtype_complex   = np.issubdtype(np.dtype(hd_out), np.complexfloating)
    except Exception:
        hilbert_out_dtype_complex       = False

    # prefer complex storage only if the symmetry eigenvalues can be complex
    if is_real_dtype and (has_complex_sym_in or has_complex_sym_out or repr_phase_out_is_complex or hilbert_in_dtype_complex or hilbert_out_dtype_complex):
        alloc_dtype = np.complex128
        
    # Allocate data array
    data    = np.zeros(max_nnz, dtype=alloc_dtype)

    # Fast JIT path when repr arrays are available and operator_func is jitted
    use_jit = False
    if getattr(hilbert_out, 'repr_idx', None) is not None and getattr(hilbert_out, 'repr_phase', None) is not None:
        try:
            if hasattr(operator_func, 'signatures') or hasattr(operator_func, 'py_func'):
                use_jit = True
        except Exception:
            use_jit = False

    if use_jit:
        repr_idx_out    = hilbert_out.repr_idx
        repr_phase_out  = hilbert_out.repr_phase
        data_idx        = _build_sparse_sector_change_jit(
            mapping_in, mapping_out, norm_in, norm_out,
            repr_idx_out, repr_phase_out,
            operator_func, ns, rows, cols, data, 0
        )
    else:
        # Python assembly that uses HilbertSpace.find_representative and conjugates the factor
        data_idx = 0
        for idx_col in range(nh_in):
            state_col           = int(hilbert_in[idx_col])
            norm_col            = hilbert_in.norm(idx_col)
            new_states, values  = operator_func(state_col, ns)

            for new_state, value in zip(new_states, values):
                if abs(value) < 1e-14:
                    continue

                idx_row, sym_factor = hilbert_out.find_representative(int(new_state), norm_col)
                if idx_row is None or sym_factor == 0:
                    continue

                # Conjugate the returned factor to follow C++ behaviour
                matrix_elem = value * np.conj(sym_factor)
                if abs(matrix_elem) > 1e-14:
                    rows[data_idx]       = int(idx_row)
                    cols[data_idx]       = int(idx_col)
                    data[data_idx]       = matrix_elem
                    data_idx            += 1

    return sp.csr_matrix(
        (data[:data_idx], (rows[:data_idx], cols[:data_idx])),
        shape=(nh_out, nh_in), dtype=alloc_dtype
    )

# -------------------------------------------------------------------------------
#! Symmetry rotation matrix construction
# -------------------------------------------------------------------------------

def get_symmetry_rotation_matrix(hilbert_space, dtype=np.complex128) -> sp.csr_matrix:
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
    
    Args:
        hilbert_space: HilbertSpace with symmetries and mapping
        dtype: Data type for matrix elements (typically complex128)
        
    Returns:
        Sparse CSR matrix U (nh_full x nh_reduced)
        
    Example:
        # Create Hilbert space with translation symmetry
        hs = HilbertSpace(
            lattice=lattice,
            sym_gen=[(SymmetryGenerators.Translation_x, 0)],
            gen_mapping=True
        )
        
        # Get rotation matrix
        U = get_symmetry_rotation_matrix(hs)
        
        # Expand reduced state to full Hilbert space
        psi_reduced = np.array([...])  # State in reduced basis
        psi_full = U @ psi_reduced     # State in full basis
    """
    if not hasattr(hilbert_space, '_sym_group') or not hilbert_space._sym_group:
        raise ValueError("Hilbert space has no symmetries. Rotation matrix not applicable.")
    
    nh_reduced = hilbert_space.dim
    nh_full = hilbert_space._nh_full if hasattr(hilbert_space, '_nh_full') else 2**hilbert_space.ns
    
    mapping = hilbert_space.mapping
    normalization = hilbert_space.normalization if hasattr(hilbert_space, 'normalization') else np.ones(nh_reduced)
    # use public property which exposes callable symmetry ops
    sym_group = hilbert_space.sym_group
    
    sym_size = len(sym_group)
    norm_factor = 1.0 / np.sqrt(sym_size)
    
    rows = []
    cols = []
    data = []
    # Build rotation matrix by casting each reduced basis unit-vector to the
    # full Hilbert space using the HilbertSpace.cast_to_full helper. This
    # ensures the same normalization/phase conventions are used as in the
    # mapping/representation machinery and avoids subtle normalization bugs.
    for k in range(nh_reduced):
        # reduced basis unit vector
        vec_reduced = np.zeros(nh_reduced, dtype=dtype)
        vec_reduced[k] = 1.0
        # cast to full basis (returns full-state vector)
        vec_full = hilbert_space.cast_to_full(vec_reduced)

        # vec_full may be backend-specific; convert to numpy if necessary
        try:
            vec_full_arr = np.asarray(vec_full)
        except Exception:
            vec_full_arr = vec_full

        nz = np.nonzero(np.abs(vec_full_arr) > 1e-14)[0]
        for i in nz:
            rows.append(int(i))
            cols.append(int(k))
            data.append(vec_full_arr[int(i)])

    return sp.csr_matrix(
        (np.array(data, dtype=dtype), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(nh_full, nh_reduced),
        dtype=dtype
    )

