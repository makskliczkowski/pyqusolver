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
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import numba
from typing import Callable, Optional, Union, Tuple, TYPE_CHECKING

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
#! NOT JITTED FUNCTIONS
# -----------------------------------------------------------------------------------------------

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
    max_nnz         = nh * max_local_changes * (ns) # estimate
    rows            = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    cols            = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    data            = np.zeros(max_nnz, dtype=alloc_dtype)

    # If there are no symmetries present (no representative_list/repr arrays and no
    # symmetry group), use a simple optimized builder that iterates the
    # HilbertSpace directly (no representative_list allocation). Otherwise use the
    # Python assembly that uses HilbertSpace.find_representative
    has_symmetry    = representative_list is not None

    if not has_symmetry:
        # Optimized no-symmetry pure-Python builder — iterate hilbert_space directly
        data_idx    = _build_sparse_same_sector_no_symmetry_py(hilbert_space, operator_func, rows, cols, data, 0)
    else:
        # Python assembly that uses HilbertSpace.find_representative
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

# ------------------------------------------------------------------------------------------

def _build_no_hilbert(operator_func       : Callable,
                      nh                  : int,
                      ns                  : int,
                      sparse              : bool,
                      max_local_changes   : int,
                      dtype               : np.dtype
                     ) -> Union[sp.csr_matrix, np.ndarray]:
    """Build matrix for operator without Hilbert space (no symmetries)."""
    
    alloc_dtype = _determine_matrix_dtype(dtype, None, operator_func, ns)
    
    if not sparse:
        # Dense matrix
        matrix = np.zeros((nh, nh), dtype=alloc_dtype)
        for k in range(nh):
            new_states, values = operator_func(k)
            for new_state, value in zip(new_states, values):
                if abs(value) < 1e-14:
                    continue
                if 0 <= new_state < nh:
                    matrix[k, new_state] += value
        return matrix
    
    # Sparse matrix
    max_nnz     = nh * max_local_changes
    rows        = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    cols        = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    data        = np.zeros(max_nnz, dtype=alloc_dtype)
    data_idx    = 0
    
    for k in range(nh):
        new_states, values = operator_func(k)
        for new_state, value in zip(new_states, values):
            if abs(value) < 1e-14:
                continue
            if 0 <= new_state < nh:
                rows[data_idx] = k
                cols[data_idx] = new_state
                data[data_idx] = value
                data_idx      += 1
    
    return sp.csr_matrix((data[:data_idx], (rows[:data_idx], cols[:data_idx])), shape=(nh, nh), dtype=alloc_dtype)

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
    if hilbert_space is None:
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