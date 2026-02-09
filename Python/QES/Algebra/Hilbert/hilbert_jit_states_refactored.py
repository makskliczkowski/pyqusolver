r"""
Support for JIT-compiled Hilbert space state amplitude calculations.

This module provides efficient routines for computing amplitudes of
various quantum many-body states, including Slater determinants,
Bogoliubov vacua, permanents, and Gaussian states using Numba for JIT compilation.

Key Features:
- Buffer/workspace preallocation support for memory efficiency
- Unified extraction using optimized binary.py helpers  
- Parallel computation support via Numba prange
- Lazy imports for optional dependencies (pfaffian, hafnian)
- Cache-friendly data layouts

----------------------------------------------------
File    : Algebra/Hilbert/hilbert_jit_states.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Version : 2.0.0
----------------------------------------------------
"""

from    __future__ import annotations
from    typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import  numba
import  numpy as np
from    numba import njit, prange

from QES.Algebra.Symmetries.base import _popcount64

# Type aliases
if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray

# Signature types for calculator functions
PfFunc              = Callable[[Array, int], Union[float, complex]]
HfFunc              = Callable[[Array], Union[float, complex]]
CallableCoefficient = Callable[[Array, Union[int, np.ndarray], int], Union[float, complex]]

__all__             = [
                    # Core bit extraction (re-exported from binary.py)
                    'extract_occupied', 'popcount_fast',
                    # Slater determinant
                    'calculate_slater_det', 'calculate_slater_det_batch',
                    # Bogoliubov
                    'bogolubov_decompose', 'pairing_matrix',
                    'calculate_bogoliubov_amp', 'calculate_bogoliubov_amp_exc',
                    # Bosonic
                    'calculate_permanent', 'calculate_bosonic_gaussian_amp',
                    # Many-body state construction
                    'ManyBodyStateType', 'many_body_states',
                    'many_body_state_full', 'many_body_state_mapping', 'many_body_state_closure',
                    'fill_many_body_state',
                    # Energy helpers
                    'nrg_particle_conserving', 'nrg_bdg',
                ]

# ============================================================================
# Configuration and constants
# ============================================================================

_TOLERANCE              = 1e-10
_ZERO_TOL               = 1e-15     # tolerance for Pfaffian pivot detection
_USE_EIGEN              = False     # Use np.linalg.det (stable) instead of eigvals product

# ============================================================================
# Import optimized helpers from binary.py (single source of truth)
# ============================================================================

try:
    from QES.general_python.common.binary import ctz64, popcount64
    _HAS_BINARY_MODULE  = True
except ImportError:
    raise ImportError("QES.general_python.common.binary module is required but not found. Ensure QES is properly installed.")

# ============================================================================
# Lazy imports for optional heavy dependencies
# ============================================================================

_pfaffian_module        = None
_hafnian_module         = None

def _get_pfaffian():
    """Lazy import of pfaffian module."""
    global _pfaffian_module
    if _pfaffian_module is None:
        try:
            from QES.general_python.algebra.utilities import pfaffian
            _pfaffian_module = pfaffian
        except ImportError as e:
            raise ImportError(
                "Pfaffian module required for Bogoliubov calculations. "
                "Install via: pip install qusolver[full]") from e
    return _pfaffian_module

def _get_hafnian():
    """Lazy import of hafnian module."""
    global _hafnian_module
    if _hafnian_module is None:
        try:
            from QES.general_python.algebra.utilities import hafnian
            _hafnian_module = hafnian
        except ImportError as e:
            raise ImportError(
                "Hafnian module required for bosonic Gaussian calculations. "
                "Install via: pip install qusolver[full]") from e
    return _hafnian_module

# ============================================================================
# JAX optional support
# ============================================================================

try:
    from QES.Algebra.Hilbert import hilbert_jit_states_jax as _jax_module
    JAX_AVAILABLE       = True
except Exception:
    _jax_module         = None
    JAX_AVAILABLE       = False

# ============================================================================
# Core bit operations - unified interface
# ============================================================================

# Re-export for backwards compatibility
popcount_fast           = popcount64

@njit(cache=True)
def extract_occupied(ns: int, basis: Union[int, np.ndarray], out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract indices of occupied sites from basis state (bitmask or array).
    
    Parameters
    ----------
    ns : int
        Number of sites (for bounds checking).
    basis : Union[int, np.ndarray]
        Basis state: integer bitmask or 0/1 array.
    out : Optional[np.ndarray]
        Pre-allocated output buffer. If None, creates new array.
        Must be large enough to hold all occupied indices.
    
    Returns
    -------
    np.ndarray
        Array of occupied site indices (int64).
    
    Notes
    -----
    Using pre-allocated `out` avoids memory allocation in tight loops.
    The returned array may be a slice of `out` with fewer elements.
    
    Examples
    --------
    >>> extract_occupied(8, 0b11010000)  # bits 4,6,7 set
    array([4, 6, 7], dtype=int64)
    
    >>> extract_occupied(4, np.array([1, 0, 1, 1]))
    array([0, 2, 3], dtype=int64)
    """

    if not hasattr(basis, 'ndim') or basis.ndim == 0:
        x = np.uint64(basis)
        count = popcount64(x)
        
        if out is None:
            occ = np.empty(count, dtype=np.int64)
        else:
            occ = out[:count]
        
        k = 0
        while x:
            pos     = ctz64(x)
            occ[k]  = pos
            k      += 1
            x      &= x - np.uint64(1) # Clear LSB
        return occ
    else:
        # Array input
        count = 0
        for i in range(basis.shape[0]):
            if basis[i] > 0:
                count += 1
        
        if out is None:
            occ = np.empty(count, dtype=np.int64)
        else:
            occ = out[:count]
        
        k = 0
        for i in range(basis.shape[0]):
            if basis[i] > 0:
                occ[k]  = i
                k      += 1
        return occ

# Backwards compatibility alias
_extract_occupied       = extract_occupied

# ============================================================================
# Slater Determinant Calculations
# ============================================================================

@njit(cache=True, fastmath=True)
def det_bareiss(M: np.ndarray) -> complex:
    """Integer-preserving determinant via Bareiss algorithm.
    Faster than LU for small matrices (N < 10)."""
    n       = M.shape[0]
    A       = M.copy()
    sign    = 1
    prev    = 1.0 + 0.0j
    
    for k in range(n - 1):
        # Partial pivoting - find max element in column k
        max_idx = k
        max_val = abs(A[k, k])
        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_idx = i
        
        # Swap rows explicitly (Numba doesn't support fancy indexing)
        if max_idx != k:
            for j in range(n):
                temp = A[k, j]
                A[k, j] = A[max_idx, j]
                A[max_idx, j] = temp
            sign *= -1
        
        if abs(A[k, k]) < 1e-14:
            return 0.0 + 0.0j
        
        # Bareiss update
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] = (A[k, k] * A[i, j] - A[i, k] * A[k, j]) / prev
        
        prev = A[k, k]
    
    return sign * A[n-1, n-1]

@njit(cache=True, fastmath=True)
def _slater_core(
    U           : np.ndarray,
    occ         : np.ndarray,
    sites       : np.ndarray,
    workspace   : Optional[np.ndarray] = None,
    use_eigen   : bool = _USE_EIGEN,
) -> complex:
    """
    Core Slater determinant computation with optional workspace.
    
    Parameters
    ----------
    U : np.ndarray
        Eigenvector matrix (ns, n_orb).
    occ : np.ndarray  
        Occupied orbital indices. Length = number of particles N.
    sites : np.ndarray
        Occupied site indices from basis state.
    workspace : Optional[np.ndarray]
        Pre-allocated (Ns, Ns) matrix for determinant.
        
    Returns
    -------
    complex
        Slater determinant value.
    """
    N = occ.shape[0]
    
    if workspace is not None:
        M = workspace[:N, :N]
    else:
        # Default to complex to avoid Numba type unification errors
        # when workspace is complex (usual case) but U is real.
        M = np.empty((N, N), dtype=np.complex128)
    
    # Build M_{jk} = U_{site_j, occ_k}
    for j in range(N):
        site_j = sites[j]
        for k in range(N):
            M[j, k] = U[site_j, occ[k]]
    
    if use_eigen:
        # Eigenvalue product method (less stable, for testing)
        eigvals = np.linalg.eigvals(M)
        det     = 1.0 + 0.0j
        for val in eigvals:
            det *= val
        return det
    
    return det_bareiss(M)
    # return np.linalg.det(M)

@njit(cache=True)
def calculate_slater_det(
    sp_eigvecs          : np.ndarray,
    occupied_orbitals   : np.ndarray,
    org_basis_state     : Union[int, np.ndarray],
    ns                  : int,
    workspace           : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Compute Slater determinant amplitude for N-fermion state.
    
    Computes the overlap between:
    - Product state in eigen-orbital basis: |psi_alpha⟩ = ∏_k bdag_{alpha_k} |0⟩
    - Fock state in site basis: |x⟩ = ∏_j adag_{x_j} |0⟩
    
    The amplitude is: ⟨x|psi_alpha⟩ = det(M) where M_{jk} = U_{x_j, alpha_k}
    
    Parameters
    ----------
    sp_eigvecs : np.ndarray
        Single-particle eigenvectors U (ns, n_orb). Columns are orbitals.
    occupied_orbitals : np.ndarray
        Indices of occupied orbitals in eigen-basis.
    org_basis_state : Union[int, np.ndarray]
        Fock state as integer bitmask or 0/1 array.
    ns : int
        Number of sites.
    workspace : Optional[np.ndarray]
        Pre-allocated (N, N) workspace matrix for determinant.
        Avoids allocation in tight loops.
    
    Returns
    -------
    complex
        Determinant value. Returns 0 if particle numbers don't match.
    
    Examples
    --------
    >>> U = np.eye(4)  # trivial transformation
    >>> occ = np.array([0, 1])  # fill first two orbitals
    >>> psi = calculate_slater_det(U, occ, 0b0011, 4)  # should be ±1
    """
    
    # Number of particles in the state
    N = occupied_orbitals.shape[0]
    
    # Handle scalar vs array input
    if not hasattr(org_basis_state, 'ndim') or org_basis_state.ndim == 0:
        # Integer bitmask
        mask    = np.uint64(org_basis_state)
        n_fock  = popcount64(mask)
        
        if n_fock != N:
            return 0.0 + 0.0j
        if N == 0:
            return 1.0 + 0.0j
        
        sites = extract_occupied(ns, int(mask))
        return _slater_core(sp_eigvecs, occupied_orbitals, sites, workspace)
    else:
        # Array basis
        n_fock  = 0
        for i in range(org_basis_state.shape[0]):
            if org_basis_state[i] > 0:
                n_fock += 1
        
        if n_fock != N:
            return 0.0 + 0.0j
        if N == 0:
            return 1.0 + 0.0j
        
        sites = extract_occupied(ns, org_basis_state)
        return _slater_core(sp_eigvecs, occupied_orbitals, sites, workspace)

@njit(cache=True, parallel=True)
def calculate_slater_det_batch(
    sp_eigvecs          : np.ndarray,
    occupied_orbitals   : np.ndarray,
    basis_states        : np.ndarray,
    ns                  : int,
    result              : Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Batch compute Slater determinants for multiple basis states.
    
    Optimized for parallel execution with Numba prange.
    
    Parameters
    ----------
    sp_eigvecs : np.ndarray
        Eigenvector matrix (ns, n_orb).
    occupied_orbitals : np.ndarray
        Occupied orbital indices.
    basis_states : np.ndarray
        Array of integer basis states.
    ns : int
        Number of sites.
    result : Optional[np.ndarray]
        Pre-allocated output array. If None, creates new array.
    
    Returns
    -------
    np.ndarray
        Array of Slater determinant values.
    """
    n_states = basis_states.shape[0]
    
    if result is None:
        result = np.empty(n_states, dtype=np.complex128)
    
    for i in prange(n_states):
        result[i] = calculate_slater_det(sp_eigvecs, occupied_orbitals, basis_states[i], ns, None)
    
    return result

# ============================================================================
# Standalone JIT-compiled Pfaffian (Parlett-Reid) and Hafnian (Gray code)
# ============================================================================
# These are module-level @njit functions extracted from the Pfaffian/Hafnian
# utility classes so that the BdG fillers can be fully JIT-compiled with prange.

@njit(cache=True)
def _pfaffian_parlett_reid(A_in: np.ndarray, N: int) -> complex:
    r"""
    Pfaffian via Parlett-Reid algorithm (standalone Numba JIT).
    
    Computes Pf(A) for an N x N skew-symmetric matrix using
    Gaussian elimination with partial pivoting.  O(N^3).
    
    Parameters
    ----------
    A_in : np.ndarray
        Skew-symmetric matrix (N, N). A copy is made internally.
    N : int
        Matrix dimension.
    
    Returns
    -------
    complex
        The Pfaffian value.
    """
    if N == 0:
        return 1.0 + 0.0j
    if N % 2 != 0:
        return 0.0 + 0.0j
    
    A            = A_in.astype(np.complex128).copy()
    pfaffian_val = 1.0 + 0.0j
    
    for k in range(0, N - 1, 2):
        # Pivoting: find largest |A[i, k]| for i in [k+1, N)
        max_val = 0.0
        kp      = k + 1
        for i in range(k + 1, N):
            av = abs(A[i, k])
            if av > max_val:
                max_val = av
                kp      = i
        
        if kp != k + 1:
            # Swap rows k+1 <-> kp
            for j in range(N):
                tmp            = A[k + 1, j]
                A[k + 1, j]    = A[kp, j]
                A[kp, j]       = tmp
            # Swap cols k+1 <-> kp
            for i in range(N):
                tmp            = A[i, k + 1]
                A[i, k + 1]    = A[i, kp]
                A[i, kp]       = tmp
            pfaffian_val *= -1.0
        
        pivot_val = A[k + 1, k]
        if abs(pivot_val) < _ZERO_TOL:
            return 0.0 + 0.0j
        
        pfaffian_val *= A[k, k + 1]
        
        if k + 2 < N:
            inv_pivot = 1.0 / A[k, k + 1]
            for i in range(k + 2, N):
                tau_i = A[k, i] * inv_pivot
                for j in range(k + 2, N):
                    A[i, j] += tau_i * A[k + 1, j] - A[i, k + 1] * (A[k, j] * inv_pivot)
    
    return pfaffian_val


@njit(cache=True)
def _hafnian_prod_cached(A: np.ndarray, mask: int) -> complex:
    r"""Recursive helper: sum over perfect matchings encoded by bitmask."""
    if mask == 0:
        return 1.0 + 0.0j
    # Find lowest set bit
    lsb  = mask & (-mask)
    i    = 0
    tmp  = lsb
    while tmp > 1:
        tmp >>= 1
        i   += 1
    mask ^= lsb
    
    acc       = 0.0 + 0.0j
    rest_mask = mask
    while rest_mask:
        lsb2      = rest_mask & (-rest_mask)
        j         = 0
        tmp2      = lsb2
        while tmp2 > 1:
            tmp2 >>= 1
            j    += 1
        rest_mask ^= lsb2
        acc       += A[i, j] * _hafnian_prod_cached(A, mask ^ lsb2)
    return acc


@njit(cache=True)
def _hafnian_gray(A: np.ndarray) -> complex:
    r"""
    Hafnian via Gray-code enumeration (standalone Numba JIT).
    
    Complexity O(2^n · n) where n is the matrix dimension.
    Practical for n ≤ 20.
    
    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix (n, n).
    
    Returns
    -------
    complex
        The Hafnian value.
    """
    n = A.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    if n & 1:
        return 0.0 + 0.0j
    
    # For small matrices, use direct recursive approach
    # Build full mask with all n bits set
    full_mask = (1 << n) - 1
    return _hafnian_prod_cached(A, full_mask)


# ============================================================================
# Bogoliubov-de Gennes Calculations
# ============================================================================

@njit(cache=True)
def bogolubov_decompose(
    eig_val : np.ndarray,
    eig_vec : np.ndarray,
    tol     : float = _TOLERANCE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Decompose BdG eigenvalues/vectors into (u, v) components.
    
    Given the eigenvalues and eigenvectors of a Bogoliubov-de Gennes (BdG) 
    Hamiltonian, selects positive eigenvalues and extracts (u, v) components.
    Each column is normalized: udagu + vdagv = 1.
    
    Parameters
    ----------
    eig_val : np.ndarray
        Array of eigenvalues (2N,).
    eig_vec : np.ndarray  
        Array of eigenvectors (2N, 2N).
    tol : float
        Tolerance for selecting positive eigenvalues.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - U: (N, N) matrix of u components
        - V: (N, N) matrix of v components  
        - eig_val_pos: (N,) positive eigenvalues
    
    Raises
    ------
    ValueError
        If number of positive eigenvalues doesn't match N.
    """
    if eig_val is None or eig_vec is None:
        raise ValueError("bogolubov_decompose: eig_val and eig_vec cannot be None")
    
    keep = eig_val > tol
    n_pos = 0
    for i in range(keep.shape[0]):
        if keep[i]:
            n_pos += 1
    
    if n_pos != eig_val.shape[0] // 2:
        raise ValueError("Degeneracy or sign problem; adjust tolerance")
    
    eig_val_pos = eig_val[keep]
    eig_vec_pos = eig_vec[:, keep]
    N = n_pos
    
    # Split into (u, v) components
    U = eig_vec_pos[:N, :].copy()
    V = eig_vec_pos[N:, :].copy()
    
    # Column-wise normalization
    for k in range(N):
        norm_u = 0.0
        norm_v = 0.0
        for i in range(N):
            norm_u += np.abs(U[i, k])**2
            norm_v += np.abs(V[i, k])**2
        s = np.sqrt(norm_u + norm_v)
        if s > 1e-15:
            for i in range(N):
                U[i, k] /= s
                V[i, k] /= s
    
    return U, V, eig_val_pos

@njit(cache=True)
def pairing_matrix(u_mat: np.ndarray, v_mat: np.ndarray) -> np.ndarray:
    r"""
    Compute pairing matrix F = V·U^{-1} via linear solve.
    
    Avoids explicit matrix inversion for numerical stability.
    Valid for both fermionic (antisymmetric F) and bosonic (symmetric G).
    
    Parameters
    ----------
    u_mat : np.ndarray
        U matrix from Bogoliubov decomposition.
    v_mat : np.ndarray
        V matrix from Bogoliubov decomposition.
    
    Returns
    -------
    np.ndarray
        Pairing matrix F (or G for bosons).
    """
    return np.linalg.solve(u_mat.T, v_mat.T).T

@njit(cache=True)
def calculate_bogoliubov_amp(
    F           : np.ndarray,
    basis       : Union[int, np.ndarray],
    ns          : int,
    enforce     : bool = True,
    workspace   : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Calculate Bogoliubov vacuum amplitude using Pfaffian (Numba JIT).
    
    For a Bogoliubov vacuum |psi⟩, computes ⟨x|psi⟩ = Pf[F_{ij}]
    where i,j are occupied sites in basis state x.
    
    Parameters
    ----------
    F : np.ndarray
        Pairing matrix (ns, ns).
    basis : Union[int, np.ndarray]
        Basis state (bitmask or array).
    ns : int
        Number of sites.
    enforce : bool
        If True, enforces skew-symmetry by averaging.
    workspace : Optional[np.ndarray]
        Pre-allocated (max_occ, max_occ) complex128 matrix for submatrix.
    
    Returns
    -------
    complex
        Pfaffian value. Returns 0 if odd number of particles.
    """
    occ = extract_occupied(ns, basis)
    m   = occ.size
    
    if m == 0:
        return 1.0 + 0.0j
    if m & 1:  # Odd number -> 0
        return 0.0 + 0.0j
    
    # Build submatrix
    if workspace is not None:
        sub = workspace[:m, :m]
    else:
        sub = np.empty((m, m), dtype=np.complex128)
    
    for p in range(m):
        ip = occ[p]
        for q in range(m):
            sub[p, q] = F[ip, occ[q]]
    
    if enforce:
        for p in range(m):
            for q in range(p + 1, m):
                avg         = 0.5 * (sub[p, q] - sub[q, p])
                sub[p, q]   =  avg
                sub[q, p]   = -avg
            sub[p, p] = 0.0
    
    return _pfaffian_parlett_reid(sub, m)


@njit(cache=True)
def calculate_bogoliubov_amp_exc(
    F           : np.ndarray,
    U           : np.ndarray,
    qp_inds     : np.ndarray,
    basis       : Union[int, np.ndarray],
    ns          : int,
    workspace   : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Calculate Bogoliubov amplitude for excited quasiparticle state (Numba JIT).
    
    Parameters
    ----------
    F : np.ndarray
        Pairing matrix (ns, ns).
    U : np.ndarray
        Bogoliubov transformation matrix (ns, N_qp).
    qp_inds : np.ndarray
        Indices of excited quasiparticles.
    basis : Union[int, np.ndarray]
        Basis state.
    ns : int
        Number of sites.
    workspace : Optional[np.ndarray]
        Pre-allocated workspace for the extended matrix (dim, dim) where
        dim = n_occupied + n_qp.
    
    Returns
    -------
    complex
        Pfaffian of extended matrix.
    """
    occ = extract_occupied(ns, basis)
    n   = occ.size
    k   = qp_inds.size
    
    if (n + k) & 1:
        return 0.0 + 0.0j
    if n == k == 0:
        return 1.0 + 0.0j
    
    dim = n + k
    
    if workspace is not None:
        M = workspace[:dim, :dim]
    else:
        M = np.empty((dim, dim), dtype=np.complex128)
    
    # F block
    for p in range(n):
        ip = occ[p]
        for q in range(n):
            M[p, q] = F[ip, occ[q]]
    
    # Cross blocks
    for p in range(n):
        ip = occ[p]
        for j in range(k):
            m_idx       = qp_inds[j]
            M[p, n + j] =  U[ip, m_idx]
            M[n + j, p] = -U[ip, m_idx]
    
    # Lower-right block = 0
    for i in range(k):
        for j in range(k):
            M[n + i, n + j] = 0.0
    
    return _pfaffian_parlett_reid(M, dim)

# ============================================================================
# Bosonic Calculations
# ============================================================================

@njit(cache=True, fastmath=True)
def _permanent_ryser(M: np.ndarray) -> complex:
    """
    Compute permanent using Ryser's formula (O(2^n * n)).
    
    Parameters
    ----------
    M : np.ndarray
        Square matrix.
    
    Returns
    -------
    complex
        Permanent value.
    """
    n = M.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    
    total = 0.0 + 0.0j
    
    for k in range(1, 1 << n):
        prod = 1.0 + 0.0j
        popcount = 0
        
        for i in range(n):
            row_sum = 0.0 + 0.0j
            mask = k
            col = 0
            while mask > 0:
                if mask & 1:
                    row_sum += M[i, col]
                mask >>= 1
                col += 1
            prod *= row_sum
        
        # Popcount
        temp = k
        while temp > 0:
            temp &= temp - 1
            popcount += 1
        
        sign = -1.0 if (n - popcount) % 2 else 1.0
        total += sign * prod
    
    return total

@njit(cache=True)
def calculate_permanent(
    sp_eigvecs          : np.ndarray,
    occupied_orbitals   : np.ndarray,
    org_basis_state     : Union[int, np.ndarray],
    ns                  : int,
    workspace           : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Compute permanent for bosonic amplitude (analogous to Slater for fermions).
    
    Uses Ryser's formula: O(2^n · n).
    
    Parameters
    ----------
    sp_eigvecs : np.ndarray
        Eigenvector matrix (ns, n_orb).
    occupied_orbitals : np.ndarray
        Occupied orbital indices.
    org_basis_state : Union[int, np.ndarray]
        Fock state.
    ns : int
        Number of sites.
    workspace : Optional[np.ndarray]
        Pre-allocated (N, N) matrix.
    
    Returns
    -------
    complex
        Permanent value.
    """
    N = occupied_orbitals.shape[0]
    
    # Extract occupied modes from basis state
    # Numba-compatible type check: use ndim check instead of isinstance with np.integer
    if not hasattr(org_basis_state, 'ndim') or org_basis_state.ndim == 0:
        mask = np.uint64(org_basis_state)
        n_fock = popcount64(mask)
        if n_fock != N:
            return 0.0 + 0.0j
        if N == 0:
            return 1.0 + 0.0j
        occupied_modes = extract_occupied(ns, int(mask))
    else:
        n_fock = 0
        for i in range(org_basis_state.shape[0]):
            if org_basis_state[i] > 0:
                n_fock += 1
        if n_fock != N:
            return 0.0 + 0.0j
        if N == 0:
            return 1.0 + 0.0j
        occupied_modes = extract_occupied(ns, org_basis_state)
    
    # Build matrix
    if workspace is not None:
        M = workspace[:N, :N]
    else:
        M = np.empty((N, N), dtype=sp_eigvecs.dtype)
    
    for j in range(N):
        site_j = occupied_modes[j]
        for k in range(N):
            M[j, k] = sp_eigvecs[site_j, occupied_orbitals[k]]
    
    return _permanent_ryser(M)

@njit(cache=True)
def calculate_bosonic_gaussian_amp(
    G           : np.ndarray,
    basis       : Union[int, np.ndarray],
    ns          : int,
    workspace   : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Calculate bosonic Gaussian state amplitude using Hafnian (Numba JIT).
    
    For a Gaussian state |psi⟩, computes ⟨x|psi⟩ = Hf[G_{ij}]
    where i,j are occupied sites.
    
    Parameters
    ----------
    G : np.ndarray
        Symmetric matrix (ns, ns).
    basis : Union[int, np.ndarray]
        Basis state.
    ns : int
        Number of modes.
    workspace : Optional[np.ndarray]
        Pre-allocated complex128 workspace (ns, ns).
    
    Returns
    -------
    complex
        Hafnian value.
    """
    occ = extract_occupied(ns, basis)
    m   = occ.size
    
    if m == 0:
        return 1.0 + 0.0j
    if m & 1:
        return 0.0 + 0.0j
    
    if workspace is not None:
        sub = workspace[:m, :m]
    else:
        sub = np.empty((m, m), dtype=np.complex128)
    
    for p in range(m):
        ip = occ[p]
        for q in range(m):
            sub[p, q] = G[ip, occ[q]]
    
    return _hafnian_gray(sub)

# ============================================================================
# Many-Body State Construction
# ============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _fill_many_body_state_slater(
    sp_eigvecs          : np.ndarray,   # (orbital_size, n_orb) - single-particle eigenvectors
    orbital_configs     : np.ndarray,   # (chunk_size, n_occupied) - pre-extracted occupied orbital indices for each target state
    ns                  : int,
    nfilling            : int,
    result              : np.ndarray,   # (chunk_size, hilbert_size) - pre-allocated output
):
    ''' Fill many-body state amplitudes for Slater determinants in parallel. '''

    chunk_size  = orbital_configs.shape[0]
    nh          = result.shape[1]       # Hilbert space size (e.g., 2^ns)

    for i in numba.prange(chunk_size):  # Parallel loop over states
        # Allocate workspace per thread (always complex for det_bareiss)
        occupied_orbitals   = orbital_configs[i]
        workspace           = np.empty((nfilling, nfilling), dtype=np.complex128)
        
        for st in range(nh):
            if _popcount64(np.uint64(st)) == nfilling:
                result[i, st] = calculate_slater_det(sp_eigvecs, occupied_orbitals, st, ns, workspace)
            else:
                result[i, st] = 0.0
        
@njit(parallel=True, cache=True, fastmath=True)
def _fill_many_body_state_permanent(
    sp_eigvecs          : np.ndarray,   # (orbital_size, n_orb) - single-particle eigenvectors
    orbital_configs     : np.ndarray,   # (chunk_size, n_occupied) - pre-extracted occupied orbital indices for each target state
    ns                  : int,
    nfilling            : int,
    result              : np.ndarray,   # (chunk_size, hilbert_size) - pre-allocated output
):
    ''' Fill many-body state amplitudes for bosonic permanents in parallel. '''

    chunk_size  = orbital_configs.shape[0]
    nh          = result.shape[1]       # Hilbert space size (e.g., 2^ns)

    for i in numba.prange(chunk_size):  # Parallel loop over states
        # Allocate workspace per thread (always complex for permanent)
        occupied_orbitals   = orbital_configs[i]
        workspace           = np.empty((nfilling, nfilling), dtype=np.complex128)
        
        for st in range(nh):
            if _popcount64(np.uint64(st)) == nfilling:
                result[i, st] = calculate_permanent(sp_eigvecs, occupied_orbitals, st, ns, workspace)
            else:
                result[i, st] = 0.0

def _fill_many_body_state_bogoliubov(
    F                   : np.ndarray,                   # (ns, ns) - antisymmetric pairing matrix
    ns                  : int,
    result              : np.ndarray,                   # (chunk_size, hilbert_size) - pre-allocated output
    qp_configs          : Optional[np.ndarray] = None,  # (chunk_size, n_qp) or None for vacuum
    u_bdg_matrix        : Optional[np.ndarray] = None,  # (ns, n_modes) for excited states
):
    r'''
    Fill many-body amplitudes for BdG fermionic states (Pfaffian-based).
    
    Physics
    -------
    For a general BdG Hamiltonian with pairing terms:
    
    .. math::
        H = \sum_{ij} h_{ij} c^\dagger_i c_j 
            + \frac{1}{2}\sum_{ij} (\Delta_{ij} c^\dagger_i c^\dagger_j + \mathrm{h.c.})
    
    The ground state (Bogoliubov vacuum |\mathrm{BCS}\rangle) satisfies 
    \gamma_k |\mathrm{BCS}\rangle = 0 for all quasiparticle operators \gamma_k.
    It can be written as:
    
    .. math::
        |\mathrm{BCS}\rangle = \mathcal{N} \exp\!\Big(\tfrac{1}{2}\sum_{ij} F_{ij}\, c^\dagger_i c^\dagger_j\Big)|0\rangle
    
    where F = V U^{-1} is the antisymmetric pairing matrix.
    
    **Vacuum amplitude** (no quasiparticle excitations):
    
    .. math::
        \langle x | \mathrm{BCS}\rangle = \mathrm{Pf}(F_{\mathrm{occ}})
    
    where F_{\mathrm{occ}} is F restricted to occupied sites in Fock state |x\rangle.
    Only even-particle-number Fock states have non-zero overlap (Pfaffian of 
    odd-dimensional matrix is zero).
    
    **Excited quasiparticle state** \gamma^\dagger_{m_1}\cdots\gamma^\dagger_{m_k}|\mathrm{BCS}\rangle:
    
    .. math::
        \langle x | \gamma^\dagger_{m_1}\cdots\gamma^\dagger_{m_k} |\mathrm{BCS}\rangle
        = \mathrm{Pf}(M_{\mathrm{ext}})
    
    where M_{\mathrm{ext}} is the (n+k)\times(n+k) matrix with blocks:
         - Upper-left (n x n):  F restricted to occupied sites
         - Off-diagonal:      U columns corresponding to excited QPs
         - Lower-right (k x k): zeros
    
    Non-zero only when n + k is even (fermion parity constraint).
    
    Parameters
    ----------
    F : np.ndarray
        Antisymmetric pairing matrix (ns, ns). Obtained from F = V @ U^{-1}.
    ns : int
        Number of single-particle modes/sites.
    result : np.ndarray
        Pre-allocated output of shape (chunk_size, 2^ns).
    qp_configs : Optional[np.ndarray]
        Quasiparticle excitation indices. Shape (chunk_size, n_qp).
        If None or n_qp == 0, computes the BCS vacuum for all configs
        (identical rows — computed once and broadcast).
    u_bdg_matrix : Optional[np.ndarray]
        Bogoliubov U matrix (ns, n_modes). Required when qp_configs 
        contains non-empty excitation patterns.
    
    Notes
    -----
    - Cannot use Numba prange because the Pfaffian routine is pure Python.
    - Vacuum is computed once and broadcast when all configs are identical.
    - Workspace is pre-allocated per-config to reduce allocation overhead.
    '''
    chunk_size  = result.shape[0]
    nh          = result.shape[1]
    
    is_vacuum   = (qp_configs is None) or (qp_configs.ndim == 2 and qp_configs.shape[1] == 0)
    
    if is_vacuum:
        # BdG vacuum: compute ONCE, then broadcast
        workspace = np.empty((ns, ns), dtype=F.dtype)
        for st in range(nh):
            n_occ = popcount64(np.uint64(st))
            if n_occ & 1:                       # odd particle number → 0
                result[0, st] = 0.0
            else:
                result[0, st] = calculate_bogoliubov_amp(F, st, ns, True, workspace)
        # Broadcast to all configs (they are all the same vacuum)
        for i in range(1, chunk_size):
            result[i, :] = result[0, :]
    else:
        # Excited quasiparticle states
        if u_bdg_matrix is None:
            raise ValueError(
                "u_bdg_matrix is required for excited BdG states. "
                "Pass the U matrix from bogolubov_decompose()."
            )
        for i in range(chunk_size):
            qp_inds         = qp_configs[i]
            n_qp            = qp_inds.size
            ws_dim          = ns + n_qp
            workspace       = np.empty((ws_dim, ws_dim), dtype=F.dtype)
            for st in range(nh):
                n_occ = popcount64(np.uint64(st))
                if (n_occ + n_qp) & 1: # parity mismatch → 0
                    result[i, st] = 0.0
                else:
                    # calculate Pfaffian of extended matrix for this config and basis state
                    result[i, st] = calculate_bogoliubov_amp_exc(
                        F, u_bdg_matrix, qp_inds, st, ns, workspace
                    )

def _fill_many_body_state_gaussian(
    G               : np.ndarray,   # (ns, ns) - symmetric pairing matrix
    ns              : int,
    result          : np.ndarray,   # (chunk_size, hilbert_size) - pre-allocated output
):
    r'''
    Fill many-body amplitudes for bosonic Gaussian states (Hafnian-based).
    
    Physics
    -------
    For a bosonic quadratic Hamiltonian with pairing (squeezing) terms:
    
    .. math::
        H = \sum_{ij} h_{ij} a^\dagger_i a_j 
            + \frac{1}{2}\sum_{ij} (G_{ij} a^\dagger_i a^\dagger_j + \mathrm{h.c.})
    
    The Gaussian ground state has occupation-basis amplitudes given by the Hafnian:
    
    .. math::
        \langle x | \mathrm{Gauss}\rangle = \mathrm{Hf}(G_{\mathrm{occ}})
    
    where G_{\mathrm{occ}} is G restricted to occupied modes in Fock state |x\rangle.
    The Hafnian is the bosonic analogue of the Pfaffian:
    - Pfaffian sums over perfect matchings with signs (fermions)
    - Hafnian sums over perfect matchings without signs (bosons)
    
    Only even-particle-number Fock states contribute (Hafnian of 
    odd-dimensional matrix is zero).
    
    Parameters
    ----------
    G : np.ndarray
        Symmetric pairing matrix (ns, ns).
    ns : int
        Number of bosonic modes.
    result : np.ndarray
        Pre-allocated output of shape (chunk_size, 2^ns).
    
    Notes
    -----
    - Bosonic Gaussian vacuum is unique for a given G; all chunk rows
      receive the same state.
    - Cannot use Numba prange (Hafnian routine is pure Python).
    '''
    chunk_size  = result.shape[0]
    nh          = result.shape[1]
    
    # Gaussian vacuum: compute once, then broadcast
    workspace = np.empty((ns, ns), dtype=G.dtype)
    for st in range(nh):
        n_occ = popcount64(np.uint64(st))
        if n_occ & 1:
            result[0, st] = 0.0
        else:
            result[0, st] = calculate_bosonic_gaussian_amp(G, st, ns, workspace)
    for i in range(1, chunk_size):
        result[i, :] = result[0, :]

# ────────────────────────────────────────────────────────────────────────────
# Many-Body State Type Enum & Unified Dispatcher
# ────────────────────────────────────────────────────────────────────────────

class ManyBodyStateType:
    r"""Identifies the physics of the many-body state.
    
    ============  ================  ===================  ================
    Type          Particles         Conserves N?         Amplitude
    ============  ================  ===================  ================
    SLATER        fermions          yes                  det(M)
    PERMANENT     bosons            yes                  perm(M)
    BOGOLIUBOV    fermions          no  (BdG / BCS)      Pf(F_occ)
    GAUSSIAN      bosons            no  (squeezing)      Hf(G_occ)
    ============  ================  ===================  ================
    """
    SLATER          = 0
    PERMANENT       = 1
    BOGOLIUBOV      = 2
    GAUSSIAN        = 3


def many_body_states(
    ns                      : int,
    *,
    state_type              : int = ManyBodyStateType.SLATER,
    orbital_configs         : Optional[np.ndarray] = None,
    result                  : Optional[np.ndarray] = None,
    dtype                   : np.dtype = np.complex128,
    # Particle-conserving (Slater / Permanent)
    single_particle_eigvecs : Optional[np.ndarray] = None,
    # BdG fermionic (Bogoliubov)
    pairing_matrix_F        : Optional[np.ndarray] = None,
    u_bdg_matrix            : Optional[np.ndarray] = None,
    # BdG bosonic (Gaussian)
    gaussian_matrix_G       : Optional[np.ndarray] = None,
    # Hilbert-space mapping
    mapping_array           : Optional[np.ndarray] = None,
) -> np.ndarray:
    r'''Construct many-body state vectors for multiple configurations in parallel.
    
    This function is the **unified dispatcher** for building occupation-basis
    representations of many-body states arising from quadratic (non-
    interacting) Hamiltonians. It covers all four physical regimes:
    
    ==========================================================================
    Regime                       What is computed
    ==========================================================================
    **SLATER** (fermion, N-cons)  Slater determinant
                                  :math:`\langle x|\Psi\rangle = \det M`
                                  where :math:`M_{jk} = U_{x_j, \alpha_k}`.
                                  Particle number is conserved; only Fock states
                                  with :math:`|x|=N_{\rm fill}` contribute.
    
    **PERMANENT** (boson, N-cons) Bosonic permanent
                                  :math:`\langle x|\Psi\rangle = \mathrm{perm}\,M`
                                  with the same matrix :math:`M`. Permanent
                                  replaces determinant (no anti-symmetry).
    
    **BOGOLIUBOV** (fermion, BdG) BCS / Bogoliubov vacuum (or excitations)
                                  :math:`\langle x|\mathrm{BCS}\rangle = \mathrm{Pf}(F_{\rm occ})`
                                  where :math:`F = V U^{-1}` is the antisymmetric
                                  pairing matrix. Only even-particle Fock
                                  states contribute. Excited QP states use an
                                  extended Pfaffian.
    
    **GAUSSIAN** (boson, BdG)     Bosonic Gaussian vacuum
                                  :math:`\langle x|\mathrm{Gauss}\rangle = \mathrm{Hf}(G_{\rm occ})`
                                  where :math:`G` is the symmetric pairing matrix
                                  and Hf is the Hafnian.
    ==========================================================================
    
    Parameters
    ----------
    ns : int
        Number of single-particle sites/modes.
    state_type : int
        One of :class:`ManyBodyStateType` constants selecting the physics.
    orbital_configs : Optional[np.ndarray]
        **Particle-conserving** (SLATER / PERMANENT):
            Required.  Shape ``(chunk_size, n_occupied)``.
            Each row lists the occupied single-particle orbital indices
            that define the many-body state.
        **BdG** (BOGOLIUBOV / GAUSSIAN):
            Optional.  Shape ``(chunk_size, n_qp)``.
            Each row lists quasiparticle indices to excite on top of the
            vacuum.  If ``None`` or ``n_qp == 0``, the BdG vacuum is
            computed (all rows identical — computed once, then broadcast).
    result : Optional[np.ndarray]
        Pre-allocated output array of shape ``(chunk_size, nh)``.
        If ``None``, a new array is allocated.
    dtype : np.dtype
        Output dtype (default ``complex128``).
    single_particle_eigvecs : Optional[np.ndarray]
        Eigenvector matrix :math:`U` of shape ``(ns, n_orb)``.
        Required for SLATER and PERMANENT.
    pairing_matrix_F : Optional[np.ndarray]
        Antisymmetric pairing matrix :math:`F` of shape ``(ns, ns)``.
        Required for BOGOLIUBOV.
    u_bdg_matrix : Optional[np.ndarray]
        Bogoliubov :math:`U` matrix of shape ``(ns, n_modes)``.
        Required for excited BOGOLIUBOV states (when ``qp_configs`` is
        non-empty).
    gaussian_matrix_G : Optional[np.ndarray]
        Symmetric pairing matrix :math:`G` of shape ``(ns, ns)``.
        Required for GAUSSIAN.
    mapping_array : Optional[np.ndarray]
        Custom Hilbert-space index mapping.  If provided, the output
        size is ``len(mapping_array)`` instead of ``2^ns``.
    
    Returns
    -------
    np.ndarray
        State vectors, shape ``(chunk_size, nh)``.
    
    Raises
    ------
    ValueError
        If required matrices are not provided for the chosen state_type.
    NotImplementedError
        If an unsupported state_type is requested.
    
    Performance
    -----------
    - SLATER / PERMANENT: fully Numba-parallel over configs × Fock states.
    - BOGOLIUBOV / GAUSSIAN: Python-level loop (Pfaffian / Hafnian are not
      yet JIT-compiled).  Vacuum is computed once and broadcast, so the cost
      is O(2^ns) regardless of ``chunk_size``.
    
    Examples
    --------
    >>> # Slater determinant for 4-site system, 2 particles
    >>> U = np.linalg.eigh(H_sp)[1]
    >>> configs = np.array([[0, 1], [0, 2], [1, 2]])
    >>> psi = many_body_states(4, orbital_configs=configs,
    ...                        single_particle_eigvecs=U,
    ...                        state_type=ManyBodyStateType.SLATER)
    >>> psi.shape
    (3, 16)
    
    >>> # BCS vacuum from pairing matrix
    >>> F = pairing_matrix(U_bdg, V_bdg)
    >>> psi_bcs = many_body_states(4, pairing_matrix_F=F,
    ...                            state_type=ManyBodyStateType.BOGOLIUBOV)
    >>> psi_bcs.shape
    (1, 16)
    '''
    
    # ── Determine chunk_size and Hilbert-space dimension ──────────────
    is_bdg = (state_type in (ManyBodyStateType.BOGOLIUBOV, ManyBodyStateType.GAUSSIAN))
    
    if orbital_configs is not None:
        if orbital_configs.ndim == 1:
            orbital_configs = orbital_configs.reshape(1, -1)
        chunk_size = orbital_configs.shape[0]
    elif is_bdg:
        # BdG vacuum with no explicit configs → single vacuum state
        chunk_size = 1
    else:
        raise ValueError(
            "orbital_configs is required for particle-conserving states "
            "(SLATER / PERMANENT)."
        )
    
    nh = mapping_array.shape[0] if mapping_array is not None else (1 << ns)
    
    # ── Allocate or validate output ───────────────────────────────────
    if result is not None:
        out = result if result.dtype == dtype else result.astype(dtype, copy=False)
    else:
        out = np.zeros((chunk_size, nh), dtype=dtype)
    
    # ── Dispatch to the appropriate filler ────────────────────────────
    if state_type == ManyBodyStateType.SLATER:
        if single_particle_eigvecs is None:
            raise ValueError("single_particle_eigvecs required for SLATER.")
        nfilling = orbital_configs.shape[1]
        _fill_many_body_state_slater(
            single_particle_eigvecs, orbital_configs, ns, nfilling, out
        )
    
    elif state_type == ManyBodyStateType.PERMANENT:
        if single_particle_eigvecs is None:
            raise ValueError("single_particle_eigvecs required for PERMANENT.")
        nfilling = orbital_configs.shape[1]
        _fill_many_body_state_permanent(
            single_particle_eigvecs, orbital_configs, ns, nfilling, out
        )
    
    elif state_type == ManyBodyStateType.BOGOLIUBOV:
        if pairing_matrix_F is None:
            raise ValueError("pairing_matrix_F required for BOGOLIUBOV.")
        _fill_many_body_state_bogoliubov(
            F               = pairing_matrix_F,
            ns              = ns,
            result          = out,
            qp_configs      = orbital_configs if (orbital_configs is not None and orbital_configs.shape[1] > 0) else None,
            u_bdg_matrix    = u_bdg_matrix,
        )
    
    elif state_type == ManyBodyStateType.GAUSSIAN:
        if gaussian_matrix_G is None:
            raise ValueError("gaussian_matrix_G required for GAUSSIAN.")
        _fill_many_body_state_gaussian(
            G       = gaussian_matrix_G,
            ns      = ns,
            result  = out,
        )
    
    else:
        raise ValueError(f"Unknown state_type: {state_type}")
    
    return out

# ============================================================================
# Energy Helpers
# ============================================================================

@njit(cache=True, inline="always")
def nrg_particle_conserving(eigvals: np.ndarray, occ: np.ndarray) -> float:
    """
    Compute energy for particle-conserving system.
    
    Parameters
    ----------
    eigvals : np.ndarray
        Single-particle eigenvalues.
    occ : np.ndarray
        Occupied orbital indices.
    
    Returns
    -------
    float
        Total energy = sum of occupied eigenvalues.
    """
    tot = 0.0
    for k in range(occ.shape[0]):
        tot += eigvals[occ[k]]
    return tot

@njit(cache=True, inline="always")
def nrg_bdg(eigvals: np.ndarray, Ns: int, occ: np.ndarray) -> float:
    """
    Compute energy for BdG system.
    
    Parameters
    ----------
    eigvals : np.ndarray
        BdG eigenvalues (sorted).
    Ns : int
        Number of single-particle states.
    occ : np.ndarray
        Occupied state indices.
    
    Returns
    -------
    float
        Total energy contribution.
    """
    tot = 0.0
    mid = Ns - 1
    for i in range(occ.shape[0]):
        tot += eigvals[mid + occ[i] + 1] - eigvals[mid - occ[i]]
    return tot

# ============================================================================
# Workspace/Buffer Preallocation Helpers
# ============================================================================

class HilbertStateWorkspace:
    """
    Pre-allocated workspace for repeated state calculations.
    
    Reduces memory allocation overhead in tight loops.
    
    Parameters
    ----------
    ns : int
        Number of sites.
    max_particles : Optional[int]
        Maximum particle number. If None, uses ns.
    dtype : np.dtype
        Matrix dtype.
    
    Examples
    --------
    >>> ws = HilbertStateWorkspace(ns=20, max_particles=10)
    >>> for occ in occupations:
    ...     psi = calculate_slater_det(U, occ, state, ns, workspace=ws.det_matrix)
    """
    
    def __init__(
        self,
        ns              : int,
        max_particles   : Optional[int] = None,
        dtype           : np.dtype = np.complex128
    ):
        self.ns             = ns
        self.max_particles  = max_particles or ns
        self.dtype          = dtype
        
        # Determinant/permanent workspace
        self._det_matrix        = np.empty((self.max_particles, self.max_particles), dtype=dtype)
        
        # Pfaffian/Hafnian workspace (for BdG: up to ns occupied)
        self._pairing_matrix    = np.empty((ns, ns), dtype=dtype)
        
        # Excited state workspace
        self._extended_matrix   = np.empty((2 * ns, 2 * ns), dtype=dtype)
        
        # Occupied indices buffer
        self._occ_buffer        = np.empty(self.max_particles, dtype=np.int64)
    
    @property
    def det_matrix(self) -> np.ndarray:
        """Workspace for determinant/permanent calculations."""
        return self._det_matrix
    
    @property
    def pairing_matrix(self) -> np.ndarray:
        """Workspace for Pfaffian/Hafnian submatrices."""
        return self._pairing_matrix
    
    @property
    def extended_matrix(self) -> np.ndarray:
        """Workspace for excited state calculations."""
        return self._extended_matrix
    
    @property
    def occ_buffer(self) -> np.ndarray:
        """Buffer for occupied index extraction."""
        return self._occ_buffer

# ============================================================================
# Test Function
# ============================================================================