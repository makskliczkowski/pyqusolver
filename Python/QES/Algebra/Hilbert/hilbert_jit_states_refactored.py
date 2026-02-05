"""
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

import  numpy as np
import  numba
from    numba import njit, prange

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
                    'many_body_state_full', 'many_body_state_mapping', 'many_body_state_closure',
                    'fill_many_body_state',
                    # Energy helpers
                    'nrg_particle_conserving', 'nrg_bdg',
                ]

# ============================================================================
# Configuration and constants
# ============================================================================

_TOLERANCE              = 1e-10
_USE_EIGEN              = False  # Use np.linalg.det (stable) instead of eigvals product

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
        Occupied orbital indices.
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

def calculate_bogoliubov_amp(
    F           : np.ndarray,
    basis       : Union[int, np.ndarray],
    ns          : int,
    enforce     : bool = True,
    workspace   : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Calculate Bogoliubov vacuum amplitude using Pfaffian.
    
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
        Pre-allocated (max_occ, max_occ) matrix for submatrix.
    
    Returns
    -------
    complex
        Pfaffian value. Returns 0 if odd number of particles.
    """
    pfaffian = _get_pfaffian()
    
    occ = extract_occupied(ns, basis)
    m = occ.size
    
    if m == 0:
        return 1.0 + 0.0j
    if m & 1:  # Odd number -> 0
        return 0.0 + 0.0j
    
    # Build submatrix
    if workspace is not None:
        sub = workspace[:m, :m]
    else:
        sub = np.empty((m, m), dtype=F.dtype)
    
    for p in range(m):
        ip = occ[p]
        for q in range(m):
            sub[p, q] = F[ip, occ[q]]
    
    if enforce:
        sub[:] = 0.5 * (sub - sub.T)
    
    return pfaffian.Pfaffian._pfaffian_parlett_reid(sub, m)

def calculate_bogoliubov_amp_exc(
    F           : np.ndarray,
    U           : np.ndarray,
    qp_inds     : np.ndarray,
    basis       : Union[int, np.ndarray],
    ns          : int,
    workspace   : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Calculate Bogoliubov amplitude for excited quasiparticle state.
    
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
        Pre-allocated workspace for the extended matrix.
    
    Returns
    -------
    complex
        Pfaffian of extended matrix.
    """
    pfaffian    = _get_pfaffian()

    occ         = extract_occupied(ns, basis)
    n           = occ.size
    k           = qp_inds.size
    
    if (n + k) & 1:
        return 0.0 + 0.0j
    if n == k == 0:
        return 1.0 + 0.0j
    
    dim = n + k
    
    if workspace is not None:
        M = workspace[:dim, :dim]
    else:
        M = np.empty((dim, dim), dtype=F.dtype)
    
    # F block
    for p in range(n):
        ip = occ[p]
        for q in range(n):
            M[p, q] = F[ip, occ[q]]
    
    # Cross blocks
    for p in range(n):
        ip = occ[p]
        for j in range(k):
            m_idx = qp_inds[j]
            M[p, n + j] = U[ip, m_idx]
            M[n + j, p] = -U[ip, m_idx]
    
    # Lower-right block = 0
    for i in range(k):
        for j in range(k):
            M[n + i, n + j] = 0.0
    
    return pfaffian.Pfaffian._pfaffian_parlett_reid(M, dim)

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

def calculate_bosonic_gaussian_amp(
    G           : np.ndarray,
    basis       : Union[int, np.ndarray],
    ns          : int,
    workspace   : Optional[np.ndarray] = None,
) -> complex:
    r"""
    Calculate bosonic Gaussian state amplitude using Hafnian.
    
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
        Pre-allocated workspace.
    
    Returns
    -------
    complex
        Hafnian value.
    """
    hafnian = _get_hafnian()
    
    occ = extract_occupied(ns, basis)
    m = occ.size
    
    if m == 0:
        return 1.0 + 0.0j
    if m & 1:
        return 0.0 + 0.0j
    
    if workspace is not None:
        sub = workspace[:m, :m]
    else:
        sub = np.empty((m, m), dtype=G.dtype)
    
    for p in range(m):
        ip = occ[p]
        for q in range(m):
            sub[p, q] = G[ip, occ[q]]
    
    return hafnian.Hafnian._hafnian_recursive(sub)

# ============================================================================
# Many-Body State Construction
# ============================================================================

@njit(parallel=True)
def fill_many_body_state(
    matrix_arg          : np.ndarray,
    occupied_orbitals   : Optional[np.ndarray],
    calculator_func     : CallableCoefficient,
    target_states       : np.ndarray,
    result              : np.ndarray,
    ns                  : int,
    indices             : Optional[np.ndarray] = None,
) -> None:
    """
    Fill result array with amplitudes for given basis states.
    
    Optimized with Numba parallel execution.
    
    Parameters
    ----------
    matrix_arg : np.ndarray
        Matrix needed by calculator (eigenvectors, pairing matrix).
    occupied_orbitals : Optional[np.ndarray]
        Orbital indices for particle-conserving systems.
    calculator_func : callable
        Amplitude calculator function.
    target_states : np.ndarray
        Integer basis states to compute.
    result : np.ndarray
        Pre-allocated output array.
    ns : int
        Number of sites.
    indices : Optional[np.ndarray]
        Mapping from target_states to result indices.
        If None, uses 0..len(target_states)-1.
    """
    n_states = target_states.shape[0]
    
    if indices is not None:
        for i in prange(n_states):
            state = target_states[i]
            idx = indices[i]
            if occupied_orbitals is not None:
                result[idx] = calculator_func(matrix_arg, occupied_orbitals, state, ns)
            else:
                result[idx] = calculator_func(matrix_arg, state, ns)
    else:
        for i in prange(n_states):
            state = target_states[i]
            if occupied_orbitals is not None:
                result[i] = calculator_func(matrix_arg, occupied_orbitals, state, ns)
            else:
                result[i] = calculator_func(matrix_arg, state, ns)

# Removed @njit because calculator might be a closure that Numba can't re-compile
def _fill_full_space_sequential(
    matrix_arg  : np.ndarray,
    calculator  : Callable[[np.ndarray, int, int], complex],
    ns          : int,
    nfilling    : Optional[int],
    result      : np.ndarray,
    workspace   : Optional[np.ndarray] = None,
) -> None:
    """Sequential fill for non-Numba calculators."""
    nh = result.size
    
    if nfilling is not None:
        
        # Pre-allocate workspace if not provided
        workspace = np.zeros((nfilling, nfilling), dtype=result.dtype) if workspace is None else workspace
        
        for st in range(nh):
            if popcount64(np.uint64(st)) == nfilling:
                result[st] = calculator(matrix_arg, int(st), ns, workspace)
            else:
                result[st] = 0.0
    else:
        for st in range(nh):
            result[st] = calculator(matrix_arg, int(st), ns, workspace)

def many_body_state_full(
    matrix_arg  : np.ndarray,
    calculator  : Callable[[np.ndarray, int, int], complex],
    ns          : int,
    resulting_s : Optional[np.ndarray] = None,
    nfilling    : Optional[int] = None,
    dtype       : np.dtype = np.complex128,
) -> np.ndarray:
    """
    Generate full many-body state vector over 2^ns basis.
    
    Parameters
    ----------
    matrix_arg : np.ndarray
        Input matrix for calculator.
    calculator : callable
        Function: (matrix, basis_state, ns) -> complex amplitude.
    ns : int
        Number of sites.
    resulting_s : Optional[np.ndarray]
        Pre-allocated output (2^ns,). If None, creates new array.
    dtype : np.dtype
        Output dtype.
    
    Returns
    -------
    np.ndarray
        State vector of shape (2^ns,).
    
    Notes
    -----
    For huge Hilbert spaces, always pass pre-allocated `resulting_s`
    to avoid repeated large allocations.
    
    Threading
    ---------
    Automatically uses ThreadPoolExecutor for nh > 4096 (ns >= 12).
    Adaptive worker count: min(8, nh // 512) to balance parallelization.
    Sequential execution for small Hilbert spaces (ns < 12) due to overhead.
    """
    
    if resulting_s is not None:
        nh  = resulting_s.shape[0]
        out = resulting_s if resulting_s.dtype == dtype else resulting_s.astype(dtype, copy=False)
    else:
        nh  = 1 << ns # 2**ns
        out = np.empty(nh, dtype=dtype)
    
    workspace = np.zeros((ns, ns), dtype=dtype) if nfilling is not None else None
    _fill_full_space_sequential(matrix_arg, calculator, ns, nfilling, out, workspace)
    
    return out

def many_body_state_mapping(
    matrix_arg      : np.ndarray,
    calculator      : CallableCoefficient,
    mapping_array   : np.ndarray,
    ns              : int,
    dtype           : np.dtype = np.complex128,
    result          : Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute many-body state for custom basis ordering.
    
    Parameters
    ----------
    matrix_arg : np.ndarray
        Input matrix.
    calculator : callable
        Amplitude calculator.
    mapping_array : np.ndarray
        mapping_array[j] = integer basis state for position j.
    ns : int
        Number of sites.
    dtype : np.dtype
        Output dtype.
    result : Optional[np.ndarray]
        Pre-allocated output. If None, creates new array.
    
    Returns
    -------
    np.ndarray
        State vector in custom ordering.
    """
    n_states = mapping_array.shape[0]
    
    if result is not None:
        out = result if result.dtype == dtype else result.astype(dtype, copy=False)
    else:
        out = np.empty(n_states, dtype=dtype)
    
    for i in range(n_states):
        out[i] = calculator(matrix_arg, mapping_array[i], ns)
    
    return out

def many_body_state_closure(
    calculator_func : Callable[[np.ndarray, np.ndarray, int, int, Optional[np.ndarray]], complex],
    matrix_arg      : Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, int, int, Optional[np.ndarray]], complex]:
    """
    Create closure that bakes in occupied orbitals.
    
    Parameters
    ----------
    calculator_func : callable
        Function: (matrix, orbitals, state, ns) -> amplitude.
    matrix_arg : Optional[np.ndarray]
        Occupied orbital indices to capture in closure.
    
    Returns
    -------
    callable
        Closure: (matrix, state, ns) -> amplitude.
    """
    if matrix_arg is not None:
        const = matrix_arg
        
        @njit(cache=True)
        def closure(U: np.ndarray, state: int, ns: int, workspace: Optional[np.ndarray] = None) -> complex:
            return calculator_func(U, const, state, ns, workspace)
        
        return closure
    
    @njit(cache=True)
    def closure(U: np.ndarray, state: int, ns: int, workspace: Optional[np.ndarray] = None) -> complex:
        return calculator_func(U, state, ns, workspace)
    
    return closure

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