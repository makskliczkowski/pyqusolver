'''
Quadratic Hamiltonian utilities.

----------------
File            : Algebra/Quadratic/hamil_quadratic_utils.py
Author          : Maksymilian Kliczkowski
Date            : 2026-02-01
----------------

Provides:
- Orbital occupation representation conversions (occ_to_indices, occ_to_mask, to_one_hot)
- Energy computation from orbital masks
- Orbital selection utilities (QuadraticSelection class)
- Haar random coefficient generation
- Solvability detection (SolvabilityInfo)

For low-level bit operations (ctz64, popcount64, mask_from_indices, indices_from_mask),
see QES.general_python.common.binary.
'''

from dataclasses    import dataclass
from collections    import defaultdict
from typing         import Optional, Sequence, Tuple, List, Union, Iterable
from enum           import Enum, unique
import numpy        as np

__all__ = [
    # Energy computation
    'energy_from_mask', 'energies_from_masks_batch',
    # Conversion helpers (high-level, orbital-specific)
    'to_one_hot', 'occ_to_indices', 'occ_to_mask',
    # Classes
    'QuadraticSelection', 'QuadraticBlockDiagonalInfo',
    'QuadraticTerm', 'SolvabilityInfo',
]

# ----------------------------------------------------------------------------
#! Numba setup
# ----------------------------------------------------------------------------

try:
    import numba
    numba_njit      = numba.njit
    HAS_NUMBA       = True
except ImportError:
    HAS_NUMBA       = False
    def numba_njit(*args, **kwargs):
        """Fallback decorator if Numba is not available."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# ----------------------------------------------------------------------------
#! Import core bit operations from binary.py (single source of truth)
# ----------------------------------------------------------------------------

try:
    from QES.general_python.common.binary   import (
                                                ctz64       as _ctz64,
                                                popcount64  as _popcount64,
                                                base2int,
                                                mask_from_indices,
                                                indices_from_mask,
                                                complement_indices,
                                                complement_mask,
                                            )
except ImportError:
    raise ImportError("Could not import bit operations from QES.general_python.common.binary. Ensure that pyqusolver is properly installed.")

try:
    from QES.qes_globals                        import get_numpy_rng
except ImportError:
    def get_numpy_rng():
        return np.random.default_rng()

##############################################################################
#! One-hot encoding
##############################################################################

def to_one_hot(positions    : Iterable[int],
                size        : int,
                *,
                asbool      : bool = True) -> np.ndarray:
    '''
    Transform a list of positions into a one-hot encoded array.

    Parameters
    ----------
    positions : Iterable[int]
        The positions to encode.
    size : int
        The size of the output array.
    asbool : bool, optional
        If True, return a boolean array. Default is True.

    Returns
    -------
    np.ndarray
        The one-hot encoded array.
    '''
    idx = list(positions)  # Ensure positions is indexable
    if asbool:
        y               = np.zeros(size, dtype=bool)
        y[idx]          = True
        return y
    else:
        y               = np.zeros(size, dtype=np.int8)
        y[idx]          = 1
        return y

# ----------------------------------------------------------------------------
#! Energy computation from orbital masks
# ----------------------------------------------------------------------------

@numba_njit(cache=True, parallel=False)
def energies_from_masks_batch(masks: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Compute energies for a batch of bitmasks.
    
    Args:
        masks (np.ndarray): Array of bitmasks (uint64).
        eps (np.ndarray): Array of single-particle energies.
        
    Returns:
        np.ndarray: Array of total energies for each mask.
    """
    n = masks.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = energy_from_mask(masks[i], eps)
    return out

# Note: complement_indices is defined locally above to avoid circular imports
# with general_python (which is a submodule and cannot import from QES)

def occ_to_indices(occ, Ns: int) -> np.ndarray:
    """
    Convert occupation specification to indices of occupied orbitals.
    Parameters
    ----------
    occ : int, array-like, or bool array
        Occupation specification:
        - If int: interpreted as number of occupied orbitals (0 to Ns).
        - If bool array: interpreted as mask of occupied orbitals.
        - If array-like of ints: interpreted as explicit indices of occupied orbitals.
    Ns : int
        Total number of orbitals.
    Returns
    -------
    np.ndarray
        Array of occupied orbital indices.
    """
    
    if isinstance(occ, (int, np.integer)):
        if occ < 0 or occ > Ns:
            raise ValueError("Integer `occ` must be in the range [0, Ns].")
        
        return indices_from_mask(np.uint64((1 << occ) - 1))

    # assume array-like
    occ = np.asarray(occ)
    if occ.dtype == np.bool_:
        if occ.ndim != 1 or occ.shape[0] != Ns:
            raise ValueError("bool occ mask must have shape (Ns,)")
        return np.nonzero(occ)[0].astype(np.int64, copy=False)

    # assume indices
    if occ.ndim != 1:
        raise ValueError("occ indices must be 1D")
    return occ.astype(np.int64, copy=False)    

def occ_to_mask(occ, Ns: int) -> np.uint64:
    """
    Convert occupation specification to a bitmask (uint64).
    Parameters
    ----------
    occ : int, array-like, or bool array
        Occupation specification:
        - If int: interpreted as number of occupied orbitals (0 to Ns).
        - If bool array: interpreted as mask of occupied orbitals.
        - If array-like of ints: interpreted as explicit indices of occupied orbitals.
    Ns : int
        Total number of orbitals.
    Returns
    -------
    np.uint64
        Bitmask representing occupied orbitals.
    """
    if isinstance(occ, (int, np.integer)):
        if occ < 0 or occ > Ns:
            raise ValueError("Integer `occ` must be in the range [0, Ns].")
        return np.uint64((1 << occ) - 1)

    # assume array-like
    occ = np.asarray(occ)
    if occ.dtype == np.bool_:
        if occ.ndim != 1 or occ.shape[0] != Ns:
            raise ValueError("bool occ mask must have shape (Ns,)")
        # Convert bool mask to indices, then to bitmask
        return mask_from_indices(np.nonzero(occ)[0].astype(np.int64, copy=False))

    # assume indices - use mask_from_indices directly (NOT recursive call)
    if occ.ndim != 1:
        raise ValueError("occ indices must be 1D")
    return mask_from_indices(occ.astype(np.int64, copy=False))


@numba_njit(cache=True)
def energy_from_mask(mask: np.uint64, eps: np.ndarray) -> float:
    '''
    Compute the total energy for a given bitmask of occupied orbitals.
    Numba-safe: uses _ctz64 instead of .bit_length().
    
    Args:
        mask (np.uint64):
            Bitmask representing occupied orbitals.
        eps (np.ndarray):
            Array of single-particle energies for each orbital.
    Returns:
        float:
            Total energy for the occupied orbitals in the mask.
    '''
    e = 0.0
    m = mask
    while m != 0:
        idx = _ctz64(m)
        e += eps[idx]
        # Clear the least significant bit
        m &= m - np.uint64(1)
    return e

@numba_njit(cache=True)
def energy_from_indices(indices: np.ndarray, eps: np.ndarray) -> float:
    '''
    Compute the total energy for a given list of occupied orbital indices.
    
    Args:
        indices (np.ndarray):
            Array of occupied orbital indices.
        eps (np.ndarray):
            Array of single-particle energies for each orbital.
    Returns:
        float:
            Total energy for the occupied orbitals.
    '''
    e = 0.0
    for idx in indices:
        e += eps[idx]
    return e

##############################################################################

@dataclass(frozen=True)
class QuadraticBlockDiagonalInfo:
    r"""
    Information for a single k-space block (or cell) of a quadratic Hamiltonian.

    Stores the diagonalized eigenvalues and eigenvectors for a small block
    (typically Nbtimes Nb for particle-conserving or 2Nbtimes 2Nb for BdG), along with
    the k-point coordinate. Useful for band structure analysis and
    sector-specific computations.

    Attributes
    ----------
    point : np.ndarray
        The k-vector or cell index for this block. Shape (3,).
        For k-space: 
            physical k-vector in reciprocal space.
        For real-space blocks: 
            cell coordinates or index.
    frac_point : Optional[np.ndarray]
        The fractional k-vector (e.g., [kx/2pi, ky/2pi, kz/2pi]). Shape (3,).
    en : np.ndarray
        Eigenvalues of this block, sorted in ascending order. Shape (M,) where
        M is the block dimension (Nb for particle-conserving, 2Nb for BdG).
    ev : np.ndarray
        Eigenvectors of this block as columns. Shape (M, M) where ev[:, i] is
        the eigenvector for eigenvalue en[i].
    block_index : Optional[Tuple[int, int, int]]
        Index (ix, iy, iz) in the lattice momentum grid. Useful for indexing
        back into the original Bloch blocks. None if not from k-space.
    is_bdg : bool
        True if this is a BdG block (2Nb x 2Nb), False if particle-conserving (Nb x Nb).
    label : Optional[str]
        Optional label for this block (e.g., "Gamma", "M", "X" for special k-points).

    Examples
    --------
    >>> # From band structure calculation
    >>> info = QuadraticBlockDiagonalInfo(
    ...     point=np.array([0.0, 0.0, 0.0]),
    ...     en=np.array([-1.0, 0.0, 1.0]),
    ...     ev=np.eye(3),
    ...     block_index=(0, 0, 0),
    ...     is_bdg=False,
    ...     label="Gamma"
    ... )
    """

    point       : np.ndarray
    frac_point  : Optional[np.ndarray]  # Fractional k-point vector
    en          : np.ndarray
    ev          : np.ndarray
    block_index : Optional[Tuple[int, int, int]] = None
    is_bdg      : bool = False
    label       : Optional[str] = None
    
    def __post_init__(self):
        """Validate dimensions."""
        if self.point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {self.point.shape}")

        if len(self.en) != self.ev.shape[0] or self.ev.shape[0] != self.ev.shape[1]:
            raise ValueError(f"ev shape {self.ev.shape} inconsistent with en shape {self.en.shape}")

    def __str__(self) -> str:
        """String representation."""
        block_type  = "BdG" if self.is_bdg else "PC"
        idx_str     = f" (idx={self.block_index})" if self.block_index else ""
        label_str   = f" [{self.label}]" if self.label else ""
        frac_str    = f" (frac={self.frac_point})" if self.frac_point is not None else ""
        return (
            f"QuadraticBlockDiagonalInfo({block_type}){label_str}{idx_str}{frac_str}\n"
            f"  q-point: {self.point}\n"
            f"  eigenvalues: {self.en}\n"
            f"  eigenvector shape: {self.ev.shape}"
        )

##############################################################################
#! Orbital selection utilities - particle-conserving sector only
##############################################################################

class QuadraticSelection:
    """Orbital selection utilities.
    For selecting orbitals, generating random combinations, and Haar-random states.
    
    Single particle-conserving sector only.
    """
    
    # ------------------------------------------------------------------------
    #! Orbital combinations - single particle-conserving sector
    # ------------------------------------------------------------------------

    @staticmethod
    def all_orbitals_size(n: int, k: int) -> int:
        """
        Binomial coefficient C(n, k).
        
        For single particle-conserving sector, this gives the number of ways to
        choose k orbitals from n total orbitals.
        
        Parameters
        ----------
        n : int
            Total number of orbitals.
        k : int
            Number of orbitals to choose.
        """
        from math import comb
        return comb(n, k)

    @staticmethod
    def all_orbitals(n: Union[int, list], k: int, *, as_array: bool = False, dtype=np.int64) -> Tuple[Union[np.ndarray, List[int], range], object]:
        """
        Generate all combinations of k orbitals from n. Takes either an integer n
        (total number of orbitals) or a list/array of orbital indices.
        
        Parameters
        ----------
        n : int or list-like
            Total number of orbitals or list of orbital indices.
        k : int
            Number of orbitals to choose.
        """
        from itertools import combinations

        if isinstance(n, (int, np.integer)):
            orbitals = range(int(n))
            combos   = combinations(orbitals, k)
            if as_array:
                # Allocate only when requested
                return np.fromiter(orbitals, dtype=dtype, count=int(n)), combos
            return orbitals, combos

        if isinstance(n, np.ndarray):
            return (n.astype(dtype, copy=False) if as_array else n), combinations(n, k)

        if isinstance(n, (list, tuple)):
            if as_array:
                return np.asarray(n, dtype=dtype), combinations(n, k)
            return n, combinations(n, k)

        n = tuple(n)
        return (np.asarray(n, dtype=dtype) if as_array else n), combinations(n, k)

    @staticmethod
    def ran_orbitals(n: Union[int, np.integer, Sequence[int], np.ndarray], k: int, rng: Optional[np.random.Generator] = None,
        *,
        return_pool_array: bool = False,
    ) -> Tuple[Union[range, np.ndarray], np.ndarray]:
        """
        Randomly select k orbitals from n (without replacement).
        
        The input n can be either an integer (total number of orbitals)
        or a list/array of orbital indices.
        
        Parameters
        ----------
        n : int or array-like
            Total number of orbitals or list/array of orbital indices.
        k : int
            Number of orbitals to select.
        rng : numpy.random.Generator or None
            Random number generator to use. If None, uses default RNG.
        """
        
        if rng is None:
            rng = get_numpy_rng()

        if isinstance(n, (int, np.integer)):
            nn          = int(n)
            selected    = rng.choice(nn, size=k, replace=False).astype(np.int64, copy=False)
            pool        = np.arange(nn, dtype=np.int64) if return_pool_array else range(nn)
            return pool, selected

        pool    = np.asarray(n, dtype=np.int64) if not isinstance(n, np.ndarray) else n.astype(np.int64, copy=False)
        idx     = rng.choice(pool.shape[0], size=k, replace=False)
        return pool, pool[idx]

    @staticmethod
    def mask_orbitals(
        ns              : int,
        n_occupation    : Union[int, float] = 0.5,
        *,
        ordered         : bool = True,
        rng             : Optional[Union[np.random.Generator, None]] = None,
        return_b        : bool = False,
    ) -> dict:
        r"""
        Generate a mask of occupied orbitals based on the number of orbitals and the occupation fraction.
        
        The mask indicates which orbitals are occupied (1) and which are unoccupied (0).
        
        Examples
        --------
        >>> QuadraticSelection.mask_orbitals(6, 3, ordered=True)
        {'mask_a': array([0, 1, 2]), 'mask_a_1h': array([1, 1, 1, 0, 0, 0]), 'mask_a_int': 7}
        >>> QuadraticSelection.mask_orbitals(6, 0.5, ordered=False)
        {'mask_a': array([3, 0, 5]), 'mask_a_1h': array([1, 0, 0, 1, 0, 1]), 'mask_a_int': 41}
        >>> QuadraticSelection.mask_orbitals(6, 2, ordered=True, return_b=True)
        {'mask_a': array([0, 1]), 'mask_a_1h': array([1, 1, 0, 0, 0, 0]), 'mask_a_int': 3,
         'mask_b': array([2, 3, 4, 5]), 'mask_b_1h': array([0, 0, 1, 1, 1, 1]), 'mask_b_int': 60,
         'order': (0, 1, 2, 3, 4, 5)}
        
        Parameters
        ----------
        ns : int
            Total number of orbitals.
        n_occupation : int or float
            Number of occupied orbitals or fraction of occupied orbitals.
            If `n_occupation` is a float, it should be in the range [0, 1].
        ordered : bool
            If True, the mask will be ordered (0s followed by 1s).
            If False, the mask will be randomly shuffled.
        rng : numpy.random.Generator or None
            Random number generator for shuffling the mask.
            If None, uses the default random generator.
        Returns
            dict
                Dictionary with keys 'mask_a' (occupied orbitals in subsystem a)
                and 'mask_b' (occupied orbitals in subsystem b - if applicable).
        """
        
        if n_occupation < 0:
            raise ValueError("`n_occupation` must be non-negative.")
        
        if isinstance(n_occupation, float) and 0.0 < n_occupation < 1.0:
            n_occupation    = int(n_occupation * ns)
            ordered         = True
            
        if n_occupation > ns:
            raise ValueError("`n_occupation` must be less than or equal to `ns`.")

        if rng is None:
            rng = get_numpy_rng()

        nocc        = int(n_occupation)

        if ordered:
            mask_a  = np.arange(nocc, dtype=np.int64)
        else:
            # no np.arange(ns) allocation; keeps random order
            mask_a  = rng.choice(ns, size=nocc, replace=False).astype(np.int64, copy=False)

        out                 = {}
        out["mask_a"]       = mask_a

        # Build one-hot directly (usually faster than a helper call, and avoids extra dtype churn).
        mask_a_1h           = np.zeros(ns, dtype=bool)
        mask_a_1h[mask_a]   = True
        out["mask_a_1h"]    = mask_a_1h

        # If base2int expects 0/1 array, bool usually works; if not, cast once.
        out["mask_a_int"]   = base2int(mask_a_1h, spin=False, spin_value=1)

        if return_b:
            # complement without setdiff1d / arange(ns)
            mask_b_1h         = ~mask_a_1h
            out["mask_b_1h"]  = mask_b_1h
            out["mask_b"]     = np.nonzero(mask_b_1h)[0].astype(np.int64, copy=False)
            out["mask_b_int"] = base2int(mask_b_1h, spin=False, spin_value=1)
            out["order"]      = tuple(mask_a.tolist()) + tuple(out["mask_b"].tolist())

        return out

    # ------------------------------------------------------------------------
    #! Haar random coefficients
    # ------------------------------------------------------------------------

    @staticmethod
    def _haar_random_coeff_cpx(
        gamma: int, *, rng: np.random.Generator | None = None, dtype=np.complex128
    ) -> np.ndarray:
        r"""
        Return a length-gamma complex vector distributed with the **Haar
        measure** on the unit sphere (i.e. what you get from the first
        column of a Haar-random unitary).

        Parameters
        ----------
        gamma : int
            Dimension of states to mix.
        rng : numpy.random.Generator, optional
            Random-number generator to use. `np.random.default_rng()` is the
            default; pass your own for reproducibility.
        dtype : np.dtype, default=np.complex128
            Precision of the returned coefficients.

        Notes
        -----
        * **Mathematical equivalence** - Drawing
        $$\psi_i = x_i + i y_i \\ (x_i,y_i\\sim N(0,1))$$
        and normalising,
        $$\\psi/\\lVert\\psi\\rVert$$
        gives exactly the same distribution as the first column of a Haar
        unitary (see, e.g., Mezzadri 2006).
        * If SciPy >= 1.4 is available we use `scipy.stats.unitary_group.rvs`
        (QR-based) instead, but the Gaussian trick is used as a fallback and
        is typically faster.

        Examples
        --------
        >>> gen_random_state_coefficients(4) # doctest: +ELLIPSIS
        array([ 0.44...+0.05...j, -0.40...-0.11...j,  0.63...+0.48...j,
                0.16...+0.29...j])
        """
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")
        if gamma == 1:
            return np.ones(1, dtype=dtype)

        rng = get_numpy_rng() if rng is None else rng

        #! fast path: SciPy's true Haar unitary, if available
        try:
            from scipy.stats import unitary_group

            return unitary_group.rvs(gamma, random_state=rng).astype(dtype)[:, 0]
        except Exception:
            # fall back to Gaussian normalise trick  (still Haar-correct)
            z   = rng.normal(size=gamma) + 1j * rng.normal(size=gamma)
            z   = z.astype(dtype, copy=False)
            z   /= np.linalg.norm(z)
            return z

    @staticmethod
    def _haar_random_unitary_cpx(
        gamma: int, *, rng: np.random.Generator | None = None, dtype=np.complex128
    ) -> np.ndarray:
        r"""
        Generate a Haar-random unitary matrix of shape (gamma, gamma).

        Parameters
        ----------
        gamma : int
            Dimension of the unitary matrix.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        dtype : np.dtype, default=np.complex128
            Desired complex dtype.

        Returns
        -------
        np.ndarray
            Haar-distributed unitary matrix of shape (gamma, gamma).

        Notes
        -----
        If SciPy >= 1.4 is available, uses `scipy.stats.unitary_group.rvs`,
        which samples unitaries via QR decomposition with Haar measure
        (Mezzadri 2006). Otherwise, performs the QR-based method manually.

        Reference
        ---------
        Mezzadri, F. (2006). How to generate random matrices from the classical groups.
        Notices of the AMS, 54(5), 592-604.

        Examples
        --------
        >>> U = haar_random_unitary(4)
        >>> np.allclose(U.conj().T @ U, np.eye(4))
        True
        """

        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")

        rng = get_numpy_rng() if rng is None else rng

        try:
            from scipy.stats import unitary_group
            return unitary_group.rvs(gamma, random_state=rng).astype(dtype)
        except Exception:
            # Fallback: generate complex Ginibre matrix and QR-decompose
            z = rng.normal(size=(gamma, gamma)) + 1j * rng.normal(size=(gamma, gamma))
            z = z.astype(dtype, copy=False)
            q, r = np.linalg.qr(z)
            # Normalize phases to ensure uniqueness
            d = np.diag(r)
            ph = d / np.abs(d)
            q *= ph[np.newaxis, :]
            return q

    @staticmethod
    def _haar_random_coeff_real(
        gamma: int, *, rng: np.random.Generator | None = None, dtype=np.float64
    ) -> np.ndarray:
        r"""
        Return a length-gamma real vector distributed with the Haar measure
        on the unit sphere (i.e. what you get from the first column of a Haar
        random orthogonal matrix).

        Parameters
        ----------
        gamma : int
            Dimension of states to mix.
        rng : numpy.random.Generator, optional
            Random-number generator to use. `np.random.default_rng()` is the
            default; pass your own for reproducibility.
        dtype : np.dtype, default=np.float64
            Precision of the returned coefficients.

        Returns
        -------
        np.ndarray
            A real vector of length `gamma` sampled from the Haar measure.

        Notes
        -----
        Uses Gaussian normalisation trick to sample from the Haar measure.

        Examples
        --------
        >>> haar_random_coeff_real(4) # doctest: +ELLIPSIS
        array([ 0.44..., -0.40...,  0.63...,  0.16...])
        """

        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")
        if gamma == 1:
            return np.ones(1, dtype=dtype)

        rng     = get_numpy_rng() if rng is None else rng
        # Generate normally distributed real numbers - normalize to unit sphere
        z       = rng.normal(size=gamma)
        z       = z.astype(dtype, copy=False)
        z      /= np.linalg.norm(z)
        return z

    @staticmethod
    def _haar_random_unitary_real(
        gamma: int, *, rng: np.random.Generator | None = None, dtype=np.float64
    ) -> np.ndarray:
        r"""
        Generate a Haar-random real orthogonal matrix of shape (gamma, gamma).
        Parameters
        ----------
        gamma : int
            Dimension of the orthogonal matrix.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        dtype : np.dtype, default=np.float64
            Desired real dtype.

        Returns
        -------
        np.ndarray
            Haar-distributed orthogonal matrix of shape (gamma, gamma).

        Notes
        -----
        Uses QR decomposition of a Gaussian matrix to sample from the Haar measure.

        Examples
        --------
        >>> U = haar_random_unitary_real(4)
        >>> np.allclose(U.T @ U, np.eye(4))
        True
        """

        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")

        rng     = get_numpy_rng() if rng is None else rng

        # Generate a random real orthogonal matrix using QR decomposition
        A       = rng.normal(size=(gamma, gamma))
        Q, R    = np.linalg.qr(A)
        d       = np.diag(R)
        Q      *= np.sign(d)

        # Return the first column as the Haar-random real vector
        return Q.astype(dtype, copy=False)

    @staticmethod
    def haar_random_unitary(gamma: int, *, rng: np.random.Generator | None = None, dtype=np.complex128) -> np.ndarray:
        r"""
        Generate a Haar-random unitary or orthogonal matrix of shape (gamma, gamma).

        Parameters
        ----------
        gamma : int
            Dimension of the unitary/orthogonal matrix.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        dtype : np.dtype, default=np.float64
            Desired dtype. If complex, generates a unitary; if real, generates an orthogonal.

        Returns
        -------
        np.ndarray
            Haar-distributed unitary (complex) or orthogonal (real) matrix of shape (gamma, gamma).

        Examples
        --------
        >>> U = haar_random_unitary(4, dtype=np.complex128)
        >>> np.allclose(U.conj().T @ U, np.eye(4))
        True
        >>> O = haar_random_unitary(4, dtype=np.float64)
        >>> np.allclose(O.T @ O, np.eye(4))
        True
        """

        if np.issubdtype(dtype, np.complexfloating):
            return QuadraticSelection._haar_random_unitary_cpx(gamma, rng=rng, dtype=dtype)
        else:
            return QuadraticSelection._haar_random_unitary_real(gamma, rng=rng, dtype=dtype)

    @staticmethod
    def haar_random_coeff(gamma: int, *, rng: np.random.Generator | None = None, dtype=np.complex128) -> np.ndarray:
        r"""
        Return a length-gamma vector (real or complex) distributed with the Haar
        measure on the unit sphere.

        Parameters
        ----------
        gamma : int
            Dimension of states to mix.
        rng : numpy.random.Generator, optional
            Random-number generator to use. `np.random.default_rng()` is the
            default; pass your own for reproducibility.
        dtype : np.dtype, default=np.float64
            Precision of the returned coefficients. If complex, returns complex
            coefficients; if real, returns real coefficients.

        Returns
        -------
        np.ndarray
            A vector of length `gamma` sampled from the Haar measure.

        Examples
        --------
        >>> haar_random_coeff(4, dtype=np.complex128) # doctest: +ELLIPSIS
        array([ 0.44...+0.05...j, -0.40...-0.11...j,  0.63...+0.48...j,
                0.16...+0.29...j])
        >>> haar_random_coeff(4, dtype=np.float64) # doctest: +ELLIPSIS
        array([ 0.44..., -0.40...,  0.63...,  0.16...])
        """

        if np.issubdtype(dtype, np.complexfloating):
            return QuadraticSelection._haar_random_coeff_cpx(gamma, rng=rng, dtype=dtype)
        else:
            return QuadraticSelection._haar_random_coeff_real(gamma, rng=rng, dtype=dtype)
    
    @staticmethod
    def haar_random_coeff_batch(
        batch: int, gamma: int, *, rng: np.random.Generator | None = None, dtype=np.complex128
    ) -> np.ndarray:
        r"""
        Generate multiple Haar-random coefficient vectors (batch generation).
        
        Vectorized version of haar_random_coeff for generating many coefficient
        sets at once. Uses the same Gaussian normalization algorithm but processes
        all vectors simultaneously for better performance.
        
        Parameters
        ----------
        batch : int
            Number of coefficient vectors to generate.
        gamma : int
            Dimension of each coefficient vector (number of states to mix).
        rng : numpy.random.Generator, optional
            Random-number generator to use. If None, uses default RNG.
        dtype : np.dtype, default=np.complex128
            Precision of the returned coefficients. If complex dtype, returns
            complex coefficients; if real dtype, returns real coefficients.
        
        Returns
        -------
        np.ndarray
            Array of shape (batch, gamma) with Haar-distributed coefficient vectors.
            Each row is independently sampled from the Haar measure on the unit sphere.
        
        Notes
        -----
        * Uses vectorized Gaussian normalization (same algorithm as haar_random_coeff):
          - Sample from N(0,1) for all batch x gamma elements
          - Normalize each row to unit sphere
          - Result is Haar-distributed (Mezzadri 2006)
        * Much faster than calling haar_random_coeff in a loop (~batch times speedup)
        * Memory: O(batch x gamma x dtype_size)
        
        Examples
        --------
        >>> coeffs = haar_random_coeff_batch(3, 4, dtype=np.complex128)
        >>> coeffs.shape
        (3, 4)
        >>> np.allclose(np.sum(np.abs(coeffs)**2, axis=1), 1.0)  # All normalized
        True
        """
        
        if batch < 1:
            raise ValueError("`batch` must be at least 1.")
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")
        
        rng = get_numpy_rng() if rng is None else rng
        
        # Vectorized generation for all batch x gamma coefficients at once
        if np.issubdtype(dtype, np.complexfloating):
            # Complex: real + imaginary parts
            real_part   = rng.standard_normal((batch, gamma))
            imag_part   = rng.standard_normal((batch, gamma))
            all_coeffs  = (real_part + 1j * imag_part).astype(dtype, copy=False)
        else:
            # Real only
            all_coeffs  = rng.standard_normal((batch, gamma)).astype(dtype, copy=False)
        
        # Vectorized normalization: normalize each row to unit sphere
        norms_sq        = np.sum(np.abs(all_coeffs)**2, axis=1, keepdims=True)
        all_coeffs     /= np.sqrt(norms_sq)
        
        return all_coeffs
        
    # ------------------------------------------------------------------------
    #! Energy concerned
    # ------------------------------------------------------------------------

    @staticmethod
    def sample_cfgs(ns: int, filling: int, rng: np.random.Generator, batch: int, sort: bool = True) -> np.ndarray:
        ''' 
        Take random samples of configurations with given filling. The samples
        are returned as arrays of orbital indices.
        
        Uses vectorized sampling when possible for better performance.
        
        Args:
            ns (int): Number of orbitals.
            filling (int): Number of occupied orbitals per configuration.
            rng (np.random.Generator): Random number generator.
            batch (int): Number of samples to generate.
            sort (bool): If True, sort each configuration (default True).
            
        Returns:
            np.ndarray: Array of shape (batch, filling) with orbital indices (dtype=int32).
        '''
        # Use uint16 for indices if ns <= 65535 (saves memory for large batches)
        idx_dtype   = np.uint16 if ns <= 65535 else np.int32
        
        # Vectorized approach: generate permutations and take first `filling` elements
        # This is faster than per-row rng.choice for large batches
        s           = np.empty((batch, filling), dtype=idx_dtype)
        
        # For small ns, can do fully vectorized with argsort trick
        if ns <= 256 and batch > 100:
            # Generate random keys for all positions, argsort gives permutation
            keys    = rng.random((batch, ns))
            perms   = np.argsort(keys, axis=1)[:, :filling]
            if sort:
                perms = np.sort(perms, axis=1)
            s[:] = perms.astype(idx_dtype, copy=False)
        else:
            # Fallback to per-row sampling
            for b in range(batch):
                x = rng.choice(ns, size=filling, replace=False)
                if sort:
                    x = np.sort(x)
                s[b] = x.astype(idx_dtype, copy=False)
        return s

    @staticmethod
    def pack_masks(samples: np.ndarray, *, ns: int = 64) -> np.ndarray:
        '''
        Convert sampled configurations (orbital indices) to bitmask representation.
        
        Args:
            samples (np.ndarray):
                Array of shape (B, k) with B samples of k occupied orbital indices.
            ns (int):
                Number of orbitals (determines dtype: uint32 if ns<=32, else uint64).
                
        Returns:
            np.ndarray:
                Array of shape (B,) with bitmask representation of each sample.
        '''
        # Choose appropriate dtype based on ns
        if ns <= 32:
            mask_dtype = np.uint32
            one = np.uint32(1)
        else:
            mask_dtype = np.uint64
            one = np.uint64(1)
        
        masks = np.zeros(samples.shape[0], dtype=mask_dtype)
        samples_cast = samples.astype(mask_dtype, copy=False)
        
        # Vectorized bit setting
        for j in range(samples.shape[1]):
            masks |= (one << samples_cast[:, j])
        return masks

    @staticmethod
    def gosper_masks(ns: int, k: int):
        '''
        Generator for all bitmasks of size `ns` with `k` bits set (1s). The
        masks are yielded in ascending order (Gosper's hack).
        
        Args:
            ns (int):
                Total number of bits (orbitals).
            k (int):
                Number of bits set (occupied orbitals).
        Yields:
            int:
                Next bitmask with `k` bits set.
                
        Examples
        --------
        >>> list(QuadraticSelection.gosper_masks(5, 3))
        [7, 11, 13, 14, 19, 21, 22, 25, 26, 28]
        >>> list(QuadraticSelection.gosper_masks(4, 4))
        [15]
        >>> list(QuadraticSelection.gosper_masks(4, 0))
        [0]
        '''
        if k < 0 or k > ns:
            return
        if k == 0:
            yield 0
            return
        if k == ns:
            # Full mask: all bits set
            yield (1 << ns) - 1
            return
        
        x       = (1 << k) - 1
        limit   = 1 << ns
        while x < limit:
            yield x
            c   = x & -x
            r   = x + c
            x   = (((r ^ x) >> 2) // c) | r
    
    # ------------------------------------------------------------------------
    #! Energy binning / manifolds
    # ------------------------------------------------------------------------

    @staticmethod
    def bin_energies(sorted_energies: np.ndarray, digits: int = 10):
        """
        Groups indices of sorted energies by their rounded values.

        Args:
            sorted_energies (list[tuple[int, float]]):
                A list of (index, energy) pairs, sorted by energy.
            digits (int):
                Number of decimal digits for rounding - used for
                binning the energies to avoid floating point errors.

        Returns:
            dict[float, list[int]]:
                Dictionary mapping rounded energy to list of indices.
        """
        binned = defaultdict(list)
        for idx, energy in sorted_energies:
            key = round(energy, digits)
            binned[key].append(idx)
        return dict(binned)

    @staticmethod
    def man_energies(binned_energies: dict, dtype = np.int32) -> Tuple[dict, np.ndarray]:
            """
            Calculates the number of elements in each energy manifold from a dictionary of binned energies.
            Args:
                binned_energies (dict):
                    A dictionary where keys represent energy bins and values are lists of items in each bin.
                dtype (np.dtype, optional):
                    The desired data type for the output array of manifold sizes. Defaults to np.int32.
            Returns:
                tuple:
                    - energy_manifolds (dict): A dictionary mapping each energy bin to the number of items in that bin.
                    - energy_manifold_values (np.ndarray): An array containing the number of items in each energy bin, in the order of iteration.
            """

            energy_manifolds        = {}
            energy_manifold_values  = []
            for k, v in binned_energies.items():
                n                   = len(v)
                energy_manifolds[k] = n
                energy_manifold_values.append(n)
            energy_manifold_values  = np.array(energy_manifold_values, dtype=dtype)
            return energy_manifolds, energy_manifold_values

##############################################################################
#! Solvability 
##############################################################################

@unique
class QuadraticTerm(Enum):
    """
    Types of terms to be added to the quadratic Hamiltonian
    """

    Onsite  = 0
    Hopping = 1
    Pairing = 2

    @property
    def mode_num(self):
        return 1 if self == QuadraticTerm.Onsite else 2
    
    @staticmethod
    def from_str(s: str) -> 'QuadraticTerm':
        s = s.lower()
        if s == 'onsite' or s.startswith('diag') or s.startswith('site'):
            return QuadraticTerm.Onsite
        elif s == 'hopping' or s.startswith('hop'):
            return QuadraticTerm.Hopping
        elif s == 'pairing' or s.startswith('pair'):
            return QuadraticTerm.Pairing
        else:
            raise ValueError(f"Unknown QuadraticTerm string: {s}")

##############################################################################

@dataclass(frozen=True)
class SolvabilityInfo:
    r"""
    Information about closed-form solvability of a quadratic Hamiltonian.

    A Hamiltonian is considered 'solvable' if eigenvalues can be obtained
    via a closed-form analytical method rather than requiring full matrix
    diagonalization.

    Attributes
    ----------
    is_solvable : bool
        True if closed-form solution is available
    method : str
        Solution method:
        - 'diagonal':
            Direct eigenvalues from diagonal elements
        - 'kspace':
            Via k-space Fourier transform (translational symmetry)
        - 'analytical':
            Known analytical form for specific model
        - 'standard':
            Requires standard matrix diagonalization
        - 'unknown':
            Method unknown
    description : str
        Human-readable explanation of solvability status
    eigenvalues : Optional[np.ndarray]
        Pre-computed eigenvalues if directly available, else None
    sparsity_pattern : Optional[str]
        Description of matrix sparsity (e.g., 'band-diagonal', 'tridiagonal')
    computation_cost : Optional[str]
        Estimated computational complexity (e.g., 'O(n)', 'O(n log n)', 'O(n^3)')

    Examples
    --------
    >>> # Diagonal Hamiltonian
    >>> info = SolvabilityInfo(
    ...     is_solvable         =   True,
    ...     method              =   'diagonal',
    ...     description         =   'Hamiltonian is diagonal in computational basis',
    ...     eigenvalues         =   np.array([1.0, 2.0, 3.0]),
    ...     sparsity_pattern    =   'full-diagonal',
    ...     computation_cost    =   'O(n)'
    ... )

    >>> # Translational invariant system
    >>> info = SolvabilityInfo(
    ...     is_solvable         =   True,
    ...     method              =   'kspace',
    ...     description         =   'System has translational symmetry, solvable via FFT'
    ... )
    """

    is_solvable                 : bool
    method                      : str
    description                 : str
    eigenvalues                 : Optional[np.ndarray]  = None
    sparsity_pattern            : Optional[str]         = None
    computation_cost            : Optional[str]         = None
    
    def __str__(self) -> str:
        """String representation."""
        status = "SOLVABLE" if self.is_solvable else "NOT_SOLVABLE"
        result = f"{status} ({self.method})\n{self.description}"
        if self.computation_cost:
            result += f"\n  Complexity: {self.computation_cost}"
        if self.sparsity_pattern:
            result += f"\n  Pattern: {self.sparsity_pattern}"
        return result

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------