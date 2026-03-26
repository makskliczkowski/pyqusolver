"""
This module contains functions for calculating statistical properties of quantum systems.

------------------------------------------------------------------------------
file    : QES/Algebra/Properties/statistical.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
version : 1.0
-------------------------------------------------------------------------------
"""

import  math
from    enum import Enum
from functools import partial
from typing import List, Optional, Tuple, Union

import numba
import numpy as np

try:

    from QES.general_python.algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    print("Error importing modules in statistical.py")

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jax = None
    jnp = np

class StatTypes(Enum):
    MEAN        = "mean"
    MEDIAN      = "median"
    VARIANCE    = "variance"
    STD         = "std"

# -----------------------------------------------------------------------------
#! LDOS
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnames=["degenerate", "tol"])
    def ldos_jax(
        energies: Array, overlaps: Array, degenerate: bool = False, tol: float = 1e-8
    ) -> Array:
        """
        JAX version of LDOS/strength function.
        """
        if not degenerate:
            return jnp.abs(overlaps) ** 2

        # for each E_i sum |overlaps[j]|^2 over j with |E_j - E_i| < tol
        def _ldos_i(E_i):
            mask = jnp.abs(energies - E_i) < tol
            return jnp.sum(jnp.abs(overlaps) ** 2 * mask)

        return jax.vmap(_ldos_i)(energies)

    @partial(jax.jit, static_argnames=["nbins"])
    def dos_jax(energies: Array, nbins: int = 100, **kwargs) -> Array:
        """
        JAX version of DOS via histogram binning.
        """
        counts, _ = jnp.histogram(energies, bins=nbins, **kwargs)
        return counts

else:
    ldos_jax = None
    dos_jax = None


def ldos(energies: Array, overlaps: Array, degenerate: bool = False, tol: float = 1e-8) -> Array:
    r"""
    Local density of states (LDOS) or strength function.

    If non-degenerate:
    .. math::
        \mathrm{LDOS}_i = |\,\langle i\,|\,\psi\rangle|^2.

    If degenerate, energies within `tol` are grouped:
    .. math::
        \mathrm{LDOS}_i = \sum_{j:|E_j - E_i|<\mathrm{tol}}
                        |\langle j|\psi\rangle|^2.

    Parameters
    ----------
    energies
        Eigenenergies \(E_n\), shape (N,).
    overlaps
        Overlap amplitudes \(\langle n|\psi\rangle\), shape (N,).
    degenerate
        Whether to sum over (nearly) degenerate levels.
    tol
        Tolerance for degeneracy grouping.

    Returns
    -------
    Array1D
        LDOS for each energy index.
    """
    if not degenerate:
        return np.abs(overlaps) ** 2

    N = energies.size
    ldos = np.empty(N, dtype=float)
    for i in range(N):
        mask = np.abs(energies - energies[i]) < tol
        ldos[i] = np.sum(np.abs(overlaps[mask]) ** 2)
    return ldos


def dos(energies: Array, nbins: int = 100, **kwargs) -> Array:
    r"""
    Density of states via histogram binning.

    Parameters
    ----------
    energies
        Eigenenergies array, shape (N,).
    nbins
        Number of bins.

    Returns
    -------
    Array1D
        Counts per energy bin.
    """
    counts, _ = np.histogram(energies, bins=nbins, **kwargs)
    return counts


# -----------------------------------------------------------------------------
#! Matrix elements
# -----------------------------------------------------------------------------


@numba.njit(fastmath=True, cache=True)
def extract_indices_window(
    start: int,
    stop: int,
    eigvals: np.ndarray,
    energy_target: float = 0.0,
    bw: float = 1.0,
    energy_diff_cut: float = 0.015,
    whole_spectrum: bool = False,
):
    """
    Extract indices of eigenvalues within a specified energy window.
    """

    if whole_spectrum:
        return np.empty((0, 3), dtype=np.int64), 0

    #! allocate -> idx_i, idx_j_start, idx_j_end
    if stop < start:
        tmp = start
        start = stop
        stop = tmp
    if stop > eigvals.shape[0]:
        stop = eigvals.shape[0]
    if start < 0:
        start = 0
    indices_alloc = np.zeros(((stop - start), 3), dtype=np.int64)

    tol = bw * energy_diff_cut
    j_lo = stop - 1
    j_hi = stop - 1

    # iterate i descending so j_lo/j_hi move forward only
    cnt = 0
    for i in range(start, stop):
        e_i = eigvals[i]
        # [|(E_i + E_j)/2 - e_target| < eps] -> [E_j < 2*e_target + eps - E_i] & [E_j > 2*e_target - eps - E_i]
        low = 2.0 * (energy_target - tol) - e_i
        high = 2.0 * (energy_target + tol) - e_i

        # advance j_hi to first eigvals[j] > high
        j_hi = stop - 1
        while eigvals[j_hi] >= high:
            j_hi -= 1

        # advance j_lo to first eigvals[j] >= low
        # we can start from j_hi!
        j_lo = j_hi
        while eigvals[j_lo] > low and j_lo > i:
            j_lo -= 1  # decrement in the upper right triangle

        if j_hi <= j_lo:
            break  # we finished the upper triangle

        indices_alloc[cnt, 0] = i
        indices_alloc[cnt, 1] = j_lo
        indices_alloc[cnt, 2] = j_hi + 1  # exclusive end
        cnt += 1
    return indices_alloc, cnt


@numba.njit(fastmath=True, cache=True, inline="always")
def _m2_hermitian(v):
    # Works for real or complex
    a = abs(v)
    return a * a


@numba.njit(fastmath=True, cache=True, inline="always")
def _m2_generic(x, y):
    # |x*y| = |x|*|y|
    return abs(x) * abs(y)


@numba.njit(cache=True, fastmath=True, inline="always")
def _bin_index(omega, bins, bin0, inv_binw, uniform_bins=False, uniform_log_bins=False):
    nBins = bins.shape[0] - 1

    if uniform_bins:
        idx = int((omega - bin0) * inv_binw) + 1  # shift by +1
        if omega < bins[0]:
            return 0
        elif omega >= bins[-1]:
            return nBins
        return idx

    elif uniform_log_bins:
        if omega <= 0.0:
            return 0  # underflow
        t = math.log(omega) - bin0
        b = int(t * inv_binw) + 1
        if omega < bins[0]:
            return 0
        elif omega >= bins[-1]:
            return nBins
        return b

    # Non-uniform: use binary search
    if omega < bins[0]:
        return 0
    elif omega >= bins[-1]:
        return nBins
    idx = np.searchsorted(bins, omega, side="right")
    return idx  # already in [1..nBins-1]


# -----------------------------------------------------------------------------


@numba.njit(fastmath=True, cache=True)
def _alloc_values_or_bins(
    nh: int, bins: Optional[np.ndarray] = None, indices_alloc: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    if bins is not None and bins.shape[0] >= 2:
        nbins = bins.shape[0]
        counts = np.zeros(nbins, dtype=np.uint64)
        sums = np.zeros(nbins, dtype=np.float64)
        empty_values = np.empty((0, 2), dtype=np.float64)
        return (counts, sums, nbins), empty_values
    else:
        if indices_alloc is not None and indices_alloc.shape[0] > 0 and indices_alloc.shape[1] == 3:
            cap = indices_alloc.shape[0]
        else:
            cap = nh * (nh - 1) // 2

        values = np.empty((cap, 2), dtype=np.float64)
        counts = np.empty(0, dtype=np.uint64)
        sums = np.empty(0, dtype=np.float64)
        return (counts, sums, 0), values


@numba.njit(fastmath=True, cache=True)
def _alloc_bin_info(
    uniform_bins: bool, uniform_log_bins: bool, bins: Optional[np.ndarray]
) -> Tuple[float, float, int]:
    """
    Allocate bin information for histogramming.
    """
    if (not uniform_bins and not uniform_log_bins) or (bins is None) or (bins.shape[0] < 2):
        return 0.0, 0.0, (False, False)

    if uniform_bins:
        bin0 = bins[0]
        binw = bins[1] - bins[0]
        inv_binw = 1.0 / binw if binw > 0.0 else 0.0
        uniform_log_bins = False
        return bin0, inv_binw, (True, False)
    elif uniform_log_bins:
        log_bin0 = math.log(bins[0]) if bins[0] > 0.0 else -np.inf
        log_binw = math.log(bins[1]) - log_bin0
        uniform_bins = False

        bin0 = log_bin0
        inv_binw = 1.0 / log_binw if log_binw > 0.0 else 0.0
        return bin0, inv_binw, (False, True)
    else:
        #! Non-uniform bins
        bin0 = 0.0
        inv_binw = 0.0
        uniform_bins = False
        uniform_log_bins = False
        return bin0, inv_binw, (False, False)


# -----------------------------------------------------------------------------


@numba.njit(fastmath=True, cache=True)
def _normalize_by_bin_width(sums: np.ndarray, bins: np.ndarray) -> None:
    r"""
    In-place divide by delta \Omega; counts- or typical-normalization can be done elsewhere.
    """
    nbins = bins.shape[0] - 1
    for i in range(nbins):
        w = bins[i + 1] - bins[i]
        if w > 0.0:
            sums[i] = sums[i] / w
        else:
            sums[i] = 0.0


# -----------------------------------------------------------------------------


@numba.njit(fastmath=True, cache=True)
def f_value(overlaps, ldos, i, j, log_eps, typical):
    m2 = _m2_hermitian(overlaps[i, j])
    return math.log(m2 + log_eps) if typical else m2


@numba.njit(fastmath=True, cache=True)
def k_value(overlaps, ldos, i, j, log_eps, typical):
    val = ldos[i] * ldos[j]
    return math.log(val + log_eps) if typical else val


@numba.njit(fastmath=True, cache=True)
def s_value(overlaps, ldos, i, j, log_eps, typical):
    val = ldos[i] * ldos[j]
    val *= _m2_hermitian(overlaps[i, j])
    return math.log(val + log_eps) if typical else val


@numba.njit(inline="always", fastmath=True)
def _value(mode, overlaps, ldos, i, j, log_eps, typical):
    """
    Combined f, k, s value function.
    """
    # mode: 0=f, 1=k, 2=s
    if mode == 0:  # f
        v = _m2_hermitian(overlaps[i, j])
    elif mode == 1:  # k
        v = ldos[i] * ldos[j]
    else:  # s
        v = ldos[i] * ldos[j]
        v *= _m2_hermitian(overlaps[i, j])
    return math.log(v + log_eps) if typical else v


@numba.njit(inline="always", fastmath=True)
def _own_contig_1d(a):
    # cheap guard; Numba can't introspect .base, so assume caller did the copy.
    return a


@numba.njit(inline="always", fastmath=True)
def _own_contig_2d(a):
    return a


@numba.njit(fastmath=True)
def pair_histogram(
    eigvals,
    overlaps,
    ldos,
    indices_alloc   =   None,
    bins            =   None,
    mode            :   int = 0,    # 0=f, 1=k, 2=s
    typical         =   False,      # uses logarithmic values if true for typical mean
    uniform_bins    =   False,      # if true, bins are uniform and we can use O(1) binning instead of binary search
    uniform_log_bins=   False,
    log_eps         =   1e-24,
):
    r"""
    Generic pairwise histogram/scatter accumulator. In
    pairwise mode (indices_alloc=None), it iterates over all pairs (i,j) with i<j and computes
    \Omega = |E_i - E_j| and value = value_fn(arr, i, j, log_eps, typical), then either
    accumulates value into the appropriate bin if bins are given, or stores (\Omega, value) pairs
    in the values array. In pre-allocated mode (indices_alloc provided), it iterates only over the specified
    (i, j) pairs given by indices_alloc, which should be an array of shape (M, 3) with rows of the form
    (i, j_start, j_end) indicating that for each i, we should consider pairs (i, j) with j in [j_start, j_end).
    
    The function for accumulation is determined by the `mode` parameter: mode=0 for f-function, mode=1 for k-function, and mode=2 for s-function.
    
    - f-function: 
    .. math::
        f_{ij} = |\langle i | O | j \rangle|^2
    - k-function:
    .. math::
        k_{ij} = \mathrm{LDOS}_i \cdot \mathrm{LDOS}_j
    - s-function:
    .. math::
        S_{ij} = \mathrm{LDOS}_i \cdot \mathrm{LDOS}_j \cdot |\langle i | O | j \rangle|^2
        
    Within a given omega bin, the values are summed, and if `typical` is True, the logarithm of the value is taken before summation 
    to compute a typical mean. The function returns either the histogram counts and sums or the list of (\Omega, value) 
    pairs depending on whether bins are provided.
    
    Parameters
    ----------
    eigvals : (N,) float
        Eigenvalues.
    arr : array
        Data array used inside `value_fn`.
    value_fn : njit function
        Must have signature (arr, i, j, log_eps, typical) -> float
    indices_alloc : (M,3) int or None
        Precomputed (i, j_start, j_end) triplets.
    bins : (B,) float or None
        Bin edges (len B >= 2).
    typical : bool
        If true, value_fn should handle log-scaling.
    uniform_bins, uniform_log_bins : bool
        Flags for binning mode.
    log_eps : float
        Small epsilon for logs.
    Returns
    -------
    values : (K,2) float
        (\Omega, value) pairs if not histogram mode, else empty.
    counts : (nbins,) uint64
        Bin counts if histogram mode.
    sums : (nbins,) float
        Bin sums (width-normalized, counts-normalization left for later).
    """
    nh                                      = eigvals.shape[0]
    use_hist                                = (bins is not None) and (bins.shape[0] >= 2)

    (counts, sums, nbins), values           = _alloc_values_or_bins(nh, bins, indices_alloc)
    bin0, inv_binw, (is_uniform, is_log)    = _alloc_bin_info(uniform_bins, uniform_log_bins, bins)

    #!path 1: indices_alloc provided
    if indices_alloc is not None and indices_alloc.shape[0] > 0 and indices_alloc.shape[1] == 3:
        if use_hist:
            for k in range(indices_alloc.shape[0]):
                i   = indices_alloc[k, 0]
                j0  = indices_alloc[k, 1]
                j1  = indices_alloc[k, 2]
                ei  = eigvals[i]
                for j in range(j0, j1):
                    omega = ei - eigvals[j]
                    if omega < 0.0:
                        omega = -omega
                    b = _bin_index(omega, bins, bin0, inv_binw, is_uniform, is_log)
                    if 0 <= b < nbins:
                        val         = _value(mode, overlaps, ldos, i, j, log_eps, typical)
                        sums[b]    += val
                        counts[b]  += 1
                        
            # _normalize_by_bin_width(sums, bins)
            return np.empty((0, 2), dtype=np.float64), counts, sums
        else:
            cnt = 0
            cap = values.shape[0]
            for k in range(indices_alloc.shape[0]):
                i   = indices_alloc[k, 0]
                j0  = indices_alloc[k, 1]
                j1  = indices_alloc[k, 2]
                for j in range(j0, j1):
                    if cnt >= cap:
                        break
                    omega = eigvals[i] - eigvals[j]
                    if omega < 0.0:
                        omega = -omega
                    val             = _value(mode, overlaps, ldos, i, j, log_eps, typical)
                    values[cnt, 0]  = omega
                    values[cnt, 1]  = val
                    cnt            += 1
            return values[:cnt], counts, sums

    #!path 2: generate pairs on the fly
    if use_hist:
        for i in range(nh):
            ei = eigvals[i]
            for j in range(i + 1, nh):
                omega = ei - eigvals[j]
                if omega < 0.0:
                    omega = -omega
                b = _bin_index(omega, bins, bin0, inv_binw, is_uniform, is_log)
                if 0 <= b < nbins:
                    val         = _value(mode, overlaps, ldos, i, j, log_eps, typical)
                    sums[b]    += val
                    counts[b]  += 1
        # _normalize_by_bin_width(sums, bins)
        return np.empty((0, 2), dtype=np.float64), counts, sums
    else:
        cnt = 0
        cap = values.shape[0]
        for i in range(nh):
            ei = eigvals[i]
            for j in range(i + 1, nh):
                if cnt >= cap:
                    break
                omega = ei - eigvals[j]
                if omega < 0.0:
                    omega = -omega
                val             = _value(mode, overlaps, ldos, i, j, log_eps, typical)
                values[cnt, 0]  = omega
                values[cnt, 1]  = val
                cnt            += 1
        return values[:cnt], counts, sums

# -----------------------------------------------------------------------------

@numba.njit(fastmath=True)
def f_function(
    overlaps,
    eigvals,
    indices_alloc=None,
    bins=None,
    typical=False,
    uniform_bins=False,
    uniform_log_bins=False,
    log_eps=1e-24,
):
    ldos = np.empty((0,), dtype=np.float64)  # dummy
    return pair_histogram(
        eigvals,
        overlaps,
        ldos,
        indices_alloc,
        bins,
        0,
        typical,
        uniform_bins,
        uniform_log_bins,
        log_eps,
    )


# -----------------------------------------------------------------------------
#! Fidelity susceptibility
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnames=["idx"])
    def fidelity_susceptibility_jax(
        energies: Array, V: Array, mu: float, idx: Optional[int] = None
    ) -> Array:
        """
        JAX version of fidelity susceptibility. If idx is given (and in-range),
        returns a scalar chi_idx; otherwise returns an Array of shape (N,) with all chi_i.
        """
        mu2 = mu * mu

        if idx is not None and 0 <= idx < energies.shape[0]:
            E = energies[idx]
            dE = energies - E  # shape (N,)
            omm = dE**2  # (E_j - E_i)^2
            denom2 = (omm + mu2) ** 2
            V2_row = jnp.abs(V[idx, :]) ** 2
            return jnp.sum(V2_row * omm / denom2)

        # full-vector version
        dE = energies[:, None] - energies[None, :]  # shape (N,N)
        omm = dE**2
        denom2 = (omm + mu2) ** 2
        V2 = jnp.abs(V) ** 2
        return jnp.sum(V2 * omm / denom2, axis=1)

else:
    fidelity_susceptibility_jax = None


def fidelity_susceptibility(
    energies: Array, V: Array, mu: float, idx: Optional[int] = None
) -> float:
    r"""
    Compute fidelity susceptibility for state `idx`:

    .. math::
        \chi_i = \sum_{j\neq i}
        \frac{|V_{ij}|^2\,(E_j - E_i)^2}
            {\bigl[(E_j - E_i)^2 + \mu^2\bigr]^2}\,.

    Parameters
    ----------
    idx
        Index of the reference eigenstate \(i\).
    energies
        1D array of eigenenergies \(E_n\).
    V
        2D overlap (or perturbation) matrix \(V_{nm}\).
    mu
        Broadening/cutoff parameter \(\mu\).

    Returns
    -------
    float
        Fidelity susceptibility \(\chi_i\).

    Examples
    --------
    >>> import numpy as np
    >>> from QES.Algebra.Properties.statistical import fidelity_susceptibility
    >>> energies = np.array([0.0, 1.0, 2.0])
    >>> V       = np.array([[0.0, 0.1, 0.2],
    ...                     [0.1, 0.0, 0.3],
    ...                     [0.2, 0.3, 0.0]])
    >>> mu      = 0.5
    >>> chi_0   = fidelity_susceptibility(energies, V, mu, idx=0)
    >>> print(chi_0)
    """
    mu2 = mu * mu

    if idx is not None:
        E = energies[idx]
        dE = energies - E
        omm = dE**2
        V_row = np.abs(V[idx]) ** 2

        mask = np.ones(len(energies), dtype=bool)
        mask[idx] = False

        denom = omm[mask] + mu2
        return np.sum(V_row[mask] * omm[mask] / denom**2)
    else:
        dE = energies[:, None] - energies[None, :]
        omm = dE**2
        denom2 = (omm + mu2) ** 2
        np.fill_diagonal(denom2, 1.0)  # avoid div-by-zero (will get zero in numerator anyway)
        V2 = np.abs(V) ** 2
        np.fill_diagonal(V2, 0.0)  # eliminate diagonal contribution explicitly
        return np.sum(V2 * omm / denom2, axis=1)


def fidelity_susceptibility_low_rank(
    energies: np.ndarray, V_overlaps: np.ndarray, mu: float, idx: Union[int, List[int], None] = None
) -> Union[float, np.ndarray]:
    r"""
    Compute fidelity susceptibility using only a subset of energies and
    projection vectors (overlaps). Useful when N_hilbert is too large.

    .. math::
        \chi_k = \sum_{n} \frac{|\langle n | V | k \rangle|^2\,(E_n - E_k)^2}
                {\bigl[(E_n - E_k)^2 + \mu^2\bigr]^2}

    Parameters
    ----------
    energies : np.ndarray
        1D array of eigenenergies \(E_n\) (size M) to sum over.
    V_overlaps : np.ndarray
        - If `idx` is an **int**: 1D array (size M) containing \(\langle n | V | k \rangle\).
        - If `idx` is a **list**: 2D array (size M, len(idx)) containing columns of overlaps.
        - If `idx` is **None**: 2D array (size M, M) (square matrix of the subspace).
    mu : float
        Broadening parameter \(\mu\).
    idx : int, list of ints, or None
        The index of the reference state(s) \(k\) within the `energies` array.

        - If `int`: Calculates scalar \(\chi_{idx}\).
        - If `list`: Calculates array of \(\chi\) for those specific indices.
        - If `None`: Calculates array of \(\chi\) for all M states (assumes V is square).

    Returns
    -------
    float or np.ndarray
        The fidelity susceptibility (or array of them).
    """

    energies = np.asarray(energies)
    V_overlaps = np.asarray(V_overlaps)
    mu2 = mu**2

    # if full ED
    if (
        len(V_overlaps.shape) == 2
        and V_overlaps.shape[0] == V_overlaps.shape[1]
        and V_overlaps.shape[0] == len(energies)
    ):
        return fidelity_susceptibility(energies, V_overlaps, mu, idx)

    # helper to compute one column
    def _compute_column(target_idx, overlap_col):
        E_k = energies[target_idx]
        dE = energies - E_k
        omm = dE**2

        # Numerator: |<n|V|k>|^2 * (En - Ek)^2
        numerator = (np.abs(overlap_col) ** 2) * omm

        # Denominator: ((En - Ek)^2 + mu^2)^2
        denom = (omm + mu2) ** 2

        # Handle the singularity/self-contribution (n=k)
        # Analytically this term is 0, numerically we force it to avoid 0/0 or noise
        # We use a mask instead of slicing to keep array shapes aligned
        mask = np.ones(len(energies), dtype=bool)
        mask[target_idx] = False

        return np.sum(numerator[mask] / denom[mask])

    # Single Target State (idx is int)
    if isinstance(idx, (int, np.integer)):
        if V_overlaps.ndim != 1:
            # Allow (M, 1) but flatten it
            if V_overlaps.ndim == 2 and V_overlaps.shape[1] == 1:
                V_overlaps = V_overlaps.flatten()
            else:
                V_overlaps = V_overlaps[:, idx]

        return _compute_column(idx, V_overlaps)

    # Specific Subset of States (idx is list)
    elif isinstance(idx, (list, tuple, np.ndarray)):
        idx = np.asarray(idx)
        if V_overlaps.ndim != 2 or V_overlaps.shape[1] != len(idx):
            raise ValueError(
                f"V_overlaps shape {V_overlaps.shape} must match (len(energies), len(idx))"
            )
        #
        results = []
        for i, target_k in enumerate(idx):
            col = V_overlaps[:, i]
            chi = _compute_column(target_k, col)
            results.append(chi)
        return np.array(results)

    # All States in subspace (idx is None)
    else:
        if V_overlaps.ndim != 2 or V_overlaps.shape[0] != V_overlaps.shape[1]:
            raise ValueError("If idx is None, V_overlaps must be a square matrix (M, M).")

        # Vectorized implementation for square matrix
        dE = energies[:, None] - energies[None, :]  # (n, k) matrix
        omm = dE**2

        numerator = (np.abs(V_overlaps) ** 2) * omm
        denom = (omm + mu2) ** 2

        # Clean diagonal
        np.fill_diagonal(numerator, 0.0)
        np.fill_diagonal(denom, 1.0)  # Avoid div by zero

        # Sum over n (rows), producing result for each k (columns)
        return np.sum(numerator / denom, axis=0)


# -----------------------------------------------------------------------------
#! State information
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnames=["q", "new_basis"])
    def inverse_participation_ratio_jax(
        state: Array, q: float = 1.0, new_basis: Optional[Array] = None
    ) -> float:
        r"""
        Compute the inverse participation ratio (IPR) of a quantum state.

        The IPR is defined as:

        .. math::
            \mathrm{IPR} = \sum_{i=1}^{N} |\psi_i|^{2q}\,.

        Parameters
        ----------
        state
            Quantum state, either a 1D array or a NumPy array.
        q
            Exponent for the IPR calculation.

        Returns
        -------
        float
            Inverse participation ratio.
        """
        if new_basis is not None:
            return jnp.sum(jnp.abs(new_basis.T @ state) ** (2 * q))
        return jnp.sum(jnp.abs(state) ** (2 * q))

else:
    inverse_participation_ratio_jax = None


@numba.njit(parallel=True, fastmath=True)
def _inverse_participation_ratio_2d(
    states: np.ndarray, q: float = 1.0, new_basis: Optional[np.ndarray] = None, square: bool = True
) -> np.ndarray:
    """
    Compute IPR_j = ∑_i |\\psi _{i j}|^{2q} for each column j of a 2D state array.
    """
    n, m    = states.shape
    out     = np.zeros(m, dtype=np.float64)
    two_q   = 2.0 * q if square else q

    if new_basis is None:
        # no transform
        for j in numba.prange(m):
            acc = 0.0
            for i in range(n):
                c    = states[i, j]
                p    = np.abs(c) ** two_q
                acc += p
            out[j] = acc
    else:
        # on-the-fly transform: φ_i = ∑_k B[k,i]*\psi _k
        # then acc += |φ_i|^(2q)
        B = new_basis
        for j in numba.prange(m):
            acc = 0.0
            for i in range(n):
                re = 0.0
                im = 0.0
                # compute (B^T\cdot \psi )_i = ∑_k B[k,i] * \psi [k,j]
                for k in range(n):
                    b   = B[k, i]
                    s   = states[k, j]
                    # complex multiply: (b_r + i b_i)*(s_r + i s_i)
                    re += b.real * s.real - b.imag * s.imag
                    im += b.real * s.imag + b.imag * s.real
                    
                # now re + i im is the transformed coefficient phi_i
                p       = re * re + im * im
                acc    += p**q
            out[j] = acc

    return out


@numba.njit(fastmath=True)
def _inverse_participation_ratio_1d(
    state: np.ndarray, q: float = 1.0, new_basis: Optional[np.ndarray] = None, square: bool = True
) -> float:
    """
    Compute the inverse participation ratio for a single state vector.
    """
    n       = state.shape[0]
    two_q   = 2.0 * q if square else q

    if new_basis is None:
        acc = 0.0
        for i in range(n):
            c    = state[i]
            acc += np.abs(c) ** two_q
        return acc

    acc = 0.0
    B   = new_basis
    for i in range(n):
        re = 0.0
        im = 0.0
        for k in range(n):
            b   = B[k, i]
            s   = state[k]
            re += b.real * s.real - b.imag * s.imag
            im += b.real * s.imag + b.imag * s.real
        acc += (re * re + im * im) ** q
    return acc


def inverse_participation_ratio(
    states: np.ndarray, q: float = 1.0, new_basis: Optional[np.ndarray] = None, square: bool = True
) -> Union[float, np.ndarray]:
    r"""
    Compute inverse participation ratios for one state vector or for a batch of states.

    The IPR is

    .. math::
        \mathrm{IPR} = \sum_i |\psi_i|^{2q}

    for a single state, or the same quantity evaluated independently for each
    column of a 2D array.

    Parameters
    ----------
    states : np.ndarray
        Complex array of shape `(n,)` or `(n, m)`.
        A 1D input is treated as one state vector.
        A 2D input is interpreted column-wise as `m` states.
    q : float
        Exponent in the IPR definition (default 1.0).
    new_basis : np.ndarray, optional
        Change-of-basis matrix of shape `(n, n)`. If provided, each state is
        transformed as `B^T @ psi` before the IPR is computed.
    square : bool
        If True, use exponent `2q`; if False, use exponent `q`.

    Returns
    -------
    float or np.ndarray
        Returns a scalar `float` for 1D input and a length-`m` array for 2D input.

    Notes
    -----
    The public wrapper dispatches to separate Numba kernels for 1D and 2D
    inputs. This avoids Numba type-unification failures when a single compiled
    function tries to treat the same variable as both rank-1 and rank-2.
    """
    states_arr = np.asarray(states)
    if states_arr.ndim == 1:
        return float(_inverse_participation_ratio_1d(states_arr, q=q, new_basis=new_basis, square=square))
    if states_arr.ndim == 2:
        return _inverse_participation_ratio_2d(states_arr, q=q, new_basis=new_basis, square=square)
    raise ValueError(f"`states` must be 1D or 2D, got shape {states_arr.shape}.")

# -----------------------------------------------------------------------------
#! Single-particle orbital statistics averaged over configurations
# -----------------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def _mean_selected_values_1d(values: np.ndarray, indices: np.ndarray) -> float:
    ''' Helper to compute the mean of selected values given by indices. '''
    acc = 0.0
    n   = indices.shape[0]
    for i in range(n):
        acc += values[indices[i]]
    return acc / n if n > 0 else 0.0

def configuration_mean_statistic(values: np.ndarray, configurations: np.ndarray) -> np.ndarray:
    r"""
    Average a per-orbital statistic over a set of occupied-orbital configurations.

    This helper is intentionally general. Assume a scalar observable
    :math:`s_\mu` is defined for each single-particle orbital :math:`\mu`, for
    example:

    - site-basis orbital IPR,
    - orbital participation entropy,
    - any other one-body scalar attached to an orbital index.

    A configuration is represented by a row of orbital indices,
    for example ``[1, 4, 7]`` meaning that orbitals ``1, 4, 7`` are occupied.
    For a configuration

    .. math::
        C_\alpha = (\mu_1, \ldots, \mu_N),

    this function returns the simple arithmetic mean

    .. math::
        \bar s(C_\alpha)
        = \frac{1}{N} \sum_{\mu \in C_\alpha} s_\mu.

    No assumptions are made about how the configurations were generated. They
    may be Slater determinants, subsets of orbitals selected from an energy
    window, momentum-constrained configurations, or any other integer-encoded
    occupied-orbital lists.

    Parameters
    ----------
    values : np.ndarray
        Per-orbital statistic, shape ``(n_orbitals,)``.
        Entry ``values[mu]`` is the scalar assigned to orbital index ``mu``.
    configurations : np.ndarray
        Integer array of shape ``(n_cfg, filling)``.
        Each row contains the occupied-orbital indices for one configuration.
        Example:

        .. code-block:: python

            configurations = np.array([
                [0, 1, 2],
                [0, 1, 5],
                [2, 4, 7],
            ])

        Here ``filling = 3`` and ``n_cfg = 3``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_cfg,)`` containing the mean statistic for each
        configuration.
    """
    values_arr  = np.asarray(values, dtype=np.float64)
    cfg_arr     = np.asarray(configurations, dtype=np.int64)
    
    # cheap guard against bad input; Numba can't introspect .ndim, so we check here before the call.
    if cfg_arr.ndim != 2:
        raise ValueError(f"`configurations` must be 2D, got shape {cfg_arr.shape}.")
    
    out = np.empty(cfg_arr.shape[0], dtype=np.float64)
    for i in range(cfg_arr.shape[0]):
        out[i]  = _mean_selected_values_1d(values_arr, cfg_arr[i])
    return out

def weighted_configuration_statistic(configuration_values: np.ndarray,
    configuration_indices: np.ndarray,
    weights: np.ndarray,
) -> float:
    r"""
    Form a weighted mean of precomputed configuration statistics.

    Suppose each available configuration :math:`C_\alpha` has already been
    assigned a scalar value :math:`x_\alpha`. A realization is then specified
    by selecting a subset of configuration labels and attaching weights
    :math:`w_\alpha`, giving

    .. math::
        X = \sum_{\alpha \in \mathcal{S}} w_\alpha x_\alpha.

    This helper implements exactly that pattern.

    The routine is generic: the ``x_\alpha`` values can represent averaged
    orbital IPRs, averaged local observables, configuration energies, or any
    other scalar defined per configuration.

    Parameters
    ----------
    configuration_values : np.ndarray
        Array of shape ``(n_cfg,)`` containing one scalar per available
        configuration.
    configuration_indices : np.ndarray
        Integer array of shape ``(m,)`` selecting which configurations are used
        in the current realization. These are indices into
        ``configuration_values``.
    weights : np.ndarray
        Weight array of shape ``(m,)`` associated with the selected
        configurations. In superposition problems this is typically
        :math:`|c_\alpha|^2`, so the result is a probabilistic average.

    Returns
    -------
    float
        Weighted average over the selected configuration values.
    """
    values  = np.asarray(configuration_values, dtype=np.float64)
    idx     = np.asarray(configuration_indices, dtype=np.int64)
    w       = np.asarray(weights, dtype=np.float64)
    return float(np.dot(w, values[idx]))

def weighted_orbital_ipr(
    orbital_vectors                 : np.ndarray,
    available_configurations        : np.ndarray,
    selected_configuration_indices  : np.ndarray,
    weights: np.ndarray,
    *,
    q: float = 2.0,
) -> float:
    r"""
    Compute a weighted mean site-basis orbital :math:`\mathrm{IPR}_q`.

    The columns of ``orbital_vectors`` are interpreted as single-particle
    orbitals :math:`\phi_\mu(x)` written in a site basis. For each orbital
    index :math:`\mu`, this function forms

    .. math::
        \mathrm{IPR}_q(\mu)
        = \sum_x |\phi_\mu(x)|^{2q}.

    A configuration

    .. math::
        C_\alpha = (\mu_1, \ldots, \mu_N)

    is encoded as a row of occupied-orbital indices, and is assigned the mean
    orbital statistic

    .. math::
        \overline{\mathrm{IPR}}_q(C_\alpha)
        = \frac{1}{N} \sum_{\mu \in C_\alpha} \mathrm{IPR}_q(\mu).

    Finally, for a selected subset of configurations with weights
    :math:`w_\alpha`, the returned scalar is

    .. math::
        \sum_{\alpha \in \mathcal{S}} w_\alpha\,
        \overline{\mathrm{IPR}}_q(C_\alpha).

    This is useful whenever a realization is built from multiple occupied
    orbital configurations, for example a superposition of Slater determinants.

    Parameters
    ----------
    orbital_vectors : np.ndarray
        Array of shape ``(n_sites, n_orbitals)``. Column ``mu`` is the
        site-basis wavefunction of orbital ``mu``.
        
        Example: if the columns are single-particle eigenstates, then
        ``orbital_vectors[:, mu]`` is the eigenstate wavefunction of orbital ``mu``
        in the site basis, and the resulting IPR is the standard orbital IPR.
        
    available_configurations : np.ndarray
        Integer array of shape ``(n_cfg, filling)``. Each row is a list of
        occupied-orbital indices defining one configuration. For example, if
        ``available_configurations[i] = [1, 4, 7]``, then configuration ``i`` corresponds to the occupation of orbitals
        ``1, 4, 7``. Then, it can be selected from orbital_vectors
        as ``orbital_vectors[:, [1, 4, 7]]`` to get the wavefunctions of the occupied orbitals in that configuration.
    selected_configuration_indices : np.ndarray
        Integer array selecting which rows of ``available_configurations`` are
        used in the current realization. These are indices into ``available_configurations`` and ``configuration_values``.
    weights : np.ndarray
        Weight array for the selected configurations. Typically
        :math:`|c_\alpha|^2`.
    q : float, optional
        Generalized IPR exponent. ``q=2`` gives the standard
        :math:`\sum_x |\phi(x)|^4`.

    Returns
    -------
    float
        Weighted mean orbital :math:`\mathrm{IPR}_q` for the realization.
    """
    orbital_ipr_q = np.asarray(inverse_participation_ratio(orbital_vectors, q=q), dtype=np.float64)
    config_values = configuration_mean_statistic(orbital_ipr_q, available_configurations)
    return weighted_configuration_statistic(config_values, selected_configuration_indices, weights)


# -----------------------------------------------------------------------------
#! K - function
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True)
def k_function(
    ldos: np.ndarray,
    eigvals: np.ndarray,
    indices_alloc: Optional[np.ndarray] = None,
    bins: Optional[np.ndarray] = None,
    # additional parameters
    typical: bool = False,
    uniform_bins: bool = False,
    uniform_log_bins: bool = False,
    log_eps: float = 1e-24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlaps = np.empty((1, 1), dtype=np.complex128)  # dummy
    return pair_histogram(
        eigvals,
        overlaps,
        ldos,
        indices_alloc,
        bins,
        1,
        typical,
        uniform_bins,
        uniform_log_bins,
        log_eps,
    )


# -----------------------------------------------------------------------------
#! Fourier spectrum function - S(omega) = \sum _{n \neq m} |c_n|^2 |c_m|^2 |O_mn|^2 \delta (omega - |E_m - E_n|)
# -----------------------------------------------------------------------------


@numba.njit(fastmath=True)
def s_function(
    ldos: np.ndarray,
    eigvals: np.ndarray,
    overlaps: np.ndarray,
    indices_alloc: Optional[np.ndarray] = None,
    bins: Optional[np.ndarray] = None,
    # additional parameters
    typical: bool = False,
    uniform_bins: bool = False,
    uniform_log_bins: bool = False,
    log_eps: float = 1e-24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pair_histogram(
        eigvals,
        overlaps,
        ldos,
        indices_alloc,
        bins,
        2,
        typical,
        uniform_bins,
        uniform_log_bins,
        log_eps,
    )


# -----------------------------------------------------------------------------
#! Spectral CDF
# -----------------------------------------------------------------------------


@staticmethod
def spectral_cdf(x, y, gammaval=0.5, BINVAL=21):
    """
    Calculate the cumulative distribution function (CDF) and find the gamma value.

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values, which may contain NaNs.
    gammaval (float, optional): The target CDF value to find the corresponding gamma value. Default is 0.5.

    Returns:
    tuple: A tuple containing:
        - x (array-like): The input independent variable values.
        - y (array-like): The input dependent variable values with NaNs removed.
        - cdf (array-like): The cumulative distribution function values.
        - gammaf (float): The value of the independent variable corresponding to the target CDF value.
    """
    # Apply the moving average to smooth y
    y_smoothed = np.convolve(y, np.ones(BINVAL) / BINVAL, mode="same")
    cdf = np.cumsum(y_smoothed * np.diff(np.insert(x, 0, 0)))
    cdf /= cdf[-1]
    y_smoothed /= cdf[-1]
    gammaf = x[np.argmin(np.abs(cdf - gammaval))]
    return x, y_smoothed, cdf, gammaf


# -----------------------------------------------------------------------------
#! Survival probability
# -----------------------------------------------------------------------------


@numba.njit(fastmath=True, cache=True)
def survival_prob(
    psi0: np.ndarray, psi_t: np.ndarray, axis: int = 0, out: np.ndarray | None = None
) -> np.ndarray:
    """
    P_k = |<psi(0) | psi(t_k)>|^2

    psi0 : (H,) complex
    psi_t : (H, N) complex
        - axis=0 -> (H, N)  columns are states at times t_k
        - axis=1 -> (N, H)  rows    are states at times t_k
    """
    if axis == 0:
        H, N = psi_t.shape[0], psi_t.shape[1]
        if psi0.shape[0] != H:
            raise ValueError("psi0 length mismatch with psi_t (axis=0).")
        # allocate output if needed
        if out is None or out.shape[0] != N:
            P = np.empty(N, dtype=psi_t.real.dtype)
        else:
            P = out
        # P_k = | sum_h conj(psi0[h]) * psi_t[h,k] |^2
        for k in range(N):
            re = 0.0
            im = 0.0
            for h in range(H):
                ar = psi_t[h, k].real
                ai = psi_t[h, k].imag
                br = psi0[h].real
                bi = psi0[h].imag
                # a * conj(b) = (ar+iai)*(br-ibi)
                re += ar * br + ai * bi
                im += -ar * bi + ai * br
            P[k] = re * re + im * im
        return P

    elif axis == 1:
        N, H = psi_t.shape[0], psi_t.shape[1]
        if psi0.shape[0] != H:
            raise ValueError("psi0 length mismatch with psi_t (axis=1).")
        if out is None or out.shape[0] != N:
            P = np.empty(N, dtype=psi_t.real.dtype)
        else:
            P = out
        # P_k = | sum_h conj(psi0[h]) * psi_t[k,h] |^2
        for k in range(N):
            re = 0.0
            im = 0.0
            for h in range(H):
                ar = psi_t[k, h].real
                ai = psi_t[k, h].imag
                br = psi0[h].real
                bi = psi0[h].imag
                re += ar * br + ai * bi
                im += -ar * bi + ai * br
            P[k] = re * re + im * im
        return P

    else:
        raise ValueError("axis must be 0 (psi_t shape (H,N)) or 1 (psi_t shape (N,H)).")


# -----------------------------------------------------------------------------
#! Structures
# -----------------------------------------------------------------------------


def spectral_structure(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the residuals of a moving average (spectral structure) for each row in the input data.

    For each row, the function subtracts a moving average from the data:
    - For the first `window` points, the moving average is computed with a growing denominator (1, 2, ..., window).
    - For the remaining points, a fixed-size window is used.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array of shape (N, T), where N is the number of rows (e.g., signals or samples)
        and T is the number of time points.
    window : int
        Size of the moving average window.

    Returns
    -------
    np.ndarray
        Array of the same shape as `data`, containing the residuals after subtracting the moving average.
    """

    N, T = data.shape
    cumsum = np.cumsum(data, axis=1)  # shape (N, T)
    residual = np.empty_like(data, dtype=float)

    # first `window` points use growing denominator (1,2,…,window)
    t0 = min(window, T)
    counts = np.arange(1, t0 + 1)  # [1, 2, …, t0]
    residual[:, :t0] = data[:, :t0] - cumsum[:, :t0] / counts

    # remaining points use fixed window
    if T > window:
        numer = cumsum[:, window:] - cumsum[:, :-window]
        ma = numer / window
        residual[:, window:] = data[:, window:] - ma

    return residual


# -----------------------------------------------------------------------------
#! Statistical properties
# -----------------------------------------------------------------------------


def microcanonical_average(
    energies: np.ndarray,
    observables: np.ndarray,
    e_mean: float,
    delta_e: float = 5e-2,
    n_closest: int = 10,
    stat: StatTypes = StatTypes.MEAN,
) -> float:
    r"""
    Compute the microcanonical ensemble average Q_ME(\mu) for a single realization \mu.

    According to Eq. (E4):
        Q_ME(\mu) = (1 / N_{\epsilon_i,\Delta\epsilon}) * Σ_{|E_n - \epsilon_i| < \Delta\epsilon} O_n,
    where O_n is the observable value corresponding to eigenstate n.

    If there are no states within |E_n - \epsilon_i| < \Delta\epsilon, the function instead selects
    the n_closest eigenstates (default: 10) closest in energy to \epsilon_i.

    Parameters
    ----------
    energies : np.ndarray
        1D array of eigenenergies E_n for a single realization.
    observables : np.ndarray
        1D array of observable values O_n corresponding to each energy E_n.
    e_mean : float
        Mean energy \epsilon_i of the initial state.
    delta_e : float, optional
        Microcanonical window width \Delta\epsilon. Default is 5e-2.
    n_closest : int, optional
        Number of nearest eigenstates to use if the window is empty. Default is 10.

    Returns
    -------
    float
        Microcanonical ensemble average Q_ME(\mu).
    """

    if len(energies) == 0 or len(observables) == 0:
        return np.nan

    if isinstance(stat, str):
        stat = StatTypes(stat.lower())

    # indices within |E_n - \epsilon_i| < \Delta\epsilon
    idx = np.where(np.abs(energies - e_mean) < delta_e)[0]

    if len(idx) == 0:
        # fallback: take n_closest states
        idx = np.argsort(np.abs(energies - e_mean))[:n_closest]

    if stat == StatTypes.MEAN:
        return np.mean(observables[idx])
    elif stat == StatTypes.MEDIAN:
        return np.median(observables[idx])
    elif stat == StatTypes.MAX:
        return np.max(observables[idx])
    elif stat == StatTypes.MIN:
        return np.min(observables[idx])
    else:
        raise ValueError(f"Unknown statistical method: {stat}")
    return np.nan


# -----------------------------------------------------------------------------
#! StatisticalModule - Hamiltonian wrapper
# -----------------------------------------------------------------------------


class StatisticalModule:
    """
    Statistical properties module for Hamiltonians.

    Provides convenient access to LDOS, DOS, matrix element statistics,
    and ensemble averages. Requires diagonalized Hamiltonian.

    Examples
    --------
    >>> hamil.diagonalize()
    >>> stats = hamil.statistical
    >>>
    >>> # LDOS for a state
    >>> psi0 = np.zeros(hamil.hilbert_size); psi0[0] = 1.0
    >>> overlaps = hamil.eig_vec.conj().T @ psi0
    >>> ldos_vals = stats.ldos(overlaps)
    >>>
    >>> # DOS histogram
    >>> dos_vals = stats.dos(nbins=50)
    >>>
    >>> # Diagonal ensemble average
    >>> observable = np.diag(hamil.hamil)
    >>> avg = stats.diagonal_ensemble(observable, overlaps)
    """

    def __init__(self, hamiltonian):
        self._hamil = hamiltonian

    def _check_diagonalized(self):
        if self._hamil._eig_val is None or len(self._hamil._eig_val) == 0:
            raise RuntimeError("Hamiltonian must be diagonalized first. Call hamil.diagonalize().")

    @property
    def energies(self) -> Array:
        """Get eigenvalues."""
        self._check_diagonalized()
        return self._hamil._eig_val

    def ldos(self, overlaps: Array, degenerate: bool = False, tol: float = 1e-8) -> Array:
        """
        Local density of states (strength function).

        Parameters
        ----------
        overlaps : Array
            Overlaps <n|psi> of initial state with eigenstates
        degenerate : bool
            If True, group degenerate states
        tol : float
            Tolerance for degeneracy

        Returns
        -------
        Array
            LDOS values |<n|psi>|^2
        """
        self._check_diagonalized()
        return ldos(self.energies, overlaps, degenerate=degenerate, tol=tol)

    def dos(self, nbins: int = 100, **kwargs) -> Array:
        """
        Density of states histogram.

        Parameters
        ----------
        nbins : int
            Number of histogram bins

        Returns
        -------
        Array
            Histogram counts
        """
        self._check_diagonalized()
        return dos(self.energies, nbins=nbins, **kwargs)

    def diagonal_ensemble(self, observable: Array, overlaps: Array) -> float:
        """
        Diagonal ensemble average <O>_DE = sum_n |c_n|^2 O_nn.

        Parameters
        ----------
        observable : Array
            Observable matrix or diagonal values
        overlaps : Array
            Overlaps <n|psi_0> with initial state

        Returns
        -------
        float
            Diagonal ensemble average
        """
        self._check_diagonalized()
        probs = np.abs(overlaps) ** 2
        if observable.ndim == 2:
            obs_diag = np.diag(observable)
        else:
            obs_diag = observable
        return np.sum(probs * obs_diag)

    def microcanonical_average(
        self, observable: Array, e_mean: float, delta_e: float = 5e-2, stat: str = "mean"
    ) -> float:
        """
        Microcanonical ensemble average around energy e_mean.

        Parameters
        ----------
        observable : Array
            Observable values for each eigenstate
        e_mean : float
            Target energy
        delta_e : float
            Energy window width
        stat : str
            'mean', 'median', 'max', or 'min'

        Returns
        -------
        float
            Microcanonical average
        """
        self._check_diagonalized()
        return microcanonical_average(
            self.energies, observable, e_mean, delta_e=delta_e, stat=stat
        )

    # -------------------------------------------------------------------------
    # Fidelity Susceptibility
    # -------------------------------------------------------------------------

    def fidelity_susceptibility(
        self, operator_matrix: Array, state_idx: int = 0, mu: float = None
    ) -> float:
        """
        Compute fidelity susceptibility χ_F for a given perturbation.

        χ_F = sum_{n≠m} |<n|V|m>|² / (E_n - E_m)²

        Parameters
        ----------
        operator_matrix : Array
            Perturbation operator V in eigenbasis (shape: n_states x n_states)
            or full Hilbert space matrix
        state_idx : int
            Index of the reference state (default: ground state)
        mu : float, optional
            Regularization parameter. If None, uses 1/N_hilbert.

        Returns
        -------
        float
            Fidelity susceptibility

        Example
        -------
        >>> # Project total S_z onto eigenbasis
        >>> Sz_proj = eig_vec.conj().T @ Sz_operator @ eig_vec
        >>> chi_F = hamil.statistical.fidelity_susceptibility(Sz_proj, state_idx=0)
        """
        self._check_diagonalized()
        if mu is None:
            mu = 1.0 / self._hamil.hilbert_space.nh
        return fidelity_susceptibility_low_rank(
            self.energies, operator_matrix, mu=mu, idx=state_idx
        )

    # -------------------------------------------------------------------------
    # Inverse Participation Ratio
    # -------------------------------------------------------------------------

    def ipr(self, state: Array = None, state_idx: int = None, q: float = 2.0) -> float:
        """
        Inverse Participation Ratio: IPR_q = sum_i |ψ_i|^(2q).

        For q=2 (default): IPR = sum_i |ψ_i|^4

        Parameters
        ----------
        state : Array, optional
            State vector. If None, uses eigenstate at state_idx.
        state_idx : int, optional
            Index of eigenstate to use. Default: 0 (ground state).
        q : float
            Rényi parameter. Default: 2.

        Returns
        -------
        float
            IPR value. 1/IPR gives effective number of basis states.
        """
        if state is None:
            self._check_diagonalized()
            state_idx   = state_idx if state_idx is not None else 0
            state       = self._hamil.eig_vec[:, state_idx]
        probs = np.abs(state) ** 2
        return np.sum(probs**q)

    def participation_entropy(self, state: Array = None, state_idx: int = None) -> float:
        """
        Participation entropy: S_p = -sum_i |ψ_i|^2 log(|ψ_i|^2).

        Parameters
        ----------
        state : Array, optional
            State vector. If None, uses eigenstate at state_idx.
        state_idx : int, optional
            Index of eigenstate. Default: 0 (ground state).

        Returns
        -------
        float
            Participation entropy.
        """
        if state is None:
            self._check_diagonalized()
            state_idx = state_idx if state_idx is not None else 0
            state = self._hamil.eig_vec[:, state_idx]
        probs = np.abs(state) ** 2
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    # -------------------------------------------------------------------------
    # Level Statistics
    # -------------------------------------------------------------------------

    def level_spacing(self, unfolded: bool = False) -> Array:
        """
        Level spacings: s_i = E_{i+1} - E_i.

        Parameters
        ----------
        unfolded : bool
            If True, normalize by mean spacing (for r-statistics).

        Returns
        -------
        Array
            Level spacings.
        """
        self._check_diagonalized()
        spacings = np.diff(self.energies)
        if unfolded:
            mean_spacing = np.mean(spacings)
            spacings = spacings / mean_spacing
        return spacings

    def level_spacing_ratio(self) -> Array:
        """
        Level spacing ratio: r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1}).

        For GOE (chaotic): <r> ≈ 0.536
        For Poisson (integrable): <r> ≈ 0.386

        Returns
        -------
        Array
            Level spacing ratios.
        """
        self._check_diagonalized()
        spacings = np.diff(self.energies)
        r = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
        return r

    def mean_level_spacing_ratio(self) -> float:
        """Average level spacing ratio <r>."""
        return np.mean(self.level_spacing_ratio())

    # -------------------------------------------------------------------------
    # Survival Probability
    # -------------------------------------------------------------------------

    def survival_probability(self, initial_state: Array, times: Array) -> Array:
        """
        Survival probability |<ψ(0)|ψ(t)>|².

        Parameters
        ----------
        initial_state : Array
            Initial state in Hilbert space basis.
        times : Array
            Time points.

        Returns
        -------
        Array
            Survival probability at each time.
        """
        self._check_diagonalized()
        overlaps = self._hamil.eig_vec.conj().T @ initial_state
        return survival_prob(initial_state, overlaps, self.energies, times)

    def help(self):
        """Print help for statistical module."""
        print("""
        StatisticalModule - Statistical properties for Hamiltonians
        ===========================================================

        Requires: hamil.diagonalize() first

        Methods:
        --------
        # Density of States
        ldos(overlaps)                  - Local density of states
        dos(nbins=100)                  - Density of states histogram

        # Ensemble Averages
        diagonal_ensemble(O, c)         - Diagonal ensemble <O>_DE
        microcanonical_average(...)     - Microcanonical average

        # Fidelity & Localization
        fidelity_susceptibility(V)      - Fidelity susceptibility χ_F
        ipr(state, q=2)                 - Inverse participation ratio
        participation_entropy(state)    - Participation entropy

        # Level Statistics
        level_spacing(unfolded=False)   - Level spacings s_i = E_{i+1} - E_i
        level_spacing_ratio()           - r_i = min/max ratio
        mean_level_spacing_ratio()      - <r> average

        # Dynamics
        survival_probability(psi0, t)   - |<ψ(0)|ψ(t)>|²

        Example:
        --------
        >>> psi0    = np.zeros(N); psi0[0] = 1.0
        >>> c       = hamil.eig_vec.conj().T @ psi0
        >>> ldos    = hamil.statistical.ldos(c)
        >>> chi_F   = hamil.statistical.fidelity_susceptibility(V_proj)
        >>> r_mean  = hamil.statistical.mean_level_spacing_ratio()
        """)


def get_statistical_module(hamiltonian) -> StatisticalModule:
    """Factory function to create statistical module."""
    return StatisticalModule(hamiltonian)


# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------
