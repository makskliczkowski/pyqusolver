"""
Diagnostics for Monte Carlo sampling.
Includes Effective Sample Size (ESS), Autocorrelation Time, and R-hat statistics.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


def _ensure_numpy(x):
    """Convert JAX array to NumPy array if needed."""
    if hasattr(x, "__array__") or (JAX_AVAILABLE and isinstance(x, jax.Array)):
        return np.array(x)
    return x


def autocorr_func_1d(x, norm=True):
    """
    Compute the autocorrelation function of a 1D time series.

    Parameters:
        x (np.ndarray): 1D array of samples.
        norm (bool): Whether to normalize so that acf[0] = 1.

    Returns:
        np.ndarray: Autocorrelation function.
    """
    x = _ensure_numpy(x)
    n = len(x)
    if n == 0:
        return np.array([1.0])

    mean = np.mean(x)
    var = np.var(x)

    if var < 1e-12:
        return np.ones(n)

    # Use FFT for efficient computation
    # Zero-pad to avoid circular correlation
    xp = x - mean
    fr = np.fft.fft(xp, n=2 * n)
    ac = np.fft.ifft(fr * np.conjugate(fr))[:n]
    ac = np.real(ac)

    # Normalize by variance * N (approximate)
    # Correct normalization for unbiased estimator is dividing by (N-k), but typical FFT method
    # divides by N.
    # We normalized by ac[0] anyway.

    if norm:
        return ac / ac[0]
    return ac


def compute_autocorr_time(x, c=5.0):
    """
    Compute the integrated autocorrelation time.
    Uses the windowing heuristic from Sokal (1989).

    tau = 1 + 2 * sum_{t=1}^M rho(t)

    The window M is chosen such that M >= c * tau.

    Parameters:
        x (np.ndarray): 1D array of samples.
        c (float): Window size factor.

    Returns:
        float: Estimated autocorrelation time.
    """
    acf = autocorr_func_1d(x)
    n = len(acf)

    # Constant/near-constant series: treat as effectively uncorrelated for
    # ESS purposes so ESS ~= N instead of collapsing to 1.
    if np.allclose(acf, 1.0, atol=1e-12):
        return 1.0

    # Find cutoff M
    tau = 1.0
    for m in range(1, n):
        tau += 2.0 * acf[m]
        if m >= c * tau:
            return tau

    # If standard window condition not met, return simplistic sum or just length/something
    # Fallback: sum until ACF goes negative or too small
    # But usually the loop above works for decent chains.
    # If it fails, it means tau is very large (comparable to N).
    return float(n)  # Worst case


def compute_ess(x):
    """
    Compute Effective Sample Size (ESS).

    ESS = N / tau

    Parameters:
        x (np.ndarray): 1D array of samples.

    Returns:
        float: Effective sample size.
    """
    x = _ensure_numpy(x)
    n = len(x)
    tau = compute_autocorr_time(x)
    return n / tau


def compute_rhat(chains):
    """
    Compute the Gelman-Rubin R-hat statistic for multiple chains.

    R-hat measures convergence by comparing within-chain variance to
    between-chain variance. R-hat close to 1.0 indicates convergence.
    Values > 1.1 usually indicate lack of convergence.

    Parameters:
        chains (np.ndarray): 2D array of shape (num_chains, num_samples).

    Returns:
        float: R-hat statistic.
    """
    chains = _ensure_numpy(chains)
    if chains.ndim != 2:
        raise ValueError("chains must be 2D array (num_chains, num_samples)")

    m, n = chains.shape  # m chains, n samples

    if m < 2:
        return np.nan  # Cannot compute R-hat with single chain
    if n < 2:
        return np.nan

    # Calculate means and variances
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)

    # Between-chain variance B
    B = n * np.var(chain_means, ddof=1)

    # Within-chain variance W
    W = np.mean(chain_vars)

    # Weighted variance estimate V_hat
    var_plus = ((n - 1) / n) * W + B / n

    if W < 1e-12:
        return 1.0  # If variances are zero, we are converged (or stuck)

    rhat = np.sqrt(var_plus / W)
    return rhat
