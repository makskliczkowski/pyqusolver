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

# ------------------------------------------------------------------------------
# Internal utilities for handling samples and estimators
# ------------------------------------------------------------------------------

def _ensure_numpy(x):
    """Convert JAX array to NumPy array if needed."""
    if hasattr(x, "__array__") or (JAX_AVAILABLE and isinstance(x, jax.Array)):
        return np.array(x)
    return x


def _move_sample_axis_first(samples, sample_axis):
    """Return samples with the sample axis moved to the leading position."""
    arr = _ensure_numpy(samples)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        raise ValueError("samples must have at least one sample axis.")
    axis = int(sample_axis)
    return np.moveaxis(arr, axis, 0)


def _apply_estimator(estimator, samples):
    """Apply an estimator and return a NumPy array."""
    return np.asarray(_ensure_numpy(estimator(samples)))


# ------------------------------------------------------------------------------
# Main diagnostic functions
# ------------------------------------------------------------------------------

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


def jackknife_estimate(samples, estimator=np.mean, sample_axis=0, return_replicates=False):
    """
    Compute a generic delete-one jackknife estimate.

    Parameters:
        samples (array-like): Input samples with one distinguished sample axis.
        estimator (callable): Function applied to the sample array with the
            sample axis moved to axis 0.
        sample_axis (int): Axis indexing independent samples.
        return_replicates (bool): Whether to include the jackknife replicates.

    Returns:
        dict: Contains `estimate`, `stderr`, `bias`, and `replicates` when requested.
    """
    data    = _move_sample_axis_first(samples, sample_axis)
    n       = data.shape[0]
    if n < 2:
        raise ValueError("jackknife_estimate requires at least two samples.")

    full_estimate   = _apply_estimator(estimator, data)
    replicates      = []
    for idx in range(n):
        reduced = np.concatenate((data[:idx], data[idx + 1 :]), axis=0)
        replicates.append(_apply_estimator(estimator, reduced))
    replicates = np.stack(replicates, axis=0)

    mean_replicate  = np.mean(replicates, axis=0)
    bias            = (n - 1) * (mean_replicate - full_estimate)
    variance        = (n - 1) * np.mean((replicates - mean_replicate) ** 2, axis=0)
    stderr          = np.sqrt(variance)

    result          = {
                        "estimate"  : full_estimate.item() if full_estimate.ndim == 0 else full_estimate,
                        "stderr"    : stderr.item() if np.ndim(stderr) == 0 else stderr,
                        "bias"      : bias.item() if np.ndim(bias) == 0 else bias,
                    }
    if return_replicates:
        result["replicates"] = replicates
    return result


def bootstrap_estimate(
    samples,
    estimator=np.mean,
    *,
    n_resamples=1000,
    sample_axis=0,
    confidence_level=0.95,
    rng=None,
    return_replicates=False,
):
    """
    Compute a generic nonparametric bootstrap estimate.

    Parameters:
        samples (array-like): Input samples with one distinguished sample axis.
        estimator (callable): Function applied to the resampled array with the
            sample axis moved to axis 0.
        n_resamples (int): Number of bootstrap resamples.
        sample_axis (int): Axis indexing independent samples.
        confidence_level (float): Central percentile interval level.
        rng (None|int|np.random.Generator): RNG control for reproducibility.
        return_replicates (bool): Whether to include bootstrap replicates.

    Returns:
        dict: Contains `estimate`, `stderr`, `bias`, `confidence_interval`,
        and `replicates` when requested.
    """
    data = _move_sample_axis_first(samples, sample_axis)
    n = data.shape[0]
    if n < 1:
        raise ValueError("bootstrap_estimate requires at least one sample.")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be at least 1.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must lie in (0, 1).")

    if isinstance(rng, np.random.Generator):
        generator = rng
    else:
        generator = np.random.default_rng(rng)

    full_estimate = _apply_estimator(estimator, data)
    replicates = []
    for _ in range(int(n_resamples)):
        indices = generator.integers(0, n, size=n)
        replicates.append(_apply_estimator(estimator, data[indices]))
    replicates = np.stack(replicates, axis=0)

    mean_replicate  = np.mean(replicates, axis=0)
    stderr          = np.std(replicates, axis=0, ddof=1 if replicates.shape[0] > 1 else 0)
    bias            = mean_replicate - full_estimate
    alpha           = 0.5 * (1.0 - confidence_level)
    ci_low          = np.quantile(replicates, alpha, axis=0)
    ci_high         = np.quantile(replicates, 1.0 - alpha, axis=0)

    result          = {
                        "estimate"  : full_estimate.item() if full_estimate.ndim == 0 else full_estimate,
                        "stderr"    : stderr.item() if np.ndim(stderr) == 0 else stderr,
                        "bias"      : bias.item() if np.ndim(bias) == 0 else bias,
                        "confidence_interval" : (
                            ci_low.item() if np.ndim(ci_low) == 0 else ci_low,
                            ci_high.item() if np.ndim(ci_high) == 0 else ci_high,
                        ),
                        "confidence_level"    : float(confidence_level),
        "n_resamples": int(n_resamples),
    }
    if return_replicates:
        result["replicates"] = replicates
    return result
