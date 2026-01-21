
import numpy as np
import pytest
from QES.Solver.MonteCarlo.diagnostics import compute_autocorr_time, compute_ess, compute_rhat

def test_autocorr_time_white_noise():
    # White noise should have tau approx 1
    np.random.seed(42)
    x = np.random.randn(10000)
    tau = compute_autocorr_time(x)
    assert 0.8 < tau < 1.2

def test_autocorr_time_correlated():
    # Correlated process (AR1)
    # x_t = alpha * x_{t-1} + epsilon
    # Theoretical tau = (1 + alpha) / (1 - alpha)
    np.random.seed(42)
    alpha = 0.9
    n = 10000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = alpha * x[i-1] + np.random.randn()

    tau = compute_autocorr_time(x)
    expected_tau = (1 + alpha) / (1 - alpha) # approx 19
    # Sokal's estimator usually good within some error
    assert expected_tau * 0.7 < tau < expected_tau * 1.3

def test_ess():
    np.random.seed(42)
    x = np.random.randn(1000)
    ess = compute_ess(x)
    # ESS should be close to N for white noise
    assert 800 < ess < 1200

def test_rhat_converged():
    # Multiple chains sampling from same distribution
    np.random.seed(42)
    chains = np.random.randn(4, 1000)
    rhat = compute_rhat(chains)
    assert 0.99 < rhat < 1.05

def test_rhat_diverged():
    # Chains with different means
    np.random.seed(42)
    c1 = np.random.randn(1000)
    c2 = np.random.randn(1000) + 10.0 # Shifted mean
    chains = np.stack([c1, c2])
    rhat = compute_rhat(chains)
    assert rhat > 1.5
