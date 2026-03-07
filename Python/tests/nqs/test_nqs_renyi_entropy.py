''' 
Tests for NQS.compute_renyi2, ensuring it correctly swaps subsystem sites, not sample rows.
'''

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jsp = pytest.importorskip("jax.scipy.special")

try:
    from QES.NQS.nqs import NQS
    from QES.NQS.src.nqs_entropy import compute_renyi_entropy
except ImportError:
    pytest.skip("QES.NQS.nqs module not found, skipping NQS tests.", allow_module_level=True)

#####################

SAMPLE_A = jnp.array(
    [
        [1, 1, 1, -1],
        [-1, -1, -1, -1],
        [-1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    dtype=jnp.float64,
)

SAMPLE_B = jnp.array(
    [
        [1, 1, 1, 1],
        [-1, 1, 1, -1],
        [-1, 1, 1, -1],
        [1, 1, 1, -1],
    ],
    dtype=jnp.float64,
)

#####################

class DummyNQS:
    """Minimal stub for testing NQS.compute_renyi2."""

    backend_str = "jax"

    def __init__(self, s1, s2):
        self._s1        = jnp.asarray(s1, dtype=jnp.float64)
        self._s2        = jnp.asarray(s2, dtype=jnp.float64)
        self.nvisible   = int(self._s1.shape[1])
        self._counter   = 0

    @staticmethod
    def ansatz(states):
        s = jnp.asarray(states, dtype=jnp.float64)
        if s.ndim == 1:
            s = s.reshape(1, -1)
            
        # Nonlinear/cross-site structure so subsystem swaps change amplitudes.
        return (
            0.7   * s[:, 0] * s[:, 2]
            + 1.1 * s[:, 1] * s[:, 2]
            - 0.4 * s[:, 2] * s[:, 3]
            + 0.3 * s[:, 0] * s[:, 1]
            + 0.8 * s[:, 0] * s[:, 3]
        )

    def sample(self, num_samples=None, num_chains=None, reset=False, **kwargs):
        if self._counter % 2 == 0:
            states = self._s1
        else:
            states = self._s2
        self._counter += 1

        log_psi = self.ansatz(states)
        probs = jnp.ones(states.shape[0], dtype=jnp.float64)
        return (None, None), (states, log_psi), probs


def _manual_s2(dummy, region):
    s1 = dummy._s1
    s2 = dummy._s2
    region = jnp.asarray(region, dtype=jnp.int32)

    s1s = s1.at[:, region].set(s2[:, region])
    s2s = s2.at[:, region].set(s1[:, region])

    log_ratio = (dummy.ansatz(s1s) + dummy.ansatz(s2s)) - (dummy.ansatz(s1) + dummy.ansatz(s2))
    log_swap = jsp.logsumexp(log_ratio) - jnp.log(log_ratio.shape[0])
    swap_mean = jnp.exp(log_swap)
    return float(jnp.real(-jnp.log(jnp.maximum(swap_mean, 1e-15))))


def _wrong_row_swap_s2(dummy, region):
    s1 = dummy._s1
    s2 = dummy._s2
    region = jnp.asarray(region, dtype=jnp.int32)

    # This is the old incorrect behavior (swapping sample rows, not subsystem sites).
    s1s = s1.at[region].set(s2[region])
    s2s = s2.at[region].set(s1[region])

    log_ratio = (dummy.ansatz(s1s) + dummy.ansatz(s2s)) - (dummy.ansatz(s1) + dummy.ansatz(s2))
    log_swap = jsp.logsumexp(log_ratio) - jnp.log(log_ratio.shape[0])
    swap_mean = jnp.exp(log_swap)
    return float(jnp.real(-jnp.log(jnp.maximum(swap_mean, 1e-15))))


def test_compute_renyi2_swaps_sites_not_rows():
    dummy = DummyNQS(SAMPLE_A, SAMPLE_B)
    region = [0, 1]

    expected = _manual_s2(dummy, region)
    wrong = _wrong_row_swap_s2(dummy, region)
    got = NQS.compute_renyi2(dummy, region=region, num_samples=4, num_chains=2)

    assert np.isclose(got, expected, rtol=1e-8, atol=1e-8)
    assert not np.isclose(got, wrong, rtol=1e-6, atol=1e-6)


def test_compute_renyi2_error_outputs_are_finite():
    dummy = DummyNQS(SAMPLE_A, SAMPLE_B)

    s2_mean, s2_err = NQS.compute_renyi2(
        dummy,
        region=[0, 1],
        num_samples=4,
        num_chains=2,
        return_error=True,
    )
    swap_mean, swap_err = NQS.compute_renyi2(
        dummy,
        region=[0, 1],
        num_samples=4,
        num_chains=2,
        return_swap_mean=True,
        return_error=True,
    )

    assert np.isfinite(s2_mean)
    assert np.isfinite(s2_err)
    assert np.isfinite(swap_mean)
    assert np.isfinite(swap_err)
    assert s2_err >= 0.0
    assert swap_err >= 0.0
    assert swap_mean > 0.0


def test_compute_mc_stats_returns_chain_diagnostics_keys():
    values = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.05, 0.95, 1.0])
    stats = NQS.compute_mc_stats(values, num_chains=2)

    for key in ["mean", "std", "stderr", "ess", "tau_int", "r_hat", "n_samples", "num_chains"]:
        assert key in stats

    assert stats["n_samples"] == values.size
    assert stats["num_chains"] == 2
    assert stats["ess"] > 0.0
    assert stats["stderr"] >= 0.0

class ProductStateDummyNQS:
    """Dummy NQS with constant log-amplitude (product state, zero entanglement)."""

    backend_str = "jax"

    def __init__(self, samples):
        self._samples = jnp.asarray(samples, dtype=jnp.float64)
        self.nvisible = int(self._samples.shape[1])

    @staticmethod
    def ansatz(states):
        s = jnp.asarray(states)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        return jnp.zeros((s.shape[0],), dtype=jnp.float64)

    def sample(self, num_samples=None, num_chains=None, reset=False):
        states = self._samples
        log_psi = self.ansatz(states)
        probs = jnp.ones(states.shape[0], dtype=jnp.float64)
        return (None, None), (states, log_psi), probs


class LargeScaleDummyNQS(DummyNQS):
    """Dummy NQS with amplified logits to stress numerical stability in q>2 errors."""

    @staticmethod
    def ansatz(states):
        s = jnp.asarray(states, dtype=jnp.float64)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        return 80.0 * (
            1.7 * s[:, 0] * s[:, 2]
            - 1.3 * s[:, 1] * s[:, 2]
            + 1.1 * s[:, 2] * s[:, 3]
            - 0.9 * s[:, 0] * s[:, 1]
            + 0.6 * s[:, 0] * s[:, 3]
        )


def test_compute_renyi2_subsystem_entropy_product_state_zero():
    """Subsystem Renyi-2 entropy should be zero for an unentangled product state."""
    samples = jnp.array(
        [
            [1, -1, 1, -1],
            [-1, 1, -1, 1],
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
        ],
        dtype=jnp.float64,
    )
    dummy = ProductStateDummyNQS(samples)

    s2, s2_err = NQS.compute_renyi2(
        dummy,
        region=[0, 1],
        num_samples=4,
        num_chains=2,
        return_error=True,
    )

    assert np.isclose(s2, 0.0, atol=1e-12)
    assert np.isclose(s2_err, 0.0, atol=1e-12)


def test_compute_renyi3_error_outputs_remain_finite_for_large_logratios():
    dummy = LargeScaleDummyNQS(SAMPLE_A, SAMPLE_B)
    s3, s3_err = compute_renyi_entropy(
        dummy,
        region=[0, 1],
        q=3,
        num_samples=4,
        num_chains=2,
        return_error=True,
    )
    raw = compute_renyi_entropy(
        dummy,
        region=[0, 1],
        q=3,
        num_samples=4,
        num_chains=2,
        return_raw=True,
    )

    assert np.isfinite(s3)
    assert np.isfinite(s3_err)
    assert s3_err >= 0.0
    assert np.isfinite(raw["trace_rho_q"])
    assert np.isfinite(raw["trace_err"])
