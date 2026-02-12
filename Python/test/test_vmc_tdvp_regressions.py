import os
import types

import numpy as np
import pytest

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from QES.Solver.MonteCarlo.vmc import VMCSampler
from QES.NQS.src.tdvp import TDVP, TDVPTimes
from QES.general_python.algebra import solvers


def test_replica_exchange_preserves_beta_permutation_per_chain():
    base_betas = jnp.array([1.0, 0.7, 0.4, 0.2], dtype=jnp.float64)
    betas = jnp.tile(base_betas[:, None], (1, 4))

    key = jax.random.PRNGKey(0)
    log_psi = jax.random.normal(jax.random.PRNGKey(777), shape=betas.shape)

    out = VMCSampler._replica_exchange_kernel_jax_betas(log_psi, betas, key, 2.0)

    out_np = np.asarray(out)
    sorted_ref = np.sort(np.asarray(base_betas))
    for chain_idx in range(out_np.shape[1]):
        assert np.allclose(np.sort(out_np[:, chain_idx]), sorted_ref, atol=1e-12)


def test_numpy_sample_uses_resolved_params_and_net_callable():
    sampler = VMCSampler.__new__(VMCSampler)

    sampler._numsamples = 5
    sampler._numchains = 2
    sampler._therm_steps = 3
    sampler._isjax = False
    sampler._states = np.zeros((2, 3), dtype=np.float64)
    sampler._logprobas = None
    sampler._num_proposed = np.zeros((2,), dtype=np.int64)
    sampler._num_accepted = np.zeros((2,), dtype=np.int64)
    sampler._mu = 2.0
    sampler._logprob_fact = 2.0
    sampler._parameters = {"w": 1.23}
    sampler._net = object()
    sampler._logger = None

    def net_callable(params, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.sum(x, axis=1)

    sampler._net_callable = net_callable

    calls = {"logprob_params": [], "generate_params": None, "generate_num_samples": None}

    def fake_logprob(x, net_callable=None, net_params=None):
        calls["logprob_params"].append(net_params)
        return net_callable(net_params, x)

    sampler.logprob = fake_logprob

    def fake_generate_samples_np(params, num_samples, multiple_of=1):
        calls["generate_params"] = params
        calls["generate_num_samples"] = num_samples

        states = np.ones((2, 3), dtype=np.float64)
        logprobas = np.zeros((2,), dtype=np.float64)
        num_proposed = np.array([11, 12], dtype=np.int64)
        num_accepted = np.array([5, 6], dtype=np.int64)
        configs = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)

        return (states, logprobas, num_proposed, num_accepted), configs

    sampler._generate_samples_np = fake_generate_samples_np

    (_, _), (configs, configs_log_ansatz), probs = VMCSampler.sample(sampler)

    assert calls["generate_params"] == sampler._parameters
    assert calls["generate_num_samples"] == sampler._numsamples
    assert all(p == sampler._parameters for p in calls["logprob_params"])

    assert configs.shape == (10, 3)
    assert configs_log_ansatz.shape == (10,)
    assert probs.shape == (10,)


def test_tdvp_solve_updates_x0_and_persists_theta0_dot():
    tdvp = TDVP.__new__(TDVP)

    tdvp.use_sr = True
    tdvp.rhs_prefactor = 1.0
    tdvp.grad_clip = None
    tdvp.use_minsr = False
    tdvp.backend = np
    tdvp.use_timing = False
    tdvp.timings = TDVPTimes()
    tdvp.logger = None

    tdvp._n_samples = 4
    tdvp._theta0 = 0.0
    tdvp._theta0_dot = 0.0
    tdvp._x0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tdvp._solution = types.SimpleNamespace(x=np.array([1.0, 2.0, 3.0], dtype=np.float64))
    tdvp.sr_solve_lin_fn = object()

    f0 = np.array([0.5, -0.5], dtype=np.float64)
    loss_c = np.zeros((4,), dtype=np.float64)
    vd_c = np.ones((4, 2), dtype=np.float64)
    vd_m = np.zeros((2,), dtype=np.float64)

    def fake_get_tdvp_standard(self, e_loc, log_deriv, **kwargs):
        self._e_local_mean = 1.5
        self._e_local_std = 0.2
        return f0, None, (loss_c, vd_c, None, vd_m)

    tdvp.get_tdvp_standard = types.MethodType(fake_get_tdvp_standard, tdvp)

    seen = {}

    def fake_solve_choice(self, vec_b, solve_func, mat_O=None, mat_a=None):
        seen["x0_shape"] = tuple(self._x0.shape)
        seen["vec_b_shape"] = tuple(vec_b.shape)
        return solvers.SolverResult(
            x=np.array([0.1, 0.2], dtype=np.float64),
            iterations=1,
            residual_norm=0.0,
            converged=True,
        )

    tdvp._solve_choice = types.MethodType(fake_solve_choice, tdvp)

    _, theta0_dot = TDVP.solve(tdvp, np.zeros((4,), dtype=np.float64), np.zeros((4, 2)))

    expected_theta0_dot = -1j * 1.5

    assert seen["x0_shape"] == seen["vec_b_shape"] == (2,)
    assert np.allclose(tdvp._x0, np.zeros((2,), dtype=np.float64))
    assert np.isclose(theta0_dot, expected_theta0_dot)
    assert np.isclose(tdvp._theta0_dot, expected_theta0_dot)
