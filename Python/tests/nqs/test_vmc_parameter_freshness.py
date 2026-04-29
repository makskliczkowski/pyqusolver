"""Regression tests for vmc parameter freshness."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
    from QES.Solver.MonteCarlo.vmc import VMCSampler
except ImportError:
    pytest.skip("Required QES VMC modules are not available.", allow_module_level=True)


def _shift_params(params, delta):
    """Shift params."""
    return jax.tree_util.tree_map(
        lambda x: x + jnp.asarray(delta, dtype=x.dtype),
        params,
    )


def test_vmc_uses_live_network_params_by_default():
    """Verify test vmc uses live network params by default."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=17,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(0),
        rng_k=jax.random.PRNGKey(0),
        numsamples=1,
        numchains=1,
        therm_steps=1,
        sweep_steps=1,
    )

    states = jnp.array([[1.0, -1.0, 1.0, -1.0]], dtype=jnp.float32)

    params_old = rbm.get_params()
    logprob_old = sampler.logprob(states)

    params_new = _shift_params(params_old, 0.125)
    rbm.set_params(params_new)

    logprob_live = sampler.logprob(states)
    logprob_explicit = sampler.logprob(states, net_params=params_new)

    assert not np.allclose(np.asarray(logprob_old), np.asarray(logprob_live))
    np.testing.assert_allclose(
        np.asarray(logprob_live),
        np.asarray(logprob_explicit),
        rtol=1e-6,
        atol=1e-6,
    )


def test_vmc_recomputes_cached_logprobs_after_parameter_change_during_sample():
    """Verify sampler log-amplitude cache follows changed network parameters."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=19,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(7),
        rng_k=jax.random.PRNGKey(7),
        numsamples=2,
        numchains=2,
        therm_steps=1,
        sweep_steps=1,
    )

    params_old = rbm.get_params()
    sampler.sample(parameters=params_old, num_samples=2, num_chains=2)
    assert sampler._logprobas is not None

    params_new = _shift_params(params_old, 0.25)
    rbm.set_params(params_new)
    final_state, _, _ = sampler.sample(parameters=params_new, num_samples=2, num_chains=2)
    final_states, final_logprobs = final_state
    direct_logprobs = sampler.logprob(final_states, net_params=params_new)

    np.testing.assert_allclose(
        np.asarray(final_logprobs),
        np.asarray(direct_logprobs),
        rtol=1e-5,
        atol=1e-5,
    )


def test_vmc_set_numsamples_rebuilds_runtime_cache_once():
    """Verify changing sample count follows one explicit sampler rebuild path."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=21,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(1),
        rng_k=jax.random.PRNGKey(1),
        numsamples=2,
        numchains=2,
        therm_steps=2,
        sweep_steps=3,
    )

    _ = sampler.get_sampler()
    assert sampler._jit_cache

    sampler.set_numsamples(5)

    assert sampler.numsamples == 5
    assert sampler._jit_cache == {}
    assert sampler._needs_thermalization is True
    assert sampler._total_sample_updates_per_chain == 5 * sampler.updates_per_sample * sampler.numchains


def test_vmc_set_update_num_invalidates_cached_sampler():
    """Verify update-count changes rebuild cached sampler state."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=23,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(2),
        rng_k=jax.random.PRNGKey(2),
        numsamples=2,
        numchains=2,
        therm_steps=1,
        sweep_steps=1,
    )

    _ = sampler.get_sampler()
    assert sampler._jit_cache

    sampler.set_update_num(2)

    assert sampler._numupd == 2
    assert sampler._jit_cache == {}
    assert sampler._needs_thermalization is True
    assert sampler._upd_fun is not sampler._org_upd_fun


def test_vmc_set_replicas_noops_for_same_schedule():
    """Verify reapplying the same PT ladder does not rebuild the sampler."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=29,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    betas = jnp.array([1.0, 0.6, 0.3], dtype=jnp.float32)
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(3),
        rng_k=jax.random.PRNGKey(3),
        numsamples=2,
        numchains=2,
        therm_steps=1,
        sweep_steps=1,
        pt_betas=betas,
    )

    _ = sampler.get_sampler()
    cached = dict(sampler._jit_cache)

    sampler.set_replicas(betas)

    assert sampler._jit_cache == cached


def test_vmc_set_pt_betas_alias_configures_replicas():
    """Verify train-facing PT beta alias configures the VMC replica ladder."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=31,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(4),
        rng_k=jax.random.PRNGKey(4),
        numsamples=1,
        numchains=2,
        therm_steps=1,
        sweep_steps=1,
        beta=1,
    )
    betas = [1.0, 0.5, 0.25]

    sampler.set_pt_betas(betas)

    assert sampler.is_pt is True
    assert sampler.n_replicas == 3
    np.testing.assert_allclose(np.asarray(sampler.pt_betas), np.asarray(betas, dtype=np.float32))
    assert sampler.states.shape == (3, 2, 4)


def test_vmc_generated_pt_ladder_starts_from_sampler_beta():
    """Verify automatic PT ladders use the sampler's physical beta."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=37,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(5),
        rng_k=jax.random.PRNGKey(5),
        numsamples=1,
        numchains=1,
        therm_steps=1,
        sweep_steps=1,
        beta=0.8,
    )

    sampler.set_replicas(None, n_replicas=3)

    betas = np.asarray(sampler.pt_betas)
    assert betas[0] == pytest.approx(0.8)
    assert betas[-1] == pytest.approx(0.08)


def test_vmc_pt_uniform_weights_detect_physical_beta_sector():
    """Verify PT samples can use uniform estimator weights at beta=1, mu=2."""
    rbm = RBM(
        input_shape=(4,),
        n_hidden=3,
        seed=41,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(6),
        rng_k=jax.random.PRNGKey(6),
        numsamples=1,
        numchains=1,
        therm_steps=1,
        sweep_steps=1,
        pt_betas=jnp.array([1.0, 0.5], dtype=jnp.float32),
    )

    assert sampler.has_uniform_weights is True
