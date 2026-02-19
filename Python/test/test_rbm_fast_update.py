import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from QES.general_python.ml.net_impl.networks.net_rbm import RBM
from QES.Solver.MonteCarlo.vmc import VMCSampler


def _flip_state(state, indices, valid_mask=None):
    out = np.array(state, copy=True)
    idx = np.asarray(indices, dtype=np.int32).reshape(-1)
    if valid_mask is None:
        valid = np.ones_like(idx, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    for i, ok in zip(idx, valid):
        if not ok:
            continue
        out[i] = -out[i]
    return out


def test_rbm_log_psi_delta_matches_direct_eval_single_flip():
    rbm = RBM(input_shape=(6,), n_hidden=4, seed=123, dtype=jnp.float32, param_dtype=jnp.float32)
    params = rbm.get_params()

    state = jnp.array([1, -1, 1, -1, 1, -1], dtype=jnp.float32)
    update_info = jnp.array([2], dtype=jnp.int32)

    current = rbm(state)
    proposed = state.at[2].multiply(-1)
    direct_delta = rbm(proposed) - current
    fast_delta = rbm.log_psi_delta(params, current, state, update_info)

    np.testing.assert_allclose(np.asarray(fast_delta), np.asarray(direct_delta), rtol=1e-6, atol=1e-6)


def test_rbm_log_psi_delta_cache_and_masked_updates_match_direct_eval():
    rbm = RBM(input_shape=(6,), n_hidden=5, seed=7, dtype=jnp.float32, param_dtype=jnp.float32)
    params = rbm.get_params()

    states = jnp.array(
        [
            [1, -1, 1, -1, 1, -1],
            [-1, -1, 1, 1, -1, 1],
            [1, 1, -1, -1, 1, -1],
        ],
        dtype=jnp.float32,
    )
    indices = jnp.array([[0, 1, 1], [2, 2, 4], [3, 5, 0]], dtype=jnp.int32)
    valid_mask = jnp.array([[True, True, False], [True, True, True], [True, False, False]])

    current = rbm(states)
    cache = rbm.init_log_psi_delta_cache(params, states)
    fast_delta, cache_new = rbm.log_psi_delta(params, current, states, (indices, valid_mask), cache)

    proposed_np = np.stack(
        [
            _flip_state(states[i], indices[i], valid_mask[i])
            for i in range(states.shape[0])
        ],
        axis=0,
    )
    proposed = jnp.asarray(proposed_np, dtype=states.dtype)
    direct_delta = rbm(proposed) - current
    expected_cache = rbm.init_log_psi_delta_cache(params, proposed)

    np.testing.assert_allclose(np.asarray(fast_delta), np.asarray(direct_delta), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(cache_new), np.asarray(expected_cache), rtol=1e-6, atol=1e-6)


def test_rbm_fast_update_supported_rules_are_flip_only():
    rbm = RBM(input_shape=(4,), n_hidden=2, seed=0, dtype=jnp.float32, param_dtype=jnp.float32)
    assert rbm.fast_update_supported_rules == {"LOCAL", "MULTI_FLIP", "BOND_FLIP", "WORM"}


def test_vmc_uses_rbm_fast_update_and_samples():
    rbm = RBM(input_shape=(4,), n_hidden=3, seed=11, dtype=jnp.float32, param_dtype=jnp.float32)
    sampler = VMCSampler(
        net=rbm,
        shape=(4,),
        backend="jax",
        rng=np.random.default_rng(0),
        rng_k=jax.random.PRNGKey(0),
        numsamples=2,
        numchains=2,
        therm_steps=1,
        sweep_steps=1,
    )

    assert sampler._log_psi_delta_fun is not None
    assert sampler._log_psi_delta_supports_cache is True
    assert "with_info" in getattr(sampler._local_upd_fun, "__name__", "")

    final_state, samples, probs = sampler.sample(
        parameters=rbm.get_params(), num_samples=2, num_chains=2
    )
    assert isinstance(final_state, tuple)
    assert isinstance(samples, tuple)
    assert probs.shape[0] == 4
