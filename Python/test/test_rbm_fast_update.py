import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from QES.general_python.ml.net_impl.networks            import net_rbm
    from QES.general_python.ml.net_impl.networks.net_rbm    import RBM
    from QES.Solver.MonteCarlo.vmc                          import VMCSampler
    from QES.Solver.MonteCarlo.updates                      import spin_jax
except ImportError:
    raise ImportError("Required modules for RBM fast update tests are not available. Please ensure JAX and the relevant QES modules are installed.")

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

def _flip_state_binary(state, indices, valid_mask=None):
    out = np.array(state, copy=True)
    idx = np.asarray(indices, dtype=np.int32).reshape(-1)
    if valid_mask is None:
        valid = np.ones_like(idx, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    for i, ok in zip(idx, valid):
        if not ok:
            continue
        out[i] = 1.0 - out[i]
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

def test_rbm_log_psi_delta_binary_with_input_activation_matches_direct_eval():
    old_spin_mode = net_rbm.BACKEND_DEF_SPIN
    old_repr = net_rbm.BACKEND_REPR
    try:
        net_rbm.BACKEND_DEF_SPIN    = False
        net_rbm.BACKEND_REPR        = 1.0
        jax.clear_caches()

        rbm = RBM(
            input_shape=(6,),
            n_hidden=5,
            seed=9,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            in_activation=True,
        )
        params = rbm.get_params()

        states = jnp.array(
            [
                [0, 1, 0, 1, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 1],
            ],
            dtype=jnp.float32,
        )
        indices = jnp.array([[0, 2, 2], [3, 5, 5], [1, 4, 4]], dtype=jnp.int32)
        valid_mask = jnp.array([[True, True, False], [True, True, False], [True, True, False]])

        current = rbm(states)
        cache = rbm.init_log_psi_delta_cache(params, states)
        fast_delta, cache_new = rbm.log_psi_delta(params, current, states, (indices, valid_mask), cache)

        proposed_np = np.stack(
            [
                _flip_state_binary(states[i], indices[i], valid_mask[i])
                for i in range(states.shape[0])
            ],
            axis=0,
        )
        proposed = jnp.asarray(proposed_np, dtype=states.dtype)
        direct_delta = rbm(proposed) - current
        expected_cache = rbm.init_log_psi_delta_cache(params, proposed)

        np.testing.assert_allclose(np.asarray(fast_delta), np.asarray(direct_delta), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(cache_new), np.asarray(expected_cache), rtol=1e-6, atol=1e-6)
    finally:
        net_rbm.BACKEND_DEF_SPIN    = old_spin_mode
        net_rbm.BACKEND_REPR        = old_repr
        jax.clear_caches()


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


def test_spin_jax_local_flip_supports_binary_representation():
    old_spin_mode = spin_jax.BACKEND_DEF_SPIN
    old_repr = spin_jax.BACKEND_REPR
    try:
        spin_jax.BACKEND_DEF_SPIN = False
        spin_jax.BACKEND_REPR = 1.0
        jax.clear_caches()

        state = jnp.array([0, 1, 0, 1], dtype=jnp.float32)
        new_state, idx = spin_jax.propose_local_flip_with_info(state, jax.random.PRNGKey(123))
        i = int(np.asarray(idx)[0])

        expected = np.asarray(state).copy()
        expected[i] = 1.0 - expected[i]
        np.testing.assert_allclose(np.asarray(new_state), expected, rtol=1e-7, atol=1e-7)
    finally:
        spin_jax.BACKEND_DEF_SPIN = old_spin_mode
        spin_jax.BACKEND_REPR = old_repr
        jax.clear_caches()


def test_spin_jax_multi_flip_supports_nonunit_binary_representation():
    old_spin_mode = spin_jax.BACKEND_DEF_SPIN
    old_repr = spin_jax.BACKEND_REPR
    try:
        spin_jax.BACKEND_DEF_SPIN = False
        spin_jax.BACKEND_REPR = 0.5
        jax.clear_caches()

        state = jnp.array([0.0, 0.5, 0.0, 0.5], dtype=jnp.float32)
        new_state, idx = spin_jax.propose_multi_flip_with_info(
            state, jax.random.PRNGKey(5), n_flip=2
        )

        expected = np.asarray(state).copy()
        for i in np.asarray(idx):
            expected[int(i)] = 0.5 - expected[int(i)]
        np.testing.assert_allclose(np.asarray(new_state), expected, rtol=1e-7, atol=1e-7)
    finally:
        spin_jax.BACKEND_DEF_SPIN = old_spin_mode
        spin_jax.BACKEND_REPR = old_repr
        jax.clear_caches()


def test_spin_jax_global_flip_supports_nonunit_binary_representation():
    old_spin_mode   = spin_jax.BACKEND_DEF_SPIN
    old_repr        = spin_jax.BACKEND_REPR
    try:
        spin_jax.BACKEND_DEF_SPIN   = False
        spin_jax.BACKEND_REPR       = 0.5
        jax.clear_caches()

        state       = jnp.array([0.0, 0.5, 0.0, 0.5], dtype=jnp.float32)
        patterns    = jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)
        new_state, safe_idx = spin_jax.propose_global_flip_with_info(state, jax.random.PRNGKey(7), patterns)

        expected    = np.asarray(state).copy()
        for i in np.asarray(safe_idx):
            expected[int(i)] = 0.5 - expected[int(i)]
        np.testing.assert_allclose(np.asarray(new_state), expected, rtol=1e-7, atol=1e-7)
    finally:
        spin_jax.BACKEND_DEF_SPIN   = old_spin_mode
        spin_jax.BACKEND_REPR       = old_repr
        jax.clear_caches()

# --------------------------------
#! EOF
# --------------------------------