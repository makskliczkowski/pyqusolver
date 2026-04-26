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
