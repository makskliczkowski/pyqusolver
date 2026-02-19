import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from QES.general_python.ml.net_impl.networks import net_cnn, net_mlp, net_mps, net_pp, net_res
from QES.general_python.ml.net_impl.networks.net_cnn import CNN
from QES.general_python.ml.net_impl.networks.net_mlp import MLP
from QES.general_python.ml.net_impl.networks.net_res import ResNet
from QES.general_python.common import binary as Binary
from QES.Solver.MonteCarlo.updates import spin_numpy


def test_mlp_transform_input_respects_backend_representation():
    old_spin_mode   = net_mlp.BACKEND_DEF_SPIN
    old_repr        = net_mlp.BACKEND_REPR
    try:
        net_mlp.BACKEND_DEF_SPIN    = False
        net_mlp.BACKEND_REPR        = 0.5
        jax.clear_caches()

        mlp_map = MLP(
            input_shape=(4,),
            hidden_dims=(4,),
            activations="tanh",
            transform_input=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            seed=11,
        )
        mlp_raw = MLP(
            input_shape=(4,),
            hidden_dims=(4,),
            activations="tanh",
            transform_input=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            seed=12,
        )
        mlp_raw.set_params(mlp_map.get_params())

        x_binary    = jnp.array([[0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0, 0.5]], dtype=jnp.float32)
        x_pm1       = net_mlp._map_state_to_pm1(x_binary)

        np.testing.assert_allclose(np.asarray(mlp_map(x_binary)), np.asarray(mlp_raw(x_pm1)), rtol=1e-6, atol=1e-6)
    finally:
        net_mlp.BACKEND_DEF_SPIN    = old_spin_mode
        net_mlp.BACKEND_REPR        = old_repr
        jax.clear_caches()


def test_cnn_transform_input_respects_backend_representation():
    old_spin_mode = net_cnn.BACKEND_DEF_SPIN
    old_repr = net_cnn.BACKEND_REPR
    try:
        net_cnn.BACKEND_DEF_SPIN    = False
        net_cnn.BACKEND_REPR        = 0.5
        jax.clear_caches()

        cnn_map = CNN(
            input_shape=(4,),
            reshape_dims=(2, 2),
            features=(2,),
            kernel_sizes=1,
            strides=1,
            activations="tanh",
            transform_input=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            seed=21,
        )
        cnn_raw = CNN(
            input_shape=(4,),
            reshape_dims=(2, 2),
            features=(2,),
            kernel_sizes=1,
            strides=1,
            activations="tanh",
            transform_input=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            seed=22,
        )
        cnn_raw.set_params(cnn_map.get_params())

        x_binary = jnp.array([[0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0, 0.5]], dtype=jnp.float32)
        x_pm1 = net_cnn._map_state_to_pm1(x_binary)

        np.testing.assert_allclose(np.asarray(cnn_map(x_binary)), np.asarray(cnn_raw(x_pm1)), rtol=1e-6, atol=1e-6)
    finally:
        net_cnn.BACKEND_DEF_SPIN = old_spin_mode
        net_cnn.BACKEND_REPR = old_repr
        jax.clear_caches()


def test_resnet_map_input_to_spin_respects_backend_representation():
    old_spin_mode = net_res.BACKEND_DEF_SPIN
    old_repr = net_res.BACKEND_REPR
    try:
        net_res.BACKEND_DEF_SPIN = False
        net_res.BACKEND_REPR = 0.5
        jax.clear_caches()

        res_map = ResNet(
            input_shape=(4,),
            reshape_dims=(2, 2),
            features=2,
            depth=1,
            kernel_size=1,
            map_input_to_spin=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            seed=31,
        )
        res_raw = ResNet(
            input_shape=(4,),
            reshape_dims=(2, 2),
            features=2,
            depth=1,
            kernel_size=1,
            map_input_to_spin=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            seed=32,
        )
        res_raw.set_params(res_map.get_params())

        x_binary = jnp.array([[0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0, 0.5]], dtype=jnp.float32)
        x_pm1 = net_res._map_state_to_pm1(x_binary)

        np.testing.assert_allclose(np.asarray(res_map(x_binary)), np.asarray(res_raw(x_pm1)), rtol=1e-6, atol=1e-6)
    finally:
        net_res.BACKEND_DEF_SPIN = old_spin_mode
        net_res.BACKEND_REPR = old_repr
        jax.clear_caches()


def test_pairproduct_binary_indices_support_nonunit_representation():
    old_spin_mode = net_pp.BACKEND_DEF_SPIN
    old_repr = net_pp.BACKEND_REPR
    try:
        net_pp.BACKEND_DEF_SPIN = False
        net_pp.BACKEND_REPR = 0.5
        idx = net_pp._state_to_binary_index(jnp.array([0.0, 0.5, 0.0, 0.5], dtype=jnp.float32))
        np.testing.assert_array_equal(np.asarray(idx), np.array([0, 1, 0, 1], dtype=np.int32))
    finally:
        net_pp.BACKEND_DEF_SPIN = old_spin_mode
        net_pp.BACKEND_REPR = old_repr


def test_mps_binary_indices_support_nonunit_representation():
    old_spin_mode = net_mps.BACKEND_DEF_SPIN
    old_repr = net_mps.BACKEND_REPR
    try:
        net_mps.BACKEND_DEF_SPIN = False
        net_mps.BACKEND_REPR = 0.5
        idx = net_mps._state_to_binary_index(jnp.array([0.0, 0.5, 0.0, 0.5], dtype=jnp.float32))
        np.testing.assert_array_equal(np.asarray(idx), np.array([0, 1, 0, 1], dtype=np.int32))
    finally:
        net_mps.BACKEND_DEF_SPIN = old_spin_mode
        net_mps.BACKEND_REPR = old_repr


def test_spin_numpy_batched_local_flip_supports_nonspin_representation():
    old_spin_mode = Binary.BACKEND_DEF_SPIN
    old_repr = Binary.BACKEND_REPR
    try:
        Binary.set_global_defaults(0.5, False)
        states = np.array(
            [[0.0, 0.5, 0.0, 0.5], [0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5]],
            dtype=np.float32,
        )
        out = spin_numpy.propose_local_flip_np.py_func(states.copy(), np.random.default_rng(0))
        assert np.all((out == 0.0) | (out == 0.5))
    finally:
        Binary.set_global_defaults(old_repr, old_spin_mode)

# --------------------------------
#! EOF
# --------------------------------