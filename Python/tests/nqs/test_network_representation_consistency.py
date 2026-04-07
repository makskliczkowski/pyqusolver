import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
	from QES.general_python.ml.net_impl.networks.net_mlp import MLP
	from QES.general_python.ml.net_impl.networks.net_cnn import CNN
	from QES.general_python.ml.net_impl.networks.net_rbm import RBM
	from QES.general_python.ml.net_impl.networks.net_mps import MPS
	from QES.general_python.ml.net_impl.networks.net_pp import PairProduct
except ImportError:
	pytest.skip("Required network modules are not available.", allow_module_level=True)


def _binary_to_spin_pm_half(x):
	return x.astype(jnp.float32) - jnp.asarray(0.5, dtype=jnp.float32)


def test_mlp_binary_and_spin_inputs_match_when_configured_explicitly():
	states_bin = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.float32)
	states_spin = _binary_to_spin_pm_half(states_bin)

	net_bin = MLP(
		input_shape=(4,),
		hidden_dims=(8,),
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=7,
		transform_input=True,
		input_spin=False,
		input_value=1.0,
	)
	net_spin = MLP(
		input_shape=(4,),
		hidden_dims=(8,),
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=7,
		transform_input=True,
		input_spin=True,
		input_value=0.5,
	)
	net_spin.set_params(net_bin.get_params())

	np.testing.assert_allclose(np.asarray(net_bin(states_bin)), np.asarray(net_spin(states_spin)), rtol=1e-6, atol=1e-6)
	assert net_bin.get_nqs_metadata()["native_representation"] == "binary_01"
	assert net_spin.get_nqs_metadata()["native_representation"] == "spin_pm"


def test_cnn_binary_and_spin_inputs_match_when_configured_explicitly():
	states_bin = jnp.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=jnp.float32)
	states_spin = _binary_to_spin_pm_half(states_bin)

	net_bin = CNN(
		input_shape=(4,),
		reshape_dims=(4,),
		features=(2,),
		kernel_sizes=(1,),
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=11,
		transform_input=True,
		input_spin=False,
		input_value=1.0,
	)
	net_spin = CNN(
		input_shape=(4,),
		reshape_dims=(4,),
		features=(2,),
		kernel_sizes=(1,),
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=11,
		transform_input=True,
		input_spin=True,
		input_value=0.5,
	)
	net_spin.set_params(net_bin.get_params())

	np.testing.assert_allclose(np.asarray(net_bin(states_bin)), np.asarray(net_spin(states_spin)), rtol=1e-6, atol=1e-6)
	assert net_bin.get_nqs_metadata()["native_representation"] == "binary_01"
	assert net_spin.get_nqs_metadata()["native_representation"] == "spin_pm"


def test_rbm_binary_and_spin_inputs_match_when_configured_explicitly():
	states_bin = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.float32)
	states_spin = _binary_to_spin_pm_half(states_bin)

	net_bin = RBM(
		input_shape=(4,),
		n_hidden=3,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=3,
		in_activation=True,
		input_spin=False,
		input_value=1.0,
	)
	net_spin = RBM(
		input_shape=(4,),
		n_hidden=3,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=3,
		in_activation=True,
		input_spin=True,
		input_value=0.5,
	)
	net_spin.set_params(net_bin.get_params())

	np.testing.assert_allclose(np.asarray(net_bin(states_bin)), np.asarray(net_spin(states_spin)), rtol=1e-6, atol=1e-6)
	assert net_bin.get_nqs_metadata()["native_representation"] == "binary_01"
	assert net_spin.get_nqs_metadata()["native_representation"] == "spin_pm"


def test_mps_and_pair_product_metadata_follow_input_convention():
	mps_bin = MPS(input_shape=(4,), bond_dim=2, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=False, input_value=1.0)
	mps_spin = MPS(input_shape=(4,), bond_dim=2, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=True, input_value=0.5)
	pp_bin = PairProduct(input_shape=(4,), use_rbm=False, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=False, input_value=1.0)
	pp_spin = PairProduct(input_shape=(4,), use_rbm=False, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=True, input_value=0.5)

	assert mps_bin.get_nqs_metadata()["native_representation"] == "binary_01"
	assert mps_spin.get_nqs_metadata()["native_representation"] == "spin_pm"
	assert pp_bin.get_nqs_metadata()["native_representation"] == "binary_01"
	assert pp_spin.get_nqs_metadata()["native_representation"] == "spin_pm"
