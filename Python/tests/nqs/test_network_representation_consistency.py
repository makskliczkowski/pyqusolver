"""Regression tests for network representation consistency."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
	from QES.NQS.ansatze.registry import resolve_ansatz_request
	from QES.NQS.src.network.adapters import choose_nqs_network_adapter, infer_network_family
	from QES.NQS.src.network.representation import ModelRepresentationInfo, apply_nqs_representation_overrides
	from QES.general_python.ml.net_impl.networks.net_mlp import MLP
	from QES.general_python.ml.net_impl.networks.net_cnn import CNN
	from QES.general_python.ml.net_impl.networks.net_rbm import RBM
	from QES.general_python.ml.net_impl.networks.net_mps import MPS
	from QES.general_python.ml.net_impl.networks.net_pp import PairProduct
except ImportError:
	pytest.skip("Required network modules are not available.", allow_module_level=True)


def _binary_to_spin_pm_half(x):
	"""Convert binary representation to to spin pm half."""
	return x.astype(jnp.float32) - jnp.asarray(0.5, dtype=jnp.float32)


def _spin_half_representation(sampler_representation="spin_pm"):
	"""Helper for spin half representation."""
	return ModelRepresentationInfo(
		basis_type="real",
		local_space_type="spin-1/2",
		vector_encoding="Length-Ns array with signed entries in {-0.5,+0.5}.",
		integer_encoding="Binary computational basis.",
		sampler_representation=sampler_representation,
		network_representation=sampler_representation,
	)


def _adapter_native(net):
	"""Adapt native."""
	return choose_nqs_network_adapter(net, _spin_half_representation()).native_representation


def test_mlp_binary_and_spin_inputs_match_when_configured_explicitly():
	"""Verify test mlp binary and spin inputs match when configured explicitly."""
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
	assert _adapter_native(net_bin) == "binary_01"
	assert _adapter_native(net_spin) == "spin_pm"


def test_cnn_binary_and_spin_inputs_match_when_configured_explicitly():
	"""Verify test cnn binary and spin inputs match when configured explicitly."""
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
	assert _adapter_native(net_bin) == "binary_01"
	assert _adapter_native(net_spin) == "spin_pm"


def test_complex64_cnn_log_cosh_is_detected_as_holomorphic():
	"""Verify test complex64 cnn log cosh is detected as holomorphic."""
	net = CNN(
		input_shape=(8,),
		reshape_dims=(8,),
		features=(2,),
		kernel_sizes=(1,),
		dtype=jnp.complex64,
		param_dtype=jnp.complex64,
		seed=31,
	)

	assert net.is_holomorphic is True


def test_rbm_binary_and_spin_inputs_match_when_configured_explicitly():
	"""Verify test rbm binary and spin inputs match when configured explicitly."""
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
	assert _adapter_native(net_bin) == "binary_01"
	assert _adapter_native(net_spin) == "spin_pm"


def test_nqs_spin_half_signed_overrides_map_cnn_family_to_unit_spins():
	"""Verify test nqs spin half signed overrides map cnn family to unit spins."""
	representation = _spin_half_representation("spin_pm")
	states = jnp.array([[-0.5, 0.5, -0.5, 0.5], [0.5, 0.5, -0.5, -0.5]], dtype=jnp.float32)
	expected = jnp.array([[-1, 1, -1, 1], [1, 1, -1, -1]], dtype=jnp.float32)

	mlp_kwargs = apply_nqs_representation_overrides("mlp", representation, {})
	cnn_kwargs = apply_nqs_representation_overrides("cnn", representation, {})

	mlp = MLP(
		input_shape=(4,),
		hidden_dims=(8,),
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=23,
		**mlp_kwargs,
	)
	cnn = CNN(
		input_shape=(4,),
		reshape_dims=(4,),
		features=(2,),
		kernel_sizes=(1,),
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=23,
		**cnn_kwargs,
	)

	assert mlp_kwargs["transform_input"] is True
	assert mlp_kwargs["input_is_spin"] is True
	assert mlp_kwargs["input_value"] == pytest.approx(0.5)
	assert cnn_kwargs["transform_input"] is True
	assert cnn_kwargs["input_is_spin"] is True
	assert cnn_kwargs["input_value"] == pytest.approx(0.5)
	np.testing.assert_allclose(
		np.asarray(mlp._flax_module.input_adapter(states)),
		np.asarray(expected),
		rtol=1e-6,
		atol=1e-6,
	)
	np.testing.assert_allclose(
		np.asarray(cnn._flax_module.input_adapter(states.reshape(2, 4, 1))),
		np.asarray(expected.reshape(2, 4, 1)),
		rtol=1e-6,
		atol=1e-6,
	)


def test_nqs_spin_half_binary_overrides_keep_unit_binary_inputs():
	"""Verify test nqs spin half binary overrides keep unit binary inputs."""
	representation = ModelRepresentationInfo(
		basis_type="real",
		local_space_type="spin-1/2",
		vector_encoding="Length-Ns array with entries in {0,1}.",
		integer_encoding="Binary computational basis.",
		sampler_representation="binary_01",
		network_representation="spin_binary_native",
	)
	states = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.float32)
	expected = jnp.array([[-1, 1, -1, 1], [1, -1, 1, -1]], dtype=jnp.float32)

	rbm_kwargs = apply_nqs_representation_overrides("rbm", representation, {})
	mlp_kwargs = apply_nqs_representation_overrides("mlp", representation, {})
	rbm = RBM(
		input_shape=(4,),
		n_hidden=3,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=13,
		**rbm_kwargs,
	)
	mlp = MLP(
		input_shape=(4,),
		hidden_dims=(8,),
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=13,
		**mlp_kwargs,
	)

	assert rbm_kwargs["input_activation"] is True
	assert rbm_kwargs["input_is_spin"] is False
	assert rbm_kwargs["input_value"] == pytest.approx(1.0)
	assert mlp_kwargs["transform_input"] is True
	assert mlp_kwargs["input_is_spin"] is False
	assert mlp_kwargs["input_value"] == pytest.approx(1.0)
	np.testing.assert_allclose(
		np.asarray(rbm._in_activation(states)),
		np.asarray(expected),
		rtol=1e-6,
		atol=1e-6,
	)
	np.testing.assert_allclose(
		np.asarray(mlp._flax_module.input_adapter(states)),
		np.asarray(expected),
		rtol=1e-6,
		atol=1e-6,
	)


def test_explicit_binary_override_beats_spin_model_default():
	"""Verify test explicit binary override beats spin model default."""
	representation = ModelRepresentationInfo(
		basis_type="real",
		local_space_type="spin-1/2",
		vector_encoding="Length-Ns array with signed entries in {-0.5,+0.5}.",
		integer_encoding="Binary computational basis.",
		sampler_representation="spin_pm",
		network_representation="spin_pm",
	)
	states = jnp.array([[0, 1, 0, 1]], dtype=jnp.float32)
	expected = jnp.array([[-1, 1, -1, 1]], dtype=jnp.float32)

	rbm_kwargs = apply_nqs_representation_overrides(
		"rbm",
		representation,
		{"state_representation": "binary_01"},
	)
	rbm = RBM(
		input_shape=(4,),
		n_hidden=3,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		seed=19,
		**rbm_kwargs,
	)

	assert rbm_kwargs["input_activation"] is True
	assert rbm_kwargs["input_is_spin"] is False
	assert rbm_kwargs["input_value"] == pytest.approx(1.0)
	np.testing.assert_allclose(
		np.asarray(rbm._in_activation(states)),
		np.asarray(expected),
		rtol=1e-6,
		atol=1e-6,
	)


def test_mps_and_pair_product_adapter_follows_input_convention():
	"""Verify test mps and pair product adapter follows input convention."""
	mps_bin = MPS(input_shape=(4,), bond_dim=2, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=False, input_value=1.0)
	mps_spin = MPS(input_shape=(4,), bond_dim=2, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=True, input_value=0.5)
	pp_bin = PairProduct(input_shape=(4,), use_rbm=False, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=False, input_value=1.0)
	pp_spin = PairProduct(input_shape=(4,), use_rbm=False, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=0, input_spin=True, input_value=0.5)

	assert _adapter_native(mps_bin) == "binary_01"
	assert _adapter_native(mps_spin) == "spin_pm"
	assert _adapter_native(pp_bin) == "binary_01"
	assert _adapter_native(pp_spin) == "spin_pm"


def test_rbm_pair_product_binary_and_spin_inputs_match():
	"""Verify test rbm pair product binary and spin inputs match."""
	states_bin = jnp.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=jnp.float32)
	states_spin = _binary_to_spin_pm_half(states_bin)

	pp_bin = PairProduct(input_shape=(4,), use_rbm=True, alpha=1.0, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=5, input_spin=False, input_value=1.0)
	pp_spin = PairProduct(input_shape=(4,), use_rbm=True, alpha=1.0, dtype=jnp.complex64, param_dtype=jnp.complex64, seed=5, input_spin=True, input_value=0.5)
	pp_spin.set_params(pp_bin.get_params())

	np.testing.assert_allclose(
		np.asarray(pp_bin(states_bin)),
		np.asarray(pp_spin(states_spin)),
		rtol=1e-6,
		atol=1e-6,
		equal_nan=True,
	)
	assert infer_network_family(pp_bin) == "pair_product"


def test_standard_ansatz_registry_resolves_to_general_backbones():
	"""Verify test standard ansatz registry resolves to general backbones."""
	rbm_cls, rbm_kwargs = resolve_ansatz_request("rbm")
	mlp_cls, mlp_kwargs = resolve_ansatz_request("mlp")
	cnn_cls, cnn_kwargs = resolve_ansatz_request("cnn")

	assert rbm_cls is RBM
	assert mlp_cls is MLP
	assert cnn_cls is CNN
	assert rbm_kwargs == {}
	assert mlp_kwargs == {}
	assert cnn_kwargs == {}
