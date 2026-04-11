import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
	from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
	from QES.Algebra.hilbert import HilbertSpace
	from QES.Algebra.Operator.impl.jax.operators_spin import sigma_x_jnp
	from QES.Solver.MonteCarlo.vmc import VMCSampler
	from QES.general_python.lattices import SquareLattice
	from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
	pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_spin_half_problem():
	lattice = SquareLattice(dim=1, lx=2, bc="obc")
	hilbert = HilbertSpace(lattice=lattice)
	model = TransverseFieldIsing(
		lattice=lattice,
		hilbert_space=hilbert,
		j=1.0,
		hx=0.7,
		hz=0.0,
		dtype=np.complex128,
	)
	net = RBM(
		input_shape=(hilbert.ns,),
		n_hidden=2,
		dtype=jnp.complex128,
		param_dtype=jnp.complex128,
		seed=0,
	)
	return hilbert, model, net


def test_hilbert_state_convention_exposes_machine_representation_fields():
	hilbert, _, _ = _build_spin_half_problem()
	convention = hilbert.state_convention

	assert convention["local_space_type"] == "spin-1/2"
	assert convention["integer_representation"] == "binary_01"
	assert convention["vector_representation"] == "spin_pm"
	assert convention["representation"] == "spin_pm"
	assert convention["spin"] is True
	assert convention["integer_mode_repr"] == pytest.approx(1.0)
	assert convention["vector_mode_repr"] == pytest.approx(0.5)
	assert convention["mode_repr"] == pytest.approx(0.5)


def test_vmc_sampler_defaults_to_hilbert_state_representation_when_unset():
	hilbert, model, net = _build_spin_half_problem()
	_ = model
	sampler = VMCSampler(
		net=net,
		shape=(hilbert.ns,),
		hilbert=hilbert,
		backend="jax",
		rng=np.random.default_rng(0),
		rng_k=jax.random.PRNGKey(0),
		numsamples=2,
		numchains=2,
		therm_steps=1,
		sweep_steps=1,
	)

	assert sampler.state_representation == "spin_pm"
	assert sampler.spin_representation is True
	assert sampler.mode_representation == pytest.approx(0.5)


def test_sigma_x_jax_respects_nonunit_binary_vector_convention():
	state = jnp.array([0.0, 0.5, 0.0, 0.5], dtype=jnp.float32)
	new_state, coeff = sigma_x_jnp(state, (1, 2), spin=False, spin_value=0.5)

	np.testing.assert_allclose(
		np.asarray(new_state),
		np.asarray([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32),
		rtol=1e-7,
		atol=1e-7,
	)
	np.testing.assert_allclose(np.asarray(coeff), np.asarray([0.25], dtype=np.float32), rtol=1e-7, atol=1e-7)
