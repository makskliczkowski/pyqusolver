"""Regression tests for nqs generic config and learning."""

import inspect
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
	from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
	from QES.Algebra.hilbert import HilbertSpace
	from QES.NQS import NQSPhysicsConfig, NQSSolverConfig, NQSTrainConfig
	from QES.NQS.ansatze.registry import list_available_ansatze, load_ansatz_class
	from QES.NQS.nqs import NQS
	from QES.NQS.src.network.presets import estimate_network_params
	from QES.general_python.lattices import SquareLattice
	from QES.general_python.ml.net_impl.networks.net_cnn import CNN
	from QES.general_python.ml.net_impl.networks.net_mlp import MLP
	from QES.general_python.ml.net_impl.utils import net_utils_jax
	from QES.Solver.MonteCarlo.vmc import VMCSampler
except ImportError:
	pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_tiny_tfim():
	"""Build tiny tfim."""
	lattice = SquareLattice(dim=1, lx=4, bc="obc")
	hilbert = HilbertSpace(lattice=lattice)
	model = TransverseFieldIsing(
		lattice=lattice,
		hilbert_space=hilbert,
		j=1.0,
		hx=0.7,
		hz=0.0,
		dtype=np.complex64,
	)
	model.diagonalize()
	return lattice, hilbert, model


def _run_short_training(net, model, hilbert, seed):
	"""Helper for run short training."""
	nqs = NQS(
		logansatz=net,
		model=model,
		hilbert=hilbert,
		sampler="vmc",
		batch_size=32,
		backend="jax",
		dtype=np.complex64,
		symmetrize=False,
		verbose=False,
		seed=seed,
		use_orbax=False,
		s_numsamples=32,
		s_numchains=4,
		s_therm_steps=4,
		s_sweep_steps=1,
	)
	return nqs.train(
		n_epochs=12,
		checkpoint_every=1000,
		lr=0.03,
		diag_shift=1e-3,
		n_batch=32,
		num_samples=32,
		num_chains=4,
		num_thermal=4,
		num_sweep=1,
		ode_solver="RK4",
		phases="default",
		use_pbar=False,
	)


def test_config_helpers_do_not_expose_model_profiles():
	"""Verify test config helpers do not expose model profiles."""
	auto_sig = inspect.signature(NQS.get_auto_config)
	preset_sig = inspect.signature(estimate_network_params)

	assert "model_hint" not in auto_sig.parameters
	assert "model_type" not in preset_sig.parameters


def test_generic_cnn_estimate_is_structure_driven():
	"""Verify test generic cnn estimate is structure driven."""
	cfg = estimate_network_params(
		net_type="cnn",
		num_sites=4,
		lattice_dims=(4, 1),
		target_accuracy="medium",
		dtype="complex64",
	)
	kw = cfg.to_factory_kwargs()

	assert kw["reshape_dims"] == (4, 1)
	assert kw["kernel_sizes"] == ((3, 1),)
	assert kw["features"] == (6,)


def test_nqs_exports_only_canonical_ansatz_names():
	"""Verify test nqs exports only canonical ansatz names."""
	assert list_available_ansatze() == [
		"rbm",
		"cnn",
		"resnet",
		"ar",
		"mlp",
		"eqgcnn",
		"transformer",
		"pp",
		"rbmpp",
		"jastrow",
		"mps",
		"amplitude_phase",
		"approx_symmetric",
	]
	with pytest.raises(ValueError):
		load_ansatz_class("autoregressive")
	with pytest.raises(ValueError):
		load_ansatz_class("pair_product")
	with pytest.raises(ValueError):
		load_ansatz_class("gcnn")


def test_train_config_emits_only_canonical_scheduler_keys():
	"""Verify test train config emits only canonical scheduler keys."""
	train_cfg = NQSTrainConfig(lr_init=1e-2, reg_init=2e-3, diag_init=3e-3, global_p=0.25)
	kwargs = train_cfg.to_train_kwargs()

	assert kwargs["lr_init"] == pytest.approx(1e-2)
	assert kwargs["reg_init"] == pytest.approx(2e-3)
	assert kwargs["diag_init"] == pytest.approx(3e-3)
	assert kwargs["global_p"] == pytest.approx(0.25)
	assert "lr_initial" not in kwargs
	assert "reg_initial" not in kwargs
	assert "diag_initial" not in kwargs
	assert "p_global" not in kwargs


def test_nqs_exports_no_legacy_local_energy_aliases():
	"""Verify test nqs exports no legacy local energy aliases."""
	assert "loc_energy" not in NQS.__dict__
	assert "local_en" not in NQS.__dict__


def test_train_accepts_canonical_scheduler_init_keys():
	"""Verify test train accepts canonical scheduler init keys."""
	_, hilbert, model = _build_tiny_tfim()
	net = MLP(
		input_shape=(hilbert.ns,),
		hidden_dims=(8,),
		activations="log_cosh",
		dtype=jnp.complex64,
		param_dtype=jnp.complex64,
		seed=4,
	)
	psi = NQS(
		logansatz=net,
		model=model,
		hilbert=hilbert,
		sampler="vmc",
		batch_size=16,
		backend="jax",
		dtype=np.complex64,
		verbose=False,
		use_orbax=False,
		seed=5,
		s_numsamples=16,
		s_numchains=4,
		s_therm_steps=2,
		s_sweep_steps=1,
	)

	stats = psi.train(
		n_epochs=2,
		checkpoint_every=1000,
		n_batch=16,
		num_samples=16,
		num_chains=4,
		num_thermal=2,
		num_sweep=1,
		ode_solver="RK4",
		phases=None,
		lr_scheduler="constant",
		reg_scheduler="constant",
		diag_scheduler="constant",
		lr_init=0.02,
		reg_init=1e-3,
		diag_init=5e-4,
		use_pbar=False,
	)

	assert len(stats.history) == 2
	assert np.all(np.isfinite(np.asarray(stats.history)))


def test_wrap_single_step_jax_does_not_mutate_nqs_batch_size():
	"""Verify test wrap single step jax does not mutate nqs batch size."""
	_, hilbert, model = _build_tiny_tfim()
	net = MLP(
		input_shape=(hilbert.ns,),
		hidden_dims=(8,),
		activations="log_cosh",
		dtype=jnp.complex64,
		param_dtype=jnp.complex64,
		seed=8,
	)
	psi = NQS(
		logansatz=net,
		model=model,
		hilbert=hilbert,
		sampler="vmc",
		batch_size=16,
		backend="jax",
		dtype=np.complex64,
		verbose=False,
		use_orbax=False,
		seed=9,
		s_numsamples=16,
		s_numchains=4,
		s_therm_steps=2,
		s_sweep_steps=1,
	)

	initial_batch_size = psi._batch_size
	_ = psi.wrap_single_step_jax(batch_size=8)

	assert psi._batch_size == initial_batch_size


def test_local_estimator_connected_state_chunking_preserves_result():
	"""Verify test local estimator connected state chunking preserves result."""
	def local_conn_fn(state):
		"""Helper for local conn fn."""
		shifts = jnp.arange(5, dtype=state.dtype).reshape(5, 1)
		new_states = state[None, :] + shifts
		new_vals = jnp.arange(1, 6, dtype=jnp.float32)
		return new_states, new_vals

	def logproba_fn(params, states):
		"""Helper for logproba fn."""
		return jnp.dot(states, params)

	params = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)
	states = jnp.array([[1.0, 0.0, -1.0], [0.5, -0.5, 1.0]], dtype=jnp.float32)
	logp_in = logproba_fn(params, states).reshape(-1, 1)
	sample_p = jnp.ones((states.shape[0], 1), dtype=jnp.float32)

	chunked, mean_chunked, std_chunked = net_utils_jax.apply_callable_batched_jax(
		local_conn_fn,
		states,
		sample_p,
		logp_in,
		logproba_fn,
		params,
		2,
	)
	full, mean_full, std_full = net_utils_jax.apply_callable_batched_jax(
		local_conn_fn,
		states,
		sample_p,
		logp_in,
		logproba_fn,
		params,
		32,
	)

	assert np.allclose(np.asarray(chunked), np.asarray(full))
	assert np.allclose(np.asarray(mean_chunked), np.asarray(mean_full))
	assert np.allclose(np.asarray(std_chunked), np.asarray(std_full))


def test_vmc_sampler_uses_configurable_network_eval_chunk_size():
	"""Verify test vmc sampler uses configurable network eval chunk size."""
	def net_apply(_params, x):
		"""Helper for net apply."""
		return jnp.sum(x, axis=tuple(range(1, x.ndim)))

	configs = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)
	out = VMCSampler._batched_network_apply(None, configs, net_apply, chunk_size=2)

	assert np.allclose(np.asarray(out), np.asarray(jnp.sum(configs, axis=1)))


def test_vmc_sampler_rejects_invalid_network_eval_chunk_size():
	"""Verify test vmc sampler rejects invalid network eval chunk size."""
	with pytest.raises(ValueError):
		VMCSampler._batched_network_apply(None, jnp.ones((2, 3), dtype=jnp.float32), lambda _p, x: x[:, 0], chunk_size=0)


def test_vmc_sample_prob_helper_matches_explicit_reweighting():
	"""Verify VMC sample-weight normalization helper matches explicit reweighting."""
	log_ansatz = jnp.array([0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j], dtype=jnp.complex64)
	total_samples = log_ansatz.shape[0]
	log_prob_exponent = 1.5

	helper = VMCSampler._compute_sample_probs_jax(
		log_ansatz,
		total_samples,
		log_prob_exponent,
		uniform_weights=False,
	)

	log_unnorm = log_prob_exponent * jnp.real(log_ansatz)
	log_max = jnp.max(log_unnorm)
	explicit = jnp.exp(log_unnorm - log_max)
	explicit = explicit / jnp.sum(explicit) * total_samples

	assert np.allclose(np.asarray(helper), np.asarray(explicit))


def test_vmc_sample_prob_helper_returns_none_for_uniform_weights():
	"""Verify VMC sample-weight normalization helper skips explicit weights when uniform."""
	log_ansatz = jnp.array([0.1, -0.2], dtype=jnp.float32)

	assert VMCSampler._compute_sample_probs_jax(
		log_ansatz,
		total_samples=2,
		log_prob_exponent=0.0,
		uniform_weights=True,
	) is None


def test_vmc_collect_pt_physical_samples_picks_target_beta_sector():
	"""Verify PT sample collection picks one physical sample per chain column."""
	states = jnp.array(
		[
			[[10.0, 0.0], [20.0, 0.0]],
			[[11.0, 0.0], [21.0, 0.0]],
			[[12.0, 0.0], [22.0, 0.0]],
		],
		dtype=jnp.float32,
	)
	logprobas = jnp.array(
		[
			[1.0, 2.0],
			[3.0, 4.0],
			[5.0, 6.0],
		],
		dtype=jnp.float32,
	)
	betas = jnp.array(
		[
			[0.5, 1.0],
			[1.0, 0.25],
			[0.25, 0.5],
		],
		dtype=jnp.float32,
	)

	collected_states, collected_logprobas = VMCSampler._collect_pt_physical_samples_jax(
		states,
		logprobas,
		betas,
		target_beta=1.0,
	)

	assert np.allclose(np.asarray(collected_states), np.asarray([[11.0, 0.0], [20.0, 0.0]]))
	assert np.allclose(np.asarray(collected_logprobas), np.asarray([3.0, 2.0]))


@pytest.mark.parametrize(
	("ansatz", "kwargs"),
	[
		("mlp", {}),
		("mps", {"bond_dim": 2}),
		(
			"amplitude_phase",
			{
				"amplitude_kwargs": {"hidden_dims": (4,)},
				"phase_kwargs": {"hidden_dims": (4,)},
			},
		),
	],
)
def test_solver_config_builds_nonpreset_ansatze_without_config_profiles(ansatz, kwargs):
	"""Verify test solver config builds nonpreset ansatze without config profiles."""
	p_cfg = NQSPhysicsConfig(model_type="tfim", lattice_type="chain", lx=4, bc="obc")
	s_cfg = NQSSolverConfig(ansatz=ansatz, dtype="complex64", backend="jax")
	states = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.float32)

	net = s_cfg.make_net(p_cfg, **kwargs)
	values = net(states)

	assert s_cfg.sota_config is None
	assert np.all(np.isfinite(np.asarray(values)))
	assert np.asarray(values).shape == (2,)


@pytest.mark.parametrize(
	("name", "builder"),
	[
		(
			"cnn",
			lambda ns: CNN(
				input_shape=(ns,),
				reshape_dims=(ns, 1),
				features=(4,),
				kernel_sizes=((3, 1),),
				dtype=jnp.complex64,
				param_dtype=jnp.complex64,
				transform_input=False,
				seed=1,
			),
		),
		(
			"mlp",
			lambda ns: MLP(
				input_shape=(ns,),
				hidden_dims=(8,),
				activations="log_cosh",
				dtype=jnp.complex64,
				param_dtype=jnp.complex64,
				seed=2,
			),
		),
	],
)
def test_non_rbm_ansatze_learn_on_tiny_tfim(name, builder):
	"""Verify test non rbm ansatze learn on tiny tfim."""
	_, hilbert, model = _build_tiny_tfim()
	stats = _run_short_training(builder(hilbert.ns), model, hilbert, seed=7)

	initial = float(np.real(stats.history[0]))
	best = min(float(np.real(v)) for v in stats.history)

	assert np.isfinite(initial)
	assert np.isfinite(best)
	assert best < initial - 0.05, f"{name} did not improve enough: initial={initial}, best={best}"
