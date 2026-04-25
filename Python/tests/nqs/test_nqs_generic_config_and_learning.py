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
except ImportError:
	pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_tiny_tfim():
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
	auto_sig = inspect.signature(NQS.get_auto_config)
	preset_sig = inspect.signature(estimate_network_params)

	assert "model_hint" not in auto_sig.parameters
	assert "model_type" not in preset_sig.parameters


def test_generic_cnn_estimate_is_structure_driven():
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
	_, hilbert, model = _build_tiny_tfim()
	stats = _run_short_training(builder(hilbert.ns), model, hilbert, seed=7)

	initial = float(np.real(stats.history[0]))
	best = min(float(np.real(v)) for v in stats.history)

	assert np.isfinite(initial)
	assert np.isfinite(best)
	assert best < initial - 0.05, f"{name} did not improve enough: initial={initial}, best={best}"
