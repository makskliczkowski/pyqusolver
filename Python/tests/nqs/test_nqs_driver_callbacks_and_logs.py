import json
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
	from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
	from QES.Algebra.hilbert import HilbertSpace
	from QES.NQS import JsonLog, NQS, RuntimeLog
	from QES.general_python.lattices import SquareLattice
	from QES.general_python.ml.net_impl.networks.net_mlp import MLP
except ImportError:
	pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_tiny_problem():
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
	net = MLP(
		input_shape=(hilbert.ns,),
		hidden_dims=(8,),
		activations="log_cosh",
		dtype=jnp.complex64,
		param_dtype=jnp.complex64,
		seed=0,
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
		seed=3,
		s_numsamples=16,
		s_numchains=4,
		s_therm_steps=2,
		s_sweep_steps=1,
	)
	return psi


def test_runtime_log_collects_epoch_data_and_callback_stops_early(tmp_path):
	psi = _build_tiny_problem()
	runtime_log = RuntimeLog()

	def stop_after_first_epoch(step, log_data, trainer):
		assert "mean" in log_data
		assert "acceptance" in log_data
		return step < 0

	stats = psi.train(
		n_epochs=6,
		checkpoint_every=1000,
		lr=0.03,
		diag_shift=1e-3,
		n_batch=16,
		num_samples=16,
		num_chains=4,
		num_thermal=2,
		num_sweep=1,
		ode_solver="RK4",
		phases="default",
		use_pbar=False,
		save_path=str(tmp_path / "ckpt"),
		out=runtime_log,
		callback=stop_after_first_epoch,
	)

	assert len(stats.history) == 1
	assert runtime_log.data["step"] == [0]
	assert len(runtime_log.data["mean"]) == 1
	assert np.isfinite(runtime_log.data["mean"][0])


def test_json_log_writes_serialized_training_history(tmp_path):
	psi = _build_tiny_problem()
	log_prefix = tmp_path / "nqs_run"

	psi.train(
		n_epochs=2,
		checkpoint_every=1000,
		lr=0.03,
		diag_shift=1e-3,
		n_batch=16,
		num_samples=16,
		num_chains=4,
		num_thermal=2,
		num_sweep=1,
		ode_solver="RK4",
		phases="default",
		use_pbar=False,
		save_path=str(tmp_path / "ckpt"),
		out=JsonLog(output_prefix=log_prefix, write_every=1),
	)

	with (log_prefix.with_suffix(".json")).open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	assert payload["step"] == [0, 1]
	assert len(payload["mean"]) == 2
	assert "global_phase" in payload
