import os
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

if os.environ.get("QES_RUN_SMALL_NQS_BENCHMARKS", "0") != "1":
	pytest.skip("Set QES_RUN_SMALL_NQS_BENCHMARKS=1 to run tiny NQS benchmarks.", allow_module_level=True)

try:
	from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
	from QES.Algebra.hilbert import HilbertSpace
	from QES.NQS.nqs import NQS
	from QES.general_python.lattices import SquareLattice
	from QES.general_python.ml.net_impl.networks.net_autoregressive import ComplexAR
	from QES.general_python.ml.net_impl.networks.net_cnn import CNN
	from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
	pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _block_tree(x):
	for leaf in jax.tree_util.tree_leaves(x):
		if hasattr(leaf, "block_until_ready"):
			leaf.block_until_ready()


def _build_tiny_model():
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
	return lattice, hilbert, model


def _make_networks(ns: int):
	return {
		"rbm": RBM(
			input_shape=(ns,),
			n_hidden=4,
			dtype=jnp.float32,
			param_dtype=jnp.float32,
			seed=0,
		),
		"cnn": CNN(
			input_shape=(ns,),
			reshape_dims=(ns, 1),
			features=(4,),
			kernel_sizes=(3,),
			dtype=jnp.float32,
			param_dtype=jnp.float32,
			transform_input=False,
			seed=1,
		),
		"autoregressive": ComplexAR(
			input_shape=(ns,),
			ar_hidden=(8,),
			phase_hidden=(8,),
			dtype=jnp.complex64,
			seed=2,
		),
	}


def _build_tiny_nqs(net, model, hilbert, seed: int):
	return NQS(
		logansatz=net,
		model=model,
		hilbert=hilbert,
		sampler=None,
		batch_size=16,
		backend="jax",
		dtype=np.complex64,
		symmetrize=False,
		verbose=False,
		seed=seed,
		s_numsamples=8,
		s_numchains=2,
		s_therm_steps=2,
		s_sweep_steps=1,
	)


def _time_call(fun, *args, repeat: int = 3, **kwargs):
	times = []
	last = None
	for _ in range(repeat):
		t0 = time.perf_counter()
		last = fun(*args, **kwargs)
		_block_tree(last)
		times.append(time.perf_counter() - t0)
	return float(np.mean(times)), last


def benchmark_tiny_nqs_paths():
	_, hilbert, model = _build_tiny_model()
	ns = hilbert.ns
	networks = _make_networks(ns)
	batch_states = jnp.asarray(np.random.default_rng(0).integers(0, 2, size=(16, ns)), dtype=jnp.float32)
	results = []

	for idx, (name, net) in enumerate(networks.items()):
		nqs = _build_tiny_nqs(net, model, hilbert, seed=idx)
		sampler = nqs.sampler
		params = net.get_params()

		# Warmup compile
		_block_tree(net(batch_states))
		_block_tree(sampler.sample(parameters=params, num_samples=8, num_chains=2))

		forward_time, _ = _time_call(net, batch_states, repeat=3)
		sample_time, sample_out = _time_call(
			sampler.sample,
			parameters=params,
			num_samples=8,
			num_chains=2,
			repeat=3,
		)
		total_samples = 16.0
		results.append(
			{
				"name": name,
				"sampler": sampler.name,
				"family": getattr(getattr(nqs, "_network_adapter", None), "family", "unknown"),
				"forward_ms": 1.0e3 * forward_time,
				"sample_ms": 1.0e3 * sample_time,
				"samples_per_sec": total_samples / sample_time if sample_time > 0 else 0.0,
				"result_shape": str(jax.tree_util.tree_structure(sample_out)),
			}
		)

	results.sort(key=lambda item: item["sample_ms"])
	return results


def test_tiny_nqs_benchmark_smoke():
	results = benchmark_tiny_nqs_paths()
	assert len(results) >= 3
	for item in results:
		assert item["forward_ms"] > 0.0
		assert item["sample_ms"] > 0.0
		assert item["samples_per_sec"] > 0.0
	assert results[0]["sample_ms"] <= results[-1]["sample_ms"]
	print("\nTiny NQS benchmark ranking:")
	for item in results:
		print(
			f"{item['name']:>16s} | sampler={item['sampler']:<3s} | "
			f"forward={item['forward_ms']:.3f} ms | sample={item['sample_ms']:.3f} ms | "
			f"samples_per_sec={item['samples_per_sec']:.2f}"
		)
