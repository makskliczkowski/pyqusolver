import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.Algebra.Operator.impl.operators_spin import sig_k
    from QES.Algebra.hilbert import HilbertSpace
    from QES.NQS.nqs import NQS
    from QES.general_python.lattices import SquareLattice
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
    pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_uniform_tfim_state(seed: int = 0):
    lattice = SquareLattice(dim=1, lx=4, bc="pbc")
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
        n_hidden=hilbert.ns,
        seed=seed,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
    )
    zero_params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), net.get_params())
    net.set_params(zero_params)

    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        sampler="vmc",
        batch_size=32,
        backend="jax",
        dtype=np.complex128,
        verbose=False,
        seed=seed,
        s_numsamples=16,
        s_numchains=4,
        s_therm_steps=2,
        s_sweep_steps=1,
    )
    nqs.set_params(zero_params)
    return model, hilbert, lattice, nqs


def test_dynamic_structure_factor_modifier_smoke():
    model, hilbert, lattice, nqs = _build_uniform_tfim_state(seed=0)
    probe_q = sig_k(np.pi, lattice=lattice, ns=hilbert.ns)
    probe_mq = sig_k(-np.pi, lattice=lattice, ns=hilbert.ns)
    times = np.linspace(0.0, 0.05, 3)

    corr = nqs.compute_dynamical_correlator(
        times,
        ket_probe_operator=probe_q,
        bra_probe_operator=probe_mq,
        exact_sum=True,
        num_samples=16,
        num_chains=4,
        n_batch=32,
        diag_shift=1e-3,
        sr_maxiter=64,
        restore=True,
    )
    spectral = nqs.dynamic_structure_factor(
        times,
        ket_probe_operator=probe_q,
        bra_probe_operator=probe_mq,
        exact_sum=True,
        num_samples=16,
        num_chains=4,
        n_batch=32,
        diag_shift=1e-3,
        sr_maxiter=64,
        eta=0.1,
        window=None,
        restore=True,
    )
    spectral_from_corr = nqs.spectrum_from_correlator(
        times,
        corr.correlator,
        eta=0.1,
        window=None,
    )
    spectral_map = nqs.dynamic_structure_factor_kspace(
        times,
        probe_operators=[
            sig_k(0.0, lattice=lattice, ns=hilbert.ns),
            probe_q,
        ],
        bra_probe_operators=[
            sig_k(-0.0, lattice=lattice, ns=hilbert.ns),
            probe_mq,
        ],
        k_values=np.asarray(
            [
                lattice.kvectors[0],
                lattice.kvectors[2],
            ]
        ),
        exact_sum=True,
        num_samples=16,
        num_chains=4,
        n_batch=32,
        diag_shift=1e-3,
        sr_maxiter=64,
        eta=0.1,
        window=None,
        restore=True,
    )

    assert corr.correlator.shape == times.shape
    assert spectral.correlator.shape == times.shape
    assert spectral_from_corr.correlator.shape == times.shape
    assert spectral.frequencies.shape == spectral.spectrum.shape
    assert np.allclose(spectral.frequencies, spectral_from_corr.frequencies)
    assert np.allclose(spectral.spectrum, spectral_from_corr.spectrum, atol=1e-10, rtol=1e-10)
    assert np.all(np.isfinite(np.real(spectral.correlator)))
    assert np.all(np.isfinite(np.imag(spectral.correlator)))
    assert np.all(np.isfinite(np.real(spectral.spectrum_complex)))
    assert np.all(np.isfinite(np.imag(spectral.spectrum_complex)))
    assert spectral_map.spectrum.shape[0] == 2
    assert spectral_map.correlator.shape == (2, times.size)
    assert spectral_map.k_values.shape == (2, 3)

    model.diagonalize()
    probe_matrix = probe_q.compute_matrix(
        hilbert_1=hilbert,
        matrix_type="dense",
        use_numpy=True,
    )
    eig_vec = np.asarray(model._eig_vec)
    psi0 = np.ones(eig_vec.shape[0], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    exact_static_weight = np.vdot(psi0, probe_matrix.conj().T @ probe_matrix @ psi0)

    assert abs(corr.correlator[0] - exact_static_weight) < 1e-8
    assert abs(spectral.correlator[0] - exact_static_weight) < 1e-8
    assert abs(spectral.metadata["static_weight"] - exact_static_weight) < 1e-8
