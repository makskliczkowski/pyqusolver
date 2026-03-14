import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from QES.Algebra.Model.Interacting.Spin.xxz import XXZ
    from QES.Algebra.Operator.impl.operators_spin import sig_x
    from QES.Algebra.Properties.time_evo import time_evo_block
    from QES.Algebra.hilbert import HilbertSpace
    from QES.NQS.nqs import NQS
    from QES.general_python.lattices import SquareLattice
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
    pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_heisenberg_chain(seed: int = 3):
    lattice = SquareLattice(dim=1, lx=4, bc="pbc")
    hilbert = HilbertSpace(lattice=lattice)
    model = XXZ(
        lattice=lattice,
        hilbert_space=hilbert,
        jxy=-1.0,
        delta=1.0,
        hx=0.0,
        hz=0.0,
        dtype=np.complex128,
    )
    net = RBM(
        input_shape=(hilbert.ns,),
        alpha=4.0,
        seed=seed,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
        in_activation=True,
        visible_bias=False,
    )
    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        sampler="vmc",
        batch_size=128,
        backend="jax",
        dtype=np.complex128,
        verbose=False,
        seed=seed,
        s_numsamples=256,
        s_numchains=16,
        s_therm_steps=8,
        s_sweep_steps=1,
        s_upd_fun="EXCHANGE",
        s_initstate="RND_FIXED",
        magnetization=0,
    )
    return model, hilbert, nqs


def _exact_local_quench_correlator(model, hilbert, probe, times):
    model.diagonalize()
    eig_vec = np.asarray(model._eig_vec, dtype=np.complex128)
    eig_val = np.asarray(model._eig_val, dtype=np.float64)
    gs = np.asarray(eig_vec[:, 0], dtype=np.complex128)
    probe_matrix = probe.compute_matrix(
        hilbert_1=hilbert,
        matrix_type="dense",
        use_numpy=True,
    )
    phi0 = np.asarray(probe_matrix @ gs, dtype=np.complex128)
    overlaps = eig_vec.conj().T @ phi0
    phi_t = time_evo_block(eig_vec, eig_val, overlaps, times)
    corr = np.einsum("i,it->t", np.conj(phi0), phi_t)
    corr = corr * np.exp(1.0j * float(np.real(eig_val[0])) * (times - times[0]))
    return corr, float(np.real(eig_val[0]))


def test_heisenberg_modifier_local_quench_smoke():
    model, hilbert, nqs = _build_heisenberg_chain()
    times = np.linspace(0.0, 0.20, 5)

    stats = nqs.train(
        n_epochs=60,
        phases="default",
        lr=0.01,
        checkpoint_every=500,
        diag_shift=1e-3,
        sr_maxiter=128,
        exact_predictions=model.eig_vals,
    )
    model.diagonalize()
    probe = sig_x(ns=hilbert.ns, sites=[0])

    probe_nqs = nqs.spawn_like(modifier=probe, verbose=False)
    trajectory = probe_nqs.time_evolve(
        times,
        num_samples=256,
        num_chains=16,
        n_batch=128,
        diag_shift=1e-3,
        sr_maxiter=128,
        ode_solver="RK4",
        max_dt=0.01,
        restore=True,
    )
    corr_nqs = nqs.compute_dynamical_correlator(
        times,
        ket_probe_operator=probe,
        bra_probe_operator=probe,
        trajectory=trajectory,
        reference_energy=float(np.real(model._eig_val[0])),
        exact_sum=True,
    )

    corr_exact, _ = _exact_local_quench_correlator(model, hilbert, probe, times)
    assert trajectory.times.shape == times.shape
    assert corr_nqs.correlator.shape == times.shape
    assert np.all(np.isfinite(np.real(corr_nqs.correlator)))
    assert np.all(np.isfinite(np.imag(corr_nqs.correlator)))
    assert abs(corr_nqs.correlator[0] - corr_exact[0]) < 0.15
    assert abs(corr_nqs.correlator[1] - corr_exact[1]) < 0.20
