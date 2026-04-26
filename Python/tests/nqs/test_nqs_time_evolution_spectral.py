"""Regression tests for nqs time evolution spectral."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from QES.Algebra.Properties.time_evo import time_evo_block
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.Algebra.Operator.impl.operators_spin import sig_k
    from QES.Algebra.hilbert import HilbertSpace
    from QES.NQS.nqs import NQS
    from QES.NQS.src.nqs_spectral import _enumerate_basis_states, _exact_wavefunction_vector
    from QES.general_python.lattices import SquareLattice
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
    pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_uniform_rbm_state(model_seed: int = 0):
    """Build uniform rbm state."""
    lattice = SquareLattice(dim=2, lx=2, ly=2, bc="obc")
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
        seed=model_seed,
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
        seed=model_seed,
        s_numsamples=8,
        s_numchains=4,
        s_therm_steps=2,
        s_sweep_steps=1,
    )
    nqs.set_params(zero_params)
    return model, nqs


def _build_uniform_rbm_chain(model_seed: int = 0):
    """Build uniform rbm chain."""
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
        seed=model_seed,
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
        seed=model_seed,
        s_numsamples=8,
        s_numchains=4,
        s_therm_steps=2,
        s_sweep_steps=1,
    )
    nqs.set_params(zero_params)
    return model, nqs


def _build_static_uniform_rbm_chain(model_seed: int = 0):
    """Build static uniform rbm chain."""
    lattice = SquareLattice(dim=1, lx=4, bc="pbc")
    hilbert = HilbertSpace(lattice=lattice)
    model = TransverseFieldIsing(
        lattice=lattice,
        hilbert_space=hilbert,
        j=0.0,
        hx=0.0,
        hz=0.0,
        dtype=np.complex128,
    )

    net = RBM(
        input_shape=(hilbert.ns,),
        n_hidden=hilbert.ns,
        seed=model_seed,
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
        seed=model_seed,
        s_numsamples=8,
        s_numchains=4,
        s_therm_steps=2,
        s_sweep_steps=1,
    )
    nqs.set_params(zero_params)
    return model, nqs


def test_square_tfim_time_evolution_and_spectral_smoke():
    """Verify test square tfim time evolution and spectral smoke."""
    model, nqs = _build_uniform_rbm_state()
    times = np.linspace(0.0, 0.10, 3)

    trajectory = nqs.time_evolve(
        times,
        num_samples=8,
        num_chains=4,
        n_batch=32,
        diag_shift=1e-3,
        sr_maxiter=64,
        ode_solver="RK4",
        restore=True,
    )
    spectral = nqs.spectral_function(
        times,
        trajectory=trajectory,
        eta=0.1,
    )

    assert trajectory.times.shape == times.shape
    assert trajectory.param_history.shape[0] == times.size
    assert spectral.correlator.shape[0] == times.size
    assert np.isclose(spectral.correlator[0], 1.0 + 0.0j, atol=1e-6)
    assert np.all(np.isfinite(np.real(spectral.correlator)))
    assert np.all(np.isfinite(np.imag(spectral.correlator)))
    assert np.all(np.isfinite(spectral.spectrum))
    assert spectral.frequencies.shape == spectral.spectrum.shape

    model.diagonalize()
    eig_vec = np.asarray(model._eig_vec)
    eig_val = np.asarray(model._eig_val)
    psi0 = np.ones(eig_vec.shape[0], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    overlaps = eig_vec.conj().T @ psi0
    psi_t = time_evo_block(eig_vec, eig_val, overlaps, times)
    exact_corr = np.einsum("i,it->t", np.conj(psi0), psi_t)

    assert np.isclose(exact_corr[0], 1.0 + 0.0j, atol=1e-12)
    assert abs(spectral.correlator[1] - exact_corr[1]) < 0.15


def test_tfim_modifier_time_evolution_smoke():
    """Verify test tfim modifier time evolution smoke."""
    model, nqs = _build_static_uniform_rbm_chain()
    times = np.linspace(0.0, 0.10, 3)
    probe = sig_k(np.pi, lattice=model.lattice, ns=model.hilbert.ns)

    probe_nqs = nqs.spawn_like(modifier=probe, verbose=False)
    trajectory = probe_nqs.time_evolve(
        times,
        num_samples=8,
        num_chains=4,
        n_batch=32,
        diag_shift=1e-3,
        sr_maxiter=64,
        ode_solver="RK4",
        restore=True,
    )
    physical_corr = nqs.compute_dynamical_correlator(
        times,
        ket_probe_operator=probe,
        bra_probe_operator=probe,
        trajectory=trajectory,
        exact_sum=True,
    )

    assert trajectory.times.shape == times.shape
    assert np.all(np.isfinite(np.real(physical_corr.correlator)))
    assert np.all(np.isfinite(np.imag(physical_corr.correlator)))

    basis_states = _enumerate_basis_states(nqs)
    psi0 = _exact_wavefunction_vector(nqs, basis_states=basis_states)
    psi0 /= np.sqrt(np.vdot(psi0, psi0))
    probe_matrix = probe.compute_matrix(
        hilbert_1=model.hilbert,
        matrix_type="dense",
        use_numpy=True,
    )
    phi0 = probe_matrix @ psi0
    exact_phys = np.full_like(times, np.vdot(phi0, phi0), dtype=np.complex128)

    assert np.allclose(physical_corr.correlator, exact_phys, atol=1e-8)
