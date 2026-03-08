import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.Algebra.Operator.impl.operators_spin import sig_x
    from QES.Algebra.Properties.time_evo import time_evo_block
    from QES.Algebra.hilbert import HilbertSpace
    from QES.NQS.nqs import NQS
    from QES.NQS.src.nqs_spectral import (
        _NQSParamView,
        _enumerate_basis_states,
        _exact_wavefunction_vector,
        _materialize_trajectory_params,
    )
    from QES.general_python.lattices import SquareLattice
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
    pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_single_spin_rabi_nqs(seed: int = 0):
    r"""
    Build a one-spin TFIM qubit for an exact real-time Rabi/Larmor benchmark.

    The Hamiltonian is

        H = h_z S^z,

    and the zero-parameter RBM represents the equal-amplitude superposition
    state ``|+x>`` in the working basis. The exact observable

        <S^x(t)>

    oscillates with the physical Schrödinger frequency and therefore provides a
    sharp regression test for TDVP time-axis conventions.
    """

    lattice = SquareLattice(dim=1, lx=1, bc="obc")
    hilbert = HilbertSpace(lattice=lattice)
    model = TransverseFieldIsing(
        lattice=lattice,
        hilbert_space=hilbert,
        j=0.0,
        hx=0.0,
        hz=1.0,
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
        batch_size=16,
        backend="jax",
        dtype=np.complex128,
        symmetrize=False,
        verbose=False,
        seed=seed,
        s_numsamples=64,
        s_numchains=8,
        s_therm_steps=1,
        s_sweep_steps=1,
    )
    nqs.set_params(zero_params)
    return model, nqs


def _observable_from_trajectory(nqs, trajectory, operator) -> np.ndarray:
    r"""
    Evaluate an observable exactly along an NQS TDVP trajectory.

    For this one-spin benchmark we use deterministic full-basis evaluation so
    the test isolates the TDVP dynamics itself rather than Monte Carlo
    estimator noise.
    """

    basis_states = _enumerate_basis_states(nqs)
    op_matrix = operator.compute_matrix(
        hilbert_1=nqs.hilbert,
        matrix_type="dense",
        use_numpy=True,
    )

    values = []
    for params_t, phase_t in zip(trajectory.param_history, trajectory.global_phase):
        state_view = _NQSParamView(
            nqs,
            _materialize_trajectory_params(nqs, trajectory, params_t),
            global_phase=complex(phase_t),
        )
        psi_t = _exact_wavefunction_vector(state_view, basis_states=basis_states)
        psi_t /= np.sqrt(np.vdot(psi_t, psi_t))
        values.append(np.vdot(psi_t, op_matrix @ psi_t))
    return np.asarray(values, dtype=np.complex128)


def _exact_ed_observable(model, times, operator) -> np.ndarray:
    model.diagonalize()
    eig_vec = np.asarray(model._eig_vec)
    eig_val = np.asarray(model._eig_val)
    psi0 = np.ones(eig_vec.shape[0], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    overlaps = eig_vec.conj().T @ psi0
    psi_t = time_evo_block(eig_vec, eig_val, overlaps, times)
    op_matrix = operator.compute_matrix(
        hilbert_1=model.hilbert,
        matrix_type="dense",
        use_numpy=True,
    )
    return np.einsum("it,ij,jt->t", np.conj(psi_t), op_matrix, psi_t)


def test_single_spin_rabi_oscillation_matches_ed():
    r"""
    The default NQS real-time TDVP path should reproduce the one-spin ED qubit.

    This is the minimal physics regression for the TDVP time convention: the
    same observation times must correspond to the same oscillation phase in ED
    and NQS, otherwise the benchmark immediately drifts.
    """

    model, nqs = _build_single_spin_rabi_nqs()
    times = np.linspace(0.0, 2.0 * np.pi, 21)
    observable = sig_x(ns=model.hilbert.ns, sites=[0])

    trajectory = nqs.time_evolve(
        times,
        num_samples=64,
        num_chains=8,
        n_batch=16,
        diag_shift=1e-8,
        sr_maxiter=64,
        ode_solver="RK4",
        n_substeps=8,
        restore=True,
    )
    nqs_values = _observable_from_trajectory(nqs, trajectory, observable)
    ed_values = _exact_ed_observable(model, times, observable)

    assert np.allclose(trajectory.times, times)
    assert np.max(np.abs(nqs_values - ed_values)) < 1e-3
    assert abs(nqs_values[0] - 0.5) < 1e-10
    assert abs(nqs_values[-1] - 0.5) < 2e-3
