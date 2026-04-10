import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

try:
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.Algebra.hilbert import HilbertSpace
    from QES.NQS.nqs import NQS
    from QES.NQS.src.nqs_entropy import (
        _enumerate_basis_states_nqs,
        compute_ed_entanglement_entropy,
        compute_renyi_entropy,
    )
    from QES.NQS.src.spectral.exact import enumerate_basis_states
    from QES.general_python.lattices import SquareLattice
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
    pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _binary_to_spin_pm_half(states):
    return np.asarray(states, dtype=np.float64) - 0.5


def _build_exact_cat_state_nqs():
    lattice = SquareLattice(dim=1, lx=2, bc="obc")
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
        input_shape=(2,),
        n_hidden=2,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
        seed=0,
    )
    # Binary-visible RBM with amplitudes proportional to [2, 0, 0, -1].
    t = np.arcsinh(1.0 / np.sqrt(2.0))
    params = net.get_params()
    params["visible_bias"] = jnp.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128)
    params["VisibleToHidden"]["kernel"] = jnp.array(
        [[t, 1j * np.pi / 2], [1j * np.pi / 2, t]],
        dtype=jnp.complex128,
    )
    params["VisibleToHidden"]["bias"] = jnp.zeros((2,), dtype=jnp.complex128)
    net.set_params(params)

    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        sampler="vmc",
        batch_size=32,
        backend="jax",
        dtype=np.complex128,
        symmetrize=False,
        verbose=False,
        seed=0,
        state_representation="binary_01",
        s_numsamples=256,
        s_numchains=8,
        s_therm_steps=4,
        s_sweep_steps=1,
    )
    nqs.set_params(params)
    psi = np.array([2.0, 0.0, 0.0, -1.0], dtype=np.complex128) / np.sqrt(5.0)
    return hilbert, nqs, psi


def _build_trained_tfim_state():
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
    model.diagonalize()

    net = RBM(
        input_shape=(hilbert.ns,),
        n_hidden=2,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
        seed=0,
    )
    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        sampler="vmc",
        batch_size=32,
        backend="jax",
        dtype=np.complex128,
        symmetrize=False,
        verbose=False,
        seed=0,
        s_numsamples=32,
        s_numchains=4,
        s_therm_steps=4,
        s_sweep_steps=1,
    )

    stats = nqs.train(
        n_epochs=20,
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
        exact_predictions=model.eig_vals,
    )
    psi_exact = np.asarray(model._eig_vec)[:, 0]
    return hilbert, nqs, model, psi_exact, stats


@pytest.mark.parametrize(
    ("region", "q"),
    [
        ([0], 2),
        ([0], 3),
        ([0], 4),
        ([1], 2),
        ([1], 3),
        ([1], 4),
        ([0, 1], 2),
        ([0, 1], 3),
        ([0, 1], 4),
    ],
)
def test_exact_sum_renyi_matches_ed_for_multiple_regions_and_q(region, q):
    hilbert, nqs, psi = _build_exact_cat_state_nqs()

    exact_sum_sq = compute_renyi_entropy(nqs, region=region, q=q, exact_sum=True)
    ed_sq = compute_ed_entanglement_entropy(
        psi,
        np.asarray(region, dtype=int),
        hilbert.ns,
        q_values=[q],
        n_states=1,
    )[f"renyi_{q}"][0]

    assert abs(exact_sum_sq - ed_sq) < 1e-10


def test_exact_sum_raw_outputs_are_consistent():
    _, nqs, _ = _build_exact_cat_state_nqs()
    raw = compute_renyi_entropy(nqs, region=[0], q=3, exact_sum=True, return_raw=True)

    assert raw["exact_sum"] is True
    assert raw["q"] == 3
    assert raw["region_size"] == 1
    assert raw["system_size"] == 2
    assert raw["trace_rho_q"] > 0.0
    assert raw["sq_err"] == 0.0
    assert raw["trace_err"] == 0.0


def test_spin_half_exact_helpers_match_sampler_binary_state_convention():
    _, nqs, _ = _build_exact_cat_state_nqs()

    entropy_basis = _enumerate_basis_states_nqs(nqs)
    spectral_basis = enumerate_basis_states(nqs)
    (_, _), (samples, _), _ = nqs.sample(num_samples=8, num_chains=2, reset=True)

    for states in (entropy_basis, spectral_basis, samples):
        unique_values = set(np.unique(np.asarray(states, dtype=np.float64)).tolist())
        assert unique_values.issubset({0.0, 1.0})


def test_hamiltonian_local_energy_uses_active_binary_state_convention():
    lattice = SquareLattice(dim=1, lx=2, bc="obc")
    hilbert = HilbertSpace(lattice=lattice)
    model = TransverseFieldIsing(
        lattice=lattice,
        hilbert_space=hilbert,
        j=0.0,
        hx=1.0,
        hz=0.0,
        dtype=np.complex128,
    )
    net = RBM(
        input_shape=(hilbert.ns,),
        n_hidden=2,
        dtype=jnp.complex128,
        param_dtype=jnp.complex128,
        seed=1,
    )
    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        sampler="vmc",
        batch_size=8,
        backend="jax",
        dtype=np.complex128,
        symmetrize=False,
        verbose=False,
        seed=1,
        state_representation="binary_01",
        s_numsamples=8,
        s_numchains=2,
        s_therm_steps=2,
        s_sweep_steps=1,
    )

    state_binary = np.array([0.0, 1.0], dtype=np.float64)
    bound_states, bound_vals = nqs.local_energy(state_binary)
    ref_states, ref_vals = model.get_loc_energy_jax_fun()(jnp.asarray(_binary_to_spin_pm_half(state_binary)))

    np.testing.assert_allclose(
        np.asarray(bound_states),
        (np.asarray(ref_states) > 0.0).astype(np.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(np.asarray(bound_vals), np.asarray(ref_vals), rtol=1e-10, atol=1e-10)
    assert set(np.unique(np.asarray(bound_states, dtype=np.float64)).tolist()).issubset({0.0, 1.0})


def test_trained_tfim_entropy_matches_ed_for_multiple_regions_and_q():
    hilbert, nqs, model, psi_exact, stats = _build_trained_tfim_state()

    assert getattr(nqs, "_state_representation", None) == "spin_pm"
    assert float(getattr(nqs, "_mode_repr", 0.0)) == 0.5

    assert abs(float(stats.history[-1]) - float(np.real(model.eig_val[0]))) < 1.0e-1

    mc_sq = compute_renyi_entropy(
        nqs,
        region=[0],
        q=2,
        num_samples=None,
        num_chains=None,
        independent_replicas=False,
    )
    ed_sq_single = compute_ed_entanglement_entropy(
        psi_exact,
        np.asarray([0], dtype=int),
        hilbert.ns,
        q_values=[2],
        n_states=1,
    )["renyi_2"][0]
    assert np.isfinite(mc_sq)
    assert abs(mc_sq - ed_sq_single) < 5.0e-2

    regions = ([0], [1], [0, 1])
    q_values = (2, 3, 4)
    for region in regions:
        ed = compute_ed_entanglement_entropy(
            psi_exact,
            np.asarray(region, dtype=int),
            hilbert.ns,
            q_values=list(q_values),
            n_states=1,
        )
        for q in q_values:
            nqs_sq = compute_renyi_entropy(nqs, region=region, q=q, exact_sum=True)
            ed_sq = ed[f"renyi_{q}"][0]
            tol = 4.0e-2 if len(region) == 1 else 1.0e-10
            assert abs(nqs_sq - ed_sq) < tol
