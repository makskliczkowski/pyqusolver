"""Regression tests for nqs dynamic structure factor."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.Algebra.Operator.impl.operators_spin import sig_k
    from QES.Algebra.hilbert import HilbertSpace
    from QES.NQS.nqs import NQS
    from QES.NQS.src.nqs_spectral import _diagonal_probe_overlap_operator, _exact_expectation_value
    from QES.general_python.lattices import SquareLattice
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
except ImportError:
    pytest.skip("Required QES modules are not available.", allow_module_level=True)


def _build_uniform_tfim_state(seed: int = 0):
    """Build uniform tfim state."""
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
    """Verify test dynamic structure factor modifier smoke."""
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
        hermitian_extension=True,
    )
    spectral_gaussian = nqs.dynamic_structure_factor(
        times,
        ket_probe_operator=probe_q,
        bra_probe_operator=probe_mq,
        exact_sum=True,
        num_samples=16,
        num_chains=4,
        n_batch=32,
        diag_shift=1e-3,
        sr_maxiter=64,
        eta=2.0 / times[-1],
        broadening_kind="gaussian",
        integration_rule="trapezoid",
        window=None,
        restore=True,
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
    assert spectral_gaussian.correlator.shape == times.shape
    assert spectral.frequencies.shape == spectral.spectrum.shape
    assert np.allclose(spectral.frequencies, spectral_from_corr.frequencies)
    assert np.allclose(spectral.spectrum, spectral_from_corr.spectrum, atol=1e-10, rtol=1e-10)
    assert np.all(np.isfinite(np.real(spectral.correlator)))
    assert np.all(np.isfinite(np.imag(spectral.correlator)))
    assert np.all(np.isfinite(np.real(spectral_gaussian.spectrum_complex)))
    assert np.all(np.isfinite(np.imag(spectral_gaussian.spectrum_complex)))
    assert spectral_gaussian.metadata["broadening_kind"] == "gaussian"
    assert spectral_gaussian.metadata["integration_rule"] == "trapezoid"
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
    exact_kernel_weight = _exact_expectation_value(
        nqs,
        _diagonal_probe_overlap_operator(probe_mq, probe_q),
    )

    assert abs(corr.correlator[0] - exact_static_weight) < 1e-8
    assert abs(spectral.correlator[0] - exact_static_weight) < 1e-8
    assert abs(spectral.metadata["static_weight"] - exact_static_weight) < 1e-8
    assert abs(exact_kernel_weight - exact_static_weight) < 1e-8


def test_spawn_like_preserves_nqs_state_convention():
    """Verify test spawn like preserves nqs state convention."""
    _, _, _, nqs = _build_uniform_tfim_state(seed=3)
    spawned = nqs.spawn_like(use_orbax=False, verbose=False)

    assert nqs.state_representation == nqs.sampler.state_representation
    assert spawned.state_representation == nqs.state_representation
    assert spawned.state_convention["spin"] == nqs.state_convention["spin"]
    assert spawned.state_convention["mode_repr"] == nqs.state_convention["mode_repr"]


def test_nqs_resolve_operator_caches_bound_operator():
    """Verify test nqs resolve operator caches bound operator."""
    _, _, _, nqs = _build_uniform_tfim_state(seed=5)

    class _BoundProbe:
        """Test helper class for BoundProbe."""
        def __init__(self):
            """Helper for init."""
            self.calls = 0

        def bind_state_convention(self, convention):
            """Helper for bind state convention."""
            self.calls += 1
            return ("bound", convention["representation"], convention["mode_repr"])

    probe = _BoundProbe()
    resolved_a = nqs.resolve_operator(probe)
    resolved_b = nqs.resolve_operator(probe)

    assert resolved_a == resolved_b
    assert probe.calls == 1


def test_compute_observable_resolves_state_bound_operator():
    """Verify test compute observable resolves state bound operator."""
    _, _, _, nqs = _build_uniform_tfim_state(seed=7)
    (_, _), (states, log_psi), probabilities = nqs.sample(num_samples=8, num_chains=4)

    class _ObservableProbe:
        """Test helper class for ObservableProbe."""
        def __init__(self):
            """Helper for init."""
            self.calls = 0

        def bind_state_convention(self, convention):
            """Helper for bind state convention."""
            self.calls += 1
            offset = 1.0 if convention["representation"] == "binary_01" else -1.0

            def _kernel(s):
                """Helper for kernel."""
                s = jnp.asarray(s)
                return s[..., 0] * 0.0 + offset

            return _kernel

    probe = _ObservableProbe()
    result = nqs.compute_observable(
        states=states,
        ansatze=log_psi,
        probabilities=probabilities,
        functions=probe,
        return_stats=True,
    )

    assert probe.calls == 1
    assert np.isfinite(result.mean)
    expected = 1.0 if nqs.state_representation == "binary_01" else -1.0
    assert np.isclose(result.mean, expected)
