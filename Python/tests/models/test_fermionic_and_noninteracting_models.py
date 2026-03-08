"""Regression tests for fermionic and noninteracting model families."""

import numpy as np

from QES.Algebra.Model import choose_model
from QES.general_python.lattices import SquareLattice
from QES.general_python.physics.eigenlevels import gap_ratio


def _to_dense(mat):
    """Convert sparse-like matrices to dense NumPy arrays."""
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _total_number_operator(ns):
    """Return the dense total-number operator in the integer occupation basis."""
    diag = np.array([bin(state).count("1") for state in range(1 << ns)], dtype=np.float64)
    return np.diag(diag.astype(np.complex128, copy=False))


def _mean_ipr(eigenvectors):
    """Return the mean inverse participation ratio over eigenvectors."""
    vecs = np.asarray(eigenvectors, dtype=np.complex128)
    return float(np.mean(np.sum(np.abs(vecs) ** 4, axis=0)))


def test_interacting_fermionic_models_conserve_total_particle_number():
    # These maintained spinless fermion models should preserve U(1) particle
    # number, so the dense Hamiltonian must commute with the total number
    # operator built in the same integer basis.
    lat = SquareLattice(dim=1, lx=4, bc="pbc")
    number = _total_number_operator(4)

    ffm = choose_model(
        "manybody_free_fermions",
        lattice=lat,
        t=1.0,
        t2=0.2,
        mu=0.0,
        dtype=np.complex128,
    )
    ffm.build(verbose=False, force=True)
    ffm_H = _to_dense(ffm.hamil)
    assert ffm_H.shape == (16, 16)
    assert np.allclose(ffm_H, ffm_H.conj().T, atol=1e-12)
    assert np.linalg.norm(ffm_H @ number - number @ ffm_H) < 1e-10

    hub = choose_model(
        "hubbard",
        lattice=lat,
        t=1.0,
        U=1.5,
        mu=0.1,
        dtype=np.complex128,
    )
    hub.build(verbose=False, force=True)
    hub_H = _to_dense(hub.hamil)
    assert hub_H.shape == (16, 16)
    assert np.allclose(hub_H, hub_H.conj().T, atol=1e-12)
    assert np.linalg.norm(hub_H @ number - number @ hub_H) < 1e-10

    ffm_old = ffm_H.copy()
    ffm.set_couplings(t2=0.4)
    ffm.build(verbose=False, force=True)
    assert not np.allclose(_to_dense(ffm.hamil), ffm_old)

    hub_old = hub_H.copy()
    hub.set_couplings(U=2.0)
    hub.build(verbose=False, force=True)
    assert not np.allclose(_to_dense(hub.hamil), hub_old)


def test_free_fermions_matches_analytic_chain_spectrum():
    # The translationally invariant 1D free-fermion chain has a closed-form
    # cosine band. The quadratic Python model should reproduce it exactly.
    model = choose_model(
        "free_fermions",
        ns=8,
        t=1.0,
        t2=0.2,
        bc="pbc",
        dtype=np.complex128,
    )
    model.diagonalize(verbose=False)

    eig = np.sort(np.real(np.asarray(model.eig_val)))
    k = np.arange(8, dtype=np.float64)
    ref = np.sort(-2.0 * np.cos(2.0 * np.pi * k / 8.0) - 0.4 * np.cos(4.0 * np.pi * k / 8.0))

    H = np.asarray(model.build_single_particle_matrix(copy=True))
    assert H.shape == (8, 8)
    assert np.allclose(H, H.conj().T, atol=1e-12)
    assert np.allclose(eig, ref, atol=1e-12)


def test_aubry_andre_localization_strength_tracks_inverse_participation():
    # Stronger quasiperiodic potential should localize single-particle states
    # more strongly, which is visible in the mean inverse participation ratio.
    weak = choose_model(
        "aubry_andre",
        lx=8,
        ly=1,
        lz=1,
        J=1.0,
        lmbd=0.4,
        beta=(np.sqrt(5.0) - 1.0) / 2.0,
        phi=0.1,
        dtype=np.float64,
    )
    weak.diagonalize(verbose=False)

    strong = choose_model(
        "aubry_andre",
        lx=8,
        ly=1,
        lz=1,
        J=1.0,
        lmbd=3.0,
        beta=(np.sqrt(5.0) - 1.0) / 2.0,
        phi=0.1,
        dtype=np.float64,
    )
    strong.diagonalize(verbose=False)

    weak_H = np.asarray(weak.build_single_particle_matrix(copy=True))
    strong_H = np.asarray(strong.build_single_particle_matrix(copy=True))
    assert np.allclose(weak_H, weak_H.T, atol=1e-12)
    assert np.allclose(strong_H, strong_H.T, atol=1e-12)
    assert _mean_ipr(strong.eig_vec) > _mean_ipr(weak.eig_vec) + 0.3


def test_random_noninteracting_models_are_seeded_and_have_finite_statistics():
    # Keep the random quadratic and dense random-matrix families covered:
    # Hermiticity, seeded reproducibility, and finite middle-spectrum gap
    # statistics should all hold for the maintained implementations.
    syk_a = choose_model("syk2", ns=18, seed=11, dtype=np.complex128)
    syk_a.diagonalize(verbose=False)
    syk_b = choose_model("syk2", ns=18, seed=11, dtype=np.complex128)
    syk_b.diagonalize(verbose=False)
    syk_c = choose_model("syk2", ns=18, seed=12, dtype=np.complex128)
    syk_c.diagonalize(verbose=False)

    syk_H = np.asarray(syk_a.build_single_particle_matrix(copy=True))
    assert np.allclose(syk_H, syk_H.conj().T, atol=1e-12)
    assert np.allclose(syk_H, np.asarray(syk_b.build_single_particle_matrix(copy=True)))
    assert not np.allclose(syk_H, np.asarray(syk_c.build_single_particle_matrix(copy=True)))
    syk_gap = gap_ratio(np.sort(np.real(np.asarray(syk_a.eig_val))), fraction=0.5, use_mean_lvl_spacing=True)
    assert 0.4 < float(syk_gap["mean"]) < 0.8

    for name, kwargs in [
        ("plrb", dict(ns=6, a=1.0, b=2.0, seed=3, dtype=np.float64)),
        ("rpm", dict(ns=6, gamma=1.5, seed=5, dtype=np.float64)),
    ]:
        model_a = choose_model(name, **kwargs)
        model_a.build(verbose=False, force=True)
        model_b = choose_model(name, **kwargs)
        model_b.build(verbose=False, force=True)

        H_a = _to_dense(model_a.hamil)
        H_b = _to_dense(model_b.hamil)
        assert H_a.shape == (64, 64)
        assert np.allclose(H_a, H_a.T.conj(), atol=1e-12)
        assert np.allclose(H_a, H_b)

        vals = np.linalg.eigvalsh(0.5 * (H_a + H_a.T.conj()))
        stats = gap_ratio(np.sort(np.real(vals)), fraction=0.5, use_mean_lvl_spacing=True)
        assert 0.3 < float(stats["mean"]) < 0.7
