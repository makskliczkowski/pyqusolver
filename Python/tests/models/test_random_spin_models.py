"""Regression tests for random interacting spin-model diagnostics."""

import numpy as np

from QES.Algebra.Model import choose_model
from QES.general_python.physics.eigenlevels import entropy_vonNeuman, gap_ratio
from QES.general_python.physics.spectral.spectral_backend import (
    greens_function_diagonal,
    operator_spectral_function_multi_omega,
)


def _to_dense(mat):
    """Convert sparse-like matrices to dense NumPy arrays."""
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _build_dense_model(name, **kwargs):
    """Instantiate and build one model, then return the dense Hamiltonian matrix."""
    model = choose_model(name, **kwargs)
    model.build(verbose=False, force=True)
    return model, _to_dense(model.hamil)


def _hermitian_eigensystem(H):
    """Return sorted eigenvalues and eigenvectors of a dense Hermitian matrix."""
    Hh = 0.5 * (H + H.conj().T)
    vals, vecs = np.linalg.eigh(Hh)
    return np.real(vals), vecs.astype(np.complex128, copy=False)


def _middle_indices(n, fraction=0.2):
    """Return a centered index window for middle-spectrum diagnostics."""
    width = max(3, int(round(float(fraction) * int(n))))
    center = int((int(n) + 1) // 2)
    start = max(0, center - width // 2 - 1)
    stop = min(int(n), start + width)
    start = max(0, stop - width)
    return slice(start, stop)


def _local_sz_operator(ns, site):
    """Return dense local sigma-z operator in the integer basis."""
    diag = np.empty(1 << ns, dtype=np.float64)
    mask = 1 << int(site)
    for state in range(1 << ns):
        diag[state] = -1.0 if (state & mask) else 1.0
    return np.diag(diag.astype(np.complex128))


def _diag_expectations(eigenvectors, operator):
    """Compute diagonal eigenstate expectation values of a dense operator."""
    return np.array(
        [np.real(np.vdot(eigenvectors[:, i], operator @ eigenvectors[:, i])) for i in range(eigenvectors.shape[1])],
        dtype=np.float64,
    )


def _collapse_dos(omega, eigenvalues, eta=0.08):
    """Collapse the level-resolved diagonal spectral helper to a DOS curve."""
    arr = -np.imag(np.asarray(greens_function_diagonal(omega, eigenvalues, eta=eta))) / np.pi
    if arr.shape == (len(eigenvalues), len(omega)):
        return np.sum(arr, axis=0)
    if arr.shape == (len(omega), len(eigenvalues)):
        return np.sum(arr, axis=1)
    raise AssertionError(f"Unexpected spectral_function_diagonal shape {arr.shape}")


def _dos_integral(omega, spectral):
    """Approximate integrated spectral weight on a uniform grid."""
    return float(np.sum(spectral) * float(omega[1] - omega[0]))


def test_random_spin_models_build_are_deterministic_and_hermitian():
    # QSM should stay Hermitian for complex dtype, and seeded builds must be
    # reproducible across repeated instantiations.
    _, qsm_a = _build_dense_model(
        "qsm",
        ns=6,
        n=2,
        gamma=0.4,
        g0=0.7,
        a=0.6,
        h=0.8,
        xi=0.2,
        seed=17,
        dtype=np.complex128,
    )
    _, qsm_b = _build_dense_model(
        "qsm",
        ns=6,
        n=2,
        gamma=0.4,
        g0=0.7,
        a=0.6,
        h=0.8,
        xi=0.2,
        seed=17,
        dtype=np.complex128,
    )
    _, qsm_c = _build_dense_model(
        "qsm",
        ns=6,
        n=2,
        gamma=0.4,
        g0=0.7,
        a=0.6,
        h=0.8,
        xi=0.2,
        seed=18,
        dtype=np.complex128,
    )
    assert qsm_a.shape == (64, 64)
    assert np.allclose(qsm_a, qsm_a.conj().T, atol=1e-10)
    assert np.allclose(qsm_a, qsm_b)
    assert not np.allclose(qsm_a, qsm_c)

    _, ultra_a = _build_dense_model(
        "ultrametric",
        ns=6,
        n=2,
        J=1.0,
        alphas=[0.5, 0.25, 0.125, 0.0625],
        gamma=0.7,
        seed=11,
        dtype=np.float64,
    )
    _, ultra_b = _build_dense_model(
        "ultrametric",
        ns=6,
        n=2,
        J=1.0,
        alphas=[0.5, 0.25, 0.125, 0.0625],
        gamma=0.7,
        seed=11,
        dtype=np.float64,
    )
    _, ultra_c = _build_dense_model(
        "ultrametric",
        ns=6,
        n=2,
        J=1.0,
        alphas=[0.5, 0.25, 0.125, 0.0625],
        gamma=0.7,
        seed=12,
        dtype=np.float64,
    )
    assert ultra_a.shape == (64, 64)
    assert np.allclose(ultra_a, ultra_a.conj().T, atol=1e-12)
    assert np.allclose(ultra_a, ultra_b)
    assert not np.allclose(ultra_a, ultra_c)


def test_random_spin_models_entropy_and_middle_spectrum_diagnostics():
    # QSM is the maintained quantum-sun surrogate: middle-spectrum states should
    # be substantially more entangled than edge states. Ultrametric should stay
    # weakly entangled and basis-local in the same diagnostics.
    _, qsm_H = _build_dense_model(
        "qsm",
        ns=6,
        n=2,
        gamma=0.4,
        g0=0.7,
        a=0.6,
        h=0.8,
        xi=0.2,
        seed=17,
        dtype=np.complex128,
    )
    _, ultra_H = _build_dense_model(
        "ultrametric",
        ns=6,
        n=2,
        J=1.0,
        alphas=[0.5, 0.25, 0.125, 0.0625],
        gamma=0.7,
        seed=11,
        dtype=np.float64,
    )

    qvals, qvecs = _hermitian_eigensystem(qsm_H)
    uvals, uvecs = _hermitian_eigensystem(ultra_H)
    qmid = _middle_indices(len(qvals), fraction=0.2)
    umid = _middle_indices(len(uvals), fraction=0.2)

    qexp = _diag_expectations(qvecs, _local_sz_operator(6, 5))
    uexp = _diag_expectations(uvecs, _local_sz_operator(6, 0))

    qmid_entropy = np.array(
        [entropy_vonNeuman(qvecs[:, i], 6, 3, TYP="SCHMIDT") for i in range(*qmid.indices(len(qvals)))]
    )
    qedge_entropy = np.array(
        [entropy_vonNeuman(qvecs[:, i], 6, 3, TYP="SCHMIDT") for i in range(qmid.stop - qmid.start)]
    )
    umid_entropy = np.array(
        [entropy_vonNeuman(uvecs[:, i], 6, 3, TYP="SCHMIDT") for i in range(*umid.indices(len(uvals)))]
    )

    qgr = gap_ratio(qvals, fraction=0.3, use_mean_lvl_spacing=True)
    ugr = gap_ratio(uvals, fraction=0.3, use_mean_lvl_spacing=True)

    assert float(qgr["mean"]) > 0.35
    assert float(np.mean(qmid_entropy)) > float(np.mean(qedge_entropy)) + 0.2
    assert float(np.mean(np.abs(qexp[qmid]))) < 0.35

    assert float(ugr["mean"]) < 0.45
    assert float(np.mean(umid_entropy)) < 0.2
    assert float(np.mean(np.abs(uexp[umid]))) > 0.45


def test_random_spin_model_spectral_diagnostics_are_finite():
    # Keep the operator spectral path covered for both maintained random spin
    # families, and ensure the broadened DOS carries the expected total weight.
    for name, kwargs, site in [
        (
            "qsm",
            dict(ns=6, n=2, gamma=0.4, g0=0.7, a=0.6, h=0.8, xi=0.2, seed=17, dtype=np.complex128),
            5,
        ),
        (
            "ultrametric",
            dict(ns=6, n=2, J=1.0, alphas=[0.5, 0.25, 0.125, 0.0625], gamma=0.7, seed=11, dtype=np.float64),
            0,
        ),
    ]:
        _, H = _build_dense_model(name, **kwargs)
        vals, vecs = _hermitian_eigensystem(H)
        omega = np.linspace(float(vals.min()) - 0.5, float(vals.max()) + 0.5, 121)
        dos = _collapse_dos(omega, vals, eta=0.08)

        assert dos.shape == omega.shape
        assert np.all(np.isfinite(dos))
        assert 0.9 * len(vals) < _dos_integral(omega, dos) < 1.1 * len(vals)

        op = _local_sz_operator(6, site)
        spec = operator_spectral_function_multi_omega(
            omega,
            vals,
            vecs,
            op,
            eta=0.08,
            temperature=0.0,
            backend="default",
        )
        assert spec.shape == omega.shape
        assert np.all(np.isfinite(spec))
        if name == "qsm":
            assert float(np.max(np.abs(spec))) > 1e-2
