"""Random interacting spin-model diagnostics for QSM and ultrametric ensembles."""

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
    raise RuntimeError(f"Unexpected spectral_function_diagonal shape {arr.shape}")


def _dos_integral(omega, spectral):
    """Approximate integrated spectral weight on a uniform grid."""
    return float(np.sum(spectral) * float(omega[1] - omega[0]))


def main():
    print("--- Random Spin Models Diagnostics ---")

    # Section: Deterministic model construction
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

    # Section: Middle-spectrum entropy and local diagnostics
    qvals, qvecs = _hermitian_eigensystem(qsm_H)
    uvals, uvecs = _hermitian_eigensystem(ultra_H)
    qmid = _middle_indices(len(qvals), fraction=0.2)
    umid = _middle_indices(len(uvals), fraction=0.2)

    qexp = _diag_expectations(qvecs, _local_sz_operator(6, 5))
    uexp = _diag_expectations(uvecs, _local_sz_operator(6, 0))

    qmid_entropy = np.array([entropy_vonNeuman(qvecs[:, i], 6, 3, TYP="SCHMIDT") for i in range(*qmid.indices(len(qvals)))])
    umid_entropy = np.array([entropy_vonNeuman(uvecs[:, i], 6, 3, TYP="SCHMIDT") for i in range(*umid.indices(len(uvals)))])

    qgr = gap_ratio(qvals, fraction=0.3, use_mean_lvl_spacing=True)
    ugr = gap_ratio(uvals, fraction=0.3, use_mean_lvl_spacing=True)

    print("QSM middle entropy mean:", float(np.mean(qmid_entropy)))
    print("QSM middle local <Sz_5>: mean(abs)=", float(np.mean(np.abs(qexp[qmid]))), "std=", float(np.std(qexp[qmid])))
    print("QSM middle gap ratio mean/std:", float(qgr["mean"]), float(qgr["std"]))

    print("Ultrametric middle entropy mean:", float(np.mean(umid_entropy)))
    print("Ultrametric middle local <Sz_0>: mean(abs)=", float(np.mean(np.abs(uexp[umid]))), "std=", float(np.std(uexp[umid])))
    print("Ultrametric middle gap ratio mean/std:", float(ugr["mean"]), float(ugr["std"]))

    # Section: DOS and operator spectral functions
    qomega = np.linspace(float(qvals.min()) - 0.5, float(qvals.max()) + 0.5, 121)
    uomega = np.linspace(float(uvals.min()) - 0.5, float(uvals.max()) + 0.5, 121)
    qdos = _collapse_dos(qomega, qvals, eta=0.08)
    udos = _collapse_dos(uomega, uvals, eta=0.08)
    qspec = operator_spectral_function_multi_omega(qomega, qvals, qvecs, _local_sz_operator(6, 5), eta=0.08)
    uspec = operator_spectral_function_multi_omega(uomega, uvals, uvecs, _local_sz_operator(6, 0), eta=0.08)

    print("QSM DOS integrated weight:", _dos_integral(qomega, qdos))
    print("QSM operator spectral peak:", float(np.max(np.abs(qspec))))
    print("Ultrametric DOS integrated weight:", _dos_integral(uomega, udos))
    print("Ultrametric operator spectral peak:", float(np.max(np.abs(uspec))))


if __name__ == "__main__":
    main()
