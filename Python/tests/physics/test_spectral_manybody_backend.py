import numpy as np
import pytest

try:
    from QES.Algebra.Model.Interacting.Spin.xxz import XXZ
    from QES.Algebra.Operator.impl.operators_spin import sig_k
    from QES.Algebra.hilbert import HilbertSpace
    from QES.general_python.lattices import SquareLattice
    from QES.general_python.physics.spectral.spectral_function import spectral_function
except ImportError:
    pytest.skip("Required QES spectral modules are not available.", allow_module_level=True)


def test_manybody_spectral_helpers_are_finite_and_nontrivial():
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
    model.diagonalize()

    probe = sig_k(np.pi, lattice=lattice, ns=hilbert.ns)
    probe_matrix = probe.compute_matrix(
        hilbert_1=hilbert,
        matrix_type="dense",
        use_numpy=True,
    )
    eigvecs = np.asarray(model._eig_vec)
    projected_probe = eigvecs.conj().T @ probe_matrix @ eigvecs[:, 0]
    omega = np.linspace(0.0, 4.0, 33)

    structure_factor = model.spectral.dynamic_structure_factor(
        omega,
        probe_matrix,
        state_idx=0,
        eta=0.1,
        use_lanczos=False,
    )
    spectral_vals, greens_vals = spectral_function(
        omega=omega,
        eta=0.1,
        eigenvalues=np.asarray(model._eig_val),
        operator_a=np.asarray(projected_probe),
        mb_states=0,
    )

    assert structure_factor.shape == omega.shape
    assert spectral_vals.shape == omega.shape
    assert greens_vals.shape == omega.shape
    assert np.all(np.isfinite(structure_factor))
    assert np.all(np.isfinite(spectral_vals))
    assert np.all(np.isfinite(np.real(greens_vals)))
    assert np.all(np.isfinite(np.imag(greens_vals)))
    assert float(np.max(np.abs(structure_factor))) > 1e-8
    assert float(np.max(np.abs(spectral_vals))) > 1e-8

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------