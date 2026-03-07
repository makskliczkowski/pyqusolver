import numpy as np

from QES.general_python.physics.density_matrix import rho, rho_spectrum
from QES.general_python.physics.entropy import Entanglement, entropy, mutual_information
from QES.general_python.physics.eigenlevels import gap_ratio
from QES.Algebra.Properties.statistical import dos, ldos


def test_bell_state_density_matrix_entropy_and_mutual_information():
    psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    psi /= np.linalg.norm(psi)

    rho_a = rho(psi, va=[0], ns=2)
    lam = rho_spectrum(rho_a)
    s_vn = entropy(lam, q=1.0, typek=Entanglement.VN)
    s_r2 = entropy(lam, q=2.0, typek=Entanglement.RENYI)
    mi, pieces = mutual_information(psi, 0, 1, 2, q=1.0)

    np.testing.assert_allclose(lam, np.array([0.5, 0.5]), rtol=1e-12, atol=1e-12)
    assert np.isclose(np.real(s_vn), np.log(2.0))
    assert np.isclose(np.real(s_r2), np.log(2.0))
    assert np.isclose(np.real(mi), 2.0 * np.log(2.0))
    assert len(pieces) == 3


def test_gap_ratio_matches_current_python_reference_values():
    vals = np.array([-2.0, -1.25, -0.2, 0.7, 1.5, 2.4, 3.1], dtype=float)
    stats = gap_ratio(vals, fraction=0.8, use_mean_lvl_spacing=True)

    assert np.isclose(float(stats["mean"]), 0.873015873015873)
    assert np.isclose(float(stats["std"]), 0.015873015873015928)
    assert len(stats["vals"]) == 2


def test_ldos_and_dos_shapes_are_stable():
    energies = np.array([-2.0, -1.0, -0.2, 0.1, 0.9, 1.8], dtype=float)
    overlaps = np.array([0.2, 0.4, 0.3, 0.6, 0.5, 0.2], dtype=np.complex128)

    local = ldos(energies, overlaps, degenerate=False)
    density = dos(energies, nbins=6)

    np.testing.assert_allclose(local, np.abs(overlaps) ** 2)
    assert density.shape == (6,)
    assert density.sum() == energies.size
