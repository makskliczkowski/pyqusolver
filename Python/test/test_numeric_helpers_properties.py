from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_autocorr_constant_series_is_well_defined():
    # Arrange
    diagnostics = _load_module("diagnostics_mod", "QES/Solver/MonteCarlo/diagnostics.py")
    x = np.full(16, 3.14159)

    # Act
    acf = diagnostics.autocorr_func_1d(x)
    tau = diagnostics.compute_autocorr_time(x)
    ess = diagnostics.compute_ess(x)

    # Assert
    np.testing.assert_allclose(acf, np.ones_like(acf))
    assert tau == pytest.approx(1.0)
    assert ess == pytest.approx(float(x.size))


@pytest.mark.parametrize(
    "chains, expected",
    [
        (np.array([[1.0, 1.1, 0.9], [1.0, 1.1, 0.9]]), np.sqrt(2.0 / 3.0)),
        (np.array([[0.0], [0.0]]), np.nan),
        (np.array([[2.0, 2.0, 2.0]]), np.nan),
    ],
)
def test_compute_rhat_edge_cases(chains, expected):
    # Arrange
    diagnostics = _load_module("diagnostics_mod_rhat", "QES/Solver/MonteCarlo/diagnostics.py")

    # Act
    rhat = diagnostics.compute_rhat(chains)

    # Assert
    if np.isnan(expected):
        assert np.isnan(rhat)
    else:
        assert rhat == pytest.approx(expected)


@pytest.mark.parametrize(
    "state, ns, site, expected",
    [
        (0b1110, 4, 1, 1),
        (0b1011, 4, 2, 1),
        (0b0001, 4, 0, 0),
    ],
)
def test_count_left_int_matches_big_endian_convention(state, ns, site, expected):
    # Arrange
    sign_mod = _load_module("sign_mod", "QES/Algebra/Operator/sign.py")

    # Act
    got = sign_mod.count_left_int(state, ns, site)

    # Assert
    assert got == expected


def test_anyon_phase_and_parity_roundtrip():
    # Arrange
    sign_mod = _load_module("sign_mod_phase", "QES/Algebra/Operator/sign.py")
    theta = np.pi / 3
    state = 0b1010

    # Act
    phase = sign_mod.anyon_phase_int(state, ns=4, site=3, statistics_angle=theta)
    parity = sign_mod.jordan_wigner_parity_int(state, ns=4, site=3)

    # Assert
    assert np.isclose(abs(phase), 1.0)
    assert parity in (-1.0, 1.0)
