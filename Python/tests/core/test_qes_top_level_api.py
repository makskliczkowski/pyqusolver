import warnings

import pytest

import QES

# ------------------------------------------------------------------
# Stable import surface
# ------------------------------------------------------------------


def test_stable_top_level_imports_resolve_without_deprecation():
    # Arrange
    stable_names = [
        "Algebra",
        "NQS",
        "Solver",
        "general_python",
        "HilbertSpace",
        "Hamiltonian",
        "Operator",
        "Timer",
    ]

    # Act / Assert
    for name in stable_names:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", DeprecationWarning)
            obj = getattr(QES, name)

        assert obj is not None
        assert not recorded


@pytest.mark.parametrize("name", ["gp_physics", "gp_lattices", "NQS_Model"])
def test_legacy_top_level_aliases_warn_once_on_access(name):
    # Legacy aliases stay alive only as compatibility shims.
    # Act
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        obj = getattr(QES, name)

    # Assert
    assert obj is not None
    assert recorded
    assert any(item.category is DeprecationWarning for item in recorded)
    assert "legacy compatibility alias" in str(recorded[0].message)


def test_deprecated_rng_wrappers_forward_and_warn(monkeypatch):
    # The old JAX-named helpers should forward to the current RNG surface.
    # Arrange
    sentinel_key = object()
    sentinel_split = ("k0", "k1", "k2")

    monkeypatch.setattr(QES, "qes_next_key", lambda: sentinel_key)
    monkeypatch.setattr(QES, "qes_split_keys", lambda n: sentinel_split[:n])

    # Act
    with warnings.catch_warnings(record=True) as key_recorded:
        warnings.simplefilter("always", DeprecationWarning)
        got_key = QES.next_jax_key()

    with warnings.catch_warnings(record=True) as split_recorded:
        warnings.simplefilter("always", DeprecationWarning)
        got_split = QES.split_jax_keys(2)

    # Assert
    assert got_key is sentinel_key
    assert got_split == sentinel_split[:2]
    assert key_recorded
    assert split_recorded
    assert "deprecated" in str(key_recorded[0].message).lower()
    assert "deprecated" in str(split_recorded[0].message).lower()


def test_dir_surface_hides_import_plumbing():
    # dir(QES) should expose the API, not internal typing/import helpers.
    # Act
    names = dir(QES)

    # Assert
    assert "Any"            not in names
    assert "Dict"           not in names
    assert "Optional"       not in names
    assert "importlib"      not in names
    assert "contextmanager" not in names
    assert "Algebra"        in names
    assert "qes_reseed"     in names
