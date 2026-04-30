"""Regression tests for package-level lazy facades."""

import pytest

import QES.Algebra as algebra_pkg
import QES.general_python as gp_pkg


def test_algebra_public_exports_resolve():
    """QES.Algebra.__all__ should contain only resolvable maintained exports."""
    for name in algebra_pkg.__all__:
        assert getattr(algebra_pkg, name) is not None, name


def test_algebra_dir_hides_legacy_symmetry_fallbacks_and_plumbing():
    """QES.Algebra should not expose all Symmetries names at package top level."""
    names = set(dir(algebra_pkg))

    assert set(algebra_pkg.__all__) == names
    assert "importlib"              not in names
    assert "TYPE_CHECKING"          not in names
    assert "TranslationSymmetry"    not in names
    assert "SymmetryContainer"      not in names
    with pytest.raises(AttributeError):
        getattr(algebra_pkg, "TranslationSymmetry")


def test_general_python_public_exports_resolve():
    """QES.general_python.__all__ should contain only resolvable maintained exports."""
    for name in gp_pkg.__all__:
        assert getattr(gp_pkg, name) is not None, name


def test_general_python_dir_hides_legacy_gp_aliases_and_plumbing():
    """The utility facade should expose subpackages and stable helpers only."""
    names = set(dir(gp_pkg))

    assert set(gp_pkg.__all__) == names
    assert "importlib"              not in names
    assert "TYPE_CHECKING"          not in names
    assert "gp_algebra"             not in names
    assert "gp_common"              not in names
    assert "gp_lattices"            not in names
    assert "gp_maths"               not in names
    assert "gp_ml"                  not in names
    assert "gp_physics"             not in names
    with pytest.raises(AttributeError):
        getattr(gp_pkg, "gp_algebra")


def test_general_python_capability_lists_are_canonical():
    """Capability helpers should not duplicate entries or list removed aliases."""
    capabilities = gp_pkg.list_capabilities()

    assert capabilities == sorted(set(capabilities))
    assert "algebra"                in capabilities
    assert "random"                 in capabilities
    assert "gp_algebra"             not in capabilities
    assert gp_pkg.list_available_modules() == [
        "algebra",
        "common",
        "lattices",
        "maths",
        "ml",
        "physics",
    ]

# -----------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------
