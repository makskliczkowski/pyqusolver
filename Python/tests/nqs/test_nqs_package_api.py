"""Regression tests for the public QES.NQS package facade."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("h5py")

import QES.NQS as nqs_pkg


def test_nqs_package_all_exports_resolve():
    """Every public QES.NQS export should be importable."""
    for name in nqs_pkg.__all__:
        assert getattr(nqs_pkg, name) is not None, name


def test_nqs_package_does_not_export_broken_legacy_aliases():
    """Broken module/class aliases should stay out of the public facade."""
    legacy_names = {
        "NQSEntropy",
        "NQSExact",
        "load_exact_impl",
        "nqs_core",
        "nqs_dataset",
        "nqs_entropy",
        "nqs_exact",
        "nqs_precision",
    }

    assert legacy_names.isdisjoint(nqs_pkg.__all__)
    for name in legacy_names:
        with pytest.raises(AttributeError):
            getattr(nqs_pkg, name)


def test_nqs_package_dir_matches_public_exports():
    """Tab completion should expose the maintained API, not import plumbing."""
    names = set(dir(nqs_pkg))

    assert set(nqs_pkg.__all__).issubset(names)
    assert "importlib"      not in names
    assert "MISSING"        not in names
    assert "fields"         not in names
    assert "NQSDataset"     in names
    assert "EDDataset"      in names
    assert "CommonDataset"  in names
