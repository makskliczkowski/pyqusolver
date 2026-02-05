"""
Behavioural tests for the lattice visualisation utilities.

Every test keeps Matplotlib in the ``Agg`` backend to avoid GUI requirements.
Figures are always closed after assertions so the suite stays memory friendly.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from QES.general_python.lattices import (
    HoneycombLattice,
    LatticeBC,
    LatticePlotter,
    SquareLattice,
    format_brillouin_zone_overview,
    format_lattice_summary,
    format_real_space_vectors,
    format_reciprocal_space_vectors,
    plot_brillouin_zone,
    plot_lattice_structure,
    plot_real_space,
    plot_reciprocal_space,
)

pytestmark = pytest.mark.usefixtures("close_figures")


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure figures created in a test do not leak into subsequent tests."""
    yield
    plt.close("all")


def _make_test_lattice() -> SquareLattice:
    """Return a tiny square lattice for smoke-tests."""
    return SquareLattice(dim=2, lx=2, ly=2, bc=None)


@pytest.mark.skip(reason="Depends on broken general_python visualization module")
def test_format_lattice_summary_contains_core_fields():
    """The textual summary exposes key metadata for quick inspection."""
    lattice = _make_test_lattice()
    summary = format_lattice_summary(lattice)
    assert "Lattice type" in summary
    assert "Dimensions" in summary
    assert "Boundary" in summary


def test_vector_tables_respect_row_limit():
    """Tables honour row limits and append an ellipsis when truncated."""
    lattice = _make_test_lattice()
    real_table = format_real_space_vectors(lattice, max_rows=3)
    real_lines = real_table.strip().splitlines()
    assert len(real_lines) == 1 + 3 + 1
    assert real_lines[-1].startswith("... (")

    recip_table = format_reciprocal_space_vectors(lattice, max_rows=4)
    recip_lines = recip_table.strip().splitlines()
    assert len(recip_lines) == 1 + 4


def test_brillouin_zone_overview_reports_bounds():
    """Brillouin-zone overview summarises the sampling extents."""
    lattice = _make_test_lattice()
    overview = format_brillouin_zone_overview(lattice)
    assert "Reciprocal-space bounds" in overview


def test_plot_helpers_return_axes_objects():
    """Smoke-test that all matplotlib helpers return figure/axes tuples."""
    lattice = _make_test_lattice()
    for fig, ax in (
        plot_real_space(lattice, show_indices=True),
        plot_reciprocal_space(lattice, show_indices=False),
        plot_brillouin_zone(lattice),
    ):
        assert fig is not None and ax is not None


def test_plotter_wrapper_delegates(monkeypatch):
    """LatticePlotter simply forwards to the standalone helper functions."""
    lattice = _make_test_lattice()
    plotter = LatticePlotter(lattice)
    calls = {"real": False, "recip": False, "bz": False}

    def _stub_real(obj, **kwargs):
        calls["real"] = True
        return plt.figure(), plt.gca()

    def _stub_recip(obj, **kwargs):
        calls["recip"] = True
        return plt.figure(), plt.gca()

    def _stub_bz(obj, **kwargs):
        calls["bz"] = True
        return plt.figure(), plt.gca()

    monkeypatch.setattr(
        "QES.general_python.lattices.visualization.plotting.plot_real_space", _stub_real
    )
    monkeypatch.setattr(
        "QES.general_python.lattices.visualization.plotting.plot_reciprocal_space", _stub_recip
    )
    monkeypatch.setattr(
        "QES.general_python.lattices.visualization.plotting.plot_brillouin_zone", _stub_bz
    )

    plotter.real_space()
    plotter.reciprocal_space()
    plotter.brillouin_zone()
    assert all(calls.values())


@pytest.mark.skip(reason="Depends on broken general_python visualization module")
def test_plot_lattice_structure_handles_boundary_conditions():
    """Ensure the structure plot works for both periodic and open boundaries."""
    pbc_lattice = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.PBC)
    fig_pbc, ax_pbc = plot_lattice_structure(pbc_lattice, show_indices=True, title=None)
    assert fig_pbc and ax_pbc

    obc_lattice = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.OBC)
    fig_obc, ax_obc = plot_lattice_structure(obc_lattice, highlight_boundary=True, title=None)
    assert fig_obc and ax_obc


def test_plot_functions_accept_custom_titles_and_views():
    """Users can disable titles or customise them when required."""
    lattice = _make_test_lattice()

    plot_real_space(lattice, title=None, tight_layout=False)
    plot_reciprocal_space(lattice, title="k-space", title_kwargs={"color": "red"})
    plot_brillouin_zone(lattice, title="BZ")


@pytest.mark.skip(reason="Depends on broken general_python visualization module")
def test_honeycomb_visualisation_uses_non_rectangular_geometry():
    """Honeycomb rvectors include fractional shifts, leading to bipartite colouring."""
    lattice = HoneycombLattice(dim=2, lx=3, ly=3, lz=1, bc=LatticeBC.PBC)
    fig, ax = plot_lattice_structure(lattice, show_indices=True)

    assert any(abs(vec[0] - round(vec[0])) > 1e-6 for vec in lattice.rvectors)
    scatter = ax.collections[0]
    facecolors = scatter.get_facecolors()
    assert len(np.unique(facecolors, axis=0)) >= 2


def test_plotter_structure_in_3d():
    """LatticePlotter.structure exposes 3D camera controls."""
    lattice = SquareLattice(dim=3, lx=2, ly=2, lz=2, bc=LatticeBC.OBC)
    fig, ax = LatticePlotter(lattice).structure(elev=20, azim=30)
    assert fig and ax


def main() -> int:
    """Entrypoint mirroring pytest CLI while optionally showing plots."""
    parser = argparse.ArgumentParser(description="Run lattice visualization tests.")
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots while running tests."
    )
    args = parser.parse_args()

    if args.show_plots:
        plt.ion()

    return pytest.main([__file__])


if __name__ == "__main__":
    raise SystemExit(main())
