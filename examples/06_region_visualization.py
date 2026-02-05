"""
Region Visualization Example
============================

Demonstrates how to define and visualize regions on a lattice, which is useful
for tasks like computing topological entanglement entropy (Kitaev-Preskill).

To run:
    python examples/06_region_visualization.py
"""

import matplotlib.pyplot as plt
import QES
# Fixed import
from QES.general_python.lattices.square import SquareLattice

def create_kitaev_preskill_regions_3x3(lattice):
    """
    Manually define Kitaev-Preskill regions A, B, C for a 3×3 lattice.
    """
    # Site layout (9 sites):
    # 0 1 2
    # 3 4 5
    # 6 7 8

    regions = {
        "A": [0, 1, 3, 4],  # Left 2 columns, top 2 rows
        "B": [1, 2, 4, 5],  # Right 2 columns, top 2 rows
        "C": [0, 1, 2],     # Top row
    }
    return regions

def validate_region_construction(lattice, regions):
    """
    Validate that regions are properly constructed.
    """
    n_sites = lattice.ns
    all_sites = set()
    site_counts = {}

    print("\n" + "=" * 60)
    print("REGION VALIDATION")
    print("=" * 60)

    for name, indices in regions.items():
        n_in_region = len(indices)
        coverage = n_in_region / n_sites * 100
        print(f"Region {name}: {n_in_region} sites ({coverage:.1f}% coverage)")

        if coverage >= 100:
            print(f"  ⚠️  WARNING: Region {name} covers entire system!")

        for idx in indices:
            all_sites.add(idx)
            site_counts[idx] = site_counts.get(idx, 0) + 1

    # Check overlaps
    overlaps = {idx: count for idx, count in site_counts.items() if count > 1}
    print(f"\nTotal unique sites covered: {len(all_sites)}/{n_sites}")
    print(f"Overlapping sites: {len(overlaps)}")

    if overlaps:
        print("Overlap details:")
        for idx, count in sorted(overlaps.items()):
            regions_with_site = [name for name, indices in regions.items() if idx in indices]
            print(f"  Site {idx}: in {count} regions {regions_with_site}")
    else:
        print("  ⚠️  WARNING: No overlaps detected!")

    print("=" * 60)

def visualize_regions_basic(lattice, regions):
    """Basic visualization without special features."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    lattice.plot.regions(
        regions=regions,
        ax=ax,
        show_system=True,
        show_complement=False,
        show_labels=True,
        show_overlaps=False,
        fill=True,
        fill_alpha=0.2,
        marker_size=200,
    )
    fig.suptitle("Basic Region Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

def main():
    print("\n" + "=" * 60)
    print("KITAEV-PRESKILL REGION VISUALIZATION DEMO")
    print("=" * 60)

    # Create 3×3 square lattice
    # Fix: Use bc='obc' and dim=2 (to workaround SquareLattice bug)
    lattice = SquareLattice(lx=3, ly=3, lz=1, bc="obc", dim=2)
    print(f"\nLattice: {lattice.ns} sites")

    # Define Kitaev-Preskill regions
    regions = create_kitaev_preskill_regions_3x3(lattice)

    # Validate regions
    validate_region_construction(lattice, regions)

    # Create visualizations
    print("\nGenerating visualizations... (close window to finish)")

    fig1 = visualize_regions_basic(lattice, regions)

    # In a script, we typically show and block, but for automated testing we might just save or return.
    # We'll just print that we are done.
    # plt.show()
    print("Visualization figures created.")

if __name__ == "__main__":
    main()
