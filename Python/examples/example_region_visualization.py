"""
Demonstration of enhanced region visualization for debugging Kitaev-Preskill
topological entanglement entropy calculations.

This example shows:
1. How to define regions manually for small lattices
2. How to visualize region construction with overlaps and complements
3. How to validate regions before entropy computation
"""

import matplotlib.pyplot as plt

from QES.general_python.lattices.lattice import SquareLattice


def create_kitaev_preskill_regions_3x3(lattice):
    """
    Manually define Kitaev-Preskill regions A, B, C for a 3×3 lattice.

    Kitaev-Preskill construction:
    - Region A: left portion
    - Region B: right portion
    - Region C: top portion
    - Overlaps: ABC, AB, AC, BC

    Returns dict with region names and site indices.
    """
    # Get site indices (assuming row-major ordering: [0,0], [0,1], [0,2], [1,0], ...)
    # 3×3 lattice has 9 sites total
    # Site layout:
    # 0 1 2
    # 3 4 5
    # 6 7 8

    regions = {
        "A": [0, 1, 3, 4],  # Left 2 columns, top 2 rows
        "B": [1, 2, 4, 5],  # Right 2 columns, top 2 rows
        "C": [0, 1, 2],  # Top row
    }

    return regions


def validate_region_construction(lattice, regions):
    """
    Validate that regions are properly constructed for TEE calculation.

    Checks:
    - No region covers entire system (would give 0 entropy)
    - Regions have overlaps (needed for MI calculation)
    - Coverage statistics
    """
    n_sites = lattice.n
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
            print("      This will give S=0 (pure state)")

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
        print("     Kitaev-Preskill TEE requires overlapping regions")

    # Check complement
    complement = set(range(n_sites)) - all_sites
    if complement:
        print(f"\nComplement (region D): {len(complement)} sites {sorted(complement)}")

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


def visualize_regions_enhanced(lattice, regions):
    """Enhanced visualization showing overlaps and complement."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    lattice.plot.regions(
        regions=regions,
        ax=ax,
        show_system=True,
        show_complement=True,
        show_labels=True,
        show_overlaps=True,
        fill=True,
        fill_alpha=0.15,
        marker_size=250,
        edge_width=2,
        complement_color="gray",
        complement_alpha=0.3,
        overlap_color="red",
    )
    fig.suptitle("Enhanced Region Plot (Overlaps + Complement)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def visualize_individual_regions(lattice, regions):
    """Show each region individually to understand structure."""
    n_regions = len(regions)
    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 5))
    if n_regions == 1:
        axes = [axes]

    for ax, (name, indices) in zip(axes, regions.items()):
        # Plot just this region
        single_region = {name: indices}
        lattice.plot.regions(
            regions=single_region,
            ax=ax,
            show_system=True,
            show_complement=True,
            show_labels=True,
            show_overlaps=False,
            fill=True,
            fill_alpha=0.3,
            marker_size=300,
            edge_width=2,
        )
        ax.set_title(f"Region {name}", fontsize=12, fontweight="bold")

    fig.suptitle("Individual Region Views", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    """Demonstrate region visualization for debugging TEE calculations."""

    print("\n" + "=" * 60)
    print("KITAEV-PRESKILL REGION VISUALIZATION DEMO")
    print("=" * 60)

    # Create 3×3 square lattice
    lattice = SquareLattice(Lx=3, Ly=3, Lz=1, boundary="open")
    print(f"\nLattice: {lattice.n} sites")

    # Define Kitaev-Preskill regions
    regions = create_kitaev_preskill_regions_3x3(lattice)

    # Validate regions
    validate_region_construction(lattice, regions)

    # Create visualizations
    print("\nGenerating visualizations...")

    fig1 = visualize_regions_basic(lattice, regions)
    fig2 = visualize_regions_enhanced(lattice, regions)
    fig3 = visualize_individual_regions(lattice, regions)

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
For Kitaev-Preskill topological entanglement entropy:

1. OVERLAPS (red circles):
   - Sites appearing in multiple regions
   - Critical for computing mutual information terms
   - Should include ABC, AB, AC, BC overlaps

2. COMPLEMENT (gray X's):
   - Sites not in any region (region D)
   - Can be used for validation
   - Not required for basic KP construction

3. COVERAGE:
   - Should be < 100% (otherwise pure state → S=0)
   - Each region should be a proper subsystem
   
4. EXPECTED TEE VALUE:
   - For Kitaev toric code ground state: ln(2) ≈ 0.693
   - NOT sqrt(2) ≈ 1.414
   - Value of 0 indicates pure state or region issues

If you get unexpected TEE values:
- Check that no single region covers entire system
- Verify overlaps exist between regions
- Ensure subsystem encoding matches region definition
- Compare symmetric vs non-symmetric paths with same regions
    """)
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
