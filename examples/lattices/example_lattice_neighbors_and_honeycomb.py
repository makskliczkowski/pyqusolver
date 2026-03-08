"""Minimal lattice-neighbor example for square and honeycomb lattices."""

from QES.general_python.lattices import HoneycombLattice, SquareLattice


def main():
    print("--- Lattice Neighbors ---")

    # Section: Build two representative lattice families
    sq = SquareLattice(dim=2, lx=3, ly=3, bc="pbc")
    hc = HoneycombLattice(lx=2, ly=2, bc="pbc")

    # Section: Inspect nearest-neighbor lists
    print("square ns:", sq.ns, "neighbors(0):", sq.get_nei(0))
    print("honeycomb ns:", hc.ns, "neighbors(0):", hc.get_nei(0))


if __name__ == "__main__":
    main()
