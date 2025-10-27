#!/usr/bin/env python3
"""
Integration tests for momentum sector analysis in HilbertSpace.

This test suite verifies that the new momentum analysis methods added to 
HilbertSpace work correctly and produce consistent results with the standalone
momentum analysis tools.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Hilbert.hilbert_local import LocalSpace
from QES.general_python.lattices.square import SquareLattice
from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.general_python.lattices.lattice import LatticeDirection
from QES.Algebra.Symmetries.translation import TranslationSymmetry
from QES.Algebra.Operator.operator import SymmetryGenerators

def test_1d_chain_momentum_analysis():
    """Test momentum analysis on a 1D chain."""
    print("\n" + "="*80)
    print("TEST 1: 1D Chain Momentum Analysis")
    print("="*80)
    
    # Create a 1D chain (4 sites)
    lat = SquareLattice(1, 4)  # dim=1, lx=4
    
    # Create Hilbert space (will analyze momentum sectors of the lattice itself)
    hs = HilbertSpace(ns=4, lattice=lat, gen_mapping=False)
    
    print(f"\nLattice: 1D chain, Lx={lat.lx}, N={lat.ns} sites")
    print(f"Hilbert space dimension: 2^{lat.ns} = {2**lat.ns}")
    
    # Analyze momentum sectors
    print("\nAnalyzing momentum sectors...")
    sectors = hs.analyze_momentum_sectors()
    
    if sectors is None:
        print("ERROR: analyze_momentum_sectors() returned None")
        return False
    
    print(f"\nFound {len(sectors)} momentum sectors")
    
    # Verify we have the expected number of sectors
    # For a 1D chain of length L, we expect L momentum sectors
    if len(sectors) != lat.lx:
        print(f"ERROR: Expected {lat.lx} sectors, got {len(sectors)}")
        return False
    
    print("✓ 1D chain momentum analysis passed")
    return True


def test_2d_square_momentum_analysis():
    """Test momentum analysis on a 2D square lattice."""
    print("\n" + "="*80)
    print("TEST 2: 2D Square Lattice Momentum Analysis")
    print("="*80)
    
    # Create a 2D square lattice (3x3)
    lat = SquareLattice(2, 3, 3)  # dim=2, lx=3, ly=3
    
    # Create Hilbert space
    hs = HilbertSpace(ns=9, lattice=lat, gen_mapping=False)
    
    print(f"\nLattice: 2D square, Lx={lat.lx}, Ly={lat.ly}, N={lat.ns} sites")
    print(f"Hilbert space dimension: 2^{lat.ns} = {2**lat.ns}")
    
    # Analyze momentum sectors
    print("\nAnalyzing momentum sectors...")
    sectors = hs.analyze_momentum_sectors()
    
    if sectors is None:
        print("ERROR: analyze_momentum_sectors() returned None")
        return False
    
    print(f"\nFound {len(sectors)} momentum sectors")
    
    # For 2D, we expect Lx × Ly momentum sectors
    expected_sectors = lat.lx * lat.ly
    if len(sectors) != expected_sectors:
        print(f"ERROR: Expected {expected_sectors} sectors, got {len(sectors)}")
        return False
    
    print("✓ 2D square lattice momentum analysis passed")
    return True


def test_honeycomb_momentum_analysis():
    """Test momentum analysis on honeycomb lattice (2 sites per unit cell)."""
    print("\n" + "="*80)
    print("TEST 3: Honeycomb Lattice Momentum Analysis (2 sites/cell)")
    print("="*80)
    
    # Create honeycomb lattice (4x1)
    lat = HoneycombLattice(dim=2, lx=4, ly=1)
    
    # Create Hilbert space (8 sites = 4 cells × 2 sites/cell)
    hs = HilbertSpace(ns=8, lattice=lat, gen_mapping=False)
    
    print(f"\nLattice: Honeycomb, Lx={lat.lx}, Ly={lat.ly}")
    print(f"Sites: {lat.ns} (= {lat.lx} × {lat.ly} × 2 sites/cell)")
    print(f"Hilbert space dimension: 2^{lat.ns} = {2**lat.ns}")
    
    # Analyze momentum sectors
    print("\nAnalyzing momentum sectors...")
    sectors = hs.analyze_momentum_sectors()
    
    if sectors is None:
        print("ERROR: analyze_momentum_sectors() returned None")
        return False
    
    print(f"\nFound {len(sectors)} momentum sectors")
    
    # Sites per cell should be auto-detected as 2
    sites_per_cell = lat.ns // (lat.lx * lat.ly)
    print(f"Auto-detected sites per cell: {sites_per_cell}")
    
    if sites_per_cell != 2:
        print(f"ERROR: Expected 2 sites/cell for honeycomb, got {sites_per_cell}")
        return False
    
    print("✓ Honeycomb lattice momentum analysis passed")
    return True


def test_comparison_with_standalone():
    """Compare HilbertSpace method results with standalone analysis."""
    print("\n" + "="*80)
    print("TEST 4: Comparison with Standalone Analysis")
    print("="*80)
    
    from QES.Algebra.Symmetries.momentum_sectors import MomentumSectorAnalyzer
    
    # Create a 2D square lattice
    lat = SquareLattice(2, 3, 2)  # dim=2, lx=3, ly=2
    
    # Create Hilbert space
    hs = HilbertSpace(ns=6, lattice=lat, gen_mapping=False)
    
    print(f"\nLattice: 2D square, Lx={lat.lx}, Ly={lat.ly}, N={lat.ns} sites")
    
    # Get results from HilbertSpace method
    print("\nUsing HilbertSpace.analyze_momentum_sectors()...")
    hs_sectors = hs.analyze_momentum_sectors()
    
    # Get results from standalone analyzer
    print("\nUsing standalone MomentumSectorAnalyzer...")
    analyzer = MomentumSectorAnalyzer(lat)
    standalone_sectors = analyzer.analyze_2d_sectors()
    
    if hs_sectors is None or standalone_sectors is None:
        print("ERROR: One of the methods returned None")
        return False
    
    # Compare number of sectors
    hs_count = len(hs_sectors)
    standalone_count = len(standalone_sectors)
    
    print(f"\nHilbertSpace method: {hs_count} sectors")
    print(f"Standalone method: {standalone_count} sectors")
    
    if hs_count != standalone_count:
        print(f"ERROR: Sector counts don't match!")
        return False
    
    print("✓ Both methods produce consistent results")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MOMENTUM SECTOR INTEGRATION TESTS")
    print("="*80)
    
    results = []
    
    try:
        results.append(("1D Chain", test_1d_chain_momentum_analysis()))
    except Exception as e:
        print(f"ERROR in test_1d_chain_momentum_analysis: {e}")
        import traceback
        traceback.print_exc()
        results.append(("1D Chain", False))
    
    try:
        results.append(("2D Square", test_2d_square_momentum_analysis()))
    except Exception as e:
        print(f"ERROR in test_2d_square_momentum_analysis: {e}")
        import traceback
        traceback.print_exc()
        results.append(("2D Square", False))
    
    try:
        results.append(("Honeycomb", test_honeycomb_momentum_analysis()))
    except Exception as e:
        print(f"ERROR in test_honeycomb_momentum_analysis: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Honeycomb", False))
    
    try:
        results.append(("Comparison", test_comparison_with_standalone()))
    except Exception as e:
        print(f"ERROR in test_comparison_with_standalone: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Comparison", False))
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80)
