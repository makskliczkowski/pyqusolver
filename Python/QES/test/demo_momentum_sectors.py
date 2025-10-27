#!/usr/bin/env python
"""
Demonstration of momentum sector analysis for translation symmetry.

Run this script to see detailed momentum sector decomposition examples.

--------------------------------------------
File        : QES/test/demo_momentum_sectors.py
Description : Demo script for momentum sector analysis
Author      : Maksymilian Kliczkowski  
Date        : 2025-10-26
--------------------------------------------
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QES.Algebra.Symmetries.momentum_analysis import demonstrate_momentum_analysis

if __name__ == "__main__":
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + " "*25 + "MOMENTUM SECTOR ANALYSIS DEMONSTRATION" + " "*36 + "█")
    print("█" + " "*98 + "█")
    print("█"*100)
    
    print("\nThis demonstrates the momentum sector decomposition for translation symmetry.")
    print("For small systems, full orbit structure is shown.")
    print("For larger systems, only block size summaries are displayed.\n")
    
    # ------------------------------------------------
    # 1D Chain Examples
    # ------------------------------------------------
    print("\n" + "="*100)
    print("=" + " "*98 + "=")
    print("=" + " "*40 + "1D CHAIN SYSTEMS" + " "*43 + "=")
    print("=" + " "*98 + "=")
    print("="*100 + "\n")
    
    demonstrate_momentum_analysis(system_sizes=[4, 6, 8, 10], lattice_type='1D')
    
    # ------------------------------------------------
    # 2D Square Lattice Examples
    # ------------------------------------------------
    print("\n" + "="*100)
    print("=" + " "*98 + "=")
    print("=" + " "*37 + "2D SQUARE LATTICE SYSTEMS" + " "*37 + "=")
    print("=" + " "*98 + "=")
    print("="*100 + "\n")
    
    print("Note: For 2D lattices, we analyze translation in x-direction only.")
    print("Full 2D momentum analysis would require both T_x and T_y symmetries.\n")
    
    # Small 2D systems: 2x2 (4 sites), 2x3 (6 sites), 3x3 (9 sites)
    demonstrate_momentum_analysis(system_sizes=[4, 6, 9, 12], lattice_type='2D_square')
    
    # ------------------------------------------------
    # Honeycomb Lattice Examples
    # ------------------------------------------------
    print("\n" + "="*100)
    print("=" + " "*98 + "=")
    print("=" + " "*38 + "HONEYCOMB LATTICE SYSTEMS" + " "*36 + "=")
    print("=" + " "*98 + "=")
    print("="*100 + "\n")
    
    print("Note: Honeycomb lattice has 2 sites per unit cell.")
    print("We analyze translation in x-direction (along unit cell rows).\n")
    
    # Small honeycomb systems
    demonstrate_momentum_analysis(system_sizes=[4, 6, 8, 12], lattice_type='honeycomb')
    
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + " "*40 + "END OF DEMONSTRATION" + " "*38 + "█")
    print("█" + " "*98 + "█")
    print("█"*100 + "\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
