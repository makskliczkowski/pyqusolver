#!/usr/bin/env python3
"""
Demonstration of 2D Momentum Sector Analysis

Shows how momentum sectors decompose under simultaneous X and Y translations
for 2D lattices (square and honeycomb).
"""

from QES.Algebra.Symmetries.momentum_analysis_2d import demonstrate_2d_momentum_analysis

if __name__ == '__main__':
    
    print("\n" + "="*100)
    print("==" + " "*96 + "==")
    print("==" + "2D MOMENTUM SECTOR ANALYSIS DEMONSTRATION".center(96) + "==")
    print("==" + " "*96 + "==")
    print("="*100)
    print("\nThis demonstrates simultaneous (k_x, k_y) momentum decomposition")
    print("for 2D lattices with translation symmetry in both X and Y directions.\n")
    
    # 2D Square lattices
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + "2D SQUARE LATTICE SYSTEMS".center(98) + "█")
    print("█" + " "*98 + "█")
    print("█"*100)
    
    demonstrate_2d_momentum_analysis(
        lattice_sizes=[(2, 2), (3, 2), (2, 3), (3, 3)],
        lattice_type='square'
    )
    
    # Honeycomb lattices
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + "HONEYCOMB LATTICE SYSTEMS (2D Translation)".center(98) + "█")
    print("█" + " "*98 + "█")
    print("█"*100)
    print("\nNote: Honeycomb has 2 sites per unit cell.")
    print("Full 2D momentum analysis applies both X and Y translations.\n")
    
    demonstrate_2d_momentum_analysis(
        lattice_sizes=[(2, 2), (3, 2), (2, 3)],
        lattice_type='honeycomb'
    )
    
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + "END OF 2D MOMENTUM DEMONSTRATION".center(98) + "█")
    print("█" + " "*98 + "█")
    print("█"*100 + "\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
