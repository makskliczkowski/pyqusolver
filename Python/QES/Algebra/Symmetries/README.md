Symmetry Operations Module
==========================

This module provides symmetry operators for quantum many-body systems with automatic
compatibility checking and momentum sector analysis.

Module Structure
----------------

Core Components:

- base.py              : Base classes, enumerations, and registry
- translation.py       : Translation symmetry (momentum sectors)
- reflection.py        : Spatial reflection/inversion
- parity.py            : Spin-flip (parity) operators
- momentum_sectors.py  : Momentum sector analysis and construction
- compatibility.py     : Automatic symmetry compatibility checking

Available Symmetries
--------------------

1. TranslationSymmetry
   - Crystal momentum sectors k = 2*pi*n/L
   - 1D, 2D, 3D lattice translations
   - Automatic Bloch state construction

2. ReflectionSymmetry
   - Spatial inversion/mirror symmetries
   - Parity quantum numbers r = +/- 1
   - Compatible with translation at k=0, pi

3. ParitySymmetry
   - Global spin-flip operators (X, Y, Z)
   - Discrete symmetries for spin systems
   - Supports spin-1/2 and spin-1

Quick Start
-----------

Basic usage:

    from QES.Algebra.Symmetries import (
        TranslationSymmetry,
        MomentumSectorAnalyzer,
    )
    from QES.general_python.lattices.square import SquareLattice
    from QES.general_python.lattices.lattice import LatticeDirection
    
    # Create lattice
    lat = SquareLattice(2, 4, 4)  # 2D, 4x4
    
    # Analyze momentum sectors
    analyzer = MomentumSectorAnalyzer(lat)
    sectors = analyzer.analyze_2d_sectors()
    
    # Create translation operator
    tx = TranslationSymmetry(lat, LatticeDirection.X)

HilbertSpace Integration:

    from QES.Algebra.hilbert import HilbertSpace
    
    # Create Hilbert space with lattice
    hs = HilbertSpace(ns=16, lattice=lat)
    
    # Analyze all momentum sectors
    sectors = hs.analyze_momentum_sectors()
    
    # Get representatives for k=(0,0) sector
    k0_reps = hs.get_momentum_representatives({
        LatticeDirection.X: 0,
        LatticeDirection.Y: 0
    })

Momentum Sector Analysis
-------------------------

The MomentumSectorAnalyzer automatically handles:

- 1D systems: Single k quantum number
- 2D systems: (k_x, k_y) quantum numbers
- 3D systems: (k_x, k_y, k_z) quantum numbers
- Multi-site unit cells (honeycomb: 2, kagome: 3, etc.)
- Orbit period calculation with GCD-based momentum selection

Example for honeycomb lattice:

    from QES.general_python.lattices.honeycomb import HoneycombLattice
    
    lat = HoneycombLattice(dim=2, lx=4, ly=2)
    analyzer = MomentumSectorAnalyzer(lat)
    
    # Automatically detects 2 sites per unit cell
    print(f"Sites per cell: {analyzer.sites_per_cell}")
    
    # Returns 2D momentum sectors
    sectors = analyzer.analyze_2d_sectors()

Compatibility Checking
----------------------

Automatic validation of symmetry combinations:

    from QES.Algebra.Symmetries import check_compatibility
    
    # Check if symmetries can be combined
    compatible = check_compatibility(
        symmetry1,
        symmetry2,
        momentum_sector=MomentumSector.ZERO
    )

The module automatically filters incompatible symmetries:

- Reflection only compatible with translation at k=0, pi
- Parity X/Y incompatible with U1 spin conservation
- Automatic detection based on momentum sector

Advanced Features
-----------------

1. Momentum Superposition Construction:
   - Build Bloch states from representatives
   - Proper normalization and phases
   - Handles degeneracies and orbit periods

2. Registry System:
   - Automatic registration of symmetry operators
   - Type-safe symmetry classification
   - Extensible for custom symmetries

3. Local Space Validation:
   - Checks symmetry applicability to local space type
   - Prevents invalid combinations (e.g., parity on fermions)
   - Automatic filtering in HilbertSpace

Testing
-------

Run integration tests:

    python QES/test/test_momentum_integration.py

Test coverage:

- 1D chain analysis
- 2D square lattice
- Honeycomb lattice (multi-site cells)
- HilbertSpace vs standalone consistency

All tests pass with proper 2D momentum analysis and sites-per-cell detection.

API Reference
-------------

See individual module docstrings for detailed API documentation:

- Each symmetry class has comprehensive docstrings
- Physical context and commutation rules documented
- Examples for common use cases

Module Metadata
---------------

Version : 1.0.0
Author  : Maksymilian Kliczkowski
Date    : 2025-10-27
