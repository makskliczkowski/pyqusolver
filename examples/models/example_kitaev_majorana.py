"""
Clean example of using KitaevGammaMajorana model with explicit basis control.

This demonstrates:
1. Creating a lattice
2. Optionally specifying the basis for HilbertSpace
3. Building and diagonalizing the model
4. Optionally transforming to different bases

NOTES:
- Basis system now supports inheritance: subclasses can use parent handlers
- Automatic basis inference: quadratic + lattice -> REAL basis
- Handler registry: extensible pattern for new transformations

Author: Maksymilian Kliczkowski
Date: 2025-11-06
"""
import sys
import os

# Ensure QES is in path if running from root
current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import numpy as np
import QES

from QES.Algebra.hilbert import HilbertSpace
# Fixed import path (lowercase conserving)
from QES.Algebra.Model.Noninteracting.conserving.Majorana.kitaev_gamma_majorana import (
    KitaevGammaMajorana,
)
from QES.general_python.common.flog import get_global_logger
from QES.general_python.lattices import HoneycombLattice

def main():
    QES.qes_reseed(42)
    # ============================================================================
    # EXAMPLE 1: Basic usage (automatic basis inference)
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic usage with automatic basis inference")
    print("=" * 70)

    logger = get_global_logger()

    # Create lattice
    lat = HoneycombLattice(dim=2, lx=2, ly=2, bc="pbc")

    # Model parameters
    Kx, Ky, Kz = 1.0, 1.0, 1.0
    Gx, Gy, Gz = None, None, None  # Use defaults (None = inferred from Kz)
    hx, hy, hz = None, None, None  # No magnetic field

    # Create model with automatic HilbertSpace and basis inference
    # For quadratic systems with lattice: basis will be REAL (position space)
    model = KitaevGammaMajorana(
        lat,
        k_x=Kx,
        k_y=Ky,
        k_z=Kz,
        gamma_x=Gx,
        gamma_y=Gy,
        gamma_z=Gz,
        h_x=hx,
        h_y=hy,
        h_z=hz,
        p_flip=0.0,
        p_zero=None,
        p_plus=None,
        logger=logger,
    )

    print(f"Model created: {model}")
    # Hilbert space might be created lazily or accessed via private attr if needed
    if model._hilbert_space:
        print(f"Hilbert space basis: {model._hilbert_space.get_basis()}")
    print(f"Current Hamiltonian basis: {model._current_basis}")

    # Build and diagonalize
    model.build()
    model.diagonalize()
    print(f"Diagonalization complete. Ground state energy: {model.eig_val[0]:.6f}")

    # ============================================================================
    # EXAMPLE 2: Explicit basis specification via HilbertSpace
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Explicit basis specification via HilbertSpace")
    print("=" * 70)

    # Create HilbertSpace with explicit basis (even though it will be overridden
    # for quadratic + lattice systems, you can do this)
    hilbert_space = HilbertSpace(
        ns=lat.ns,
        lattice=lat,
        is_manybody=False,  # Quadratic system
        basis="real",  # Explicit basis specification
    )

    print(f"HilbertSpace created with basis: {hilbert_space.get_basis()}")

    # Create model with explicit HilbertSpace
    model2 = KitaevGammaMajorana(
        lat,
        hilbert_space=hilbert_space,  # Pass explicit HilbertSpace
        k_x=Kx,
        k_y=Ky,
        k_z=Kz,
        gamma_x=Gx,
        gamma_y=Gy,
        gamma_z=Gz,
        h_x=hx,
        h_y=hy,
        h_z=hz,
        p_flip=0.0,
        logger=logger,
    )

    print("Model created with explicit HilbertSpace")
    print(f"Hamiltonian inherited basis: {model2._current_basis}")

    # Build and diagonalize
    model2.build()
    model2.diagonalize()
    print(f"Diagonalization complete. Ground state energy: {model2.eig_val[0]:.6f}")

    # ============================================================================
    # EXAMPLE 3: Real-space to k-space transformation
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Transform from real-space to k-space")
    print("=" * 70)

    model3 = KitaevGammaMajorana(
        lat,
        k_x=Kx,
        k_y=Ky,
        k_z=Kz,
        gamma_x=Gx,
        gamma_y=Gy,
        gamma_z=Gz,
        logger=logger,
    )

    # Build in real space
    model3.build()
    print(f"Built Hamiltonian in basis: {model3._current_basis}")
    print(f"Real-space Hamiltonian shape: {model3._hamil_sp.shape}")

    # Transform to k-space
    print("\nTransforming to k-space...")
    try:
        model3.to_basis("k-space", enforce=True)
        print(f"Transformed to basis: {model3._current_basis}")
        if model3._hamil_transformed is not None:
            print(f"Transformed Hamiltonian stored, shape: {model3._hamil_transformed.shape}")
        if model3._transformed_grid is not None:
            print(f"Transformed grid (k-space points) shape: {model3._transformed_grid.shape}")
    except Exception as e:
        print(f"Transformation failed: {e}")

    # Diagonalize in k-space representation
    print("\nDiagonalizing in k-space representation...")
    model3.diagonalize()
    print("Diagonalization complete.")

    # ============================================================================
    # EXAMPLE 3B: Understanding k-space points
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3B: Understanding k-space grid points")
    print("=" * 70)

    if model3._transformed_grid is not None:
        k_grid = model3._transformed_grid
        print(f"K-space grid shape: {k_grid.shape}")
        print(
            f"  Interpretation: ({k_grid.shape[0]}, {k_grid.shape[1]}, {k_grid.shape[2]}) lattice momentum points"
        )
        print(f"  Each point has 3D coordinates: {k_grid.shape[3]}")

        print("\nFirst few k-points:")
        for i in range(min(3, k_grid.shape[0])):
            for j in range(min(3, k_grid.shape[1])):
                k_vec = k_grid[i, j, 0]
                k_mag = np.linalg.norm(k_vec)
                print(f"  k[{i},{j}] = {k_vec} (|k| = {k_mag:.4f})")

        print("\nUnderstanding k-space points:")
        print("  - Each point k represents a Bloch state in momentum space")
        print("  - Computed from reciprocal lattice vectors (b1, b2, b3)")
        print("  - For Lxtimes Lytimes Lz real-space cells: Lxtimes Lytimes Lz k-points in BZ")
        print("  - H(k) is Nbtimes Nb matrix at each k-point (Nb = sublattices)")
        print("  - Diagonalizing H(k) gives single-particle band structure")

    # ============================================================================
    # EXAMPLE 4: Custom configuration (no random field)
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom configuration without random disorder")
    print("=" * 70)

    model4 = KitaevGammaMajorana(
        lat,
        k_x=Kx,
        k_y=Ky,
        k_z=Kz,
        gamma_x=Gx,
        gamma_y=Gy,
        gamma_z=Gz,
        h_x=None,  # No magnetic field
        h_y=None,
        h_z=None,
        p_flip=0.0,  # No disorder in u field
        p_zero=None,  # No random g field
        p_plus=None,
        logger=logger,
    )

    model4.build()
    model4.diagonalize()
    print("Model built and diagonalized")
    print(f"Ground state energy: {model4.eig_val[0]:.6f}")
    print("First 5 energy levels:")
    for i in range(min(5, len(model4.eig_val))):
        print(f"  E[{i}] = {model4.eig_val[i]:.6f}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Key points about basis handling:

    1. AUTOMATIC INFERENCE (Most Common):
    - For quadratic systems with lattice: automatically uses REAL basis
    - HilbertSpace infers basis from system type
    - Hamiltonian inherits basis from HilbertSpace

    2. EXPLICIT BASIS:
    - Pass `basis="real"` to HilbertSpace constructor
    - Pass HilbertSpace to model constructor
    - Hamiltonian inherits the explicit basis

    3. BASIS TRANSFORMATIONS:
    - Call `model.to_basis("k-space")` to transform
    - Transformed representation stored in `_hamil_transformed`
    - Associated grid stored in `_transformed_grid`
    - Current basis tracked in `_current_basis`

    4. ARCHITECTURE:
    - Parent Hamiltonian has handler registry system
    - QuadraticHamiltonian registers REAL <-> KSPACE transformations
    - New basis types can be added by registering handlers
    - No need to modify core `to_basis()` method

    5. CLEANUP:
    - Old attributes (_H_k, _k_grid) removed
    - General storage (_hamil_transformed, _transformed_grid) in parent
    - Handler registration at module end (clean separation)
    - Symmetry information tracked separately
    """)

    print("✅ All examples completed successfully!")

if __name__ == "__main__":
    main()
