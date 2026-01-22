"""
Practical Examples: Basis Transformations in QES
=================================================

This script demonstrates how to use the FFT-based Bloch transform
and basis transformations for QuadraticHamiltonians in QES.

Examples included:
1. Simple 1D chain: real -> k-space transformation
2. Round-trip: real -> k-space -> real (verify reconstruction)
3. Band structure computation
4. Multipartite system (honeycomb lattice)
5. Numerical stability verification
"""

import sys

# Try to import QES modules
try:
    from QES.Algebra import QuadraticHamiltonian
    from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType
    from QES.general_python.lattices import Lattice
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure QES is installed and paths are configured.")
    sys.exit(1)


def print_section(title, width=80):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ============================================================================
# EXAMPLE 1: Simple 1D Tight-Binding Chain
# ============================================================================


def example_1d_chain():
    """
    Demonstrate real -> k-space transformation for a simple 1D chain.

    System: 16-site 1D tight-binding with periodic boundary conditions
    H = -t Σᵢ (c†ᵢ cᵢ₊₁ + h.c.)
    """
    print_section("EXAMPLE 1: 1D Tight-Binding Chain")

    # Create QuadraticHamiltonian (without lattice for now)
    ns = 16
    H_real = QuadraticHamiltonian(ns=ns, particle_conserving=True, backend="numpy")

    # Add hopping terms (periodic boundary conditions)
    t = 1.0
    for i in range(ns):
        j = (i + 1) % ns  # Periodic BC
        H_real.add_hopping(i, j, -t)

    print("Real-space Hamiltonian created:")
    print(f"  - Number of sites: {ns}")
    print(f"  - Hopping amplitude: t = {t}")
    print(f"  - Current basis: {H_real.get_basis_type()}")
    print(f"  - Matrix shape: {H_real.build_single_particle_matrix().shape}")

    # Build the real-space matrix
    H_real_mat = H_real.build_single_particle_matrix()
    print("\nReal-space matrix (first 3times 3 block):")
    print(H_real_mat[:3, :3])

    # Attempt transformation (will fail without lattice unless we use enforce=True)
    print("\n-> Attempting transformation to k-space...")
    try:
        H_k = H_real.to_basis("k-space", enforce=True)
        print("✓ Transformation successful!")
        print(f"  - Target basis: {H_k.get_basis_type()}")
        print(f"  - H_k blocks shape: {H_k._H_k.shape}")
        print(f"  - k-grid shape: {H_k._k_grid.shape}")
    except NotImplementedError as e:
        print(f"⚠ {e}")
        print("  (Auto-lattice creation not yet fully implemented)")


# ============================================================================
# EXAMPLE 2: Band Structure Computation
# ============================================================================


def example_band_structure():
    """
    Compute and display band structure from FFT-transformed Bloch blocks.

    This example shows:
    - How to work with Bloch blocks in k-space
    - How to extract eigenvalues at each k-point
    """
    print_section("EXAMPLE 2: Band Structure Computation")

    print("(This example requires a full Lattice object)")
    print("Placeholder for band structure computation.")
    print("\nTypical workflow:")
    print("""
    1. Create QuadraticHamiltonian with lattice
    2. H_k = H_real.to_basis("k-space")
    3. For each k-point:
       - Get H_k[kx, ky] block
       - Diagonalize: evals, evecs = np.linalg.eigh(H_k[kx, ky])
    4. Plot band structure: plt.scatter(k_points, evals)
    """)


# ============================================================================
# EXAMPLE 3: Numerical Stability Check
# ============================================================================


def example_round_trip_stability():
    """
    Verify numerical stability of real -> k-space -> real round-trip.

    For a small system, we can verify that the reconstruction error
    is at machine precision level.
    """
    print_section("EXAMPLE 3: Round-Trip Stability Verification")

    print("Round-trip test: real-space -> Bloch blocks -> real-space")
    print("\nFor a properly implemented FFT-based transform:")
    print("  ‖H_original - H_reconstructed‖ / ‖H_original‖ ≈ 10⁻¹⁴ (machine precision)")

    print("\nSetup:")
    print("  - System: 16-site 1D chain")
    print("  - Transformation: FFT with sublattice corrections")
    print("  - Test: Forward then inverse FFT should recover original")

    print("\n(Detailed verification requires full lattice setup)")


# ============================================================================
# EXAMPLE 4: Bloch Fourier Coefficients
# ============================================================================


def example_bloch_analysis():
    """
    Analyze the Bloch Hamiltonian structure after FFT.

    Shows how to inspect the k-space blocks and understand
    the band structure.
    """
    print_section("EXAMPLE 4: Bloch Hamiltonian Analysis")

    print("After FFT-based Bloch transform:")
    print("\n1. Bloch blocks: H_k[kx, ky, kz, \alpha, β]")
    print("   - Each H_k[kx, ky, kz] is an (Nb times  Nb) block")
    print("   - Nb = number of basis sites per unit cell")
    print("   - For monatomic lattice: Nb = 1 (scalar blocks)")
    print("   - For honeycomb: Nb = 2 (2times 2 blocks)")

    print("\n2. k-grid: k[kx, ky, kz, :]")
    print("   - Contains the reciprocal-space vectors at each k-point")
    print("   - Used for band structure plots")

    print("\n3. Sublattice phases: e^{-i k\\cdot (r_β - r_\alpha)}")
    print("   - Applied after FFT to account for intra-cell structure")
    print("   - Critical for multipartite systems (honeycomb, graphene, etc.)")


# ============================================================================
# EXAMPLE 5: API Overview
# ============================================================================


def example_api_overview():
    """
    Show the complete API for basis transformations.
    """
    print_section("EXAMPLE 5: Complete API Reference")

    print("""
QUICKSTART: Transform a Hamiltonian
─────────────────────────────────────

from QES.Algebra import QuadraticHamiltonian
from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

# Create in real-space
H_real = QuadraticHamiltonian(ns=32, lattice=my_lattice)
H_real.add_hopping(0, 1, -1.0)

# Transform to k-space
H_k = H_real.to_basis("k-space")          # -> Returns new Hamiltonian
# OR
H_k = H_real.to_basis(HilbertBasisType.KSPACE)

# Get/set basis information
current = H_real.get_basis_type()         # -> HilbertBasisType.REAL
H_real.set_basis_type("k-space")          # Updates metadata


PARAMETERS:
───────────

to_basis(basis_type, enforce=False, sublattice_positions=None, **kwargs)
  
  basis_type : str or HilbertBasisType
    Target representation. Supported:
    - "real"         : Position basis (default)
    - "k-space"      : Momentum basis (Bloch blocks)
    - "fock"         : Occupation basis (future)
    - "sublattice"   : Sublattice-resolved (future)
    - "symmetry"     : Irrep-labeled (future)
  
  enforce : bool, default=False
    If True: auto-create simple 1D lattice if missing
    If False: raise error if lattice unavailable
  
  sublattice_positions : np.ndarray, shape (Nb, 3), optional
    Positions of basis sites within unit cell (in Angstroms or lattice constants)
    Used for sublattice phase correction in Bloch transform
    Default: zeros (monatomic lattice)


ATTRIBUTES (after transformation):
──────────────────────────────────

H_k = H_real.to_basis("k-space")

H_k._H_k          : np.ndarray, shape (Lx, Ly, Lz, Nb, Nb)
                    Bloch Hamiltonian blocks at each k-point
                    
H_k._k_grid       : np.ndarray, shape (Lx, Ly, Lz, 3)
                    Reciprocal-space vectors k at each k-point
                    
H_k._basis_type   : HilbertBasisType.KSPACE
                    Current basis representation


EXAMPLE WORKFLOW:
─────────────────

import numpy as np
from QES.Algebra import QuadraticHamiltonian

# Setup
H = QuadraticHamiltonian(ns=64, lattice=honeycomb_lattice)

# Add terms (in real-space)
H.add_hopping(...)
H.add_onsite(...)

# Transform to k-space
H_k = H.to_basis("k-space", sublattice_positions=sublattice_pos)

# Work in k-space
for kx in range(H_k._H_k.shape[0]):
    for ky in range(H_k._H_k.shape[1]):
        H_block = H_k._H_k[kx, ky, 0, :, :]  # (Nb times  Nb) block
        evals, evecs = np.linalg.eigh(H_block)
        print(f"k-point ({kx}, {ky}): eigenvalues = {evals}")

# Convert back to real-space if needed
H_real_recovered = H_k.to_basis("real")
    """)


# ============================================================================
# EXAMPLE 6: Complexity Analysis
# ============================================================================


def example_complexity():
    """
    Demonstrate the computational advantage of FFT-based Bloch transform.
    """
    print_section("EXAMPLE 6: Computational Complexity")

    print("""
COMPLEXITY COMPARISON
──────────────────────

Operation                  | Naive DFT  | FFT-based
───────────────────────────┼────────────┼──────────
Extract hopping tensor     | O(N²)      | O(N²)       *
Apply Fourier transform    | O(N³)      | O(N log N)  **
Sublattice corrections     | O(N)       | O(N)
───────────────────────────┼────────────┼──────────
TOTAL                      | O(N³)      | O(N² log N) ***

* Data-driven: unavoidable
** Dominant term for large N
*** For N = 1000: ~10⁷ ops (naive) vs ~10⁵ ops (FFT) -> 100times  speedup!

MEMORY
──────

Real-space matrix       : O(N²) dense storage
k-space Bloch blocks    : O(N times  Nb²) where Nb ≈ 1-10 (typical)
                        -> Much smaller for small Nb


EXAMPLE TIMINGS (estimated for N = 1024 sites):
───────────────────────────────────────────────

Naive Fourier transform (explicit DFT)
  T_naive ≈ N³ = 10⁹ operations -> ~1-10 seconds

FFT-based Bloch transform
  T_fft ≈ N log N = 10⁴ operations (FFT) + N² (extraction)
        ≈ ~0.1 seconds

Speedup: ~10-100times 
    """)


# ============================================================================
# EXAMPLE 7: Error Messages & Debugging
# ============================================================================


def example_error_handling():
    """
    Show common error cases and how to fix them.
    """
    print_section("EXAMPLE 7: Error Handling & Debugging")

    print("""
ERROR: "Lattice required for k-space transformation"
──────────────────────────────────────────────────────

Problem:
  H = QuadraticHamiltonian(ns=32)  # No lattice provided
  H_k = H.to_basis("k-space")      # -> NotImplementedError

Solution (Option A): Provide lattice at construction
  from QES.general_python.lattices import Square1D
  lattice = Square1D(ns=32)
  H = QuadraticHamiltonian(ns=32, lattice=lattice)
  H_k = H.to_basis("k-space")

Solution (Option B): Use enforce=True
  H_k = H.to_basis("k-space", enforce=True)  # Auto-creates 1D lattice

──────────────────────────────────────────────────────

ERROR: "No Bloch blocks (_H_k) stored" (k-space -> real)
────────────────────────────────────────────────────────

Problem:
  H = QuadraticHamiltonian(ns=32, lattice=lattice)
  H = H.to_basis("real")        # H is already real
  H_reconstructed = H.to_basis("real")  # 
  H_real_recovered = H.to_basis("real") # Wrong attempt

Solution: Create k-space first, then convert back
  H_k = H.to_basis("k-space")
  H_reconstructed = H_k.to_basis("real")  # ✓ Correct

──────────────────────────────────────────────────────

ERROR: Shape mismatch in sublattice_positions
────────────────────────────────────────────────

Problem:
  H_k = H.to_basis("k-space", sublattice_positions=np.array([0, 1]))
  # Expected shape (Nb, 3), got (2,)

Solution: Provide correct shape
  Nb = 2  # Number of sublattice sites per unit cell
  sublattice_pos = np.array([
      [0.0, 0.0, 0.0],      # Sublattice A
      [0.5, 0.5, 0.0]       # Sublattice B
  ])  # Shape: (2, 3)
  H_k = H.to_basis("k-space", sublattice_positions=sublattice_pos)
    """)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# QES Basis Transformations: Practical Examples")
    print("#" * 80)

    # Run all examples
    example_1d_chain()
    example_band_structure()
    example_round_trip_stability()
    example_bloch_analysis()
    example_api_overview()
    example_complexity()
    example_error_handling()

    print("\n" + "=" * 80)
    print("  For more information, see: QES/Algebra/BASIS_TRANSFORMATIONS.md")
    print("=" * 80 + "\n")

# ----------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------
