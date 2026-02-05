"""
Exact Diagonalization Example
=============================

This example demonstrates how to:
1. Define a Hilbert space on a lattice.
2. Construct a Heisenberg-Kitaev Hamiltonian.
3. Compute the exact spectrum (eigenvalues and eigenvectors).

To run:
    python examples/01_exact_diagonalization.py
"""

import numpy as np
import QES
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
from QES.general_python.lattices.honeycomb import HoneycombLattice

def main():
    # 1. Setup the Lattice (2x2 Honeycomb, 8 sites)
    # Using 'pbc' (Periodic Boundary Conditions)
    lattice = HoneycombLattice(dim=2, lx=2, ly=2, bc="pbc")
    print(f"Lattice created: {lattice}")
    print(f"Number of sites: {lattice.ns}")

    # 2. Setup Hilbert Space
    # Spin-1/2 system (default)
    # Use complex128 to ensure compatibility with complex Hamiltonians
    hilbert = HilbertSpace(
        lattice=lattice,
        is_manybody=True,
        dtype=np.complex128
    )
    print(f"Hilbert Space dimension: {hilbert.nh}")

    # 3. Instantiate Hamiltonian
    # Heisenberg-Kitaev model:
    # H = \sum_{<ij>_\gamma} K S_i^\gamma S_j^\gamma + J \vec{S}_i \cdot \vec{S}_j
    # Here we set J=0 (Pure Kitaev) and K=1.0
    # Explicitly set dtype to complex128 to avoid Numba type mismatches
    hamiltonian = HeisenbergKitaev(
        lattice=lattice,
        hilbert_space=hilbert,
        J=0.0,
        K=1.0,
        hz=0.1,  # Small magnetic field to break degeneracy
        dtype=np.complex128
    )
    print(f"Hamiltonian initialized: {hamiltonian.name}")

    # 4. Diagonalize
    print("\nDiagonalizing Hamiltonian...")
    # Calculate lowest k=5 eigenvalues
    # method='exact' uses full dense matrix diagonalization (for small systems)
    # method='lanczos' uses sparse iterative solver (for larger systems)
    hamiltonian.diagonalize(k=5, method='exact')

    print("\nLowest 5 Eigenvalues:")
    for i, E in enumerate(hamiltonian.eigenvalues):
        print(f"  E[{i}] = {E:.6f}")

    # Ground state vector
    psi_gs = hamiltonian.eig_vec[:, 0]
    norm = np.linalg.norm(psi_gs)
    print(f"\nGround state norm: {norm:.6f}")

if __name__ == "__main__":
    main()
