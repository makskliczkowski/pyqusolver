"""
Example: Exact Diagonalization of a Custom Hamiltonian
======================================================

This example demonstrates how to:
1. Define a Lattice and Hilbert Space.
2. Construct a custom Hamiltonian (Heisenberg chain) using local operators.
3. Build the Hamiltonian matrix.
4. Diagonalize it to find eigenvalues and eigenvectors.
"""
import sys
import os
import numpy as np

# Ensure QES is in path if running from root
current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import QES
from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice

def main():
    QES.qes_reseed(42)
    print("="*60)
    print("Exact Diagonalization: Heisenberg Chain (N=4)")
    print("="*60)

    # 1. Setup Lattice and Hilbert Space
    ns = 4
    # 1D lattice with periodic boundary conditions
    lattice = SquareLattice(dim=1, lx=ns, bc="pbc")
    hilbert = HilbertSpace(lattice=lattice, ns=ns, is_manybody=True)

    print(f"Hilbert space dimension: {hilbert.nh}")

    # 2. Create Hamiltonian
    # Must use complex128 because Pauli Y involves imaginary numbers
    ham = Hamiltonian(hilbert_space=hilbert, name="Heisenberg", dtype=np.complex128)

    # Define operators
    # Note: Operators factory creates operators compatible with the Hilbert space
    ops = ham.operators

    # We need correlation operators (acting on pairs of sites)
    sx_corr = ops.sig_x(ns=ns, type_act='correlation')
    sy_corr = ops.sig_y(ns=ns, type_act='correlation')
    sz_corr = ops.sig_z(ns=ns, type_act='correlation')

    # Add Heisenberg interaction: H = J sum_{i} S_i . S_{i+1}
    # where S = (1/2) * sigma
    # So S_i . S_j = (1/4) * (sigma_x_i sigma_x_j + sigma_y_i sigma_y_j + sigma_z_i sigma_z_j)

    # We set J=4.0 so that the effective coupling for Pauli matrices is 1.0
    # This aligns the ground state energy with standard literature (E0 = -2.0 for N=4, J=1)
    J_phys = 1.0
    J_eff  = 4.0 * J_phys

    print(f"Adding terms with physical J={J_phys} (effective Pauli J={J_eff})...")

    for i in range(ns):
        j = (i + 1) % ns  # Periodic neighbor

        # Add terms (0.25 * J_eff for each Pauli product => 1.0 * J_phys)
        coeff = 0.25 * J_eff
        ham.add(sx_corr, sites=[i, j], multiplier=coeff)
        ham.add(sy_corr, sites=[i, j], multiplier=coeff)
        ham.add(sz_corr, sites=[i, j], multiplier=coeff)

    # 3. Build Matrix
    print("Building Hamiltonian matrix...")
    ham.build()

    # 4. Diagonalize
    print("Diagonalizing...")
    # Calculate lowest k eigenvalues
    ham.diagonalize(k=min(hilbert.nh, 10))

    print("\nEigenvalues:")
    for i, e in enumerate(ham.eigenvalues):
        print(f"  E[{i}] = {e.real:.6f}")

    print("\nGround state energy:", ham.eigenvalues[0].real)

    # Theoretical ground state energy for N=4 Heisenberg chain (PBC): E0 = -2.0 * J
    # Note:
    # For N=2, E0 = -0.75
    # For N=4, Bethe ansatz gives E = -2.0 (singlet ground state) with H = sum S_i.S_{i+1}
    # Our H uses J=1.0.

    expected_E0 = -2.0 if ns == 4 else (-0.75 if ns == 2 else None)

    if expected_E0 is not None:
        if np.isclose(ham.eigenvalues[0].real, expected_E0, atol=1e-5):
            print(f"✓ Matches expected ground state energy ({expected_E0})")
        else:
            print(f"⚠ Difference from expected ({expected_E0}): {abs(ham.eigenvalues[0].real - expected_E0)}")
            print("Note: If using Pauli matrices directly, H = 1/4 * sum sigma_i sigma_j.")
            print("If you expected -8.0 or -0.5, check spin magnitude factors.")

if __name__ == "__main__":
    main()
