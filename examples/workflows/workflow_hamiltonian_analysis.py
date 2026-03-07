"""
Workflow: Hamiltonian Analysis (Spin & Quadratic)
=================================================

This example demonstrates how to construct and analyze different types of Hamiltonians:
1. Spin Hamiltonians (e.g., Transverse Field Ising Model).
2. Quadratic Hamiltonians (e.g., Free Fermions).

It covers:
- Operator definition.
- Matrix construction (Sparse/Dense).
- Spectrum analysis (Ground state energy).
- Hermiticity checks.
"""
import sys
import os
import numpy as np

# Add Python paths
current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))
    sys.path.append(os.path.join(current_dir, "Python", "QES"))

import QES
from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice

def analyze_tfim(N=8, h=1.0):
    """Constructs and analyzes the Transverse Field Ising Model."""
    print(f"\n[1] Analysing TFIM (N={N}, h={h})...")

    # 1. Define Hilbert Space
    lattice = SquareLattice(dim=1, lx=N, bc="pbc")
    hilbert = HilbertSpace(lattice=lattice, ns=N, is_manybody=True)

    # 2. Construct Hamiltonian: H = - sum Z_i Z_{i+1} - h sum X_i
    ham = Hamiltonian(hilbert_space=hilbert, name="TFIM", dtype=np.complex128)
    ops = ham.operators

    # Interaction term (ZZ)
    sz_corr = ops.sig_z(ns=N, type_act='correlation')
    for i in range(N):
        j = (i + 1) % N
        ham.add(sz_corr, sites=[i, j], multiplier=-1.0)

    # Transverse field (X)
    sx = ops.sig_x(ns=N, type_act='local')
    for i in range(N):
        ham.add(sx, sites=[i], multiplier=-h)

    # 3. Build Matrix
    print("    Building sparse matrix...")
    ham.build()

    # 4. Check Hermiticity
    print("    Checking Hermiticity...")
    H_mat = ham.hamil
    if H_mat is not None:
        if hasattr(H_mat, "toarray"):
            H_dense = H_mat.toarray()
        else:
            H_dense = H_mat

        diff = np.linalg.norm(H_dense - H_dense.conj().T)
        is_hermitian = diff < 1e-10
        print(f"    Hermitian: {is_hermitian} (diff={diff:.2e})")
        if not is_hermitian:
            print("    WARNING: Hamiltonian is not Hermitian!")
    else:
        print("    Matrix not available.")

    # 5. Diagonalize (Low-lying spectrum)
    print("    Calculating ground state...")
    ham.diagonalize(k=2)
    E0 = ham.eigenvalues[0].real
    E1 = ham.eigenvalues[1].real
    gap = E1 - E0
    print(f"    E0: {E0:.6f}")
    print(f"    Gap: {gap:.6f}")
    return E0

def analyze_free_fermions(N=8, t=1.0):
    """Constructs and analyzes a Free Fermion hopping model."""
    print(f"\n[2] Analysing Free Fermions (N={N}, t={t})...")

    # 1. Define Hilbert Space (Spinless Fermions)
    # Note: QuadraticHamiltonian handles particle number conservation implicitly via BdG
    lattice = SquareLattice(dim=1, lx=N, bc="pbc")
    # Hilbert space is technicaly defined, but QuadraticHamiltonian works on single-particle sector usually

    # 2. Construct via BdG Matrices
    # H = -t sum (c^dag_i c_{i+1} + h.c.)
    # In BdG form, we define the hopping matrix A where H = c^dag A c

    # Matrix A_{ij} corresponds to hopping from j to i
    A = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        j = (i + 1) % N
        A[i, j] = -t
        A[j, i] = -t # Hermitian conjugate part

    # Antisymmetric part B is zero for standard hopping
    B = np.zeros((N, N), dtype=np.complex128)

    print("    Constructing QuadraticHamiltonian from matrices...")
    ham_q = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=A,
        antisymmetric_part=B,
        constant=0.0,
        dtype=np.complex128
    )

    # 3. Diagonalize (Single particle spectrum)
    # QuadraticHamiltonian automatically diagonalizes the single-particle matrix
    print("    Single-particle eigenvalues (orbital energies):")
    energies = ham_q.eig_val.copy()
    # Sort them
    energies.sort()
    print(f"    Min E: {energies[0]:.6f}, Max E: {energies[-1]:.6f}")

    # 4. Many-body Ground State Energy (Half-filling approx sum of negative energies)
    E_mb = np.sum(energies[energies < 0])
    print(f"    Sum of negative energies (Many-body E0 estimate): {E_mb:.6f}")

def main():
    QES.qes_reseed(42)
    analyze_tfim(N=8, h=1.0)
    analyze_free_fermions(N=10, t=1.0)
    print("\nWorkflow completed successfully.")

if __name__ == "__main__":
    main()
