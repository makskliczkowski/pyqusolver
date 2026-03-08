'''Quadratic interop workflow for Qiskit-style and OpenFermion-style usage.

This example demonstrates:
- construction of a BdG quadratic Hamiltonian,
- inspection of the single-particle and BdG matrix shapes,
- round-trip export/import through optional Qiskit and OpenFermion bridges.
'''

import numpy as np
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian

def main():
    # Section: Define a small BdG problem
    K = np.array(
        [
            [0.2, -1.0],
            [-1.0, -0.1],
        ],
        dtype=np.complex128,
    )
    Delta = np.array(
        [
            [0.0, 0.25],
            [-0.25, 0.0],
        ],
        dtype=np.complex128,
    )

    qh = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        constant=0.3,
        dtype=np.complex128,
    )

    print("--- Quadratic Interop Example ---")
    print("particle conserving:", qh.particle_conserving)
    print("single-particle block shape:", qh.build_single_particle_matrix().shape)
    print("bdg matrix shape:", qh.build_bdg_matrix().shape)

    # Section: Optional Qiskit-style round trip
    #
    # This path is skipped cleanly when Qiskit Nature is not installed.
    try:
        qiskit_ham  = qh.to_qiskit_hamiltonian()
        qh_qiskit   = QuadraticHamiltonian.from_qiskit_hamiltonian(qiskit_ham, dtype=np.complex128)
        print("qiskit round trip shape:", qh_qiskit.build_bdg_matrix().shape)
    except ImportError:
        print("qiskit nature not installed, skipping qiskit round trip")

    # Section: Optional OpenFermion-style round trip
    #
    # This path is skipped cleanly when OpenFermion is not installed.
    try:
        of_ham      = qh.to_openfermion_hamiltonian()
        qh_of       = QuadraticHamiltonian.from_openfermion_hamiltonian(of_ham, num_orbitals=2, dtype=np.complex128)
        print("openfermion round trip shape:", qh_of.build_bdg_matrix().shape)
    except ImportError:
        print("openfermion not installed, skipping openfermion round trip")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
#! EOF
# ---------------------------------------------------------------------------