import numpy as np

from QES.Algebra.hamil_quadratic import QuadraticHamiltonian


def main():
    print("--- Quadratic Single Particle Hamiltonian ---")

    ns = 12
    qh = QuadraticHamiltonian(ns=ns, particle_conserving=True, particles="fermions", dtype=np.complex128)

    for i in range(ns):
        qh.add_hopping(i, (i + 1) % ns, -1.0)

    qh.build()
    qh.diagonalize()

    evals = np.sort(np.real(np.array(qh.eig_val)))
    print("single-particle spectrum head:", evals[:6])


if __name__ == "__main__":
    main()
