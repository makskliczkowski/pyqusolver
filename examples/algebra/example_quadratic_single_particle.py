import numpy as np

from QES.Algebra.hamil_quadratic import QuadraticHamiltonian


def main():
    print("--- Quadratic Single Particle Hamiltonian ---")

    ns = 12
    qh = QuadraticHamiltonian(ns=ns, particle_conserving=True, particles="fermions", dtype=np.complex128)

    for i in range(ns):
        qh.add_hopping(i, (i + 1) % ns, -1.0)
    qh.add_onsite(0, 0.25)

    h_single = qh.build_single_particle_matrix()
    W_pc, eps_pc, const_pc = qh.diagonalizing_bogoliubov_transform()
    print("single-particle matrix shape:", h_single.shape)
    print("particle-conserving transform shape:", W_pc.shape)
    print("single-particle spectrum head:", np.sort(np.real(np.array(eps_pc)))[:6])
    print("stored constant:", const_pc)

    K = np.zeros((4, 4), dtype=np.complex128)
    Delta = np.zeros((4, 4), dtype=np.complex128)
    for i in range(3):
        K[i, i + 1] = -1.0
        K[i + 1, i] = -1.0
        Delta[i, i + 1] = 0.2
        Delta[i + 1, i] = -0.2
    K[1, 1] = 0.4

    qh_bdg = QuadraticHamiltonian.from_bdg_matrices(
        hermitian_part=K,
        antisymmetric_part=Delta,
        constant=0.1,
        dtype=np.complex128,
    )
    h_bdg = qh_bdg.build_bdg_matrix()
    W_bdg, eps_bdg, const_bdg = qh_bdg.diagonalizing_bogoliubov_transform()
    print("BdG matrix shape:", h_bdg.shape)
    print("BdG transform shape:", W_bdg.shape)
    print("BdG orbital energies:", np.array(eps_bdg))
    print("BdG transformed constant:", const_bdg)


if __name__ == "__main__":
    main()
