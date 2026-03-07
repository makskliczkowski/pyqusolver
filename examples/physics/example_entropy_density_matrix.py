import numpy as np

from QES.general_python.physics.density_matrix import rho, rho_spectrum
from QES.general_python.physics.entropy import Entanglement, entropy, mutual_information


def main():
    print("--- Density Matrix And Entropy ---")

    psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    psi /= np.linalg.norm(psi)

    rho_a = rho(psi, va=[0], ns=2)
    lam = rho_spectrum(rho_a)

    s_vn = entropy(lam, q=1.0, typek=Entanglement.VN)
    s_r2 = entropy(lam, q=2.0, typek=Entanglement.RENYI)
    mi, _ = mutual_information(psi, 0, 1, 2, q=1.0)

    print("rho_A eigenvalues:", lam)
    print("S_vN:", float(np.real(s_vn)))
    print("S_R2:", float(np.real(s_r2)))
    print("I(0:1):", float(np.real(mi)))


if __name__ == "__main__":
    main()
