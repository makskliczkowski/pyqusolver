'''Example of using spectral and statistical tools.'''

import numpy as np

from QES.Algebra.Properties.statistical import dos, ldos
from QES.general_python.common.plotters.spectral_utils import compute_spectral_broadening


def main():
    print("--- Spectral And Statistical Tools ---")

    energies = np.array([-2.0, -1.0, -0.2, 0.1, 0.9, 1.8], dtype=float)
    overlaps = np.array([0.2, 0.4, 0.3, 0.6, 0.5, 0.2], dtype=np.complex128)

    l = ldos(energies, overlaps, degenerate=False)
    d = dos(energies, nbins=6)
    print("ldos:", l)
    print("dos bins:", d)

    omega   = np.linspace(-3.0, 3.0, 121)
    weights = np.abs(overlaps) ** 2
    aw      = compute_spectral_broadening(energies, weights, omega, eta=0.08, kind="lorentzian")
    print("A(omega=0):", float(aw[len(omega) // 2]))


if __name__ == "__main__":
    main()
