"""Many-body time-evolution and spectral-statistics workflow."""

import numpy as np

from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice
from QES.general_python.physics.eigenlevels import gap_ratio


def main():
    print("--- Time Evolution And Spectral Statistics ---")

    # Section: Build a small transverse-field Ising chain
    ns  = 6
    lat = SquareLattice(dim=1, lx=ns, bc="pbc")
    hs  = HilbertSpace(lattice=lat, ns=ns, is_manybody=True)

    ham = Hamiltonian(hilbert_space=hs, dtype=np.complex128, name="tfim")
    ops = ham.operators
    sx  = ops.sig_x(ns=ns, type_act="local")
    sz_corr = ops.sig_z(ns=ns, type_act="correlation")

    for i in range(ns):
        j = (i + 1) % ns
        ham.add(sz_corr, sites=[i, j], multiplier=1.0)
        ham.add(sx, sites=[i], multiplier=0.35)

    # Section: Diagonalize and compute spectral statistics
    ham.build()
    ham.diagonalize(k=min(ham.hilbert_space.nh, 16))
    ham.eigenstates = ham.eigenvectors

    vals    = np.real(np.array(ham.eigenvalues))
    stats   = gap_ratio(vals, fraction=0.8, use_mean_lvl_spacing=True)

    psi0    = ham.eigenvectors[:, 0]
    times   = np.linspace(0.0, 1.0, 5)
    psi_t   = ham.time_evo.evolve_batch(psi0, times)
    norms   = [np.linalg.norm(psi_t[:, i]) for i in range(psi_t.shape[1])]

    # Section: Report the small reference workflow
    print("gap ratio mean/std:", float(stats["mean"]), float(stats["std"]))
    print("state norms:", norms)

if __name__ == "__main__":
    main()
