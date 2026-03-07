import numpy as np

from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice


def main():
    print("--- Lattice-Driven Hamiltonian Build ---")

    lat = SquareLattice(dim=1, lx=6, bc="pbc")
    hs = HilbertSpace(lattice=lat, ns=lat.ns, is_manybody=True)

    ham = Hamiltonian(hilbert_space=hs, dtype=np.complex128, name="lattice_tfim")
    ops = ham.operators
    sz_corr = ops.sig_z(ns=lat.ns, type_act="correlation")
    sx = ops.sig_x(ns=lat.ns, type_act="local")

    for i in range(lat.ns):
        j = int(lat.get_nei(i, direction=0))
        if i < j:
            ham.add(sz_corr, sites=[i, j], multiplier=1.0)
        ham.add(sx, sites=[i], multiplier=0.3)

    ham.build()
    ham.diagonalize(k=6)

    vals = np.sort(np.real(np.array(ham.eigenvalues)))
    print("first eigenvalues:", vals)


if __name__ == "__main__":
    main()
