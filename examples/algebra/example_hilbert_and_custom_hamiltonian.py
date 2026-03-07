import numpy as np

from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice


def main():
    print("--- Hilbert And Custom Hamiltonian ---")

    ns = 6
    lat = SquareLattice(dim=1, lx=ns, bc="pbc")

    hs_full = HilbertSpace(lattice=lat, ns=ns, is_manybody=True)
    hs_sym = HilbertSpace(lattice=lat, ns=ns, is_manybody=True, sym_gen={"translation": 0}, gen_mapping=True)

    print("hilbert full dim:", hs_full.nh, "full basis:", hs_full.nhfull)
    print("hilbert symmetry dim:", hs_sym.nh, "full basis:", hs_sym.nhfull)

    ham = Hamiltonian(hilbert_space=hs_full, dtype=np.complex128, name="custom_tfim")
    ops = ham.operators
    sx = ops.sig_x(ns=ns, type_act="local")
    sz_corr = ops.sig_z(ns=ns, type_act="correlation")

    for i in range(ns):
        j = (i + 1) % ns
        ham.add(sz_corr, sites=[i, j], multiplier=1.0)
        ham.add(sx, sites=[i], multiplier=0.35)

    ham.build()
    ham.diagonalize(k=4)

    vals = np.real(np.array(ham.eigenvalues))
    print("lowest eigenvalues:", np.sort(vals))


if __name__ == "__main__":
    main()
