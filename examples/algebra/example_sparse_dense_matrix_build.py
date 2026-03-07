'''Example of building both sparse and dense Hamiltonian matrices using the same code.'''

import numpy as np

from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice

def _build_example(is_sparse: bool):
    ns  = 4
    lat = SquareLattice(dim=1, lx=ns, bc="pbc")
    hs  = HilbertSpace(lattice=lat, ns=ns, is_manybody=True)

    ham = Hamiltonian(hilbert_space=hs, dtype=np.complex128, is_sparse=is_sparse, name=f"matrix_mode_{is_sparse}")
    ops = ham.operators
    sx  = ops.sig_x(ns=ns, type_act="local")

    for i in range(ns):
        ham.add(sx, sites=[i], multiplier=0.2)

    ham.build()
    return ham.hamil

def main():
    print("--- Sparse And Dense Matrix Build ---")

    m_sparse            = _build_example(True)
    m_dense             = _build_example(False)

    print("sparse type:", type(m_sparse), "shape:", m_sparse.shape)
    print("dense type:", type(m_dense), "shape:", m_dense.shape)

    dense_from_sparse   = m_sparse.toarray() if hasattr(m_sparse, "toarray") else np.array(m_sparse)
    diff                = np.linalg.norm(dense_from_sparse - np.asarray(m_dense))
    print("dense-sparse consistency norm:", float(diff))

if __name__ == "__main__":
    main()
