import sys
sys.path.insert(0, '/Users/makskliczkowski/Codes/pyqusolver/Python')

import numpy as np
from scipy.sparse import csr_matrix
from QES.Algebra.hilbert import HilbertSpace, SymmetryGenerators
from QES.general_python.lattices.square import SquareLattice
from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix


def build_full_sigma_x(ns):
    nh = 2 ** ns
    row = []
    col = []
    data = []
    for state in range(nh):
        for i in range(ns):
            new_state = state ^ (1 << i)
            row.append(state)
            col.append(new_state)
            data.append(1.0)
    return csr_matrix((data, (row, col)), shape=(nh, nh))


def reconstruct_from_sectors(ns):
    lattice = SquareLattice(1, ns)
    H_full = build_full_sigma_x(ns)

    # Accumulator for reconstructed full matrix
    nh = H_full.shape[0]
    H_rec = np.zeros((nh, nh), dtype=np.complex128)

    for k in range(ns):
        # Build Hilbert for sector k
        hil = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, k)], gen_mapping=True)
        H_k = build_operator_matrix(hil, sigma_x_op, sparse=True)

        # Build momentum-resolved basis mapping: rep -> {state: coeff}
        momenta = {lattice.active_directions[0]: k} if hasattr(lattice, 'active_directions') else {0: k}
        basis_map = hil.build_momentum_basis(momenta)

        # Convert H_k to dense for simplicity (small sizes)
        Hk_dense = H_k.toarray() if hasattr(H_k, 'toarray') else np.array(H_k)

        reps = list(basis_map.keys())
        for i, rep_i in enumerate(reps):
            vec_i = np.zeros(nh, dtype=np.complex128)
            for state, coeff in basis_map[rep_i].items():
                vec_i[int(state)] = coeff
            for j, rep_j in enumerate(reps):
                vec_j = np.zeros(nh, dtype=np.complex128)
                for state, coeff in basis_map[rep_j].items():
                    vec_j[int(state)] = coeff
                H_rec += Hk_dense[i, j] * np.outer(vec_i, np.conjugate(vec_j))

    return H_full.toarray(), H_rec


# numba operator used by matrix_builder (same as other tests)
import numba
@numba.njit
def sigma_x_op(state, ns):
    new_states = np.empty(ns, dtype=np.int64)
    for i in range(ns):
        new_states[i] = state ^ (1 << i)
    return new_states, np.ones(ns, dtype=np.float64)


def test_reconstruction_small():
    ns = 6
    H_full, H_rec = reconstruct_from_sectors(ns)
    diff = np.max(np.abs(H_full - H_rec))
    print('max_diff:', diff)
    assert diff < 1e-8


if __name__ == '__main__':
    test_reconstruction_small()
