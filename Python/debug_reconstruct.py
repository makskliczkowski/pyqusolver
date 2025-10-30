import numpy as np
from scipy.sparse import csr_matrix
from QES.Algebra.hilbert import HilbertSpace, SymmetryGenerators
from QES.general_python.lattices.square import SquareLattice
from QES.general_python.lattices.lattice import LatticeDirection
from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix, get_symmetry_rotation_matrix


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


def reconstruct_with_U(ns):
    lattice = SquareLattice(1, ns)
    H_full = build_full_sigma_x(ns).toarray()
    nh = H_full.shape[0]
    H_rec = np.zeros((nh, nh), dtype=np.complex128)

    for k_out in range(ns):
        for k_in in range(ns):
            hil_in = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, k_in)], gen_mapping=True)
            hil_out = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, k_out)], gen_mapping=True)
            H_block = build_operator_matrix(hil_in, sigma_x_op, hilbert_space_out=hil_out, sparse=True)

            # build U matrices via canonical function
            U_in = get_symmetry_rotation_matrix(hil_in)
            U_out = get_symmetry_rotation_matrix(hil_out)

            # Convert sparse rotation matrices to dense arrays for clarity in
            # this debug reconstruction (small system)
            U_in_arr = U_in.toarray() if hasattr(U_in, 'toarray') else np.asarray(U_in)
            U_out_arr = U_out.toarray() if hasattr(U_out, 'toarray') else np.asarray(U_out)

            H_block_dense = H_block.toarray() if hasattr(H_block, 'toarray') else np.array(H_block)
            H_block_dense = np.asarray(H_block_dense, dtype=np.complex128)
            H_rec += U_out_arr.dot(H_block_dense).dot(U_in_arr.conjugate().T)

    return H_full, H_rec

# numba operator
import numba
@numba.njit
def sigma_x_op(state, ns):
    new_states = np.empty(ns, dtype=np.int64)
    for i in range(ns):
        new_states[i] = state ^ (1 << i)
    return new_states, np.ones(ns, dtype=np.float64)

if __name__ == '__main__':
    H_full, H_rec = reconstruct_with_U(6)
    diff = np.max(np.abs(H_full - H_rec))
    print('max_diff:', diff)
    print('||H_full||_max:', np.max(np.abs(H_full)))
    print('||H_rec||_max:', np.max(np.abs(H_rec)))
    nz_full = np.count_nonzero(np.abs(H_full) > 1e-12)
    nz_rec = np.count_nonzero(np.abs(H_rec) > 1e-12)
    print('nnz full:', nz_full, 'nnz rec:', nz_rec)
    idxs = np.argwhere(np.abs(H_full - H_rec) > 1e-8)
    print('examples diffs (up to 10):')
    for r, c in idxs[:10]:
        print(r, c, H_full[r, c], H_rec[r, c])
