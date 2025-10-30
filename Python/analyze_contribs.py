import numpy as np
from QES.Algebra.hilbert import HilbertSpace, SymmetryGenerators
from QES.general_python.lattices.square import SquareLattice
from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix, get_symmetry_rotation_matrix
import numba

@numba.njit
def sigma_x_op(state, ns):
    new_states = np.empty(ns, dtype=np.int64)
    for i in range(ns):
        new_states[i] = state ^ (1 << i)
    return new_states, np.ones(ns, dtype=np.float64)


def contribution_for_pair(ns, k_in, k_out, i_full=0, j_full=1):
    lattice = SquareLattice(1, ns)
    hil_in = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, k_in)], gen_mapping=True)
    hil_out = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, k_out)], gen_mapping=True)

    H_block = build_operator_matrix(hil_in, sigma_x_op, hilbert_space_out=hil_out, sparse=True)
    U_in = get_symmetry_rotation_matrix(hil_in)
    U_out = get_symmetry_rotation_matrix(hil_out)

    U_in_arr = U_in.toarray() if hasattr(U_in, 'toarray') else np.asarray(U_in)
    U_out_arr = U_out.toarray() if hasattr(U_out, 'toarray') else np.asarray(U_out)

    H_block_dense = H_block.toarray() if hasattr(H_block, 'toarray') else np.array(H_block)
    H_block_dense = np.asarray(H_block_dense, dtype=np.complex128)

    # compute partial contribution to element
    contrib = np.zeros((), dtype=np.complex128)
    for a in range(H_block_dense.shape[0]):
        for b in range(H_block_dense.shape[1]):
            val = H_block_dense[a,b]
            if abs(val) < 1e-14:
                continue
            contrib += U_out_arr[i_full, a] * val * np.conjugate(U_in_arr[j_full, b])
    return contrib

if __name__ == '__main__':
    ns = 6
    total = 0
    contribs = []
    for k_out in range(ns):
        for k_in in range(ns):
            c = contribution_for_pair(ns, k_in, k_out)
            if abs(c) > 1e-12:
                contribs.append((k_out, k_in, c))
            total += c
    print('Non-zero per-pair contributions:')
    for k_out, k_in, c in contribs:
        print('k_out={}, k_in={}, contrib={}'.format(k_out, k_in, c))
    print('Sum total:', total)
    # Compare with H_full
    from scipy.sparse import csr_matrix
    H_full = csr_matrix(([], ([], [])), shape=(1,1))
    # Quick direct compute of H_full element
    from QES.debug_reconstruct import build_full_sigma_x
    H_full = build_full_sigma_x(ns).toarray()
    print('H_full[0,1]=', H_full[0,1])
