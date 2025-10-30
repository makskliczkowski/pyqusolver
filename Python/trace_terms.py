import numpy as np
from scipy.sparse import csr_matrix
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


def trace_one_element(ns=6, k_in=0, k_out=0, i_full=0, j_full=1):
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

    print('U_in nonzeros (idx: val):')
    nz_in = np.nonzero(np.abs(U_in_arr) > 1e-14)
    for r,c in zip(nz_in[0], nz_in[1]):
        if c == 0:
            print(r, U_in_arr[r,c])
    print('\nU_out nonzeros (idx: val):')
    nz_out = np.nonzero(np.abs(U_out_arr) > 1e-14)
    for r,c in zip(nz_out[0], nz_out[1]):
        if c == 0:
            print(r, U_out_arr[r,c])

    print('\nHilbert in mapping and normalization (first 20):')
    print('mapping_in[:20]=', getattr(hil_in, 'mapping')[:20])
    print('normalization_in[:20]=', getattr(hil_in, 'normalization')[:20])
    print('\nHilbert out mapping and normalization (first 20):')
    print('mapping_out[:20]=', getattr(hil_out, 'mapping')[:20])
    print('normalization_out[:20]=', getattr(hil_out, 'normalization')[:20])

    print('\nH_block nonzeros (a,b,val):')
    rows, cols = np.nonzero(np.abs(H_block_dense) > 1e-14)
    for a,b in zip(rows, cols):
        print(a,b, H_block_dense[a,b])

    # compute contribution to full element (i_full, j_full)
    contribs = []
    for a in range(H_block_dense.shape[0]):
        for b in range(H_block_dense.shape[1]):
            val = H_block_dense[a,b]
            if abs(val) < 1e-14:
                continue
            u_out = U_out_arr[i_full, a]
            u_in = U_in_arr[j_full, b]
            term = u_out * val * np.conjugate(u_in)
            if abs(term) > 1e-14:
                contribs.append(((a,b), u_out, val, u_in, term))

    print('\nContributions to H_rec[{},{}] from sector pair k_out={},k_in={}:'.format(i_full, j_full, k_out, k_in))
    total = 0
    for (a,b), uout, val, uin, term in contribs:
        print('a,b=',a,b,'u_out=',uout,'H_block=',val,'u_in=',uin,'term=',term)
        total += term
    print('Sum of contributions:', total)

    # Print H_full element
    H_full = build_full_sigma_x(ns).toarray()
    print('H_full[{},{}] = {}'.format(i_full, j_full, H_full[i_full, j_full]))

if __name__ == '__main__':
    trace_one_element(ns=6, k_in=0, k_out=0, i_full=0, j_full=1)
