import numpy as np
from QES.Algebra.hilbert import HilbertSpace, SymmetryGenerators
from QES.general_python.lattices.square import SquareLattice
from QES.Algebra.Hilbert.matrix_builder import get_symmetry_rotation_matrix


def diagnose(ns=6, sector_index=0):
    lattice = SquareLattice(1, ns)
    # build a Hilbert space for a given momentum sector
    hil = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, sector_index)], gen_mapping=True)

    print('nh_reduced (dim):', hil.dim)
    print('nh_full:', getattr(hil, '_nh_full', 2**hil.ns))

    sym_group = hil.sym_group
    print('sym_group size:', len(sym_group))

    k = 0
    rep = int(hil.mapping[k])
    norm_k = hil.normalization[k] if hil.normalization is not None else 1.0
    print('\nColumn k={}, representative={}, norm_k={}'.format(k, rep, norm_k))

    # Fallback expansion contributions (what matrix_builder does when cast_to_full not used)
    contributions = {}
    for i, g in enumerate(sym_group):
        try:
            state_i, phase = g(rep)
        except Exception as e:
            print('group element call failed', e)
            continue
        val = np.conjugate(phase) / (norm_k * np.sqrt(len(sym_group)))
        contributions.setdefault(int(state_i), 0.0)
        contributions[int(state_i)] += val
        print('g[{}]: state={}, phase={}, contrib={}'.format(i, int(state_i), phase, val))

    print('\nAggregated fallback contributions (index: value):')
    for idx in sorted(contributions.keys()):
        print(idx, contributions[idx])

    # Cast_to_full expansion (if available)
    print()    
    if hasattr(hil, 'cast_to_full'):
        vec_reduced = np.zeros(hil.dim, dtype=np.complex128)
        vec_reduced[k] = 1.0
        vec_full = hil.cast_to_full(vec_reduced)
        nz = np.nonzero(np.abs(vec_full) > 1e-14)[0]
        print('cast_to_full nonzeros (index: value):')
        for idx in nz:
            print(idx, vec_full[int(idx)])
    else:
        print('Hilbert has no cast_to_full')

    # Rotation matrix U column from canonical builder
    print()    
    U = get_symmetry_rotation_matrix(hil)
    U_arr = U.toarray()
    col = U_arr[:, k]
    nz = np.nonzero(np.abs(col) > 1e-14)[0]
    print('U column nonzeros (index: value):')
    for idx in nz:
        print(idx, col[int(idx)])

    # Compare per-index values
    print('\nComparison (index: fallback, cast_to_full, U):')
    all_idxs = set(list(contributions.keys()) + (list(nz) if len(nz) else []))
    if hasattr(hil, 'cast_to_full'):
        all_idxs |= set(np.nonzero(np.abs(vec_full) > 1e-14)[0].tolist())
    for idx in sorted(all_idxs):
        f = contributions.get(idx, 0.0)
        c = vec_full[int(idx)] if 'vec_full' in locals() else None
        u = col[int(idx)] if idx in nz else 0.0
        print(idx, f, c, u)


if __name__ == '__main__':
    diagnose(ns=6, sector_index=0)
