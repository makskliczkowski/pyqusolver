import numpy as np

from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Operator.impl.operators_spin import sigma_x_int_np, sigma_z_int_np
from QES.Algebra.Operator.impl.operators_spin_1 import s1_z
from QES.Algebra.Operator.impl.operators_spinless_fermions import c_dag_int_np, c_int_np, n_int_np
from QES.Algebra.Operator.operator import OperatorTypeActing
from QES.general_python.lattices import SquareLattice


def _build_tfim(ns: int, is_sparse: bool):
    lat = SquareLattice(dim=1, lx=ns, bc="pbc")
    hs = HilbertSpace(lattice=lat, ns=ns, is_manybody=True)

    ham = Hamiltonian(
        hilbert_space=hs,
        dtype=np.complex128,
        is_sparse=is_sparse,
        name=f"tfim_sparse_{is_sparse}",
    )
    ops = ham.operators
    sx = ops.sig_x(ns=ns, type_act="local")
    sz_corr = ops.sig_z(ns=ns, type_act="correlation")

    for i in range(ns):
        j = (i + 1) % ns
        ham.add(sz_corr, sites=[i, j], multiplier=1.0)
        ham.add(sx, sites=[i], multiplier=0.35)

    ham.build()
    return hs, ham


def test_hilbert_symmetry_reduction_changes_effective_dimension():
    lat = SquareLattice(dim=1, lx=6, bc="pbc")
    hs_full = HilbertSpace(lattice=lat, ns=6, is_manybody=True)
    hs_sym = HilbertSpace(
        lattice=lat,
        ns=6,
        is_manybody=True,
        sym_gen={"translation": 0},
        gen_mapping=True,
    )

    assert hs_full.nh == hs_full.nhfull
    assert hs_sym.nh <= hs_sym.nhfull
    assert hs_sym.nh < hs_full.nh


def test_sparse_and_dense_hamiltonian_builds_are_consistent():
    _, ham_sparse = _build_tfim(ns=4, is_sparse=True)
    _, ham_dense = _build_tfim(ns=4, is_sparse=False)

    dense_sparse = ham_sparse.hamil.toarray() if hasattr(ham_sparse.hamil, "toarray") else np.asarray(ham_sparse.hamil)
    dense_direct = np.asarray(ham_dense.hamil)

    np.testing.assert_allclose(dense_sparse, dense_direct, rtol=1e-10, atol=1e-10)


def test_quadratic_tight_binding_spectrum_is_sorted_and_symmetric():
    qh = QuadraticHamiltonian(ns=8, particle_conserving=True, particles="fermions", dtype=np.complex128)
    for i in range(8):
        qh.add_hopping(i, (i + 1) % 8, -1.0)

    qh.build()
    qh.diagonalize()
    evals = np.sort(np.real(np.asarray(qh.eig_val)))

    assert evals.shape == (8,)
    np.testing.assert_allclose(evals, -evals[::-1], atol=1e-10)


def test_operator_actions_on_integer_and_vector_states_are_stable():
    sx_states, sx_coeffs = sigma_x_int_np(0b0010, 4, [1])
    sz_states, sz_coeffs = sigma_z_int_np(0b0010, 4, [1])
    cd_states, cd_coeffs = c_dag_int_np(0b0010, 4, [0])
    c_states, c_coeffs = c_int_np(0b0010, 4, [2])
    n_states, n_coeffs = n_int_np(0b0010, 4, [2])

    assert sx_states == (0b0110,)
    assert sz_states == (0b0010,)
    assert sx_coeffs == (0.5,)
    assert sz_coeffs == (0.5,)
    assert cd_states == (0b1010,)
    assert cd_coeffs == (1.0,)
    assert c_states == (0,)
    assert c_coeffs == (1.0,)
    assert n_states == (0b0010,)
    assert n_coeffs == 1.0

    op_s1z = s1_z(ns=1, type_act=OperatorTypeActing.Local, sites=[0])
    psi_s1 = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
    out_states, out_coeffs = op_s1z(psi_s1)
    np.testing.assert_allclose(out_states, np.array([[0.0, 1.0, 0.0]], dtype=np.complex128))
    np.testing.assert_allclose(out_coeffs, np.array([1.0]))


def test_lattice_neighbor_driven_build_produces_finite_spectrum():
    lat = SquareLattice(dim=1, lx=6, bc="pbc")
    hs = HilbertSpace(lattice=lat, ns=lat.ns, is_manybody=True)
    ham = Hamiltonian(hilbert_space=hs, dtype=np.complex128, name="neighbor_tfim")
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
    vals = np.real(np.asarray(ham.eigenvalues))

    assert np.all(np.isfinite(vals))
    assert vals.size > 0
