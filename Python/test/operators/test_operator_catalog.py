import numpy as np

from QES.Algebra.Hilbert.hilbert_local import LocalSpace
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Operator.operator import OperatorTypeActing


def test_spin_catalog_keys():
    space = LocalSpace.default_spin_half()
    keys = space.list_operator_keys()
    for required in {"sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus"}:
        assert required in keys
    sigma_plus = space.get_op("sigma_plus")
    assert "raising" in sigma_plus.tags


def test_fermion_creation_sign():
    space = LocalSpace.default_fermion_spinless()
    creation = space.get_op("c_dag").kernels
    # initial state 0b100, create on site 1 (middle)
    out_state, coeff = creation.fun_int(0b100, 3, [1])
    assert out_state[0] == 0b110
    assert coeff[0] == -1.0
    # attempt to create on already occupied site gives zero coefficient
    out_state, coeff = creation.fun_int(0b100, 3, [0])
    assert coeff[0] == 0.0


def test_anyon_phase_semion():
    theta = np.pi / 2
    space = LocalSpace.default_abelian_anyon(statistics_angle=theta)
    creation = space.get_op("a_dag").kernels
    out_state, coeff = creation.fun_int(0b100, 3, [1])
    assert out_state[0] == 0b110
    expected = np.exp(1j * theta)
    assert np.allclose(coeff[0], expected)


def test_hilbert_build_local_operator():
    space = LocalSpace.default_fermion_spinless()
    hilbert = HilbertSpace(ns=3, local_space=space, backend="default")
    op = hilbert.build_local_operator("c_dag")
    assert op.type_acting == OperatorTypeActing.Local
    new_state, amplitude = op(0b100, 1)
    assert new_state[0] == 0b110
    assert amplitude[0] == -1.0

