import numpy as np

from QES.Algebra.Operator.impl.operators_spin import sigma_x_int_np, sigma_z_int_np
from QES.Algebra.Operator.impl.operators_spin_1 import s1_z
from QES.Algebra.Operator.impl.operators_spinless_fermions import c_dag_int_np, c_int_np, n_int_np
from QES.Algebra.Operator.operator import OperatorTypeActing


def main():
    print("--- Operator Actions On States ---")

    ns = 4
    state = 0b0010

    sx_states, sx_coeffs = sigma_x_int_np(state, ns, [1])
    sz_states, sz_coeffs = sigma_z_int_np(state, ns, [1])
    print("spin-1/2 sigma_x_int:", sx_states, sx_coeffs)
    print("spin-1/2 sigma_z_int:", sz_states, sz_coeffs)

    cd_states, cd_coeffs = c_dag_int_np(state, ns, [0])
    c_states, c_coeffs = c_int_np(state, ns, [2])
    n_states, n_coeffs = n_int_np(state, ns, [2])
    print("fermion c_dag:", cd_states, cd_coeffs)
    print("fermion c:", c_states, c_coeffs)
    print("fermion n:", n_states, n_coeffs)

    op_s1z = s1_z(ns=1, type_act=OperatorTypeActing.Local, sites=[0])
    psi_s1 = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
    s1_states, s1_coeffs = op_s1z(psi_s1)
    print("spin-1 S_z on vector:", s1_states, s1_coeffs)


if __name__ == "__main__":
    main()
