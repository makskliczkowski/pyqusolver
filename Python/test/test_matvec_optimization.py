
import numpy as np
import pytest
from QES.Algebra.Operator.operator import create_operator, OperatorTypeActing
from QES.Algebra.hilbert import HilbertSpace
import numba

# Simple operator for testing: Sum of Z_i
@numba.njit
def op_z_int(state, ns, sites):
    # This operator does not change the state, but multiplies by a value
    # Value is sum of (2*s_i - 1)
    val = 0.0
    for i in range(ns):
        # bit extraction: (state >> i) & 1
        bit = (state >> i) & 1
        val += (2.0 * bit - 1.0)

    st = np.empty(1, dtype=np.int64)
    st[0] = state
    v = np.empty(1, dtype=np.float64)
    v[0] = val
    return st, v

def op_z_np(state, sites):
    # state is (batch, )
    # Not implemented for this test
    raise NotImplementedError

def test_matvec_correctness():
    ns = 8
    nh = 2**ns
    hilbert = HilbertSpace(ns=ns)

    op = create_operator(
        type_act=OperatorTypeActing.Global,
        op_func_int=op_z_int,
        op_func_np=op_z_np,
        op_func_jnp=None,
        ns=ns
    )

    # 1. Build dense matrix to verify against
    # Force dense build by setting internal flag
    op._is_sparse = False
    op.build(hilbert=hilbert, force=True)
    dense_mat = op.matrix_data

    # 2. Generate random vector
    np.random.seed(42)
    vec = np.random.rand(nh) + 1j * np.random.rand(nh)

    # 3. Compute matvec using the optimized function (multithreaded=True)
    # Note: matvec uses the operator function, not the built matrix
    res_matvec = op.matvec(vec, hilbert_in=hilbert, multithreaded=True)

    # 4. Compute reference using dense matrix multiplication
    res_ref = dense_mat @ vec

    # 5. Compare
    diff = np.linalg.norm(res_matvec - res_ref)
    assert diff < 1e-10, f"Matvec mismatch: {diff}"
    print(f"Matvec correctness verified. Diff: {diff}")

def test_matvec_batch_correctness():
    ns = 8
    nh = 2**ns
    hilbert = HilbertSpace(ns=ns)

    op = create_operator(
        type_act=OperatorTypeActing.Global,
        op_func_int=op_z_int,
        op_func_np=op_z_np,
        op_func_jnp=None,
        ns=ns
    )

    op._is_sparse = False
    op.build(hilbert=hilbert, force=True)
    dense_mat = op.matrix_data

    # Batch size > chunk size usually used in optimization
    batch_size = 10
    vecs = np.random.rand(nh, batch_size) + 1j * np.random.rand(nh, batch_size)

    res_matvec = op.matvec(vecs, hilbert_in=hilbert, multithreaded=True)
    res_ref = dense_mat @ vecs

    diff = np.linalg.norm(res_matvec - res_ref)
    assert diff < 1e-10, f"Batch matvec mismatch: {diff}"
    print(f"Batch matvec correctness verified. Diff: {diff}")

if __name__ == "__main__":
    test_matvec_correctness()
    test_matvec_batch_correctness()
