import sys
import os
import numpy as np

# Ensure path
current_dir = os.getcwd()
python_path = os.path.join(current_dir, "Python")
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    import jax.numpy as jnp
    backend = jnp
    print("Using JAX backend")
except ImportError:
    backend = np
    print("Using NumPy backend")

from QES.Algebra.Operator.matrix import DummyVector

print("--- Testing DummyVector.astype ---")
# Reproduce the DummyVector.astype issue
v = DummyVector(1.0, ns=5, backend=backend)

# Test with type
try:
    v.astype(np.float64)
    print("DummyVector.astype(np.float64) OK")
except Exception as e:
    print(f"DummyVector.astype(np.float64) FAILED: {e}")

# Test with dtype object
try:
    v.astype(np.dtype(np.float64))
    print("DummyVector.astype(np.dtype(float64)) OK")
except Exception as e:
    print(f"DummyVector.astype(np.dtype(float64)) FAILED: {e}")

# Test complex
try:
    v.astype(np.complex128)
    print("DummyVector.astype(np.complex128) OK")
except Exception as e:
    print(f"DummyVector.astype(np.complex128) FAILED: {e}")


print("\n--- Testing matrix_builder ---")
from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix
import numba

# Define a simple JIT function
@numba.njit
def simple_op(state):
    return (np.array([state], dtype=np.int64), np.array([1.0], dtype=np.complex128))

class MockOperator:
    def __init__(self):
        self.int = simple_op
        self.fun = "something"

op = MockOperator()

try:
    # nh=2, ns=1
    # This calls _build_no_hilbert internally which uses JIT functions
    mat = build_operator_matrix(op, nh=2, ns=1, sparse=True, dtype=np.complex128)
    print("build_operator_matrix with Operator-like object OK")
    print(f"Matrix shape: {mat.shape}")
except Exception as e:
    print(f"build_operator_matrix FAILED: {e}")
    import traceback
    traceback.print_exc()
