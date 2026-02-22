
import sys
import os
import time
import numpy as np

# Ensure QES is in path if running from root
current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import QES
from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Operator.operator_loader import get_operator_module
from QES.Algebra.hamil_cache import generate_cache_key

# Ensure reproducible run
QES.qes_reseed(42)

def benchmark():
    # Setup
    ns = 8
    # Use complex dtype to avoid Numba compilation issues with spin operators
    hilbert = HilbertSpace(ns=ns, is_manybody=True, dtype=np.complex128)

    # Create Hamiltonian 1
    h1 = Hamiltonian(hilbert_space=hilbert, name="H1")
    ops = h1.operators

    # Create generic local operators
    op_x = ops.sig_x(ns=ns, type_act='local')
    op_z = ops.sig_z(ns=ns, type_act='local')

    # Add terms
    for i in range(ns):
        h1.add(op_x, 0.5, sites=[i])
        h1.add(op_z, 1.0, sites=[i])

    print(f"Signature H1: {h1.signature}")
    key1 = generate_cache_key(h1)

    # First build
    t0 = time.perf_counter()
    h1.build()
    t1 = time.perf_counter()
    print(f"H1 first build time: {t1 - t0:.6f}s")

    # Test matvec
    v = np.random.rand(hilbert.nh).astype(np.complex128)
    # This uses Operator.matvec -> adapter -> Numba kernels
    res = h1.matvec(v, hilbert_in=hilbert)
    print(f"Matvec result norm: {np.linalg.norm(res)}")

    # Second build (should be instant)
    t0 = time.perf_counter()
    h1.build()
    t1 = time.perf_counter()
    print(f"H1 second build time (cached): {t1 - t0:.6f}s")

    # Create Hamiltonian 2 (Identical)
    h2 = Hamiltonian(hilbert_space=hilbert, name="H2")
    for i in range(ns):
        h2.add(op_x, 0.5, sites=[i])
        h2.add(op_z, 1.0, sites=[i])

    print(f"Signature H2: {h2.signature}")
    key2 = generate_cache_key(h2)

    if key1 == key2:
        print("Keys are identical.")
    else:
        print("Keys differ!")

    # Build H2 (should hit cache)
    t0 = time.perf_counter()
    h2.build()
    t1 = time.perf_counter()
    print(f"H2 build time (should hit cache): {t1 - t0:.6f}s")

    if h1.hamil is h2.hamil:
        print("Matrices are identical objects (shared).")
    else:
        print("Matrices are different objects (copied?).")

if __name__ == "__main__":
    benchmark()
