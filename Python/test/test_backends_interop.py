"""
Test script for backend switcher and interoperability features.
"""

import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, '/Users/makskliczkowski/Codes/pyqusolver/Python')

try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
    from QES.Algebra.backends import get_backend, get_backend_registry, get_available_backends
except ImportError as e:
    print(f"ImportError: {e}")
    print("Make sure QES package is correctly installed or PYTHONPATH is set.")
    sys.exit(1)

print("=" * 70)
print("BACKEND SWITCHER AND INTEROPERABILITY TEST")
print("=" * 70)

# Test 1: Check available backends
print("\n[Test 1] Available Backends")
print("-" * 70)

backend_list = get_available_backends()
print(f"Registered backends: {backend_list}")

# Test 2: Get backends
print("\n[Test 2] Getting Backend Instances")
print("-" * 70)

numpy_backend = get_backend('numpy')
print(f"(ok) NumPy backend: {numpy_backend}")

try:
    jax_backend = get_backend('jax')
    print(f"(ok) JAX backend: {jax_backend}")
except ValueError as e:
    print(f"(x) JAX backend: {e}")

# Test 3: Create QuadraticHamiltonian with backend selection
print("\n[Test 3] QuadraticHamiltonian with Backends")
print("-" * 70)

qh = QuadraticHamiltonian(ns=4, particle_conserving=True, backend='numpy')
qh.add_hopping(0, 1, -1.0)
qh.add_hopping(1, 2, -1.0)
qh.add_hopping(2, 3, -1.0)
qh.add_hopping(3, 0, -1.0)

print(f"(ok) Created QuadraticHamiltonian: {qh}")

# Test 4: Diagonalize and get eigenvalues
print("\n[Test 4] Diagonalization")
print("-" * 70)

qh.diagonalize(verbose=False)
print(f"(ok) Diagonalization successful")
print(f"  Ground state energy: {qh.eig_val[0]:.6f}")
print(f"  First 4 eigenvalues: {qh.eig_val[:4]}")

# Test 5: Backend list from Hamiltonian
print("\n[Test 5] Backend List from Hamiltonian")
print("-" * 70)

backend_list = qh.get_backend_list()
print(f"Available backends from Hamiltonian:")
for name, available in backend_list:
    status = "(ok)" if available else "(x)"
    print(f"  {status} {name}: {available}")

# Test 6: Qiskit Interoperability
print("\n[Test 6] Qiskit Interoperability")
print("-" * 70)

try:
    qiskit_op = qh.to_qiskit_hamiltonian()
    print(f"(ok) Converted to Qiskit operator: {type(qiskit_op).__name__}")
except ImportError as e:
    print(f"(x) Qiskit conversion skipped: {e}")
except Exception as e:
    print(f"(x) Qiskit conversion failed: {e}")

# Test 7: OpenFermion Interoperability
print("\n[Test 7] OpenFermion Interoperability")
print("-" * 70)

try:
    of_ham = qh.to_openfermion_hamiltonian()
    print(f"(ok) Converted to OpenFermion operator: {type(of_ham).__name__}")
    print(f"  Number of terms: {len(of_ham.terms)}")
except ImportError as e:
    print(f"(x) OpenFermion conversion skipped: {e}")
except Exception as e:
    print(f"(x) OpenFermion conversion failed: {e}")

# Test 8: BdG case with interoperability
print("\n[Test 8] BdG Case with Interoperability")
print("-" * 70)

qh_bdg = QuadraticHamiltonian(ns=2, particle_conserving=False)
qh_bdg.add_hopping(0, 1, 1.0)
qh_bdg.add_hopping(1, 0, 1.0)
qh_bdg.add_pairing(0, 1, 0.5)
qh_bdg.add_pairing(1, 0, 0.5)

print(f"Created BdG Hamiltonian: {qh_bdg}")

qh_bdg.diagonalize(verbose=False)
print(f"(ok) Diagonalization successful")

try:
    qiskit_op_bdg = qh_bdg.to_qiskit_hamiltonian()
    print(f"(ok) BdG converted to Qiskit: {type(qiskit_op_bdg).__name__}")
except Exception as e:
    print(f"(x) BdG Qiskit conversion: {e}")

# Test 9: Direct matrix construction
print("\n[Test 9] Direct Matrix Construction")
print("-" * 70)

# Particle-conserving case
h_matrix = np.array([
    [0, -1, 0, 0],
    [-1, 0, -1, 0],
    [0, -1, 0, -1],
    [0, 0, -1, 0]
], dtype=complex)

qh_direct = QuadraticHamiltonian.from_hermitian_matrix(h_matrix, constant=1.0)
qh_direct.diagonalize(verbose=False)

print(f"(ok) Created from Hermitian matrix")
print(f"  Ground state energy: {qh_direct.eig_val[0]:.6f}")
print(f"  Constant offset: {qh_direct._constant_offset}")

# BdG case
h_matrix_bdg = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

v_matrix_bdg = np.array([
    [0, 0.5],
    [0.5, 0]
], dtype=complex)

qh_bdg_direct = QuadraticHamiltonian.from_bdg_matrices(h_matrix_bdg, v_matrix_bdg, constant=0.5)
qh_bdg_direct.diagonalize(verbose=False)

print(f"(ok) Created from BdG matrices")
print(f"  Eigenvalues shape: {qh_bdg_direct.eig_val.shape}")
print(f"  Ground state energy: {qh_bdg_direct.eig_val[0]:.6f}")

print("\n" + "=" * 70)
print("(ok)(ok) All backend and interoperability tests completed!")
print("=" * 70)

# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------