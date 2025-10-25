"""
Quick test for native MINRES implementation.

This test compares the native MINRES solver against SciPy's implementation
on simple symmetric positive definite and indefinite systems.

File        : Python/test_minres_native.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
"""
import numpy as np
import sys
import os

# Ensure the module path is correct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'QES'))

from QES.general_python.algebra.solvers.minres import MinresSolver, MinresSolverScipy

# -------------------------------------------------------------------

def test_minres_simple():
    """Test MINRES on a simple symmetric system."""
    print("\n=== Testing Native MINRES ===")
    
    # Create a simple SPD system
    n = 10
    A = np.random.rand(n, n)
    A = A + A.T + n * np.eye(n)  # Make symmetric positive definite
    
    b = np.random.rand(n)
    x_true = np.linalg.solve(A, b)
    x0 = np.zeros(n)
    
    # Define matvec
    def matvec(v):
        return A @ v
    
    # Test native MINRES
    result = MinresSolver.solve(
        matvec=matvec,
        b=b,
        x0=x0,
        tol=1e-10,
        maxiter=n*2,
        backend_module=np
    )
    
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Residual norm: {result.residual_norm:.2e}")
    print(f"Solution error: {np.linalg.norm(result.x - x_true):.2e}")
    print(f"||Ax - b||: {np.linalg.norm(A @ result.x - b):.2e}")
    
    assert result.converged, "MINRES did not converge"
    assert np.linalg.norm(result.x - x_true) < 1e-6, "Solution error too large"
    print("(v) Native MINRES passed!")
    
def test_minres_vs_scipy():
    """Compare native MINRES with SciPy MINRES."""
    print("\n=== Comparing Native vs SciPy MINRES ===")
    
    n = 20
    A = np.random.rand(n, n)
    A = A + A.T + n * np.eye(n)
    
    b = np.random.rand(n)
    x0 = np.zeros(n)
    
    def matvec(v):
        return A @ v
    
    # Native MINRES
    result_native = MinresSolver.solve(
        matvec=matvec,
        b=b,
        x0=x0,
        tol=1e-10,
        maxiter=n*2,
        backend_module=np
    )
    
    # SciPy MINRES
    result_scipy = MinresSolverScipy.solve(
        matvec=matvec,
        b=b,
        x0=x0,
        tol=1e-10,
        maxiter=n*2,
        backend_module=np
    )
    
    print(f"Native - Converged: {result_native.converged}, Iterations: {result_native.iterations}, Residual: {result_native.residual_norm:.2e}")
    print(f"SciPy  - Converged: {result_scipy.converged}, Iterations: {result_scipy.iterations}, Residual: {result_scipy.residual_norm if result_scipy.residual_norm else 'N/A'}")
    print(f"Solution difference: {np.linalg.norm(result_native.x - result_scipy.x):.2e}")
    
    assert np.allclose(result_native.x, result_scipy.x, rtol=1e-6), "Solutions differ significantly"
    print("(v) Native and SciPy MINRES produce similar results!")

def test_minres_indefinite():
    """Test MINRES on an indefinite symmetric system."""
    print("\n=== Testing MINRES on Indefinite System ===")
    
    n = 15
    # Create indefinite symmetric matrix
    A = np.random.rand(n, n)
    A = A + A.T
    # Make it have mixed eigenvalues
    eigs = np.random.rand(n) * 2 - 1  # Range [-1, 1]
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    A = Q @ np.diag(eigs) @ Q.T
    
    b = np.random.rand(n)
    x_true = np.linalg.solve(A, b)
    x0 = np.zeros(n)
    
    def matvec(v):
        return A @ v
    
    result = MinresSolver.solve(
        matvec=matvec,
        b=b,
        x0=x0,
        tol=1e-8,
        maxiter=n*3,
        backend_module=np
    )
    
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Residual norm: {result.residual_norm:.2e}")
    print(f"Solution error: {np.linalg.norm(result.x - x_true):.2e}")
    
    assert np.linalg.norm(result.x - x_true) < 1e-5, "Solution error too large for indefinite system"
    print("(v) MINRES works on indefinite systems!")

if __name__ == "__main__":
    test_minres_simple()
    test_minres_vs_scipy()
    test_minres_indefinite()
    print("\n🎉 All MINRES tests passed!")

# -------------------------------------------------------------------
#! End of file
# -------------------------------------------------------------------