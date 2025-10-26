"""
Simple integration test for the modular diagonalization system.

Tests basic functionality of the new DiagonalizationEngine integration
with the Hamiltonian class.

File        : QES/Algebra/Hamil/test_diag_integration.py
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

import numpy as np
import scipy.sparse as sp
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# ----------------------------------
#! Test functions
# ----------------------------------

def test_diagonalization_engine_import():
    """Test that DiagonalizationEngine can be imported."""
    try:
        from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
        print("(ok)  DiagonalizationEngine imported successfully")
        return True
    except ImportError as e:
        print(f"(x) Failed to import DiagonalizationEngine: {e}")
        return False

# ----------------------------------
#! Test eigen solvers availability
# ----------------------------------

def test_eigensolvers_available():
    """Test that eigen solvers module is available."""
    try:
        from QES.general_python.algebra.eigen.factory import choose_eigensolver
        from QES.general_python.algebra.eigen.result import EigenResult
        print("(ok)  Eigen solvers module available")
        return True
    except ImportError as e:
        print(f"(x) Eigen solvers not available: {e}")
        return False

# ----------------------------------
#! Integration tests for diagonalization methods
# ----------------------------------

def test_simple_matrix_exact():
    """Test exact diagonalization of a simple matrix."""
    try:
        from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
        
        # Create simple symmetric matrix
        n = 50
        np.random.seed(42)
        A = np.random.randn(n, n)
        A = 0.5 * (A + A.T)  # Make symmetric
        
        # Create engine and diagonalize
        engine = DiagonalizationEngine(method='exact', backend='numpy', verbose=False)
        result = engine.diagonalize(A, hermitian=True)
        
        # Verify
        assert result.eigenvalues is not None
        assert result.eigenvectors is not None
        assert len(result.eigenvalues) == n
        assert result.eigenvectors.shape == (n, n)
        
        # Check eigenvalue equation
        for i in range(min(5, n)):
            v           = result.eigenvectors[:, i]
            lam         = result.eigenvalues[i]
            residual    = np.linalg.norm(A @ v - lam * v)
            assert residual < 1e-10, f"Eigenvalue equation not satisfied: residual={residual}"
        
        print("(ok)  Exact diagonalization test passed")
        return True
        
    except Exception as e:
        print(f"(x) Exact diagonalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------
#! Lanczos test
# ----------------------------------

def test_simple_matrix_lanczos():
    """Test Lanczos diagonalization of a simple sparse matrix."""
    try:
        from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
        
        # Create simple symmetric sparse matrix
        n       = 200
        k       = 10
        np.random.seed(42)
        
        # Diagonal + random sparse perturbation
        diag    = np.arange(n, dtype=float)
        A       = sp.diags(diag, format='csr')
        
        # Create engine and diagonalize
        engine  = DiagonalizationEngine(method='lanczos', backend='scipy', verbose=False)
        result  = engine.diagonalize(A, k=k, which='smallest', hermitian=True)
        
        # Verify
        assert result.eigenvalues is not None
        assert result.eigenvectors is not None
        assert len(result.eigenvalues) == k
        
        # Check that we got the k smallest
        expected = np.arange(k, dtype=float)
        assert np.allclose(result.eigenvalues, expected, rtol=1e-8)
        
        print("(ok)  Lanczos diagonalization test passed")
        return True
        
    except Exception as e:
        print(f"(x) Lanczos diagonalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------
#! Basis transformation test
# ----------------------------------

def test_basis_transformations():
    """Test Krylov basis transformations."""
    try:
        from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
        
        # Create simple matrix
        n           = 100
        k           = 10
        np.random.seed(42)
        A           = sp.diags(np.arange(n, dtype=float), format='csr')

        # Diagonalize with basis storage
        engine      = DiagonalizationEngine(method='lanczos', backend='scipy', verbose=False)
        result      = engine.diagonalize(A, k=k, which='smallest', store_basis=True)

        # Check basis availability
        assert engine.has_krylov_basis(), "Krylov basis should be available"
        
        # Get basis
        V           = engine.get_krylov_basis()
        assert V is not None, "Krylov basis should not be None"
        
        # Test transformation: create vector in Krylov basis
        v_krylov    = np.zeros(k)
        v_krylov[0] = 1.0  # First basis vector
        
        # Transform to original
        v_original  = engine.to_original_basis(v_krylov)
        assert v_original.shape == (n,), f"Original vector wrong shape: {v_original.shape}"
        
        # Transform back
        v_krylov_reconstructed = engine.to_krylov_basis(v_original)
        assert np.allclose(v_krylov, v_krylov_reconstructed, rtol=1e-10), \
            "Round-trip transformation failed"
        
        print("(ok)  Basis transformation test passed")
        return True
        
    except Exception as e:
        print(f"(x) Basis transformation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------
#! Auto method selection test
# ----------------------------------

def test_method_auto_selection():
    """Test automatic method selection."""
    try:
        from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
        from QES.general_python.algebra.eigen.factory import decide_method
        
        # Small matrix -> exact
        method = decide_method(n=100, k=None, hermitian=True)
        assert method == 'exact', f"Expected 'exact' for small matrix, got '{method}'"
        
        # Large matrix, few eigenvalues -> lanczos
        method = decide_method(n=1000, k=5, hermitian=True)
        assert method in ['lanczos', 'block_lanczos'], \
            f"Expected iterative method for large matrix, got '{method}'"
        
        # Large matrix, many eigenvalues -> block_lanczos
        method = decide_method(n=5000, k=50, hermitian=True)
        assert method == 'block_lanczos', \
            f"Expected 'block_lanczos' for many eigenvalues, got '{method}'"
        
        print("(ok)  Auto method selection test passed")
        return True
        
    except Exception as e:
        print(f"(x) Auto method selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------
#! Main test runner
# ----------------------------------

def main():
    """Run all tests."""
    print("=" * 80)
    print("DIAGONALIZATION INTEGRATION TESTS")
    print("=" * 80)
    print()
    
    tests = [
        ("Import DiagonalizationEngine",    test_diagonalization_engine_import),
        ("Eigen solvers available",         test_eigensolvers_available),
        ("Exact diagonalization",           test_simple_matrix_exact),
        ("Lanczos diagonalization",         test_simple_matrix_lanczos),
        ("Basis transformations",           test_basis_transformations),
        ("Auto method selection",           test_method_auto_selection),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        print("-" * 80)
        passed = test_func()
        results.append((name, passed))
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "(ok) " if passed else "(x)"
        print(f"{symbol} {name:40s} {status}")
    
    total           = len(results)
    passed_count    = sum(1 for _, p in results if p)
    print()
    print(f"Total: {passed_count}/{total} tests passed")
    print("=" * 80)
    
    return all(p for _, p in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------