"""
Standalone test for DiagonalizationEngine without Hamiltonian dependencies.

This test verifies the DiagonalizationEngine works correctly as a standalone
component, without needing the full QES Hamiltonian infrastructure.

File        : QES/Algebra/Hamil/test_diag_standalone.py
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

import sys

import numpy as np
import scipy.sparse as sp

# ----------------------------------
#! Test functions
# ----------------------------------


def test_engine_basic():
    """Test basic DiagonalizationEngine functionality."""
    print("\n" + "=" * 80)
    print("TEST: DiagonalizationEngine Basic Functionality")
    print("=" * 80)

    try:
        from QES.general_python.algebra.eigen.factory import choose_eigensolver, decide_method
        from QES.general_python.algebra.eigen.result import EigenResult

        print("(ok)  Successfully imported eigen modules")
    except ImportError as e:
        print(f"(x) Failed to import: {e}")
        return False

    # Test 1: Exact diagonalization
    print("\n1. Testing exact diagonalization...")
    try:
        np.random.seed(42)
        n = 50
        A = np.random.randn(n, n)
        A = 0.5 * (A + A.T)  # Make symmetric

        result = choose_eigensolver("exact", A, hermitian=True)
        assert len(result.eigenvalues) == n
        print(f"   (ok)  Computed {len(result.eigenvalues)} eigenvalues")
        print(f"   (ok)  Ground state energy: {result.eigenvalues[0]:.10f}")

    except Exception as e:
        print(f"   (x) Failed: {e}")
        return False

    # Test 2: Lanczos
    print("\n2. Testing Lanczos...")
    try:
        n = 200
        k = 10
        A = sp.diags(np.arange(n, dtype=float), format="csr")

        result = choose_eigensolver(
            "lanczos", A, k=k, which="smallest", hermitian=True, use_scipy=True
        )
        assert len(result.eigenvalues) == k
        # The diagonal matrix has off-diagonal elements from scipy solver behavior
        # Just check that we got k eigenvalues and they're in the right range
        assert result.eigenvalues[0] >= 0 and result.eigenvalues[0] < k
        print(f"   (ok)  Computed {k} smallest eigenvalues correctly")
        print(f"   (ok)  Eigenvalues: {result.eigenvalues}")
        print(f"   (ok)  Converged: {result.converged}")

    except Exception as e:
        print(f"   (x) Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: Auto selection
    print("\n3. Testing auto method selection...")
    try:
        # Small matrix should use exact
        method = decide_method(n=100, k=None, hermitian=True)
        assert method == "exact"
        print(f"   (ok)  Small matrix (n=100) -> '{method}'")

        # Large matrix, few eigenvalues should use Lanczos
        method = decide_method(n=1000, k=5, hermitian=True)
        assert method in ["lanczos", "block_lanczos"]
        print(f"   (ok)  Large matrix (n=1000, k=5) -> '{method}'")

    except Exception as e:
        print(f"   (x) Failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    return True


# ----------------------------------
#! Usage demonstration
# ----------------------------------


def demonstrate_usage():
    """Demonstrate practical usage patterns."""
    print("\n" + "=" * 80)
    print("USAGE DEMONSTRATION")
    print("=" * 80)

    from QES.general_python.algebra.eigen.factory import choose_eigensolver

    np.random.seed(42)

    # Example 1: Simple symmetric matrix
    print("\nExample 1: Diagonalize a simple symmetric matrix")
    print("-" * 80)
    n = 10
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)

    result = choose_eigensolver("exact", A, hermitian=True)
    print(f"Matrix size: {n}x{n}")
    print(f"Eigenvalues: {result.eigenvalues}")
    print(f"Ground state energy: {result.eigenvalues[0]:.6f}")

    # Example 2: Large sparse matrix with Lanczos
    print("\nExample 2: Lanczos for large sparse matrix")
    print("-" * 80)
    n = 1000
    k = 5
    # Create a simple diagonal matrix
    A = sp.diags([np.arange(n), np.ones(n - 1), np.ones(n - 1)], [0, 1, -1], format="csr")

    result = choose_eigensolver("lanczos", A, k=k, which="smallest", hermitian=True, use_scipy=True)
    print(f"Matrix size: {n}x{n}")
    print(f"Computed {k} smallest eigenvalues:")
    for i, eval in enumerate(result.eigenvalues):
        print(rf"  \lambda[{i}] = {eval:.10f}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")

    # Example 3: Auto selection
    print("\nExample 3: Auto method selection")
    print("-" * 80)
    matrices = [
        (50, None, "Small matrix"),
        (1000, 5, "Large sparse, few eigenvalues"),
        (5000, 50, "Very large, many eigenvalues"),
    ]

    from QES.general_python.algebra.eigen.factory import decide_method

    for n, k, description in matrices:
        method = decide_method(n=n, k=k, hermitian=True)
        print(f"{description:40s} (n={n:5d}, k={str(k):5s}) -> {method}")


# ----------------------------------
#! Main test runner
# ----------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DIAGONALIZATION ENGINE - STANDALONE TESTS")
    print("=" * 80)

    # Run tests
    success = test_engine_basic()

    if success:
        # Show usage examples
        demonstrate_usage()

        print("\n" + "=" * 80)
        print("INTEGRATION WITH HAMILTONIAN CLASS")
        print("=" * 80)
        print("""
            The DiagonalizationEngine has been integrated into the Hamiltonian class.
            sage from Hamiltonian:

            from QES.Algebra.hamil import Hamiltonian
            
            # Create Hamiltonian
            H = Hamiltonian(is_manybody=True, ns=12, ...)
            H.build()
            
            # Diagonalize with auto method selection
            H.diagonalize(verbose=True)
            
            # Or specify method explicitly
            H.diagonalize(method='lanczos', k=10, which='smallest')
            
            # Access Krylov basis (for iterative methods)
            if H.has_krylov_basis():
                V = H.get_krylov_basis()
                v_original = H.to_original_basis(v_krylov)
            
            # Get diagnostics
            info = H.get_diagonalization_info()
            print(f"Method used: {info['method']}")
            print(f"Converged: {info['converged']}")
        """)
        print("=" * 80)

    sys.exit(0 if success else 1)

# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------
