r"""
Examples and demonstrations of the modular diagonalization system for Hamiltonians.

This file demonstrates how to use the different diagonalization methods available
in the Hamiltonian class, including basis transformations and method selection.

File        : QES/Algebra/Hamil/examples_diagonalization.py
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

import numpy as np

# Minimal runnable example using the DiagonalizationEngine without requiring a full Hamiltonian
try:
    from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
except Exception:
    # Fallback for relative import if running file directly
    from .hamil_diag_engine import DiagonalizationEngine

# Assuming QES imports work
# from QES.Algebra.hamil import Hamiltonian
# from QES.Algebra.hilbert import HilbertSpace

# ----------------------------------------------------------------------------------------
#! Example 1: Auto-selection of diagonalization method
# ----------------------------------------------------------------------------------------


def example_auto_selection():
    """
    Demonstrate automatic method selection based on matrix size.
    """
    print("=" * 80)
    print("Example 1: Automatic Method Selection")
    print("=" * 80)

    # Small system (n < 500) -> Exact diagonalization
    print("\n1a. Small system (n=100):")
    # H_small = Hamiltonian(is_manybody=True, ns=10, ...)
    # H_small.build()
    # H_small.diagonalize(method='auto', verbose=True)
    # info = H_small.get_diagonalization_info()
    # print(f"Method used: {info['method']}")  # Expected: 'exact'

    # Large sparse system -> Lanczos
    print("\n1b. Large sparse system (n=1000):")
    # H_large = Hamiltonian(is_manybody=True, ns=20, ...)
    # H_large.build()
    # H_large.diagonalize(method='auto', k=10, verbose=True)
    # info = H_large.get_diagonalization_info()
    # print(f"Method used: {info['method']}")  # Expected: 'lanczos' or 'block_lanczos'


# ----------------------------------------------------------------------------------------
#! Example 2: Exact diagonalization
# ----------------------------------------------------------------------------------------


def example_exact_diagonalization():
    """
    Full diagonalization for complete spectrum.
    """
    print("\n" + "=" * 80)
    print("Example 2: Exact Diagonalization")
    print("=" * 80)

    # Create small Hamiltonian
    # H = Hamiltonian(is_manybody=True, ns=8, ...)
    # H.build()

    # Exact diagonalization (all eigenvalues)
    # H.diagonalize(method='exact', verbose=True)

    # Access results
    # evals = H.get_eigval()
    # evecs = H.get_eigvec()
    # print(f"\nComputed {len(evals)} eigenvalues")
    # print(f"Ground state energy: {evals[0]:.10f}")
    # print(f"Energy gap: {evals[1] - evals[0]:.10f}")

    # No Krylov basis for exact diagonalization
    # print(f"Has Krylov basis: {H.has_krylov_basis()}")  # False


# ----------------------------------------------------------------------------------------
#! Example 3: Lanczos iteration
# ----------------------------------------------------------------------------------------


def example_lanczos():
    """
    Lanczos for extremal eigenvalues of large sparse symmetric matrices.
    """
    print("\n" + "=" * 80)
    print("Example 3: Lanczos Iteration")
    print("=" * 80)

    # Create large sparse Hamiltonian
    # H = Hamiltonian(is_manybody=True, ns=16, is_sparse=True, ...)
    # H.build()

    # Lanczos for 10 smallest eigenvalues
    # H.diagonalize(
    #     method='lanczos',
    #     k=10,
    #     which='smallest',
    #     tol=1e-10,
    #     store_basis=True,
    #     verbose=True
    # )

    # Get results
    # evals = H.get_eigval()
    # print(f"\n10 smallest eigenvalues: {evals}")

    # Check convergence
    # info = H.get_diagonalization_info()
    # print(f"Converged: {info['converged']}")
    # print(f"Iterations: {info['iterations']}")

    # Krylov basis is available
    # if H.has_krylov_basis():
    #     V = H.get_krylov_basis()
    #     print(f"Krylov basis shape: {V.shape}")  # (n, k) or larger


# ----------------------------------------------------------------------------------------
#! Example 4: Block Lanczos
# ----------------------------------------------------------------------------------------


def example_block_lanczos():
    """
    Block Lanczos for multiple degenerate or clustered eigenvalues.
    """
    print("\n" + "=" * 80)
    print("Example 4: Block Lanczos")
    print("=" * 80)

    # Create Hamiltonian with degenerate eigenvalues
    # H = Hamiltonian(is_manybody=True, ns=14, ...)
    # H.build()

    # Block Lanczos with block size 5
    # H.diagonalize(
    #     method='block_lanczos',
    #     k=20,
    #     block_size=5,
    #     which='smallest',
    #     tol=1e-10,
    #     reorthogonalize=True,
    #     verbose=True
    # )

    # Check for degeneracies
    # evals = H.get_eigval()
    # print(f"\nFirst 20 eigenvalues:")
    # for i, e in enumerate(evals):
    #     print(f"  E[{i}] = {e:.10f}")


# ----------------------------------------------------------------------------------------
#! Example 5: Basis transformations
# ----------------------------------------------------------------------------------------


def example_basis_transformations():
    """
    Demonstrate transformations between Krylov and original basis.
    """
    print("\n" + "=" * 80)
    print("Example 5: Basis Transformations")
    print("=" * 80)

    # Create and diagonalize with Lanczos
    # H = Hamiltonian(is_manybody=True, ns=12, ...)
    # H.build()
    # H.diagonalize(method='lanczos', k=10, store_basis=True)

    # Example: Transform Ritz vector to original basis
    # ritz_vec = np.zeros(10)
    # ritz_vec[0] = 1.0  # First Ritz vector (ground state in Krylov basis)
    # state_original = H.to_original_basis(ritz_vec)
    # print(f"Original space vector shape: {state_original.shape}")

    # Example: Project arbitrary state onto Krylov subspace
    # arbitrary_state = np.random.randn(H.nh)
    # arbitrary_state /= np.linalg.norm(arbitrary_state)
    # krylov_coeffs = H.to_krylov_basis(arbitrary_state)
    # print(f"Krylov coefficients shape: {krylov_coeffs.shape}")

    # Verify reconstruction
    # reconstructed = H.to_original_basis(krylov_coeffs)
    # projection_quality = np.linalg.norm(reconstructed - arbitrary_state)
    # print(f"Projection residual: {projection_quality:.10e}")


# ----------------------------------------------------------------------------------------
#! Example 6: Comparing methods
# ----------------------------------------------------------------------------------------


def example_compare_methods():
    """
    Compare different methods on the same Hamiltonian.
    """
    print("\n" + "=" * 80)
    print("Example 6: Comparing Methods")
    print("=" * 80)

    # Create Hamiltonian
    # H = Hamiltonian(is_manybody=True, ns=10, ...)
    # H.build()

    # Method 1: Exact
    # import time
    # t0 = time.perf_counter()
    # H.diagonalize(method='exact')
    # t_exact = time.perf_counter() - t0
    # evals_exact = H.get_eigval()[:10]
    # print(f"Exact: {t_exact:.4f}s, E0 = {evals_exact[0]:.10f}")

    # Method 2: Lanczos
    # t0 = time.perf_counter()
    # H.diagonalize(method='lanczos', k=10, which='smallest')
    # t_lanczos = time.perf_counter() - t0
    # evals_lanczos = H.get_eigval()
    # print(f"Lanczos: {t_lanczos:.4f}s, E0 = {evals_lanczos[0]:.10f}")

    # Compare accuracy
    # error = np.abs(evals_exact - evals_lanczos)
    # print(f"Max error: {np.max(error):.10e}")


# ----------------------------------------------------------------------------------------
#! Example 7: Using with matrix-vector product
# ----------------------------------------------------------------------------------------


def example_matvec():
    """
    Diagonalize using only matrix-vector product (no explicit matrix).
    """
    print("\n" + "=" * 80)
    print("Example 7: Matrix-Vector Product Interface")
    print("=" * 80)

    # Define matrix-vector product function
    # def matvec(v):
    #     # Custom implementation of H @ v
    #     # Could use sparse operations, operator-based methods, etc.
    #     return H_implicit @ v

    # Create Hamiltonian without building full matrix
    # H = Hamiltonian(is_manybody=True, ns=12, ...)
    # # Don't call H.build()

    # Note: Current implementation requires H._hamil to be set
    # Future enhancement could support matvec-only mode

    print("Note: Matvec-only mode requires modification to diagonalize() method")
    print("      to handle case where self._hamil is None but matvec is provided")


# ----------------------------------------------------------------------------------------
#! Example 8: Backend selection
# ----------------------------------------------------------------------------------------


def example_backends():
    """
    Demonstrate using different computational backends.
    """
    print("\n" + "=" * 80)
    print("Example 8: Backend Selection")
    print("=" * 80)

    # Create Hamiltonian
    # H = Hamiltonian(is_manybody=True, ns=10, backend='numpy', ...)
    # H.build()

    # Use NumPy backend
    # H.diagonalize(method='exact', backend='numpy', verbose=True)

    # Use SciPy backend (for sparse)
    # H.diagonalize(method='lanczos', k=10, backend='scipy', use_scipy=True)

    # Use JAX backend (if available)
    # try:
    #     H.diagonalize(method='exact', backend='jax', verbose=True)
    # except ImportError:
    #     print("JAX backend not available")


# ----------------------------------------------------------------------------------------
#! Example 9: Working with quadratic Hamiltonians
# ----------------------------------------------------------------------------------------


def example_quadratic():
    """
    Diagonalization of quadratic (non-interacting) Hamiltonians.
    """
    print("\n" + "=" * 80)
    print("Example 9: Quadratic Hamiltonians")
    print("=" * 80)

    # Create quadratic Hamiltonian
    # H = Hamiltonian(is_manybody=False, ns=20, ...)
    # H.build()

    # Exact diagonalization (single-particle spectrum)
    # H.diagonalize(method='exact', verbose=True)

    # The eigenvectors represent single-particle states
    # evals = H.get_eigval()
    # evecs = H.get_eigvec()
    # print(f"Single-particle energies: {evals}")


# ----------------------------------------------------------------------------------------
#! Example 10: Advanced usage
# ----------------------------------------------------------------------------------------


def example_advanced():
    """
    Advanced features and customization.
    """
    print("\n" + "=" * 80)
    print("Example 10: Advanced Usage")
    print("=" * 80)

    # 1. Custom convergence criteria
    # H.diagonalize(
    #     method='lanczos',
    #     k=10,
    #     tol=1e-12,
    #     max_iter=500,
    #     reorthogonalize=True
    # )

    # 2. Get detailed diagnostics
    # info = H.get_diagonalization_info()
    # print(f"Method: {info['method']}")
    # print(f"Converged: {info['converged']}")
    # print(f"Iterations: {info['iterations']}")
    # if info['residual_norms'] is not None:
    #     print(f"Max residual: {np.max(info['residual_norms']):.10e}")

    # 3. Work with Krylov subspace
    # if H.has_krylov_basis():
    #     V = H.get_krylov_basis()
    #     # Build reduced Hamiltonian in Krylov basis
    #     H_reduced = V.T.conj() @ H._hamil @ V
    #     # Verify eigenvalues match
    #     evals_reduced = np.linalg.eigvalsh(H_reduced)
    #     print(f"Krylov eigenvalues match: {np.allclose(evals_reduced, H.get_eigval())}")


# ----------------------------------------------------------------------------------------
#! Main
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run all examples.

    Note: These are template examples. To run them, you need to:
    1. Import the actual Hamiltonian class
    2. Uncomment the code blocks
    3. Provide appropriate Hamiltonian parameters for your system
    """

    print("\n" + "=" * 80)
    print("MODULAR DIAGONALIZATION SYSTEM - EXAMPLES")
    print("=" * 80)

    print("\nThese examples demonstrate the modular diagonalization features.")
    print("Below is a minimal runnable demo using a synthetic symmetric matrix.")

    # --- Minimal runnable demo (synthetic matrix) ---
    n = 120
    k = 6
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    A = 0.5 * (A + A.T)  # Symmetrize

    engine = DiagonalizationEngine(method="lanczos", verbose=True)
    result = engine.diagonalize(A=A, k=k, hermitian=True, store_basis=True)

    print(
        f"\nComputed {len(result.eigenvalues)} eigenvalues (smallest). Ground state: {result.eigenvalues[0]:.6f}"
    )
    print(f"Converged: {result.converged}, iterations: {result.iterations}")
    if engine.has_krylov_basis():
        V = engine.get_krylov_basis()
        print(f"Krylov basis available: V shape = {V.shape}")
        # Transform first Ritz basis vector (unit vector in Krylov space) to original basis
        e1 = np.zeros(k)
        e1[0] = 1.0
        vec_original = engine.to_original_basis(e1)
        print(f"Example transform: ||vec_original|| = {np.linalg.norm(vec_original):.6f}")
    else:
        print("Krylov basis not available for this method/backend.")

    # Uncomment to run the template examples below with your Hamiltonian class wired:
    # example_auto_selection()
    # example_exact_diagonalization()
    # example_lanczos()
    # example_block_lanczos()
    # example_basis_transformations()
    # example_compare_methods()
    # example_matvec()
    # example_backends()
    # example_quadratic()
    # example_advanced()

    print("\n" + "=" * 80)
    print("Key Features Summary:")
    print("=" * 80)
    print("(ok)  Automatic method selection based on matrix properties")
    print("(ok)  Multiple diagonalization methods: exact, Lanczos, Block Lanczos, Arnoldi")
    print("(ok)  Krylov basis storage and transformation utilities")
    print("(ok)  Backend support: NumPy, SciPy, JAX")
    print("(ok)  Detailed convergence and diagnostic information")
    print("(ok)  Memory-efficient handling of large sparse matrices")
    print("(ok)  Easy-to-use API similar to standard libraries")
    print("=" * 80)

# ----------------------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------------------
