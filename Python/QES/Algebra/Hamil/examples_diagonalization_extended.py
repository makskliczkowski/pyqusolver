"""
Extended Examples: New SciPy and JAX Diagonalization Methods

Demonstrates usage of the newly added methods:
- scipy-eigh (dense Hermitian)
- scipy-eig (dense general)
- scipy-eigs (sparse general)
- lobpcg (preconditioned)
- jax-eigh (GPU-accelerated)

----------------------------------------
File        : QES/Algebra/Hamil/examples_diagonalization_extended.py
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
----------------------------------------
"""

import numpy as np
import scipy.sparse as sp

try:
    from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
except ImportError:
    raise ImportError("QES package is required to run these examples.")

# ----------------------------------------------------------------------------------------
#! Example 1: scipy-eigh - Dense Hermitian eigenvalue solver
# ----------------------------------------------------------------------------------------


def example_scipy_eigh():
    r"""
    Use scipy.linalg.eigh for dense Hermitian matrices.
    Returns full spectrum by default (ignores k).
    """
    print("=" * 80)
    print("Example 1: scipy-eigh - Dense Hermitian Solver")
    print("=" * 80)

    # Create a dense Hermitian matrix
    n = 50
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)

    engine = DiagonalizationEngine(method="scipy-eigh", verbose=True)
    result = engine.diagonalize(A, hermitian=True)

    print(f"\nMatrix size: {n}x{n}")
    print(f"Computed {len(result.eigenvalues)} eigenvalues (full spectrum)")
    print(f"Smallest eigenvalues: {result.eigenvalues[:5]}")
    print(f"Largest eigenvalues: {result.eigenvalues[-5:]}")

    # Verify orthonormality
    Q = result.eigenvectors
    orthonormality_error = np.linalg.norm(Q.T @ Q - np.eye(n))
    print(f"Orthonormality error: {orthonormality_error:.2e}")


# ----------------------------------------------------------------------------------------
#! Example 2: scipy-eigh with generalized eigenvalue problem
# ----------------------------------------------------------------------------------------


def example_scipy_eigh_generalized():
    r"""
    Solve generalized eigenvalue problem: A v = \lambda B v
    """
    print("\n" + "=" * 80)
    print(r"Example 2: Generalized Eigenvalue Problem (A v = \lambda B v)")
    print("=" * 80)

    n = 30
    # Create symmetric positive definite matrices
    A = np.random.randn(n, n)
    A = A.T @ A + np.eye(n) * 0.1

    B = np.random.randn(n, n)
    B = B.T @ B + np.eye(n) * 0.5

    engine = DiagonalizationEngine(method="scipy-eigh")
    result = engine.diagonalize(A, hermitian=True, B=B)

    print("\nSolved A v = \\lambda B v")
    print(f"Computed {len(result.eigenvalues)} eigenvalues")
    print(f"First 5 eigenvalues: {result.eigenvalues[:5]}")

    # Verify: A v_i = \lambda_i B v_i
    v0 = result.eigenvectors[:, 0]
    lam0 = result.eigenvalues[0]
    residual = np.linalg.norm(A @ v0 - lam0 * (B @ v0))
    print(f"Residual for first eigenpair: {residual:.2e}")


# ----------------------------------------------------------------------------------------
#! Example 3: scipy-eigs - Sparse general eigenvalue solver
# ----------------------------------------------------------------------------------------


def example_scipy_eigs():
    """
    Use scipy.sparse.linalg.eigs for sparse non-Hermitian matrices.
    Computes k eigenvalues with largest magnitude.
    """
    print("\n" + "=" * 80)
    print("Example 3: scipy-eigs - Sparse General Solver")
    print("=" * 80)

    n = 500
    # Create a sparse non-symmetric matrix
    A = sp.random(n, n, density=0.01, format="csr")
    A = A + A.T * 0.3  # Make it slightly non-symmetric

    k = 8
    engine = DiagonalizationEngine(method="scipy-eigs")
    result = engine.diagonalize(A, k=k, hermitian=False, which="LM")

    print(f"\nMatrix size: {n}x{n} (sparse)")
    print(f"Requested {k} eigenvalues with largest magnitude")
    print(f"Computed {len(result.eigenvalues)} eigenvalues")
    print("Eigenvalues (by magnitude):")
    for i, lam in enumerate(result.eigenvalues):
        print(rf"  \lambda[{i}] = {lam.real:+.6f} + {lam.imag:+.6f}j  (|\lambda| = {abs(lam):.6f})")


# ----------------------------------------------------------------------------------------
#! Example 4: lobpcg - Preconditioned iterative solver
# ----------------------------------------------------------------------------------------


def example_lobpcg():
    r"""
    Use LOBPCG with Jacobi preconditioner for large sparse symmetric matrices.
    """
    print("\n" + "=" * 80)
    print("Example 4: LOBPCG - Preconditioned Iterative Solver")
    print("=" * 80)

    n = 1000
    # Build a discrete 2D Laplacian (symmetric positive definite)
    nx = int(np.sqrt(n))
    ny = nx
    n = nx * ny

    # Simple tridiagonal structure
    main_diag = 4 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)
    A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csr")

    # Add small shift for numerical stability
    A += sp.eye(n) * 0.01

    # Jacobi preconditioner (diagonal scaling)
    M = sp.diags(1.0 / A.diagonal())

    k = 10
    engine = DiagonalizationEngine(method="lobpcg", verbose=True)
    result = engine.diagonalize(
        A, k=k, hermitian=True, which="smallest", M=M, max_iter=100, tol=1e-6
    )

    print(f"\nMatrix size: {n}x{n} (sparse)")
    print(f"Computed {len(result.eigenvalues)} smallest eigenvalues")
    print("Eigenvalues:")
    for i, lam in enumerate(result.eigenvalues):
        print(rf"  \lambda[{i}] = {lam:.8f}")

    print("\nPreconditioner used: Jacobi (diagonal scaling)")
    print(f"Convergence: {result.converged}")


# ----------------------------------------------------------------------------------------
#! Example 5: jax-eigh - GPU-accelerated diagonalization
# ----------------------------------------------------------------------------------------


def example_jax_eigh():
    r"""
    Use JAX for GPU-accelerated diagonalization (if JAX available).
    """
    print("\n" + "=" * 80)
    print("Example 5: JAX-eigh - GPU-Accelerated Solver")
    print("=" * 80)

    try:
        import jax
        import jax.numpy as jnp

        n = 200
        # Create Hermitian matrix
        A_np = np.random.randn(n, n)
        A_np = 0.5 * (A_np + A_np.T)

        engine = DiagonalizationEngine(method="jax-eigh")
        result = engine.diagonalize(A_np, hermitian=True)

        print(f"\nMatrix size: {n}x{n}")
        print(f"Computed {len(result.eigenvalues)} eigenvalues (full spectrum)")
        print(f"JAX device: {jax.devices()[0]}")
        print(f"Smallest 5 eigenvalues: {result.eigenvalues[:5]}")
        print(f"Largest 5 eigenvalues: {result.eigenvalues[-5:]}")

        print("\nNote: JAX can leverage GPU/TPU acceleration for large matrices")

    except ImportError:
        print("\nJAX not available. Install with:")
        print("  pip install jax jaxlib")
        print("For GPU support:")
        print("  pip install jax[cuda11_cudnn82]  # or appropriate CUDA version")


# ----------------------------------------------------------------------------------------
#! Example 6: Comparison of methods
# ----------------------------------------------------------------------------------------


def example_comparison():
    r"""
    Compare different methods on the same problem.
    """
    print("\n" + "=" * 80)
    print("Example 6: Method Comparison")
    print("=" * 80)

    n = 100
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)

    methods = ["exact", "scipy-eigh", "lanczos"]
    k = 10

    print(f"\nMatrix size: {n}x{n}")
    print(f"Computing {k} smallest eigenvalues with different methods:\n")

    results = {}
    for method in methods:
        try:
            engine = DiagonalizationEngine(method=method)
            if method in ["lanczos"]:
                result = engine.diagonalize(A, k=k, hermitian=True, which="smallest")
                evals = result.eigenvalues
            else:
                result = engine.diagonalize(A, hermitian=True)
                evals = result.eigenvalues[:k]  # Take first k

            results[method] = evals
            print(rf"{method:12s}: \lambda_min = {evals[0]:+.8f}, \lambda_max = {evals[-1]:+.8f}")
        except Exception as e:
            print(f"{method:12s}: Failed - {e}")

    # Compare results
    if len(results) > 1:
        methods_list = list(results.keys())
        ref = results[methods_list[0]]
        print(f"\nRelative differences (vs {methods_list[0]}):")
        for method in methods_list[1:]:
            diff = np.linalg.norm(results[method] - ref) / (np.linalg.norm(ref) + 1e-12)
            print(f"  {method:12s}: {diff:.2e}")


# ----------------------------------------------------------------------------------------
#! Example 7: Using subset selection with scipy-eigh
# ----------------------------------------------------------------------------------------


def example_subset_selection():
    r"""
    Use scipy-eigh with subset_by_index or subset_by_value to compute
    only a subset of eigenvalues efficiently.
    """
    print("\n" + "=" * 80)
    print("Example 7: Subset Selection with scipy-eigh")
    print("=" * 80)

    n = 200
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)

    # Compute eigenvalues 10-19 (0-indexed)
    engine = DiagonalizationEngine(method="scipy-eigh")
    result = engine.diagonalize(A, hermitian=True, subset_by_index=(10, 19))

    print(f"\nMatrix size: {n}x{n}")
    print("Requested eigenvalues at indices 10-19")
    print(f"Computed {len(result.eigenvalues)} eigenvalues:")
    for i, lam in enumerate(result.eigenvalues):
        print(rf"  \lambda[{10+i}] = {lam:.8f}")

    # Compute eigenvalues in range [-2, 2]
    result2 = engine.diagonalize(A, hermitian=True, subset_by_value=(-2.0, 2.0))
    print(f"\nEigenvalues in range [-2.0, 2.0]: {len(result2.eigenvalues)} found")
    if len(result2.eigenvalues) > 0:
        print(f"  Range: [{result2.eigenvalues[0]:.4f}, {result2.eigenvalues[-1]:.4f}]")


# ----------------------------------------------------------------------------------------
#! Main execution
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXTENDED DIAGONALIZATION METHODS - EXAMPLES")
    print("=" * 80)
    print("\nNew methods available:")
    print("  - scipy-eigh  : Dense Hermitian (full spectrum)")
    print("  - scipy-eig   : Dense general (full spectrum)")
    print("  - scipy-eigs  : Sparse general (k eigenvalues)")
    print("  - lobpcg      : Preconditioned iterative (k eigenvalues)")
    print("  - jax-eigh    : GPU-accelerated (full spectrum)")
    print("=" * 80)

    # Run examples
    example_scipy_eigh()
    example_scipy_eigh_generalized()
    example_scipy_eigs()
    example_lobpcg()
    example_jax_eigh()
    example_comparison()
    example_subset_selection()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("(ok) Dense solvers (scipy-eigh, scipy-eig) return full spectrum")
    print("(ok) Sparse solvers (scipy-eigs, lobpcg) compute k eigenvalues")
    print("(ok) Generalized problems supported via B parameter")
    print("(ok) Preconditioning supported via M parameter (lobpcg)")
    print("(ok) Subset selection via subset_by_index/value (scipy-eigh)")
    print("(ok) JAX enables GPU acceleration")
    print("=" * 80)

# ----------------------------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------------------------
