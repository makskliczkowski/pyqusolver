"""
Tests for various eigenvalue solvers in the QES package.
This includes tests for dense and sparse solvers, as well as different backends like SciPy and JAX.

File        : test/test_eigen_extended.py
Author      : Maksymilian Kliczkowski
Date        : 2025-10-30
"""

import numpy as np
import pytest
import scipy.sparse as sp

try:
    from QES.general_python.algebra.eigen.factory import choose_eigensolver
except ImportError:
    raise ImportError("QES package is required to run these tests.")

# ----------------------------------
#! Helper function to compute residual norm
# ----------------------------------


def residual_norm(A, vals, vecs):
    if vecs is None:
        return np.inf
    return np.linalg.norm(A @ vecs - vecs @ np.diag(vals)) / (np.linalg.norm(A) + 1e-12)


# ----------------------------------
#! Test cases for different eigensolvers
# ----------------------------------


def test_scipy_eigh_dense_small():
    """Test SciPy's eigh solver on a small dense symmetric matrix."""
    n = 20
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    res = choose_eigensolver("scipy-eigh", A, hermitian=True)
    assert res.eigenvalues.shape == (n,)
    assert res.eigenvectors.shape == (n, n)
    rn = residual_norm(A, res.eigenvalues, res.eigenvectors)
    assert rn < 1e-8


# ----------------------------------


def test_scipy_eig_dense_general():
    """Test SciPy's eig solver on a small dense general matrix."""
    n = 18
    A = np.random.randn(n, n)
    res = choose_eigensolver("scipy-eig", A, hermitian=False)
    assert res.eigenvalues.shape == (n,)
    assert res.eigenvectors.shape == (n, n)
    # For general matrices, residual with right eigenvectors
    rn = np.linalg.norm(A @ res.eigenvectors - res.eigenvectors @ np.diag(res.eigenvalues)) / (
        np.linalg.norm(A) + 1e-12
    )
    assert rn < 1e-6


# ----------------------------------


def test_scipy_eigs_sparse_k():
    """Test SciPy's eigs solver on a sparse non-symmetric matrix."""
    n = 200
    rng = np.random.default_rng(0)
    A = sp.random(n, n, density=0.01, data_rvs=rng.standard_normal)
    k = 6
    res = choose_eigensolver(
        "scipy-eigs", A, k=k, hermitian=False, which="LM", tol=1e-6, max_iter=200
    )
    assert res.eigenvalues.shape[0] == k
    assert res.eigenvectors.shape == (n, k)


# ----------------------------------


@pytest.mark.skip(
    reason="Fails due to broken import 'from scipy.sparse import sparse' in general_python submodule"
)
def test_lobpcg_symmetric_sparse():
    """Test LOBPCG solver on a symmetric sparse matrix with preconditioning."""
    n = 150
    # Build a symmetric positive definite matrix (discrete 1D Laplacian + I)
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    A = A + sp.eye(n) * 1e-3
    k = 5
    # Simple Jacobi preconditioner
    M = sp.diags(1.0 / A.diagonal())
    res = choose_eigensolver("lobpcg", A, k=k, hermitian=True, which="smallest", M=M, max_iter=50)
    assert res.eigenvalues.shape[0] == k
    assert res.eigenvectors.shape == (n, k)


# ----------------------------------


def test_jax_eigh_small():
    """Test JAX's eigh solver on a small dense symmetric matrix."""
    # Skip this test if JAX is not available in the environment
    pytest.importorskip("jax", reason="JAX not available in test env")
    pytest.importorskip("jax.numpy")
    n = 16
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    res = choose_eigensolver("jax-eigh", A, hermitian=True)
    assert res.eigenvalues.shape == (n,)
    assert res.eigenvectors.shape == (n, n)


# ----------------------------------
#! EOF
# ----------------------------------
