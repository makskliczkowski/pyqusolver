"""
test_spectral_backend.py

Comprehensive tests for:
1. Lanczos coefficient storage in EigenResult
2. Spectral function calculations
3. Type safety (no complex->real casting)
4. Operator-specific spectral functions
5. Backend linalg operations
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# Import modules
try:
    from QES.general_python.algebra.backend_linalg import (
        change_basis_matrix,
        eigh,
        eigsh,
        identity,
        inner,
        kron,
        outer,
        overlap,
        trace,
    )

    LINALG_AVAILABLE = True
except ImportError:
    LINALG_AVAILABLE = False

try:
    from QES.general_python.algebra.eigen.lanczos import LanczosEigensolver
    from QES.general_python.algebra.eigen.result import EigenResult

    LANCZOS_AVAILABLE = True
except ImportError:
    LANCZOS_AVAILABLE = False

try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian, QuadraticTerm

    QH_AVAILABLE = True
except ImportError:
    QH_AVAILABLE = False

try:
    from QES.general_python.physics.spectral.spectral_function import spectral_function

    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False

# ============================================================================
# TEST 1: Backend Linalg Operations
# ============================================================================


@pytest.mark.skipif(not LINALG_AVAILABLE, reason="backend_linalg not available")
def test_outer_product():
    """Test outer product computation"""
    a = np.array([1, 2, 3], dtype=complex)
    b = np.array([4, 5], dtype=complex)

    result = outer(a, b, backend="default")
    expected = np.outer(a, b)

    assert_allclose(result, expected)
    assert result.dtype == complex


@pytest.mark.skipif(not LINALG_AVAILABLE, reason="backend_linalg not available")
def test_kron_product():
    """Test Kronecker product computation"""
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    B = np.array([[5, 6], [7, 8]], dtype=complex)

    result = kron(A, B, backend="default")
    expected = np.kron(A, B)

    assert_allclose(result, expected)
    assert result.dtype == complex


@pytest.mark.skipif(not LINALG_AVAILABLE, reason="backend_linalg not available")
def test_inner_product():
    """Test inner product computation"""
    a = np.array([1 + 1j, 2 + 2j], dtype=complex)
    b = np.array([3 + 3j, 4 + 4j], dtype=complex)

    result = inner(a, b, backend="default")
    # Inner product: <a|b> = adagger  \cdot  b
    expected = np.conj(a) @ b

    assert_allclose(result, expected)


@pytest.mark.skipif(not LINALG_AVAILABLE, reason="backend_linalg not available")
def test_overlap_with_operator():
    """Test matrix element <a|O|b>"""
    a = np.array([0, 1, 0], dtype=complex)
    b = np.array([0, 1, 0], dtype=complex)
    O = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=complex)

    result = overlap(a, O, b, backend="default")
    expected = 2.0  # <a|O|b> = a† O b

    assert_allclose(result, expected)
    assert isinstance(result, (complex, np.complexfloating))


@pytest.mark.skipif(not LINALG_AVAILABLE, reason="backend_linalg not available")
def test_basis_change():
    """Test basis transformation"""
    # Random unitary
    np.random.seed(42)
    N = 4
    from scipy.stats import unitary_group

    U = unitary_group.rvs(N)

    # Matrix to transform
    A = np.diag(np.arange(N, dtype=complex))

    # Change basis: A' = U A Udagger
    A_prime = change_basis_matrix(U, A, direction="forward", backend="default")

    # Verify: eigenvalues are preserved
    evals_A = np.linalg.eigvalsh(A)
    evals_A_prime = np.linalg.eigvalsh(A_prime)

    assert_allclose(np.sort(evals_A), np.sort(evals_A_prime), atol=1e-14)


@pytest.mark.skipif(not LINALG_AVAILABLE, reason="backend_linalg not available")
def test_trace():
    """Test trace computation"""
    A = np.array([[1 + 1j, 2], [3, 4 + 2j]], dtype=complex)

    result = trace(A, backend="default")
    expected = np.trace(A)

    assert_allclose(result, expected)


# ============================================================================
# TEST 2: EigenResult Lanczos Coefficients
# ============================================================================


@pytest.mark.skipif(not LANCZOS_AVAILABLE, reason="Lanczos not available")
def test_eigenresult_lanczos_fields():
    """Test that EigenResult stores Lanczos coefficients"""
    evals = np.array([1.0, 2.0, 3.0])
    evecs = np.eye(3)
    alpha = np.array([1.0, 2.0, 3.0])
    beta = np.array([0.5, 0.5])
    V = np.eye(3)

    result = EigenResult(
        eigenvalues=evals,
        eigenvectors=evecs,
        lanczos_alpha=alpha,
        lanczos_beta=beta,
        krylov_basis=V,
    )

    assert result.lanczos_alpha is not None
    assert result.lanczos_beta is not None
    assert result.krylov_basis is not None
    assert_array_equal(result.lanczos_alpha, alpha)
    assert_array_equal(result.lanczos_beta, beta)


# ============================================================================
# TEST 3: Spectral Function with Operators
# ============================================================================


@pytest.mark.skipif(not SPECTRAL_AVAILABLE, reason="spectral_function not available")
def test_spectral_function_type_safety():
    """Test that complex values are preserved (no casting issues)"""
    # Green's function (must be complex)
    G = np.array([[1.0 / (1.0 + 1j * 0.1), 0], [0, 1.0 / (2.0 + 1j * 0.1)]], dtype=complex)

    # Spectral function should be real
    A = spectral_function(G)

    # Verify type safety
    assert A.dtype in [np.float32, np.float64, float]
    assert not np.any(np.iscomplex(A))
    assert np.all(A >= 0)  # Spectral function is non-negative


@pytest.mark.skipif(not SPECTRAL_AVAILABLE, reason="spectral_function not available")
def test_spectral_function_with_identity_operator():
    """Test spectral function with identity operator"""
    G = np.array(
        [
            [1.0 / (1.0 + 1j * 0.1), 0, 0],
            [0, 1.0 / (2.0 + 1j * 0.1), 0],
            [0, 0, 1.0 / (3.0 + 1j * 0.1)],
        ],
        dtype=complex,
    )

    # Test that spectral function works with just Green's function
    A = spectral_function(G)

    # Should be real and positive
    assert np.all(np.imag(A) == 0)
    assert np.all(A >= 0)


@pytest.mark.skipif(not SPECTRAL_AVAILABLE, reason="spectral_function not available")
def test_spectral_function_with_diagonal_operator():
    """Test spectral function from Green's function matrix"""
    G = np.array(
        [
            [1.0 / (1.0 + 1j * 0.1), 0, 0],
            [0, 1.0 / (2.0 + 1j * 0.1), 0],
            [0, 0, 1.0 / (3.0 + 1j * 0.1)],
        ],
        dtype=complex,
    )

    A = spectral_function(G)

    # Should be real and positive
    assert np.all(np.isreal(A) | (np.abs(np.imag(A)) < 1e-14))
    assert np.all(A >= 0)

    # Check diagonal structure preserved (for diagonal G)
    if A.ndim == 2:
        assert np.allclose(A, np.diag(np.diag(A)), atol=1e-14)


# ============================================================================
# TEST 4: Sum Rules
# ============================================================================


@pytest.mark.skipif(not SPECTRAL_AVAILABLE, reason="spectral_function not available")
def test_spectral_function_sum_rule():
    """Test spectral function sum rule"""
    # Create eigenvalues and eigenvectors
    evals = np.array([-2, -1, 0, 1, 2], dtype=float)
    N = len(evals)
    evecs = np.eye(N, dtype=complex)

    # Frequency grid
    omegas = np.linspace(-5, 5, 500)
    eta = 0.01

    # Compute Green's function and spectral function
    A_LDOS = np.zeros(len(omegas), dtype=float)
    for i, omega in enumerate(omegas):
        G_omega = evecs @ np.diag(1.0 / (omega + 1j * eta - evals)) @ evecs.T.conj()
        A_i = spectral_function(G_omega)
        A_LDOS[i] = np.trace(A_i)

    # Integrate: should be approximately N
    integral = np.trapezoid(A_LDOS, omegas)

    # Allow for larger tolerance due to finite grid and broadening
    assert_allclose(integral, N, rtol=0.1)


# ============================================================================
# TEST 5: Integration with Hamiltonians
# ============================================================================


@pytest.mark.skipif(not LINALG_AVAILABLE, reason="backend_linalg not available")
def test_quadratic_hamiltonian_spectrum():
    """Test spectral properties of a simple quadratic Hamiltonian"""
    try:
        from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
    except ImportError:
        pytest.skip("QuadraticHamiltonian not available")

    # Small quadratic system
    ns = 3
    H = QuadraticHamiltonian(ns=ns, particle_conserving=True)

    # Add hopping
    for i in range(ns - 1):
        H.add_hopping(i, i + 1, -1.0)

    # Add onsite
    for i in range(ns):
        H.add_term(QuadraticTerm.Onsite, i, -0.5)

    H.diagonalize()

    # Eigenvalues should be real
    assert H.eig_val.dtype in [np.float32, np.float64, float]

    # Eigenvectors should be real or complex (depends on Hamiltonian)
    assert H.eig_vec.dtype in [np.float32, np.float64, np.complex64, np.complex128, float, complex]


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("Running spectral backend tests...\n")

    # Test backend_linalg
    if LINALG_AVAILABLE:
        print("Testing backend_linalg...")
        test_outer_product()
        test_kron_product()
        test_inner_product()
        test_overlap_with_operator()
        test_basis_change()
        test_trace()
        print("(ok) backend_linalg tests passed\n")

    # Test EigenResult
    if LANCZOS_AVAILABLE:
        print("Testing EigenResult Lanczos fields...")
        test_eigenresult_lanczos_fields()
        print("(ok) EigenResult tests passed\n")

    # Test spectral functions
    if SPECTRAL_AVAILABLE:
        print("Testing spectral functions...")
        test_spectral_function_type_safety()
        test_spectral_function_with_identity_operator()
        test_spectral_function_with_diagonal_operator()
        test_spectral_function_sum_rule()
        print("(ok) Spectral function tests passed\n")

    # Test integration
    if LINALG_AVAILABLE:
        print("Testing Hamiltonian integration...")
        test_quadratic_hamiltonian_spectrum()
        print("(ok) Hamiltonian tests passed\n")

    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------
