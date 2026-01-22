"""
Test backend_ops integration with utils.py infrastructure.

This test demonstrates that backend_ops.py correctly integrates with
the existing BackendManager from utils.py.

File        : Python/test/test_backend_integration.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 15.10.25
"""

import numpy as np

try:
    from QES.general_python.algebra.solvers.backend_ops import (
        BackendOps,
        default_ops,
        get_backend_ops,
        numpy_ops,
    )
    from QES.general_python.algebra.utils import JAX_AVAILABLE, Array, backend_mgr
except ImportError as e:
    raise ImportError(
        "Failed to import necessary modules for backend_ops testing. Ensure QES package is correctly installed."
    ) from e

# ------------------------------------------------------------------------------
#! TESTS
# ------------------------------------------------------------------------------


def test_default_backend_matches_manager():
    """Test that default backend matches BackendManager."""
    ops = get_backend_ops("default")
    assert ops.backend_name == backend_mgr.name
    print("(v) Default backend matches BackendManager")


# ------------------------------------------------------------------------------


def test_from_array_detection():
    """Test automatic backend detection from array type."""
    x_np = np.array([1, 2, 3])
    ops = BackendOps.from_array(x_np)
    assert ops.backend_name == "numpy"
    print("(v) from_array() correctly detects NumPy arrays")


# ------------------------------------------------------------------------------


def test_basic_operations():
    """Test basic linear algebra operations."""
    ops = get_backend_ops("numpy")

    # Array creation
    x = ops.zeros(5)
    assert len(x) == 5
    assert np.allclose(x, 0)

    y = ops.ones(3)
    assert np.allclose(y, 1)

    # Dot product
    a = ops.array([1.0, 2.0, 3.0])
    b = ops.array([4.0, 5.0, 6.0])
    dot = ops.dot(a, b)
    assert np.isclose(dot, 32.0)  # 1*4 + 2*5 + 3*6 = 32

    # Norm
    norm = ops.norm(a)
    assert np.isclose(norm, np.sqrt(14))  # sqrt(1^2 + 2^2 + 3^2)

    print("(v) Basic operations work correctly")


# ------------------------------------------------------------------------------


def test_givens_rotation():
    """Test Givens rotation computation."""
    ops = get_backend_ops("numpy")

    # Test case: orthogonalize (3, 4) -> expect (c, s, r) = (0.6, 0.8, 5)
    c, s, r = ops.sym_ortho(3.0, 4.0)

    assert np.isclose(c, 0.6)
    assert np.isclose(s, 0.8)
    assert np.isclose(r, 5.0)
    assert np.isclose(c**2 + s**2, 1.0)  # Orthonormality

    # Verify rotation zeros out second component
    h1_new, h2_new = ops.apply_givens_rotation(3.0, 4.0, c, s)
    assert np.isclose(h1_new, 5.0)
    assert np.isclose(h2_new, 0.0)

    print("(v) Givens rotation works correctly")


# ------------------------------------------------------------------------------


def test_global_instances():
    """Test pre-created global instances."""
    assert numpy_ops.backend_name == "numpy"
    assert default_ops.backend_name == backend_mgr.name

    if JAX_AVAILABLE:
        from QES.general_python.algebra.solvers.backend_ops import jax_ops

        assert jax_ops.backend_name == "jax"

    print("(v) Global instances initialized correctly")


# ------------------------------------------------------------------------------


def test_backend_switching():
    """Test backend switching integration."""
    original = backend_mgr.name

    # NumPy backend
    backend_mgr.set_active_backend("numpy")
    ops = get_backend_ops()
    assert ops.backend_name == "numpy"

    # JAX backend (if available)
    if JAX_AVAILABLE:
        backend_mgr.set_active_backend("jax")
        ops = get_backend_ops()
        assert ops.backend_name == "jax"

    # Restore original
    backend_mgr.set_active_backend(original)
    print("(v) Backend switching works correctly")


# ------------------------------------------------------------------------------


def test_linear_algebra_advanced():
    """Test advanced linear algebra operations."""
    ops = get_backend_ops("numpy")

    # Solve linear system
    A = ops.array([[3.0, 1.0], [1.0, 2.0]])
    b = ops.array([9.0, 8.0])
    x = ops.solve(A, b)
    assert np.allclose(x, [2.0, 3.0], rtol=1e-10)

    # Verify solution
    Ax = A @ x
    assert np.allclose(Ax, b, rtol=1e-10)

    # Cholesky decomposition (A is SPD)
    L = ops.cholesky(A, lower=True)
    A_reconstructed = L @ L.T
    assert np.allclose(A_reconstructed, A, rtol=1e-10)

    print("(v) Advanced linear algebra operations work")


# ------------------------------------------------------------------------------


def test_element_wise_operations():
    """Test element-wise operations."""
    ops = get_backend_ops("numpy")
    x = ops.array([1.0, 4.0, 9.0])

    # Square root
    sqrt_x = ops.sqrt(x)
    assert np.allclose(sqrt_x, [1.0, 2.0, 3.0])

    # Absolute value
    y = ops.array([-1.0, 2.0, -3.0])
    abs_y = ops.abs(y)
    assert np.allclose(abs_y, [1.0, 2.0, 3.0])

    # Complex operations
    z = ops.array([1 + 2j, 3 + 4j])
    conj_z = ops.conj(z)
    assert np.allclose(conj_z, [1 - 2j, 3 - 4j])

    print("(v) Element-wise operations work correctly")


# ------------------------------------------------------------------------------


def test_utility_operations():
    """Test utility operations."""
    ops = get_backend_ops("numpy")

    a = ops.array([1.0, 2.0, 3.0])
    b = ops.array([1.0, 2.0, 3.00001])

    # allclose
    assert ops.allclose(a, b, rtol=1e-3)
    assert not ops.allclose(a, b, rtol=1e-10)

    # where
    x = ops.array([1.0, 2.0, 3.0])
    y = ops.where(x > 1.5, x, 0.0)
    assert np.allclose(y, [0.0, 2.0, 3.0])

    # isnan, isinf
    z = ops.array([1.0, np.nan, np.inf])
    assert ops.any(ops.isnan(z))
    assert ops.any(ops.isinf(z))

    print("(v) Utility operations work correctly")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing backend_ops integration with utils.py\n")
    print("=" * 60)

    test_default_backend_matches_manager()
    test_from_array_detection()
    test_global_instances()
    test_backend_switching()
    test_basic_operations()
    test_givens_rotation()
    test_linear_algebra_advanced()
    test_element_wise_operations()
    test_utility_operations()

    print("=" * 60)
    print("\nv All tests passed!")
    print("\nBackend configuration:")
    print(f"  Active backend: {backend_mgr.name}")
    print(f"  JAX available: {JAX_AVAILABLE}")
    print(f"  Default seed: {backend_mgr.default_seed}")

# ------------------------------------------------------------------------------
#! END OF FILE
# ------------------------------------------------------------------------------
