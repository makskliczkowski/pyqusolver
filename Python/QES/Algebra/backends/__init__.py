"""
Unified Backend Abstraction Layer

This module provides a consolidated interface to the global backend system
defined in QES.general_python.algebra.utils. It wraps the BackendManager
singleton to provide type-safe backend access for the QES.Algebra package.

Supported backends:
    - 'numpy'         : Pure NumPy (default)
    - 'jax'           : JAX with automatic differentiation

Architecture
------------
This module delegates to QES.general_python.algebra.utils.BackendManager,
which is the single source of truth for backend management. This ensures:

1. Consistent backend state across entire package
2. Unified dtype handling (numpy and jax types)
3. Centralized RNG and JIT compilation management
4. No duplicate backend registries

To use different backends in general_python.algebra, set environment variables:
    - PY_BACKEND='numpy' or 'jax'
    - PY_FLOATING_POINT='float32' or 'float64'

Examples
--------
>>> from QES.Algebra.backends import get_backend, get_available_backends
>>> 
>>> # List available backends
>>> backends = get_available_backends()
>>> for name, available in backends:
...     print(f"{name}: {'Available' if available else 'Not installed'}")
>>> 
>>> # Get backend instance
>>> backend = get_backend('numpy')
>>> 
>>> # Access backend manager directly (advanced)
>>> from QES.general_python.algebra.utils import backend_mgr
>>> print(f"Active backend: {backend_mgr.name}")

Notes
-----
The Backend abstract class is preserved for type compatibility, but
actual backend logic is managed by BackendManager in utils.py
    - through general_python.algebra.utils.BackendManager package.
Backend implementations (NumpyBackend, JaxBackend) wrap the global
backend manager's capabilities.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

# Import from the central backend management system
try:
    from QES.general_python.algebra.utils import (
        backend_mgr,
        jnp,
        JAX_AVAILABLE,
        _DTYPE_REGISTRY,
        _TYPE_TO_NAME,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import backend manager from QES.general_python.algebra.utils. "
        "Ensure the QES package is correctly installed."
    ) from e

# ============================================================================
# Backward Compatibility: Abstract Backend Class
# ============================================================================

class Backend(ABC):
    """
    Abstract base class for computation backends.
    
    This class provides the interface that backend implementations must follow.
    Actual backend logic is delegated to BackendManager in utils.py.
    
    Note
    ----
    This is primarily for type compatibility. The actual backend management
    is centralized in QES.general_python.algebra.utils.BackendManager.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass
    
    @abstractmethod
    def eigh(self, H: np.ndarray, k: Optional[int] = None, 
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition of Hermitian matrix."""
        pass
    
    @abstractmethod
    def eigsh_gapped(self, H: np.ndarray, gap: float,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Find eigenvalues around a spectral gap."""
        pass
    
    @abstractmethod
    def expm(self, H: np.ndarray, t: float) -> np.ndarray:
        """Matrix exponential: exp(-i*H*t)."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are installed."""
        pass

# ============================================================================
# Backward Compatibility: Registry (delegates to BackendManager)
# ============================================================================

class BackendRegistry:
    """
    Backward-compatible registry interface that delegates to BackendManager.
    
    This class provides the same API as before but internally uses
    the global BackendManager from utils.py.
    
    Warning
    -------
    This is primarily for backward compatibility. New code should use
    the functions below (get_backend, get_available_backends) instead.
    """
    
    def __init__(self):
        """Initialize registry (uses global backend_mgr)."""
        self._manager = backend_mgr
    
    def register(self, name: str, backend_class: type) -> None:
        """
        Register a backend class.
        
        Warning: This does not actually register backends in the global
        manager. It's a no-op for backward compatibility.
        """
        pass # No-op - backends are managed globally
    
    def get(self, name: str) -> Backend:
        """
        Get or create a backend instance.
        
        Delegates to the global BackendManager.
        """
        if name not in ['numpy', 'jax']:
            available = ['numpy']
            if JAX_AVAILABLE:
                available.append('jax')
            raise ValueError(
                f"Backend '{name}' not registered. "
                f"Available: {available}"
            )
        
        if name == 'numpy':
            from .numpy_backend import NumpyBackend
            return NumpyBackend()
        elif name == 'jax' and JAX_AVAILABLE:
            from .jax_backend import JaxBackend
            return JaxBackend()
        else:
            raise ValueError(f"Backend '{name}' not available")
    
    def available(self) -> List[Tuple[str, bool]]:
        """
        List all registered backends with availability status.
        
        Returns
        -------
        list
            [(name, available), ...]
        """
        return [
            ('numpy', True),
            ('jax', JAX_AVAILABLE),
        ]

# ============================================================================
# Primary Public API (delegates to BackendManager)
# ============================================================================

def get_backend(name: str) -> Backend:
    """
    Get a backend instance by name.
    
    This function provides access to backend implementations that wrap
    the global BackendManager from utils.py.
    
    Parameters
    ----------
    name : str
        Backend name ('numpy' or 'jax')
        
    Returns
    -------
    Backend
        Backend instance ready for use
        
    Raises
    ------
    ValueError
        If backend name is not registered or not available
        
    Examples
    --------
    >>> numpy_backend = get_backend('numpy')
    >>> print(f"Using backend: {numpy_backend.name}")
    Using backend: numpy
    
    >>> if JAX_AVAILABLE:
    ...     jax_backend = get_backend('jax')
    """
    if name == 'numpy':
        from .numpy_backend import NumpyBackend
        return NumpyBackend()
    elif name == 'jax':
        if not JAX_AVAILABLE:
            raise ValueError("JAX backend requested but not installed")
        from .jax_backend import JaxBackend
        return JaxBackend()
    else:
        available = ['numpy']
        if JAX_AVAILABLE:
            available.append('jax')
        raise ValueError(
            f"Backend '{name}' not recognized. "
            f"Available backends: {available}"
        )

def register_backend(name: str, backend_class: type) -> None:
    """
    Register a new backend.
    
    Warning: The central backend system (BackendManager in utils.py)
    manages available backends. This function is for backward compatibility
    and does not actually register new backends.
    
    Parameters
    ----------
    name : str
        Backend identifier
    backend_class : type
        Backend class (must inherit from Backend)
        
    Notes
    -----
    To add custom backends, modify BackendManager in
    QES/general_python/algebra/utils.py
    """
    pass  # No-op - backends are managed centrally


def get_backend_registry() -> BackendRegistry:
    """
    Access the backend registry (backward compatibility).
    
    Returns
    -------
    BackendRegistry
        Registry instance that delegates to global BackendManager
        
    Warning
    -------
    This is primarily for backward compatibility. New code should use
    get_backend() and get_available_backends() functions.
    """
    return BackendRegistry()


def get_available_backends() -> List[Tuple[str, bool]]:
    """
    List all available backends with status.
    
    This queries the global backend manager to determine which backends
    are available.
    
    Returns
    -------
    list
        [(name, available), ...] tuples
        
    Examples
    --------
    >>> from QES.Algebra.backends import get_available_backends
    >>> 
    >>> for name, available in get_available_backends():
    ...     status = "Available" if available else "Not installed"
    ...     print(f"  {name}: {status}")
    Available
      numpy: Available
      jax: Available
    """
    return [
        ('numpy', True),  # Always available
        ('jax', JAX_AVAILABLE),
    ]


def get_dtype_registry() -> Dict[str, Dict[str, Any]]:
    """
    Access the unified dtype registry.
    
    This provides access to the centralized type system that maps
    between numpy and jax dtypes.
    
    Returns
    -------
    dict
        Dtype registry from BackendManager
        
    Notes
    -----
    This is shared with QES.general_python.algebra.utils to ensure
    consistent type handling across the entire package.
    """
    return _DTYPE_REGISTRY


def get_dtype_map() -> Dict[Any, str]:
    """
    Access the dtype-to-name mapping.
    
    Returns
    -------
    dict
        Mapping from dtype to name (e.g., np.float64 -> 'float64')
    """
    return _TYPE_TO_NAME


def get_global_backend_manager():
    """
    Access the global backend manager.
    
    This is the central point for backend state management.
    
    Returns
    -------
    BackendManager
        Global backend manager instance from utils.py
        
    Notes
    -----
    Advanced use only. Most users should use get_backend() instead.
    """
    return backend_mgr


# ============================================================================
# Linear Algebra Operations (via backend_linalg)
# ============================================================================

def get_linalg_backend(backend: str = 'default'):
    """
    Get linear algebra operations for specified backend.
    
    Parameters
    ----------
    backend : str, optional
        Backend name ('default', 'numpy', 'jax'). Default auto-detects.
        
    Returns
    -------
    module
        Linear algebra module with operations: outer, kron, inner, overlap,
        eigh, eigsh, trace, identity, etc.
        
    Examples
    --------
    >>> from QES.Algebra.backends import get_linalg_backend
    >>> linalg = get_linalg_backend('numpy')
    >>> A = np.array([[1, 0], [0, -1]])
    >>> evals, evecs = linalg.eigh(A)
    """
    try:
        from QES.general_python.algebra import backend_linalg
        return backend_linalg
    except ImportError:
        raise ImportError(
            "backend_linalg module not available. "
            "Ensure QES.general_python.algebra is installed."
        )


# Convenience functions for common linalg operations
def outer(A, B, backend='default'):
    """Outer product using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.outer(A, B)

def kron(A, B, backend='default'):
    """Kronecker product using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.kron(A, B)

def inner(a, b, backend='default'):
    """Inner product using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.inner(a, b)

def overlap(a, O, b, backend='default'):
    """Matrix element <a|O|b> using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.overlap(a, O, b)

def eigh(A, backend='default'):
    """Hermitian eigendecomposition using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.eigh(A)

def eigsh(A, k, which='SA', backend='default'):
    """Partial eigendecomposition (sparse) using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.eigsh(A, k, which)

def trace(A, backend='default'):
    """Matrix trace using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.trace(A)

def identity(n, dtype=None, backend='default'):
    """Identity matrix using backend_linalg."""
    linalg = get_linalg_backend(backend)
    return linalg.identity(n, dtype)

# ============================================================================

__all__ = [
    'Backend',
    'BackendRegistry',
    'get_backend',
    'register_backend',
    'get_backend_registry',
    'get_available_backends',
    'get_dtype_registry',
    'get_dtype_map',
    'get_global_backend_manager',
    'JAX_AVAILABLE',
    # Linear algebra operations
    'get_linalg_backend',
    'outer',
    'kron',
    'inner',
    'overlap',
    'eigh',
    'eigsh',
    'trace',
    'identity',
]

# ============================================================================
#! End of module
# ============================================================================
