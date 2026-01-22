"""
Abstract base class for backends (DEPRECATED - use __init__.py instead).

This module defines the Backend ABC for type compatibility. Actual backend
management is centralized in QES.general_python.algebra.utils.BackendManager.

The BackendRegistry class here is kept for backward compatibility but should
not be used in new code. Use get_backend() and get_available_backends() from
__init__.py instead.

Architecture Note
-----------------
This module is part of the consolidated backend system. All actual backend
state and management is delegated to the global BackendManager singleton
in QES/general_python/algebra/utils.py to ensure:
- Single source of truth for backend state
- Consistent dtype handling (numpy and jax types)
- Unified RNG and JIT compilation
- No duplicate registries or state
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class Backend(ABC):
    """Abstract base class for computation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    @abstractmethod
    def eigh(
        self, H: np.ndarray, k: Optional[int] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition of Hermitian matrix.

        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix
        k : int, optional
            Number of eigenvalues to compute (None = all)
        **kwargs
            Backend-specific options

        Returns
        -------
        eig_val : np.ndarray
            Eigenvalues
        eig_vec : np.ndarray
            Eigenvectors (columns)
        """
        pass

    @abstractmethod
    def eigsh_gapped(self, H: np.ndarray, gap: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Find eigenvalues around a spectral gap.

        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix
        gap : float
            Target energy gap
        **kwargs
            Backend-specific options

        Returns
        -------
        eig_val : np.ndarray
            Eigenvalues near gap
        eig_vec : np.ndarray
            Eigenvectors (columns)
        """
        pass

    @abstractmethod
    def expm(self, H: np.ndarray, t: float) -> np.ndarray:
        """Matrix exponential: exp(-i*H*t).

        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix
        t : float
            Evolution time

        Returns
        -------
        np.ndarray
            exp(-i*H*t)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are installed."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# DEPRECATED: BackendRegistry (use functions in __init__.py instead)
# ============================================================================


class BackendRegistry:
    """
    DEPRECATED: Registry for available backends.

    This class is kept for backward compatibility only.
    Use get_backend() and get_available_backends() from __init__.py instead.

    The actual backend management is centralized in
    QES.general_python.algebra.utils.BackendManager.
    """

    def __init__(self):
        """Initialize backend registry (backward compatibility only)."""
        self._backends: Dict[str, type] = {}
        self._instances: Dict[str, Backend] = {}

    def register(self, name: str, backend_class: type) -> None:
        """
        Register a backend class (DEPRECATED - no-op).

        This method does nothing. Backends are managed centrally in
        QES.general_python.algebra.utils.BackendManager.
        """
        pass

    def get(self, name: str) -> Backend:
        """Get or create a backend instance."""
        from . import get_backend

        return get_backend(name)

    def available(self) -> list:
        """List all registered backends with availability status."""
        from . import get_available_backends

        return get_available_backends()
