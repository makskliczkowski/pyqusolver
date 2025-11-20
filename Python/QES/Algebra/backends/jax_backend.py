"""
JAX backend for automatic differentiation support.
"""

import numpy as np
from typing import Tuple, Optional
from .base import Backend

# --------------------------------------------------------------------------

class JaxBackend(Backend):
    """Backend using JAX for autodiff and GPU support."""
    
    def __init__(self):
        """Initialize JAX backend."""
        self._jax = None
        if self.is_available():
            import jax
            import jax.numpy as jnp
            self._jax = jax
            self._jnp = jnp
    
    @property
    def name(self) -> str:
        """Backend name: 'jax'."""
        return 'jax'
    
    def eigh(self, H: np.ndarray, k: Optional[int] = None,
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition using jax.numpy.linalg.eigh.
        
        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix
        k : int, optional
            Number of eigenvalues (ignored, JAX computes all)
        **kwargs
            Ignored
            
        Returns
        -------
        eig_val : np.ndarray
            Sorted eigenvalues
        eig_vec : np.ndarray
            Eigenvectors (columns)
        """
        H_jax = self._jnp.asarray(H)
        eig_val, eig_vec = self._jax.numpy.linalg.eigh(H_jax)
        return np.asarray(eig_val), np.asarray(eig_vec)
    
    def eigsh_gapped(self, H: np.ndarray, gap: float,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Find eigenvalues near gap using JAX (computes all).
        
        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix
        gap : float
            Target energy gap
        **kwargs
            Ignored
            
        Returns
        -------
        eig_val : np.ndarray
            All eigenvalues
        eig_vec : np.ndarray
            All eigenvectors (columns)
        """
        return self.eigh(H, **kwargs)
    
    def expm(self, H: np.ndarray, t: float) -> np.ndarray:
        """Matrix exponential using JAX.
        
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
        H_jax = self._jnp.asarray(H, dtype=complex)
        result = self._jax.scipy.linalg.expm(-1j * H_jax * t)
        return np.asarray(result)
    
    def is_available(self) -> bool:
        """Check if JAX is installed."""
        try:
            import jax
            return True
        except ImportError:
            return False

# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------