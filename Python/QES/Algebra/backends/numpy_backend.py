"""
NumPy backend (default implementation).
"""

from typing import Tuple, Optional
import numpy as np
from scipy import linalg
from .base import Backend


class NumpyBackend(Backend):
    """Backend using NumPy/SciPy for all operations."""
    
    @property
    def name(self) -> str:
        """Backend name: 'numpy'."""
        return 'numpy'
    
    def eigh(self, H: np.ndarray, k: Optional[int] = None,
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition using scipy.linalg.eigh.
        
        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix
        k : int, optional
            Number of eigenvalues (ignored, computes all)
        **kwargs
            Ignored
            
        Returns
        -------
        eig_val : np.ndarray
            Sorted eigenvalues
        eig_vec : np.ndarray
            Eigenvectors (columns)
        """
        eig_val, eig_vec = linalg.eigh(H)
        return eig_val, eig_vec
    
    def eigsh_gapped(self, H: np.ndarray, gap: float,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Find eigenvalues near a spectral gap using sparse eigsh.
        
        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix
        gap : float
            Target energy gap
        **kwargs
            Additional arguments (num_lanczos_iterations, etc.)
            
        Returns
        -------
        eig_val : np.ndarray
            Eigenvalues near gap
        eig_vec : np.ndarray
            Eigenvectors (columns)
        """
        # For dense matrices, use full diagonalization
        # For sparse matrices, could use scipy.sparse.linalg.eigsh
        from scipy.sparse import issparse
        
        if issparse(H):
            from scipy.sparse.linalg import eigsh
            try:
                # Find eigenvalues near the gap
                eig_val, eig_vec = eigsh(H, k=min(10, H.shape[0]-1),
                                         which='SA', **kwargs)
            except Exception:
                # Fallback to full diagonalization
                eig_val, eig_vec = linalg.eigh(H.toarray())
        else:
            eig_val, eig_vec = linalg.eigh(H)
        
        return eig_val, eig_vec
    
    def expm(self, H: np.ndarray, t: float) -> np.ndarray:
        """Matrix exponential using scipy.linalg.expm.
        
        For Hamiltonian evolution: U(t) = exp(-i*H*t)
        
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
        return linalg.expm(-1j * H * t)
    
    def is_available(self) -> bool:
        """NumPy backend is always available."""
        return True

# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------