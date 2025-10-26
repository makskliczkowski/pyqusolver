"""
Helper functions and methods for Hamiltonian diagonalization.

This module contains utility functions for basis transformations and
diagnostics related to the diagonalization engine, extracted from the
main Hamiltonian class to keep the code organized.

-----------------------------------------------------------------------------------------
File        : QES/Algebra/Hamil/hamil_diag_helpers.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 2025-10-26
-----------------------------------------------------------------------------------------
"""

import numpy as np
from typing import Optional, Dict, Any
from numpy.typing import NDArray

# ----------------------------------------------------------------------------------------
#! Basis Transformation Utilities
# ----------------------------------------------------------------------------------------

def has_krylov_basis(diag_engine, krylov_stored: Optional[NDArray]) -> bool:
    """
    Check if a Krylov basis is available.
    
    Parameters:
    -----------
        diag_engine : DiagonalizationEngine or None
            The diagonalization engine instance
        krylov_stored : ndarray or None
            Stored Krylov basis matrix
    
    Returns:
    --------
        bool : True if Krylov basis is available, False otherwise
    """
    if diag_engine is not None:
        return diag_engine.has_krylov_basis()
    return krylov_stored is not None

def get_krylov_basis(diag_engine, krylov_stored: Optional[NDArray]) -> Optional[NDArray]:
    """
    Get the Krylov basis.
    
    Parameters:
    -----------
        diag_engine : DiagonalizationEngine or None
            The diagonalization engine instance
        krylov_stored : ndarray or None
            Stored Krylov basis matrix
    
    Returns:
    --------
        ndarray or None : Krylov basis matrix V (shape n x k)
    """
    if diag_engine is not None:
        return diag_engine.get_krylov_basis()
    return krylov_stored

def to_original_basis(vec           : NDArray,
                     diag_engine    : Optional['DiagonalizationEngine'],
                     method_used    : Optional[str]) -> NDArray:
    """
    Transform a vector from Krylov/computational basis to original basis.
    
    Parameters:
    -----------
        vec : ndarray
            Vector(s) in Krylov basis
        diag_engine : DiagonalizationEngine or None
            The diagonalization engine instance
        method_used : str or None
            Method that was used for diagonalization
    
    Returns:
    --------
        ndarray : Vector(s) in original basis
    """
    if diag_engine is None:
        # No engine, assume exact or no transformation needed
        return vec
    return diag_engine.to_original_basis(vec)

def to_krylov_basis(vec: NDArray, diag_engine: Optional['DiagonalizationEngine']) -> NDArray:
    """
    Transform a vector from original basis to Krylov/computational basis.
    
    Parameters:
    -----------
        vec : ndarray
            Vector(s) in original basis
        diag_engine : DiagonalizationEngine or None
            The diagonalization engine instance
    
    Returns:
    --------
        ndarray : Vector(s) in Krylov basis
    
    Raises:
    -------
        ValueError : If no engine or Krylov basis available
    """
    if diag_engine is None:
        raise ValueError("No diagonalization engine available. Run diagonalize() with an iterative method first.")
    return diag_engine.to_krylov_basis(vec)

def get_basis_transform(diag_engine: Optional['DiagonalizationEngine'], krylov_stored: Optional[NDArray]) -> Optional[NDArray]:
    """
    Get the transformation matrix from Krylov to original basis.
    
    Parameters:
    -----------
        diag_engine : DiagonalizationEngine or None
            The diagonalization engine instance
        krylov_stored : ndarray or None
            Stored Krylov basis matrix
    
    Returns:
    --------
        ndarray or None : Transformation matrix V (shape n x k)
    """
    if diag_engine is not None:
        return diag_engine.get_basis_transform()
    return krylov_stored

# ----------------------------------------------------------------------------------------
#! Diagnostic Information
# ----------------------------------------------------------------------------------------

def get_diagonalization_method(diag_engine) -> Optional[str]:
    """
    Get the diagonalization method used.
    
    Parameters:
    -----------
        diag_engine : DiagonalizationEngine or None
            The diagonalization engine instance
    
    Returns:
    --------
        str or None : Method name or None if not yet diagonalized
    """
    if diag_engine is not None:
        return diag_engine.get_method_used()
    return None

def get_diagonalization_info(diag_engine    : Optional['DiagonalizationEngine'],
                            eig_val         : Optional[NDArray],
                            krylov_stored   : Optional[NDArray]) -> Dict[str, Any]:
    """
    Get detailed information about the last diagonalization.
    
    Parameters:
    -----------
        diag_engine : DiagonalizationEngine or None
            The diagonalization engine instance
        eig_val : ndarray or None
            Stored eigenvalues
        krylov_stored : ndarray or None
            Stored Krylov basis
    
    Returns:
    --------
        dict : Dictionary with diagnostic information
    """
    info = {
        'method': None,
        'converged': True,
        'iterations': None,
        'residual_norms': None,
        'has_krylov_basis': has_krylov_basis(diag_engine, krylov_stored),
        'num_eigenvalues': len(eig_val) if eig_val is not None else 0
    }
    
    if diag_engine is not None:
        info['method'] = diag_engine.get_method_used()
        info['converged'] = diag_engine.converged()
        result = diag_engine.get_result()
        if result is not None:
            if hasattr(result, 'iterations'):
                info['iterations'] = result.iterations
            if hasattr(result, 'residual_norms'):
                info['residual_norms'] = result.residual_norms
    
    return info

# ----------------------------------------------------------------------------------------
#! Validation and Utilities
# ----------------------------------------------------------------------------------------

def validate_diagonalization_params(method: str, k: Optional[int], n: int) -> None:
    """
    Validate diagonalization parameters.
    
    Parameters:
    -----------
        method : str
            Diagonalization method
        k : int or None
            Number of eigenvalues to compute
        n : int
            Matrix dimension
    
    Raises:
    -------
        ValueError : If parameters are invalid
    """
    if method in ['lanczos', 'arnoldi', 'block_lanczos', 'shift-invert']:
        if k is None:
            raise ValueError(f"Method '{method}' requires specifying k (number of eigenvalues)")
        if k >= n:
            raise ValueError(f"k ({k}) must be less than matrix dimension n ({n})")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

def suggest_diagonalization_method(n        : int,
                                  k         : Optional[int],
                                  is_sparse : bool,
                                  hermitian : bool) -> str:
    """
    Suggest an appropriate diagonalization method based on problem characteristics.
    
    Parameters:
    -----------
        n : int
            Matrix dimension
        k : int or None
            Number of eigenvalues needed
        is_sparse : bool
            Whether matrix is sparse
        hermitian : bool
            Whether matrix is symmetric/Hermitian
    
    Returns:
    --------
        str : Suggested method name
    """
    
    # Small matrices: exact
    if n <= 500:
        return 'exact'
    
    # Large non-symmetric: arnoldi
    if not hermitian:
        return 'arnoldi'
    
    # Large sparse symmetric
    if k is None or k > n * 0.5:
        # Need most eigenvalues
        if is_sparse and n > 1000:
            return 'block_lanczos'
        return 'exact'
    
    # Few eigenvalues
    if k == 1:
        return 'lanczos'
    elif k >= 10 or n > 5000:
        return 'block_lanczos'
    else:
        return 'lanczos'

# ----------------------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------------------
