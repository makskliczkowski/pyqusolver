'''
Methods for computing the entanglement spectrum of a quantum state, 
which is the set of eigenvalues of the reduced density matrix of a subsystem. 
The entanglement spectrum provides insights into the entanglement structure and topological properties of quantum states.

--------------------------------
Authors:            Maksymilian Kliczkowski
Created:            2026-02-12
--------------------------------
'''

import  numpy as np
import  scipy.linalg as la
from    typing import List, Tuple, Union, Optional

try:
    from QES.general_python.physics.density_matrix import psi_numpy, mask_subsystem, schmidt
except ImportError:
    raise ImportError("Could not import psi_numpy and mask_subsystem from QES.general_python.physics.density_matrix. Please ensure the module is available.")

# -----------------------------------------------------------------------------
#! Main functions
# -----------------------------------------------------------------------------

def calculate_entanglement_spectrum(
    psi         : np.ndarray, 
    region      : List[int], 
    ns          : int, 
    eps         : float = 1e-14,
    sort        : bool = True,
) -> np.ndarray:
    """
    Compute the entanglement spectrum (log of eigenvalues of reduced density matrix).
    
    The entanglement spectrum is defined as xi_i = -ln(lambda_i), where lambda_i are
    the eigenvalues of the reduced density matrix rho_A.
    
    Parameters
    ----------
    psi : np.ndarray
        The quantum state vector.
    region : List[int]
        List of site indices in subsystem A.
    ns : int
        Total number of sites.
    eps : float
        Threshold for small eigenvalues (to avoid log of zero).
    sort : bool
        If True, sort the spectrum in ascending order (descending eigenvalues).
        
    Returns
    -------
    np.ndarray
        The entanglement spectrum xi_i.
    """
    if len(region) == 0 or len(region) == ns:
        return np.array([])
    
    # Get eigenvalues of the reduced density matrix using Schmidt decomposition
    eigvals     = schmidt(psi, va=region, ns=ns, eig=False, contiguous=False, square=True, return_vecs=False)
            
    # Filter and take -log
    eigvals     = eigvals[eigvals > eps]
    spectrum    = -np.log(eigvals)
    
    if sort:
        spectrum = np.sort(spectrum)
        
    return spectrum

def calculate_entanglement_spectrum_manifold(
    V           : np.ndarray, 
    coeffs      : np.ndarray,
    region      : List[int], 
    ns          : int, 
    eps         : float = 1e-14
) -> np.ndarray:
    """
    Compute the entanglement spectrum for a state in a manifold defined by coefficients.
    
    psi = V @ coeffs
    """
    psi = V @ coeffs
    return calculate_entanglement_spectrum(psi, region, ns, eps=eps)

def entanglement_entropy_from_spectrum(spectrum: np.ndarray, q: float = 1.0) -> float:
    """
    Compute Rényi entropy from entanglement spectrum.
    
    S_q = 1/(1-q) * ln(sum exp(-q * xi_i))
    For q=1, S_1 = sum xi_i * exp(-xi_i)
    """
    if len(spectrum) == 0:
        return 0.0
        
    if abs(q - 1.0) < 1e-9:
        # Von Neumann: sum lambda_i * (-ln lambda_i) = sum exp(-xi) * xi
        eigvals = np.exp(-spectrum)
        return np.sum(eigvals * spectrum)
    else:
        # Renyi: 1/(1-q) * ln(sum lambda_i^q) = 1/(1-q) * ln(sum exp(-q * xi))
        # Use log-sum-exp trick for numerical stability
        max_val = np.min(spectrum) # min xi corresponds to max lambda
        s       = np.sum(np.exp(-q * (spectrum - max_val)))
        return (np.log(s) - q * max_val) / (1.0 - q)

# ---------------------------------------
#! EOF
# ---------------------------------------
