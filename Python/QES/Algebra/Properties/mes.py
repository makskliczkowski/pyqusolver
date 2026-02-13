r'''

'''

import  numpy as np
from    typing import Callable, List, Tuple

def normalize(c):
    ''' Normalize a complex vector. '''
    return c / np.linalg.norm(c)

def random_c(m):
    ''' Generate random complex vector and normalize it. '''
    c = np.random.randn(m) + 1j*np.random.randn(m)
    return normalize(c)

def random(m):
    ''' Generate random vector without normalization. '''
    return np.random.randn(m)

# ---------------------------------------
# Entropy minimization to find MES
# ---------------------------------------

def entropy_of_c(c, V, S_func):
    ''' Compute the entropy S(V @ c) for a given vector c. '''
    psi = V @ c
    return S_func(psi)

# ---------------------------------------
# Projected gradient descent on the unit sphere
# ---------------------------------------

def projected_gradient_step(c, V, S_func, eps=1e-6, lr=0.1):
    ''' Perform a projected gradient step to minimize entropy on the unit sphere. '''
    m       = len(c)
    grad    = np.zeros_like(c, dtype=np.complex128)
    base    = entropy_of_c(c, V, S_func)

    # finite-difference gradient
    for i in range(m):
        dc          = np.zeros_like(c)
        dc[i]       = eps
        grad[i]     = (entropy_of_c(normalize(c + dc), V, S_func) - base) / eps

        dc[i]       = 1j * eps
        grad[i]    += 1j * (entropy_of_c(normalize(c + dc), V, S_func) - base) / eps

    # project gradient to tangent space of unit sphere
    grad    = grad - np.vdot(c, grad) * c
    c_new   = normalize(c - lr * grad)
    return c_new

# ---------------------------------------
# Main function to find MES
# ---------------------------------------

def minimize_entropy(V, S_func: Callable, max_iter=200, tol=1e-10):
    ''' Minimize the entropy S(V @ c) over normalized vectors c using projected gradient descent. '''
    
    m       = V.shape[1]
    c       = random_c(m)
    prev    = entropy_of_c(c, V, S_func)

    for _ in range(max_iter):
        c       = projected_gradient_step(c, V, S_func)
        val     = entropy_of_c(c, V, S_func)

        if abs(val - prev) < tol:
            break
        prev = val

    return normalize(c), prev

# ---------------------------------------
# Wrapper to find multiple MES by random restarts
# ---------------------------------------

def find_mes(V, S_func, n_trials=30, overlap_tol=1e-6, state_max=10):
    ''' 
    Find multiple MES by performing multiple runs of entropy minimization with random initializations.
    Parameters    
    ----------
    V : np.ndarray
        The matrix whose columns span the subspace of interest (e.g., low-energy eigenstates).
    S_func : Callable
        A function that takes a state vector and returns its entanglement entropy.
    n_trials : int, optional
        The number of random initializations to try for finding MES. Default is 30.
    overlap_tol : float, optional
        The tolerance for considering two MES as distinct based on their overlap. Default is 1e-6.
    state_max : int, optional
        The maximum number of MES to find. Default is 10.
    Returns
    -------
    List[np.ndarray]
        A list of MES state vectors found.
    List[float]
        The corresponding entanglement entropy values for the MES found.
    '''    
    
    minima = []
    values = []

    for _ in range(n_trials):
        
        # Minimize entropy to find one MES
        c, val  = minimize_entropy(V, S_func)
        psi     = V @ c

        # check uniqueness
        unique = True
        for psi_prev in minima:
            if abs(np.vdot(psi_prev, psi)) > 1 - overlap_tol:
                unique = False
                break

        if unique:
            minima.append(psi)
            values.append(val)
            
        if len(minima) >= state_max:
            break

    # Orthonormalize MES
    if minima:
        Q, _    = np.linalg.qr(np.column_stack(minima))
        minima  = [Q[:, i] for i in range(Q.shape[1])]

    return minima, values

# ---------------------------------------
#! EOF
# ---------------------------------------
