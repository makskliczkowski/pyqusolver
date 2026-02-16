r'''
Here, a minimum entangled state (MES) finder is implemented using projected gradient 
descent on the unit sphere of coefficients. The main function `find_mes` 
performs multiple random restarts to find distinct MES, which are then orthonormalized. 
This approach allows us to identify states in the ground state manifold that minimize 
the entanglement entropy for a given partition, which is crucial 
for extracting topological entanglement entropy and understanding the underlying topological order.

--------------------------------
Author      : Maksymilian Kliczkowski
Created     : 2026-02-12
--------------------------------
'''

import  numpy as np
from    typing import Callable, List, Tuple, Optional, Dict, Any
from    dataclasses import dataclass

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------------------
#! Topological Results Dataclass
# ---------------------------------------

@dataclass
class TopologicalResults:
    """
    Data class to store results of topological analysis from MES.
    
    Attributes
    ----------
    S_matrix : np.ndarray
        The modular S-matrix (normalized overlap matrix). S_ij = <MES_i|MES_j> / D, where D is the total quantum dimension.
    overlap_matrix : np.ndarray
        The raw overlap matrix between two MES bases. This is the unnormalized version of S_matrix
        and can be used to extract quantum dimensions.
    quantum_dimensions : np.ndarray
        Individual quantum dimensions d_i for each anyon type.
    total_quantum_dimension : float
        The total quantum dimension D = sqrt(sum d_i^2). This is a key quantity characterizing the topological order.
    is_abelian : bool
        True if all d_i are approximately 1.
    is_non_abelian : bool
        True if any d_i > 1 (within tolerance).
    """
    S_matrix                : np.ndarray
    overlap_matrix          : np.ndarray
    quantum_dimensions      : np.ndarray
    total_quantum_dimension : float
    is_abelian              : bool
    is_non_abelian          : bool

    def __repr__(self) -> str:
        return (f"TopologicalResults(D={self.total_quantum_dimension:.4f}, "
                f"Abelian={self.is_abelian}, Non-Abelian={self.is_non_abelian}, "
                f"d_i={self.quantum_dimensions})")

# ---------------------------------------
#! Minimum Entangled States (MES) Finder
# ---------------------------------------

def _params_to_complex(params: np.ndarray, m: int):
    ''' Convert real parameters to a complex vector of length m. '''
    if params.size != 2*m:
        raise ValueError(f"Expected 2*m parameters, got {params.size}")

    c   = params[:m] + 1j * params[m:]
    nrm = np.linalg.norm(c)
    
    if nrm < 1e-13:
        c       = np.zeros(m, dtype=np.complex128)
        c[0]    = 1.0
        return c
    return c / nrm

class MESFinder:
    ''' Class to find Minimum Entangled States (MES) by minimizing the entanglement entropy. '''
    
    def __init__(self, V: np.ndarray, S_func: Callable[[np.ndarray], float]):
        """
        Initialize the MESFinder.
        
        Parameters
        ----------
        V : np.ndarray
            A matrix of shape (Ns, m) where m is the dimension of the degenerate manifold.
            Each column is a basis state of the manifold.
        S_func : Callable[[np.ndarray], float]
            A function that takes a state vector of size Ns and returns its entanglement entropy.
        """
        self.V          = V
        self.S_func     = S_func
        self.m          = V.shape[1]
        self.Ns         = V.shape[0]
        # Pre-allocate buffer for state vector to avoid multiple large allocations.
        # Ensure it is complex as coefficients c are complex.
        self._psi_buf   = np.zeros(self.Ns, dtype=np.complex128 if np.isrealobj(V) else V.dtype)

    @staticmethod
    def normalize(c):
        ''' Normalize a complex vector. '''
        nrm = np.linalg.norm(c)
        if nrm < 1e-15: return c
        return c / nrm

    @staticmethod
    def random_c(m, rng=None):
        ''' Generate random complex vector and normalize it. '''
        if rng is None:
            c = np.random.randn(m) + 1j*np.random.randn(m)
        else:
            c = rng.standard_normal(m) + 1j*rng.standard_normal(m)
        return MESFinder.normalize(c)

    # --------------------------------
    #! Find
    # --------------------------------
    
    def __call__(self, c):
        ''' Compute the entropy S(V @ c) for a given vector c. '''
        
        # Optimized matrix-vector multiplication into buffer
        # psi = self.V @ c
        np.dot(self.V, c, out=self._psi_buf)
        return self.S_func(self._psi_buf)

    # --------------------------------
    #! Optimization methods
    # --------------------------------

    def _minimize_projected_gradient(self, c, eps=1e-6, lr=0.1, max_iter=200, tol=1e-8):
        ''' Perform a projected gradient step to minimize entropy on the unit sphere. '''
        m           = len(c)
        c_curr      = c.copy()
        val_curr    = self(c_curr)
        
        for _ in range(max_iter):
            grad    = np.zeros_like(c_curr, dtype=np.complex128)
            # finite-difference gradient
            for i in range(m):
                # Real part
                old_val     = c_curr[i]
                c_curr[i]  += eps
                val_plus    = self(self.normalize(c_curr))
                grad[i]     = (val_plus - val_curr) / eps
                c_curr[i]   = old_val
                
                # Imaginary part
                c_curr[i]  += 1j * eps
                val_plus_im = self(self.normalize(c_curr))
                grad[i]    += 1j * (val_plus_im - val_curr) / eps
                c_curr[i]   = old_val

            # project gradient to tangent space of unit sphere
            grad        = grad - np.vdot(c_curr, grad) * c_curr
            
            c_next      = self.normalize(c_curr - lr * grad)
            val_next    = self(c_next)
            
            if abs(val_next - val_curr) < tol:
                break
                
            c_curr      = c_next
            val_curr    = val_next
            
        return c_curr

    # --------------------------------
    
    def minimize_entropy(
        self,
        max_iter            : int=200, 
        tol                 : float=1e-10,
        n_restarts          : int=5,
        rng                 : Optional[np.random.Generator] = None,
        seed                : Optional[int] = None,
        # projected gradient options
        eps                 : float = 1e-6,
        lr                  : float = 0.1,
        # scipy options
        method              : str = 'L-BFGS-B',
        options             : Optional[Dict[str, Any]] = None,
        verbose             : bool = False
    ):
        ''' Minimize the entropy S(V @ c) over normalized vectors c using projected gradient descent. '''
        if options is None:
            options = {'maxiter': max_iter, 'ftol': tol}
        
        m           = self.m
        rng         = np.random.default_rng(seed) if rng is None else rng
        
        best_c      = None      # best vector found across restarts
        best_val    = np.inf    # best value found across restarts
        nfev_total  = 0         # total number of function evaluations across restarts
        
        if verbose:
            print(f"Starting MES optimization with method={method}, max_iter={max_iter}, tol={tol}, n_restarts={n_restarts}")
        
        def obj(params: np.ndarray):
            c   = _params_to_complex(params, m)
            return self(c)
        
        # Optimize using scipy if available, otherwise use manual projected gradient descent
        for i in range(n_restarts):
            c0      = self.random_c(m, rng=rng)
            if SCIPY_AVAILABLE:
                p0      = np.concatenate([c0.real, c0.imag])
                res     = minimize(obj, p0, method=method, options=options)
                c_opt   = _params_to_complex(res.x, m)
                val     = res.fun
                nfev    = res.nfev
            else:
                c_opt   = self._minimize_projected_gradient(c0, eps=eps, lr=lr, max_iter=max_iter, tol=tol)
                val     = self(c_opt)
                nfev    = max_iter # Approximate
            
            nfev_total += nfev
            if verbose:
                print(f"\tRestart {i+1}/{n_restarts}: val={val:.8f}, nfev={nfev}")
                
            if val < best_val:
                best_val = val
                best_c   = c_opt
                
        info = {
            "method"        : method if SCIPY_AVAILABLE else "projected_gradient",
            "nfev_total"    : nfev_total,
            "best_val"      : best_val,
            "best_c_norm"   : np.linalg.norm(best_c) if best_c is not None else None,
            "n_restarts"    : n_restarts
        }
        
        if verbose:
            print(f"Best value found: {best_val:.8f} with method={info['method']}")
        
        return self.normalize(best_c), float(best_val), info

    def find(
        self,
        n_trials        : int = 20, 
        overlap_tol     : float = 1e-4, 
        state_max       : int = 10,
        # optimization options
        rng             : Optional[np.random.Generator] = None,
        seed            : Optional[int] = None,
        max_iter        : int = 200,
        tol             : float = 1e-10,
        n_restarts      : int = 3,
        method          : str = 'L-BFGS-B',
        options         : Optional[Dict[str, Any]] = None,
        verbose         : bool = False,
    ) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[Dict]]:
        r''' 
        Find multiple MES by performing multiple runs of entropy minimization with random initializations.
        '''    
        
        rng         = np.random.default_rng(seed) if rng is None else rng    
        minima      = []
        values      = []
        coeffs      = []
        diagnostics = []

        # To find multiple MES, we perform multiple optimization runs with random initializations.
        for trial in range(n_trials):
            if verbose:
                print(f"Trial {trial+1}/{n_trials}...")
                
            c_opt, val, info = self.minimize_entropy(
                max_iter=max_iter, tol=tol, n_restarts=n_restarts, rng=rng, seed=None,
                method=method, options=options, verbose=verbose
            )
            
            # Check for uniqueness against found coefficients in the subspace
            # This is more efficient than checking full state vectors
            is_unique = True
            for c_prev in coeffs:
                
                # In the subspace, we check overlap |<c_prev|c_opt>|
                if abs(np.vdot(c_prev, c_opt)) > 1.0 - overlap_tol:
                    is_unique = False
                    break
            
            if is_unique:
                psi = self.V @ c_opt
                minima.append(psi)
                coeffs.append(c_opt)
                values.append(float(val))
                diagnostics.append(info)
                if verbose:
                    print(f"  -> Found unique MES with S={val:.6f}")
            
            if len(minima) >= state_max:
                break

        # Orthonormalize the found MES manifold if requested or needed
        # Often MES states are not exactly orthogonal, but they represent the manifold.
        # But if the user wants an orthonormal basis of MES, we can do QR.
        # Note: Orthonormalizing might change the entropy of the states slightly if they were not orthogonal.
        # For Kitaev model, they should be approximately orthogonal for large systems.
        if len(minima) > 1:
            Q, _    = np.linalg.qr(np.column_stack(coeffs))
            coeffs  = [Q[:, i] for i in range(Q.shape[1])]
            minima  = [self.V @ c for c in coeffs]

        return minima, values, coeffs, diagnostics

def find_mes(V: np.ndarray, S_func: Callable[[np.ndarray], float], **kwargs):
    """
    Convenience function to find MES using MESFinder.
    """
    finder = MESFinder(V, S_func)
    return finder.find(**kwargs)

# ---------------------------------------
#! Modular S-matrix and Topological Statistics
# ---------------------------------------

def compute_modular_s_matrix(
    mes_x   : List[np.ndarray], 
    mes_y   : List[np.ndarray], 
    tol     : float = 1e-8
) -> TopologicalResults:
    r"""
    Compute the modular S-matrix from two bases of Minimum Entangled States (MES)
    obtained from perpendicular cuts (e.g., along x and y).
    
    The modular S-matrix S_ij encodes the topological mutual statistics between anyons.
    It is proportional to the overlap matrix:
    S_ij = <Xi_i(x) | Xi_j(y)> / D
    
    Parameters
    ----------
    mes_x : List[np.ndarray]
        List of MES obtained from a cut along the x-direction.
    mes_y : List[np.ndarray]
        List of MES obtained from a cut along the y-direction.
    tol : float
        Numerical tolerance for Abelian/Non-Abelian classification.
        
    Returns
    -------
    TopologicalResults
        Dataclass containing the modular S-matrix, quantum dimensions, and total quantum dimension.
    """
    n_x     = len(mes_x)
    n_y     = len(mes_y)
    
    if n_x != n_y:
        raise ValueError(f"Number of MES in x and y bases must be equal (got {n_x} and {n_y}).")
    
    m       = n_x
    overlap = np.zeros((m, m), dtype=np.complex128)
    
    for i in range(m):
        for j in range(m):
            overlap[i, j] = np.vdot(mes_x[i], mes_y[j])
            
    # The modular S-matrix is unitary. The first row (or column) contains d_i / D.
    # Since d_i >= 1, the largest values in the first row correspond to d_i.
    
    row1        = np.abs(overlap[0, :])
    d_tilde     = row1
    d_1_tilde   = d_tilde[0]
    
    if d_1_tilde < 1e-10:
        d_1_tilde = np.max(d_tilde)
    
    quantum_dims    = d_tilde / d_1_tilde
    total_D         = 1.0 / d_1_tilde
    
    # Normalized S-matrix
    s_matrix        = overlap / total_D
    
    is_abelian      = np.all(np.abs(quantum_dims - 1.0) < tol)
    is_non_abelian  = np.any(quantum_dims > 1.0 + tol)
    
    return TopologicalResults(
        S_matrix                = s_matrix,
        overlap_matrix          = overlap,
        quantum_dimensions      = quantum_dims,
        total_quantum_dimension = total_D,
        is_abelian              = is_abelian,
        is_non_abelian          = is_non_abelian
    )

# ---------------------------------------
#! EOF
# ---------------------------------------
