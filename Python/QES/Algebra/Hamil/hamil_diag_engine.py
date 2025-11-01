r"""
Diagonalization Engine for Hamiltonian class. It integrates various
eigenvalue solvers into a unified interface (factory pattern) and
automatically selects the most suitable method based on the problem size
and characteristics.

Provides a modular interface for different diagonalization methods including:
- Exact diagonalization (full spectrum)
- Lanczos iteration (sparse symmetric/Hermitian)
- Block Lanczos (multiple eigenpairs, sparse symmetric)
- Arnoldi iteration (general matrices)
- Shift-invert methods for interior eigenvalues
- SciPy backends: scipy-eigh, scipy-eig, scipy-eigs, lobpcg
    - Exact methods: scipy-eigh, scipy-eig
    - Iterative methods: scipy-eigs, lobpcg
- JAX GPU-accelerated methods: jax-eigh
The engine automatically decides the best method based on matrix size, sparsity, and requested number of eigenvalues.

The engine stores Krylov basis information when using iterative methods and
provides utilities to transform between Krylov and original basis.

----------------------------------------
File        : QES/Algebra/Hamil/hamil_diag_engine.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 2025-10-26
----------------------------------------
"""

import numpy as np
import scipy as sp
import logging
from typing import Optional, Literal, Union, Callable, Any
from numpy.typing import NDArray
from enum import Enum

# Import eigensolvers from the general_python module
try:
    from QES.general_python.algebra.eigen.factory import choose_eigensolver, decide_method
    from QES.general_python.algebra.eigen.result import EigenResult
    EIGEN_SOLVERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import eigensolvers: {e}")
    EIGEN_SOLVERS_AVAILABLE = False
    EigenResult             = None
    choose_eigensolver      = None
    decide_method           = None

# JAX support
try:
    import jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    JAX_AVAILABLE   = False
    jnp             = None

# ----------------------------------------------------------------------------------------
#! Diagonalization Engine
# ----------------------------------------------------------------------------------------

class DiagonalizationMethods(Enum):
    AUTO            = 'auto'
    EXACT           = 'exact'
    LANCZOS         = 'lanczos'
    BLOCK_LANCZOS   = 'block_lanczos'
    ARNOLDI         = 'arnoldi'
    SHIFT_INVERT    = 'shift-invert'
    SCIPY_EIGH      = 'scipy-eigh'
    SCIPY_EIG       = 'scipy-eig'
    SCIPY_EIGS      = 'scipy-eigs'
    LOBPCG          = 'lobpcg'
    JAX_EIGH        = 'jax-eigh'
    
    def __str__(self) -> str:       return self.value
    def __repr__(self) -> str:      return self.value

class DiagonalizationEngine:
    r"""
    Modular diagonalization engine for Hamiltonian matrices.
    
    This class provides a unified interface for various eigenvalue solvers,
    automatically selecting the most appropriate method based on problem size
    and characteristics. It stores Krylov basis information for iterative methods
    and provides utilities for basis transformations.
    
    Features:
    ---------
        - Automatic method selection based on matrix size and properties
            - Support for exact, Lanczos, Block Lanczos, Arnoldi, shift-invert methods
            - Support for SciPy methods: scipy-eigh, scipy-eig, scipy-eigs, lobpcg
            - Support for JAX GPU-accelerated methods: jax-eigh
        - Krylov basis storage and transformation utilities
        - Backend support: NumPy, JAX, SciPy
        - Memory-efficient handling of large sparse matrices
        - Verbose logging and progress reporting
    
    Parameters:
    -----------
        method : str, optional
            Diagonalization method: 'auto', 'exact', 'lanczos', 'block_lanczos', 'arnoldi',
                'shift-invert', 'scipy-eigh', 'scipy-eig', 'scipy-eigs', 'lobpcg', 'jax-eigh'
            Default is 'auto' which automatically selects based on matrix properties.
        backend : str, optional
            Computational backend: 'numpy', 'scipy', 'jax'. Default is 'numpy'.
        use_scipy : bool, optional
            Prefer SciPy implementations when available. Default is True.
        verbose : bool, optional
            Enable verbose output. Default is False.
    
    Example:
    --------
        >>> engine          = DiagonalizationEngine(method='auto', verbose=True)
        >>> result          = engine.diagonalize(H, k=10, hermitian=True)
        >>> evals, evecs    = result.eigenvalues, result.eigenvectors
        >>> # Transform a vector from Krylov to original basis
        >>> v_original      = engine.to_original_basis(v_krylov)
        
        >>> # Use JAX for GPU acceleration
        >>> engine_jax      = DiagonalizationEngine(method='jax-eigh')
        >>> result          = engine_jax.diagonalize(H)
    
        >>> # Use LOBPCG with preconditioner
        >>> M               = scipy.sparse.diags([1.0/np.diag(H)])
        >>> result          = engine.diagonalize(H, k=10, method='lobpcg', M=M)
    """
    
    def __init__(self,
                method         : DiagonalizationMethods             = DiagonalizationMethods.AUTO,
                backend        : Literal['numpy', 'scipy', 'jax']   = 'numpy',
                use_scipy      : bool                               = True,
                verbose        : bool                               = False,
                logger         : Optional[Callable[[str], Any]]     = None
                ):
        """Initialize the diagonalization engine."""
        
        if not EIGEN_SOLVERS_AVAILABLE:
            raise ImportError("Eigenvalue solvers not available. Ensure QES.general_python.algebra.eigen module is properly installed.")
        
        self.method         = method
        self.backend        = backend
        self.use_scipy      = use_scipy
        self.verbose        = verbose
        self.logger         = logger if logger is not None else logging.getLogger(__name__)
        
        # Storage for diagonalization results and basis information
        self._result        : Optional[EigenResult] = None
        self._krylov_basis  : Optional[NDArray]     = None
        self._method_used   : Optional[str]         = None
        self._n             : Optional[int]         = None  # Original dimension
        self._k             : Optional[int]         = None  # Number of eigenvalues computed

    # ------------------------------------------------------------------------------------
    #! Main Diagonalization Method
    # ------------------------------------------------------------------------------------
    
    def diagonalize(self,
                    A               : Optional[NDArray]                         = None,
                    matvec          : Optional[Callable[[NDArray], NDArray]]    = None,
                    n               : Optional[int]                             = None,
                    k               : Optional[int]                             = None,
                    hermitian       : bool                                      = True,
                    which           : Union[str, Literal['smallest', 'largest', 'both']] = 'smallest',
                    store_basis     : bool                                      = True,
                    **kwargs)       -> EigenResult:
        r"""
        Diagonalize a matrix using the specified or auto-selected method.
        
        Parameters:
        -----------
            A : ndarray or sparse matrix, optional
                Matrix to diagonalize. Either A or matvec must be provided.
                The matrix can be dense or sparse.
            matvec : callable, optional
                Matrix-vector product function. Either A or matvec must be provided.
                The function should take a vector and return the matrix-vector product.
            n : int, optional
                Dimension of the problem (required if matvec provided without A).
                Must match the number of rows/columns of A if provided.
            k : int, optional
                Number of eigenvalues to compute. If None:
                - For 'exact': all eigenvalues computed
                - For iterative methods: default is 6
            hermitian : bool, optional
                Whether matrix is symmetric/Hermitian. Default is True.
            which : str, optional
                Which eigenvalues to find:
                - For Lanczos/Block Lanczos: 'smallest', 'largest', 'both'
                - For Arnoldi: 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
                Default is 'smallest'.
            store_basis : bool, optional
                Store Krylov basis for iterative methods. Default is True.
            **kwargs : dict
                Additional arguments passed to the solver:
                - tol               : float - Convergence tolerance
                - max_iter          : int   - Maximum iterations
                - block_size        : int   - Block size for Block Lanczos
                - reorthogonalize   : bool  - Enable reorthogonalization

        Returns:
        --------
            EigenResult containing:
                - eigenvalues       : ndarray   - Computed eigenvalues
                - eigenvectors      : ndarray   - Computed eigenvectors
                - iterations        : int       - Number of iterations (for iterative methods)
                - converged         : bool      - Convergence status
                - residual_norms    : ndarray   - Residual norms (if available)

        Example:
        --------
            >>> # Exact diagonalization
            >>> result = engine.diagonalize(A, hermitian=True)
            
            >>> # Lanczos for 10 smallest eigenvalues
            >>> result = engine.diagonalize(A, k=10, which='smallest')
            
            >>> # Block Lanczos with custom parameters
            >>> result = engine.diagonalize(A, k=20, method='block_lanczos', 
            >>>                             block_size=5, tol=1e-8)
        """
        
        # Determine dimension
        if A is not None:
            self._n = A.shape[0]
        elif n is not None:
            self._n = n
        else:
            raise ValueError("Must provide either matrix A or dimension n")
        
        # Set number of eigenvalues to compute
        # Auto-select method if needed
        self._k     = k if k is not None else (self._n if self.method == DiagonalizationMethods.EXACT else 6)
        method      = self.method
        if method == 'auto':
            method  = decide_method(self._n, k=k, hermitian=hermitian)
            if self.verbose:
                self.logger.info(f"Auto-selected diagonalization method: {method}")

        # Store method used
        self._method_used = method
        
        # Call the appropriate solver via factory
        try:
            self._result = choose_eigensolver(
                method      = method,
                A           = A,
                matvec      = matvec,
                n           = self._n,
                k           = self._k,
                hermitian   = hermitian,
                which       = which,
                backend     = self.backend,
                use_scipy   = self.use_scipy,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Diagonalization failed with method '{method}': {e}") from e
        
        # Store Krylov basis if requested and applicable
        if store_basis and method in ['lanczos', 'block_lanczos', 'arnoldi']:
            self._extract_krylov_basis(A, matvec)
        
        if self.verbose:
            self.logger.info(f"Diagonalization completed using {method}")
            self.logger.info(f"  Computed {len(self._result.eigenvalues)} eigenvalues")
            if self._result.converged:
                self.logger.info(f"  Converged in {self._result.iterations} iterations")
            else:
                self.logger.warning(f"  Warning: Did not converge after {self._result.iterations} iterations")

        return self._result
    
    # ------------------------------------------------------------------------------------
    #! Krylov Basis Management
    # ------------------------------------------------------------------------------------
    
    def _extract_krylov_basis(self,
                             A      : Optional[NDArray],
                             matvec : Optional[Callable]):
        r"""
        Extract and store the Krylov basis from eigenvectors.
        
        For iterative methods, the eigenvectors are expressed in the Krylov subspace.
        We reconstruct the Krylov basis V from the eigenvectors and eigenvalues.
        
        Note: This is an approximation. For exact Krylov basis, the solver would
        need to return it directly. Here we use the eigenvectors as a proxy.
        #!TODO: Improve by modifying solvers to return Krylov basis directly.
        """
        
        if self._result is None or self._result.eigenvectors is None:
            return
        
        # For iterative methods, eigenvectors form an orthonormal basis
        # In the full implementation, we'd get V from the solver directly
        # For now, we use the eigenvectors as the basis
        self._krylov_basis = self._result.subspacevectors if hasattr(self._result, 'subspacevectors') else None
        
    def has_krylov_basis(self) -> bool:
        """
        Check if a Krylov basis is available.
        
        Returns:
        --------
            bool : True if Krylov basis is stored, False otherwise.
        """
        return self._krylov_basis is not None
    
    def get_krylov_basis(self) -> Optional[NDArray]:
        """
        Get the stored Krylov basis.
        
        Returns:
        --------
            ndarray or None : Krylov basis matrix V, or None if not available.
        """
        return self._krylov_basis
    
    def to_original_basis(self, vec: NDArray) -> NDArray:
        """
        Transform a vector from Krylov/computational basis to original basis.
        
        For exact diagonalization:
            - This is a no-op (returns the same vector).
        For iterative methods with Krylov basis V:
            - Returns V @ vec
        
        Parameters:
        -----------
            vec : ndarray
                Vector in Krylov/computational basis. Can be:
                - 1D array of shape (k,)    - single vector
                - 2D array of shape (k, m)  - multiple vectors
        
        Returns:
        --------
            ndarray: Vector(s) in original basis.
        
        Example:
        --------
            >>> # Get a Ritz vector (in Krylov basis)
            >>> ritz_vec        = engine.get_result().eigenvectors[:, 0]
            >>> # Transform to original basis
            >>> original_vec    = engine.to_original_basis(ritz_vec)
        """

        if self._method_used == 'exact':    # For exact diagonalization, eigenvectors are already in original basis
            return vec
        
        if self._krylov_basis is None:      # If no Krylov basis stored, assume already in original basis
            return vec
        
        #! Transform: original = V @ krylov
        if vec.ndim == 1:
            return self._krylov_basis @ vec
        else:
            return self._krylov_basis @ vec
    
    def to_krylov_basis(self, vec: NDArray) -> NDArray:
        """
        Transform a vector from original basis to Krylov/computational basis.
        For exact diagonalization:
            - This is a no-op (returns the same vector).
        For iterative methods with Krylov basis V:
            - Returns V.T @ vec (or V.H @ vec for complex)
        
        Parameters:
        -----------
            vec : ndarray
                Vector in original basis. Can be:
                - 1D array of shape (n,) - single vector
                - 2D array of shape (n, m) - multiple vectors
        
        Returns:
        --------
            ndarray : Vector(s) in Krylov basis.
        
        Example:
        --------
            >>> # Project a state onto the Krylov subspace
            >>> state = np.random.randn(n)
            >>> krylov_coeffs = engine.to_krylov_basis(state)
        """
        
        if self._method_used == 'exact':    # For exact diagonalization, no transformation needed
            return vec
        
        if self._krylov_basis is None:
            raise ValueError("No Krylov basis available for transformation")
        
        # Transform: krylov = V.H @ original (or V.T for real)
        V = self._krylov_basis
        if np.iscomplexobj(V):
            if vec.ndim == 1:
                return V.conj().T @ vec
            else:
                return V.conj().T @ vec
        else:
            if vec.ndim == 1:
                return V.T @ vec
            else:
                return V.T @ vec
    
    def get_basis_transform(self) -> Optional[NDArray]:
        """
        Get the transformation matrix from Krylov to original basis.
        
        Returns:
        --------
            ndarray or None : 
                Transformation matrix V (shape n x k), or None if not applicable.
                To transform from Krylov to original: v_original = V @ v_krylov
        """
        return self._krylov_basis
    
    # ------------------------------------------------------------------------------------
    #! Result Access
    # ------------------------------------------------------------------------------------
    
    def get_result(self) -> Optional[EigenResult]:
        """
        Get the diagonalization result.
        
        Returns:
        --------
            EigenResult or None : Result object with eigenvalues, eigenvectors, etc.
        """
        return self._result
    
    def get_eigenvalues(self) -> Optional[NDArray]:
        """
        Get computed eigenvalues.
        
        Returns:
        --------
            ndarray or None : Eigenvalues array.
        """
        return self._result.eigenvalues if self._result else None
    
    def get_eigenvectors(self) -> Optional[NDArray]:
        """
        Get computed eigenvectors.
        
        Returns:
        --------
            ndarray or None : Eigenvectors matrix (each column is an eigenvector).
        """
        return self._result.eigenvectors if self._result else None
    
    def get_method_used(self) -> Optional[str]:
        """
        Get the diagonalization method that was used.
        
        Returns:
        --------
            str or None : Method name ('exact', 'lanczos', 'block_lanczos', 'arnoldi').
        """
        return self._method_used
    
    def converged(self) -> bool:
        """
        Check if the diagonalization converged.
        
        Returns:
        --------
            bool : True if converged, False otherwise (or if not applicable).
        """
        if self._result is None:
            return False
        return self._result.converged if hasattr(self._result, 'converged') else True
    
    def get_residual_norms(self) -> Optional[NDArray]:
        """
        Get residual norms for computed eigenpairs.
        
        Returns:
        --------
            ndarray or None : Residual norms, or None if not available.
        """
        if self._result is None:
            return None
        return self._result.residual_norms if hasattr(self._result, 'residual_norms') else None
    
    # ------------------------------------------------------------------------------------
    #! Utility Methods
    # ------------------------------------------------------------------------------------
    
    def reset(self):
        """Reset the engine, clearing all stored results and basis information."""
        self._result        = None
        self._krylov_basis  = None
        self._method_used   = None
        self._n             = None
        self._k             = None
    
    def __repr__(self) -> str:
        """String representation of the engine."""
        
        status = "Not yet used"
        if self._krylov_basis is not None:
            status = f"Krylov basis stored (shape: {self._krylov_basis.shape})"
            
        if self._method_used:
            status = f"Last used: {self._method_used}"
            if self._result:
                status += f" ({len(self._result.eigenvalues)} eigenvalues)"
                
                if self._result.converged:
                    status += ", converged"
                else:
                    status += ", not converged"
        
        return f"DiagonalizationEngine(method={self.method}, backend={self.backend}, {status})"

    def __str__(self) -> str:
        return f'DiagonalizationEngine(method={self.method}, backend={self.backend})'

    # ------------------------------------------------------------------------------------
    #! Properties
    # ------------------------------------------------------------------------------------
    
    @property
    def method_used(self) -> Optional[str]:
        """Get the diagonalization method that was used."""
        return self._method_used
    
    @property
    def krylov_basis(self) -> Optional[NDArray]:
        """Get the stored Krylov basis."""
        return self._krylov_basis

    @property
    def result(self) -> Optional[EigenResult]:
        """Get the result of the diagonalization."""
        return self._result
    
    @property
    def dimension(self) -> Optional[int]:
        """Get the dimension of the problem."""
        return self._n

    @property
    def k(self) -> Optional[int]:
        """Get the number of desired eigenvalues."""
        return self._k

# ----------------------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------------------
