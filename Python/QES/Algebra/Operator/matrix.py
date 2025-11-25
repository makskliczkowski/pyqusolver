'''
We move most of the definitions of matrices to this file. Operators will derive
from it and be able to create matrices as needed.

-------------------------------------------------------------------------------
File            : Algebra/Operator/matrix.py
Author          : Maksymilian Kliczkowski
Date            : 2025-11-24
Copyright       : (c) 2025
License         : MIT
-------------------------------------------------------------------------------
'''

from __future__ import annotations
import numpy as np

import time
import numba
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import TYPE_CHECKING, Optional, Union, Callable, Any

try:
    from QES.general_python.algebra.utils import get_backend, JAX_AVAILABLE, Array
    if TYPE_CHECKING:
        from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
        from QES.general_python.common.flog import Logger
        import QES.Algebra.Hamil.hamil_diag_helpers as diag_helpers
        
except ImportError:
    raise ImportError("QES.general_python.algebra.utils could not be imported. Ensure QES is properly installed.")

# -------------------------------

if JAX_AVAILABLE:
    import jax
    import jax.lax as lax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO, CSR
else:
    jax                     = None
    jnp                     = None
    lax                     = None
    BCOO                    = None
    CSR                     = None
    local_energy_jax_wrap   = None

# -------------------------------

class DummyVector:
    """
    Constant vector of length `ns` with all entries == `val`.
    A thin dummy so scalar couplings can be broadcast like arrays.
    """

    #! construction
    def __init__(self, val, ns: int | None = None, *, backend=None):
        """
        Initialize the object with the given value, number of sites, and backend.
        Args:
            val:
                The value to be assigned to the object.
            ns (int, optional):
                The number of sites. If not provided, defaults to 1.
            backend (optional):
                The backend module to use. If not provided, defaults to 'numpy'.
        """
        
        self.val      = val
        self.ns       = int(ns) if ns is not None else 1
        self._backend = backend or __import__("numpy")

    # ---------------------------------------------------------------------------
    
    @property
    def dtype(self):
        return getattr(self.val, "dtype", type(self.val))

    def astype(self, dtype, copy: bool = False, *, backend=None):
        """
        Return a `DummyVector` with the same length but `val` cast to `dtype`.

        Parameters
        ----------
        dtype : str | numpy.dtype | jax.numpy.dtype | type
            Desired element dtype.
        copy  : bool, default False
            If False and the dtype is unchanged -> return self.
            If True  -> always return a *new* `DummyVector`.
        backend : optional
            Backend module (`numpy`, `jax.numpy`, …) controlling the cast.
            If None, use the instance's backend.

        Notes
        -----
        *The method never materialises a full array*, so it's O(1) in memory.
        """
        backend = backend or self._backend
        tgt_dt  = distinguish_type(dtype)

        if (backend.iscomplexobj(self.val) and not backend.iscomplexobj(dtype)):
            # If we're casting from complex to real, take the real part
            self.val = backend.real(self.val)
        elif (not backend.iscomplexobj(self.val) and backend.iscomplexobj(dtype)):
            # If we're casting from real to complex, add a zero imaginary part
            self.val = backend.asarray(self.val, dtype=backend.complex128)

        # fast path: nothing to change
        if not copy and tgt_dt == self.dtype:
            return self

        new_val = backend.asarray(self.val, dtype=tgt_dt).item()
        return DummyVector(new_val, ns=self.ns, backend=backend)

    # ---------------------------------------------------------------------------
    
    def __array__(self, dtype=None):
        return self._backend.full(self.ns, self.val, dtype=dtype)

    def __array_priority__(self):
        return 100.0

    # ---------------------------------------------------------------------------
    
    def __repr__(self):
        return f"DummyVector(val={self.val!r}, ns={self.ns})"

    def __str__(self):
        return f"[{self.val}] * {self.ns}"

    # ---------------------------------------------------------------------------
    
    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            length = len(range(*idx.indices(self.ns)))
            return DummyVector(self.val, ns=length, backend=self._backend)
        elif isinstance(idx, int):
            return self.val
        else:
            raise TypeError("DummyVector supports int or slice indices only.")

    def __iter__(self):
        yield from (self.val for _ in range(self.ns))

    # ---------------------------------------------------------------------------
    
    def _binary(self, other, op):
        if isinstance(other, DummyVector):
            if self.ns != other.ns:
                raise ValueError("DummyVector: size mismatch")
            other_val = other.val
        else:  # assume scalar
            other_val = other
        return DummyVector(op(self.val, other_val), ns=self.ns, backend=self._backend)

    def __add__(self, other):       return self._binary(other, lambda a, b: a + b)
    def __radd__(self, other):      return self.__add__(other)
    def __sub__(self, other):       return self._binary(other, lambda a, b: a - b)
    def __rsub__(self, other):      return self._binary(other, lambda a, b: b - a)
    def __mul__(self, other):       return self._binary(other, lambda a, b: a * b)
    def __rmul__(self, other):      return self.__mul__(other)
    def __truediv__(self, other):   return self._binary(other, lambda a, b: a / b)
    def __rtruediv__(self, other):  return self._binary(other, lambda a, b: b / a)
    # (add more as needed)

    # ---------------------------------------------------------------------------
    
    def __eq__(self, other):
        return (
            isinstance(other, DummyVector)
            and self.ns == other.ns
            and self.val == other.val
        )

    def __hash__(self):
        return hash((self.val, self.ns))
    
    def to_array(self, dtype=None, backend=None):
        """
        Convert the DummyVector to a numpy array.
        """
        backend = backend if backend is not None else __import__('numpy')
        return backend.full(self.ns, self.val, dtype=dtype)
    
##################################################################################
#! General Matrix Class
##################################################################################

class GeneralMatrix(spla.LinearOperator):
    ''' General matrix representation for operators. '''
    
    _ERR_MATRIX_NOT_BUILT        = "Matrix representation has not been built. Call build() first."
    _ERR_INVALID_BACKEND         = "Invalid backend specified."
    _ERR_UNSUPPORTED_OPERATION   = "The requested operation is not supported for this matrix type."
    
    def __init__(self, 
                shape           : tuple[int, int],
                *,
                matvec          : Optional[Callable[[np.ndarray], np.ndarray]]  = None,
                is_sparse       : bool                                          = True,
                backend         : str                                           = 'default',
                logger          : Optional[Logger]                              = None,
                seed            : Optional[int]                                 = None,
                dtype           : Optional[Union[str, np.dtype]]                = None,
            ) -> None:
        
        (self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k)) = GeneralMatrix._set_backend(backend, seed)        

        self._dim               = shape[0]
        self._is_jax            = JAX_AVAILABLE and self._backend != np
        self._is_numpy          = not self._is_jax
        self._is_sparse         = is_sparse
        self._shape             = shape
        
        self._dtypeint          = self._backend.int64
        self._dtype             = dtype if dtype is not None else self._backend.float64
        
        self._logger: Logger    = logger
        
        self._name              = "GeneralMatrix"
        self._handle_dtype(dtype)
        
        # This sets self.shape and self.dtype for SciPy's internals
        super().__init__(dtype=self._dtype, shape=shape)
        
        # If a fast matvec kernel (Numba wrapper) is provided, store it.
        self._custom_matvec     = matvec
        
        # Storage - cache if built
        self._is_built          = False
        self._matrix            : Optional[Union[np.ndarray, Any]] = None # Could be jax array or scipy sparse matrix
        self._eigvecs           : Optional[Union[np.ndarray, Any]] = None
        self._eigval            : Optional[np.ndarray]             = None
        self._krylov            : Optional[Any]                    = None
        self._max_local_ch      : Optional[int]                    = 1
        self._max_local_ch_o    : Optional[int]                    = 1
        
        # Diagonalization engine 
        self._diag_engine       : Optional[Any]                    = None
        self._diag_method       : Optional[str]                    = 'exact'
        
        # Basis tracking
        # These attributes enable flexible basis transformations for any Hamiltonian subclass
        # (Quadratic, ManyBody, etc.) and any basis type (REAL, KSPACE, FOCK, etc.)
        self._original_basis    : Optional[Any] = None
        self._current_basis     : Optional[Any] = None
        self._basis_metadata    : dict = {}
        self._is_transformed    = False
        self._transformed_grid  : Optional[Any] = None
        self._symmetry_info     : dict = {}
        
    # -------------------------------------------------------
    # SciPy LinearOperator Interface Implementation
    # -------------------------------------------------------
    
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        The method SciPy calls when doing A @ x.
        """
        
        if self._custom_matvec is not None:
            
            if self._is_numpy:
                x_in = np.ascontiguousarray(x, dtype=self._dtype)
                return self._custom_matvec(x_in)
            else:
                return self._custom_matvec(x)
        else:
            raise NotImplementedError("Matrix-vector product not defined. Call build() or provide matvec.")

    def _adjoint(self):
        """ One can implement Hermitian conjugate logic. """
        # For real symmetric operators, self.H == self.H
        return self

    # -------------------------------------------------------
    #! VIRTUAL
    # -------------------------------------------------------
    
    @staticmethod
    def _set_backend(backend: str, seed = None):
        '''
        Get the backend, scipy, and random number generator for the backend.
        
        Parameters:
        -----------
            backend (str):
                The backend to use.
            seed (int, optional):
                The seed for the random number generator.
        
        Returns:
            tuple : 
                The backend, scipy, and random number generator for the backend.
                
        Examples:
        ---------
            >>> GeneralMatrix._set_backend('np')
            ('np', <module 'numpy' from '...'>, None, (None, None))
            >>> GeneralMatrix._set_backend('jax', seed=42)
            ('jax', <module 'jax.numpy' from '...'>, None, (<jax.random.PRNGKey object at ...>, None))
        '''
        if isinstance(backend, str):
            bck = get_backend(backend, scipy=True, random=True, seed=seed)
            if isinstance(bck, tuple):
                _backend, _backend_sp = bck[0], bck[1]
                if isinstance(bck[2], tuple):
                    _rng, _rng_k = bck[2][0], bck[2][1]
                else:
                    _rng, _rng_k = bck[2], None
            else:
                _backend, _backend_sp   = bck, None
                _rng, _rng_k            = None, None
            return backend, _backend, _backend_sp, (_rng, _rng_k)
        if JAX_AVAILABLE and backend == 'default':
            _backendstr = 'jax'
        else:
            _backendstr = 'np'
        return GeneralMatrix._set_backend(_backendstr)    
    
    def _handle_dtype(self, dtype: Optional[Union[str, np.dtype]]) -> np.dtype:
        ''' Handle dtype selection based on backend and user input. '''
        if dtype is not None:
            self._dtype = dtype
            
            if self._is_jax:
                self._iscpx = jnp.issubdtype(jnp.dtype(self._dtype), jnp.complexfloating)
            elif self._is_numpy:
                self._iscpx = np.issubdtype(np.dtype(self._dtype), np.complexfloating)
        else:
            self._dtype = self._backend.float64
            self._iscpx = False
        
    def _log(self, msg : str, log : str = 'info', lvl : int = 0, color : str = "white"):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (str) : The logging level. Default is 'info'.
            lvl (int) : The level of the message.
        """
        if self._logger is not None:
            msg = f"[{self.name}] {msg}"
            self._logger.info(msg, lvl=lvl, log=log, color=color)
            
    # -------------------------------------------------------
    #! BUILD        
    # -------------------------------------------------------
    
    def build(self) -> None:
        ''' Build the matrix representation. '''
        raise NotImplementedError("The build method must be implemented in subclasses of GeneralMatrix.")
    
    # -------------------------------------------------------
    #! CLEAR
    # -------------------------------------------------------
    
    def clear(self) -> None:
        ''' Clear the built matrix and eigen decomposition. '''
        self._log("Clearing the built matrix and eigen decomposition...", lvl=2, log='debug')
        self._is_built      = False
        self._matrix        = None
        self._eigvecs       = None
        self._eigval        = None
        self._krylov        = None
        self._diag_engine   = None
        self._diag_method   = None
        
    # -------------------------------------------------------
    #! DIAGONALIZE
    # -------------------------------------------------------
    
    def diagonalize(self, verbose: bool = False, **kwargs) -> None:
        ''' Diagonalize the matrix representation. '''
        raise NotImplementedError("The diagonalize method must be implemented in subclasses of GeneralMatrix.")
    
    # -------------------------------------------------------
    #! Properties
    # -------------------------------------------------------
    
    @property
    def name(self) -> str:              return self._name
    @name.setter
    def name(self, value: str) -> None: self._name = value
    
    @property
    def dtype(self):                    return self._dtype
    @property
    def dtypeint(self):                 return self._dtypeint
    @property
    def inttype(self):                  return self._dtypeint
    
    @property
    def backend(self):                  return self._backendstr
    
    @property
    def sparse(self):                   return self._is_sparse
    def is_sparse(self):                return self._is_sparse
    
    @property
    def max_local_changes(self):        return self._max_local_ch
    @property
    def max_local(self):                return self.max_local_changes    
    @property
    def max_operator_changes(self):     return self._max_local_ch_o
    @property
    def max_operator(self):             return self.max_operator_changes

    @property
    def energies(self):                 return self._eig_val
    @property
    def eigenvalues(self):              return self._eig_val
    @property
    def eig_val(self):                  return self._eig_val
    @property
    def eigenvals(self):                return self._eig_val
    @property
    def eig_vals(self):                 return self._eig_val
    @property
    def eigen_vals(self):               return self._eig_val
    @property 
    def energies(self):                 return self._eig_val

    @property
    def eig_vec(self):                  return self._eig_vec
    @property
    def eigenvectors(self):             return self._eig_vec
    @property
    def eigenvecs(self):                return self._eig_vec
    @property
    def krylov(self):                   return self._krylov

    @property
    def matrix(self)                    -> Union[np.ndarray, Any]:
        ''' Get the Hamiltonian matrix representation. '''
        if not self._is_built or self._matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)
        return self._matrix

    @property
    def diag(self)                      -> Optional[Union[np.ndarray, Any]]:
        '''
        Returns the diagonal of the matrix.
        Distinguish between JAX and NumPy/SciPy. 
        '''
        
        target = self._matrix
        if target is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)
    
        if JAX_AVAILABLE and self._backend != np:
            if isinstance(target, BCOO):
                return target.diagonal()
            elif jnp is not None and isinstance(target, jnp.ndarray):
                return jnp.diag(target)
            else:
                # dunnno what to do here
                return None
        elif sp.sparse.issparse(target):
            return target.diagonal()
        elif isinstance(target, np.ndarray):
            return target.diagonal()
        else:
            return None

    @property
    def mat_memory(self)                -> float:
        ''' Estimate the memory used by the Hamiltonian matrix in bytes. '''
        
        matrix_to_check = self.hamil if (self._is_manybody) else self.hamil_sp
        if matrix_to_check is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)
        
        # Dense matrix: use nbytes if available, otherwise compute from shape.
        # self._log(f"Checking the memory used by the Hamiltonian matrix of type {type(self._hamil)}", lvl=1)
        memory = 0
        
        if not self._is_sparse:
            if hasattr(matrix_to_check, "nbytes"):
                return matrix_to_check.nbytes
            else:
                return int(np.prod(matrix_to_check.shape)) * matrix_to_check.dtype.itemsize
        else:
            self._log("It is not a dense matrix...", lvl=2, log='debug')
            
            # Sparse matrix:
            # For NumPy (or when JAX is unavailable) we assume a scipy sparse matrix (e.g. CSR)
            if self._is_numpy:
                memory = 0
                for attr in ('data', 'indices', 'indptr'):
                    if hasattr(matrix_to_check, attr):
                        arr = getattr(matrix_to_check, attr)
                        if hasattr(arr, 'nbytes'):
                            memory += arr.nbytes
                        else:
                            memory += int(np.prod(arr.shape)) * arr.dtype.itemsize
            elif self._is_jax:
                # For JAX sparse matrices (e.g. BCOO), we assume they have data and indices attributes.
                data_arr        = matrix_to_check.data
                indices_arr     = matrix_to_check.indices
                if hasattr(data_arr, 'nbytes'):
                    data_bytes  = data_arr.nbytes
                else:
                    data_bytes  = int(np.prod(data_arr.shape)) * data_arr.dtype.itemsize
                if hasattr(indices_arr, 'nbytes'):
                    indices_bytes = indices_arr.nbytes
                else:
                    indices_bytes = int(np.prod(indices_arr.shape)) * indices_arr.dtype.itemsize
                memory = data_bytes + indices_bytes
            else:
                return 0 # Unknown type, return 0
        return memory # for BdG Hamiltonian        
        
    @property
    def eigvec_memory(self)             -> float:
        ''' Estimate the memory used by the eigenvectors in bytes. '''
        if self._eigvecs is None:
            return 0
        if hasattr(self._eigvecs, "nbytes"):
            return self._eigvecs.nbytes
        else:
            return int(np.prod(self._eigvecs.shape)) * self._eigvecs.dtype.itemsize
    
    @property
    def eigval_memory(self)             -> float:
        ''' Estimate the memory used by the eigenvalues in bytes. '''
        if self._eigval is None:
            return 0
        if hasattr(self._eigval, "nbytes"):
            return self._eigval.nbytes
        else:
            return int(np.prod(self._eigval.shape)) * self._eigval.dtype.itemsize    
        
    @property
    def mat_memory_mb(self):            return self.mat_memory      / (1024 ** 2)    
    @property
    def eigvec_memory_mb(self):         return self.eigvec_memory   / (1024 ** 2)    
    @property
    def eigval_memory_mb(self):         return self.eigval_memory   / (1024 ** 2)    
    @property
    def mat_memory_gb(self):            return self.mat_memory      / (1024 ** 3)
    @property
    def eigvec_memory_gb(self):         return self.eigvec_memory   / (1024 ** 3)
    @property
    def eigval_memory_gb(self):         return self.eigval_memory   / (1024 ** 3)
    
    @property
    def memory(self):                   return self.mat_memory + self.eigvec_memory + self.eigval_memory
    @property
    def memory_mb(self):                return self.memory          / (1024 ** 2)
    @property
    def memory_gb(self):                return self.memory          / (1024 ** 3)
    
    # -------------------------------------------------------
    #! Basis 
    # -------------------------------------------------------
    
    def has_krylov_basis(self) -> bool:
        """
        Check if a Krylov basis is available from the last diagonalization.
        
        The Krylov basis is available when using iterative methods (Lanczos,
        Block Lanczos, Arnoldi, Shift-Invert) with store_basis=True.
        
        Returns:
        --------
            bool : True if Krylov basis is available, False otherwise.
        
        Example:
        --------
            >>> hamil.diagonalize(method='lanczos', k=10, store_basis=True)
            >>> if hamil.has_krylov_basis():
            ...     print("Krylov basis available for transformations")
        """
        return diag_helpers.has_krylov_basis(self._diag_engine, self._krylov)
    
    def get_krylov_basis(self) -> Optional[Array]:
        """
        Get the Krylov basis from the last diagonalization.
        This corespons to the matrix V whose columns are the Krylov basis vectors.
        
        To be available, the Krylov basis must have been stored during
        diagonalization (store_basis=True) using an iterative method.
        
        Returns:
        --------
            ndarray or None : 
                Krylov basis matrix V (shape n x k), or None if not available.
        
        Example:
        --------
            >>> V = hamil.get_krylov_basis()
            >>> if V is not None:
            ...     print(f"Krylov basis shape: {V.shape}")
            
            >>> # Manual transformation
            >>> v_krylov    = np.array([1, 0, 0, ...])  # First Ritz vector
            >>> v_original  = V @ v_krylov              # Transform to original basis
            >>> print(f"Original vector: {v_original}")
            
            >>> # Matrix reconstruction
            >>> H_reconstructed = V @ np.diag(hamil.energies) @ V.T
            >>> print(f"Reconstructed Hamiltonian shape: {H_reconstructed.shape}")
        """
        return diag_helpers.get_krylov_basis(self._diag_engine, self._krylov)
    
    def to_original_basis(self, vec: Array) -> Array:
        """
        Transform a vector from Krylov/computational basis to original basis.
        
        For exact diagonalization, this is a no-op (returns the same vector).
        For iterative methods with Krylov basis V: returns V @ vec.
        
        This is useful when working with Ritz vectors or when projecting
        computations done in the reduced Krylov subspace back to the full
        Hilbert space.
        
        Parameters:
        -----------
            vec : ndarray
                Vector(s) in Krylov/computational basis.
                - 1D array of shape (k,): single vector
                - 2D array of shape (k, m): m vectors as columns
        
        Returns:
        --------
            ndarray : Vector(s) in original Hilbert space basis.
        
        Examples:
        ---------
            >>> # Diagonalize using Lanczos
            >>> hamil.diagonalize(method='lanczos', k=10)
            >>> 
            >>> # Get first Ritz vector (in Krylov basis)
            >>> ritz_vec = np.zeros(10)
            >>> ritz_vec[0] = 1.0  # First Ritz vector
            >>> 
            >>> # Transform to original basis
            >>> state = hamil.to_original_basis(ritz_vec)
            >>> print(f"State in full Hilbert space: shape {state.shape}")
        
        See Also:
        ---------
            to_krylov_basis : Inverse transformation
            has_krylov_basis : Check if basis is available
        """
        return diag_helpers.to_original_basis(vec, self._diag_engine, self.get_diagonalization_method())

    def to_krylov_basis(self, vec: Array) -> Array:
        """
        Transform a vector from original basis to Krylov/computational basis.
        
        For exact diagonalization, this is a no-op (returns the same vector).
        For iterative methods with Krylov basis V: returns V.H @ vec (or V.T for real).
        
        This is useful for projecting states from the full Hilbert space onto
        the reduced Krylov subspace for efficient computations.
        
        Parameters:
        -----------
            vec : ndarray
                Vector(s) in original Hilbert space basis.
                - 1D array of shape (n,): single vector
                - 2D array of shape (n, m): m vectors as columns
        
        Returns:
        --------
            ndarray : Vector(s) in Krylov basis.
        
        Examples:
        ---------
            >>> # Diagonalize using Lanczos
            >>> hamil.diagonalize(method='lanczos', k=10)
            >>> 
            >>> # Create a random state in full Hilbert space
            >>> state = np.random.randn(hamil.nh)
            >>> state /= np.linalg.norm(state)
            >>> 
            >>> # Project onto Krylov subspace
            >>> krylov_coeffs = hamil.to_krylov_basis(state)
            >>> print(f"Krylov coefficients: shape {krylov_coeffs.shape}")
        
        Raises:
        -------
            ValueError : If no Krylov basis is available.
        
        See Also:
        ---------
            to_original_basis : Inverse transformation
            has_krylov_basis : Check if basis is available
        """
        return diag_helpers.to_krylov_basis(vec, self._diag_engine)

    def get_basis_transform(self) -> Optional[Array]:
        """
        Get the transformation matrix from Krylov to original basis.
        
        Returns the Krylov basis matrix V such that:
            v_original = V @ v_krylov
        
        Returns:
        --------
            ndarray or None : 
                Transformation matrix V (shape n x k), or None if not applicable.
        
        Example:
        --------
            >>> V = hamil.get_basis_transform()
            >>> if V is not None:
            ...     # Manual transformation
            ...     v_krylov = np.array([1, 0, 0, ...])  # First Ritz vector
            ...     v_original = V @ v_krylov
        """
        return diag_helpers.get_basis_transform(self._diag_engine, self._krylov)
    
    def get_diagonalization_method(self) -> Optional[str]:
        """
        Get the diagonalization method used in the last diagonalize() call.
        
        Returns:
        --------
            str or None : 
                Method name ('exact', 'lanczos', 'block_lanczos', 'arnoldi', 'shift-invert'),
                or None if not yet diagonalized.
        
        Example:
        --------
            >>> hamil.diagonalize(method='auto', k=10)
            >>> method = hamil.get_diagonalization_method()
            >>> print(f"Used method: {method}")
        """
        return diag_helpers.get_diagonalization_method(self._diag_engine)
    
    def get_diagonalization_info(self) -> dict:
        """
        Get detailed information about the last diagonalization.
        
        Returns:
        --------
            dict : Dictionary containing:
                - method            : str - Method used
                - converged         : bool - Convergence status
                - iterations        : int - Number of iterations (if applicable)
                - residual_norms    : ndarray - Residual norms (if available)
                - has_krylov_basis  : bool - Whether Krylov basis is available
                - num_eigenvalues   : int - Number of computed eigenvalues
        
        Example:
        --------
            >>> hamil.diagonalize(method='lanczos', k=10)
            >>> info = hamil.get_diagonalization_info()
            >>> print(f"Method: {info['method']}")
            >>> print(f"Converged: {info['converged']}")
        """
        return diag_helpers.get_diagonalization_info(self._diag_engine, self._eig_val, self._krylov)
        
    # -------------------------------------------------------
    
    def to_dense(self):
        '''
        Converts the Hamiltonian matrix to a dense matrix.
        '''
        self._is_sparse = False
        self._log("Converting the Hamiltonian matrix to a dense matrix... Run build...", lvl = 1)
        self.clear()
        
    def to_sparse(self):
        '''
        Converts the Hamiltonian matrix to a sparse matrix.
        '''
        self._is_sparse = True
        self._log("Converting the Hamiltonian matrix to a sparse matrix... Run build...", lvl = 1)
        self.clear()
    
        # ----------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------
    #! FORMATTING
    # -------------------------------------------------------
    
    @staticmethod
    def _fmt_scalar(name, val, prec=1):
        """
        Formats a scalar value with a given name and precision.

        Args:
            name (str):
                The name to display alongside the value.
            val (float):
                The scalar value to format.
            prec (int, optional):
                The number of decimal places to display. Defaults to 1.

        Returns:
            str: A formatted string in the form 'name=value' with the specified precision.
        """
        return f"{name}={val:.{prec}f}"

    @staticmethod
    def _fmt_array(name, arr, prec=1, tol=1e-6):
        """
        Formats a NumPy array or DummyVector into a concise string representation for display.
        Parameters:
            name (str):
                The name to prefix the formatted output.
            arr (array-like or DummyVector):
                The array or vector to format.
            prec (int, optional):
                Number of decimal places for min/max values. Default is 1.
            tol (float, optional):
                Tolerance for determining if all elements are equal. Default is 1e-6.
        Returns:
            str: A formatted string representing the array:
                - If arr is a DummyVector, returns a scalar format.
                - If arr is empty, returns 'name=[]'.
                - If all elements are (approximately) equal, returns a scalar format.
                - Otherwise, returns 'name[min=..., max=...]' with specified precision.
        """
        if isinstance(arr, DummyVector):
            return Hamiltonian._fmt_scalar(name, arr[0])
        
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return f"{name}=[]"
        if np.allclose(arr, arr.flat[0], atol=tol, rtol=0):
            return Hamiltonian._fmt_scalar(name, float(arr.flat[0]), prec=prec)
        return f"{name}[min={arr.min():.{prec}f},max={arr.max():.{prec}f}]"

    @staticmethod
    def fmt(name, value, prec=1):
        """Choose scalar vs array formatter."""
        return Hamiltonian._fmt_scalar(name, value, prec=prec) if np.isscalar(value) else Hamiltonian._fmt_array(name, value, prec=prec)
        
# -------------------------------------------------------
#! EOF
# -------------------------------------------------------