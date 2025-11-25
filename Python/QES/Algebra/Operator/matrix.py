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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Union, Callable, Any

try:
    from QES.general_python.algebra.utils import get_backend, JAX_AVAILABLE, Array
    from QES.general_python.common.flog import Logger
    from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine
    import QES.Algebra.Hamil.hamil_diag_helpers as diag_helpers

except ImportError as exc:
    raise ImportError("QES.general_python.algebra.utils could not be imported. Ensure QES is properly installed.") from exc

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
    """Generic linear-algebra helper providing matrix storage, diagonalization, and logging."""

    _ERR_MATRIX_NOT_BUILT        = "Matrix representation has not been built. Call build() first."
    _ERR_INVALID_BACKEND         = "Invalid backend specified."
    _ERR_UNSUPPORTED_OPERATION   = "The requested operation is not supported for this matrix type."

    def __init__(self,
                 shape              : Optional[tuple[int, int]]                         = None,
                 *,
                 matvec             : Optional[Callable[[np.ndarray], np.ndarray]]      = None,
                 is_sparse          : bool                                              = True,
                 backend            : str                                               = 'default',
                 backend_components : Optional[tuple[Any, Any, Any, tuple[Any, Any]]]   = None,
                 logger             : Optional[Logger]                                  = None,
                 seed               : Optional[int]                                     = None,
                 dtype              : Optional[Union[str, np.dtype]]                    = None) -> None:

        self._shape = tuple(shape) if shape is not None else (0, 0)
        self._dim   = self._shape[0]

        if backend_components is not None:
            (self._backendstr,
             self._backend,
             self._backend_sp,
             (self._rng, self._rng_k)) = backend_components
        else:
            (self._backendstr,
             self._backend,
             self._backend_sp,
             (self._rng, self._rng_k)) = GeneralMatrix._set_backend(backend, seed)

        self._is_jax       = JAX_AVAILABLE and self._backend is not np
        self._is_numpy     = not self._is_jax
        self._is_sparse    = is_sparse
        self._dtypeint     = self._backend.int64
        self._logger       = logger
        self._name         = "GeneralMatrix"
        self._custom_matvec= matvec

        self._matrix       : Optional[Union[np.ndarray, Any]] = None
        self._is_built                                              = False
        self._eig_vec           : Optional[Union[np.ndarray, Any]]  = None
        self._eig_val           : Optional[Union[np.ndarray, Any]]  = None
        self._krylov            : Optional[Any]                     = None
        self._max_local_ch = 1
        self._max_local_ch_o = 1

        self._diag_engine       : Optional[DiagonalizationEngine]   = None
        self._diag_method       : str                               = 'exact'

        self._original_basis    : Optional[Any]                     = None
        self._current_basis     : Optional[Any]                     = None
        self._basis_metadata    : dict                              = {}
        self._is_transformed    : bool                              = False
        self._transformed_grid  : Optional[Any]                     = None
        self._symmetry_info     : dict                              = {}

        self._handle_dtype(dtype)
        super().__init__(dtype=self._dtype, shape=self._shape)

    # -------------------------------------------------------
    # SciPy LinearOperator Interface
    # -------------------------------------------------------

    def _matvec_context(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Arguments passed to matvec closures. Subclasses can override."""
        return (), {}

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if self._custom_matvec is not None:
            if self._is_numpy:
                x_in = np.ascontiguousarray(x, dtype=self._dtype)
                return self._custom_matvec(x_in)
            return self._custom_matvec(x)

        matvec_impl = getattr(self, "matvec", None)
        if matvec_impl is None:
            raise NotImplementedError("Matrix-vector product not defined. Provide a matvec implementation.")

        args, kwargs = self._matvec_context()
        return matvec_impl(x, *args, **kwargs)

    def _adjoint(self):
        return self

    # -------------------------------------------------------
    # Helpers / backend
    # -------------------------------------------------------

    @staticmethod
    def _set_backend(backend: str, seed: Optional[int] = None):
        if isinstance(backend, str):
            bck = get_backend(backend, scipy=True, random=True, seed=seed)
            if isinstance(bck, tuple):
                module, module_sp = bck[0], bck[1]
                if isinstance(bck[2], tuple):
                    _rng, _rng_k = bck[2]
                else:
                    _rng, _rng_k = bck[2], None
            else:
                module, module_sp = bck, None
                _rng, _rng_k      = None, None
            return backend, module, module_sp, (_rng, _rng_k)

        target = 'jax' if JAX_AVAILABLE and backend == 'default' else 'np'
        return GeneralMatrix._set_backend(target)

    def _handle_dtype(self, dtype: Optional[Union[str, np.dtype]]) -> None:
        if dtype is not None:
            self._dtype = dtype
            if self._is_jax:
                self._iscpx = jnp.issubdtype(jnp.dtype(self._dtype), jnp.complexfloating)
            else:
                self._iscpx = np.issubdtype(np.dtype(self._dtype), np.complexfloating)
            return

        self._dtype = self._backend.float64
        self._iscpx = False

    def _log(self, msg: str, log: str = 'info', lvl: int = 0, color: str = "white") -> None:
        if self._logger is None:
            return
        self._logger.info(f"[{self.name}] {msg}", lvl=lvl, log=log, color=color)

    # -------------------------------------------------------
    # Matrix storage
    # -------------------------------------------------------

    def set_matrix_shape(self, shape: tuple[int, int]) -> None:
        self._shape     = tuple(shape)
        self._dim       = self._shape[0]
        self.shape      = self._shape

    def _get_matrix_reference(self):
        return self._matrix

    def _set_matrix_reference(self, matrix: Optional[Any]) -> None:
        self._matrix    = matrix
        self._is_built  = matrix is not None

    @property
    def matrix_data(self) -> Union[np.ndarray, Any]:
        matrix = self._get_matrix_reference()
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)
        return matrix

    # -------------------------------------------------------
    # Abstract build
    # -------------------------------------------------------

    def build(self) -> None:
        raise NotImplementedError("Subclasses must implement build().")

    def clear(self) -> None:
        self._log("Clearing cached matrix and eigen-decomposition...", lvl=2, log='debug')
        self._is_built    = False
        self._matrix      = None
        self._eig_vec     = None
        self._eig_val     = None
        self._krylov      = None
        self._diag_engine = None
        self._diag_method = 'exact'

    # -------------------------------------------------------
    # Properties
    # -------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def dtype(self):
        return self._dtype

    @property
    def dtypeint(self):
        return self._dtypeint

    @property
    def backend(self):
        return self._backendstr

    @property
    def sparse(self) -> bool:
        return self._is_sparse

    def is_sparse(self) -> bool:
        return self._is_sparse

    @property
    def max_local_changes(self):
        return self._max_local_ch

    @property
    def max_operator_changes(self):
        return self._max_local_ch_o

    @property
    def energies(self):
        return self._eig_val

    @property
    def eigenvalues(self):
        return self._eig_val

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eigenvals(self):
        return self._eig_val

    @property
    def eig_vals(self):
        return self._eig_val

    @property
    def eigen_vals(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def eigenvectors(self):
        return self._eig_vec

    @property
    def eigenvecs(self):
        return self._eig_vec

    @property
    def krylov(self):
        return self._krylov

    # -------------------------------------------------------
    # Diagonalization
    # -------------------------------------------------------

    def _prepare_diagonalization(self, **kwargs) -> None:
        """Hook executed before running the diagonalization engine."""

    def _get_diagonalization_matrix(self):
        return self.matrix_data

    def _default_diagonalization_backend(self, use_scipy: bool = True) -> str:
        if self._is_jax:
            return 'jax'
        return 'scipy' if use_scipy else 'numpy'

    def _ensure_diag_engine(self, method: str, backend: str, use_scipy: bool, verbose: bool) -> DiagonalizationEngine:
        if self._diag_engine is not None and self._diag_engine.method == method:
            return self._diag_engine

        self._diag_engine = DiagonalizationEngine(
            method    = method,
            backend   = backend,
            use_scipy = use_scipy,
            verbose   = verbose,
            logger    = self._logger,
        )
        return self._diag_engine

    def _on_diagonalized(self, result: Any, diag_duration: float, verbose: bool) -> None:
        """Hook executed after diagonalization to allow subclasses to log extra info."""
        if verbose:
            method_used = self.get_diagonalization_method()
            self._log(f"Diagonalization ({method_used}) completed in {diag_duration:.6f} seconds.", lvl=2, color="green")

    def diagonalize(self, verbose: bool = False, **kwargs) -> None:
        diag_start = time.perf_counter()

        method      = kwargs.pop("method", self._diag_method)
        backend_str = kwargs.pop("backend", None)
        use_scipy   = kwargs.pop("use_scipy", True)
        store_basis = kwargs.pop("store_basis", True)
        k           = kwargs.pop("k", None)
        which       = kwargs.pop("which", "smallest")
        hermitian   = kwargs.pop("hermitian", True)

        if backend_str is None:
            backend_str = self._default_diagonalization_backend(use_scipy=use_scipy)

        self._prepare_diagonalization(method=method, backend=backend_str, **kwargs)

        matrix_to_diag = self._get_diagonalization_matrix()
        if matrix_to_diag is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if method == 'exact':
            if JAX_AVAILABLE and BCOO is not None and isinstance(matrix_to_diag, BCOO):
                if verbose:
                    self._log("Converting JAX sparse matrix to dense for exact diagonalization.", lvl=2, color="yellow")
                matrix_to_diag = np.asarray(matrix_to_diag.todense())
            elif sp.sparse.issparse(matrix_to_diag):
                if verbose:
                    self._log("Converting SciPy sparse matrix to dense for exact diagonalization.", lvl=2, color="yellow")
                matrix_to_diag = matrix_to_diag.toarray()

        engine          = self._ensure_diag_engine(method=method, backend=backend_str, use_scipy=use_scipy, verbose=verbose)
        solver_kwargs   = {key: val for key, val in kwargs.items()
                        if key not in {'method', 'backend', 'use_scipy', 'store_basis', 'hermitian', 'k', 'which'}}

        matvec_callable = None
        try:
            matvec_callable = self.matvec_fun
        except NotImplementedError:
            pass

        size = getattr(matrix_to_diag, 'shape', (self._dim, self._dim))[0]

        try:
            result = engine.diagonalize(
                A           = matrix_to_diag,
                matvec      = matvec_callable,
                n           = size,
                k           = k,
                hermitian   = hermitian,
                which       = which,
                store_basis = store_basis,
                dtype       = self._dtype,
                **solver_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Diagonalization failed with method '{method}': {exc}") from exc

        self._diag_method   = method
        self._eig_val       = result.eigenvalues
        self._eig_vec       = result.eigenvectors

        if store_basis and engine.has_krylov_basis():
            self._krylov    = engine.get_krylov_basis()
        else:
            self._krylov    = None

        if JAX_AVAILABLE:
            if hasattr(self._eig_val, "block_until_ready"):
                self._eig_val = self._eig_val.block_until_ready()
            if hasattr(self._eig_vec, "block_until_ready"):
                self._eig_vec = self._eig_vec.block_until_ready()

        diag_duration = time.perf_counter() - diag_start
        self._on_diagonalized(result, diag_duration, verbose)

    # -------------------------------------------------------
    # Matrix/vector helpers
    # -------------------------------------------------------

    @property
    def matvec_fun(self):
        matvec_impl = getattr(self, "matvec", None)
        if matvec_impl is None:
            raise NotImplementedError("Subclasses must implement matvec() to use matvec_fun.")

        args, kwargs = self._matvec_context()

        def _matvec(x):
            return matvec_impl(x, *args, **kwargs)

        return _matvec

    @property
    def diag(self) -> Optional[Union[np.ndarray, Any]]:
        target = self._get_matrix_reference()
        if target is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if JAX_AVAILABLE and self._backend is not np:
            if isinstance(target, BCOO):
                return target.diagonal()
            if jnp is not None and isinstance(target, jnp.ndarray):
                return jnp.diag(target)
            return None

        if sp.issparse(target):
            return target.diagonal()
        if isinstance(target, np.ndarray):
            return target.diagonal()
        return None

    # -------------------------------------------------------
    # Memory helpers
    # -------------------------------------------------------

    def _estimate_matrix_memory(self, matrix: Any) -> float:
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if not self._is_sparse:
            if hasattr(matrix, 'nbytes'):
                return matrix.nbytes
            return int(np.prod(matrix.shape)) * matrix.dtype.itemsize

        if self._is_numpy:
            memory = 0
            for attr in ('data', 'indices', 'indptr'):
                if hasattr(matrix, attr):
                    arr = getattr(matrix, attr)
                    if hasattr(arr, 'nbytes'):
                        memory += arr.nbytes
                    else:
                        memory += int(np.prod(arr.shape)) * arr.dtype.itemsize
            return memory

        if self._is_jax and hasattr(matrix, 'data') and hasattr(matrix, 'indices'):
            data_arr    = matrix.data
            indices_arr = matrix.indices
            data_bytes  = data_arr.nbytes if hasattr(data_arr, 'nbytes') else int(np.prod(data_arr.shape)) * data_arr.dtype.itemsize
            ind_bytes   = indices_arr.nbytes if hasattr(indices_arr, 'nbytes') else int(np.prod(indices_arr.shape)) * indices_arr.dtype.itemsize
            return data_bytes + ind_bytes

        return 0.0

    @property
    def mat_memory(self) -> float:
        return self._estimate_matrix_memory(self._get_matrix_reference())

    @property
    def eigvec_memory(self) -> float:
        if self._eig_vec is None:
            return 0.0
        if hasattr(self._eig_vec, 'nbytes'):
            return self._eig_vec.nbytes
        return int(np.prod(self._eig_vec.shape)) * self._eig_vec.dtype.itemsize

    @property
    def eigval_memory(self) -> float:
        if self._eig_val is None:
            return 0.0
        if hasattr(self._eig_val, 'nbytes'):
            return self._eig_val.nbytes
        return int(np.prod(self._eig_val.shape)) * self._eig_val.dtype.itemsize

    @property
    def mat_memory_mb(self) -> float:
        return self.mat_memory / (1024 ** 2)

    @property
    def eigvec_memory_mb(self) -> float:
        return self.eigvec_memory / (1024 ** 2)

    @property
    def eigval_memory_mb(self) -> float:
        return self.eigval_memory / (1024 ** 2)

    @property
    def mat_memory_gb(self) -> float:
        return self.mat_memory / (1024 ** 3)

    @property
    def eigvec_memory_gb(self) -> float:
        return self.eigvec_memory / (1024 ** 3)

    @property
    def eigval_memory_gb(self) -> float:
        return self.eigval_memory / (1024 ** 3)

    @property
    def memory(self) -> float:
        return self.mat_memory + self.eigvec_memory + self.eigval_memory

    @property
    def memory_mb(self) -> float:
        return self.memory / (1024 ** 2)

    @property
    def memory_gb(self) -> float:
        return self.memory / (1024 ** 3)

    # -------------------------------------------------------
    # Krylov helpers
    # -------------------------------------------------------

    def has_krylov_basis(self) -> bool:
        return diag_helpers.has_krylov_basis(self._diag_engine, self._krylov)

    def get_krylov_basis(self) -> Optional[Array]:
        return diag_helpers.get_krylov_basis(self._diag_engine, self._krylov)

    def to_original_basis(self, vec: Array) -> Array:
        return diag_helpers.to_original_basis(vec, self._diag_engine, self.get_diagonalization_method())

    def to_krylov_basis(self, vec: Array) -> Array:
        return diag_helpers.to_krylov_basis(vec, self._diag_engine)

    def get_basis_transform(self) -> Optional[Array]:
        return diag_helpers.get_basis_transform(self._diag_engine, self._krylov)

    def get_diagonalization_method(self) -> Optional[str]:
        return diag_helpers.get_diagonalization_method(self._diag_engine)

    def get_diagonalization_info(self) -> dict:
        return diag_helpers.get_diagonalization_info(self._diag_engine, self._eig_val, self._krylov)

    # -------------------------------------------------------
    # Sparsity controls
    # -------------------------------------------------------

    def to_dense(self) -> None:
        self._is_sparse = False
        self._log("Switching to dense representation; clearing cached data.", lvl=1)
        self.clear()

    def to_sparse(self) -> None:
        self._is_sparse = True
        self._log("Switching to sparse representation; clearing cached data.", lvl=1)
        self.clear()

    # -------------------------------------------------------
    # Formatting helpers
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
            return GeneralMatrix._fmt_scalar(name, arr[0])
        
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return f"{name}=[]"
        if np.allclose(arr, arr.flat[0], atol=tol, rtol=0):
            return GeneralMatrix._fmt_scalar(name, float(arr.flat[0]), prec=prec)
        return f"{name}[min={arr.min():.{prec}f},max={arr.max():.{prec}f}]"

    @staticmethod
    def fmt(name, value, prec=1):
        """Choose scalar vs array formatter."""
        return GeneralMatrix._fmt_scalar(name, value, prec=prec) if np.isscalar(value) else GeneralMatrix._fmt_array(name, value, prec=prec)
        
# -------------------------------------------------------
#! EOF
# -------------------------------------------------------
