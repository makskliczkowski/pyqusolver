"""
We move most of the definitions of matrices to this file. Operators will derive
from it and be able to create matrices as needed.

-------------------------------------------------------------------------------
File            : Algebra/Operator/matrix.py
Author          : Maksymilian Kliczkowski
Date            : 2025-11-24
Copyright       : (c) 2025
License         : MIT
Description     : Generic matrix class for operators and Hamiltonians.
                It provides matrix storage, diagonalization, and logging.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing     import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy                as np
import scipy.sparse         as sp
import scipy.sparse.linalg  as spla

try:
    if TYPE_CHECKING:
        from QES.general_python.algebra.utils   import Array
        from QES.general_python.common.flog     import Logger

    import QES.Algebra.Hamil.hamil_diag_helpers as diag_helpers
    from QES.Algebra.Hamil.hamil_diag_engine    import DiagonalizationEngine

except ImportError as exc:
    raise ImportError("QES.general_python.algebra.utils could not be imported. Ensure QES is properly installed.") from exc

# --------------------------------------------------------------------------------

try:
    import jax
    import jax.lax as lax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO, CSR

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    lax = None
    BCOO = None
    CSR = None
    JAX_AVAILABLE = False

# --------------------------------------------------------------------------------


class DummyVector:
    """
    Constant vector of length `ns` with all entries == `val`.
    A thin dummy so scalar couplings can be broadcast like arrays.
    
    Parameters
    ----------
    val : scalar
        The constant value for all entries in the vector.
    ns : int, optional
        The number of sites (length of the vector). Defaults to 1.
    backend : module, optional
        The backend module to use (e.g., `numpy`, `jax.numpy`). Defaults to `numpy`.
    Notes
    -----
    This class mimics a full array without actually storing all elements,
    enabling efficient operations and type casting.
    
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

        self.val        = val
        self.ns         = int(ns) if ns is not None else 1
        self._backend   = backend or __import__("numpy")

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
        try:
            from QES.general_python.algebra.utils import distinguish_type
        except ImportError:
            raise ImportError(
                "QES.general_python.algebra.utils could not be imported. Ensure QES is properly installed."
            )

        backend = backend or self._backend
        tgt_dt = distinguish_type(dtype)

        # Robust check for complex dtype that works for both NumPy and JAX
        # backend.iscomplexobj(dtype) can fail if dtype is a type object (especially in JAX)
        is_tgt_complex = np.dtype(tgt_dt).kind == 'c'
        is_val_complex = backend.iscomplexobj(self.val)

        if is_val_complex and not is_tgt_complex:
            # If we're casting from complex to real, take the real part
            self.val = backend.real(self.val)
        elif not is_val_complex and is_tgt_complex:
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
    
    def _invalidate_cache(self):
        """Wipe eigenvalues, eigenvectors, cached many-body calculator."""
        self._eig_val       = None
        self._eig_vec       = None
        self._krylov        = None

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

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary(other, lambda a, b: b / a)

    # (add more as needed)

    # ---------------------------------------------------------------------------

    def __eq__(self, other):
        return isinstance(other, DummyVector) and self.ns == other.ns and self.val == other.val

    def __hash__(self):
        return hash((self.val, self.ns))

    def to_array(self, dtype=None, backend=None):
        """
        Convert the DummyVector to a numpy array.
        """
        backend = backend if backend is not None else __import__("numpy")
        return backend.full(self.ns, self.val, dtype=dtype)


##################################################################################
#! General Matrix Class
##################################################################################


class GeneralMatrix(spla.LinearOperator):
    """
    Generic linear-algebra helper providing matrix storage, diagonalization,
    and logging. This class serves as the foundation for operators and
    Hamiltonians that need matrix representations.

    Provides:
        - Backend handling (NumPy, JAX, SciPy)
        - Matrix storage (sparse/dense)
        - Diagonalization via DiagonalizationEngine
        - Eigenvalue/eigenvector storage
        - Krylov basis management
        - Memory estimation
        - SciPy LinearOperator interface

    Subclasses (Operator, Hamiltonian) can override specific methods while
    inheriting the common matrix infrastructure.
    """

    _ERR_MATRIX_NOT_BUILT       = "Matrix representation has not been built. Call build() first."
    _ERR_INVALID_BACKEND        = "Invalid backend specified."
    _ERR_UNSUPPORTED_OPERATION  = "The requested operation is not supported for this matrix type."

    # -------------------------------------------------------

    def __init__(
        self,
        shape: Optional[tuple[int, int]] = None,
        *,
        ns: Optional[int] = None,
        matvec: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        is_sparse: bool = True,
        backend: str = "default",
        backend_components: Optional[tuple[Any, Any, Any, tuple[Any, Any]]] = None,
        logger: Optional["Logger"] = None,
        seed: Optional[int] = None,
        dtype: Optional[Union[str, np.dtype]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the GeneralMatrix.

        Parameters
        ----------
        shape : tuple[int, int], optional
            Shape of the matrix. If None, defaults to (0, 0).
        ns : int, optional
            Number of sites/modes. Used by subclasses for physical systems.
        matvec : callable, optional
            Custom matrix-vector product function.
        is_sparse : bool, default True
            Whether to use sparse matrix representation.
        backend : str, default 'default'
            Computational backend ('default', 'np', 'numpy', 'jax').
        backend_components : tuple, optional
            Pre-computed backend components (backendstr, backend, backend_sp, (rng, rng_k)).
        logger : Logger, optional
            Logger instance for logging messages.
        seed : int, optional
            Random seed for reproducibility.
        dtype : dtype, optional
            Data type for matrix elements.
        """

        logger      = GeneralMatrix._check_logger(logger)
        self._shape = tuple(shape) if shape is not None else (0, 0)
        self._dim   = self._shape[0]
        self._ns    = ns  # Number of sites/modes (optional, for physical systems)

        if backend_components is not None:
            self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k) = (
                backend_components
            )
        else:
            self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k) = (
                GeneralMatrix._set_backend(backend, seed)
            )

        self._seed          = seed
        self._is_jax        = JAX_AVAILABLE and self._backend is not np
        self._is_numpy      = not self._is_jax
        self._is_sparse     = is_sparse
        self._dtypeint      = self._backend.int64
        self._name          = "GeneralMatrix"
        self._custom_matvec = matvec

        self._is_built      = False
        self._matrix: Optional[Union[np.ndarray, Any]]  = None
        self._eig_vec: Optional[Union[np.ndarray, Any]] = None
        self._eig_val: Optional[np.ndarray]             = None
        self._krylov: Optional[Any]                     = None
        self._max_local_ch: int                         = 1
        self._max_local_ch_o: int                       = 1

        self._diag_engine: Optional[DiagonalizationEngine]  = None
        self._diag_method: str                              = "exact"

        self._original_basis: Optional[Any]             = None
        self._current_basis: Optional[Any]              = None
        self._basis_metadata: dict                      = {}
        self._is_transformed: bool                      = False
        self._transformed_grid: Optional[Any]           = None
        self._symmetry_info: dict                       = {}

        # -----------------------

        self._av_en_idx     = 0
        self._av_en         = 0.0
        self._std_en        = 0.0
        self._min_en        = 0.0
        self._max_en        = 0.0
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
            raise NotImplementedError(
                "Matrix-vector product not defined. Provide a matvec implementation."
            )

        args, kwargs = self._matvec_context()
        return matvec_impl(x, *args, **kwargs)

    def _adjoint(self):
        return self

    # -------------------------------------------------------
    # Helpers / backend
    # -------------------------------------------------------

    @staticmethod
    def _check_logger(logger: Optional["Logger"]) -> Optional["Logger"]:
        """Ensure a valid logger is available."""

        if logger is None:
            try:
                from QES.qes_globals import get_logger as get_global_logger

                logger = get_global_logger()
            except ImportError:
                pass
        return logger

    @staticmethod
    def _set_backend(backend: str, seed: Optional[int] = None):
        """
        Set the computational backend.
        - backend : str
            Backend to use ('default', 'np', 'numpy', 'jax').
        - seed : Optional[int]
            Random seed for reproducibility.
        """

        if isinstance(backend, str):

            try:
                from QES.general_python.algebra.utils import get_backend
            except ImportError:
                raise ImportError(
                    "QES.general_python.algebra.utils could not be imported. Ensure QES is properly installed."
                )

            bck = get_backend(backend, scipy=True, random=True, seed=seed)
            if isinstance(bck, tuple):
                module, module_sp = bck[0], bck[1]
                if isinstance(bck[2], tuple):
                    _rng, _rng_k = bck[2]
                else:
                    _rng, _rng_k = bck[2], None
            else:
                module, module_sp = bck, None
                _rng, _rng_k = None, None
            return backend, module, module_sp, (_rng, _rng_k)
        if isinstance(backend, str):
            target = (
                "np"
                if backend.lower() in ["default", "np", "numpy"] or not JAX_AVAILABLE
                else "jax"
            )
        else:
            target = "jax" if JAX_AVAILABLE else "np"
        return GeneralMatrix._set_backend(target)

    def _handle_dtype(self, dtype: Optional[Union[str, np.dtype]]) -> None:
        """Handle the data type for the matrix elements."""
        if dtype is not None:
            self._dtype = dtype
            if self._is_jax:
                self._iscpx = jnp.issubdtype(jnp.dtype(self._dtype), jnp.complexfloating)
            else:
                self._iscpx = np.issubdtype(np.dtype(self._dtype), np.complexfloating)
            return

        self._dtype = self._backend.float64
        self._iscpx = False

    def _log(self, msg: str, log: str = "info", lvl: int = 0, color: str = "white") -> None:
        # Get logger safely (if _logger is not available on self, use None)
        logger = getattr(self, "_logger", None)
        if logger is None:
            return
        logger.info(f"[{self.name}] {msg}", lvl=lvl, log=log, color=color)

    # -------------------------------------------------------
    # Matrix storage
    # -------------------------------------------------------

    def set_matrix_shape(self, shape: tuple[int, int]) -> None:
        self._shape = tuple(shape)
        self._dim = self._shape[0]
        self.shape = self._shape

    def _get_matrix_reference(self):
        return self._matrix

    def _set_matrix_reference(self, matrix: Optional[Any]) -> None:
        self._matrix = matrix
        self._is_built = matrix is not None

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
        self._log("Clearing cached matrix and eigen-decomposition...", lvl=2, log="debug")
        self._is_built = False
        self._matrix = None
        self._eig_vec = None
        self._eig_val = None
        self._krylov = None
        self._diag_engine = None
        self._av_en_idx = 0
        self._av_en = 0.0
        self._std_en = 0.0
        self._min_en = 0.0
        self._max_en = 0.0
        self._diag_method = "exact"

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

    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = value

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

    # -------------------------------------------------------

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

    # -------------------------------------------------------

    @property
    def av_en_idx(self):
        """
        Returns the index of the average energy/eigenvalue of the matrix.
        This is used to track the average energy during calculations.
        """
        if self._av_en_idx is not None or self._av_en_idx == 0:
            self._av_en_idx = int(np.argmin(np.abs(self.eig_val - self.av_en)))
        return self._av_en_idx

    @property
    def av_en_idx_and_value(self):
        """
        Returns the index and value of the average energy of the Hamiltonian.
        This is used to track the average energy during calculations.

        Returns:
            tuple : (index, value) of the average energy.
        """
        return self.av_en_idx, self.av

    # energy properties
    @property
    def av_en(self):
        return self._av_en

    @property
    def std_en(self):
        return self._std_en

    @property
    def min_en(self):
        return self._min_en

    @property
    def max_en(self):
        return self._max_en

    # -------------------------------------------------------

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
    # Eigenvalue/Eigenvector Getters
    # -------------------------------------------------------

    def get_eigvec(self, *args):
        """
        Returns the eigenvectors of the matrix.

        - No arguments: return all eigenvectors (matrix Nh x Nh, each column is an eigenvector)
        - One argument: return the eigenvector at that index
        - Two arguments: return a specific element

        Parameters
        ----------
        *args : int
            Optional indices for selecting specific eigenvectors or elements.

        Returns
        -------
        ndarray
            The requested eigenvector(s) or element.

        Raises
        ------
        ValueError
            If eigenvalues are not available or invalid arguments provided.
        """
        if self._eig_vec is None:
            raise ValueError("Eigenvectors not available. Call diagonalize() first.")
        if len(args) == 0:
            return self._eig_vec
        elif len(args) == 1 and len(self._eig_vec.shape) == 2:
            return self._eig_vec[:, args[0]]
        elif len(args) == 2 and len(self._eig_vec.shape) == 2:
            return self._eig_vec[args[0], args[1]]
        else:
            raise ValueError("Invalid arguments provided for eigenvector retrieval.")

    def get_eigval(self, *args):
        """
        Returns the eigenvalues of the matrix.

        - No arguments: return all eigenvalues (a vector in ascending order)
        - One argument: return a single eigenvalue at that index

        Parameters
        ----------
        *args : int
            Optional index for selecting a specific eigenvalue.

        Returns
        -------
        ndarray or scalar
            The requested eigenvalue(s).

        Raises
        ------
        ValueError
            If eigenvalues are not available or invalid arguments provided.
        """
        if self._eig_val is None:
            raise ValueError("Eigenvalues not available. Call diagonalize() first.")
        if len(args) == 0:
            return self._eig_val
        elif len(args) == 1 and len(self._eig_val) > 0:
            return self._eig_val[args[0]]
        else:
            raise ValueError("Invalid arguments provided for eigenvalue retrieval.")

    # -------------------------------------------------------
    # Ground State and Excited States
    # -------------------------------------------------------

    @property
    def ground_state(self) -> np.ndarray:
        """
        Return the ground state (first eigenvector corresponding to the lowest eigenvalue).

        Returns
        -------
        ndarray
            The ground state eigenvector.

        Raises
        ------
        ValueError
            If eigenvectors are not available (diagonalize() not called).
        """
        if self._eig_vec is None:
            raise ValueError("Ground state not available. Call diagonalize() first.")
        return self._eig_vec[:, 0]

    @property
    def ground_energy(self) -> float:
        """
        Return the ground state energy (lowest eigenvalue).

        Returns
        -------
        float
            The ground state energy.

        Raises
        ------
        ValueError
            If eigenvalues are not available (diagonalize() not called).
        """
        if self._eig_val is None:
            raise ValueError("Ground energy not available. Call diagonalize() first.")
        return float(self._eig_val[0])

    def excited_state(self, n: int) -> np.ndarray:
        """
        Return the n-th excited state eigenvector.

        Parameters
        ----------
        n : int
            The excitation index (0 = ground state, 1 = first excited, etc.).

        Returns
        -------
        ndarray
            The n-th excited state eigenvector.

        Raises
        ------
        ValueError
            If eigenvectors are not available or index is out of bounds.
        """
        if self._eig_vec is None:
            raise ValueError("Excited states not available. Call diagonalize() first.")
        if n < 0 or n >= self._eig_vec.shape[1]:
            raise IndexError(
                f"Excited state index {n} out of bounds (0 to {self._eig_vec.shape[1] - 1})."
            )
        return self._eig_vec[:, n]

    def excited_energy(self, n: int) -> float:
        """
        Return the n-th excited state energy.

        Parameters
        ----------
        n : int
            The excitation index (0 = ground energy, 1 = first excited energy, etc.).

        Returns
        -------
        float
            The n-th excited state energy.

        Raises
        ------
        ValueError
            If eigenvalues are not available or index is out of bounds.
        """
        if self._eig_val is None:
            raise ValueError("Excited energies not available. Call diagonalize() first.")
        if n < 0 or n >= len(self._eig_val):
            raise IndexError(f"Energy index {n} out of bounds (0 to {len(self._eig_val) - 1}).")
        return float(self._eig_val[n])

    @property
    def spectral_gap(self) -> float:
        """
        Return the spectral gap (difference between first excited and ground state energies).

        Returns
        -------
        float
            The spectral gap E_1 - E_0.

        Raises
        ------
        ValueError
            If fewer than 2 eigenvalues are available.
        """
        if self._eig_val is None or len(self._eig_val) < 2:
            raise ValueError(
                "Spectral gap requires at least 2 eigenvalues. Call diagonalize() first."
            )
        return float(self._eig_val[1] - self._eig_val[0])

    @property
    def spectral_width(self) -> float:
        """
        Return the spectral width (difference between largest and smallest eigenvalues).

        Returns
        -------
        float
            The spectral width E_max - E_min.

        Raises
        ------
        ValueError
            If eigenvalues are not available.
        """
        if self._eig_val is None:
            raise ValueError("Spectral width not available. Call diagonalize() first.")
        return float(self._eig_val[-1] - self._eig_val[0])

    # -------------------------------------------------------
    # Diagonalization
    # -------------------------------------------------------

    def _prepare_diagonalization(self, **kwargs) -> None:
        """Hook executed before running the diagonalization engine."""

    def _get_diagonalization_matrix(self):
        return self.matrix_data

    def _default_diagonalization_backend(self, use_scipy: bool = True) -> str:
        if self._is_jax:
            return "jax"
        return "scipy" if use_scipy else "numpy"

    def _ensure_diag_engine(
        self, method: str, backend: str, use_scipy: bool, verbose: bool
    ) -> DiagonalizationEngine:
        if self._diag_engine is not None and self._diag_engine.method == method:
            return self._diag_engine

        # Get logger safely (if _logger is not available on self, use global logger)
        logger = getattr(self, "_logger", None)

        self._diag_engine = DiagonalizationEngine(
            method=method,
            backend=backend,
            use_scipy=use_scipy,
            verbose=verbose,
            logger=logger,
        )
        return self._diag_engine

    def _on_diagonalized(self, result: Any, diag_duration: float, verbose: bool) -> None:
        """Hook executed after diagonalization to allow subclasses to log extra info."""
        if verbose:
            method_used = self.get_diagonalization_method()
            self._log(
                f"Diagonalization ({method_used}) completed in {diag_duration:.6f} seconds.",
                lvl=2,
                color="green",
            )

    def diagonalize(self, verbose: bool = False, **kwargs) -> None:
        diag_start = time.perf_counter()

        method = kwargs.pop("method", self._diag_method)
        backend_str = kwargs.pop("backend", None)
        use_scipy = kwargs.pop("use_scipy", True)
        store_basis = kwargs.pop("store_basis", True)
        k = kwargs.pop("k", None)
        which = kwargs.pop("which", "smallest")
        hermitian = kwargs.pop("hermitian", True)

        if backend_str is None:
            backend_str = self._default_diagonalization_backend(use_scipy=use_scipy)

        self._prepare_diagonalization(method=method, backend=backend_str, **kwargs)

        matrix_to_diag = self._get_diagonalization_matrix()
        if matrix_to_diag is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if method == "exact":
            if JAX_AVAILABLE and BCOO is not None and isinstance(matrix_to_diag, BCOO):
                if verbose:
                    self._log(
                        "Converting JAX sparse matrix to dense for exact diagonalization.",
                        lvl=2,
                        color="yellow",
                    )
                matrix_to_diag = np.asarray(matrix_to_diag.todense())
            elif sp.sparse.issparse(matrix_to_diag):
                if verbose:
                    self._log(
                        "Converting SciPy sparse matrix to dense for exact diagonalization.",
                        lvl=2,
                        color="yellow",
                    )
                matrix_to_diag = matrix_to_diag.toarray()

        engine = self._ensure_diag_engine(
            method=method, backend=backend_str, use_scipy=use_scipy, verbose=verbose
        )
        solver_kwargs = {
            key: val
            for key, val in kwargs.items()
            if key
            not in {"method", "backend", "use_scipy", "store_basis", "hermitian", "k", "which"}
        }

        matvec_callable = None
        try:
            matvec_callable = self.matvec_fun
        except NotImplementedError:
            pass

        size = getattr(matrix_to_diag, "shape", (self._dim, self._dim))[0]

        try:
            result = engine.diagonalize(
                A=matrix_to_diag,
                matvec=matvec_callable,
                n=size,
                k=k,
                hermitian=hermitian,
                which=which,
                store_basis=store_basis,
                dtype=self._dtype,
                **solver_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Diagonalization failed with method '{method}': {exc}") from exc

        self._diag_method = method
        self._eig_val = result.eigenvalues
        self._eig_vec = result.eigenvectors

        if store_basis and engine.has_krylov_basis():
            self._krylov = engine.get_krylov_basis()
        else:
            self._krylov = None

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
    # Expectation Values and Matrix Operations
    # -------------------------------------------------------

    def expectation_value(
        self, state: np.ndarray, other_state: Optional[np.ndarray] = None
    ) -> complex:
        """
        Compute the expectation value ⟨state|M|other_state⟩ or ⟨state|M|state⟩.

        Parameters
        ----------
        state : ndarray
            The ket vector |ψ⟩.
        other_state : ndarray, optional
            The bra vector. If None, uses state (computes ⟨ψ|M|ψ⟩).

        Returns
        -------
        complex
            The expectation value.

        Raises
        ------
        ValueError
            If matrix is not built.
        """
        if other_state is None:
            other_state = state

        matrix = self._get_matrix_reference()
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        # Compute M|other_state⟩
        if sp.issparse(matrix):
            M_state = matrix @ other_state
        elif JAX_AVAILABLE and BCOO is not None and isinstance(matrix, BCOO):
            M_state = matrix @ other_state
        else:
            M_state = self._backend.dot(matrix, other_state)

        # Compute <state|M|other_state>
        return self._backend.vdot(state, M_state)

    def trace_matrix(self) -> complex:
        r"""
        Compute the trace of the matrix.

        Returns
        -------
        complex
            Tr(M) = \sum_i M_ii.

        Raises
        ------
        ValueError
            If matrix is not built.
        """
        matrix = self._get_matrix_reference()
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if sp.issparse(matrix):
            return matrix.diagonal().sum()

        if JAX_AVAILABLE and BCOO is not None and isinstance(matrix, BCOO):
            return matrix.todense().trace()

        return self._backend.trace(matrix)

    def frobenius_norm(self) -> float:
        r"""
        Compute the Frobenius norm of the matrix.

        Returns
        -------
        float
            ||M||_F = \sqrt{\sum_{i,j} |M_{i,j}|^2} = \sqrt{\mathrm{Tr}(M^\dagger M)}.

        Raises
        ------
        ValueError
            If matrix is not built.
        """

        matrix = self._get_matrix_reference()
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if sp.issparse(matrix):
            return np.sqrt(np.abs(matrix.data**2).sum())

        if JAX_AVAILABLE and BCOO is not None and isinstance(matrix, BCOO):
            return float(jnp.sqrt(jnp.sum(jnp.abs(matrix.data) ** 2)))

        return float(self._backend.linalg.norm(matrix, "fro"))

    def spectral_norm(self) -> float:
        """
        Compute the spectral norm of the matrix (largest singular value).

        Returns
        -------
        float
            ||M||_2 = largest singular value.

        Raises
        ------
        ValueError
            If matrix is not built.
        """
        matrix = self._get_matrix_reference()
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if sp.issparse(matrix):
            # Use sparse SVD for largest singular value
            try:
                from scipy.sparse.linalg import svds

                u, s, v = svds(matrix, k=1, which="LM")
                return float(s[0])
            except Exception:
                # Fall back to dense computation
                matrix = matrix.toarray()

        if JAX_AVAILABLE and BCOO is not None and isinstance(matrix, BCOO):
            matrix = np.asarray(matrix.todense())

        return float(np.linalg.norm(matrix, 2))

    def eigenvector_norm(self, n: int = 0) -> float:
        """
        Compute the norm of the n-th eigenvector.

        Parameters
        ----------
        n : int, default 0
            Index of the eigenvector to compute norm for.

        Returns
        -------
        float
            ||ψₙ||₂ = √(⟨ψₙ|ψₙ⟩).

        Raises
        ------
        ValueError
            If eigenvectors are not available.
        """
        if self._eig_vec is None:
            raise ValueError("Eigenvector norm not available. Call diagonalize() first.")
        if n < 0 or n >= self._eig_vec.shape[1]:
            raise IndexError(f"Eigenvector index {n} out of bounds.")

        vec = self._eig_vec[:, n]
        return float(self._backend.linalg.norm(vec))

    def overlap(self, vec1: np.ndarray, vec2: Optional[np.ndarray] = None) -> complex:
        """
        Compute the overlap ⟨vec1|vec2⟩.

        Parameters
        ----------
        vec1 : ndarray
            First vector.
        vec2 : ndarray, optional
            Second vector. If None, computes ⟨vec1|vec1⟩ = ||vec1||².

        Returns
        -------
        complex
            The inner product ⟨vec1|vec2⟩.
        """
        if vec2 is None:
            vec2 = vec1
        return self._backend.vdot(vec1, vec2)

    # -------------------------------------------------------
    # Commutators and Anticommutators
    # -------------------------------------------------------

    def commutator(self, other: "GeneralMatrix") -> np.ndarray:
        """
        Compute the commutator [self, other] = self @ other - other @ self.

        Parameters
        ----------
        other : GeneralMatrix
            The other matrix/operator.

        Returns
        -------
        ndarray
            The commutator matrix [M, O].

        Raises
        ------
        ValueError
            If either matrix is not built.
        """
        M = self._get_matrix_reference()
        if M is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        O = other._get_matrix_reference() if hasattr(other, "_get_matrix_reference") else other
        if O is None:
            raise ValueError("Other matrix representation has not been built.")

        return M @ O - O @ M

    def anticommutator(self, other: "GeneralMatrix") -> np.ndarray:
        """
        Compute the anticommutator {self, other} = self @ other + other @ self.

        Parameters
        ----------
        other : GeneralMatrix
            The other matrix/operator.

        Returns
        -------
        ndarray
            The anticommutator matrix {M, O}.

        Raises
        ------
        ValueError
            If either matrix is not built.
        """
        M = self._get_matrix_reference()
        if M is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        O = other._get_matrix_reference() if hasattr(other, "_get_matrix_reference") else other
        if O is None:
            raise ValueError("Other matrix representation has not been built.")

        return M @ O + O @ M

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply the matrix to a state vector: M|ψ⟩.

        This is equivalent to matrix-vector multiplication.

        Parameters
        ----------
        state : ndarray
            The state vector to apply the matrix to.

        Returns
        -------
        ndarray
            The resulting state M|ψ⟩.

        Raises
        ------
        ValueError
            If matrix is not built.
        """
        matrix = self._get_matrix_reference()
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)
        return matrix @ state

    # -------------------------------------------------------
    # Spectral Analysis
    # -------------------------------------------------------

    def density_of_states(
        self, bins: int = 50, range: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the density of states (histogram of eigenvalues).

        Parameters
        ----------
        bins : int, default 50
            Number of bins for the histogram.
        range : tuple of float, optional
            Range (min, max) for the histogram. If None, uses eigenvalue range.

        Returns
        -------
        tuple of ndarray
            (counts, bin_edges) - the histogram counts and bin edges.

        Raises
        ------
        ValueError
            If eigenvalues are not available.
        """
        if self._eig_val is None:
            raise ValueError("Density of states not available. Call diagonalize() first.")

        return np.histogram(self._eig_val, bins=bins, range=range)

    def participation_ratio(self, n: int = 0) -> float:
        """
        Compute the inverse participation ratio (IPR) for the n-th eigenstate.

        IPR = 1 / Σᵢ |ψᵢ|⁴

        The IPR measures localization:
        - IPR ≈ 1: fully localized on one site
        - IPR ≈ N: fully delocalized (uniform distribution)

        Parameters
        ----------
        n : int, default 0
            Index of the eigenvector (0 = ground state).

        Returns
        -------
        float
            The inverse participation ratio.

        Raises
        ------
        ValueError
            If eigenvectors are not available.
        """
        if self._eig_vec is None:
            raise ValueError("Participation ratio not available. Call diagonalize() first.")
        if n < 0 or n >= self._eig_vec.shape[1]:
            raise IndexError(f"Eigenvector index {n} out of bounds.")

        vec = self._eig_vec[:, n]
        probs_sq = np.abs(vec) ** 4
        ipr_inv = np.sum(probs_sq)

        if ipr_inv < 1e-15:
            return float("inf")
        return 1.0 / ipr_inv

    def degeneracy(self, tol: float = 1e-10) -> Dict[float, int]:
        """
        Find degenerate energy levels.

        Parameters
        ----------
        tol : float, default 1e-10
            Tolerance for considering eigenvalues equal.

        Returns
        -------
        dict
            Dictionary mapping unique energy levels to their degeneracy count.

        Raises
        ------
        ValueError
            If eigenvalues are not available.
        """
        if self._eig_val is None:
            raise ValueError("Degeneracy analysis not available. Call diagonalize() first.")

        energies = np.sort(self._eig_val)
        unique_energies = []
        degeneracies = []

        i = 0
        while i < len(energies):
            current = energies[i]
            count = 1
            while i + count < len(energies) and abs(energies[i + count] - current) < tol:
                count += 1
            unique_energies.append(float(current))
            degeneracies.append(count)
            i += count

        return dict(zip(unique_energies, degeneracies))

    def degenerate_subspace(self, energy: float, tol: float = 1e-10) -> np.ndarray:
        """
        Get all eigenvectors corresponding to a degenerate energy level.

        Parameters
        ----------
        energy : float
            The energy level to find eigenvectors for.
        tol : float, default 1e-10
            Tolerance for energy matching.

        Returns
        -------
        ndarray
            Matrix of eigenvectors (columns) with the given energy.

        Raises
        ------
        ValueError
            If eigenvalues/eigenvectors are not available or energy not found.
        """
        if self._eig_val is None or self._eig_vec is None:
            raise ValueError("Degenerate subspace not available. Call diagonalize() first.")

        mask = np.abs(self._eig_val - energy) < tol
        if not np.any(mask):
            raise ValueError(f"No eigenvalue found within tolerance of {energy}.")

        return self._eig_vec[:, mask]

    def level_spacing(self) -> np.ndarray:
        """
        Compute the level spacings (differences between consecutive eigenvalues).

        Returns
        -------
        ndarray
            Array of level spacings Δₙ = Eₙ₊₁ - Eₙ.

        Raises
        ------
        ValueError
            If eigenvalues are not available.
        """
        if self._eig_val is None:
            raise ValueError("Level spacing not available. Call diagonalize() first.")
        return np.diff(self._eig_val)

    def level_spacing_ratio(self) -> np.ndarray:
        """
        Compute the level spacing ratios rₙ = min(sₙ, sₙ₊₁) / max(sₙ, sₙ₊₁).

        This is used to distinguish integrable (Poisson) from chaotic (GOE/GUE) systems.
        - Poisson (integrable): ⟨r⟩ ≈ 0.386
        - GOE (chaotic, real symmetric): ⟨r⟩ ≈ 0.530
        - GUE (chaotic, complex Hermitian): ⟨r⟩ ≈ 0.603

        Returns
        -------
        ndarray
            Array of level spacing ratios.

        Raises
        ------
        ValueError
            If fewer than 3 eigenvalues are available.
        """
        if self._eig_val is None or len(self._eig_val) < 3:
            raise ValueError("Level spacing ratio requires at least 3 eigenvalues.")

        spacings = np.diff(self._eig_val)
        ratios = np.zeros(len(spacings) - 1)

        for i in range(len(spacings) - 1):
            s1, s2 = spacings[i], spacings[i + 1]
            ratios[i] = min(s1, s2) / max(s1, s2) if max(s1, s2) > 1e-15 else 0.0

        return ratios

    # -------------------------------------------------------
    # Memory helpers
    # -------------------------------------------------------

    def _estimate_matrix_memory(self, matrix: Any) -> float:
        if matrix is None:
            raise ValueError(self._ERR_MATRIX_NOT_BUILT)

        if not self._is_sparse:
            if hasattr(matrix, "nbytes"):
                return matrix.nbytes
            return int(np.prod(matrix.shape)) * matrix.dtype.itemsize

        if self._is_numpy:
            memory = 0
            for attr in ("data", "indices", "indptr"):
                if hasattr(matrix, attr):
                    arr = getattr(matrix, attr)
                    if hasattr(arr, "nbytes"):
                        memory += arr.nbytes
                    else:
                        memory += int(np.prod(arr.shape)) * arr.dtype.itemsize
            return memory

        if self._is_jax and hasattr(matrix, "data") and hasattr(matrix, "indices"):
            data_arr = matrix.data
            indices_arr = matrix.indices
            data_bytes = (
                data_arr.nbytes
                if hasattr(data_arr, "nbytes")
                else int(np.prod(data_arr.shape)) * data_arr.dtype.itemsize
            )
            ind_bytes = (
                indices_arr.nbytes
                if hasattr(indices_arr, "nbytes")
                else int(np.prod(indices_arr.shape)) * indices_arr.dtype.itemsize
            )
            return data_bytes + ind_bytes

        return 0.0

    @property
    def mat_memory(self) -> float:
        return self._estimate_matrix_memory(self._get_matrix_reference())

    @property
    def eigvec_memory(self) -> float:
        if self._eig_vec is None:
            return 0.0
        if hasattr(self._eig_vec, "nbytes"):
            return self._eig_vec.nbytes
        return int(np.prod(self._eig_vec.shape)) * self._eig_vec.dtype.itemsize

    @property
    def eigval_memory(self) -> float:
        if self._eig_val is None:
            return 0.0
        if hasattr(self._eig_val, "nbytes"):
            return self._eig_val.nbytes
        return int(np.prod(self._eig_val.shape)) * self._eig_val.dtype.itemsize

    @property
    def mat_memory_mb(self) -> float:
        return self.mat_memory / (1024**2)

    @property
    def eigvec_memory_mb(self) -> float:
        return self.eigvec_memory / (1024**2)

    @property
    def eigval_memory_mb(self) -> float:
        return self.eigval_memory / (1024**2)

    @property
    def mat_memory_gb(self) -> float:
        return self.mat_memory / (1024**3)

    @property
    def eigvec_memory_gb(self) -> float:
        return self.eigvec_memory / (1024**3)

    @property
    def eigval_memory_gb(self) -> float:
        return self.eigval_memory / (1024**3)

    @property
    def memory(self) -> float:
        return self.mat_memory + self.eigvec_memory + self.eigval_memory

    @property
    def memory_mb(self) -> float:
        return self.memory / (1024**2)

    @property
    def memory_gb(self) -> float:
        return self.memory / (1024**3)

    # -------------------------------------------------------
    # Krylov helpers
    # -------------------------------------------------------

    def has_krylov_basis(self) -> bool:
        return diag_helpers.has_krylov_basis(self._diag_engine, self._krylov)

    def get_krylov_basis(self) -> Optional[Array]:
        return diag_helpers.get_krylov_basis(self._diag_engine, self._krylov)

    def to_original_basis(self, vec: Array) -> Array:
        return diag_helpers.to_original_basis(
            vec, self._diag_engine, self.get_diagonalization_method()
        )

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
        return (
            GeneralMatrix._fmt_scalar(name, value, prec=prec)
            if np.isscalar(value)
            else GeneralMatrix._fmt_array(name, value, prec=prec)
        )

    # -------------------------------------------------------
    # Help and Documentation
    # -------------------------------------------------------

    @classmethod
    def help(cls, topic: Optional[str] = None) -> str:
        """
        Display help information about GeneralMatrix capabilities.

        Parameters
        ----------
        topic : str, optional
            Specific topic to get help on. Options:
            - None or 'all':
                Full overview
            - 'properties':
                Available properties
            - 'diagonalization':
                Diagonalization methods
            - 'spectral':
                Spectral analysis methods
            - 'memory':
                Memory estimation
            - 'matrix':
                Matrix operations

        Returns
        -------
        str
            Help text for the requested topic.

        Examples
        --------
        >>> GeneralMatrix.help()            # Full overview
        >>> GeneralMatrix.help('spectral')  # Spectral analysis help
        """
        topics = {
            "properties": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         GeneralMatrix: Properties                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Eigenvalue/Eigenvector Access:                                              ║
║    .eigenvalues, .eig_val    - All eigenvalues (sorted ascending)            ║
║    .eigenvectors, .eig_vec   - All eigenvectors (columns)                    ║
║    .ground_state             - Ground state eigenvector (first eigenvector)  ║
║    .ground_energy            - Ground state energy (lowest eigenvalue)       ║
║    .get_eigval(n)            - Get n-th eigenvalue                           ║
║    .get_eigvec(n)            - Get n-th eigenvector                          ║
║    .excited_state(n)         - Get n-th excited state                        ║
║    .excited_energy(n)        - Get n-th excited energy                       ║
║                                                                              ║
║  Matrix Properties:                                                          ║
║    .shape                    - Matrix dimensions (N, N)                      ║
║    .dtype                    - Data type (float64, complex128, etc.)         ║
║    .sparse                   - Whether using sparse representation           ║
║    .backend                  - Computational backend ('numpy', 'jax')        ║
║    .diag                     - Diagonal elements of the matrix               ║
║    .matrix_data              - Raw matrix data (raises if not built)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "diagonalization": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      GeneralMatrix: Diagonalization                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Main Method:                                                                ║
║    .diagonalize(method='auto', k=None, which='smallest', ...)                ║
║                                                                              ║
║  Methods Available:                                                          ║
║    'auto'   - Automatic selection based on matrix size                       ║
║    'exact'  - Full diagonalization (all eigenvalues)                         ║
║    'lanczos'- Lanczos iteration for sparse symmetric matrices                ║
║    'arnoldi'- Arnoldi iteration for general matrices                         ║
║                                                                              ║
║  Key Parameters:                                                             ║
║    method     : str  - Diagonalization method                                ║
║    k          : int  - Number of eigenvalues to compute                      ║
║    which      : str  - 'smallest', 'largest', or 'both'                      ║
║    tol        : float- Convergence tolerance (default: 1e-10)                ║
║    backend    : str  - 'numpy', 'scipy', or 'jax'                            ║
║    store_basis: bool - Store Krylov basis for transformations                ║
║                                                                              ║
║  Krylov Basis Methods (after iterative diagonalization):                     ║
║    .has_krylov_basis()       - Check if Krylov basis available               ║
║    .get_krylov_basis()       - Get the Krylov basis                          ║
║    .to_original_basis(vec)   - Transform from Krylov to original basis       ║
║    .to_krylov_basis(vec)     - Transform from original to Krylov basis       ║
║    .get_diagonalization_method() - Get method used                           ║
║    .get_diagonalization_info()   - Get full diagonalization info             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "spectral": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      GeneralMatrix: Spectral Analysis                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Energy Gaps and Widths:                                                     ║
║    .spectral_gap             - E₁ - E₀ (first excited - ground)              ║
║    .spectral_width           - E_max - E_min (total width)                   ║
║    .level_spacing()          - Array of \Delta_n = E_{n+1} - E_n             ║
║    .level_spacing_ratio()    - Array of r_n for chaos analysis               ║
║                               (Poisson ≈ 0.386, GOE ≈ 0.530, GUE ≈ 0.603)    ║
║                                                                              ║
║  Degeneracy Analysis:                                                        ║
║    .degeneracy(tol)          - Dict of {energy: degeneracy_count}            ║
║    .degenerate_subspace(E)   - All eigenvectors at energy E                  ║
║                                                                              ║
║  Localization:                                                               ║
║    .participation_ratio(n)   - IPR = 1/\sum_i|\psi_i|2 for n-th state        ║
║                               (1 = localized, N = delocalized)               ║
║    .density_of_states(bins)  - Histogram of eigenvalues                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "matrix": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      GeneralMatrix: Matrix Operations                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Expectation Values:                                                         ║
║    .expectation_value(ψ, φ)  - Compute ⟨ψ|M|φ⟩ (or ⟨ψ|M|ψ⟩ if φ=None)        ║
║    .overlap(v1, v2)          - Compute ⟨v1|v2⟩                               ║
║                                                                              ║
║  Matrix Norms:                                                               ║
║    .trace_matrix()           - Tr(M) = Σᵢ Mᵢᵢ                                ║
║    .frobenius_norm()         - ||M||_F = √(Σᵢⱼ |Mᵢⱼ|²)                       ║
║    .spectral_norm()          - ||M||₂ = largest singular value               ║
║    .eigenvector_norm(n)      - ||ψₙ||₂                                       ║
║                                                                              ║
║  Commutators:                                                                ║
║    .commutator(O)            - [M, O] = MO - OM                              ║
║    .anticommutator(O)        - {M, O} = MO + OM                              ║
║                                                                              ║
║  Application:                                                                ║
║    .apply(ψ)                 - Compute M|ψ⟩                                  ║
║    .matvec_fun               - Get matvec function for LinearOperator        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "memory": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      GeneralMatrix: Memory Estimation                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Memory Properties (in bytes):                                               ║
║    .mat_memory               - Memory used by matrix                         ║
║    .eigvec_memory            - Memory used by eigenvectors                   ║
║    .eigval_memory            - Memory used by eigenvalues                    ║
║    .memory                   - Total memory (matrix + eigenvectors + vals)   ║
║                                                                              ║
║  Memory Properties (in MB):                                                  ║
║    .mat_memory_mb, .eigvec_memory_mb, .eigval_memory_mb, .memory_mb          ║
║                                                                              ║
║  Memory Properties (in GB):                                                  ║
║    .mat_memory_gb, .eigvec_memory_gb, .eigval_memory_gb, .memory_gb          ║
║                                                                              ║
║  Representation Control:                                                     ║
║    .to_sparse()              - Switch to sparse representation               ║
║    .to_dense()               - Switch to dense representation                ║
║    .clear()                  - Clear cached matrix and eigendecomposition    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
        }

        overview = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              GeneralMatrix                                   ║
║          Base class for matrix operations, diagonalization, and              ║
║          spectral analysis in quantum systems.                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Inheritance: GeneralMatrix -> scipy.sparse.linalg.LinearOperator            ║
║  Subclasses:  Operator -> Hamiltonian                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quick Start:                                                                ║
║    1. Build the matrix:     obj.build()                                      ║
║    2. Diagonalize:          obj.diagonalize()                                ║
║    3. Access results:       obj.ground_state, obj.eigenvalues                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Topics (use .help('topic') for details):                                    ║
║    'properties'      - Eigenvalue/eigenvector access, matrix properties      ║
║    'diagonalization' - Diagonalization methods and Krylov basis              ║
║    'spectral'        - Spectral analysis (gaps, degeneracy, localization)    ║
║    'matrix'          - Matrix operations (expectation values, norms)         ║
║    'memory'          - Memory estimation and representation control          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

        if topic is None or topic == "all":
            result = overview
            for t in topics.values():
                result += t
            print(result)
            return result

        if topic in topics:
            print(topics[topic])
            return topics[topic]

        print(f"Unknown topic '{topic}'. Available: {list(topics.keys())}")
        return f"Unknown topic '{topic}'. Available: {list(topics.keys())}"


# --------------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------------
