"""
This module defines a set of classes and functions for
handling general operators in quantum mechanics,
particularly in the context of Hilbert spaces.

Main components of this module include:

Classes:
    - GeneralOperator: Describes a general operator. It can be expanded for more complicated operators acting on Hilbert space or other spaces. It supports various constructors for different initializations and provides methods for setting and getting operator properties.
    - OperatorContainer: Stores elements in a matrix form and provides methods for updating, sampling, and normalizing operator values.

Enumerations:
    - SymGenerators     : Enumerates various implemented symmetry types - used for symmetry analysis.
    - FermionicOperators: Enumerates various implemented fermionic operators.

This module is under constant development and is intended to be expanded for more complex operators and functionalities.

-----------------------------------------------------------------
File    : Algebra/Operator/operator.py
Date    : April 2025
Author  : Maksymilian Kliczkowski, WUST, Poland
-----------------------------------------------------------------
"""

from __future__ import annotations

import copy
import numbers
import time

#####################################################################################################
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numba
import numpy as np
from numba.core.registry import CPUDispatcher

####################################################################################################

try:
    if TYPE_CHECKING:
        from QES.Algebra.hilbert                import HilbertSpace
        from QES.general_python.algebra.utils   import Array
        from QES.general_python.lattices        import Lattice
        from QES.Algebra.Hilbert.hilbert_local  import LocalSpaceTypes

    from QES.Algebra.Operator.matrix            import GeneralMatrix
    from QES.Algebra.Operator.operator_function import OperatorFunction, OperatorTypeActing, make_jax_operator_closure, _PYTHON_SCALARS
    from QES.Algebra.Operator.adapters          import NumpyAdapter, JaxAdapter
except ImportError as e:
    raise ImportError(
        "QES modules are required for this module to function properly. Please ensure QES is installed."
    ) from e

####################################################################################################

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = np

####################################################################################################


class SymmetryGenerators(Enum):
    """
    Available symmetry generators for symmetry analysis.
    """

    # lattice / local operators
    E             = auto()
    Translation_x = auto()
    Translation_y = auto()
    Translation_z = auto()
    Reflection    = auto()
    Inversion     = auto()  # spatial inversion (general lattice)
    ParityX       = auto()  # spin-only (sigma-x parity)
    ParityY       = auto()  # spin-only (sigma-y parity)
    ParityZ       = auto()  # spin-only (sigma-z parity)

    # fermion-specific
    FermionParity = auto()  # (-1)^{N}
    ParticleHole  = auto()  # PH transform
    TimeReversal  = auto()  # optional placeholder, depends on model

    # other symmetries - fallback
    Other         = auto()

    # ---------------
    #! HASERS
    # ---------------

    def has_translation(self):
        return self in [
            SymmetryGenerators.Translation_x,
            SymmetryGenerators.Translation_y,
            SymmetryGenerators.Translation_z,
        ]

    def has_reflection(self):
        return self in [SymmetryGenerators.Reflection]

    def has_inversion(self):
        return self in [SymmetryGenerators.Inversion]

    def supported_kind(self) -> Set["LocalSpaceTypes"]:
        """Return the set of LocalSpaceTypes supported by this symmetry generator"""

        from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes

        if self in [
            SymmetryGenerators.ParityX,
            SymmetryGenerators.ParityY,
            SymmetryGenerators.ParityZ,
        ]:
            return {LocalSpaceTypes.SPIN_1_2, LocalSpaceTypes.SPIN_1}
        elif self in [SymmetryGenerators.FermionParity, SymmetryGenerators.ParticleHole]:
            return {LocalSpaceTypes.SPINLESS_FERMIONS}
        else:
            return set()

    # -----------


class GlobalSymmetries(Enum):
    """
    Global symmetries for representing different symmetry groups.
    """

    U1 = 1
    Z2_PARITY = 2
    Other = auto()


####################################################################################################

class Operator(GeneralMatrix):
    """
    A class to represent a general operator acting on a Hilbert space.

    Inherits from GeneralMatrix to gain matrix storage, diagonalization,
    backend handling, and eigenvalue/eigenvector management.

    Attributes:
        - op_fun (OperatorFunction):
            The operator function that defines the operator.
        - fun_int (Callable):
            The function defining the operator for integer states.
        - fun_np (Optional[Callable]):
            The function defining the operator for NumPy array states.
        - fun_jnp (Optional[Callable]):
            The function defining the operator for JAX array states.
        - eigval (float):
            The eigenvalue of the operator (distinct from matrix eigenvalues).
        - lattice (Optional[Lattice]):
            The lattice object representing the physical system.
        - ns (Optional[int]):
            The number of sites in the system.
        - typek (Optional[SymmetryGenerators]):
            The symmetry generators of the operator.
        - name (str):
            The name of the operator.
        - modifies (bool):
            Flag for the operator that modifies the state.
        - quadratic (bool):
            Flag for the quadratic operator.
        - backend (str):
            The backend for the operator (default is 'default').
        - backend_sp (str):
            The backend for the operator (default is 'default').
    Examples:
        >>> op = Operator(fun_int=my_operator_function, ns=4, name='MyOperator')
        >>> print(op)
        Operator(MyOperator, type_acting=Global, eigval=1.0, type=Other)
        >>> op_result = op.op_fun.apply(state)

        >>> # Matrix representation and diagonalization (inherited from GeneralMatrix)
        >>> matrix = op.matrix(dim=hilbert.nh, hilbert_1=hilbert)
        >>> op.diagonalize()  # Now available for all operators
    """

    _INVALID_OPERATION_TYPE_ERROR   = "Invalid type for function. Expected a callable function."
    _INVALID_SYSTEM_SIZE_PROVIDED   = (
        "Invalid system size provided. Number of sites or a lattice object must be provided."
    )
    _INVALID_FUNCTION_NONE          = "Invalid number of necessary arguments for the operator function."

    #################################

    def __init__(
        self,
        op_fun                : OperatorFunction = None,
        fun_int               : Callable = None,
        fun_np                : Optional[Callable] = None,
        fun_jnp               : Optional[Callable] = None,
        eigval                = 1.0,
        lattice               : Optional["Lattice"] = None,
        ns                    : Optional[int] = None,
        typek                 : Optional[SymmetryGenerators] = SymmetryGenerators.Other,
        name                  : str = "Operator",
        modifies              : bool = True,
        quadratic             : bool = False,
        backend               : str = "default",
        is_sparse             : bool = True,
        dtype                 : Optional[Union[str, np.dtype]] = None,
        logger                : Optional[Any] = None,
        seed                  : Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the Operator object.

        Parameters
        ----------
        op_fun : OperatorFunction, optional
            Pre-built operator function object.
        fun_int : callable, optional
            The function defining the operator on integer states.
            Signature: `f(state, *args) -> (new_state, value)`.
        fun_np : callable, optional
            The function defining the operator on NumPy arrays.
        fun_jnp : callable, optional
            The function defining the operator on JAX arrays.
        eigval : float, default=1.0
            Scalar multiplier for the operator.
        lattice : Lattice, optional
            The lattice object defining the geometry.
        ns : int, optional
            The number of sites. Required if lattice is not provided.
        typek : SymmetryGenerators, optional
            The symmetry type of the operator.
        name : str, default="Operator"
            Name of the operator.
        modifies : bool, default=True
            Whether the operator modifies the basis states (e.g. flip spin).
        quadratic : bool, default=False
            Whether the operator is quadratic (single-particle).
        backend : str, default="default"
            Computation backend ('numpy', 'jax').
        is_sparse : bool, default=True
            Whether to use sparse matrix representation.
        dtype : dtype, optional
            Data type for matrix elements.
        logger : Logger, optional
            Logger instance.
        seed : int, optional
            Random seed for reproducibility.
        """

        # handle the system physical size dimension and the lattice
        if lattice is None and ns is not None:
            _ns = ns
            _lattice = lattice
        elif lattice is not None:
            _lattice = lattice
            _ns = _lattice.ns
        else:
            raise ValueError(Operator._INVALID_SYSTEM_SIZE_PROVIDED)

        # Filter out Operator-specific kwargs before passing to GeneralMatrix
        _operator_only_kwargs = {
            "acton",
            "necessary_args",
            "instr_code",
            "shape",
            "ns",
            "lattice",
            "_shape",
            "_ns",
            "_lattice",
            "backend",
            "is_sparse",
            "logger",
            "seed",
            "dtype",
            "op_fun",
            "fun_int",
            "fun_np",
            "fun_jax",
            "fun_jnp",  # function kwargs handled by Operator
        }
        _general_matrix_kwargs  = {k: v for k, v in kwargs.items() if k not in _operator_only_kwargs}

        # Initialize GeneralMatrix parent class
        # Shape is set to (0, 0) initially - will be set when matrix is built
        GeneralMatrix.__init__(
            self,
            shape               = (0, 0),     # Will be determined when matrix is built
            ns                  = _ns,        # number of sites - from lattice or provided
            is_sparse           = is_sparse,  # sparse matrix representation
            backend             = backend,    # backend
            logger              = logger,     # logger, already initialized or None
            seed                = seed,       # random seed
            dtype               = dtype,      # data type for matrix elements
            **_general_matrix_kwargs,
        )

        # Store lattice (not in GeneralMatrix, specific to Operator)
        self._lattice           = _lattice

        # property of the operator itself
        self._eigval            = eigval      #! operator's eigenvalue (NOT matrix eigenvalue)
        self._opeigval          = eigval      # backward compatibility
        self._name              = name        # the name of the operator
        self._type              = typek
        if self._type != SymmetryGenerators.Other and self._name == "Operator":
            self._name = self._type.name

        # property for the behavior of the operator - e.g., quadratic, action, etc.
        self._quadratic         = quadratic   # flag for the quadratic operator - this enables different matrix representation
        self._acton             = kwargs.get("acton", False)  # flag for the action of the operator on the local physical space
        self._modifies          = modifies    # flag for the operator that modifies the state
        self._matrix_fun        = None        # the function that defines the matrix form of the operator - if not provided, the matrix is generated from the function fun
        self._necessary_args    = kwargs.get("necessary_args", 0)  # number of necessary arguments for the operator function
        self._fun               = (
                                  None  # the function that defines the operator - it is set to None if not provided
                                )
        self._jit_wrapper_cache = {}  # cache for JIT wrappers

        #! IMPORTANT
        self._instr_code        = kwargs.get("instr_code", None)    # instruction code for the operator - used in operator builder - linear algebraic operations
        self._init_functions(op_fun, fun_int, fun_np, fun_jnp)      # initialize the operator function

        # Initialize backend adapter
        if self._is_jax:
            self._adapter = JaxAdapter()
        else:
            self._adapter = NumpyAdapter()

    def __repr__(self):
        """
        String representation of the operator.
        """
        eigval_str    = f"eigval={self._eigval:.4g}" if not np.isclose(self._eigval, 1.0) else ""
        type_str      = f"type={self._type.name}" if self._type != SymmetryGenerators.Other else ""
        type_act_str  = f"act={self.type_acting.name}"
        name_str      = f"{self._name}" if self._name else "Operator"
        props         = ",".join(filter(None, [type_act_str, eigval_str, type_str]))
        return f"{name_str}({props})"

    #################################
    #! Initialize functions
    #################################

    def _init_functions(self, op_fun=None, fun_int=None, fun_np=None, fun_jax=None):
        """
        Initializes the operator functions and determines the operator type based on the
        number of necessary arguments.

        ---
        Parameters:
            op_fun (OperatorFunction, optional):
                An instance of `OperatorFunction`. If provided,
                it is directly assigned to the operator.
            fun_int (callable, optional):
                A Python function representing the internal implementation
                of the operator. Must be provided if `op_fun` is not specified.
            fun_np (callable, optional):
                A NumPy-compatible implementation of the operator function.
            fun_jax (callable, optional):
                A JAX-compatible implementation of the operator function.
        Raises:
            ValueError: If both `op_fun` and `fun_int` are `None`.
            NotImplementedError: If the number of necessary arguments exceeds 2.
        Notes:
            - The `necessary_args` attribute is determined based on the number of arguments
                required by `fun_int`, excluding the first argument (assumed to be `self`).
            - The operator type (`_type_acting`) is set based on the number of necessary arguments:
                - 0 arguments: `OperatorTypeActing.Global`
                - 1 argument: `OperatorTypeActing.Local`
                - 2 arguments: `OperatorTypeActing.Correlation`
        """

        if op_fun is not None and isinstance(op_fun, OperatorFunction):
            self._fun = op_fun
            self._necessary_args = op_fun.necessary_args
        else:
            if fun_int is None:
                # Allow initialization without function (e.g. for Hamiltonian base class)
                # The function must be set later before use.
                self._fun = None
                return

            # get the necessary args
            self._necessary_args = fun_int.__code__.co_argcount - 1
            self._fun = OperatorFunction(
                fun_int,
                fun_np,
                fun_jax,
                modifies_state=self._modifies,
                necessary_args=self._necessary_args,
            )
        # set the operator function type
        if self._necessary_args == 0:
            self._type_acting = OperatorTypeActing.Global
        elif self._necessary_args == 1:
            self._type_acting = OperatorTypeActing.Local
        elif self._necessary_args == 2:
            self._type_acting = OperatorTypeActing.Correlation
        else:
            raise NotImplementedError()

    #################################
    #! GeneralMatrix interface implementation
    #################################

    def build(
        self,
        dim: int = None,
        hilbert: "HilbertSpace" = None,
        verbose: bool = False,
        force: bool = False,
        **kwargs,
    ) -> None:
        """
        Build the matrix representation of the operator and store it.

        This implements the GeneralMatrix.build() interface, allowing
        operators to participate in the same workflow as Hamiltonians and other matrix-based objects.

        Parameters
        ----------
        dim : int, optional
            Dimension of the matrix. Required if hilbert is not provided.
        hilbert : HilbertSpace, optional
            Hilbert space for matrix construction. Takes precedence over dim.
        verbose : bool, default False
            Whether to print progress messages.
        force : bool, default False
            If True, rebuild even if matrix already exists.
        **kwargs
            Additional arguments passed to matrix() method.
        """
        kwargs.pop("verbose", None)

        if self._matrix is not None and not force:
            self._log("Matrix already built. Use force=True to rebuild.", lvl=1)
            return

        if hilbert is not None:
            dim = hilbert.nh
        elif dim is None:
            raise ValueError("Either 'dim' or 'hilbert' must be provided to build the matrix.")

        # Build the matrix using existing compute_matrix() method
        matrix_type = "sparse" if self._is_sparse else "dense"
        # Remove verbose from kwargs if present to avoid duplication

        built_matrix = self.compute_matrix(
            dim=dim, matrix_type=matrix_type, hilbert_1=hilbert, verbose=verbose, **kwargs
        )

        # Store in GeneralMatrix's storage
        self._set_matrix_reference(built_matrix)
        self.set_matrix_shape((dim, dim))

        if verbose:
            self._log(f"Operator matrix built with shape {self._shape}", lvl=1, color="green")

    def _get_diagonalization_matrix(self):
        """
        Get the matrix to use for diagonalization.

        For Operator, this returns the stored matrix from build().
        """
        if self._matrix is None:
            raise ValueError("Matrix not built. Call build() first before diagonalizing.")
        return self._matrix

    def _matvec_context(self) -> tuple:
        """
        Provide context for matvec operations.

        Returns tuple of (args, kwargs) to pass to matvec.
        """
        # For Operator, we may need a Hilbert space for matvec
        # This can be overridden by subclasses
        return (), {"hilbert": None}

    #################################
    #! Static methods
    #################################

    @staticmethod
    def idn(state, *args):
        """
        Identity operator function.
        """
        return (state,), (1.0,)

    @staticmethod
    def idn_f(state, *args):
        """
        Identity operator function.
        """
        return 1.0

    #################################
    #! Copying and cloning
    #################################

    def copy(self):
        """
        Create a copy of the operator.
        """
        return copy.deepcopy(self)

    def clone(self):
        """
        Clone the operator.
        """
        return Operator(**self.__dict__)

    #################################
    #! Operators that modify the operator class itself
    #################################

    def _multiply_const(self, constant):
        """
        Multiply the operator by a constant value and return a new Operator.

        Parameters:
            constant (numeric): The constant value to multiply with the operator.

        Returns:
            Operator: A new Operator instance with the multiplied function and eigenvalue.
        """
        new_kwargs = self.__dict__.copy()
        new_kwargs.pop("_fun", None)
        new_kwargs.pop("_name", None)
        new_kwargs.pop("_eigval", None)

        new_fun = self._fun._multiply_const(constant)
        new_eigval = self._eigval * constant
        new_name = f"({constant} * {self._name})"

        return Operator(
            op_fun=new_fun,
            name=new_name,
            eigval=new_eigval,
            ns=new_kwargs["_ns"],
            lattice=new_kwargs["_lattice"],
            modifies=new_kwargs["_modifies"],
            backend=new_kwargs["_backend_str"],
            **new_kwargs,
        )

    def __imul__(self, scalar):
        """*= Operator for a general operator"""
        self._fun = self._fun * scalar
        return self

    def __itruediv__(self, scalar):
        """/= Operator with Division by Zero Check"""

        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__itruediv__")
            self._fun = self._fun * (1.0 / scalar)
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
        return self

    # -------------------------------

    def __mul__(self, other):
        new_kwargs = self.__dict__.copy()
        new_kwargs.pop("_fun", None)  # Remove _fun, it will be new
        new_kwargs.pop("_name", None)  # Remove _name, it will be new
        new_kwargs.pop("_eigval", None)  # Eigval handled separately or combined if meaningful
        dtype = new_kwargs.get("_dtype", None)

        if isinstance(other, Operator):
            new_fun     = self._fun * other._fun
            #! If eigvals are simple scalars, product makes sense.
            new_eigval  = self._eigval * other._eigval
            new_name    = f"({self._name} * {other._name})"
            # Modifies state is handled by OperatorFunction composition
            # Quadratic, acton etc. might need rules for combining
            # For now, inherit from self, or define combination rules
            new_kwargs["quadratic"] = self._quadratic or other._quadratic  # ? what to do here?
        elif isinstance(other, _PYTHON_SCALARS) or (
            JAX_AVAILABLE and isinstance(other, jax.Array) and other.ndim == 0
        ):
            new_fun     = self._fun * other
            new_eigval  = self._eigval * other
            new_name    = f"({self._name} * {other})"
        else:
            return NotImplementedError("Incompatible operator function")
        return Operator(
            op_fun      = new_fun,
            name        = new_name,
            eigval      = new_eigval,
            ns          = new_kwargs["_ns"],
            lattice     = new_kwargs["_lattice"],
            modifies    = new_kwargs["_modifies"],
            backend     = new_kwargs["_backend"],
            logger      = new_kwargs.get("_logger", None),
            **new_kwargs,
        )

    def __rmul__(self, other):
        new_kwargs = self.__dict__.copy()
        new_kwargs.pop("_fun", None)
        new_kwargs.pop("_name", None)
        new_kwargs.pop("_eigval", None)

        if isinstance(other, _PYTHON_SCALARS) or (
            JAX_AVAILABLE and isinstance(other, jax.Array) and other.ndim == 0
        ):
            new_fun     = other * self._fun  # OperatorFunction scalar rmul (same as mul)
            new_eigval  = other * self._eigval
            new_name    = f"({other} * {self._name})"
        elif isinstance(other, Operator):
            new_fun     = other._fun * self._fun
            #! If eigvals are simple scalars, product makes sense.
            new_eigval  = other._eigval * self._eigval
            new_name    = f"({other._name} * {self._name})"
        else:
            return NotImplementedError("Incompatible operator function")

        return Operator(op_fun=new_fun, name=new_name, eigval=new_eigval, **new_kwargs)

    # -------------------------------

    def __truediv__(self, scalar):
        """
        Division of the operator by a scalar.
        """

        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__truediv__")
            return self._multiply_const(1.0 / scalar)
        elif isinstance(scalar, Operator):
            raise NotImplementedError("Division of two operators is not implemented.")
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
        return None

    def __rtruediv__(self, scalar):
        """
        Division of a scalar by the operator.
        """
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__rtruediv__")
            return self._multiply_const(scalar)
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
        return None

    # -------------------------------

    def __add__(self, other):
        """
        Addition of two operators.

        Parameters:
            other: Another Operator instance to add to this operator.

        Returns:
            Operator: A new Operator instance representing the sum.
        """
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot add Operator with type {type(other)}")

        new_kwargs = self.__dict__.copy()
        new_kwargs.pop("_fun", None)
        new_kwargs.pop("_name", None)
        new_kwargs.pop("_eigval", None)

        new_fun = self._fun + other._fun
        new_name = f"({self._name} + {other._name})"
        # Note: eigenvalue addition doesn't have clear physical meaning for general operators
        # Set to None or keep the first operator's eigenvalue
        new_eigval = None

        return Operator(
            op_fun=new_fun,
            name=new_name,
            eigval=new_eigval,
            ns=new_kwargs["_ns"],
            lattice=new_kwargs["_lattice"],
            modifies=new_kwargs["_modifies"],
            backend=new_kwargs["_backend_str"],
            **new_kwargs,
        )

    def __sub__(self, other):
        """
        Subtraction of two operators.

        Parameters:
            other: Another Operator instance to subtract from this operator.

        Returns:
            Operator: A new Operator instance representing the difference.
        """
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot subtract type {type(other)} from Operator")

        new_kwargs = self.__dict__.copy()
        new_kwargs.pop("_fun", None)
        new_kwargs.pop("_name", None)
        new_kwargs.pop("_eigval", None)

        new_fun = self._fun - other._fun
        new_name = f"({self._name} - {other._name})"
        # Note: eigenvalue subtraction doesn't have clear physical meaning for general operators
        # Set to None or keep the first operator's eigenvalue
        new_eigval = None

        return Operator(
            op_fun=new_fun,
            name=new_name,
            eigval=new_eigval,
            ns=new_kwargs["_ns"],
            lattice=new_kwargs["_lattice"],
            modifies=new_kwargs["_modifies"],
            backend=new_kwargs["_backend_str"],
            **new_kwargs,
        )

    #################################
    #! Setters and Getters
    #################################

    @property
    def eigval(self):
        return self._eigval

    @property
    def opeigval(self):
        return self._eigval

    @eigval.setter
    def eigval(self, val):
        self._eigval = val

    @opeigval.setter
    def opeigval(self, val):
        self._eigval = val

    # -------------------------------
    # Operator-specific properties
    # -------------------------------

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, val):
        self._lattice = val

    @property
    def ns(self):
        return self._ns

    @ns.setter
    def ns(self, val):
        self._ns = val

    @property
    def sites(self):
        return self.ns

    # -------------------------------

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    # -------------------------------

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, val):
        self._type = val

    # -------------------------------

    @property
    def code(self):
        return self._instr_code

    @code.setter
    def code(self, val):
        self._instr_code = val

    # -------------------------------

    @property
    def quadratic(self):
        return self._quadratic

    @quadratic.setter
    def quadratic(self, val):
        self._quadratic = val

    # -------------------------------

    @property
    def acton(self):
        return self._acton

    @acton.setter
    def acton(self, val):
        self._acton = val

    # -------------------------------

    @property
    def modifies(self):
        return self._modifies

    @modifies.setter
    def modifies(self, val):
        self._modifies = val

    # -------------------------------

    @property
    def type_acting(self):
        return self._type_acting

    def get_acting_type(self):
        return self._type_acting

    # -------------------------------

    @property
    def fun(self):
        return self._fun

    @fun.setter
    def fun(self, val):
        self._fun = val

    @property
    def int(self):
        self._backend = np
        return self._fun.fun

    @property
    def npy(self):
        self._backend = np
        return self._fun.npy

    @property
    def jax(self):
        self._backend = jnp
        return self._fun.jax

    @property
    def matvec_fun(self):
        """
        Returns a pre-optimized matrix-vector multiplication function.
        Pre-allocates thread buffers only when multithreaded=True to avoid wasting memory.
        """

        # Pre-allocate thread buffers once for reuse (critical for memory efficiency in Lanczos/Arnoldi)
        _thread_buffer_cache = None

        def _matvec(x, *args, hilbert=None, **kwargs):
            nonlocal _thread_buffer_cache

            # Only allocate thread buffer if multithreaded mode is requested
            multithreaded = kwargs.get("multithreaded", False)

            if multithreaded and hilbert is not None and _thread_buffer_cache is None:
                try:
                    from QES.Algebra.Hilbert.matrix_builder import ensure_thread_buffer

                    nh = hilbert.nh
                    chunk_size = kwargs.get("chunk_size", 6)
                    dtype = kwargs.get("dtype", np.complex128)
                    _thread_buffer_cache = ensure_thread_buffer(nh, chunk_size, dtype=dtype)
                except ImportError:
                    pass  # Fall back to per-call allocation if imports unavailable

            # Pass pre-allocated buffer to avoid reallocation (only if allocated)
            if _thread_buffer_cache is not None and "thread_buffer" not in kwargs:
                kwargs["thread_buffer"] = _thread_buffer_cache

            return self.matvec(x, *args, hilbert_in=hilbert, **kwargs)

        return _matvec

    # -------------------------------

    def override_matrix_function(self, function: Callable):
        """
        Override the matrix function of the operator.

        Args:
            function (Callable): The new matrix function to set.

        Example:
            >>> def new_matrix_fun(state):
            >>>     # Define the new matrix function here
            >>>     pass
            >>> op.override_matrix_function(new_matrix_fun)
        """
        self._matrix_fun = function

    #################################
    #! Apply the operator
    #################################

    def _apply_global(self, states):
        """
        Applies a function to a given state or a collection of states.

        This method processes either a single state or an iterable of states,
        applying the `_fun` function to each state. If the input is a single
        state, the result of `_fun` is returned directly. If the input is a
        collection of states, the method returns two lists: one containing
        the transformed states and the other containing the corresponding
        values.

        Args:
            states (Union[int, list, np.ndarray, jnp.ndarray]):
                A single state or a collection of states to which the `_fun`
                function will be applied. Can be an integer, list, numpy
                array, or jax.numpy array.

        Returns:
            tuple[list, list]:
                Two lists: the first contains the transformed states, the second contains the corresponding values.
        """
        if (hasattr(states, "shape") and len(states.shape) <= 1) or isinstance(
            states, (int, np.integer)
        ):
            # if the state is a single state (scalar or 0-d/1-d array), apply the function directly
            st, val = self._fun(states)
            return st, self._backend.asarray(val) * self._eigval

        # if the state is a collection of states, apply the function to each state
        results = [self._fun(state) for state in states]
        out, val = zip(*results) if results else ([], [])
        return (
            (self._backend.stack(list(out)), self._backend.stack([v * self._eigval for v in val]))
            if list(out)
            else (self._backend.array([]), self._backend.array([]))
        )

    def _apply_local(self, states, i):
        """
        Applies a local operation to a given state or a collection of states.
        Parameters:
            states (Union[int, list, np.ndarray, jnp.ndarray]): The input state(s) to which the operation is applied.
                Can be a single integer, a list of integers, or a NumPy/JAX array.
            i (int): The index or parameter used by the local operation.
        Returns:
            tuple[list, list]: Two lists: the resulting states and the corresponding values.
        """
        if (hasattr(states, "shape") and len(states.shape) <= 1) or isinstance(
            states, (int, np.integer)
        ):
            # if the state is a single state (scalar or 0-d/1-d array), apply the function directly
            st, val = self._fun(states, i)
            return st, self._backend.asarray(val) * self._eigval
        results = [self._fun(state, i) for state in states]
        out, val = zip(*results) if results else ([], [])
        return (
            (self._backend.stack(list(out)), self._backend.stack([v * self._eigval for v in val]))
            if list(out)
            else (self._backend.array([]), self._backend.array([]))
        )

    def _apply_correlation(self, states, i, j):
        """
        Applies a correlation function to a given state or a collection of states.
        Parameters:
            states (Union[int, list, np.ndarray, jnp.ndarray]): The input state(s) to which the correlation function
                will be applied. Can be a single integer, a list of integers, or a NumPy/JAX array of states.
            i (int): The first index parameter for the correlation function.
            j (int): The second index parameter for the correlation function.
        Returns:
            tuple[list, list]: Two lists: the output states and the corresponding values.
        """
        if (hasattr(states, "shape") and len(states.shape) == 1) or isinstance(
            states, (int, np.int8, np.int16, np.int32, np.int64)
        ):
            # if the state is a single state, apply the function directly
            st, val = self._fun(states, i, j)
            return st, self._backend.asarray(val) * self._eigval

        results = [self._fun(state, i, j) for state in states]
        out, val = zip(*results) if results else ([], [])
        return (
            (self._backend.stack(list(out)), self._backend.stack([v * self._eigval for v in val]))
            if list(out)
            else (self._backend.array([]), self._backend.array([]))
        )

    def apply(self, states: list | Array, *args):
        """
        Apply the operator to a state or batch of states.

        Parameters
        ----------
        states : list or array-like
            The state(s) to which the operator is applied. Can be:
            - Integer (basis state index)
            - NumPy/JAX array (dense vector or batch of vectors)
        *args : int
            Additional arguments specifying the site(s) on which the operator acts.
            - Global operator: No args.
            - Local operator: `site_index`.
            - Correlation operator: `site_i, site_j`.

        Returns
        -------
        tuple
            (new_states, coefficients)
            - new_states: The resulting states after operator application.
            - coefficients: The matrix elements (amplitudes) for the transitions.
        """
        if self._type_acting.is_global():
            return self._apply_global(states)
        elif self._type_acting.is_local():
            return self._apply_local(states, *args)
        elif self._type_acting.is_correlation():
            return self._apply_correlation(states, *args)
        else:
            raise NotImplementedError("Invalid operator acting type.")
        return [None], [0.0]

    # -------------------------------
    #! Operator call interface
    # -------------------------------

    def __call__(self, states: list | Array, *args):
        """
        Apply the operator to the state.

        Args:
            states            : list of states to which the operator is applied.
            args              : Additional arguments for the operator - inform how to act on a state.
                                If there no arguments, the operator acts on the state as a whole - global operator.
                                If there are arguments, the operator acts on the state locally - local operator (e.g., site-dependent).
        """
        return self.apply(states, *args)

    def __getitem__(
        self,
        states: list | Array,
    ):
        """
        Apply the operator to the state - returns modified state only.

        Args:
            states:
                list of states to which the operator is applied.

        Returns:
            list: The first element is the transformed state and the second element is the value - thus only the modified state is returned.
        """
        if isinstance(states, tuple):
            return self.apply(states[0], *states[1:])[0]
        return self.apply(states)[0]

    def __mod__(self, other):
        """
        Apply the operator to a given state and return the modified state values.
        """
        if isinstance(other, tuple):
            return self.apply(other[0], *other[1:])[1]
        return self.apply(other)[1]

    #################################
    #! Generate matrix form of the operator
    #################################

    def _matrix_no_hilbert_np(
        self,
        dim: int,
        is_sparse: bool,
        wrapped_funct,
        dtype,
        max_loc_upd: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate the matrix form of the operator without Hilbert space.
        """
        # create a dummy Hilbert space for convenience
        from QES.Algebra.hilbert import HilbertSpace
        from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix

        dummy_hilbert = HilbertSpace(nh=dim, backend=self._backend)
        if verbose:
            dummy_hilbert._log("Calculating the Operator matrix using NumPy...", lvl=2)

        # calculate the time to create the matrix
        t1 = time.time()
        matrix = build_operator_matrix(
            hilbert_space=dummy_hilbert,
            operator_func=wrapped_funct,
            sparse=is_sparse,
            max_local_changes=max_loc_upd,
            dtype=dtype,
        )
        time_taken = time.time() - t1
        if verbose:
            dummy_hilbert._log(
                f"Time taken to create the matrix {self._name}: {time_taken:.2e} seconds", lvl=2
            )
        return matrix

    @property
    def matrix(self):
        """
        Return the stored matrix representation of the operator.
        Requires build() to be called first.
        """
        return self._matrix

    def compute_matrix(
        self,
        *args,
        dim=None,
        matrix_type="sparse",
        dtype=None,
        hilbert_1=None,
        hilbert_2=None,
        use_numpy: bool = True,
        **kwargs,
    ) -> Array | None:
        """
        Generates the matrix representation of the operator.

        Parameters
        ----------
        dim : int, optional
            The dimension of the matrix. Required if Hilbert spaces are not provided.
        matrix_type : str, default='sparse'
            The type of matrix to generate ('sparse' or 'dense').
        dtype : dtype, optional
            The data type for matrix elements. Defaults to complex128.
        hilbert_1 : HilbertSpace, optional
            The source Hilbert space.
        hilbert_2 : HilbertSpace, optional
            The target Hilbert space (if different).
        use_numpy : bool, default=True
            Whether to force NumPy backend for matrix construction.
        **kwargs : dict
            Additional arguments passed to the matrix builder.

        Returns
        -------
        array-like
            The matrix representation (scipy.sparse.csr_matrix or numpy/jax array).
        """

        # Check if the operator instance has attached Hilbert space (e.g. from Hamiltonian)
        if hilbert_1 is None and hilbert_2 is None:
            if hasattr(self, "hilbert_space") and self.hilbert_space is not None:
                hilbert_1 = self.hilbert_space
            elif hasattr(self, "_hilbert_space") and self._hilbert_space is not None:
                hilbert_1 = self._hilbert_space

        # check the dimension of the matrix
        dim1, dim2 = None, None
        matrix_hilbert = "None"
        if hilbert_1 is not None and hilbert_2 is not None:
            dim1, dim2 = hilbert_1.nh, hilbert_2.nh
            matrix_hilbert = "double"
        elif hilbert_1 is not None and hilbert_2 is None:
            dim1, dim2 = hilbert_1.nh, hilbert_1.nh
            matrix_hilbert = "single"
        elif hilbert_1 is None and hilbert_2 is not None:
            hilbert_1 = hilbert_2
            dim1, dim2 = hilbert_2.nh, hilbert_2.nh
            matrix_hilbert = "single"
        else:
            if dim is None:
                raise ValueError("Dimension or at least one Hilbert space must be provided.")
            dim1, dim2 = dim, dim
            matrix_hilbert = "None"

        verbose = kwargs.pop("verbose", False)

        # check if there are functions from the Hilbert space
        jax_maybe_av = JAX_AVAILABLE and self._backend != np
        is_sparse = matrix_type == "sparse"
        use_numpy = use_numpy or (not jax_maybe_av)

        # check if the matrix function is provided and skips kwargs if unnecessary
        if self._matrix_fun is not None:
            if is_sparse:
                return self._matrix_fun(dim1, matrix_type, *args)
            else:
                return self._backend.asarray(
                    self._matrix_fun(dim1, matrix_type, *args), dtype=dtype
                )

        # Create JIT wrapper for matrix builder
        # This ensures we pass a JIT-compiled function to the JIT-compiled builder
        # wrap the function with necessary parameters
        dtype = dtype if dtype is not None else self._backend.complex128
        max_loc_upd = kwargs.get("max_loc_upd", 1)

        if self._fun is None:
             raise RuntimeError(f"Operator function is not initialized for '{self._name}'.")

        op_int_jit = self._fun._fun_int

        if op_int_jit is None:
            # If function is not available, we cannot proceed with JIT matrix building
            # Try to see if we can use the python function? But matrix_builder requires JIT.
            raise RuntimeError(f"Operator integer function (JIT) is None. Operator compilation likely failed for '{self._name}'. check logs for details.")

        # Check cache first
        cache_key = args if not any(isinstance(a, (np.ndarray, list, dict)) for a in args) else None
        op_wrapper_jit = None

        if cache_key is not None and cache_key in self._jit_wrapper_cache:
            op_wrapper_jit = self._jit_wrapper_cache[cache_key]
        else:

            @numba.njit
            def op_wrapper_jit(state):
                return op_int_jit(state, *args)

            if cache_key is not None:
                self._jit_wrapper_cache[cache_key] = op_wrapper_jit

        # Case1: easiest case - no Hilbert space provided
        if matrix_hilbert == "None":
            # maximum local updates - how many states does the operator create - for sparse
            return self._matrix_no_hilbert_np(
                dim1, is_sparse, op_wrapper_jit, dtype, max_loc_upd, verbose, **kwargs
            )

        # Case2: one Hilbert space provided
        elif matrix_hilbert == "single":
            if not jax_maybe_av or use_numpy:
                from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix

                if verbose:
                    hilbert_1.log(
                        f"Calculating the operator matrix {self._name} using NumPy with single Hilbert space...",
                        lvl=2,
                    )

                t1 = time.time()
                matrix = build_operator_matrix(
                    operator_func=op_wrapper_jit,
                    hilbert_space=hilbert_1,
                    sparse=is_sparse,
                    max_local_changes=max_loc_upd,
                    dtype=dtype,
                    nh=hilbert_1.nh if hilbert_1 is not None else dim1,
                    ns=hilbert_1.ns if hilbert_1 is not None else None,
                )
                if verbose:
                    hilbert_1.log(
                        f"Time taken to create the matrix {self._name}: {time.time() - t1:.2e} seconds",
                        lvl=2,
                    )
                return matrix
            else:
                raise NotImplementedError(
                    "JAX backend for single Hilbert space matrix construction is not yet implemented."
                )

        # Case3: two Hilbert spaces provided
        elif matrix_hilbert == "double":
            if not jax_maybe_av or use_numpy:
                from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix

                if verbose:
                    if hasattr(hilbert_1, "log"):
                        hilbert_1.log(
                            f"Calculating the operator matrix {self._name} using NumPy with two Hilbert spaces...",
                            lvl=2,
                        )

                t1 = time.time()
                matrix = build_operator_matrix(
                    operator_func=op_wrapper_jit,
                    hilbert_space=hilbert_1,
                    hilbert_space_out=hilbert_2,
                    sparse=is_sparse,
                    max_local_changes=max_loc_upd,
                    dtype=dtype,
                    nh=hilbert_1.nh if hilbert_1 is not None else dim1,
                    ns=hilbert_1.ns if hilbert_1 is not None else None,
                )
                if verbose:
                    if hasattr(hilbert_1, "log"):
                        hilbert_1.log(
                            f"Time taken to create the matrix {self._name}: {time.time() - t1:.2e} seconds",
                            lvl=2,
                        )
                return matrix
            else:
                raise NotImplementedError(
                    "JAX backend for two Hilbert spaces matrix construction is not yet implemented."
                )
        else:
            raise ValueError("Invalid Hilbert space provided.")
        return None

    #################################

    def standardize_matrix(self, matrix):
        """
        Standardizes the given matrix representation of the operator.

        This method can be used to apply normalization, symmetrization, or other
        standardization procedures to a matrix representation of the operator.
        Currently, this is a placeholder for future implementation.

        Parameters:
            matrix: The matrix to standardize (sparse or dense array).

        Returns:
            The standardized matrix (currently returns the input unchanged).

        Note:
            This method is reserved for future extensions that may require
            matrix standardization based on specific operator properties or
            symmetries. Override in subclasses if needed.
        """
        # Placeholder - return matrix unchanged
        # Future implementations might include:
        # - Normalization by trace or norm
        # - Symmetrization for hermitian operators
        # - Removal of numerical noise below threshold
        return matrix

    #################################

    @staticmethod
    def apply_sequence(
        vecs: Array,
        operators: List["Operator"],
        hilbert: Optional["HilbertSpace"] = None,
        *,
        symmetry_mode: str = "auto",
        out: Optional[Array] = None,
        tmp: Optional[Array] = None,
        **kwargs,
    ) -> Array:

        if not operators:
            return vecs

        is_1d = vecs.ndim == 1
        A = vecs[:, None] if is_1d else vecs

        nh, n_batch = A.shape
        B = out if out is not None else np.zeros((nh, n_batch), dtype=np.complex128)
        T = tmp if tmp is not None else np.zeros((nh, n_batch), dtype=np.complex128)

        cur = A
        nxt = B

        for i, op in enumerate(operators):
            nxt.fill(0.0)
            op.matvec(cur, hilbert=hilbert, out=nxt, symmetry_mode=symmetry_mode, **kwargs)
            cur, nxt = nxt, (T if nxt is B else B)

        if is_1d:
            return cur[:, 0]
        return cur

    # -----------------------------------
    #! Matrix-vector multiplication
    # -----------------------------------

    def matvec(
        self,
        vecs          : 'Array',
        *args,
        hilbert_in    : HilbertSpace  = None,
        hilbert_out   : HilbertSpace  = None,
        symmetry_mode : str           = "auto",
        multithreaded : bool          = False,
        out           : Array         = None,
        thread_buffer : Array         = None,
        chunk_size    : int           = 1,
        dtype         = None,
        # alias for hilbert_in
        hilbert: Optional[HilbertSpace] = None,
    ) -> Array:
        """
        Apply the operator matrix to a vector using the backend adapter.
        Delegates to self._adapter.matvec.
        """
        logger          = hilbert_in._logger if hilbert_in is not None else None
        hilbert_in      = hilbert_in if hilbert_in is not None else hilbert

        # No Hilbert space provided
        if hilbert_in is None:
            if isinstance(vecs, list) or (isinstance(vecs, np.ndarray) and vecs.ndim == 1 and "int" in str(vecs.dtype)):
                # Assuming vecs are state INDICES, not amplitudes
                return self.apply(vecs, *args)

            # For now assuming the user knows what they are doing.
            if logger is not None:
                logger.warning("Hilbert space not provided. Assuming single-particle picture for matvec.")

        # Runtime backend fallback: allow NumPy vectors even when operator backend is JAX.
        # This keeps ED/Lanczos post-processing paths working without forcing explicit
        # backend switches on the operator instance.
        adapter = self._adapter
        if isinstance(vecs, np.ndarray) and isinstance(adapter, JaxAdapter):
            adapter = NumpyAdapter()

        return adapter.matvec(
            self, vecs, hilbert_in, hilbert_out, *args,
            symmetry_mode=symmetry_mode, multithreaded=multithreaded,
            out=out, thread_buffer=thread_buffer, chunk_size=chunk_size, dtype=dtype
        )

    def matvec_fourier(
        self,
        phases        : Array,
        vec           : Array,
        hilbert       : HilbertSpace,
        *,
        symmetry_mode : str = "auto",
        multithreaded : bool = False,
        out           : Optional['Array'] = None,
        thread_buffer : Optional['Array'] = None,
        chunk_size    : int = 4,
    ) -> 'Array':
        r"""
        Computes |out> = O_q |in> using backend adapter.
        O_q = (1/sqrt(N)) * sum_j exp(i * k * r_j) * sigma_j
        """
        if hilbert is None:
            raise ValueError("Hilbert space must be provided for Fourier operator application.")

        return self._adapter.matvec_fourier(self, phases, vec, hilbert,
            symmetry_mode=symmetry_mode, multithreaded=multithreaded,
            out=out, thread_buffer=thread_buffer, chunk_size=chunk_size
        )

    #################################
    # K-space Transformation Methods
    #################################

    def _is_single_particle_matrix(self) -> bool:
        """
        Check if the current matrix has single-particle dimensions.

        A matrix is considered single-particle if its dimension equals:
        - ns (number of sites) for standard operators
        - 2*ns for BdG (Bogoliubov-de Gennes) operators with pairing

        Returns
        -------
        bool
            True if matrix has single-particle dimensions, False otherwise.
        """
        matrix = self._get_matrix_reference()
        if matrix is None:
            return False

        dim = matrix.shape[0]
        ns = (
            self._ns
            if self._ns is not None
            else (self._lattice.ns if self._lattice is not None else None)
        )

        if ns is None:
            return False

        return dim == ns or dim == 2 * ns

    def to_kspace(self, return_transform: bool = False, **kwargs):
        r"""
        Transform operator matrix to k-space (momentum space).

        This method transforms a real-space operator to k-space using Bloch's theorem.
        Only valid for operators with single-particle matrix dimensions (dim = ns or 2*ns).

        Parameters
        ----------
        return_transform : bool, optional
            If True, also return the Bloch unitary matrix W for operator transformations.
            Default is False.
        **kwargs
            Additional arguments passed to kspace_from_realspace (e.g., unitary_norm, use_cache)

        Returns
        -------
        O_k : np.ndarray
            K-space operator, shape (Lx, Ly, Lz, Nb, Nb) where Nb is bands per unit cell
        kgrid : np.ndarray
            K-points in Cartesian coordinates, shape (Lx, Ly, Lz, 3)
        kgrid_frac : np.ndarray
            K-points in fractional coordinates, shape (Lx, Ly, Lz, 3)
        W : np.ndarray, optional
            Bloch unitary matrix (only if return_transform=True), shape (Lx, Ly, Lz, Ns, Nb)

        Examples
        --------
        >>> # For a quadratic Hamiltonian or single-particle operator
        >>> H_k, kgrid, kgrid_frac = operator.to_kspace()
        >>>
        >>> # Get W for transforming other operators consistently
        >>> H_k, kgrid, kgrid_frac, W = operator.to_kspace(return_transform=True)
        >>> # Transform another operator: O_k = W^dagger @ O @ W

        Raises
        ------
        ValueError
            If matrix is not built, lattice is not available, or dimensions are wrong.

        Notes
        -----
        The operator matrix must have dimensions ns \times ns (single-particle) or
        2ns \times 2ns (BdG with particle-hole doubling). Many-body matrices with
        dimension 2^ns are NOT supported - use single-particle representations.
        """
        matrix = self._get_matrix_reference()

        if matrix is None:
            raise ValueError("Operator matrix not built. Call build() first.")

        if self._lattice is None:
            raise ValueError("Lattice information required for k-space transformation.")

        if not self._is_single_particle_matrix():
            dim = matrix.shape[0]
            ns = self._ns or self._lattice.ns
            raise ValueError(
                f"K-space transformation requires single-particle matrix dimensions. "
                f"Got dim={dim}, expected ns={ns} or 2*ns={2*ns}. "
                f"For many-body Hamiltonians (dim=2^ns), use quadratic representation."
            )

        from QES.general_python.lattices.tools.lattice_kspace import kspace_from_realspace

        # Convert sparse to dense if needed
        O_real = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        return kspace_from_realspace(
            self._lattice, O_real, return_transform=return_transform, **kwargs
        )

    def from_kspace(self, O_k, kgrid=None):
        """
        Transform k-space operator back to real space.

        Parameters
        ----------
        O_k : np.ndarray
            K-space operator, shape (Lx, Ly, Lz, Nb, Nb)
        kgrid : np.ndarray, optional
            K-point grid (not used, kept for API compatibility)

        Returns
        -------
        O_real : np.ndarray
            Real-space operator, shape (Ns, Ns)

        Examples
        --------
        >>> O_k, kgrid, kgrid_frac = operator.to_kspace()
        >>> # Modify O_k in k-space...
        >>> O_real_new = operator.from_kspace(O_k)

        Raises
        ------
        ValueError
            If lattice is not available.
        """
        if self._lattice is None:
            raise ValueError("Lattice information required for inverse k-space transformation.")

        from QES.general_python.lattices.tools.lattice_kspace import realspace_from_kspace

        return realspace_from_kspace(self._lattice, O_k, kgrid)

    def transform_to_kspace(self, return_grid: bool = True, **kwargs):
        """
        Alias for to_kspace() for backward compatibility.

        Parameters
        ----------
        return_grid : bool, optional
            If True (default), return k-grid along with transformed operator.
        **kwargs
            Additional arguments for kspace_from_realspace

        Returns
        -------
        O_k : np.ndarray
            K-space operator, shape (Lx, Ly, Lz, Nb, Nb)
        kgrid : np.ndarray, optional
            K-point grid (only if return_grid=True)
        kgrid_frac : np.ndarray, optional
            Fractional k-grid (only if return_grid=True)
        """
        result = self.to_kspace(**kwargs)

        if return_grid:
            return result  # Returns (O_k, kgrid, kgrid_frac) or with W
        else:
            return result[0]  # Returns only O_k

    #################################
    # Help and Documentation
    #################################

    @classmethod
    def help(cls, topic: Optional[str] = None) -> str:
        r"""
        Display help information about Operator capabilities.

        Parameters
        ----------
        topic : str, optional
            Specific topic to get help on. Options:
            - None or 'all': Full overview
            - 'types':
                Operator types (Global, Local, Correlation)
            - 'creation':
                Creating operators
            - 'application':
                Applying operators to states
            - 'matrix':
                Matrix representation
            - 'inherited':
                Methods inherited from GeneralMatrix

        Returns
        -------
        str
            Help text for the requested topic.

        Examples
        --------
        >>> Operator.help()  # Full overview
        >>> Operator.help('types')  # Operator types help
        """
        topics = {
            "types": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           Operator: Types                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OperatorTypeActing (determines how operator acts on states):                ║
║                                                                              ║
║  Global (0 site arguments):                                                  ║
║    - Acts on entire system at once                                           ║
║    - Example: Total spin Sₓ = \sum_i \sigma^x_i                              ║
║    - Usage: op(state)                                                        ║
║                                                                              ║
║  Local (1 site argument):                                                    ║
║    - Acts on a single specified site                                         ║
║    - Example: Local spin \sigma_x^i at site i                                ║
║    - Usage: op(state, site_index)                                            ║
║                                                                              ║
║  Correlation (2 site arguments):                                             ║
║    - Acts on pairs of sites                                                  ║
║    - Example: Two-site correlation \sigma_x^i \sigma_x^j                     ║
║    - Usage: op(state, site_i, site_j)                                        ║
║                                                                              ║
║  Properties:                                                                 ║
║    .type_acting              - Get the operator type                         ║
║    .necessary_args           - Number of site arguments needed               ║
║    .modifies_state           - Whether operator changes state                ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "creation": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         Operator: Creation                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Constructor:                                                                ║
║    Operator(fun_int, fun_np=None, fun_jax=None, ...)                         ║
║                                                                              ║
║  Key Parameters:                                                             ║
║    fun_int    : Callable - Integer state function (required)                 ║
║    fun_np     : Callable - NumPy array function   (optional)                 ║
║    fun_jax    : Callable - JAX array function     (optional)                 ║
║    eigval     : float    - Operator eigenvalue    (default: 1.0)             ║
║    modifies   : bool     - Whether operator modifies state                   ║
║    name       : str      - Operator name                                     ║
║    lattice    : Lattice  - Lattice for site information                      ║
║    ns         : int      - Number of sites        (if no lattice)            ║
║                                                                              ║
║  Function Signature (returns (new_states, coefficients)):                    ║
║    Global:      fun(state)        ->  (states_array, values_array)           ║
║    Local:       fun(state, i)     ->  (states_array, values_array)           ║
║    Correlation: fun(state, i, j)  ->  (states_array, values_array)           ║
║                                                                              ║
║  Factory Functions (from operator modules):                                  ║
║    sig_x(ns=N, type_act='local')  - Pauli X operator                        ║
║    sig_y(ns=N, type_act='local')  - Pauli Y operator                        ║
║    sig_z(ns=N, type_act='local')  - Pauli Z operator                        ║
║    cdag(ns=N, type_act='local')   - Fermion creation operator               ║
║    c(ns=N, type_act='local')      - Fermion annihilation operator           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "application": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Operator: Application                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Primary Methods:                                                            ║
║    .apply(states, *args)     - Apply operator to state(s)                    ║
║    op(states, *args)         - Same as apply (callable interface)            ║
║    op[states]                - Returns only modified states                  ║
║    op % states               - Returns only coefficients                     ║
║                                                                              ║
║  Return Format:                                                              ║
║    (new_states, coefficients) - Tuple of arrays                              ║
║                                                                              ║
║  State Types Supported:                                                      ║
║    - Integer      (basis state index)                                        ║
║    - NumPy array  (occupation numbers)                                       ║
║    - JAX array    (for GPU acceleration)                                     ║
║                                                                              ║
║  Examples:                                                                   ║
║    >>> op = sig_x(ns=4, type_act='local')                                    ║
║    >>> states, vals = op(5, 2) # Apply σ_x at site 2 to state |0101⟩         ║
║    >>> states, vals = op.apply([5, 6, 7], 2) # Batch application             ║
║                                                                              ║
║  Fourier Transform:                                                          ║
║    .apply_fourier(k, hilbert, vec)  - Apply momentum-space operator          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "matrix": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Operator: Matrix Representation                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Building Matrix:                                                            ║
║    .build(dim=N, hilbert=H)       - Build and store matrix representation    ║
║    .matrix(dim=N, hilbert=H)      - Generate matrix (returns without storing)║
║                                                                              ║
║  Matrix-Vector Product:                                                      ║
║    .matvec(v, hilbert=H)          - Compute matrix-vector product            ║
║    .matvec_fun                    - Get matvec function for scipy            ║
║                                                                              ║
║  Key Parameters for matrix():                                                ║
║    dim         : int              - Matrix dimension (Hilbert space size)    ║
║    hilbert     : HilbertSpace     - Hilbert space for symmetry handling      ║
║    matrix_type : str              - 'sparse' or 'dense'                      ║
║    verbose     : bool             - Print progress information               ║
║                                                                              ║
║  Properties (after build):                                                   ║
║    .matrix_data                   - Get the stored matrix                    ║
║    .shape                         - Matrix dimensions                        ║
║    .sparse                        - Whether using sparse format              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "inherited": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Operator: Inherited from GeneralMatrix                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Operator inherits all GeneralMatrix functionality:                          ║
║                                                                              ║
║  Diagonalization:                                                            ║
║    .diagonalize(method='auto', k=None, ...)                                  ║
║    .eigenvalues, .eigenvectors                                               ║
║    .ground_state, .ground_energy                                             ║
║                                                                              ║
║  Spectral Analysis:                                                          ║
║    .spectral_gap, .spectral_width                                            ║
║    .participation_ratio(n), .degeneracy(tol)                                 ║
║    .level_spacing(), .level_spacing_ratio()                                  ║
║                                                                              ║
║  Matrix Operations:                                                          ║
║    .expectation_value(ψ, φ), .overlap(v1, v2)                                ║
║    .trace_matrix(), .frobenius_norm(), .spectral_norm()                      ║
║    .commutator(O), .anticommutator(O)                                        ║
║                                                                              ║
║  Memory & Control:                                                           ║
║    .memory, .memory_mb, .memory_gb                                           ║
║    .to_sparse(), .to_dense(), .clear()                                       ║
║                                                                              ║
║  Use GeneralMatrix.help() for full details on inherited methods.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "kspace": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      Operator: K-Space Transformations                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Transform operators between real-space and momentum-space (k-space).        ║
║  Only valid for single-particle operators (dimension = ns or 2*ns).          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Methods:                                                                    ║
║    .to_kspace(return_transform=False)                                        ║
║        Transform operator matrix to k-space                                  ║
║        Returns: (O_k, kgrid, kgrid_frac) or with W if return_transform=True  ║
║                                                                              ║
║    .from_kspace(O_k, kgrid=None)                                             ║
║        Transform k-space operator back to real space                         ║
║        Returns: O_real (ns x ns matrix)                                      ║
║                                                                              ║
║    .transform_to_kspace(return_grid=True)                                    ║
║        Alias for to_kspace() for backward compatibility                      ║
║                                                                              ║
║  Dimension Requirements:                                                     ║
║    Matrix dimension must be:                                                 ║
║    • ns     : Standard single-particle operators                             ║
║    • 2*ns   : BdG (Bogoliubov-de Gennes) with particle-hole doubling         ║
║    Many-body matrices (dim = 2^ns) are NOT supported!                        ║
║                                                                              ║
║  Helper:                                                                     ║
║    ._is_single_particle_matrix()  - Check if matrix has valid dimensions     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Examples:                                                                   ║
║    >>> model.build()                                                         ║
║    >>> H_k, kgrid, kgrid_frac = model.to_kspace()                            ║
║    >>>                                                                       ║
║    >>> # Get Bloch unitary for consistent operator transforms                ║
║    >>> H_k, kgrid, kgrid_frac, W = model.to_kspace(return_transform=True)    ║
║    >>> O_k = W.conj().T @ O_real @ W  # Transform another operator           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
        }

        overview = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                 Operator                                     ║
║           Quantum operator class for acting on basis states.                 ║
║       Supports integer, NumPy, and JAX representations.                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Inheritance: Operator -> GeneralMatrix -> LinearOperator                      ║
║  Subclasses:  Hamiltonian                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quick Start:                                                                ║
║    1. Create operator: op = sig_x(ns=4, type_act='local')                    ║
║    2. Apply to state:  states, vals = op(state, site_index)                  ║
║    3. Build matrix:    op.build(dim=16)                                      ║
║    4. Diagonalize:     op.diagonalize()                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Topics (use .help('topic') for details):                                    ║
║    'types'       - Operator types (Global, Local, Correlation)               ║
║    'creation'    - How to create operators                                   ║
║    'application' - Applying operators to states                              ║
║    'matrix'      - Matrix representation and matvec                          ║
║    'kspace'      - K-space (momentum space) transformations                  ║
║    'inherited'   - Methods inherited from GeneralMatrix                      ║
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

####################################################################################################


def operator_identity(backend: str = "default") -> Operator:
    """
    Generate the identity operator.
    Parameters:
    - backend (str)     : The backend for the operator - for using linear algebra libraries, not integer representation.
    Returns:
    - Operator          : The identity operator.
    """

    def identity_fun(state):
        return state, 1.0

    return Operator(
        fun_int=identity_fun,
        fun_np=identity_fun,
        fun_jax=identity_fun,
        eigval=1.0,
        ns=1,
        backend=backend,
        name=SymmetryGenerators.E,
        modifies=False,
        quadratic=False,
    )


####################################################################################################


def _make_global_closure(op, ns_val, sites_val, args):
    # Unpack args in Python to avoid Numba unpacking/branching issues
    n = len(args)
    if n == 2:
        a0, a1 = args

        @numba.njit
        def impl(state):
            return op(state, ns_val, sites_val, a0, a1)

        return impl
    elif n == 3:
        a0, a1, a2 = args

        @numba.njit
        def impl(state):
            return op(state, ns_val, sites_val, a0, a1, a2)

        return impl
    elif n == 4:
        a0, a1, a2, a3 = args

        @numba.njit
        def impl(state):
            return op(state, ns_val, sites_val, a0, a1, a2, a3)

        return impl
    elif n == 1:
        a0 = args[0]

        @numba.njit
        def impl(state):
            return op(state, ns_val, sites_val, a0)

        return impl
    elif n == 0:

        @numba.njit
        def impl(state):
            return op(state, ns_val, sites_val)

        return impl
    else:
        # Fallback for unusual cases (might fail in njit if *args not supported)
        @numba.njit
        def impl(state):
            return op(state, ns_val, sites_val, *args)

        return impl


def _make_local_closure(op, ns_val, args):
    n = len(args)
    if n == 2:
        a0, a1 = args

        @numba.njit(inline="always")
        def impl(state, i):
            return op(state, ns_val, (i,), a0, a1)

        return impl
    elif n == 3:
        a0, a1, a2 = args

        @numba.njit(inline="always")
        def impl(state, i):
            return op(state, ns_val, (i,), a0, a1, a2)

        return impl
    elif n == 1:
        a0 = args[0]

        @numba.njit(inline="always")
        def impl(state, i):
            return op(state, ns_val, (i,), a0)

        return impl
    elif n == 0:

        @numba.njit(inline="always")
        def impl(state, i):
            return op(state, ns_val, (i,))

        return impl
    else:

        @numba.njit(inline="always")
        def impl(state, i):
            return op(state, ns_val, (i,), *args)

        return impl


def _make_corr_closure(op, ns_val, args):
    n = len(args)
    if n == 2:
        a0, a1 = args

        @numba.njit(inline="always")
        def impl(state, i, j):
            return op(state, ns_val, (i, j), a0, a1)

        return impl
    elif n == 3:
        a0, a1, a2 = args

        @numba.njit(inline="always")
        def impl(state, i, j):
            return op(state, ns_val, (i, j), a0, a1, a2)

        return impl
    elif n == 1:
        a0 = args[0]

        @numba.njit(inline="always")
        def impl(state, i, j):
            return op(state, ns_val, (i, j), a0)

        return impl
    elif n == 0:

        @numba.njit(inline="always")
        def impl(state, i, j):
            return op(state, ns_val, (i, j))

        return impl
    else:

        @numba.njit(inline="always")
        def impl(state, i, j):
            return op(state, ns_val, (i, j), *args)

        return impl


def create_operator(
    type_act: int | OperatorTypeActing,
    op_func_int: Callable,
    op_func_np: Callable,
    op_func_jnp: Callable,
    lattice: Optional[Any] = None,
    ns: Optional[int] = None,
    sites: Optional[List[int]] = None,
    extra_args: Tuple[Any, ...] = (),
    name: Optional[str] = None,
    modifies: bool = True,
    code: Optional[int] = None,
) -> Operator:
    """
    Create a general operator that distinguishes the type of operator action (global, local, correlation)
    and wraps the provided operator functions for int, NumPy, and JAX. The operator functions must have
    the signature:
        - (state, *args) -> (new_state, op_value) for global
        - or (state, site, *args) for local,
        - and (state, site1, site2, *args) for correlation.

    ---
    Note:
        - The operator functions should be defined to handle the specific state representation (int, NumPy, JAX).
        - The operator functions should return a tuple of the new state and the operator value.
        - The global operator should have either a list of sites or act on all sites.
        - The local operator should have a single site argument (additional to the state).
        - The correlation operator should have two site arguments (additional to the state).
        - The operator functions should be able to handle the extra arguments passed to them.

    The extra arguments (extra_args) are passed on to the operator functions (apart from the sites,
    which are provided dynamically). For global operators, if no sites are provided the operator is applied
    to all sites (0, 1, ..., ns-1).

    ---
    Parameters:
        type_act (int):
            The type of operator acting. Use OperatorTypeActing.Global, .Local, or .Correlation.
        op_func_int (Callable):
            Operator function for integer-based states.
        op_func_np (Callable):
            Operator function for NumPy-based states.
        op_func_jnp (Callable):
            Operator function for JAX-based states.
        lattice (Optional[Any]):
            A lattice object; if provided, ns is set from lattice.ns.
        ns (Optional[int]):
            The number of sites. Required if lattice is None.
        sites (Optional[List[int]]):
            For global operators: a list of sites on which to act. If None, all sites are used.
            (For local or correlation operators, sites are provided when the operator is applied.)
        extra_args (Tuple[Any,...]):
            Extra parameters to be forwarded to the operator functions.
        name (Optional[str]):
            A base name for the operator. If not provided, op_func_int.__name__ is used.
        modifies (bool):
            Whether the operator modifies the state.

    Returns:
        Operator:
            An operator object with fun_int, fun_np, fun_jnp methods appropriately wrapped.
    """

    # Ensure we know ns - the number of sites or modes for the operator
    if lattice is not None:
        ns = lattice.ns
    else:
        assert ns is not None, "Either lattice or ns must be provided."

    if isinstance(type_act, str):
        type_act = OperatorTypeActing.from_string(type_act)

    #! Global operator: the operator acts on a specified set of sites (or all if sites is None)
    if OperatorTypeActing.is_type_global(type_act) or sites is not None:

        # If sites is None, we act on all sites.
        if isinstance(sites, int):
            sites = [sites]
        if sites is None or len(sites) == 0:
            sites = list(range(ns))

        sites = tuple(sites) if isinstance(sites, list) else sites
        sites_np = (
            np.array(sites, dtype=np.int32) if sites is not None else np.array([], dtype=np.int32)
        )

        if isinstance(op_func_int, CPUDispatcher):
            fun_int = (
                _make_global_closure(op_func_int, ns, sites, extra_args)
                if sites is not None
                else _make_global_closure(op_func_int, ns, extra_args)
            )
        else:

            def fun_int(state):
                return op_func_int(state, ns, sites, *extra_args)

        def fun_np(state):
            return op_func_np(state, sites_np, *extra_args)

        # ----------

        if JAX_AVAILABLE and op_func_jnp is not None:
            fun_jnp = make_jax_operator_closure(op_func_jnp, sites, *extra_args)
        else:

            def fun_jnp(state):
                return state, 0.0

        op_name = (name if name is not None else op_func_int.__name__) + "/"
        op_name += "-".join(str(site) for site in sites)
        return Operator(
            fun_int=fun_int,
            fun_np=fun_np,
            fun_jnp=fun_jnp,
            eigval=1.0,
            lattice=lattice,
            ns=ns,
            name=op_name,
            typek=SymmetryGenerators.Other,
            modifies=modifies,
            instr_code=code,
        )

    #! Local operator: the operator acts on one specific site. The returned functions expect an extra site argument.
    elif OperatorTypeActing.is_type_local(type_act):

        if isinstance(op_func_int, CPUDispatcher):
            fun_int = _make_local_closure(op_func_int, ns, extra_args)
        else:

            def fun_int(state, i):
                return op_func_int(state, ns, (i,), *extra_args)

        def fun_np(state, i):
            sites_1 = np.array([i], dtype=np.int32)
            return op_func_np(state, sites_1, *extra_args)

        if JAX_AVAILABLE and op_func_jnp is not None:

            @jax.jit
            def fun_jnp(state, i):
                # Pass tuple - operator will convert via jnp.asarray() internally
                return op_func_jnp(state, (i,), *extra_args)

        else:

            def fun_jnp(state, i):
                return state, 0.0

        op_name = (name if name is not None else op_func_int.__name__) + "/L"
        return Operator(
            fun_int=fun_int,
            fun_np=fun_np,
            fun_jnp=fun_jnp,
            eigval=1.0,
            lattice=lattice,
            ns=ns,
            name=op_name,
            typek=SymmetryGenerators.Other,
            modifies=modifies,
            instr_code=code,
        )

    #! Correlation operator: the operator acts on a pair of sites.
    elif OperatorTypeActing.is_type_correlation(type_act):

        if isinstance(op_func_int, CPUDispatcher):
            fun_int = _make_corr_closure(op_func_int, ns, extra_args)
        else:

            def fun_int(state, i, j):
                sites_2 = np.array([i, j], dtype=np.int32)
                return op_func_int(state, ns, sites_2, *extra_args)

        def fun_np(state, i, j):
            sites_2 = np.array([i, j], dtype=np.int32)
            return op_func_np(state, sites_2, *extra_args)

        if JAX_AVAILABLE and op_func_jnp is not None:

            @jax.jit
            def fun_jnp(state, i, j):
                # Pass tuple - operator will convert via jnp.asarray() internally
                return op_func_jnp(state, (i, j), *extra_args)

        else:

            def fun_jnp(state, i, j):
                return state, 0.0

        op_name = (name if name is not None else op_func_int.__name__) + "/C"
        return Operator(
            fun_int=fun_int,
            fun_np=fun_np,
            fun_jnp=fun_jnp,
            eigval=1.0,
            lattice=lattice,
            ns=ns,
            name=op_name,
            typek=SymmetryGenerators.Other,
            modifies=modifies,
            instr_code=code,
        )

    else:
        raise ValueError("Invalid OperatorTypeActing")


# Example usage:
# (Assume sigma_x_int_np, sigma_x_np, sigma_x_jnp are defined elsewhere and JAX_AVAILABLE is set.)

# For a global operator:
#   op = create_operator(OperatorTypeActing.Global, sigma_x_int_np, sigma_x_np, sigma_x_jnp,
#                        lattice=my_lattice, sites=[0, 2, 3], extra_args=(spin_value,))
#
# For a local operator:
#   op = create_operator(OperatorTypeActing.Local, sigma_x_int_np, sigma_x_np, sigma_x_jnp,
#                        ns=16, extra_args=(spin_value,))
#
# For a correlation operator:
#   op = create_operator(OperatorTypeActing.Correlation, sigma_x_int_np, sigma_x_np, sigma_x_jnp,
#                        ns=16, extra_args=(spin_value,))


def create_add_operator(operator: Operator, multiplier: Union[float, int, complex], sites=None):
    """
    Create a tuple representing an operator with its associated sites and multiplier.
    This function takes an operator instance along with a multiplier and an optional list of sites,
    and returns a tuple containing the operator, its sites, and the multiplier. If the operator is of
    Global type, the provided sites are ignored and replaced with an empty list.
    Parameters:
        operator (Operator):
            The operator instance to be added.
        multiplier (Union[float, int, complex]):
            The scalar multiplier associated with the operator.
        sites (Optional[List[Any]], optional):
            The list of sites where the operator acts. Defaults to None. If the operator's type is Global, sites will be overridden with an empty list.

    ---
    Returns:
        Tuple[Operator, List[Any], Union[float, int, complex]]:
            A tuple containing the operator, the adjusted list of sites, and the multiplier.
    """

    # if the operator is of Global type, we don't want to add the states to the argument, pass empty list

    if operator.type == OperatorTypeActing.Global:
        return (operator, [], multiplier)

    # if sites is None, we pass an empty list
    if sites is None:
        sites_arg = []
    else:
        sites_arg = sites
    # create the operator tuple
    return (operator, sites_arg, multiplier)


####################################################################################################
#! Operator shape
####################################################################################################


@numba.njit
def ensure_operator_output_shape_numba(state_out: np.ndarray, coeff_out: np.ndarray):
    """
    Ensure output is consistently 2D for states and 1D for coefficients.
    Works only when inputs already have correct shape or are known to be 1D scalars.
    """
    # if isinstance(coeff_out, (int, float, complex, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128)):
    #     coeff_out = np.array([coeff_out])

    if state_out.ndim == 1 and coeff_out.ndim == 1:
        out_state = state_out.reshape(1, state_out.shape[0])
        out_coeff = coeff_out
    elif state_out.ndim == 1 and coeff_out.ndim == 0:
        out_state = state_out.reshape(1, state_out.shape[0])
        out_coeff = np.empty(1, dtype=coeff_out.dtype)
        out_coeff[0] = coeff_out
    elif state_out.ndim == 2 and coeff_out.ndim == 1:
        out_state = state_out
        out_coeff = coeff_out
    else:
        raise ValueError("Unsupported shape combination in ensure_operator_output_shape_numba")

    return out_state, out_coeff


if JAX_AVAILABLE:

    @jax.jit
    def ensure_operator_output_shape_jax(state_out: jnp.ndarray, coeff_out: jnp.ndarray):
        r"""
        Ensure (state, coeff) output is 2D-batched: (N, L), (N,)
        where N is the number of states and L is the dimension of the state.
        Using JAX for JIT compilation.

        Parameters:
            state_out (jnp.ndarray):
            The output state array.
            coeff_out (jnp.ndarray):
            The output coefficient array.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
            A tuple containing the reshaped state and coefficient arrays.
            With the first dimension being the number of states that the operator returns
            and the second dimension being the dimension of the state.
            This corresponds to matrix elements of the operator:
            .. math::
                \{\, |s'\rangle,\ \langle s'|O|s\rangle\,\}
        """
        # Convert scalars to arrays
        state_out = jnp.atleast_1d(state_out)
        coeff_out = jnp.atleast_1d(coeff_out)

        # Handle scalar state (single integer)
        if state_out.ndim == 1 and coeff_out.shape[0] == 1:
            state_out = state_out.reshape(1, -1)  # (1, 1) or (1, L)
        elif state_out.ndim == 2:
            # already batched: do nothing
            pass
        else:
            raise ValueError("Unsupported state_out shape for JAX.")

        return state_out, coeff_out

else:

    def ensure_operator_output_shape_jax(state_out: np.ndarray, coeff_out: np.ndarray):
        r"""
        Ensure (state, coeff) output is 2D-batched: (N, L), (N,)
        where N is the number of states and L is the dimension of the state.
        Using JAX for JIT compilation.

        Parameters:
            state_out (jnp.ndarray):
            The output state array.
            coeff_out (jnp.ndarray):
            The output coefficient array.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
            A tuple containing the reshaped state and coefficient arrays.
            With the first dimension being the number of states that the operator returns
            and the second dimension being the dimension of the state.
            This corresponds to matrix elements of the operator:
            .. math::
                \{\, |s'\rangle,\ \langle s'|O|s\rangle\,\}
        """
        return ensure_operator_output_shape_numba(state_out, coeff_out)


# ##################################################################################################


def initial_states(
    ns: int,
    display: bool = False,
    int_state: Optional[int] = None,
    np_state: Optional[np.ndarray] = None,
) -> tuple:
    """
    Create initial states for testing the operator. It generates:
    - int_state:
        a random integer state (int) in the range [0, 2**ns)
    - np_state:
        a random NumPy state (np.ndarray) of size ns
    - jnp_state:
        a random JAX state (jnp.ndarray) of size ns

    The function also checks if the integer state is out of bounds and generates a random state if necessary.
    The function returns the generated states as a tuple.
    Parameters:
        ns (int):
            The number of sites in the system.
        display (bool):
            Whether to display the generated states.
        int_state (Optional[int]):
            The integer state to be used. If None, a random state is generated.
        np_state (Optional[np.ndarray]):
            The NumPy state to be used. If None, a random state is generated.
    Returns:
        tuple:
        A tuple containing the generated states:
        - int_state:
            a random integer state (int) in the range [0, 2**ns)
        - np_state:
            a random NumPy state (np.ndarray) of size ns
        - jnp_state:
            a random JAX state (jnp.ndarray) of size ns
    """
    import QES.general_python.common.binary as _bin_mod
    from QES.general_python.common.display import display_state

    #! Take the integer state as input
    int_state = (
        np.random.randint(0, 2 ** (ns % 64), dtype=np.int32) if int_state is None else int_state
    )

    #! Check the integer state whether it is out of bounds
    if np_state is None:
        if ns >= 64:
            np_state = np.random.randint(0, 2, size=ns).astype(np.float32)
        else:
            np_state = _bin_mod.int2base_np(int_state, size=ns, value_true=1, value_false=0).astype(
                np.float32
            )

    if JAX_AVAILABLE:
        jnp_state = jnp.array(np_state, dtype=jnp.float32) if np_state is not None else None
    else:
        jnp_state = None

    if display:
        display_state(int_state, ns, label="Integer state")
        display_state(np_state, ns, label="NumPy state")
        display_state(jnp_state, ns, label="JAX state")

    return int_state, np_state, jnp_state


####################################################################################################
#! Test the operator
####################################################################################################


def _dispatch(
    op: Operator,
    state: Union[int, np.ndarray],
    lat: "Lattice",
    is_int: bool,
    to_bin: Optional[Callable[[int, int], str]] = None,
    lab: Optional[str] = None,
    *,
    i=None,
    j=None,
    sites=None,
    just_time=False,
):
    """
    Call *op* with the right signature and send its first output to
    `display_operator_action`.

    Parameters
    ----------
    op : Operator
        The operator acting on the state.
    state : int | np.ndarray
        The state on which the operator acts.
    lat : Lattice
        The lattice object providing the number of sites.
    is_int : bool
        Whether the state is an integer or not.
    to_bin : Callable[[int, int], str], optional
        Function to convert integer to binary string.
    lab : str, optional
        The label for the operator.
    i : int, optional
        The first site index for the operator.
    j : int, optional
        The second site index for the operator.
    just_time : bool, default = False
        If True, only measure the time taken for the operation without
        displaying the results. This is useful for benchmarking.
    Returns
    -------
    Tuple[Union[int, np.ndarray], float]
        The new state and the coefficient after applying the operator.
    Notes
    -----
    * The function dispatches the operator call based on the type of operator
        (global, local, or correlation) and the type of state (integer or array).
    * The function also handles the display of the operator action using
        `display_operator_action` if `just_time` is False.
    * The function returns the new state and the coefficient after applying
        the operator.
    * The function uses `numba` for JIT compilation to speed up the operator
        application.
    * The function handles both NumPy and JAX arrays for the state.
    * The function uses `jnp` for JAX arrays if available.
    """
    if not just_time:
        from QES.general_python.common.display import display_operator_action

    state_act = state
    # call signature depends on OperatorTypeActing
    if i is None:  # Global operator
        st_out, coeff = op(state_act)
    elif j is None:  # Local operator
        if is_int:
            st_out, coeff = op(state_act, i)
        elif isinstance(state_act, (np.ndarray)):
            sites = np.array([i])
            st_out, coeff = op(state_act, *sites)
        elif isinstance(state_act, (jnp.ndarray)):
            sites = jnp.array([i])
            st_out, coeff = op(state_act, sites[0])
    else:  # Correlation operator
        if is_int:
            st_out, coeff = op(state_act, i, j)
        elif isinstance(state_act, (np.ndarray)):
            sites = np.array([i, j])
            st_out, coeff = op(state_act, *sites)
        elif isinstance(state_act, (jnp.ndarray)):
            sites = jnp.array([i, j])
            st_out, coeff = op(state_act, sites[0], sites[1])

    #! choose what to show depending on state representation
    new_state = st_out
    new_coeff = coeff
    if not just_time:
        site_lbl = ""
        if i is not None:
            site_lbl = str(i)
        if j is not None:
            site_lbl += f",{j}"
        if sites is not None:
            site_lbl = ",".join(map(str, sites))

        display_operator_action(
            f"\\quad \\quad {lab}", site_lbl, state, lat.ns, new_state, new_coeff, to_bin=to_bin
        )
    return new_state, new_coeff


def test_operator_on_state(
    op: Union[Operator, Sequence[Operator]],
    lat: "Lattice",
    state: Union[int, np.ndarray],
    *,
    ns: Optional[int] = None,
    op_acting: "OperatorTypeActing" = OperatorTypeActing.Local,
    op_label=None,
    to_bin=None,
    just_time=False,
    sites: Optional[List[int]] = None,
) -> None:
    r"""
    Pretty-print the action of *one or several* lattice operators
    on a basis state or wave-function.

    Parameters
    ----------
    op : Operator or sequence[Operator]
        The operator(s) Ô acting on 0, 1, or 2 sites.
    lat : Lattice
        Provides the number of sites ``lat.ns`` = :math:`N_s`.
    ns : int, optional
        Number of sites.  If *None*, uses ``lat.ns``.
    state : int | np.ndarray | jax.numpy.ndarray
        *Basis state* (integer encoding) or *wave-function* :math:`|\psi\rangle`.
    op_acting : OperatorTypeActing, default = ``Local``
        How Ô acts: Local (Ôᵢ), Correlation (Ôᵢⱼ), or Global (Ô).
    op_label : str | sequence[str], optional
        LaTeX label(s).  If *None*, uses ``op.name`` for every operator.
    to_bin : Callable[[int, int], str], optional
        Integer -> binary-string formatter.  Defaults to
        ``lambda k,L: format(k, f'0{L}b')``.
    just_time : bool, default = False
        If True, only measure the time taken for the operation without
        displaying the results.  This is useful for benchmarking.

    Notes
    -----
    * For **integer states** we reproduce the coefficient table you had before.
    * For **array states** (NumPy / JAX) we show only the *first* non-zero
        coefficient returned by the operator.  Adjust if you need more detail.

    Examples
    --------
    >>> test_operator_on_state(op, state, lat, ns=16, op_acting=OperatorTypeActing.Local)
    >>> test_operator_on_state(op, state, lat, just_time=True)
    """

    from QES.general_python.common.timer import Timer

    try:
        if not just_time:
            from IPython.display import Math, display

            from QES.general_python.common.display import display_state, prepare_labels
    except:
        raise ImportError("IPython is required for displaying operator actions.")

    # ------------------------------------------------------------------
    ops = (op,) if not isinstance(op, Sequence) else tuple(op)
    is_int = isinstance(state, (numbers.Integral, int, np.integer))
    ns = lat.ns if ns is None else ns
    labels = prepare_labels(ops, op_label) if not just_time else [""] * len(ops)

    # ------------------------------------------------------------------

    if not just_time:
        display_state(
            state,
            ns,
            label=f"Initial integer state (Ns={ns})",
            to_bin=to_bin,
            verbose=not just_time,
        )

    # ------------------------------------------------------------------
    with Timer(verbose=True, name="Operator action"):
        for cur_op, lab in zip(ops, labels):
            if not just_time:
                display(Math(rf"\text{{Operator: }} {lab}, \text{{typeacting}}: {op_acting}"))

            if op_acting == OperatorTypeActing.Local.value:
                for i in range(ns):
                    if not just_time:
                        display(Math(rf"\quad \text{{Site index: }} {i}"))
                    s, c = _dispatch(
                        cur_op, state, lat, is_int, to_bin, lab, i=i, just_time=just_time
                    )

            elif op_acting == OperatorTypeActing.Correlation.value:
                for i in range(ns):
                    for j in range(ns):
                        if not just_time:
                            display(Math(rf"\text{{Site indices: }} {i}, {j}"))
                        s, c = _dispatch(
                            cur_op, state, lat, is_int, to_bin, lab, i=i, j=j, just_time=just_time
                        )

            elif op_acting == OperatorTypeActing.Global.value:
                s, c = _dispatch(
                    cur_op, state, lat, is_int, to_bin, lab, just_time=just_time, sites=sites
                )
            else:
                raise ValueError(f"Operator acting type {op_acting!r} not supported.")


# --------------------------------------------------------------------------------------------------


def test_operators(
    op,
    state,
    ns: Optional[int] = None,
    output_format: str = "tabs",
    r=5,
    n=5,
    add_args: Optional[Tuple[Any, ...]] = None,
):
    """
    Test the operator using three syntaxes:
        - op(state),
        - op[state],
        - op % state
    and display the results either in separate tabs or as a combined Markdown output.

    This function:
        - Shows the initial state.
        - Describes the operator being applied.
        - Measures execution time using %timeit.
        - Displays results (if the operator returns a tuple, it prints both parts).

    It works for any state (e.g., NumPy, JAX, etc.) by displaying the raw state.

    Parameters:
        op:
            The operator to be applied.
        state:
            The initial state.
        ns:
            The number of sites in the system.
        r:
            The number of repetitions for %timeit.
        n:
            The number of loops for %timeit.
        output_format:
            'tabs' (default) to display results in ipywidgets tabs,
                    or 'markdown' to display a combined Markdown output.
    Returns:
        pd.DataFrame:
            A DataFrame containing the test results.
    """

    import io
    from contextlib import redirect_stdout

    import pandas as pd
    from IPython import get_ipython
    from IPython.display import Markdown, display

    import QES.general_python.common.binary as bin_mod

    if ns is None:
        ns = 32

    # Get the IPython shell
    ip = get_ipython()

    # Define tests for each operator syntax.
    if add_args is None:
        tests = {
            "op(state)": lambda: op(state),
            "op[state]": lambda: op[state],
            "op[op[state]]": lambda: (
                op[op[state][0]] if isinstance(state, (int, np.integer)) else op[op[state]]
            ),
            "op % state": lambda: op % state,
        }
    else:
        # If add_args is provided, we need to adjust the tests accordingly.
        # Note: We assume that add_args is a tuple of additional arguments.
        #       The first argument is the state, and the rest are additional arguments.
        #       The operator should be able to handle these additional arguments.
        params_str = ", ".join(map(str, add_args))
        tests = {
            f"op(state, {params_str})": lambda: op(state, *add_args),
            # f"op[state, {params_str}]"     : lambda: op[state, *add_args],
            f"op % state, {params_str}": lambda: op % (state, *add_args),
        }

    # List to collect the data for each test.
    results_data = []

    # Go through each test.
    for method_expr, func in tests.items():
        # Capture the timing output from %timeit.
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                ip.run_line_magic("timeit", f"-r {r} -n {n} {method_expr}")
            except Exception as e:
                print(f"Timeit error: {e}")

        # Get the captured output.
        timing_output = f.getvalue().strip()

        # Execute the operator call to get the result.
        try:
            result = func()
        except Exception as e:
            result = f"Error: {e}"

        # Check the state type
        if isinstance(state, (int, np.integer)):
            state_str = f"({state}), which is b{bin_mod.int2binstr(state, ns)}"
            if isinstance(result, tuple):
                state_str_r = f"{[r for r in result[0]]}, which is {['b' + bin_mod.int2binstr(r, ns) for r in result[0]]}"
                val_str = f"{[v for v in result[1]]}"
            else:
                state_str_r = ""
                val_str = f"{[v for v in result]}"
        else:
            state_str = str(state)
            state_str_r = str(result[0]) if isinstance(result, tuple) else str(result)
            val_str = str(result[1]) if isinstance(result, tuple) else ""

        if len(state_str_r) > 0 and len(val_str) > 0:
            result_str = f"{state_str_r} with {val_str}"
        elif len(state_str_r) > 0:
            result_str = f"{state_str_r}"
        elif len(val_str) > 0:
            result_str = f"{val_str}"
        else:
            result_str = f"{result}"

        # Add the test information to our list.
        results_data.append(
            {
                "Test Expression": method_expr,
                "Initial State": state_str,
                "Operator": op.name if hasattr(op, "name") else str(op),
                "Result": result_str,
                "Time Measurement": timing_output,
            }
        )

    # Create a DataFrame from the collected results.
    df = pd.DataFrame(results_data)

    if output_format == "markdown":
        # Convert the DataFrame to a Markdown table and display.
        from IPython.display import Markdown, display

        display(Markdown(df.to_markdown(index=False)))
    else:
        # Display the DataFrame directly.
        from IPython.display import display

        display(df)
    return df


####################################################################################################
#! End of file
####################################################################################################
