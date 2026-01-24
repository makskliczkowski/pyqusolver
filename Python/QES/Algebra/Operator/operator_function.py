"""
A module defining the OperatorFunction class for quantum operators.
This class allows the definition and manipulation of quantum operators
that can act on quantum states represented as integers, NumPy arrays, or JAX arrays.
It supports various operations such as addition, subtraction, multiplication,
and composition of operators, enabling flexible quantum state transformations.

--------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-01
Description     : OperatorFunction class for quantum operators.
--------------------------------
"""

from __future__ import annotations

#####################################################################################################
from enum import Enum, auto, unique
from functools import partial  # partial function application for operator composition
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import numba
import numpy as np

####################################################################################################

try:
    if TYPE_CHECKING:
        from QES.Algebra.hilbert import HilbertSpace
        from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
        from QES.general_python.algebra.utils import Array
        from QES.general_python.lattices import Lattice
except ImportError as e:
    raise ImportError(
        "QES modules are required for this module to function properly. Please ensure QES is installed."
    ) from e

####################################################################################################

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True

    def make_jax_operator_closure(
        op_func: Callable, sites: Sequence[int], *static_args: Any, **static_kwargs: Any
    ) -> Callable:
        """
        Create a JAX-compiled closure with fixed `sites`, args, and kwargs.
        All additional parameters are assumed to be static (compile-time constants).

        Parameters
        ----------
        op_func : Callable
            The operator function to compile. Must accept (state, sites, *args, **kwargs).
        sites : Sequence[int]
            List or tuple of site indices to be treated as static.
        *static_args : Any
            Additional positional arguments to fix.
        **static_kwargs : Any
            Additional keyword arguments to fix.

        Returns
        -------
        Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
            A JIT-compiled function accepting only `state`.
        """
        static_sites = tuple(int(i) for i in sites)
        kwarg_keys = tuple(sorted(static_kwargs))
        kwarg_vals = tuple(static_kwargs[k] for k in kwarg_keys)

        @partial(jax.jit, static_argnums=range(1, 1 + 1 + len(static_args) + len(kwarg_vals)))
        def op_func_jax(state, static_sites_, *args_and_kwargs):
            args = args_and_kwargs[: len(static_args)]
            kwargs_vals = args_and_kwargs[len(static_args) :]
            kwargs = dict(zip(kwarg_keys, kwargs_vals))
            return op_func(state, static_sites_, *args, **kwargs)

        #! Precompile the full closure with static args fixed
        def compiled_op(state):
            return op_func_jax(state, static_sites, *static_args, *kwarg_vals)

        return jax.jit(compiled_op)

except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = np
    make_jax_operator_closure = lambda op_func, sites, *args, **kwargs: op_func

####################################################################################################
#! Distinguish type of function
####################################################################################################


def op_func_wrapper(op_func: Callable, *args: Any) -> Callable:
    """
    Wraps the operator function to handle different argument counts.
    Parameters:
        op_func (Callable):
            The operator function to be wrapped.
        *args (Any):
            Additional arguments to be passed to the operator function.
    """
    if op_func.__code__.co_argcount == 1:
        return op_func
    else:
        return lambda k: op_func(k, *args)


####################################################################################################
#! Constants
####################################################################################################

_PYTHON_SCALARS = (int, float, complex, np.number)
_FUN_NUMPY: TypeAlias = Optional[
    Callable[[np.ndarray, Any], List[Tuple[Optional[np.ndarray], Union[float, complex]]]]
]
_FUN_INT: TypeAlias = Callable[[int, Any], List[Tuple[Optional[int], Union[float, complex]]]]
if JAX_AVAILABLE:
    _FUN_JAX: TypeAlias = Optional[
        Callable[[jnp.ndarray, Any], List[Tuple[Optional[jnp.ndarray], Union[float, complex]]]]
    ]
else:
    _FUN_JAX: TypeAlias = None

####################################################################################################


@unique
class OperatorTypeActing(Enum):
    """
    Enumerates the types of operators acting on the system.
    """

    Global = (
        auto()
    )  # Global operator - acts on the whole system (does not need additional arguments).
    Local = (
        auto()
    )  # Local operator - acts on the local physical space (needs additional argument - 1).
    Correlation = (
        auto()
    )  # Correlation operator - acts on the correlation space (needs additional argument - 2).
    # -----------

    @staticmethod
    def from_string(name: str) -> "OperatorTypeActing":
        """
        Create an OperatorTypeActing from a string representation.

        Parameters
        ----------
        name : str
            The string representation of the operator type.

        Returns
        -------
        OperatorTypeActing
            The corresponding OperatorTypeActing enum member.

        Raises
        ------
        ValueError
            If the provided name does not correspond to any OperatorTypeActing member.
        """
        name = name.lower()
        if name.startswith("g"):
            return OperatorTypeActing.Global
        elif name.startswith("l"):
            return OperatorTypeActing.Local
        elif name.startswith("c"):
            return OperatorTypeActing.Correlation
        else:
            raise ValueError(f"Unknown OperatorTypeActing name: {name}")

    @staticmethod
    def is_type_global(val_str: Union[str, "OperatorTypeActing"]):
        return (isinstance(val_str, str) and val_str.lower().startswith("g")) or (
            val_str == OperatorTypeActing.Global
        )

    def is_global(self):
        """
        Check if the operator is a global operator.
        """
        return self == OperatorTypeActing.Global

    @staticmethod
    def is_type_local(val_str: Union[str, "OperatorTypeActing"]):
        return (isinstance(val_str, str) and val_str.lower().startswith("l")) or (
            val_str == OperatorTypeActing.Local
        )

    def is_local(self):
        """
        Check if the operator is a local operator.
        """
        return self == OperatorTypeActing.Local

    @staticmethod
    def is_type_correlation(val_str: Union[str, "OperatorTypeActing"]):
        return (isinstance(val_str, str) and val_str.lower().startswith("c")) or (
            val_str == OperatorTypeActing.Correlation
        )

    def is_correlation(self):
        """
        Check if the operator is a correlation operator.
        """
        return self == OperatorTypeActing.Correlation

    def __eq__(self, other):
        """
        Compare OperatorTypeActing with another OperatorTypeActing or with an integer.
        """
        if isinstance(other, OperatorTypeActing):
            return super().__eq__(other)
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"


####################################################################################################
#! OperatorFunction
####################################################################################################


class OperatorFunction:
    """
    OperatorFunction is a class that represents a mathematical operator that can be applied to a state.
    The operator can be defined for different backends (integer, NumPy, JAX) and supports various operations
    such as addition, subtraction, multiplication, and composition.
    The class provides a flexible way to define and apply operators to quantum states, allowing for
    different manipulations and analyses of quantum systems.

    Attr:
    ---
    - fun_int (Callable):
        The function defining the operator for integer states.
    - fun_np (Optional[Callable]):
        The function defining the operator for NumPy array states.
    - fun_jax (Optional[Callable]):
        The function defining the operator for JAX array states.
    - modifies_state (bool):
        A flag indicating whether the operator modifies the state.
    - necessary_args (int):
        The number of necessary arguments for the operator function.

    Methods:
    ---
    - __init__(fun_int, fun_np=None, fun_jax=None, modifies_state=True, necessary_args=0):
        Initializes the OperatorFunction object with the provided functions and attributes.
    - apply(s, *args):
        Applies the operator function to a given state with the specified arguments.
    - __call__(s, *args):
        Calls the operator function on a given state. Equivalent to `apply`.
    - __mul__(other):
        Composes the current operator with another operator or scales it by a scalar.
    - __rmul__(other):
        Reverse composition of the current operator with another operator or scales it by a scalar.
    - __add__(other):
        Adds two operator functions, combining their effects.
    - __sub__(other):
        Subtracts one operator function from another, combining their effects.
    - wrap(*args):
        Wraps the operator function with additional arguments.
    - fun (property):
        Gets or sets the function defining the operator for integer states.
    - np (property):
        Gets or sets the function defining the operator for NumPy array states.
    - jax (property):
        Gets or sets the function defining the operator for JAX array states.
    - modifies_state (property):
        Gets or sets the flag indicating whether the operator modifies the state.
    - necessary_args (property):
        Gets or sets the number of necessary arguments for the operator function.
    """

    _ERR_INVALID_ARG_NUMBER = (
        "Invalid number of arguments for the operator function. Expected {}, got {}."
    )
    _ERR_WRONG_MULTIPLICATION = "Invalid multiplication with type {}."

    # -----------

    def __init__(
        self,
        fun_int: _FUN_INT,
        fun_np: Optional[_FUN_NUMPY] = None,
        fun_jax: Optional[_FUN_JAX] = None,
        modifies_state: bool = True,
        necessary_args: int = 0,
    ):
        """
        Initialize the OperatorFunction object.

        Params:
        - fun (callable)    : The function that defines the operator - it shall take a state
            (or a list of states) and return the transformed state (or a list of states). States can be
            represented as integers or numpy arrays or JAX arrays. This enables the user to define
            any operator that can be applied to the state. The function shall return a list of pairs (state, value).
        """
        self._fun_int = fun_int
        self._fun_np = fun_np
        self._fun_jax = fun_jax  # JAX function for the operator
        self._modifies_state = modifies_state  # flag for the operator that modifies the state
        self._necessary_args = int(
            necessary_args
        )  # number of necessary arguments for the operator function
        self._acting_type = (
            OperatorTypeActing.Global
            if self._necessary_args == 0
            else (
                OperatorTypeActing.Local
                if self._necessary_args == 1
                else (
                    OperatorTypeActing.Correlation
                    if self._necessary_args == 2
                    else OperatorTypeActing.Global
                )
            )
        )
        self._dispatch = (
            self._apply_local
            if self._necessary_args == 1
            else (
                self._apply_correlation
                if self._necessary_args == 2
                else self._apply_global if self._necessary_args == 0 else self._apply_any
            )
        )

    # -----------

    def _choose_apply(self, s: Union[int, np.ndarray]) -> Callable:
        """
        Choose the appropriate function to apply based on the type of state and arguments.

        Parameters:
            s (int or np.ndarray):
                The state to which the operator is applied.
        Returns:
            A callable function that applies the operator to the state.
        """
        if isinstance(s, jnp.ndarray):
            return self._fun_jax
        elif isinstance(s, np.ndarray):
            return self._fun_np
        else:
            return self._fun_int

    # -----------

    def _apply_global(
        self, s: Union[int, np.ndarray]
    ) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or np.ndarray or jnp.ndarray):
                The state to which the operator is applied.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state
            (or None if not applicable) and its corresponding value.
        """

        # If the state is a JAX array, use the JAX function
        if JAX_AVAILABLE and isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            return self._fun_jax(s)
        # If the state is a NumPy array, use the NumPy function
        elif isinstance(s, (np.ndarray, List)) and self._fun_np is not None:
            return self._fun_np(s)
        # Fallback to the integer function
        return self._fun_int(s)

    # -----------

    def _apply_local(
        self, s: Union[int, np.ndarray], i
    ) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            i: Additional argument for the operator.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state
            (or None if not applicable) and its corresponding value.
        """

        # If the state is a JAX array, use the JAX function
        if JAX_AVAILABLE and isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            return self._fun_jax(s, i)
        # If the state is a NumPy array, use the NumPy function
        elif isinstance(s, np.ndarray) and self._fun_np is not None:
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s, i)
        # If the state is an integer, use the integer function
        # Note: The integer function should be able to handle the additional argument
        return self._fun_int(s, i)

    # -----------

    def _apply_correlation(
        self, s: Union[int], i, j
    ) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            i, j: Additional arguments for the operator.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state
            (or None if not applicable) and its corresponding value.
        """
        # If the state is a JAX array, use the JAX function
        if JAX_AVAILABLE and isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            # If the state is a JAX array, use the JAX function
            return self._fun_jax(s, i, j)
        elif isinstance(s, np.ndarray) and self._fun_np is not None:
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s, i, j)
        # If the state is an integer, use the integer function
        # Note: The integer function should be able to handle the additional arguments
        return self._fun_int(s, i, j)

    # -----------

    def _apply_any(
        self, s: Union[int, np.ndarray], *args
    ) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or np.ndarray or jnp.ndarray):
                The state to which the operator is applied.
            args:
                Additional arguments for the operator. The number of arguments must equal self._necessary_args.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state
            (or None if not applicable) and its corresponding value.
        """
        if isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            result = self._fun_jax(s, *args)
        elif isinstance(s, np.ndarray) and self._fun_np is not None:
            result = self._fun_np(s, *args)
        else:
            result = self._fun_int(s, *args)
        return result

    def apply(
        self, s: Union[int, np.ndarray], *args
    ) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or np.ndarray or jnp.ndarray):
                The state to which the operator is applied.
            args:
                Additional arguments for the operator. The number of arguments must equal self._necessary_args.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state
            (or None if not applicable) and its corresponding value.

        Raises:
            ValueError: If the number of provided arguments does not equal self._necessary_args,
                        or if the return type from the operator function is not recognized.
        """
        if len(args) != self._necessary_args:
            raise ValueError(self._ERR_INVALID_ARG_NUMBER.format(self._necessary_args, len(args)))

        # apply the operator function based on the number of necessary arguments
        result = self._dispatch(s, *args)

        # Check if result is a valid (state, value) tuple
        # Accept int, np.integer (for np.int64, etc.), np.ndarray, or jnp.ndarray for state
        if isinstance(result, tuple) and len(result) == 2:
            s_res, c_res = result

            # Convert tuples to numpy arrays for consistency
            if isinstance(s_res, tuple):
                s_res = np.array(s_res)
            if isinstance(c_res, tuple):
                c_res = np.array(c_res)

            state_valid = isinstance(s_res, (int, np.integer, np.ndarray, jnp.ndarray))

            if state_valid:
                return (s_res, c_res)

        elif isinstance(result, list) and all(isinstance(item, tuple) and len(item) == 2 for item in result):
            return result
        raise ValueError(
            "Operator function returned an invalid type. Expected a tuple or a list of (state, value) pairs."
        )

    # -----------

    def __call__(
        self, s: Union[int], *args
    ) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            args: Additional arguments for the operator. The number of arguments must equal self._necessary_args.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state
            (or None if not applicable) and its corresponding value.

        Raises:
            ValueError: If the number of provided arguments does not equal self._necessary_args,
                        or if the return type from the operator function is not recognized.
        """
        return self.apply(s, *args)

    # -----------
    #! Getters and Setters
    # -----------

    @property
    def fun(self):
        """Set the function that defines the operator"""
        return self._fun_int

    @fun.setter
    def fun(self, val):
        self._fun_int = val

    @property
    def npy(self):
        """Set the function that defines the operator"""
        return self._fun_np

    @npy.setter
    def npy(self, val):
        self._fun_np = val

    @property
    def jax(self):
        """Set the function that defines the operator"""
        return self._fun_jax

    @jax.setter
    def jax(self, val):
        self._fun_jax = val

    @property
    def modifies_state(self):
        """Set the flag for the operator that modifies the state"""
        return self._modifies_state

    @modifies_state.setter
    def modifies_state(self, val):
        self._modifies_state = val

    @property
    def necessary_args(self):
        """Set the number of necessary arguments for the operator function"""
        return self._necessary_args

    @necessary_args.setter
    def necessary_args(self, val):
        self._necessary_args = val

    # -----------
    #! Composition
    # -----------

    def _multiply_const(self, constant):
        """
        Multiply the operator by a constant value.
        This method constructs and returns a new OperatorFunction instance where the underlying
        functions for different processing backends (integer, NumPy, and JAX) are modified to multiply
        their computed values by the specified constant. Specifically, it defines three nested functions:

        - mul_int:
            Applies the constant multiplication to the result of self._fun_int.
        - mul_np:
            Applies the constant multiplication to the result of self._fun_np and is accelerated using Numba.
        - mul_jax:
            Applies the constant multiplication to the result of self._fun_jax and, if JAX is available,
            is compiled with jax.jit for performance.

        Parameters:
                constant (numeric): The constant value to multiply with the operator's result.
        Returns:
                OperatorFunction: A new instance of OperatorFunction encapsulating the modified functions,
                                                    along with metadata about state modifications and necessary arguments.
        """

        if JAX_AVAILABLE and isinstance(constant, jax.Array):
            pass
        elif isinstance(constant, np.ndarray):  # Numpy array scalar
            constant = constant.item()  # Convert to Python scalar

        def mul_int(s, *args):
            st, val = self._fun_int(s, *args)
            return st, np.array(val) * constant  # Ensure val is array for broadcasting

        @numba.njit(cache=True)
        def mul_np(s, *args):
            st, val = self._fun_np(s, *args)  # Assume fun_np returns (np.array, np.array)
            return st, val * constant

        mul_jax_defined = False
        if JAX_AVAILABLE and self._fun_jax is not None:

            @jax.jit
            def mul_jax_impl(s_jax, *args_jax):
                st_jax, val_jax = self._fun_jax(s_jax, *args_jax)
                return st_jax, val_jax * constant

            mul_jax = jax.jit(mul_jax_impl)
            mul_jax_defined = True
        else:
            mul_jax = None

        return OperatorFunction(
            mul_int,
            mul_np,
            mul_jax if mul_jax_defined else None,
            modifies_state=self._modifies_state,
            necessary_args=self._necessary_args,
        )

    # =========================================================================
    #! Composition: f * g  (i.e. (f * g)(s) = f(g(s)) )
    # =========================================================================

    def __mul__(self, other: Union[float, "OperatorFunction"]) -> "OperatorFunction":

        is_python_scalar = isinstance(other, _PYTHON_SCALARS)
        is_jax_scalar = False
        if JAX_AVAILABLE:
            is_jax_scalar = isinstance(other, jax.Array) and other.ndim == 0

        if is_python_scalar or is_jax_scalar:
            return self._multiply_const(other)

        if not isinstance(other, OperatorFunction):
            return NotImplementedError("Incompatible operator function")

        #! Type consistency check for operator composition f * g (self is f, other is g)
        if self._acting_type != other._acting_type:
            raise ValueError(
                f"Operator acting type mismatch for composition: {self._acting_type} (f) vs {other._acting_type} (g)"
            )

        try:
            from .operator_wrap import _make_mul_int_njit, _make_mul_jax_vmap, _make_mul_np_njit
        except ImportError as e:
            raise ImportError(
                "Failed to import multiplication helper functions for OperatorFunction. Ensure that operator_wrap module is available."
            ) from e

        # The composed operator (f*g) modifies state if f modifies g's output, or if g modifies initial state.
        # If f is diagonal (modifies_state=False), it doesn't change g's state structure.
        # If f can change state (modifies_state=True), it can change g's output state.
        composed_modifies_state = self._modifies_state or other._modifies_state

        # Number of arguments for the composed operator
        # This is correct due to the acting_type check ensuring they are compatible.
        composed_necessary_args = self._necessary_args
        # ? or max(self._necessary_args, other._necessary_args)

        # Create composed functions: (f * g)(s) = f(g(s))
        # self._fun_xxx is f, other._fun_xxx is g

        # Note on branching: Current _make_mul_int/np_njit take f_coeff[0], f_state[0].
        # This means if f (self) branches, only its first branch is propagated.
        # The new _make_mul_jax_vmap handles branching in f.

        composed_fun_int = _make_mul_int_njit(self._fun_int, other._fun_int)
        composed_fun_np = _make_mul_np_njit(self._fun_np, other._fun_np)
        composed_fun_jax = (
            _make_mul_jax_vmap(self._fun_jax, other._fun_jax) if JAX_AVAILABLE else None
        )

        return OperatorFunction(
            composed_fun_int,
            composed_fun_np,
            composed_fun_jax,
            modifies_state=composed_modifies_state,
            necessary_args=composed_necessary_args,
        )

    # -----------

    def __rmul__(
        self,
        other: Union[int, float, complex, np.int64, np.float64, np.complex128, "OperatorFunction"],
    ):
        """
        Reverse composition of two operator functions: g * f ≡ g(f(n,...), ...).
        That is, the current operator (self) is applied first, and then the left-hand operator (other)
        is applied to the resulting states.

        If other is a scalar, the operator's output values are simply scaled by that scalar.

        JIT compilation is applied for the JAX version when available.

        Parameters
        ----------
        other : int, float, complex, np.int64, np.float64, np.complex128, or OperatorFunction
            Either a scalar multiplier or an operator function to compose with self in reverse order.

        Returns
        -------
        OperatorFunction
            A new operator function representing the reverse composition.
        """
        is_python_scalar = isinstance(other, _PYTHON_SCALARS)
        is_jax_scalar = False
        if JAX_AVAILABLE:
            is_jax_scalar = isinstance(other, jax.Array) and other.ndim == 0

        if is_python_scalar or is_jax_scalar:
            return self._multiply_const(other)

        if not isinstance(other, OperatorFunction):
            return NotImplementedError("Incompatible operator function")

        # For other * self (g * f), where g is `other`, f is `self`
        if other._acting_type != self._acting_type:
            raise ValueError(
                f"Operator acting type mismatch for composition: {other._acting_type} (g) vs {self._acting_type} (f)"
            )

        try:
            from .operator_wrap import _make_mul_int_njit, _make_mul_jax_vmap, _make_mul_np_njit
        except ImportError as e:
            raise ImportError(
                "Failed to import multiplication helper functions for OperatorFunction. Ensure that operator_wrap module is available."
            ) from e

        composed_modifies_state = other._modifies_state or self._modifies_state
        composed_necessary_args = self._necessary_args  # From check, same as other._necessary_args

        # Create composed functions: (g * f)(s) = g(f(s))
        # other._fun_xxx is g, self._fun_xxx is f
        composed_fun_int = _make_mul_int_njit(other._fun_int, self._fun_int)
        composed_fun_np = _make_mul_np_njit(other._fun_np, self._fun_np)
        composed_fun_jax = (
            _make_mul_jax_vmap(other._fun_jax, self._fun_jax) if JAX_AVAILABLE else None
        )

        return OperatorFunction(
            composed_fun_int,
            composed_fun_np,
            composed_fun_jax,
            modifies_state=composed_modifies_state,
            necessary_args=composed_necessary_args,
        )

    # -----------

    def __getitem__(self, other):
        """
        Applies operator to a given - returns the first element of the result.
        """
        return self.apply(other)[0]

    # ----------

    def __mod__(self, other):
        """
        Applies operator to a given - returns the second element of the result.
        """
        return self.apply(other)[1]

    # -----------
    #! Addition
    # -----------

    def __add__(self, other: "OperatorFunction") -> "OperatorFunction":
        """
        Add two operator functions: (A + B).
        This results in a function that returns the concatenation of the results of A and B.
        """

        if not isinstance(other, OperatorFunction):
            return NotImplementedError(f"Cannot add OperatorFunction with {type(other)}")

        try:
            from .operator_wrap import _make_add_int_njit, _make_add_jax, _make_add_np_njit
        except ImportError as e:
            raise ImportError(
                "Failed to import addition helper functions for OperatorFunction. Ensure that operator_wrap module is available."
            ) from e

        # Compatibility Checks
        if self._necessary_args != other._necessary_args:
            raise ValueError(
                f"Argument number mismatch for addition: {self._necessary_args} vs {other._necessary_args}"
            )

        # Construct composed functions
        composed_fun_int = _make_add_int_njit(self._fun_int, other._fun_int)
        composed_fun_np = _make_add_np_njit(self._fun_np, other._fun_np)
        composed_fun_jax = _make_add_jax(self._fun_jax, other._fun_jax) if JAX_AVAILABLE else None

        # The combined operator modifies state if either modifies it
        new_modifies = self._modifies_state or other._modifies_state

        return OperatorFunction(
            composed_fun_int,
            composed_fun_np,
            composed_fun_jax,
            modifies_state=new_modifies,
            necessary_args=self._necessary_args,
        )

    def __sub__(self, other: "OperatorFunction") -> "OperatorFunction":
        """
        Subtract two operator functions: (A - B).
        This results in a function that returns the concatenation of A and B,
        with B's coefficients negated.
        """

        if not isinstance(other, OperatorFunction):
            return NotImplementedError(f"Cannot subtract OperatorFunction with {type(other)}")

        try:
            from .operator_wrap import _make_sub_int_njit, _make_sub_jax, _make_sub_np_njit
        except ImportError as e:
            raise ImportError(
                "Failed to import subtraction helper functions for OperatorFunction. Ensure that operator_wrap module is available."
            ) from e

        # Compatibility Checks
        if self._necessary_args != other._necessary_args:
            raise ValueError(
                f"Argument number mismatch for subtraction: {self._necessary_args} vs {other._necessary_args}"
            )

        # Construct composed functions
        composed_fun_int = _make_sub_int_njit(self._fun_int, other._fun_int)
        composed_fun_np = _make_sub_np_njit(self._fun_np, other._fun_np)
        composed_fun_jax = _make_sub_jax(self._fun_jax, other._fun_jax) if JAX_AVAILABLE else None

        new_modifies = self._modifies_state or other._modifies_state

        return OperatorFunction(
            composed_fun_int,
            composed_fun_np,
            composed_fun_jax,
            modifies_state=new_modifies,
            necessary_args=self._necessary_args,
        )

    # -----------
    #! Wrapping
    # -----------

    def wrap(self, *fixed_args):
        """
        Wraps the operator functions with additional fixed arguments, returning a new OperatorFunction
        that encapsulates integer, numpy, and (if available) JAX implementations.
        Parameters:
            *args: Additional arguments to be bound to the underlying operator functions.
        Returns:
            OperatorFunction: An object constructed with three wrapper functions:
                - wrap_int: Calls self._fun_int with the initial fixed arguments (from *args) followed
                    by any additional arguments provided during its invocation.
                - wrap_np: Calls self._fun_np in a similar manner for numpy-compatible operations.
                - wrap_jax: Calls self._fun_jax with the fixed and additional arguments, and, if JAX is available,
                    it is decorated with jax.jit for just-in-time compilation.
        Notes:
            - The inner wrapper functions merge the fixed arguments passed to wrap with any further arguments.
            - If JAX_AVAILABLE is True, the wrap_jax function is optimized using jax.jit.
            - The returned OperatorFunction also propagates metadata such as 'modifies_state' and 'necessary_args'
                from the current operator instance.
        """

        new_necessary_args = self._necessary_args - len(fixed_args)
        if new_necessary_args < 0:
            raise ValueError("Too many arguments provided for wrapping.")

        def wrap_int(s, *runtime_args):
            return self._fun_int(s, *(fixed_args + runtime_args))

        def wrap_np(s, *runtime_args):
            return self._fun_np(s, *(fixed_args + runtime_args))

        wrapped_fun_jax = None
        if JAX_AVAILABLE and self._fun_jax is not None:

            def wrap_jax_impl(s_jax, *runtime_args_jax):
                return self._fun_jax(s_jax, *(fixed_args + runtime_args_jax))

            wrapped_fun_jax = jax.jit(wrap_jax_impl)

        return OperatorFunction(
            wrap_int,
            wrap_np,
            wrapped_fun_jax,
            modifies_state=self._modifies_state,
            necessary_args=new_necessary_args,
        )


####################################################################################################
#! EOF
####################################################################################################
