"""
file: Algebra/global.py
Contains the GlobalSymmetry class for defining and checking global symmetries on states.
"""

# Import the necessary modules
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Callable, Optional, Union

import numba
import numpy as np

# operator module for operator overloading (absolute import for reliability)
try:
    if TYPE_CHECKING:
        from QES.Algebra.Operator.operator import GlobalSymmetries
    # from QES.Algebra.Symmetries.base            import _popcount64
except ImportError as e:
    raise ImportError(
        "Failed to import Operator module. Ensure QES package is correctly installed."
    ) from e

# from general Python modules
try:
    from QES.general_python.algebra.utils import get_backend
    from QES.general_python.lattices.lattice import Lattice
except ImportError as e:
    raise ImportError(
        "Failed to import general_python modules. Ensure QES package is correctly installed."
    ) from e


@numba.njit(cache=True, fastmath=True)
def _popcount64(x: np.int64) -> np.int64:
    # portable popcount (Numba supports bit ops; this is fine)
    c = np.int64(0)
    while x:
        x &= x - 1
        c += 1
    return c


# ---------------------------


class GlobalSymmetry(ABC):
    """
    GlobalSymmetry represents a global symmetry check on a state.
    It stores:
        - a lattice (lat)   (if needed),
        - a symmetry value  (val),
        - a symmetry name   (an element of GlobalSymmetries),
        - and a checking function (check) that takes (state, val) and returns a bool (True if the state satisfies the symmetry).

    The symmetry check supports integer states as well as NumPy or JAX arrays through the backend.
    """

    def __init__(
        self,
        lat: Optional[Lattice] = None,
        ns: Optional[int] = None,
        val: float = 0.0,
        name: "GlobalSymmetries" = "Other",
        backend: str = "default",
    ):
        """
        Initialize the GlobalSymmetry object.
        Parameters:
        - lat : Optional[Lattice]     : The lattice associated with the symmetry.
        - ns  : Optional[int]         : The number of states (if needed).
        - val : float                 : The symmetry value (default is 0.0).
        - name: GlobalSymmetries      : The name of the symmetry (default is GlobalSymmetries.Other).
        """
        if lat is not None:
            self._lat = lat
            ns = lat.ns
            self._ns = ns
        elif ns is None:
            raise ValueError("Either the lattice or the number of states must be provided!")

        if isinstance(backend, str):
            self._backend_str = backend
            self._backend = get_backend(backend)
        else:
            self._backend_str = "np" if backend == np else "jax"
            self._backend = backend

        self.ns = ns
        self.val = val
        self.name = name
        self.check: Optional[Callable[[Union[int, np.ndarray], float], bool]] = None

    # ---------- SETTERS -----------

    def set_fun(self, fun: Callable[[Union[int, np.ndarray], float], bool]) -> None:
        """Set the checking function."""
        self.check = fun

    def set_name(self, name: "GlobalSymmetries") -> None:
        """Set the name of the symmetry."""
        self.name = name

    # ---------- GETTERS -----------
    def get_name(self) -> "GlobalSymmetries":
        """Return the symmetry name (enum element)."""
        return self.name

    def get_name_str(self) -> str:
        """Return the string representation of the symmetry name."""
        return self.name.name

    def get_val(self) -> float:
        """Return the symmetry value."""
        return self.val

    @property
    def lat(self) -> Lattice:
        """Return the lattice associated with the symmetry."""
        return self._lat

    @property
    def ns(self) -> int:
        """Return the number of states."""
        return self._ns

    @ns.setter
    def ns(self, ns: int) -> None:
        """Set the number of states."""
        self._ns = ns

    @property
    def backend(self) -> str:
        """Return the backend used for the symmetry check."""
        return self._backend

    @backend.setter
    def backend(self, backend: str) -> None:
        """Set the backend used for the symmetry check."""
        self._backend = backend

    # ---------- CHECKER -----------
    def __call__(self, state: Union[int, np.ndarray]) -> bool:
        """
        When the object is called with a state, the checking function is applied.
        Raises a ValueError if no check function is set.
        """
        if self.check is None:
            raise ValueError("No symmetry check function has been set!")
        return self.check(state, self.val)

    def check_state(self, state: Union[int, np.ndarray], out_cond: bool) -> bool:
        """
        Returns True if the state satisfies the symmetry and the additional condition outCond.
        """
        return self(state) and out_cond


# ---------------------------
#! Global U(1) Symmetry
# ---------------------------


def u1_sym(state: Union[int, np.ndarray], val: float) -> bool:
    """
    Global U(1) symmetry check.

    For a given state, returns True if the popcount (number of 1-bits or up spins)
    equals the given value 'val'. This works for both integer states and array-like states.
    """
    return _popcount64(state) == val


def get_u1_sym(lat: Lattice, val: float) -> GlobalSymmetry:
    """
    Factory function that creates a U(1) global symmetry object.

    Parameters:
        lat: Lattice on which the symmetry is defined.
        val: The symmetry value (typically the required number of 1-bits).

    Returns:
        An instance of GlobalSymmetry with name U1, value val, and the checking function set to U1_sym.
    """
    from QES.Algebra.Operator.operator import GlobalSymmetries

    sym = GlobalSymmetry(lat=lat, val=val, name=GlobalSymmetries.U1)
    sym.set_fun(u1_sym)
    return sym


# --------------------------
#! Global Z2 Symmetry
# --------------------------


def z2_parity_sym(state: Union[int, np.ndarray], val: float) -> bool:
    """
    Global Z2 parity symmetry.

    Checks (-1)^N == val, where N = popcount(state).

    Works for:
    - spin-1/2 systems
    - spinless fermions

    Parameters
    ----------
    state : int or ndarray
        Integer-encoded many-body state
    val : float
        Parity sector: +1 (even), -1 (odd)

    Returns
    -------
    bool
    """
    # N = number of occupied sites / down spins
    parity = _popcount64(state) & 1

    # (-1)^N : even -> +1, odd -> -1
    return (1 if parity == 0 else -1) == int(val)


def get_z2_parity_sym(lat: Lattice, val: int) -> GlobalSymmetry:
    """
    Factory for global Z2 parity symmetry.

    Parameters
    ----------
    lat : Lattice
        Lattice (used only for ns)
    val : int
        Parity sector: +1 (even), -1 (odd)

    Returns
    -------
    GlobalSymmetry
    """
    from QES.Algebra.Operator.operator import GlobalSymmetries

    if val not in (+1, -1):
        raise ValueError("Z2 parity sector must be +1 (even) or -1 (odd).")

    sym = GlobalSymmetry(lat=lat, val=val, name=GlobalSymmetries.Z2_PARITY)
    sym.set_fun(z2_parity_sym)
    return sym


# --------------------------


def parse_global_syms(lat: Lattice, sym_dict: dict) -> list["GlobalSymmetry"]:
    """
    Parse a dictionary of global symmetries into a list of GlobalSymmetry objects.

    Parameters
    ----------
    lat : Lattice
        The lattice on which the symmetries are defined.
    sym_dict : dict
        Dictionary where keys are symmetry names (str) and values are symmetry values (float).

    Returns
    -------
    list[GlobalSymmetry]
        List of GlobalSymmetry objects.
    """
    syms = []
    for name_str, val in sym_dict.items():
        name_str = name_str.upper()

        # Create the appropriate GlobalSymmetry object based on the name
        if name_str == "U1":
            sym = get_u1_sym(lat, val)
        elif name_str == "Z2_PARITY":
            sym = get_z2_parity_sym(lat, int(val))
        else:
            raise ValueError(f"Unknown global symmetry name: {name_str}")
        syms.append(sym)
    return syms


# --------------------------


def to_codes(syms: list["GlobalSymmetry"]) -> np.ndarray:
    """
    Convert a list of GlobalSymmetry objects into a 2D numpy array of codes.

    Each row in the output array corresponds to a symmetry, with the first element
    being the symmetry type (as an integer) and the second element being the symmetry value.

    Parameters:
        syms:
            List of GlobalSymmetry objects.

    Returns:
        A tuple of two 1D numpy arrays:
        - codes: Array of symmetry types (as integers) of shape (n_syms,)
        - values: Array of symmetry values of shape (n_syms,)
    """
    n_syms = len(syms)
    codes = np.zeros((n_syms,), dtype=np.int8)
    values = np.zeros((n_syms,), dtype=np.float64)

    for i, sym in enumerate(syms):
        codes[i] = sym.get_name().value
        values[i] = sym.get_val()

    return codes, values


@numba.njit(cache=True, fastmath=True)
def violates_global_syms(state: int, codes: np.ndarray, values: np.ndarray, arg0: int) -> bool:
    """
    Check if the given integer state violates any of the global symmetries defined by the codes.

    Parameters:
        state:
            The integer representation of the state to check.
        codes:
            A 2D numpy array where each row represents a symmetry code.
            The first element of each row is the symmetry type (as an integer),
            and the second element is the symmetry value (Any type).
        arg0: An additional argument, if needed for future extensions.
    """

    n_syms = codes.shape[0]
    for i in range(n_syms):
        sym_type = codes[i]
        sym_val = values[i]

        if sym_type == 1:  # GlobalSymmetries.U1.value
            if _popcount64(state) != int(sym_val):
                return True  # Violates U(1) symmetry
        elif sym_type == 2:  # GlobalSymmetries.Z2_PARITY.value
            parity = _popcount64(state) & 1
            if (1 if parity == 0 else -1) != int(sym_val):
                return True
        # Additional symmetry types can be added here with elif blocks.

    return False  # Does not violate any symmetries


# ---------------------------
#! End of globals.py
# ---------------------------
