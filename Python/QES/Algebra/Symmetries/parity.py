"""
Parity (flip) symmetry operators and utilities for Hilbert space construction.

--------------------------------------------
File        : QES/Algebra/Symmetries/parity.py
Description : Parity (flip) symmetry operators and utilities for Hilbert space construction.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
--------------------------------------------
"""

from typing import TYPE_CHECKING, Tuple

import numba
import numpy as np

# ------------------------------------------

try:
    from QES.Algebra.Symmetries.base import (
        LocalSpaceTypes,
        MomentumSector,
        SymmetryApplicationCodes,
        SymmetryClass,
        SymmetryOperator,
        _popcount64,
    )
    from QES.general_python.common.binary import flip_all, popcount
except ImportError:
    raise ImportError("QES.general_python.common.binary module is required for parity operations.")

# ------------------------------------------

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice import Lattice

####################################################################################################
# Parity (flip) symmetry operator class
####################################################################################################


@numba.njit(cache=True, fastmath=True)
def _apply_parity_prim(
    state: np.int64, ns: np.int64, axis_code: np.uint8
) -> tuple[np.int64, np.complex128]:
    mask = (np.int64(1) << ns) - 1
    new_state = (~state) & mask

    # phases: keep exactly what you intend physically; below matches your current intent loosely.
    # axis_code: 0=x, 1=y, 2=z
    if axis_code == 2:  # z
        return new_state, np.complex128(1.0 + 0.0j)
    elif axis_code == 0:  # x
        return new_state, np.complex128(1.0 + 0.0j)
    else:  # y
        # your old code: spin_ups = ns - popcount(state)
        spin_ups = ns - _popcount64(state)
        # keep your convention; tweak if needed
        if (spin_ups & 1) == 0:
            ph = 1j
        else:
            ph = -1j
        return new_state, np.complex128(ph)


####################################################################################################


class ParitySymmetry(SymmetryOperator):
    """
    Global spin-flip (parity) symmetry operators.

    Physical Context
    ----------------
    Applies to discrete spin-flip symmetries in quantum spin systems:
    - Parity X (sigma^x):
        Flips all spins in x-basis -> (1)^n phase, n=spin-ups
    - Parity Y (sigma^y):
        Flips all spins in y-basis -> (+/- i)^n phase
    - Parity Z (sigma^z):
        Flips all spins in z-basis -> (-1)^n phase

    Examples:
    - XXZ model:
        Pz symmetry (conserves Sz but not Sx, Sy)
    - Transverse-field Ising:
        Px symmetry
    - XY model:
        May have Pz symmetry

    Commutation Rules
    -----------------
    Always commutes with:
    - TRANSLATION               : Abelian group [T, P] = 0 for discrete lattice translations
    - REFLECTION                : Spatial symmetry commutes with internal spin symmetry
    - PARITY                    : Different Pauli matrices commute
    - INVERSION                 : Spatial inversion independent of spin flip
    - U1_GLOBAL, U1_PARTICLE    : Backward compatibility alias

    Conditionally commutes with:
    - U1_SPIN                   : Only Pz commutes with Sz conservation (X, Y break it)
    - U1_PARTICLE               : Only if system is at half-filling N = Ns/2 (even Ns)
        -> Reason: Particle-hole transformation maps N <-> Ns - N
    """

    code = SymmetryApplicationCodes.OP_PARITY
    symmetry_class = SymmetryClass.PARITY
    compatible_with = {
        SymmetryClass.TRANSLATION,
        SymmetryClass.REFLECTION,
        SymmetryClass.U1_PARTICLE,
        SymmetryClass.U1_SPIN,
        SymmetryClass.PARITY,
        SymmetryClass.INVERSION,
        SymmetryClass.POINT_GROUP,
    }
    supported_local_spaces = {
        LocalSpaceTypes.SPIN_1_2,
        LocalSpaceTypes.SPIN_1,
        # NOT supported for fermions/bosons (different parity operators needed)
    }

    # -------------------------

    @staticmethod
    def get_sectors(axis: str = "z") -> list:
        """
        Return the valid parity sectors for the given axis.

        Parameters
        ----------
        axis : str, default='z'
            Parity axis ('x', 'y', or 'z').

        Returns
        -------
        list
            Valid parity sectors.
            - For axis='z' or 'x': [+1, -1]
            - For axis='y': [+1, -1] (with complex phases in application)

        Examples
        --------
        >>> ParitySymmetry.get_sectors('z')
        [1, -1]
        """
        return [1, -1]

    @staticmethod
    def get_axes() -> list:
        """
        Return the available parity axes.

        Returns
        -------
        list
            Available axes: ['x', 'y', 'z'].
        """
        return ["x", "y", "z"]

    # -------------------------

    def __init__(self, axis: str, sector: int, ns: int, nhl: int, lattice: "Lattice" = None):
        """
        Initialize parity symmetry operator.

        Parameters
        ----------
        axis : str
            Parity axis ('x', 'y', or 'z')
        sector : int
            Parity quantum number (+1 or -1)
        ns : int
            Number of sites in the system
        nhl : int
            Hilbert space dimension (local_hspace_size^ns)
        lattice : Lattice, optional
            Lattice instance for compatibility checking
        """
        self.axis = axis
        self.sector = sector
        self.ns = ns
        self.nhl = nhl
        self.lattice = lattice
        self.perm = None

    def __call__(self, state: int, ns: int = None, **kwargs) -> Tuple[int, complex]:
        return self.apply_int(state, ns or self.ns, **kwargs)

    # -------------------------

    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """Apply parity operator to integer state representation."""
        if flip_all is None or popcount is None:
            raise ImportError(
                "flip_all/popcount not available. Ensure QES.general_python.common.binary is installed."
            )
        if self.axis == "z":
            # Parity Z: flip all spins, phase +1
            new_state = int(flip_all(state, ns, backend="default")) & ((1 << ns) - 1)
            phase = 1.0
        elif self.axis == "y":
            # Parity Y: flip all spins, phase depends on spin-ups
            spin_ups = ns - popcount(state, backend="default")
            phase = (1 - 2 * (spin_ups & 1)) * (1j if ns % 2 == 0 else -1j)
            new_state = int(flip_all(state, ns, backend="default")) & ((1 << ns) - 1)
        elif self.axis == "x":
            # Parity X: flip all spins, phase +1
            new_state = int(flip_all(state, ns, backend="default")) & ((1 << ns) - 1)
            phase = 1.0
        else:
            raise ValueError(f"Unknown parity axis: {self.axis}")
        return new_state, phase

    def apply_numpy(self, state, **kwargs):
        """Apply parity operator to numpy array state representation."""
        import numpy as np

        if not isinstance(state, np.ndarray):
            state = np.array(state)
        # For numpy arrays, delegate to integer operations
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return np.array(new_state), phase

    def apply_jax(self, state, **kwargs):
        """
        Apply parity operator to JAX array state representation.

        Parameters
        ----------
        state : jnp.ndarray
            State vector (batch, ns) or (ns,)

        Returns
        -------
        new_state : jnp.ndarray
            Transformed state
        phase : jnp.ndarray
            Phase factor
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX is required for apply_jax. Install with: pip install jax")

        # Vector encoding handling
        if state.ndim >= 1 and state.shape[-1] == self.ns:
            if self.axis == "z":
                # Flip spins in Z basis
                new_state = state
                phase = jnp.ones(state.shape[:-1], dtype=complex)
                phase *= jnp.prod(state, axis=-1)
                return new_state, phase

            elif self.axis == "x":
                # Flip spins in X basis
                new_state = -state
                phase = jnp.ones(state.shape[:-1], dtype=complex)
                return new_state, phase

            elif self.axis == "y":
                # Py = prod sigma_y
                # new_state is flipped
                new_state = -state

                # Phase calculation per site
                # phase_i = i if s_i=+1 else -i
                # phase_i = i * s_i
                # Total phase = prod (i * s_i) = i^N * prod s_i

                ns = self.ns
                i_factor = 1j**ns
                prod_s = jnp.prod(state, axis=-1)
                phase = i_factor * prod_s
                return new_state, phase

        # Fallback to integer
        new_state, phase = self.apply_int(int(state), self.ns, **kwargs)
        return jnp.array(new_state), phase

    def is_compatible_with_global_symmetry(self, global_symmetry, **kwargs):
        """
        Check compatibility with U(1) particle number conservation.

        Particle-hole symmetry maps N -> Ns - N, so parity X/Y are only
        compatible at half-filling (N = Ns/2, even Ns).
        Parity Z acts trivially with U(1) so it's removed.
        """
        # Check if this is U(1) symmetry
        has_u1 = hasattr(global_symmetry, "name") and "U1" in str(global_symmetry.name)
        if not has_u1:
            return True, "Compatible"  # Not U(1, no restriction

        # Get U(1) sector (particle number) and ns from kwargs or instance
        u1_sector = getattr(global_symmetry, "sector", None) or getattr(
            global_symmetry, "val", None
        )
        ns = kwargs.get("ns", self.ns)
        if self.lattice:
            ns = ns or getattr(self.lattice, "Ns", None) or getattr(self.lattice, "ns", None)

        if self.axis in ["x", "y"]:
            # Parity X/Y require half-filling with U(1)
            if u1_sector is None or ns is None:
                return True, "Compatible (cannot determine filling)"
            if int(u1_sector) != ns // 2 or ns % 2 != 0:
                return (
                    False,
                    f"Parity {self.axis.upper()} incompatible with U(1): requires half-filling N=Ns/2={ns//2} (even Ns), but N={int(u1_sector)}",
                )
            return True, "Compatible at half-filling"
        elif self.axis == "z":
            # Parity Z acts trivially with U(1) (preserves particle number)
            return False, "Parity Z acts trivially with U(1) (redundant)"

        return True, "Compatible"

    # -------------------------

    @property
    def directory_name(self) -> str:
        """
        Return a clean string suitable for directory names.

        Format: 'p{axis}_{p|m}' e.g., 'pz_p', 'px_m', 'py_p'

        Returns
        -------
        str
            Filesystem-safe string for this parity symmetry sector.
        """
        sector_str = self._sector_to_str(self.sector)
        return f"p{self.axis}_{sector_str}"

    def __repr__(self) -> str:
        return f"ParitySymmetry(axis={self.axis}, sector={self.sector:+d}, ns={self.ns})"

    def __str__(self) -> str:
        parity_str = "even" if self.sector == 1 else "odd"
        return f"Parity{self.axis.upper()}({parity_str})"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
