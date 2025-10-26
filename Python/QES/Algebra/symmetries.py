'''
file: Algebra/symmetries.py
Description: 
    This module contains functions to compute the symmetries of a given state at a given representation.
    This includes translation, reflection, and parity symmetries for spin-1/2 systems. Implementations
    are provided for both the state vector and the density matrix representations.
    
File        : Algebra/symmetries.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-05 
'''

# Import the necessary modules
import math
import cmath
import numpy as np
import time
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from QES.Algebra.hilbert import LocalSpace

#! operator module for operator overloading
try:
    from QES.Algebra.Operator.operator import Operator, SymmetryGenerators
except ImportError as e:
    raise ImportError("Failed to import Operator module. Ensure QES package is correctly installed.") from e

#! from general Python modules
try:
    from QES.general_python.lattices.lattice import Lattice, LatticeDirection
    from QES.general_python.common.binary import flip_all, rev, popcount, BACKEND_REPR, BACKEND_DEF_SPIN
except ImportError as e:
    raise ImportError("Failed to import general_python modules. Ensure QES package is correctly installed.") from e

####################################################################################################
#! Constants
####################################################################################################

_LATTICE_NONE_ERROR         = "Lattice object is None."
_LATTICE_DIM2_ERROR         = "Lattice dimension must be at least 2..."
_LATTICE_DIM3_ERROR         = "Lattice dimension must be at least 3..."
_TRANSLATION_CACHE_ATTR     = '_translation_cache'

####################################################################################################
#! Fermionic helpers
####################################################################################################

def _fermionic_boundary_sign_int_1d(state: int, ns: int, site_moving_mask: int) -> int:
    """
    Returns (-1)^{n_cross}, where n_cross is the number of particles that cross the boundary
    during a cyclic shift. For a left shift by 1 in 1D, particles at site 0 move to site Ns-1 (or vice versa depending on encoding).
    site_moving_mask marks those crossing sites (usually a single bit).
    """
    # Count how many occupied bits cross the boundary:
    return -1 if ((state & site_moving_mask) != 0) else +1

def _fermionic_boundary_sign_array_1d(state_vec: np.ndarray, crossing_indices: np.ndarray) -> int:
    # crossing_indices are the indices that wrap around; count occupations mod 2
    n_cross = int(state_vec[crossing_indices].sum()) & 1
    return -1 if n_cross == 1 else +1

####################################################################################################
#! Tanslational Symmetries - spin-1/2
####################################################################################################

def _axis_index(lat: Lattice, direction: LatticeDirection) -> int:
    axis_map = {
        LatticeDirection.X: 0,
        LatticeDirection.Y: 1,
        LatticeDirection.Z: 2,
    }
    axis = axis_map[direction]
    if axis >= lat.dim:
        raise ValueError(f"Direction {direction} is not supported for lattice dimension {lat.dim}.")
    return axis

def _compute_translation_data(lat: Lattice, direction: LatticeDirection):
    """
    Pre-compute permutation data for a unit translation.
    Returns ``(perm, crossing_mask)`` where ``perm`` maps source indices to
    destination indices and ``crossing_mask`` marks sites that cross the
    boundary when translated.
    """
    cache = getattr(lat, _TRANSLATION_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(lat, _TRANSLATION_CACHE_ATTR, cache)
    if direction in cache:
        return cache[direction]

    if not lat.is_periodic(direction):
        raise ValueError(f"Translation along {direction.name} requires periodic boundary conditions.")

    axis = _axis_index(lat, direction)
    dims = (lat.lx, lat.ly if lat.dim > 1 else 1, lat.lz if lat.dim > 2 else 1)
    size_axis = dims[axis]
    perm = np.empty(lat.ns, dtype=np.int64)
    crossing_mask = np.zeros(lat.ns, dtype=bool)

    for site in range(lat.ns):
        coord = list(lat.get_coordinates(site))
        while len(coord) < 3:
            coord.append(0)
        new_coord = coord.copy()
        new_coord[axis] += 1
        wrap = False
        if new_coord[axis] >= size_axis:
            new_coord[axis] -= size_axis
            wrap = True
        dest = lat.site_index(int(new_coord[0]), int(new_coord[1]), int(new_coord[2]))
        perm[site] = dest
        if wrap:
            crossing_mask[site] = True

    cache[direction] = (perm, crossing_mask)
    return perm, crossing_mask


def _apply_translation_int(state: int, perm: np.ndarray, crossing_mask: np.ndarray,
                           lat: Lattice, direction: LatticeDirection) -> Tuple[int, complex]:
    """Apply translation permutation to integer-encoded basis states."""
    ns = lat.ns
    new_state = 0
    for src in range(ns):
        if (state >> (ns - 1 - src)) & 1:
            dest = int(perm[src])
            new_state |= 1 << (ns - 1 - dest)

    if crossing_mask.any():
        crossing_sites = np.nonzero(crossing_mask)[0]
        occ_cross = sum(1 for src in crossing_sites if (state >> (ns - 1 - src)) & 1)
    else:
        occ_cross = 0
    phase = lat.boundary_phase(direction, occ_cross)
    return new_state, phase


def _apply_translation_array(state, perm: np.ndarray, crossing_mask: np.ndarray,
                             lat: Lattice, direction: LatticeDirection):
    """Apply translation permutation to array-encoded basis states."""
    state_arr = np.asarray(state)
    flat = state_arr.reshape(-1)
    if flat.size != lat.ns:
        raise ValueError(f"State size {flat.size} incompatible with lattice size {lat.ns}.")
    new_flat = np.empty_like(flat)
    new_flat[perm] = flat

    if crossing_mask.any():
        occ_cross = int(np.count_nonzero(flat[crossing_mask]))
    else:
        occ_cross = 0
    phase = lat.boundary_phase(direction, occ_cross)
    return new_flat.reshape(state_arr.shape), phase


def _translation_operator(lat: Lattice, direction: LatticeDirection, backend='default'):
    if lat is None:
        raise ValueError(_LATTICE_NONE_ERROR)
    perm, crossing_mask = _compute_translation_data(lat, direction)

    def op(state):
        if isinstance(state, (int, np.integer)):
            return _apply_translation_int(int(state), perm, crossing_mask, lat, direction)
        return _apply_translation_array(state, perm, crossing_mask, lat, direction)

    return op


def translation_x(lat: Lattice, backend='default', local_space: Optional['LocalSpace'] = None):
    """
    Translation in the X direction using permutation data derived from the lattice.
    """
    return _translation_operator(lat, LatticeDirection.X, backend=backend)

def translation_y(lat, backend='default'):
    """
    Translation in the Y direction using permutation data derived from the lattice.
    """
    if lat.dim == 1:
        raise ValueError(_LATTICE_DIM2_ERROR)
    return _translation_operator(lat, LatticeDirection.Y, backend=backend)

def translation_z(lat, backend='default'):
    """
    Translation in the Z direction using permutation data derived from the lattice.
    """
    if lat.dim != 3:
        raise ValueError(_LATTICE_DIM3_ERROR)
    return _translation_operator(lat, LatticeDirection.Z, backend=backend)

def translation(lat : Lattice,
                kx  : int,
                ky  : Optional[int] = 0,
                kz  : Optional[int] = 0,
                dim : Optional[int] = None,
                direction           = LatticeDirection.X,
                backend             = 'default'):
    """
    Generates a translation operator with a momentum phase factor.
    The phase is defined as exp(i * 2π * (k / L)) along the chosen direction.
    Boundary flux phases stored on the lattice are incorporated automatically.

    Parameters:
        - lat:
            lattice object.
        - kx, ky, kz: 
            momentum quantum numbers.
        - dim: 
            lattice dimension (if not provided, lat.get_Dim() is used).
        - direction: 
            one of 'x', 'y', or 'z' (default is 'x').
        - backend: 
            backend specifier for array operations.
    Returns:
        Operator: The translation operator with the defined momentum phase factor.    
    """
    if lat is None:
        raise ValueError(_LATTICE_NONE_ERROR)

    if dim is None:
        dim = lat.dim
    lx = lat.lx
    ly = lat.ly if dim > 1 and hasattr(lat, 'ly') else 1
    lz = lat.lz if dim > 2 and hasattr(lat, 'lz') else 1
    kx = 2 * math.pi * kx / lx
    ky = 2 * math.pi * ky / ly if dim > 1 and ky is not None else 0
    kz = 2 * math.pi * kz / lz if dim > 2 and kz is not None else 0
    k  = kx
    
    if direction == LatticeDirection.X:
        op_fun      = translation_x(lat, backend)
    elif direction == LatticeDirection.Y:
        op_fun      = translation_y(lat, backend)
        k           = ky
    elif direction == LatticeDirection.Z:
        op_fun      = translation_z(lat, backend)
        k           = kz
    else:
        op_fun      = translation_x(lat, backend)
    phase = cmath.exp(1j * k)  # Use cmath.exp for complex exponentials
    
    # get the symmetry generator type
    typek = SymmetryGenerators.Translation_x
    if direction == LatticeDirection.Y:
        typek = SymmetryGenerators.Translation_y
    elif direction == LatticeDirection.Z:
        typek = SymmetryGenerators.Translation_z
    
    name = f'T_{direction.name}'
    
    return Operator(lattice = lat, eigval = phase, fun_int = op_fun,
            typek = typek, backend = backend, name = name)

####################################################################################################
#! Reflection Symmetries - spin-1/2
####################################################################################################

def _reflection(ns : int, backend : str = 'default'):
    """ 
    Reflection operator for a given state.
    """
    def op(state):
        return rev(state, ns, backend)
    return op

def reflection(sec : int, lat : Optional[Lattice] = None, ns : Optional[int] = None, backend : str = 'default'):
    """
    Generates a reflection operator with eigenvalue 'sec'.
    Parameters:
        - lat: lattice object.
        - sec: eigenvalue of the reflection operator.
        - base: base of the integer representation (default is 2).
    Returns:
        Operator: The reflection operator defined by the parameters.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(_LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun_int=_reflection(ns, backend), typek=SymmetryGenerators.Reflection)

####################################################################################################
#! Parity Symmetries - spin-1/2
####################################################################################################

# --- Parity Z ---

def _flip_z(ns: int, backend: str = 'default', spin: bool = BACKEND_DEF_SPIN, spin_value: float = BACKEND_REPR):
    """
    Global spin-flip (parity Z) operator.

    For integer-encoded spin configurations, implement parity as flipping all spins and
    returning a phase (here set to +1 for simplicity). This matches the expected behavior
    in tests where parity reduces the Hilbert space by pairing bitstrings with their complements.
    """
    def op(state):
        # Flip all spins; treat arrays/ints via helper
        return (flip_all(state, ns, backend=backend, spin=spin, spin_value=spin_value), 1.0)
    return op

def parity_z(sec : int, ns : Optional[int] = None, lat : Optional[Lattice] = None,
            backend : str = 'default', spin: bool = BACKEND_DEF_SPIN, spin_value : float = BACKEND_REPR):
    """
    Generates a partity operator by applying the Pauli-Z flip operator to the state.
    The state is assumed to be in the Pauli-Z basis.
    Parameters:
        - lat: lattice object.
        - sec: eigenvalue of the parity operator.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(_LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun_int=_flip_z(ns, backend, spin, spin_value), typek=SymmetryGenerators.ParityZ)

# --- Parity Y ---

def _flip_y(ns: int, backend: str = 'default', 
        spin: bool = BACKEND_DEF_SPIN, spin_value: float = BACKEND_REPR):
    """
    Creates the behavior that checks the parity of the state by applying the Y-flip operator
    to the state and returning the phase factor. The state is assumed to be in the
    Pauli-Z basis.
    For integer states the binary popcount is used;
    for array-like states the helper popcount is applied.
    """
    phase_factor    = 1j if ns % 2 == 0 else -1j
    
    def op(state):
        spin_ups    = ns - popcount(state, spin=spin, backend=backend)
        phase       = (1 - 2 * (spin_ups & 1)) * phase_factor
        return (flip_all(state, ns, backend=backend, spin=spin, spin_value=spin_value), phase)
    return op

def parity_y(sec : int, ns : Optional[int] = None, lat : Optional[Lattice] = None,
            backend : str = 'default', spin: bool = BACKEND_DEF_SPIN, spin_value : float = BACKEND_REPR):
    """
    Generates a parity operator by applying the Y-flip operator to the state.
    The state is assumed to be in the Pauli-Z basis.
    Parameters:
        - lat: lattice object.
        - ns: number of sites in the lattice.
        - sec: eigenvalue of the parity operator.
        - backend: backend specifier for array operations.
        - spin: boolean flag to indicate whether the state is a spin state.
        - spin_value: value of the spin.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(_LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun_int=_flip_y(ns, backend, spin, spin_value), typek=SymmetryGenerators.ParityY)

# --- Parity X ---

def _flip_x(ns: int, backend: str = 'default',
            spin: bool = BACKEND_DEF_SPIN, spin_value : float = BACKEND_REPR):
    """
    Creates the behavior that checks the parity of the state by applying the X-flip operator
    to the state and returning the phase factor. The state is assumed to be in the
    Pauli-Z basis.
    For integer states the binary popcount is used;
    for array-like states the helper popcount is applied.
    """
        
    def op(state):
        return (flip_all(state, ns, backend=backend, spin=spin, spin_value=spin_value), 1)
    return op

def parity_x(sec : int, ns : Optional[int] = None, lat : Optional[Lattice] = None,
            backend : str = 'default', spin: bool = BACKEND_DEF_SPIN, spin_value : float = BACKEND_REPR):
    """
    Generates a parity operator by applying the X-flip operator to the state.
    The state is assumed to be in the Pauli-Z basis.
    Parameters:
        - lat: lattice object.
        - sec: eigenvalue of the parity operator.
    """
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(_LATTICE_NONE_ERROR)
    return Operator(lattice = lat, eigval = sec, fun_int=_flip_x(ns, backend, spin, spin_value), typek=SymmetryGenerators.ParityX)

####################################################################################################
# Choose Symmetry
####################################################################################################

def choose(sym_specifier : Tuple[SymmetryGenerators, int], 
        ns : Optional[int] = None, lat : Optional[Lattice] = None,
        backend : str = 'default', spin_value : Optional[float] = BACKEND_REPR, spin : Optional[bool] = BACKEND_DEF_SPIN):
    """
    Given a symmetry specification (a tuple of (SymmetryGenerators, eigenvalue))
    and a lattice, returns the corresponding symmetry operator.
    Parameters:
        - sym_specifier: 
            a tuple of (SymmetryGenerators, eigenvalue) specifying the symmetry.
        - ns: 
            number of sites in the lattice.
        - lat: 
            lattice object.
    Returns:
        Operator: The symmetry operator corresponding to the given specification.    
    
    Example sym_spec values:
        (SymmetryGenerators.T, k)           -> Translation with momentum sector k
        (SymmetryGenerators.R, sec)         -> Reflection with eigenvalue sec
        (SymmetryGenerators.PX, sec)        -> Parity (σ^x) with eigenvalue sec
        (SymmetryGenerators.PY, sec)        -> Parity (σ^y) with eigenvalue sec
        (SymmetryGenerators.PZ, sec)        -> Parity (σ^z) with eigenvalue sec
        (SymmetryGenerators.E, _)           -> Identity
    """
    gen, eig = sym_specifier
    
    if lat is not None:
        ns = lat.sites
    elif ns is None:
        raise ValueError(_LATTICE_NONE_ERROR)
    
    if gen == SymmetryGenerators.Translation_x:
        return translation(lat, kx=eig, dim = lat.dim, direction = LatticeDirection.X)
    elif gen == SymmetryGenerators.Translation_y:
        return translation(lat, kx=0, ky=eig, dim = lat.dim, direction = LatticeDirection.Y)
    elif gen == SymmetryGenerators.Translation_z:
        return translation(lat, kx=0, ky=0, kz=eig, dim = lat.dim, direction = LatticeDirection.Z)
    elif gen == SymmetryGenerators.Reflection:
        return reflection(sec=eig, lat=lat, ns=ns, backend=backend)
    elif gen == SymmetryGenerators.ParityX:
        return parity_x(sec=eig, lat=lat, ns=ns, backend=backend, spin=spin, spin_value=spin_value)
    elif gen == SymmetryGenerators.ParityY:
        return parity_y(sec=eig, lat=lat, ns=ns, backend=backend, spin=spin, spin_value=spin_value)
    elif gen == SymmetryGenerators.ParityZ:
        return parity_z(sec=eig, lat=lat, ns=ns, backend=backend, spin=spin, spin_value=spin_value)
    elif gen == SymmetryGenerators.E:
        return Operator(lat)
    # default return
    return Operator(lat)

####################################################################################################

from QES.general_python.common.tests import GeneralAlgebraicTest

class SymmetryTests(GeneralAlgebraicTest):
    """
    Class for testing the symmetry functions.
    """
    
    def add_tests(self):
        """
        Adds all symmetry function tests to the test list.
        """
        self.tests.append(self.test_translation_x_int)
        self.tests_dict[self.test_translation_x_int.__name__] = self.test_translation_x_int

        self.tests.append(self.test_translation_x_array)
        self.tests_dict[self.test_translation_x_array.__name__] = self.test_translation_x_array

        self.tests.append(self.test_translation_y_array)
        self.tests_dict[self.test_translation_y_array.__name__] = self.test_translation_y_array

        self.tests.append(self.test_translation_z_array)
        self.tests_dict[self.test_translation_z_array.__name__] = self.test_translation_z_array

        self.tests.append(self.test_translation_operator)
        self.tests_dict[self.test_translation_operator.__name__] = self.test_translation_operator

        self.tests.append(self.test_reflection)
        self.tests_dict[self.test_reflection.__name__] = self.test_reflection

        self.tests.append(self.test_parity_z)
        self.tests_dict[self.test_parity_z.__name__] = self.test_parity_z

        self.tests.append(self.test_parity_y)
        self.tests_dict[self.test_parity_y.__name__] = self.test_parity_y

        self.tests.append(self.test_parity_x)
        self.tests_dict[self.test_parity_x.__name__] = self.test_parity_x

        self.tests.append(self.test_choose)
        self.tests_dict[self.test_choose.__name__] = self.test_choose

    # --- Translation Tests ---

    def test_translation_x_int(self, lat, state_int):
        """
        Test the translation_x operator on an INTEGER state.
        """
        from QES.general_python.common.binary import int2binstr
        
        self._log("Testing translation_x on INTEGER state", self.test_count, "blue")
        op_func             = translation_x(lat, self.backend)
        new_state, phase    = op_func(state_int)
        self._log(f"Input (int): {int2binstr(state_int, lat.sites)}", self.test_count)
        self._log(f"Output (int): {int2binstr(new_state, lat.sites)}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_translation_x_array(self, lat, state_arr):
        """
        Test the translation_x operator on an ARRAY state.
        """
        self._log("Testing translation_x on ARRAY state", self.test_count, "blue")
        op_func             = translation_x(lat, self.backend)
        new_state, phase    = op_func(state_arr)
        self._log(f"Input (array): {state_arr}", self.test_count)
        self._log(f"Output (array): {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_translation_y_array(self, lat, state_arr):
        """
        Test the translation_y operator on an ARRAY state.
        (Integer version is not implemented.)
        """
        if lat.dim < 2:
            self._log("Skipping translation_y: lattice dim < 2", self.test_count, "yellow")
            self.test_count += 1
            return
        
        self._log("Testing translation_y on ARRAY state", self.test_count, "blue")
        op_func             = translation_y(lat, self.backend)
        new_state, phase    = op_func(state_arr)
        self._log(f"Input (array): {state_arr}", self.test_count)
        self._log(f"Output (array): {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count     += 1

    def test_translation_z_array(self, lat, state_arr):
        """
        Test the translation_z operator on an ARRAY state.
        """
        if lat.dim != 3:
            self._log("Skipping translation_z: lattice dim != 3", self.test_count, "yellow")
            self.test_count     += 1
            return
        self._log("Testing translation_z on ARRAY state", self.test_count, "blue")
        op_func             = translation_z(lat, self.backend)
        new_state, phase    = op_func(state_arr)
        self._log(f"Input (array): {state_arr}", self.test_count)
        self._log(f"Output (array): {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_translation_operator(self, lat, state, kx, ky, kz, direction):
        """
        Test the full translation operator (with momentum phase) using its apply() method.
        """
        self._log(f"Testing translation operator for direction {direction.name}", self.test_count, "blue")
        op                  = translation(lat, kx, ky, kz, dim=lat.dim, direction=direction, backend=self.backend)
        new_state, phase    = op.apply(state)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Output state: {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    # --- Reflection Test ---

    def test_reflection(self, lat, state):
        """
        Test the reflection operator.
        """
        self._log("Testing reflection operator", self.test_count, "blue")
        op = reflection(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Reflected state: {new_state}", self.test_count)
        self._log(f"Phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    # --- Parity Tests ---

    def test_parity_z(self, lat, state):
        """
        Test the parity_z operator.
        """
        self._log("Testing parity_z operator", self.test_count, "blue")
        op = parity_z(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        expected_phase = 1.0 - 2.0 * ((lat.sites - popcount(state, backend=self.backend)) & 1)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Parity_z output: {new_state}", self.test_count)
        self._log(f"Expected phase: {expected_phase}, Obtained phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_parity_y(self, lat, state):
        """
        Test the parity_y operator.
        """
        self._log("Testing parity_y operator", self.test_count, "blue")
        op = parity_y(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        expected_phase = (1 - 2 * ((lat.sites - popcount(state, backend=self.backend)) & 1)) * (1j ** lat.sites)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Parity_y output: {new_state}", self.test_count)
        self._log(f"Expected phase: {expected_phase}, Obtained phase: {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def test_parity_x(self, lat, state):
        """
        Test the parity_x operator.
        """
        self._log("Testing parity_x operator", self.test_count, "blue")
        op = parity_x(sec=1, lat=lat, backend=self.backend)
        new_state, phase = op.apply(state)
        self._log(f"Input state: {state}", self.test_count)
        self._log(f"Parity_x output: {new_state}", self.test_count)
        self._log(f"Phase (should be 1): {phase}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    # --- Choose() Test ---

    def test_choose(self, lat, state):
        """
        Test the choose() function to select a symmetry operator.
        """
        self._log("Testing choose() function for symmetry operators", self.test_count, "blue")
        # Test Translation_x
        op_tx = choose((SymmetryGenerators.Translation_x, 1), lat=lat, backend=self.backend)
        st_tx, ph_tx = op_tx.apply(state)
        # Test Reflection
        op_ref = choose((SymmetryGenerators.Reflection, 1), lat=lat, backend=self.backend)
        st_ref, ph_ref = op_ref.apply(state)
        # Test Parity_x
        op_px = choose((SymmetryGenerators.ParityX, 1), lat=lat, backend=self.backend)
        st_px, ph_px = op_px.apply(state)
        self._log(f"Translation_x: Output: {st_tx}, Phase: {ph_tx}", self.test_count)
        self._log(f"Reflection:     Output: {st_ref}, Phase: {ph_ref}", self.test_count)
        self._log(f"Parity_x:       Output: {st_px}, Phase: {ph_px}", self.test_count)
        self._log("-" * 50, self.test_count)
        self.test_count += 1

    def run_tests(self):
        """
        Runs all symmetry function tests.
        """
        from QES.general_python.common.binary import int2binstr
        self.test_count = 1
        separator       = "=" * 50
        self._log(separator, 0)
        self._log("TESTING SYMMETRY FUNCTIONS", 0, "green")
        self._log(separator, 0)

        # Create sample lattices.
        lat1d = Lattice(4)            # 1D lattice with 4 sites
        lat2d = Lattice(4, 4)           # 2D lattice: 4x4
        lat3d = Lattice(2, 2, 2)        # 3D lattice: 2x2x2

        # Create sample states.
        # For integer representations, generate random integers with appropriate bit-length.
        state_1d_int = np.random.randint(0, 2 ** lat1d.sites)
        state_2d_int = np.random.randint(0, 2 ** lat2d.sites)
        state_3d_int = np.random.randint(0, 2 ** lat3d.sites)
        # For array representations, convert the integer to a binary vector.
        state_1d_arr = np.array([int(b) for b in int2binstr(state_1d_int, lat1d.sites)])
        state_2d_arr = np.array([int(b) for b in int2binstr(state_2d_int, lat2d.sites)])
        state_3d_arr = np.array([int(b) for b in int2binstr(state_3d_int, lat3d.sites)])

        # Run translation_x tests.
        self.test_translation_x_int(lat1d, state_1d_int)
        self.test_translation_x_array(lat1d, state_1d_arr)
        self.test_translation_x_int(lat2d, state_2d_int)
        self.test_translation_x_array(lat2d, state_2d_arr)
        self.test_translation_x_int(lat3d, state_3d_int)
        self.test_translation_x_array(lat3d, state_3d_arr)

        # Run translation_y tests (only array versions).
        self.test_translation_y_array(lat2d, state_2d_arr)
        self.test_translation_y_array(lat3d, state_3d_arr)

        # Run translation_z test (3D only, array version).
        self.test_translation_z_array(lat3d, state_3d_arr)

        # Test translation operator with momentum.
        self.test_translation_operator(lat2d, state_2d_arr, 1, 0, 0, LatticeDirection.X)
        self.test_translation_operator(lat2d, state_2d_arr, 0, 1, 0, LatticeDirection.Y)
        self.test_translation_operator(lat3d, state_3d_arr, 0, 0, 1, LatticeDirection.Z)

        # Test reflection operator.
        self.test_reflection(lat2d, state_2d_arr)
        self.test_reflection(lat3d, state_3d_arr)

        # Test parity operators.
        self.test_parity_z(lat2d, state_2d_arr)
        self.test_parity_y(lat2d, state_2d_arr)
        self.test_parity_x(lat2d, state_2d_arr)

        # Test choose() function.
        self.test_choose(lat2d, state_2d_arr)

        self._log(separator, 0)
        total_time = time.time()  # For simplicity, you can record start time before tests.
        self._log(f"Total testing time: {total_time:.6f} sec", 0, "green")
        self._log(separator, 0)
        self._log("Testing completed.", 0, "green")

# -------------------------------------------------------------------------------------------------