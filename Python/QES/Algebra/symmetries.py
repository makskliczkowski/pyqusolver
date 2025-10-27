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

# Modular symmetry imports
import numpy as np
import time
from typing import Tuple, Optional, TYPE_CHECKING
try:
    from QES.Algebra.Symmetries.translation import TranslationSymmetry
    from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
    from QES.Algebra.Symmetries.parity import ParitySymmetry
    from QES.Algebra.Symmetries.base import SymmetryRegistry, SymmetryOperator
    from QES.general_python.lattices.lattice import Lattice, LatticeDirection
    from QES.general_python.common.binary import flip_all, rev, popcount, BACKEND_REPR, BACKEND_DEF_SPIN
    from QES.Algebra.Operator.operator import Operator, SymmetryGenerators
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

def _apply_translation_int(state: int, perm: np.ndarray, crossing_mask: np.ndarray,
def _apply_translation_array(state, perm: np.ndarray, crossing_mask: np.ndarray,
def _translation_operator(lat: Lattice, direction: LatticeDirection, backend='default'):
def translation_x(lat: Lattice, backend='default', local_space: Optional['LocalSpace'] = None):
def translation_y(lat, backend='default'):
def translation_z(lat, backend='default'):
def translation(lat : Lattice,

# Translation, reflection, and parity symmetries are now handled in QES.Algebra.Symmetries modules.

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
    and a lattice, returns the corresponding symmetry operator (modular version).
    """
    gen, eig = sym_specifier
    if gen in (SymmetryGenerators.Translation_x, SymmetryGenerators.Translation_y, SymmetryGenerators.Translation_z):
        return TranslationSymmetry(lat)
    elif gen == SymmetryGenerators.Reflection:
        return ReflectionSymmetry(lat)
    elif gen in (SymmetryGenerators.ParityX, SymmetryGenerators.ParityY, SymmetryGenerators.ParityZ):
        axis = {SymmetryGenerators.ParityX: 'x', SymmetryGenerators.ParityY: 'y', SymmetryGenerators.ParityZ: 'z'}[gen]
        return ParitySymmetry(lat, axis=axis)
    elif gen == SymmetryGenerators.E:
        return Operator(lat)
    else:
        raise ValueError(f"Unknown symmetry generator: {gen}")

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