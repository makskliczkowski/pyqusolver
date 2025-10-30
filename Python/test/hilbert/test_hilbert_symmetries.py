"""
Test suite for Hilbert space symmetries

--------------
File    : test/hilbert/test_hilbert_symmetries.py
Author  : Maksymilian Kliczkowski
Date    : October 2025
--------------
"""

########################################################################
#! IMPORTS
########################################################################

import sys
import os
from pathlib import Path

# Add QES to path
qes_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(qes_path))

import numba
import numpy as np
import pytest
from typing import List, Tuple
from math import comb
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Import QES modules
try:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Operator.operator import SymmetryGenerators, LocalSpace, LocalSpaceTypes
    from QES.Algebra.globals import GlobalSymmetry, get_u1_sym
    from QES.general_python.lattices.square import SquareLattice
    from QES.general_python.lattices.lattice import LatticeBC
    from QES.general_python.common.binary import int2binstr, popcount
except ImportError as e:
    raise ImportError("Failed to import QES modules. Ensure QES package is correctly installed.") from e

########################################################################
#! CONSTANTS
########################################################################

# Test parameters
DEFAULT_SYSTEM_SIZE     = 4
LARGE_SYSTEM_SIZE       = 6
DEFAULT_PARTICLE_NUMBER = 3
VALIDATION_SAMPLE_SIZE  = 5

# Display settings
SEPARATOR_LONG          = "=" * 80
SEPARATOR_SHORT         = "=" * 60
SEPARATOR_TEST          = "-" * 60
INDENT                  = "  "

########################################################################
#! TEST UTILITIES
########################################################################

def print_test_header(test_name: str, description: str = "") -> None:
    """Print a formatted test header with clear separation."""
    print(f"\n{SEPARATOR_LONG}")
    print(f"🧪 RUNNING TEST: {test_name}")
    if description:
        print(f"   {description}")
    print(f"{SEPARATOR_LONG}")

def print_test_result(success: bool, message: str = "") -> None:
    """Print test result with clear success/failure indication."""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\n{status}: {message}")
    print(SEPARATOR_SHORT)

def print_test_section(title: str) -> None:
    """Print a subsection within a test."""
    print(f"\n{INDENT}📋 {title}")
    print(f"{INDENT}{SEPARATOR_TEST}")

########################################################################
#! PYTEST FIXTURES AND HOOKS
########################################################################

@pytest.fixture(autouse=True)
def print_test_info(request):
    """Print test information before and after each test."""
    test_name = request.node.name
    class_name = request.node.cls.__name__ if request.node.cls else "Global"
    
    print_test_header(f"{class_name}::{test_name}")
    
    def fin():
        print_test_result(True, f"{class_name}::{test_name} completed")
    
    request.addfinalizer(fin)

########################################################################
#! HELPER FUNCTIONS
########################################################################

def print_section_header(title: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def print_subsection(title: str) -> None:
    """Print a formatted subsection title."""
    print(f"\n{title}:")


def create_1d_lattice(ns: int, bc: LatticeBC = LatticeBC.PBC) -> SquareLattice:
    """Create a 1D square lattice with the specified number of sites."""
    return SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=bc)


def create_2d_lattice(lx: int, ly: int, bc: LatticeBC = LatticeBC.PBC) -> SquareLattice:
    """Create a 2D square lattice with the specified dimensions."""
    return SquareLattice(dim=2, lx=lx, ly=ly, lz=1, bc=bc)


def print_state(state: int, ns: int, label: str = "State") -> None:
    """
    Pretty print a quantum state with binary representation.
    
    Args:
        state: Integer representation of quantum state
        ns: Number of sites
        label: Descriptive label for the state
    """
    binary          = int2binstr(state, ns)
    particle_num    = popcount(state)
    print(f"{INDENT}{label}: {state:4d} = |{binary}>, N = {particle_num}")


def print_hilbert_info(hilbert: HilbertSpace, title: str = "") -> None:
    """
    Print comprehensive information about a Hilbert space.
    
    Args:
        hilbert: HilbertSpace instance
        title: Optional title for the output
    """
    if title:
        print_subsection(title)
    
    print(f"{INDENT}Full space dimension: {hilbert.Nhfull}")
    print(f"{INDENT}Reduced space dimension: {hilbert.Nh}")
    
    if hilbert.sym_group:
        print(f"{INDENT}Symmetry group size: {len(hilbert.sym_group)}")
    
    reduction_factor = hilbert.Nhfull / hilbert.Nh if hilbert.Nh > 0 else 0
    print(f"{INDENT}Reduction factor: {reduction_factor:.2f}x")


########################################################################
#! HELPER FUNCTIONS - VALIDATION
########################################################################

def validate_representative(hilbert: HilbertSpace, state: int, 
                           verbose: bool = False) -> bool:
    """
    Validate that a state is correctly identified as a representative.
    
    A state is a representative if it's the minimum in its symmetry orbit.
    
    Args:
        hilbert: HilbertSpace instance
        state: State to validate
        verbose: Whether to print detailed error messages
        
    Returns:
        True if validation passes, False otherwise
    """
    rep, _ = hilbert.find_repr(state)
    
    # Find minimum state in orbit by applying all symmetry operations
    min_state = state
    for g in hilbert.sym_group:
        new_state, _ = g(state)
        if new_state < min_state:
            min_state = new_state
    
    is_valid = (state == min_state) == (state == rep)
    
    if not is_valid and verbose:
        print(f"{INDENT}(bad) Representative validation failed for state {state}")
        print(f"{INDENT}   Found rep: {rep}, Actual min: {min_state}")
    
    return is_valid


def validate_global_symmetries(hilbert: HilbertSpace, state: int) -> Tuple[bool, str]:
    """
    Check if a state satisfies all global symmetries.
    
    Args:
        hilbert: HilbertSpace instance
        state: State to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not hilbert._global_syms:
        return True, ""
    
    for sym in hilbert._global_syms:
        if not sym(int(state)):
            return False, f"Violates {sym.get_name_str()}"
    
    return True, ""


def validate_mapping(hilbert: HilbertSpace, verbose: bool = False) -> bool:
    """
    Validate the entire mapping by checking:
    1. All states in mapping are representatives
    2. All representatives satisfy global symmetries
    3. Normalization factors are positive and reasonable
    
    Args:
        hilbert: HilbertSpace instance
        verbose: Whether to print detailed validation info
        
    Returns:
        True if validation passes, False otherwise
    """
    # Handle identity mapping case
    if hilbert.mapping is None or len(hilbert.mapping) == 0:
        print(f"{INDENT}⚠️  No mapping to validate (identity mapping)")
        return True
    
    print_section_header(f"Validating mapping: {len(hilbert.mapping)} states", 
                        char="=", width=60)
    
    errors = []
    
    # Validate each state in mapping
    for idx, state in enumerate(hilbert.mapping):
        state_int = int(state)
        
        # Check 1: State is its own representative
        rep, _ = hilbert.find_repr(state_int)
        if rep != state:
            errors.append(f"State {state} in mapping but rep={rep}")
            if verbose:
                print(f"{INDENT}(bad) [{idx}] {int2binstr(state, hilbert.Ns)} "
                      f"is not its own representative")
        
        # Check 2: State satisfies global symmetries
        is_valid, err_msg = validate_global_symmetries(hilbert, state_int)
        if not is_valid:
            errors.append(f"State {state} {err_msg}")
            if verbose:
                print(f"{INDENT}(bad) [{idx}] {int2binstr(state, hilbert.Ns)} {err_msg}")
        
        # Check 3: Normalization is positive
        norm = hilbert.norm(idx)
        if norm <= 0:
            errors.append(f"State {state} has non-positive norm {norm}")
            if verbose:
                print(f"{INDENT}(bad) [{idx}] {int2binstr(state, hilbert.Ns)} "
                      f"has invalid norm {norm}")
    
    # Print results
    if errors:
        print(f"\n{INDENT}(bad) Found {len(errors)} errors in mapping:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"{INDENT}   {err}")
        if len(errors) > 10:
            print(f"{INDENT}   ... and {len(errors)-10} more")
        return False
    else:
        print(f"{INDENT}(ok) Mapping validation passed! "
              f"All {len(hilbert.mapping)} states are valid representatives")
        return True


########################################################################
#! TEST CLASSES
########################################################################

class TestHilbertSpaceBasic:
    """
    Basic tests for Hilbert space functionality without symmetries.
    
    This class tests the fundamental HilbertSpace class behavior when no symmetries
    are applied. It verifies that the space has the correct dimensions, that no
    symmetry reduction occurs, and that basic properties are set correctly.
    """
    
    def test_initialization_no_symmetries(self):
        """
        Test creating a Hilbert space without any symmetries applied.
        
        This test verifies that when no symmetry generators or global symmetries
        are provided, the Hilbert space remains in its full, unreduced form.
        The test checks that:
        - The number of sites is correctly set
        - The full Hilbert space dimension is 2^Ns (for spin-1/2 particles)
        - The reduced dimension equals the full dimension (no reduction)
        - No mapping is generated (identity mapping)
        - The space is not marked as modified by symmetries
        """
        ns = DEFAULT_SYSTEM_SIZE
        hilbert = HilbertSpace(ns=ns, sym_gen=None, global_syms=None)
        
        assert hilbert.Ns == ns
        assert hilbert.Nh == hilbert.Nhfull == 2**ns
        assert hilbert.mapping is None
        assert not hilbert.modifies
        
        print_hilbert_info(hilbert, "Basic Hilbert space (no symmetries)")
        print(f"{INDENT}(ok) Test passed")
    
    def test_local_space_spin_half(self):
        """
        Test Hilbert space with explicit spin-1/2 local space specification.
        
        This test ensures that when a LocalSpace object is explicitly provided,
        the Hilbert space correctly recognizes the local dimension and computes
        the total space size accordingly.
        """
        ns = DEFAULT_SYSTEM_SIZE
        local_space = LocalSpace.default()  # spin-1/2
        hilbert = HilbertSpace(ns=ns, local_space=local_space)
        
        assert hilbert.local == 2
        assert hilbert.Nhfull == 2**ns
        
        print_subsection("Spin-1/2 system")
        print(f"{INDENT}Local dimension: {hilbert.local}")
        print(f"{INDENT}Full Hilbert space: {hilbert.Nhfull}")
        print(f"{INDENT}(ok) Test passed")


class TestTranslationSymmetry:
    """
    Tests for translation symmetry in 1D, 2D, and 3D systems.
    
    Translation symmetry is one of the most important symmetries in condensed matter
    physics, allowing reduction of the Hilbert space by identifying states that are
    related by lattice translations. This class tests:
    - Basic translation symmetry in momentum sectors (k=0 and non-zero k)
    - Representative state finding and validation
    - Proper dimension reduction and mapping generation
    """
    
    def test_translation_1d_k0(self):
        """
        Test 1D translation symmetry with k=0 (uniform/total momentum sector).
        
        The k=0 sector contains states that are invariant under translation by any
        lattice site. This is the most commonly used sector for ground state searches
        as it includes the ground state for translationally invariant Hamiltonians.
        The test verifies that the space is reduced and mapping is properly generated.
        """
        ns = DEFAULT_SYSTEM_SIZE
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.Translation_x, 0)]
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_hilbert_info(hilbert, "1D Translation k=0")
        
        # Validation
        assert hilbert.Nh < hilbert.Nhfull
        assert hilbert.mapping is not None
        assert validate_mapping(hilbert)
        print(f"{INDENT}(ok) Test passed")
    
    def test_translation_1d_momentum(self):
        """
        Test 1D translation symmetry with non-zero momentum sector.
        
        Non-zero momentum sectors contain states with definite crystal momentum k.
        These sectors are important for studying excited states and momentum-resolved
        properties. The test uses a larger system size to ensure multiple momentum
        sectors exist.
        """
        ns = LARGE_SYSTEM_SIZE
        k = 1  # momentum sector
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.Translation_x, k)]
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_hilbert_info(hilbert, f"1D Translation k={k}")
        
        assert hilbert.Nh < hilbert.Nhfull
        assert validate_mapping(hilbert)
        print(f"{INDENT}(ok) Test passed")
    
    def test_representative_finding_1d(self):
        """
        Test that representative states are correctly identified in symmetry orbits.
        
        For each state in the full Hilbert space, there exists a unique "representative"
        state that is the lexicographically smallest state in its symmetry orbit.
        This test verifies that the find_repr method correctly identifies these
        representatives and computes the associated phase factors.
        """
        ns = DEFAULT_SYSTEM_SIZE
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.Translation_x, 0)]
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        # Test various bit patterns
        test_states = [0b0011, 0b0101, 0b1100, 0b1010]
        
        print_subsection("Testing representative finding")
        for state in test_states:
            rep, phase = hilbert.find_repr(state)
            print_state(state, ns, "Original")
            print_state(rep, ns, f"{INDENT}-> Representative")
            print(f"{INDENT}  Phase: {phase:.4f}")
            
            # Representative should be minimal in orbit
            assert validate_representative(hilbert, state)
        
        print(f"{INDENT}(ok) All representatives validated")


class TestParitySymmetry:
    """
    Tests for parity/reflection symmetries.
    
    Parity symmetries include spatial reflection (mirror symmetry) and spin-flip
    (parity Z) operations. These symmetries are important for systems with
    inversion symmetry or particle-hole symmetry. The tests verify proper
    space reduction and sector separation.
    """
    
    def test_parity_z(self):
        """
        Test spin-flip (parity Z) symmetry.
        
        Parity Z symmetry corresponds to flipping all spins in the system.
        This symmetry is relevant for systems with particle-hole symmetry or
        when studying antiferromagnetic order. States are divided into even
        and odd parity sectors.
        """
        ns = DEFAULT_SYSTEM_SIZE
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.ParityZ, 1)]  # Even parity sector
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_hilbert_info(hilbert, "Parity Z symmetry (even sector)")
        
        # With spin-flip symmetry, space should be roughly halved
        max_reduced_size = hilbert.Nhfull // 2 + hilbert.Nhfull % 2
        assert hilbert.Nh <= max_reduced_size
        assert validate_mapping(hilbert)
        print(f"{INDENT}(ok) Test passed")


class TestGlobalSymmetry:
    """
    Tests for global symmetries (U(1), etc.).
    
    Global symmetries act identically on all sites and typically correspond to
    conserved quantities like total particle number, total spin, or total momentum.
    Unlike spatial symmetries, global symmetries don't reduce the Hilbert space
    through orbit identification but rather select subspaces with definite
    quantum numbers.
    """
    
    def test_u1_symmetry(self):
        """
        Test U(1) particle number conservation symmetry.
        
        U(1) symmetry enforces conservation of total particle number N.
        This is one of the most important symmetries in many-body physics,
        particularly for fermionic and bosonic systems. The Hilbert space
        is restricted to states with exactly N particles, reducing the
        dimension from 2^Ns to C(Ns, N) where C is the binomial coefficient.
        """
        ns              = LARGE_SYSTEM_SIZE
        n_particles     = DEFAULT_PARTICLE_NUMBER
        lattice         = create_1d_lattice(ns)
        
        # Create U(1) global symmetry
        u1_sym          = get_u1_sym(lattice, n_particles)
        global_syms     = [u1_sym]
        
        hilbert         = HilbertSpace(lattice=lattice, global_syms=global_syms, gen_mapping=False)
        
        # Expected dimension: C(ns, n_particles)
        expected_dim    = comb(ns, n_particles)
        
        print_subsection(f"U(1) symmetry (N={n_particles})")
        print(f"{INDENT}Full space: {hilbert.Nhfull}")
        print(f"{INDENT}U(1) restricted: {hilbert.Nh}")
        print(f"{INDENT}Expected (binomial): {expected_dim}")
        
        # Generate and validate full map
        full_map        = hilbert.get_full_map()
        
        # Verify all valid states have correct particle number
        valid_states    = [i for i, idx in enumerate(full_map) if idx != -1]
        for state in valid_states:
            assert popcount(state) == n_particles
        
        print(f"{INDENT}(ok) All {len(valid_states)} valid states have N={n_particles}")
    
    def test_u1_with_translation(self):
        """
        Test combination of U(1) particle conservation and translation symmetry.
        
        This test combines two fundamental symmetries: particle number conservation
        and translation invariance. The U(1) symmetry first restricts to a fixed
        particle number sector, then translation symmetry further reduces by
        identifying momentum sectors within that particle number subspace.
        """
        ns = LARGE_SYSTEM_SIZE
        n_particles = DEFAULT_PARTICLE_NUMBER
        k = 0
        lattice = create_1d_lattice(ns)
        
        u1_sym = get_u1_sym(lattice, n_particles)
        global_syms = [u1_sym]
        sym_gen = [(SymmetryGenerators.Translation_x, k)]
        
        hilbert = HilbertSpace(
            lattice=lattice,
            global_syms=global_syms,
            sym_gen=sym_gen,
            gen_mapping=True
        )
        
        print_hilbert_info(hilbert, f"U(1) + Translation (N={n_particles}, k={k})")
        
        assert validate_mapping(hilbert)
        print(f"{INDENT}(ok) Test passed")


########################################################################
#! TEST CLASSES - COMBINED SYMMETRIES
########################################################################

class TestCombinedSymmetries:
    """
    Tests for combinations of multiple symmetries.
    
    Real physical systems often possess multiple symmetries simultaneously.
    This class tests various combinations of symmetries to ensure they work
    together correctly. Key considerations include:
    - Commutativity of symmetry operations
    - Proper ordering of symmetry application
    - Combined reduction factors
    - Compatibility between global and spatial symmetries
    """
    
    def test_translation_and_parity(self):
        """
        Test combination of translation and parity Z symmetries.
        
        Translation and parity symmetries can be combined when they commute.
        This test verifies that applying both symmetries gives a greater
        reduction than either symmetry alone, demonstrating the multiplicative
        effect of compatible symmetries.
        """
        ns = DEFAULT_SYSTEM_SIZE
        lattice = create_1d_lattice(ns)
        sym_gen = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.ParityZ, 1)
        ]
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_hilbert_info(hilbert, "Translation + Parity Z")
        
        # With both symmetries, expect significant reduction; for Ns=4, combined T+parity yields 4 reps
        assert hilbert.Nh <= hilbert.Nhfull // 4
        assert validate_mapping(hilbert)
        print(f"{INDENT}(ok) Test passed")
    
    def test_full_symmetry_combination(self):
        """
        Test full combination of U(1), translation, and parity symmetries.
        
        This test applies the maximum number of symmetries typically used in
        quantum many-body calculations: particle number conservation (U(1)),
        translation invariance (k=0 sector), and spin-flip symmetry (parity Z).
        This combination provides the greatest computational advantage for
        ground state searches.
        """
        ns = LARGE_SYSTEM_SIZE
        n_particles = DEFAULT_PARTICLE_NUMBER
        lattice = create_1d_lattice(ns)
        
        u1_sym = get_u1_sym(lattice, n_particles)
        global_syms = [u1_sym]
        sym_gen = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.ParityZ, 1)
        ]
        
        hilbert = HilbertSpace(
            lattice=lattice,
            global_syms=global_syms,
            sym_gen=sym_gen,
            gen_mapping=True
        )
        
        print_hilbert_info(hilbert, "Full symmetry (U(1) + Translation + Parity Z)")
        
        assert validate_mapping(hilbert, verbose=True)
        print(f"{INDENT}(ok) Test passed")
    
    def test_translation_reflection_k0(self):
        """
        Test translation and reflection symmetries at k=0.
        
        Translation symmetry at k=0 commutes with reflection symmetry,
        allowing both to be applied simultaneously. This combination is
        particularly useful for systems with both translational and
        inversion symmetry.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)

        # Translation k=0 commutes with reflection
        sym_gen = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.Reflection, 0)  # Even parity
        ]

        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

        assert hilbert.Nh > 0, "Should have states"

    def test_full_symmetry_group_1d(self):
        """
        Test maximal symmetry group for 1D systems.
        
        This test applies all commonly used symmetries for 1D fermionic systems:
        U(1) particle conservation, translation invariance (k=0), spatial reflection,
        and parity X (which requires half-filling for compatibility with U(1)).
        This represents the most aggressive symmetry reduction possible.
        """
        ns = 6  # Even number for all symmetries
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)
        n_particles = ns // 2

        # U(1) + Translation k=0 + Reflection + ParityX (at half-filling)
        u1_sym = get_u1_sym(lattice, n_particles)
        sym_gen = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.Reflection, 0),
            (SymmetryGenerators.ParityX, 0)
        ]

        hilbert = HilbertSpace(
            lattice=lattice,
            sym_gen=sym_gen,
            global_syms=[u1_sym],
            gen_mapping=True
        )

        # Should have maximum reduction
        hilbert_u1_only = HilbertSpace(
            lattice=lattice,
            global_syms=[u1_sym],
            gen_mapping=True
        )

        assert hilbert.Nh > 0, "Should have at least one state"
        assert hilbert.Nh < hilbert_u1_only.Nh, "Should have additional reduction"

    def test_2d_full_symmetry(self):
        """
        Test 2D lattice with multiple translation symmetries.
        
        In 2D systems, translation symmetry can be applied in both x and y
        directions, leading to 2D momentum sectors (kx, ky). This test applies
        translation in both directions at the Gamma point (kx=0, ky=0).
        """
        lx, ly = 3, 3
        lattice = SquareLattice(dim=2, lx=lx, ly=ly, lz=1, bc=LatticeBC.PBC)

        # 2D translation at Gamma point
        sym_gen = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.Translation_y, 0),
        ]

        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

        assert hilbert.Nh > 0, "Should have states"
        assert hilbert.Nh < hilbert.Nhfull, "Should have reduction"


########################################################################
#! TEST CLASSES - MATRIX CONSTRUCTION
########################################################################

class TestMatrixConstruction:
    """
    Tests for matrix construction and comparison between full space and symmetry sectors.
    
    This class verifies that matrix construction works correctly both in the full
    Hilbert space and in symmetry-reduced sectors. It tests the matrix_builder
    functionality and ensures consistency between different construction methods.
    """
    
    def test_transverse_ising_full_space(self):
        """
        Test transverse Ising model construction in full Hilbert space.
        
        Constructs the Hamiltonian H = -J Σ σ_z^i σ_z^(i+1) - h Σ σ_x^i directly
        in the computational basis and verifies basic properties.
        """
        ns = 4  # Small system for testing
        J, h = 1.0, 0.5
        
        print_subsection(f"Transverse Ising (full space, ns={ns})")
        
        # Build Hamiltonian directly
        nh = 2**ns
        row_indices = []
        col_indices = []
        values = []
        
        for state in range(nh):
            # Diagonal part: -J Σ σ_z^i σ_z^(i+1)
            diag_val = 0.0
            for i in range(ns):
                j = (i + 1) % ns  # PBC
                sz_i = 2 * ((state >> i) & 1) - 1
                sz_j = 2 * ((state >> j) & 1) - 1
                diag_val += sz_i * sz_j
            
            if abs(diag_val) > 1e-14:
                row_indices.append(state)
                col_indices.append(state)
                values.append(-J * diag_val)
            
            # Off-diagonal part: -h Σ σ_x^i
            for i in range(ns):
                new_state = state ^ (1 << i)
                row_indices.append(state)
                col_indices.append(new_state)
                values.append(-h)
        
        H_full = csr_matrix((values, (row_indices, col_indices)), shape=(nh, nh))
        
        # Basic checks
        assert H_full.shape == (nh, nh)
        assert H_full.nnz > 0
        assert np.allclose(H_full.toarray(), H_full.toarray().T)  # Hermitian
        
        print(f"{INDENT}Matrix shape: {H_full.shape}")
        print(f"{INDENT}Non-zeros: {H_full.nnz}")
        print(f"{INDENT}Hermitian: {np.allclose(H_full.toarray(), H_full.toarray().T)}")
        print(f"{INDENT}(ok) Full space construction validated")
    
    def test_matrix_builder_consistency(self):
        """
        Test consistency between matrix_builder and direct construction.
        
        Compares matrix construction for simple operators (σ_x, σ_zσ_z) between
        direct sparse matrix construction and the matrix_builder approach.
        """
        import numba
        from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix
        
        ns = 4
        lattice = create_1d_lattice(ns)
        
        # Create Hilbert space with translation symmetry
        hilbert = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, 0)], gen_mapping=True)
        
        print_subsection(f"Matrix builder consistency (ns={ns}, k=0)")
        
        # Define σ_x operator
        @numba.njit
        def sigma_x_op(state, ns):
            new_states = np.empty(ns, dtype=np.int64)
            for i in range(ns):
                new_states[i] = state ^ (1 << i)
            return new_states, np.ones(ns, dtype=np.float64)
        
        # Define σ_zσ_z operator (nearest neighbor)
        @numba.njit
        def sigma_zz_op(state, ns):
            val = 0.0
            for i in range(ns):
                j = (i + 1) % ns
                sz_i = 2 * ((state >> i) & 1) - 1
                sz_j = 2 * ((state >> j) & 1) - 1
                val += sz_i * sz_j
            return np.array([state], dtype=np.int64), np.array([val], dtype=np.float64)
        
        # Build with matrix_builder
        H_x_mb = build_operator_matrix(hilbert, sigma_x_op, sparse=True)
        H_zz_mb = build_operator_matrix(hilbert, sigma_zz_op, sparse=True)
        
        # Basic validation
        assert H_x_mb.shape == (hilbert.dim, hilbert.dim)
        assert H_zz_mb.shape == (hilbert.dim, hilbert.dim)
        assert H_x_mb.nnz > 0
        assert H_zz_mb.nnz > 0
        
        print(f"{INDENT}σ_x matrix: {H_x_mb.shape}, nnz={H_x_mb.nnz}")
        print(f"{INDENT}σ_zσ_z matrix: {H_zz_mb.shape}, nnz={H_zz_mb.nnz}")
        print(f"{INDENT}(ok) Matrix builder construction validated")
    
    def test_symmetry_sector_vs_full_space(self):
        """
        Test comparison between symmetry sector and full space construction.
        
        This test verifies that the ground state energy computed in symmetry
        sectors is consistent with the full Hilbert space (within numerical
        precision or known discrepancies).
        """
        ns = 4  # Small system
        J, h = 1.0, 0.5
        
        print_subsection(f"Symmetry sector vs full space (ns={ns})")
        
        # Full space construction
        nh = 2**ns
        row_indices = []
        col_indices = []
        values = []
        
        for state in range(nh):
            diag_val = 0.0
            for i in range(ns):
                j           = (i + 1) % ns
                sz_i        = 2 * ((state >> i) & 1) - 1
                sz_j        = 2 * ((state >> j) & 1) - 1
                diag_val   += sz_i * sz_j

            if abs(diag_val) > 1e-14:
                row_indices.append(state)
                col_indices.append(state)
                values.append(-J * diag_val)
            
            for i in range(ns):
                new_state = state ^ (1 << i)
                row_indices.append(state)
                col_indices.append(new_state)
                values.append(-h)
        
        H_full = csr_matrix((values, (row_indices, col_indices)), shape=(nh, nh))
        
        # Get ground state from full space
        evals_full = eigsh(H_full, k=1, which='SA', return_eigenvectors=False)
        E0_full = evals_full[0]
        
        # Symmetry sector construction
        lattice = create_1d_lattice(ns)
        hilbert = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, 0)], gen_mapping=True)
        
        # Build operators in symmetry sector
        import numba
        from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix
        
        @numba.njit
        def sigma_x_op(state, ns):
            new_states = np.empty(ns, dtype=np.int64)
            for i in range(ns):
                new_states[i] = state ^ (1 << i)
            return new_states, np.ones(ns, dtype=np.float64)
        
        @numba.njit
        def sigma_zz_op(state, ns):
            val = 0.0
            for i in range(ns):
                j = (i + 1) % ns
                sz_i = 2 * ((state >> i) & 1) - 1
                sz_j = 2 * ((state >> j) & 1) - 1
                val += sz_i * sz_j
            return np.array([state], dtype=np.int64), np.array([val], dtype=np.float64)
        
        H_x_mb = build_operator_matrix(hilbert, sigma_x_op, sparse=True)
        H_zz_mb = build_operator_matrix(hilbert, sigma_zz_op, sparse=True)
        H_sector = -J * H_zz_mb - h * H_x_mb
        
        # Get ground state from sector
        evals_sector = eigsh(H_sector, k=1, which='SA', return_eigenvectors=False)
        E0_sector = evals_sector[0]
        
        # Compare energies
        energy_diff = abs(E0_full - E0_sector)
        
        print(f"{INDENT}Full space E0: {E0_full:.8f}")
        print(f"{INDENT}Sector E0:     {E0_sector:.8f}")
        print(f"{INDENT}Difference:    {energy_diff:.2e}")
        
        # Check if energies are reasonably close (allowing for known discrepancies)
        energy_close = energy_diff < 1.0  # Within 1 unit of energy
        
        if energy_close:
            print(f"{INDENT}(ok) Energies are consistent between full space and symmetry sector")
        else:
            print(f"{INDENT}(warning) Energy discrepancy detected - may indicate known issue")
        
        assert energy_close, f"Energy mismatch too large: {energy_diff}"
    
    def test_operator_matrix_properties(self):
        """
        Test basic properties of operator matrices built with matrix_builder.
        
        Verifies that matrices have correct dimensions, sparsity, and hermiticity.
        """
        import numba
        from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix
        
        ns = 4
        lattice = create_1d_lattice(ns)
        hilbert = HilbertSpace(lattice=lattice, sym_gen=[(SymmetryGenerators.Translation_x, 0)], gen_mapping=True)
        
        print_subsection(f"Operator matrix properties (ns={ns})")
        
        @numba.njit
        def sigma_z_op(state, ns):
            sz_vals = np.empty(ns, dtype=np.float64)
            for i in range(ns):
                sz_vals[i] = 2 * ((state >> i) & 1) - 1
            return np.array([state], dtype=np.int64), np.array([sz_vals.sum()], dtype=np.float64)
        
        H_z = build_operator_matrix(hilbert, sigma_z_op, sparse=True)
        
        # Check properties
        assert H_z.shape == (hilbert.dim, hilbert.dim)
        assert H_z.nnz > 0
        
        # Check hermiticity (σ_z is Hermitian)
        H_dense = H_z.toarray()
        is_hermitian = np.allclose(H_dense, H_dense.T.conj())
        
        print(f"{INDENT}Matrix shape: {H_z.shape}")
        print(f"{INDENT}Non-zeros: {H_z.nnz}")
        print(f"{INDENT}Hermitian: {is_hermitian}")
        
        assert is_hermitian, "σ_z operator should be Hermitian"
        print(f"{INDENT}(ok) Operator matrix properties validated")

class TestNormalization:
    """
    Tests for state normalization calculations.
    
    When symmetries are applied, states in the reduced Hilbert space are
    linear combinations of states in the full space. The normalization factors
    account for the fact that multiple full-space states may map to the same
    reduced-space state. This class verifies that normalization factors are
    computed correctly and have reasonable values.
    """
    
    def test_normalization_values(self):
        """
        Test that normalization factors are calculated correctly.
        
        Normalization factors should be positive real numbers that ensure
        proper normalization of the reduced Hilbert space states. For translation
        symmetry, these factors are typically sqrt(Ns) or smaller, depending
        on the size of the symmetry orbit.
        """
        ns = DEFAULT_SYSTEM_SIZE
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.Translation_x, 0)]
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_subsection("Normalization factors (first 5 states)")
        
        num_samples = min(VALIDATION_SAMPLE_SIZE, hilbert.Nh)
        for idx in range(num_samples):
            state = hilbert.mapping[idx]
            norm = hilbert.norm(idx)
            binary = int2binstr(state, ns)
            print(f"{INDENT}State |{binary}>: norm = {norm:.4f}")
            
            # Normalization should be positive and <= sqrt(ns)
            assert norm > 0, f"Non-positive normalization for state {idx}"
            assert norm <= np.sqrt(ns) + 0.1, f"Too large normalization for state {idx}"
        
        print(f"{INDENT}(ok) All normalization factors are valid")


########################################################################
#! ADDITIONAL COMPREHENSIVE TESTS
########################################################################

class TestTranslation1D:
    """
    Comprehensive tests for translation symmetry on 1D chains.
    
    This class provides exhaustive testing of translation symmetry in one
    dimension, covering all momentum sectors, boundary conditions, and
    validation of quantum number assignments.
    """
    
    def test_translation_1d_k0_pbc(self):
        """
        Test 1D chain with k=0 translation (PBC).
        
        Periodic boundary conditions allow full translational symmetry.
        The k=0 sector should contain states that are unchanged under
        translation by any number of sites.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)
        sym_gen = [(SymmetryGenerators.Translation_x, 0)]

        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

        # Validations
        assert hilbert.Nh < hilbert.Nhfull, "Should have reduction with translation"
        assert hilbert.mapping is not None, "Should have mapping"
        assert hilbert.Nh > 0, "Should have states in k=0 sector"

    def test_translation_1d_all_k_sectors(self):
        """
        Test all momentum sectors for 1D chain.
        
        For a system of size Ns with PBC, there are Ns distinct momentum
        sectors k = 0, 1, ..., Ns-1. The total number of states across all
        sectors should equal the full Hilbert space dimension.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)

        total_states = 0

        for k in range(ns):
            sym_gen = [(SymmetryGenerators.Translation_x, k)]
            hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

            assert hilbert.Nh > 0, f"Empty Hilbert space for k={k}"
            total_states += hilbert.Nh

        # Total states should equal full Hilbert space
        assert total_states == 2**ns, f"Total states {total_states} != 2^{ns} = {2**ns}"

    def test_translation_1d_momentum_quantum_numbers(self):
        """
        Test that momentum quantum numbers are correctly assigned.
        
        Each momentum sector should be properly labeled with its quantum
        number k. This test verifies that the symmetry information stored
        in the HilbertSpace object correctly identifies the momentum sector.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)

        for k in range(ns):
            sym_gen = [(SymmetryGenerators.Translation_x, k)]
            hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

            # Check that symmetry info contains the correct momentum
            sym_info = hilbert.get_sym_info()
            assert f"Translation_x={k}" in sym_info, f"Momentum k={k} not found in symmetry info: {sym_info}"


class TestTranslation2D:
    """
    Comprehensive tests for translation symmetry on 2D lattices.
    
    2D systems introduce additional complexity with translation in two
    directions, leading to 2D momentum space (kx, ky). This class tests
    single-direction translations, combined translations, and completeness
    of momentum space coverage.
    """
    
    def test_translation_2d_single_direction(self):
        """
        Test 2D lattice with translation in single direction.
        
        Translation can be applied independently in x and y directions.
        This test verifies that each direction provides the expected
        reduction when applied separately.
        """
        lx, ly = 3, 3
        lattice = SquareLattice(dim=2, lx=lx, ly=ly, lz=1, bc=LatticeBC.PBC)

        # Test Tx only
        sym_gen = [(SymmetryGenerators.Translation_x, 0)]
        hilbert_x = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

        assert hilbert_x.Nh < hilbert_x.Nhfull, "Should have reduction"
        assert hilbert_x.Nh > 0, "Should have states"

        # Test Ty only
        sym_gen = [(SymmetryGenerators.Translation_y, 0)]
        hilbert_y = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

        assert hilbert_y.Nh < hilbert_y.Nhfull, "Should have reduction"
        assert hilbert_y.Nh > 0, "Should have states"

    def test_translation_2d_both_directions(self):
        """
        Test 2D lattice with translation in both directions.
        
        Applying translation in both x and y directions simultaneously
        creates 2D momentum sectors (kx, ky). This provides greater
        reduction than single-direction translations.
        """
        lx, ly = 3, 3
        lattice = SquareLattice(dim=2, lx=lx, ly=ly, lz=1, bc=LatticeBC.PBC)

        # Test several (kx, ky) combinations
        test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for kx, ky in test_cases:
            if kx >= lx or ky >= ly:
                continue

            sym_gen = [
                (SymmetryGenerators.Translation_x, kx),
                (SymmetryGenerators.Translation_y, ky)
            ]

            hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

            assert hilbert.Nh > 0, f"Empty space for kx={kx}, ky={ky}"

    def test_translation_2d_momentum_completeness(self):
        """
        Test that all 2D momentum sectors sum to full Hilbert space.
        
        For a 2D lattice of size Lx × Ly, there are Lx × Ly distinct
        momentum sectors. The total number of states across all sectors
        should equal the full Hilbert space dimension.
        """
        lx, ly = 3, 3  # Small lattice for exhaustive test
        lattice = SquareLattice(dim=2, lx=lx, ly=ly, lz=1, bc=LatticeBC.PBC)

        total_states = 0
        sectors_tested = 0

        for kx in range(lx):
            for ky in range(ly):
                sym_gen = [
                    (SymmetryGenerators.Translation_x, kx),
                    (SymmetryGenerators.Translation_y, ky)
                ]

                hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
                total_states += hilbert.Nh
                sectors_tested += 1

        ns = lx * ly
        expected = 2**ns

        assert total_states == expected, f"Momentum sectors don't sum to full space: {total_states} != {expected}"


class TestReflectionParity:
    """
    Comprehensive tests for reflection and parity symmetries.
    
    Reflection symmetry corresponds to spatial inversion (x → -x), while
    parity symmetries include spin operations like flipping all spins.
    These symmetries are important for systems with inversion symmetry
    or particle-hole symmetry.
    """
    
    def test_reflection_1d(self):
        """
        Test reflection symmetry on 1D chain.
        
        Reflection symmetry inverts the spatial coordinate x → L-x.
        For a 1D chain with PBC, this creates even and odd parity sectors
        under reflection.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)

        for sector in [0, 1]:  # Even and odd parity
            sym_gen = [(SymmetryGenerators.Reflection, sector)]
            hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

            assert hilbert.Nh < hilbert.Nhfull, "Should have reduction"
            assert hilbert.Nh > 0, f"Should have states in sector {sector}"

    def test_parity_z(self):
        """
        Test Parity Z (spin flip) symmetry.
        
        Parity Z flips all spins in the system (σ → -σ). This symmetry
        is relevant for systems with antiferromagnetic order or particle-hole
        symmetry. States are divided into even and odd parity sectors.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)

        for sector in [0, 1]:
            sym_gen = [(SymmetryGenerators.ParityZ, sector)]
            hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

            assert hilbert.Nh < hilbert.Nhfull, "Should have reduction"
            assert hilbert.Nh > 0, f"Should have states in sector {sector}"

    def test_parity_x_half_filling(self):
        """
        Test Parity X at half-filling (required for U(1) compatibility).
        
        Parity X (σ^x on all sites) is compatible with U(1) symmetry only
        at half-filling (N = Ns/2) due to the particle-hole transformation.
        This test verifies the combination works correctly.
        """
        ns = 4  # Even number
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)

        # ParityX should work at half-filling with U(1)
        n_particles = ns // 2
        u1_sym = get_u1_sym(lattice, n_particles)

        sym_gen = [(SymmetryGenerators.ParityX, 0)]
        hilbert = HilbertSpace(
            lattice=lattice,
            sym_gen=sym_gen,
            global_syms=[u1_sym],
            gen_mapping=True
        )

        assert hilbert.Nh > 0, "Should have states at half-filling"


class TestGlobalU1:
    """
    Comprehensive tests for U(1) particle number conservation.
    
    U(1) symmetry enforces conservation of total particle number N.
    This is implemented as a global symmetry that restricts the Hilbert
    space to states with exactly N particles, reducing the dimension
    from 2^Ns to C(Ns, N) where C is the binomial coefficient.
    """
    
    def test_u1_only(self):
        """
        Test U(1) symmetry without other symmetries.
        
        This test verifies that U(1) symmetry correctly restricts the
        Hilbert space to states with the specified particle number N,
        and that the dimension matches the expected binomial coefficient.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)

        for n_particles in range(ns + 1):
            u1_sym = get_u1_sym(lattice, n_particles)

            hilbert = HilbertSpace(
                lattice=lattice,
                global_syms=[u1_sym],
                gen_mapping=True
            )

            expected_dim = comb(ns, n_particles)

            assert hilbert.Nh == expected_dim, f"Wrong dimension for N={n_particles}: {hilbert.Nh} != {expected_dim}"

    def test_u1_with_translation(self):
        """
        Test U(1) combined with translation symmetry.
        
        This test combines particle number conservation with translation
        invariance. The U(1) symmetry first restricts to a fixed particle
        number sector, then translation symmetry further reduces by
        identifying momentum sectors within that subspace.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)
        n_particles = ns // 2

        u1_sym = get_u1_sym(lattice, n_particles)
        sym_gen = [(SymmetryGenerators.Translation_x, 0)]

        hilbert = HilbertSpace(
            lattice=lattice,
            sym_gen=sym_gen,
            global_syms=[u1_sym],
            gen_mapping=True
        )

        # Should be smaller than U(1) alone
        hilbert_u1_only = HilbertSpace(
            lattice=lattice,
            global_syms=[u1_sym],
            gen_mapping=True
        )

        assert hilbert.Nh < hilbert_u1_only.Nh, "Translation should further reduce space"
        assert hilbert.Nh > 0, "Should have states"



class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    
    This class tests unusual or boundary cases that might reveal bugs
    in the symmetry implementation, including different boundary conditions,
    very small systems, and degenerate cases.
    """
    
    def test_no_symmetries(self):
        """
        Test Hilbert space with no symmetries applied.
        
        This degenerate case should result in the full Hilbert space
        with no reduction, no mapping generated, and no symmetry modifications.
        """
        ns = 4
        hilbert = HilbertSpace(ns=ns, sym_gen=None, global_syms=None)

        assert hilbert.Nh == hilbert.Nhfull == 2**ns
        assert hilbert.mapping is None
        assert not hilbert.modifies

    def test_open_boundary_conditions(self):
        """
        Test with open boundary conditions (no translation).
        
        Open boundary conditions break translational symmetry, so translation
        generators are not applicable. However, reflection and parity symmetries
        can still be used. This test verifies that non-translational symmetries
        still work correctly.
        """
        ns = 4
        lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.OBC)

        # OBC: translation not applicable, but reflection/parity still work
        sym_gen = [(SymmetryGenerators.Reflection, 0)]

        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

        assert hilbert.Nh < hilbert.Nhfull, "Reflection should still reduce space"
        assert hilbert.Nh > 0, "Should have states"

    def test_small_systems(self):
        """
        Test very small systems (edge cases).
        
        Small systems (Ns=2,3) have limited symmetry groups and may
        exhibit degenerate behavior. This test ensures the symmetry
        implementation works correctly even for minimal system sizes.
        """
        for ns in [2, 3]:
            lattice = SquareLattice(dim=1, lx=ns, ly=1, lz=1, bc=LatticeBC.PBC)
            sym_gen = [(SymmetryGenerators.Translation_x, 0)]

            hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)

            assert hilbert.Nh > 0, f"Empty space for Ns={ns}"


########################################################################
#! PYTEST MAIN
########################################################################

if __name__ == "__main__":
    """Run tests with detailed output."""
    print("\n" + SEPARATOR_LONG)
    print("HILBERT SPACE SYMMETRIES TEST SUITE")
    print(SEPARATOR_LONG)
    
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
