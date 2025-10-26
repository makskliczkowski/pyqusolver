"""
Test suite for Hilbert space symmetries

This module tests the symmetry implementation in the HilbertSpace class,
including:
- Translation symmetries (1D, 2D, 3D)
- Parity/Reflection symmetries
- U(1) global symmetries (particle number conservation)
- Combined symmetries
- Representative finding and normalization
- Mapping generation and validation

File    : test/hilbert/test_hilbert_symmetries.py
Author  : Maksymilian Kliczkowski
Date    : October 2025
"""

########################################################################
#! RESOLVE PATH
########################################################################

import sys
import os
from pathlib import Path

# Add QES to path
qes_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(qes_path))

########################################################################
#! IMPORTS
########################################################################

import numpy as np
import pytest
from typing import List, Tuple
from math import comb

# Import QES modules
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Operator.operator import SymmetryGenerators, LocalSpace, LocalSpaceTypes
from QES.Algebra.globals import GlobalSymmetry, get_u1_sym
from QES.general_python.lattices.square import SquareLattice
from QES.general_python.lattices.lattice import LatticeBC
from QES.general_python.common.binary import int2binstr, popcount

########################################################################
#! CONSTANTS & CONFIGURATION
########################################################################

# Test parameters
DEFAULT_SYSTEM_SIZE = 4
LARGE_SYSTEM_SIZE = 6
DEFAULT_PARTICLE_NUMBER = 3
VALIDATION_SAMPLE_SIZE = 5

# Display settings
SEPARATOR_LONG = "=" * 70
SEPARATOR_SHORT = "=" * 60
INDENT = "  "

########################################################################
#! HELPER FUNCTIONS - OUTPUT
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
    binary = int2binstr(state, ns)
    particle_num = popcount(state)
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
        print(f"{INDENT}❌ Representative validation failed for state {state}")
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
                print(f"{INDENT}❌ [{idx}] {int2binstr(state, hilbert.Ns)} "
                      f"is not its own representative")
        
        # Check 2: State satisfies global symmetries
        is_valid, err_msg = validate_global_symmetries(hilbert, state_int)
        if not is_valid:
            errors.append(f"State {state} {err_msg}")
            if verbose:
                print(f"{INDENT}❌ [{idx}] {int2binstr(state, hilbert.Ns)} {err_msg}")
        
        # Check 3: Normalization is positive
        norm = hilbert.norm(idx)
        if norm <= 0:
            errors.append(f"State {state} has non-positive norm {norm}")
            if verbose:
                print(f"{INDENT}❌ [{idx}] {int2binstr(state, hilbert.Ns)} "
                      f"has invalid norm {norm}")
    
    # Print results
    if errors:
        print(f"\n{INDENT}❌ Found {len(errors)} errors in mapping:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"{INDENT}   {err}")
        if len(errors) > 10:
            print(f"{INDENT}   ... and {len(errors)-10} more")
        return False
    else:
        print(f"{INDENT}✅ Mapping validation passed! "
              f"All {len(hilbert.mapping)} states are valid representatives")
        return True


########################################################################
#! TEST CLASSES - BASIC
########################################################################

class TestHilbertSpaceBasic:
    """Basic tests for Hilbert space without symmetries."""
    
    def test_initialization_no_symmetries(self):
        """Test creating a Hilbert space without symmetries."""
        ns = DEFAULT_SYSTEM_SIZE
        hilbert = HilbertSpace(ns=ns, sym_gen=None, global_syms=None)
        
        assert hilbert.Ns == ns
        assert hilbert.Nh == hilbert.Nhfull == 2**ns
        assert hilbert.mapping is None
        assert not hilbert.modifies
        
        print_hilbert_info(hilbert, "Basic Hilbert space (no symmetries)")
        print(f"{INDENT}✅ Test passed")
    
    def test_local_space_spin_half(self):
        """Test Hilbert space with spin-1/2 local space."""
        ns = DEFAULT_SYSTEM_SIZE
        local_space = LocalSpace.default()  # spin-1/2
        hilbert = HilbertSpace(ns=ns, local_space=local_space)
        
        assert hilbert.local == 2
        assert hilbert.Nhfull == 2**ns
        
        print_subsection("Spin-1/2 system")
        print(f"{INDENT}Local dimension: {hilbert.local}")
        print(f"{INDENT}Full Hilbert space: {hilbert.Nhfull}")
        print(f"{INDENT}✅ Test passed")


########################################################################
#! TEST CLASSES - TRANSLATION SYMMETRY
########################################################################

class TestTranslationSymmetry:
    """Tests for translation symmetry in 1D, 2D, and 3D."""
    
    def test_translation_1d_k0(self):
        """Test 1D translation with k=0 (uniform sector)."""
        ns = DEFAULT_SYSTEM_SIZE
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.Translation_x, 0)]
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_hilbert_info(hilbert, "1D Translation k=0")
        
        # Validation
        assert hilbert.Nh < hilbert.Nhfull
        assert hilbert.mapping is not None
        assert validate_mapping(hilbert)
        print(f"{INDENT}✅ Test passed")
    
    def test_translation_1d_momentum(self):
        """Test 1D translation with non-zero momentum."""
        ns = LARGE_SYSTEM_SIZE
        k = 1  # momentum sector
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.Translation_x, k)]
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_hilbert_info(hilbert, f"1D Translation k={k}")
        
        assert hilbert.Nh < hilbert.Nhfull
        assert validate_mapping(hilbert)
        print(f"{INDENT}✅ Test passed")
    
    def test_representative_finding_1d(self):
        """Test that representatives are correctly identified."""
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
        
        print(f"{INDENT}✅ All representatives validated")


########################################################################
#! TEST CLASSES - PARITY SYMMETRY
########################################################################

class TestParitySymmetry:
    """Tests for parity/reflection symmetries."""
    
    def test_parity_z(self):
        """Test spin-flip (parity Z) symmetry."""
        ns = DEFAULT_SYSTEM_SIZE
        lattice = create_1d_lattice(ns)
        sym_gen = [(SymmetryGenerators.ParityZ, 1)]  # Even parity sector
        
        hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, gen_mapping=True)
        
        print_hilbert_info(hilbert, "Parity Z symmetry (even sector)")
        
        # With spin-flip symmetry, space should be roughly halved
        max_reduced_size = hilbert.Nhfull // 2 + hilbert.Nhfull % 2
        assert hilbert.Nh <= max_reduced_size
        assert validate_mapping(hilbert)
        print(f"{INDENT}✅ Test passed")


########################################################################
#! TEST CLASSES - GLOBAL SYMMETRY
########################################################################

class TestGlobalSymmetry:
    """Tests for global symmetries (U(1), etc.)."""
    
    def test_u1_symmetry(self):
        """Test U(1) particle number conservation."""
        ns = LARGE_SYSTEM_SIZE
        n_particles = DEFAULT_PARTICLE_NUMBER
        lattice = create_1d_lattice(ns)
        
        # Create U(1) global symmetry
        u1_sym = get_u1_sym(lattice, n_particles)
        global_syms = [u1_sym]
        
        hilbert = HilbertSpace(lattice=lattice, global_syms=global_syms, 
                              gen_mapping=False)
        
        # Expected dimension: C(ns, n_particles)
        expected_dim = comb(ns, n_particles)
        
        print_subsection(f"U(1) symmetry (N={n_particles})")
        print(f"{INDENT}Full space: {hilbert.Nhfull}")
        print(f"{INDENT}U(1) restricted: {hilbert.Nh}")
        print(f"{INDENT}Expected (binomial): {expected_dim}")
        
        # Generate and validate full map
        full_map = hilbert.get_full_map_int()
        
        # Verify all states have correct particle number
        for state in full_map:
            assert popcount(state) == n_particles
        
        print(f"{INDENT}✅ All {len(full_map)} states have N={n_particles}")
    
    def test_u1_with_translation(self):
        """Test combination of U(1) and translation symmetries."""
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
        print(f"{INDENT}✅ Test passed")


########################################################################
#! TEST CLASSES - COMBINED SYMMETRIES
########################################################################

class TestCombinedSymmetries:
    """Tests for combinations of multiple symmetries."""
    
    def test_translation_and_parity(self):
        """Test translation + parity Z."""
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
        print(f"{INDENT}✅ Test passed")
    
    def test_full_symmetry_combination(self):
        """Test U(1) + Translation + Parity."""
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
        print(f"{INDENT}✅ Test passed")


########################################################################
#! TEST CLASSES - NORMALIZATION
########################################################################

class TestNormalization:
    """Tests for state normalization calculations."""
    
    def test_normalization_values(self):
        """Test that normalization factors are calculated correctly."""
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
        
        print(f"{INDENT}✅ All normalization factors are valid")


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
