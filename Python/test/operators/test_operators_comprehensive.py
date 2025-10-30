#!/usr/bin/env python3
"""
Comprehensive Operator Test Suite
==================================

This module provides a comprehensive test suite for quantum operators across all backends.
Tests cover:
- spin operators, 
- fermionic operators, 
- operator combinations, and 
- backend compatibility.

Tests are organized by functionality:
- Catalog tests         : Verify operator availability and metadata
- Backend tests         : Test int, NumPy, and JAX backends
- Combination tests     : Test operator multiplication and composition
- Fermionic tests       : Test creation/annihilation with correct signs
- Momentum tests        : Test k-space operators

All tests include verbose assertions with detailed error messages.

----------------------------------------------------------------------------
File        : Python/test/operators/test_operators_comprehensive.py
Author      : Maksymilian Kliczkowski
Date        : 30.10.2025
Version     : 1.0
----------------------------------------------------------------------------
run: pytest Python/test/operators/test_operators_comprehensive.py
"""

import numpy as np
import pytest
from typing import Union, List

# Core imports
try:
    from QES.Algebra.Hilbert.hilbert_local      import LocalSpace
    from QES.Algebra.hilbert                    import HilbertSpace
    from QES.Algebra.Operator.operator          import OperatorTypeActing
    from QES.general_python.lattices.square     import SquareLattice
    from QES.general_python.lattices.lattice    import LatticeBC
    from QES.general_python.algebra.utils       import JAX_AVAILABLE
except ImportError as e:
    raise ImportError(f"Required QES modules not found: {e}")

# Operator imports
try:
    from QES.Algebra.Operator import operators_spin as op_spin
    from QES.Algebra.Operator import operators_spinless_fermions as op_ferm
except ImportError as e:
    raise ImportError(f"Required QES operator modules not found: {e}")

# JAX imports (conditional)
if JAX_AVAILABLE:
    import jax.numpy as jnp
else:
    jnp = None

# ----------------------------------------------------------------------------
#! Comprehensive Operator Test Suite
# ----------------------------------------------------------------------------

class TestOperatorCatalog:
    """Test operator catalog and metadata."""

    def test_spin_catalog_keys(self):
        """Test that all required spin operators are available in the catalog."""
        space               = LocalSpace.default_spin_half()
        keys                = set(space.list_operator_keys())

        required_operators  = {"sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus"}
        missing_operators   = required_operators - keys

        assert not missing_operators, f"Missing spin operators: {missing_operators}"
        print(f"(ok)  Found all required spin operators: {required_operators}")

        # Test operator metadata
        sigma_plus          = space.get_op("sigma_plus")
        assert "raising" in sigma_plus.tags, "sigma_plus should have 'raising' tag"
        print("(ok)  sigma_plus has correct 'raising' tag")

    # ----------------------------------------------------------------------------

    def test_fermion_catalog_keys(self):
        """Test that fermionic operators are available."""
        space               = LocalSpace.default_fermion_spinless()
        keys                = set(space.list_operator_keys())
        required_operators  = {"c", "c_dag", "n"}
        missing_operators   = required_operators - keys

        assert not missing_operators, f"Missing fermion operators: {missing_operators}"
        print(f"(ok)  Found all required fermion operators: {required_operators}")

    # ----------------------------------------------------------------------------

    def test_anyon_catalog_keys(self):
        """Test that anyon operators are available."""
        theta               = np.pi / 2  # semion
        space               = LocalSpace.default_abelian_anyon(statistics_angle=theta)
        keys                = set(space.list_operator_keys())
        required_operators  = {"a", "a_dag"}
        missing_operators   = required_operators - keys

        assert not missing_operators, f"Missing anyon operators: {missing_operators}"
        print(f"(ok)  Found all required anyon operators: {required_operators}")


# --------------------------------------------------------------------------------

class TestFermionicOperators:
    """Test fermionic creation/annihilation operators with correct signs."""

    def test_fermion_creation_sign(self):
        """Test fermionic creation operator with correct Jordan-Wigner signs."""
        space = LocalSpace.default_fermion_spinless()
        creation = space.get_op("c_dag").kernels

        # Test: initial state |100⟩, create on site 1 (middle)
        # Should give |110⟩ with coefficient -1 (parity from site 0)
        out_state, coeff = creation.fun_int(0b100, 3, [1])

        assert out_state[0] == 0b110, f"Expected state 0b110, got {out_state[0]:0b}"
        assert coeff[0] == -1.0, f"Expected coefficient -1.0, got {coeff[0]}"
        print("(ok)  Fermionic creation sign correct: |100⟩ → |110⟩ with coeff -1")

        # Test: attempt to create on already occupied site gives zero
        out_state, coeff = creation.fun_int(0b100, 3, [0])
        assert coeff[0] == 0.0, f"Expected zero coefficient for double occupation, got {coeff[0]}"
        print("(ok)  Double occupation correctly gives zero coefficient")

    def test_fermion_annihilation_sign(self):
        """Test fermionic annihilation operator."""
        space = LocalSpace.default_fermion_spinless()
        annihilation = space.get_op("c").kernels

        # Test: state |110⟩, annihilate site 1
        # Should give |100⟩ with coefficient -1
        out_state, coeff = annihilation.fun_int(0b110, 3, [1])

        assert out_state[0] == 0b100, f"Expected state 0b100, got {out_state[0]:0b}"
        assert coeff[0] == -1.0, f"Expected coefficient -1.0, got {coeff[0]}"
        print("(ok)  Fermionic annihilation sign correct: |110⟩ → |100⟩ with coeff -1")

        # Test: annihilate empty site gives zero
        out_state, coeff = annihilation.fun_int(0b100, 3, [1])
        assert coeff[0] == 0.0, f"Expected zero coefficient for empty site, got {coeff[0]}"
        print("(ok)  Annihilation of empty site correctly gives zero coefficient")

    def test_anyon_phase_semion(self):
        """Test anyon statistics with semion phase."""
        theta = np.pi / 2
        space = LocalSpace.default_abelian_anyon(statistics_angle=theta)
        creation = space.get_op("a_dag").kernels

        out_state, coeff = creation.fun_int(0b100, 3, [1])
        expected_coeff = np.exp(1j * theta)

        assert out_state[0] == 0b110, f"Expected state 0b110, got {out_state[0]:0b}"
        assert np.allclose(coeff[0], expected_coeff), f"Expected {expected_coeff}, got {coeff[0]}"
        print(f"(ok)  Semion phase correct: coefficient = {coeff[0]} ≈ exp(iπ/2) = {expected_coeff}")


class TestOperatorCombinations:
    """Test operator combinations and global operators across backends."""

    @pytest.fixture
    def lattice(self):
        """Create a small 1D lattice for testing."""
        return SquareLattice(dim=1, lx=3, ly=1, lz=1, bc=LatticeBC.PBC)

    def test_spin_global_operators_int_backend(self, lattice):
        """Test global spin operators on integer states."""
        # Create global operators
        sig_x_g = op_spin.sig_x(lattice=lattice, type_act=OperatorTypeActing.Global)
        sig_z_g = op_spin.sig_z(lattice=lattice, type_act=OperatorTypeActing.Global)

        # Test single int state |000⟩
        state = 0
        result_x = sig_x_g(state)
        assert isinstance(result_x[0], np.ndarray), "Result states should be array"
        assert isinstance(result_x[1], np.ndarray), "Result coeffs should be array"
        assert result_x[0].shape == (1,), f"Expected shape (1,), got {result_x[0].shape}"
        assert result_x[1].shape == (1,), f"Expected shape (1,), got {result_x[1].shape}"
        print(f"(ok)  σₓ on |000⟩: {len(result_x[0])} states generated")

        result_z = sig_z_g(state)
        assert isinstance(result_z[0], np.ndarray), "Result states should be array"
        assert isinstance(result_z[1], np.ndarray), "Result coeffs should be array"
        print(f"(ok)  σz on |000⟩: {len(result_z[0])} states generated")

        # Test combination
        sig_x_sig_z = sig_x_g * sig_z_g
        result_comb = sig_x_sig_z(state)
        assert isinstance(result_comb[0], np.ndarray), "Combined result states should be array"
        assert isinstance(result_comb[1], np.ndarray), "Combined result coeffs should be array"
        print(f"(ok)  σₓσz on |000⟩: {len(result_comb[0])} states generated")

    def test_spin_global_operators_numpy_backend(self, lattice):
        """Test global spin operators on NumPy states."""
        sig_x_g = op_spin.sig_x(lattice=lattice, type_act=OperatorTypeActing.Global)
        sig_z_g = op_spin.sig_z(lattice=lattice, type_act=OperatorTypeActing.Global)

        # Test numpy state |000⟩
        np_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result_x = sig_x_g(np_state)
        assert result_x[0].shape == (1, 8), f"Expected shape (1, 8), got {result_x[0].shape}"
        assert result_x[1].shape == (1,), f"Expected shape (1,), got {result_x[1].shape}"
        print(f"(ok)  σₓ on NumPy |000⟩: output shape {result_x[0].shape}")

        # Test combination
        sig_x_sig_z = sig_x_g * sig_z_g
        result_comb = sig_x_sig_z(np_state)
        assert result_comb[0].shape == (1, 8), f"Expected shape (1, 8), got {result_comb[0].shape}"
        assert result_comb[1].shape == (1,), f"Expected shape (1,), got {result_comb[1].shape}"
        print(f"(ok)  σₓσz on NumPy |000⟩: output shape {result_comb[0].shape}")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_spin_global_operators_jax_backend(self, lattice):
        """Test global spin operators on JAX states."""
        sig_x_g = op_spin.sig_x(lattice=lattice, type_act=OperatorTypeActing.Global)
        sig_z_g = op_spin.sig_z(lattice=lattice, type_act=OperatorTypeActing.Global)

        # Test JAX state |000⟩
        jnp_state = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result_x = sig_x_g(jnp_state)
        assert result_x[0].shape == (1, 8), f"Expected shape (1, 8), got {result_x[0].shape}"
        assert result_x[1].shape == (1,), f"Expected shape (1,), got {result_x[1].shape}"
        print(f"(ok)  σₓ on JAX |000⟩: output shape {result_x[0].shape}")

        # Test combination
        sig_x_sig_z = sig_x_g * sig_z_g
        result_comb = sig_x_sig_z(jnp_state)
        assert result_comb[0].shape == (1, 8), f"Expected shape (1, 8), got {result_comb[0].shape}"
        assert result_comb[1].shape == (1,), f"Expected shape (1,), got {result_comb[1].shape}"
        print(f"(ok)  σₓσz on JAX |000⟩: output shape {result_comb[0].shape}")


class TestFermionOperators:
    """Test fermion operators across backends."""

    @pytest.fixture
    def lattice(self):
        """Create a small 1D lattice for testing."""
        return SquareLattice(dim=1, lx=3, ly=1, lz=1, bc=LatticeBC.PBC)

    def test_fermion_creation_annihilation_int(self, lattice):
        """Test fermion creation and annihilation on integer states."""
        c_dag = op_ferm.cdag(lattice=lattice, type_act=OperatorTypeActing.Local)
        c = op_ferm.c(lattice=lattice, type_act=OperatorTypeActing.Local)

        # Test creation on empty site
        state = 0b000  # empty
        result = c_dag(state, 0)
        assert result[0][0] == 0b100, f"Expected 0b100, got {result[0][0]:0b}"
        assert result[1][0] == 1.0, f"Expected 1.0, got {result[1][0]}"
        print("(ok)  Creation on empty site: |000⟩ → |100⟩")

        # Test annihilation on occupied site
        state = 0b100
        result = c(state, 0)
        assert result[0][0] == 0b000, f"Expected 0b000, got {result[0][0]:0b}"
        assert result[1][0] == 1.0, f"Expected 1.0, got {result[1][0]}"
        print("(ok)  Annihilation on occupied site: |100⟩ → |000⟩")

        # Test fermionic sign
        state = 0b100  # site 0 occupied
        result = c_dag(state, 1)
        assert result[0][0] == 0b110, f"Expected 0b110, got {result[0][0]:0b}"
        assert result[1][0] == -1.0, f"Expected -1.0, got {result[1][0]}"
        print("(ok)  Fermionic sign: |100⟩ → |110⟩ with coefficient -1")

    def test_fermion_operators_numpy(self, lattice):
        """Test fermion operators on NumPy arrays."""
        c_dag = op_ferm.cdag(lattice=lattice, type_act=OperatorTypeActing.Local)

        # Test numpy state
        np_state = np.array([0.0, 0.0, 0.0])  # empty
        result = c_dag(np_state, 0)
        expected_state = np.array([1.0, 0.0, 0.0])
        assert np.allclose(result[0], expected_state), f"State mismatch: expected {expected_state}, got {result[0]}"
        assert result[1][0] == 1.0, f"Expected coefficient 1.0, got {result[1][0]}"
        print("(ok)  Creation on NumPy empty state: [0,0,0] → [1,0,0]")


class TestHilbertIntegration:
    """Test integration with HilbertSpace."""

    def test_hilbert_build_local_operator(self):
        """Test building operators through HilbertSpace."""
        space = LocalSpace.default_fermion_spinless()
        hilbert = HilbertSpace(ns=3, local_space=space, backend="default")

        op = hilbert.build_local_operator("c_dag")
        assert op.type_acting == OperatorTypeActing.Local, f"Expected Local, got {op.type_acting}"

        # Test operator application
        new_state, amplitude = op(0b100, 1)
        assert new_state[0] == 0b110, f"Expected 0b110, got {new_state[0]:0b}"
        assert amplitude[0] == -1.0, f"Expected -1.0, got {amplitude[0]}"
        print("(ok)  HilbertSpace operator creation and application works")


class TestMomentumSpaceOperators:
    """Test momentum space operators for spins."""

    @pytest.fixture
    def lattice(self):
        """Create a small 1D lattice for testing."""
        return SquareLattice(dim=1, lx=3, ly=1, lz=1, bc=LatticeBC.PBC)

    def test_spin_momentum_operators(self, lattice):
        """Test momentum space spin operators."""
        # Test sigma_k
        k = np.pi  # k=π
        sigma_k_op = op_spin.sig_k(k=k, lattice=lattice, type_act=OperatorTypeActing.Local)

        # Test on numpy state
        state = np.array([1.0, 0.0, 0.0])  # |↑00⟩

        result = sigma_k_op(state, 0)  # site 0 (k is already set in operator)
        assert len(result) == 2, "Should return (states, coeffs) tuple"
        assert isinstance(result[1], np.ndarray) and result[1].dtype in [np.complex64, np.complex128], "Should return complex coefficient array"
        print(f"(ok)  Spin momentum operator at k=π: coefficient = {result[1]}")


class TestBackendCompatibility:
    """Test backend compatibility and type consistency."""

    @pytest.fixture
    def lattice(self):
        """Create a small 1D lattice for testing."""
        return SquareLattice(dim=1, lx=3, ly=1, lz=1, bc=LatticeBC.PBC)

    def test_operator_output_types(self, lattice):
        """Test that operators return consistent output types."""
        sig_x = op_spin.sig_x(lattice=lattice, type_act=OperatorTypeActing.Local)

        # Test int input
        int_result = sig_x(0, 0)
        assert isinstance(int_result[0], np.ndarray), "Int input should return ndarray states"
        assert isinstance(int_result[1], np.ndarray), "Int input should return ndarray coeffs"

        # Test numpy input
        np_state = np.array([1.0, 0.0, 0.0])
        np_result = sig_x(np_state, 0)
        assert isinstance(np_result[0], np.ndarray), "NumPy input should return ndarray states"
        assert isinstance(np_result[1], np.ndarray), "NumPy input should return ndarray coeffs"

        if JAX_AVAILABLE:
            # Test JAX input
            jnp_state = jnp.array([1.0, 0.0, 0.0])
            jnp_result = sig_x(jnp_state, 0)
            assert hasattr(jnp_result[0], 'device'), "JAX input should return JAX arrays"
            assert hasattr(jnp_result[1], 'device'), "JAX input should return JAX arrays"

        print("(ok)  All backends return consistent output types")

    def test_operator_shape_consistency(self, lattice):
        """Test that operator outputs have consistent shapes."""
        c_dag = op_ferm.cdag(lattice=lattice, type_act=OperatorTypeActing.Local)

        # All backends should return (states, coeffs) with consistent shapes
        int_result = c_dag(0, 0)
        assert int_result[0].ndim == 1, "States should be 1D array"
        assert int_result[1].ndim == 1, "Coeffs should be 1D array"

        np_state = np.zeros(3)
        np_result = c_dag(np_state, 0)
        assert np_result[0].shape == (1, 3), "NumPy states should return (1, n_sites)"
        assert np_result[1].shape == (1,), "NumPy coeffs should be scalar array"

        print("(ok)  All backends maintain consistent output shapes")


class TestTranslationSymmetry:
    """Test translation symmetry operators and their properties."""

    @pytest.fixture
    def lattice_1d(self):
        """Create a 1D lattice for testing."""
        from QES.general_python.lattices.square import SquareLattice
        from QES.general_python.lattices.lattice import LatticeBC
        return SquareLattice(dim=1, lx=4, ly=1, lz=1, bc=LatticeBC.PBC)

    @pytest.fixture
    def lattice_2d(self):
        """Create a 2D lattice for testing."""
        from QES.general_python.lattices.square import SquareLattice
        from QES.general_python.lattices.lattice import LatticeBC
        return SquareLattice(dim=2, lx=4, ly=4, lz=1, bc=LatticeBC.PBC)

    def test_translation_real_coefficients_zero_sector(self, lattice_1d, lattice_2d):
        """Test that translation operators return real coefficients at k=0."""
        from QES.Algebra.Symmetries.translation import TranslationSymmetry
        from QES.general_python.lattices.lattice import LatticeDirection

        # Test 1D k=0 sector
        trans_1d = TranslationSymmetry(lattice_1d, sector=0, ns=lattice_1d.Ns, direction='x')
        assert trans_1d.get_momentum_sector().name == 'ZERO'

        # Test several states
        for state in [0b0000, 0b0001, 0b0101, 0b1111]:
            new_state, phase = trans_1d.apply_int(state, lattice_1d.Ns)
            assert np.isreal(phase), f"Phase {phase} is not real for state {state:04b} at k=0"
            assert phase.imag == 0.0, f"Phase {phase} has non-zero imaginary part for state {state:04b} at k=0"
            print(f"(ok)  1D k=0: state {state:04b} → phase {phase}")

        # Test 2D k=(0,0) sector
        trans_2d_x = TranslationSymmetry(lattice_2d, sector=0, ns=lattice_2d.Ns, direction='x')
        trans_2d_y = TranslationSymmetry(lattice_2d, sector=0, ns=lattice_2d.Ns, direction='y')

        # Test combined translation (should still be real)
        for state in [0, 1, 15]:  # Some test states
            # Apply T_x then T_y
            state_x, phase_x = trans_2d_x.apply_int(state, lattice_2d.Ns)
            state_xy, phase_y = trans_2d_y.apply_int(state_x, lattice_2d.Ns)
            total_phase = phase_x * phase_y

            assert np.isreal(total_phase), f"Combined phase {total_phase} is not real for state {state} at k=(0,0)"
            assert total_phase.imag == 0.0, f"Combined phase {total_phase} has non-zero imaginary part for state {state} at k=(0,0)"
            print(f"(ok)  2D k=(0,0): state {state} → combined phase {total_phase}")

    def test_translation_real_coefficients_pi_sector(self, lattice_1d, lattice_2d):
        """Test that translation operators return real coefficients at k=π."""
        from QES.Algebra.Symmetries.translation import TranslationSymmetry

        # Test 1D k=π sector (sector = L/2 = 2 for L=4)
        trans_1d = TranslationSymmetry(lattice_1d, sector=2, ns=lattice_1d.Ns, direction='x')
        assert trans_1d.get_momentum_sector().name == 'PI'

        # Test several states
        for state in [0b0000, 0b0001, 0b0101, 0b1111]:
            new_state, phase = trans_1d.apply_int(state, lattice_1d.Ns)
            assert np.isreal(phase), f"Phase {phase} is not real for state {state:04b} at k=π"
            assert phase.imag == 0.0, f"Phase {phase} has non-zero imaginary part for state {state:04b} at k=π"
            print(f"(ok)  1D k=π: state {state:04b} → phase {phase}")

        # Test 2D k=(π,0) and k=(0,π) sectors
        trans_2d_x_pi = TranslationSymmetry(lattice_2d, sector=2, ns=lattice_2d.Ns, direction='x')  # k_x = π
        trans_2d_y_pi = TranslationSymmetry(lattice_2d, sector=2, ns=lattice_2d.Ns, direction='y')  # k_y = π

        for state in [0, 1, 15]:  # Some test states
            # Test individual directions
            _, phase_x = trans_2d_x_pi.apply_int(state, lattice_2d.Ns)
            _, phase_y = trans_2d_y_pi.apply_int(state, lattice_2d.Ns)

            assert np.isreal(phase_x), f"X-direction phase {phase_x} is not real for state {state} at k=(π,0)"
            assert np.isreal(phase_y), f"Y-direction phase {phase_y} is not real for state {state} at k=(0,π)"
            assert phase_x.imag == 0.0, f"X-direction phase {phase_x} has non-zero imaginary part for state {state} at k=(π,0)"
            assert phase_y.imag == 0.0, f"Y-direction phase {phase_y} has non-zero imaginary part for state {state} at k=(0,π)"

            print(f"(ok)  2D k=(π,0): state {state} → phase {phase_x}")
            print(f"(ok)  2D k=(0,π): state {state} → phase {phase_y}")

    def test_translation_generic_sector_complex(self, lattice_1d):
        """Test that translation operators can return complex coefficients at generic k."""
        from QES.Algebra.Symmetries.translation import TranslationSymmetry

        # Test 1D generic k sector (sector = 1 for L=4, k=π/2)
        trans_1d = TranslationSymmetry(lattice_1d, sector=1, ns=lattice_1d.Ns, direction='x')
        assert trans_1d.get_momentum_sector().name == 'GENERIC'

        # For generic k, phases can be complex
        state = 0b0001
        new_state, phase = trans_1d.apply_int(state, lattice_1d.Ns)

        # Phase can be complex for generic k
        print(f"(ok)  1D k=π/2: state {state:04b} → phase {phase} (can be complex)")

    def test_momentum_sector_detection(self, lattice_1d, lattice_2d):
        """Test that momentum sectors are correctly identified."""
        from QES.Algebra.Symmetries.translation import TranslationSymmetry
        from QES.Algebra.Symmetries.base import MomentumSector

        # 1D lattice L=4
        trans_k0 = TranslationSymmetry(lattice_1d, sector=0, ns=lattice_1d.Ns, direction='x')
        trans_k1 = TranslationSymmetry(lattice_1d, sector=1, ns=lattice_1d.Ns, direction='x')
        trans_k2 = TranslationSymmetry(lattice_1d, sector=2, ns=lattice_1d.Ns, direction='x')

        assert trans_k0.get_momentum_sector() == MomentumSector.ZERO
        assert trans_k1.get_momentum_sector() == MomentumSector.GENERIC
        assert trans_k2.get_momentum_sector() == MomentumSector.PI

        print("(ok)  Momentum sector detection works correctly for 1D")

        # 2D lattice 4x4
        trans_2d_k0x = TranslationSymmetry(lattice_2d, sector=0, ns=lattice_2d.Ns, direction='x')
        trans_2d_k2x = TranslationSymmetry(lattice_2d, sector=2, ns=lattice_2d.Ns, direction='x')
        trans_2d_k0y = TranslationSymmetry(lattice_2d, sector=0, ns=lattice_2d.Ns, direction='y')
        trans_2d_k2y = TranslationSymmetry(lattice_2d, sector=2, ns=lattice_2d.Ns, direction='y')

        assert trans_2d_k0x.get_momentum_sector() == MomentumSector.ZERO
        assert trans_2d_k2x.get_momentum_sector() == MomentumSector.PI
        assert trans_2d_k0y.get_momentum_sector() == MomentumSector.ZERO
        assert trans_2d_k2y.get_momentum_sector() == MomentumSector.PI

        print("(ok)  Momentum sector detection works correctly for 2D")


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short"])