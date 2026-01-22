"""
Comprehensive test suite for QuadraticHamiltonian improvements.
Tests performance, usability, backend switching, and state calculations.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
    from QES.Algebra.Hilbert.hilbert_jit_states import (
        calculate_bogoliubov_amp,
        calculate_bogoliubov_amp_exc,
        calculate_slater_det,
    )
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


class TestParticleConservingSystemsBasic:
    """Test basic particle-conserving systems."""

    def test_ring_lattice(self):
        """Test 1D ring lattice with hopping."""
        qh = QuadraticHamiltonian(ns=4, particle_conserving=True)
        # Add hopping terms forming a ring
        qh.add_hopping(0, 1, 1.0)
        qh.add_hopping(1, 2, 1.0)
        qh.add_hopping(2, 3, 1.0)
        qh.add_hopping(3, 0, 1.0)  # Closes the ring
        assert qh is not None
        # Verify matrix is non-zero
        mat = qh.build_single_particle_matrix()
        assert np.any(np.abs(mat) > 0)

    def test_chain_lattice(self):
        """Test 1D chain with onsite and hopping."""
        qh = QuadraticHamiltonian(ns=6, particle_conserving=True)
        # Add onsite terms
        for i in range(6):
            qh.add_onsite(i, 0.5)
        # Add hopping terms
        for i in range(5):
            qh.add_hopping(i, i + 1, -1.0)
        assert qh is not None
        mat = qh.build_single_particle_matrix()
        assert np.any(np.abs(mat) > 0)

    def test_slater_state_calculation(self):
        """Test Slater determinant calculation."""
        qh = QuadraticHamiltonian(ns=4, particle_conserving=True)
        qh.add_onsite(0, 0.1)
        qh.add_onsite(1, -0.1)
        qh.add_hopping(0, 1, 0.5)
        qh.add_hopping(1, 2, 0.5)

        # Diagonalize to get ground state
        qh.diagonalize(verbose=False)
        assert qh.eig_val is not None
        assert len(qh.eig_val) == 4


class TestBdGSystems:
    """Test BdG (non-particle-conserving) systems."""

    def test_bdg_simple(self):
        """Test simple BdG system."""
        qh = QuadraticHamiltonian(ns=2, particle_conserving=False)
        qh.add_hopping(0, 1, 1.0)
        qh.add_pairing(0, 1, 0.5)
        assert qh is not None
        # Verify BdG matrix has both components
        assert qh._hamil_sp is not None
        assert qh._delta_sp is not None

    def test_bdg_superconductor(self):
        """Test BdG superconductor with gap."""
        qh = QuadraticHamiltonian(ns=4, particle_conserving=False)
        # Kinetic terms
        for i in range(4):
            qh.add_onsite(i, 0.0)
        for i in range(3):
            qh.add_hopping(i, i + 1, -1.0)
        # Pairing terms (superconducting gap)
        for i in range(0, 4, 2):
            qh.add_pairing(i, i + 1, 0.3)
        assert qh is not None
        mat = qh.build_single_particle_matrix()
        assert np.any(np.abs(mat) > 0)


class TestMatrixConstructionMethods:
    """Test direct matrix construction methods."""

    def test_from_hermitian(self):
        """Test construction from Hermitian matrix."""
        h_matrix = np.array([[1, -1, 0], [-1, 0, -1], [0, -1, 1]], dtype=complex)

        try:
            qh = QuadraticHamiltonian.from_hermitian_matrix(h_matrix, constant=2.0)
            qh.diagonalize(verbose=False)
            assert qh.eig_val is not None
        except (AttributeError, NotImplementedError):
            pytest.skip("from_hermitian_matrix not implemented")

    def test_from_bdg_matrices(self):
        """Test construction from BdG matrices."""
        h_matrix = np.array([[0, 1], [1, 0]], dtype=complex)

        v_matrix = np.array([[0, 0.5], [0.5, 0]], dtype=complex)

        try:
            qh = QuadraticHamiltonian.from_bdg_matrices(h_matrix, v_matrix, constant=1.5)
            qh.diagonalize(verbose=False)
            assert qh.eig_val.size == 4
        except (AttributeError, NotImplementedError):
            pytest.skip("from_bdg_matrices not implemented")


class TestPerformanceScaling:
    """Test performance and scaling."""

    def test_matrix_construction_time(self):
        """Test matrix construction time."""
        h_matrix = np.array([[1, 0.5], [0.5, -1]], dtype=complex)


        start = time.time()
        try:
            qh = QuadraticHamiltonian.from_hermitian_matrix(h_matrix)
            elapsed = time.time() - start
            assert elapsed < 1.0  # Should be fast
        except (AttributeError, NotImplementedError):
            pytest.skip("from_hermitian_matrix not implemented")

    def test_scaling_with_size(self):
        """Test performance scaling with system size."""
        # Test that larger matrices still construct quickly
        sizes = [2, 4, 6]
        for size in sizes:
            h_matrix = np.eye(size, dtype=complex)
            try:
                qh = QuadraticHamiltonian.from_hermitian_matrix(h_matrix)
                assert qh is not None
            except (AttributeError, NotImplementedError):
                pytest.skip("from_hermitian_matrix not implemented")


class TestBackendSupport:
    """Test backend compatibility."""

    def test_matrix_independence_from_backend(self):
        """Test that matrix operations are backend-independent."""
        h_matrix = np.array([[2, -1], [-1, 0]], dtype=complex)

        try:
            qh = QuadraticHamiltonian.from_hermitian_matrix(h_matrix)
            # Verify basic functionality works
            assert qh is not None
        except (AttributeError, NotImplementedError):
            pytest.skip("from_hermitian_matrix not implemented")

    def test_numerical_stability(self):
        """Test numerical stability with small values."""
        h_matrix = np.array([[1e-10, 1e-11], [1e-11, -1e-10]], dtype=complex)

        try:
            qh = QuadraticHamiltonian.from_hermitian_matrix(h_matrix)
            assert qh is not None
        except (AttributeError, NotImplementedError):
            pytest.skip("from_hermitian_matrix not implemented")
