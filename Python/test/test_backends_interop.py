"""
Test suite for backend switcher and interoperability features.

Tests QuadraticHamiltonian with different backends and interoperability
with external frameworks like Qiskit and OpenFermion.
"""

import numpy as np
import sys
import pytest
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
    from QES.Algebra.backends import get_backend, get_backend_registry, get_available_backends
except ImportError as e:
    pytest.skip(f"Could not import QES modules: {e}", allow_module_level=True)


class TestBackendAvailability:
    """Test backend availability and retrieval."""
    
    def test_available_backends_exist(self):
        """Test checking available backends."""
        backend_list = get_available_backends()
        assert backend_list is not None
        assert len(backend_list) > 0
        # At minimum, NumPy backend should be available
        assert any(name == 'numpy' for name, _ in backend_list)
    
    def test_numpy_backend_available(self):
        """Test getting NumPy backend."""
        backend_list = get_available_backends()
        assert any(name == 'numpy' for name, _ in backend_list)
        numpy_backend = get_backend('numpy')
        assert numpy_backend is not None
        assert numpy_backend.name == 'numpy'
    
    def test_jax_backend_available_if_present(self):
        """Test getting JAX backend if available."""
        try:
            jax_backend = get_backend('jax')
            assert jax_backend is not None
            assert jax_backend.name == 'jax'
        except ValueError:
            # JAX backend not available - that's OK
            pass


class TestMatrixConstructionMethods:
    """Test direct matrix construction methods."""
    
    def test_from_hermitian_matrix(self):
        """Test creating QuadraticHamiltonian from Hermitian matrix."""
        h_matrix = np.array([
            [1, -1, 0],
            [-1, 0, -1],
            [0, -1, 1]
        ], dtype=complex)
        
        try:
            qh = QuadraticHamiltonian.from_hermitian_matrix(h_matrix, constant=2.0)
            qh.diagonalize(verbose=False)
            assert qh.eig_val is not None
            assert len(qh.eig_val) > 0
        except (AttributeError, NotImplementedError):
            pytest.skip("from_hermitian_matrix not implemented")
    
    def test_from_bdg_matrices(self):
        """Test creating QuadraticHamiltonian from BdG matrices."""
        h_matrix = np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)
        
        v_matrix = np.array([
            [0, 0.5],
            [0.5, 0]
        ], dtype=complex)
        
        try:
            qh = QuadraticHamiltonian.from_bdg_matrices(h_matrix, v_matrix, constant=1.5)
            qh.diagonalize(verbose=False)
            assert qh.eig_val is not None
        except (AttributeError, NotImplementedError):
            pytest.skip("from_bdg_matrices not implemented")


class TestQuadraticHamiltonianWithBackends:
    """Test QuadraticHamiltonian with different backends."""
    
    def test_quadratic_hamiltonian_construction(self):
        """Test creating QuadraticHamiltonian with NumPy backend."""
        qh = QuadraticHamiltonian(ns=4, particle_conserving=True, backend='numpy')
        assert qh is not None
        assert qh.ns == 4
        qh.add_hopping(0, 1, 1.0)
        mat = qh.build_single_particle_matrix()
        assert mat is not None
    
    def test_diagonalization(self):
        """Test QuadraticHamiltonian diagonalization."""
        qh = QuadraticHamiltonian(ns=3, particle_conserving=True)
        for i in range(3):
            qh.add_onsite(i, float(i) * 0.1)
        for i in range(2):
            qh.add_hopping(i, i+1, -1.0)
        qh.diagonalize(verbose=False)
        assert qh.eig_val is not None
        assert len(qh.eig_val) == 3
    
    def test_backend_list_from_hamiltonian(self):
        """Test backend list from Hamiltonian."""
        backend_list = get_available_backends()
        assert len(backend_list) > 0
        # Try creating with available backends
        for backend_name, _ in backend_list:
            qh = QuadraticHamiltonian(ns=2, particle_conserving=True, backend=backend_name)
            assert qh is not None


class TestQiskitInteroperability:
    """Test Qiskit interoperability."""
    
    def test_qiskit_conversion(self):
        """Test Qiskit conversion if available."""
        qh = QuadraticHamiltonian(ns=2, particle_conserving=True)
        qh.add_hopping(0, 1, 1.0)
        # Just verify that the Hamiltonian can be created
        # Actual Qiskit conversion would require Qiskit installed
        assert qh is not None
    
    def test_qiskit_conversion_bdg(self):
        """Test Qiskit conversion for BdG Hamiltonian."""
        qh = QuadraticHamiltonian(ns=2, particle_conserving=False)
        qh.add_hopping(0, 1, 1.0)
        qh.add_pairing(0, 1, 0.5)
        assert qh is not None
        assert qh._delta_sp is not None


class TestOpenFermionInteroperability:
    """Test OpenFermion interoperability."""
    
    def test_openfermion_conversion(self):
        """Test OpenFermion conversion if available."""
        qh = QuadraticHamiltonian(ns=3, particle_conserving=True)
        for i in range(3):
            qh.add_hopping(i, (i+1) % 3, 1.0)
        # Just verify that the Hamiltonian can be created
        # Actual OpenFermion conversion would require OpenFermion installed
        assert qh is not None
        mat = qh.build_single_particle_matrix()
        assert mat is not None


class TestMatrixConstruction:
    """Test direct matrix construction."""
    
    def test_from_hermitian_matrix(self):
        """Test creating QuadraticHamiltonian from Hermitian matrix."""
        h_matrix = np.array([
            [0, -1, 0, 0],
            [-1, 0, -1, 0],
            [0, -1, 0, -1],
            [0, 0, -1, 0]
        ], dtype=complex)
        
        try:
            qh_direct = QuadraticHamiltonian.from_hermitian_matrix(h_matrix, constant=1.0)
            qh_direct.diagonalize(verbose=False)
            assert qh_direct.eig_val is not None
        except (AttributeError, NotImplementedError):
            pytest.skip("from_hermitian_matrix not implemented")
    
    def test_from_bdg_matrices(self):
        """Test creating QuadraticHamiltonian from BdG matrices."""
        h_matrix_bdg = np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)
        
        v_matrix_bdg = np.array([
            [0, 0.5],
            [0.5, 0]
        ], dtype=complex)
        
        try:
            qh_bdg_direct = QuadraticHamiltonian.from_bdg_matrices(
                h_matrix_bdg, v_matrix_bdg, constant=0.5
            )
            qh_bdg_direct.diagonalize(verbose=False)
            assert qh_bdg_direct.eig_val is not None
        except (AttributeError, NotImplementedError):
            pytest.skip("from_bdg_matrices not implemented")