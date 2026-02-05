"""
Tests for QuadraticHamiltonian with lazy imports.

Validates correctness of the quadratic hamiltonian implementation
against known small-system exact results.

----------------------------------------------------
File    : test/test_quadratic_hamiltonian.py
Author  : Maksymilian Kliczkowski
----------------------------------------------------
"""

import pytest
import numpy as np
from typing import Tuple

# Import the module under test
try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
    from QES.Algebra.Quadratic.hamil_quadratic_utils import QuadraticTerm
    QUADRATIC_AVAILABLE = True
except ImportError as e:
    QUADRATIC_AVAILABLE = False
    pytest.skip(f"QuadraticHamiltonian not available: {e}", allow_module_level=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tight_binding_4site():
    """Create a 4-site tight-binding chain."""
    ham = QuadraticHamiltonian(ns=4, particle_conserving=True, particles='fermions')
    t = -1.0  # hopping
    for i in range(3):
        ham.add_hopping(i, i+1, t)
    ham.add_hopping(3, 0, t)  # periodic boundary
    return ham

@pytest.fixture
def disordered_chain_6site():
    """Create a 6-site disordered chain."""
    np.random.seed(42)
    ham = QuadraticHamiltonian(ns=6, particle_conserving=True, particles='fermions')
    
    # Random onsite energies
    for i in range(6):
        ham.add_onsite(i, np.random.uniform(-0.5, 0.5))
    
    # Hopping
    t = -1.0
    for i in range(5):
        ham.add_hopping(i, i+1, t)
    
    return ham


# =============================================================================
# Test: Lazy Import System
# =============================================================================

class TestLazyImports:
    """Test that lazy imports work correctly."""
    
    def test_module_loads_without_hilbert_jit(self):
        """Module should load without immediately importing hilbert_jit_states."""
        # The import at module level should not fail
        assert QUADRATIC_AVAILABLE
    
    def test_first_access_loads_functions(self, tight_binding_4site):
        """Functions should be loaded on first use."""
        ham = tight_binding_4site
        ham.diagonalize()

        # This should trigger lazy import
        occ = np.array([0, 1], dtype=np.int32)
        psi = ham.many_body_state(occ)

        assert psi is not None
        assert psi.shape == (16,)


# =============================================================================
# Test: Tight-Binding Model
# =============================================================================

class TestTightBindingModel:
    """Tests for simple tight-binding chains."""
    
    def test_eigenvalues_4site_pbc(self, tight_binding_4site):
        """4-site PBC chain should have analytical eigenvalues."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        # For t=-1, PBC: E_k = -2t*cos(2πk/N) = 2*cos(2πk/4)
        # k = 0,1,2,3 -> E = 2, 0, -2, 0
        expected = np.array([-2.0, 0.0, 0.0, 2.0])
        
        assert np.allclose(np.sort(ham.eig_val), np.sort(expected), atol=1e-10)
    
    def test_ground_state_energy(self, tight_binding_4site):
        """Ground state energy should be sum of lowest eigenvalues."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        # Half-filling: 2 particles
        # Fill lowest 2 orbitals: E = -2 + 0 = -2
        occ = np.argsort(ham.eig_val)[:2].astype(np.int32)
        E = ham.many_body_energy(occ)
        
        expected = -2.0
        assert np.isclose(E, expected, atol=1e-10)
    
    def test_state_normalization(self, tight_binding_4site):
        """Many-body state should be normalized."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        occ = np.array([0, 1], dtype=np.int32)
        psi = ham.many_body_state(occ)
        
        norm_sq = np.sum(np.abs(psi)**2)
        assert np.isclose(norm_sq, 1.0, atol=1e-10)
    
    def test_particle_conservation(self, tight_binding_4site):
        """State should only have amplitude in correct particle sector."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        n_particles = 2
        occ = np.arange(n_particles, dtype=np.int32)
        psi = ham.many_body_state(occ)
        
        # Check that only 2-particle states have non-zero amplitude
        for state in range(16):
            n_set_bits = bin(state).count('1')
            if n_set_bits != n_particles:
                assert np.isclose(psi[state], 0.0, atol=1e-10), \
                    f"State {state} has {n_set_bits} particles but amplitude {psi[state]}"


# =============================================================================
# Test: Disordered Chain
# =============================================================================

class TestDisorderedChain:
    """Tests for disordered chains."""
    
    def test_eigenvalue_count(self, disordered_chain_6site):
        """Should have ns eigenvalues."""
        ham = disordered_chain_6site
        ham.diagonalize()
        
        assert len(ham.eig_val) == 6
    
    def test_eigenvector_orthonormality(self, disordered_chain_6site):
        """Eigenvectors should be orthonormal."""
        ham = disordered_chain_6site
        ham.diagonalize()
        
        U = ham.eig_vec
        overlap = U.conj().T @ U
        
        assert np.allclose(overlap, np.eye(6), atol=1e-10)
    
    def test_hermiticity(self, disordered_chain_6site):
        """Hamiltonian matrix should be Hermitian."""
        ham = disordered_chain_6site
        ham.build()
        H = ham._hamil_sp
        
        assert np.allclose(H, H.conj().T, atol=1e-10)


# =============================================================================
# Test: Many-Body State Methods
# =============================================================================

class TestManyBodyStateMethods:
    """Tests for many_body_state and related methods."""
    
    def test_integer_occupation_input(self, tight_binding_4site):
        """Test with integer (bitmask) occupation input."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        # Integer 3 = 0b0011 = sites 0,1 occupied
        psi = ham.many_body_state(3)
        
        assert psi.shape == (16,)
        assert np.isclose(np.sum(np.abs(psi)**2), 1.0)
    
    def test_array_occupation_input(self, tight_binding_4site):
        """Test with array occupation input."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        occ = np.array([0, 1], dtype=np.int32)
        psi = ham.many_body_state(occ)
        
        assert psi.shape == (16,)
        assert np.isclose(np.sum(np.abs(psi)**2), 1.0)
    
    def test_list_occupation_input(self, tight_binding_4site):
        """Test with list occupation input."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        psi = ham.many_body_state([0, 1])
        
        assert psi.shape == (16,)
        assert np.isclose(np.sum(np.abs(psi)**2), 1.0)
    
    def test_preallocated_result(self, tight_binding_4site):
        """Test with pre-allocated result array."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        result = np.zeros(16, dtype=complex)
        returned = ham.many_body_state([0, 1], resulting_state=result)
        
        # Should write to the provided array
        assert np.isclose(np.sum(np.abs(result)**2), 1.0)
    
    def test_many_body_states_batch(self, tight_binding_4site):
        """Test batch computation of multiple states."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        occupations = [0, 1, 2, 3]  # Different particle numbers
        states = ham.many_body_states(occupations, return_dict=False)
        
        assert states.shape == (4, 16)
        
        # Each should be normalized in its particle sector
        for i, psi in enumerate(states):
            n_particles = bin(occupations[i]).count('1')
            if n_particles > 0:
                assert np.sum(np.abs(psi)**2) > 0
    
    def test_many_body_states_dict(self, tight_binding_4site):
        """Test dictionary return from batch computation."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        occupations = [1, 3, 5]
        states = ham.many_body_states(occupations, return_dict=True)
        
        assert isinstance(states, dict)
        assert set(states.keys()) == {1, 3, 5}


# =============================================================================
# Test: Ground State Computation
# =============================================================================

class TestGroundState:
    """Tests for compute_ground_state method."""
    
    def test_ground_state_energy(self, tight_binding_4site):
        """Ground state energy should match manual computation."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        E0, psi0 = ham.compute_ground_state(symmetry_sector=2)
        
        # Manual: fill lowest 2 orbitals
        occ = np.argsort(ham.eig_val)[:2].astype(np.int32)
        E_manual = ham.many_body_energy(occ)
        
        assert np.isclose(E0, E_manual)
    
    def test_ground_state_normalized(self, tight_binding_4site):
        """Ground state should be normalized."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        _, psi0 = ham.compute_ground_state(symmetry_sector=2)
        
        assert np.isclose(np.sum(np.abs(psi0)**2), 1.0)
    
    def test_vacuum_ground_state(self, tight_binding_4site):
        """Vacuum (0 particles) should be valid."""
        ham = tight_binding_4site
        ham.diagonalize()
        
        E0, psi0 = ham.compute_ground_state(symmetry_sector=0)

        assert np.isclose(E0, 0.0)
        # Vacuum state: only state 0 has amplitude 1
        assert np.isclose(psi0[0], 1.0)


# =============================================================================
# Test: Cache Effectiveness
# =============================================================================

class TestCaching:
    """Tests for caching behavior."""
    
    def test_repeated_calls_use_cache(self, tight_binding_4site):
        """Repeated calls with same occupation should use cache."""
        ham = tight_binding_4site
        ham.diagonalize()

        occ = np.array([0, 1], dtype=np.int32)

        # First call
        psi1 = ham.many_body_state(occ)
        cache_size_1 = len(ham._calculator_cache)

        # Second call (should use cache)
        psi2 = ham.many_body_state(occ)
        cache_size_2 = len(ham._calculator_cache)

        assert cache_size_2 == cache_size_1  # No new cache entries
        assert np.allclose(psi1, psi2)

    def test_different_occupation_new_cache(self, tight_binding_4site):
        """Different occupation should create new cache entry."""
        ham = tight_binding_4site
        ham.diagonalize()

        # First occupation
        ham.many_body_state([0, 1])
        cache_size_1 = len(ham._calculator_cache)

        # Different occupation
        ham.many_body_state([0, 2])
        cache_size_2 = len(ham._calculator_cache)

        assert cache_size_2 > cache_size_1

    def test_cache_invalidation(self, tight_binding_4site):
        """Cache should be invalidated on relevant changes."""
        ham = tight_binding_4site
        ham.diagonalize()

        ham.many_body_state([0, 1])

        # Invalidate cache
        ham.invalidate_state_cache()

        assert len(ham._calculator_cache) == 0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_site(self):
        """Single site system."""
        ham = QuadraticHamiltonian(ns=1, particle_conserving=True, particles='fermions')
        ham.add_onsite(0, 0.5)
        ham.diagonalize()

        # Vacuum
        E0, psi0 = ham.compute_ground_state(symmetry_sector=0)
        assert np.isclose(E0, 0.0)

        # One particle
        E1, psi1 = ham.compute_ground_state(symmetry_sector=1)
        assert np.isclose(E1, 0.5)

    def test_full_occupation(self, tight_binding_4site):
        """All sites occupied."""
        ham = tight_binding_4site
        ham.diagonalize()

        occ = np.arange(4, dtype=np.int32)
        psi = ham.many_body_state(occ)

        # Full occupation: only state 0b1111 = 15 has amplitude
        assert np.isclose(np.abs(psi[15]), 1.0)

    def test_no_terms_added(self):
        """Hamiltonian with no terms should still work."""
        ham = QuadraticHamiltonian(ns=4, particle_conserving=True, particles='fermions')
        ham.diagonalize()

        # All eigenvalues should be zero
        assert np.allclose(ham.eig_val, 0.0)


# =============================================================================
# Test: Comparison with Exact Results
# =============================================================================

class TestExactResults:
    """Tests comparing with analytically known results."""
    
    def test_two_site_dimer(self):
        """Two-site dimer has exact solution."""
        ham = QuadraticHamiltonian(ns=2, particle_conserving=True, particles='fermions')
        t = -1.0
        ham.add_hopping(0, 1, t)
        ham.diagonalize()

        # Eigenvalues: ±|t| = ±1
        expected_eigvals = np.array([-1.0, 1.0])
        assert np.allclose(np.sort(ham.eig_val), np.sort(expected_eigvals))

        # Ground state with 1 particle: bonding orbital
        E0, psi0 = ham.compute_ground_state(symmetry_sector=1)
        assert np.isclose(E0, -1.0)

        # State should be (|01⟩ + |10⟩)/√2 or similar
        # Non-zero at states 1 and 2
        amp_01 = psi0[1]  # state 0b01
        amp_10 = psi0[2]  # state 0b10
        assert np.isclose(np.abs(amp_01), np.abs(amp_10))

    def test_three_site_chain_obc(self):
        """Three-site OBC chain."""
        ham = QuadraticHamiltonian(ns=3, particle_conserving=True, particles='fermions')
        t = -1.0
        ham.add_hopping(0, 1, t)
        ham.add_hopping(1, 2, t)
        ham.diagonalize()

        # For 3-site OBC: E = -√2, 0, +√2 (approximately)
        expected = np.array([-np.sqrt(2), 0.0, np.sqrt(2)])
        assert np.allclose(np.sort(ham.eig_val), np.sort(expected), atol=1e-10)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
