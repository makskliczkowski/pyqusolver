"""
Comprehensive symmetry tests for the Kitaev-Heisenberg model on honeycomb lattice.

Tests validate that symmetry-reduced Hamiltonians correctly reproduce the full spectrum
when all sectors are combined.

File: test/test_kitaev_symmetries.py
"""

import pytest
import numpy as np
from math import comb

from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.Algebra.globals import get_u1_sym


class TestKitaevSymmetries:
    """
    Test suite for Kitaev model symmetries on honeycomb lattice.
    
    The Kitaev model has the following symmetries:
    - Translation (for uniform couplings with PBC)
    - U(1) particle conservation (when hx=hz=0)
    - Time-reversal (complex conjugation)
    - Inversion symmetry
    """
    
    @pytest.fixture
    def small_honeycomb(self):
        """Small honeycomb lattice for testing (2x3 unit cells = 12 sites)."""
        return HoneycombLattice(dim=2, lx=2, ly=3, bc='pbc')
    
    # --------------------------------------------------------------------------------------
    # Translation Symmetry Tests
    # --------------------------------------------------------------------------------------
    
    def test_kitaev_translation_all_k_sectors(self, small_honeycomb):
        """
        Test Kitaev model with translation symmetry across all momentum sectors.
        Verify spectrum reconstruction from all k-sectors.
        """
        L = small_honeycomb.ns
        
        # Full spectrum (no symmetry)
        h_full = HilbertSpace(lattice=small_honeycomb)
        kitaev_full = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_full,
            K=(1.0, 1.0, 1.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True
        )
        E_full = np.linalg.eigvalsh(kitaev_full.matrix.toarray())
        E_full_sorted = np.sort(E_full)
        
        print(f"\nFull Hilbert space: {h_full.Nh} states")
        
        # Collect eigenvalues from all k-sectors
        E_all_k = []
        total_states = 0
        
        # For honeycomb, translation is more complex due to two sublattices
        # We test momentum conservation in the x-direction
        for kx in range(small_honeycomb.lx):
            h_k = HilbertSpace(
                lattice=small_honeycomb,
                sym_gen={'translation': kx}
            )
            
            if h_k.Nh == 0:
                continue
                
            kitaev_k = HeisenbergKitaev(
                lattice=small_honeycomb,
                hilbert_space=h_k,
                K=(1.0, 1.0, 1.0),
                J=None,
                Gamma=None,
                hx=None,
                hz=None,
                dtype=np.float64,
                use_forward=True
            )
            
            H_k = kitaev_k.matrix.toarray()
            E_k = np.real(np.linalg.eigvals(H_k))  # Use eigvals, take real part
            E_all_k.extend(E_k)
            total_states += len(E_k)
            
            print(f"kx={kx}: dim={h_k.Nh}, n_eigs={len(E_k)}, "
                  f"GS energy={E_k[0]:.6f}")
        
        # Sort and compare
        E_all_k_sorted = np.sort(E_all_k)
        
        max_error = np.max(np.abs(E_all_k_sorted - E_full_sorted))
        print(f"\nTranslation spectrum reconstruction max error: {max_error:.2e}")
        print(f"Hilbert space: {len(E_full_sorted)} -> {total_states} states collected")
        print(f"Reduction factor: ~{len(E_full_sorted) / total_states:.1f}×")
        
        assert max_error < 1e-12, f"Translation symmetry validation failed: {max_error:.2e}"
    
    def test_kitaev_translation_k0_sector(self, small_honeycomb):
        """
        Test Kitaev model in k=0 momentum sector only.
        Note: Symmetry-reduced Hamiltonians may not be Hermitian in the reduced basis
        due to representation normalization, but eigenvalues should still be real.
        """
        h_k0 = HilbertSpace(
            lattice=small_honeycomb,
            sym_gen={'translation': 0}
        )
        
        kitaev_k0 = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_k0,
            K=(1.0, 1.0, 1.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True
        )
        
        H_k0 = kitaev_k0.matrix.toarray()
        
        # Eigenvalues should be real even if representation is not Hermitian
        E_k0 = np.linalg.eigvals(H_k0)
        imag_part = np.max(np.abs(np.imag(E_k0)))
        print(f"\nk=0 sector: {h_k0.Nh} states, GS energy = {np.real(E_k0[np.argmin(np.real(E_k0))]):.6f}")
        print(f"Max imaginary part of eigenvalues: {imag_part:.2e}")
        print(f"Energy range: [{np.min(np.real(E_k0)):.4f}, {np.max(np.real(E_k0)):.4f}]")
        
        assert imag_part < 1e-10, f"Eigenvalues should be real, max imag = {imag_part:.2e}"
        assert h_k0.Nh > 0, "k=0 sector should not be empty"
    
    # --------------------------------------------------------------------------------------
    # U(1) Symmetry Tests
    # --------------------------------------------------------------------------------------
    
    def test_kitaev_u1_particle_conservation(self, small_honeycomb):
        """
        Test Kitaev model with U(1) particle number conservation.
        Valid when hx=hz=0 (no fields breaking particle number).
        """
        L = small_honeycomb.ns
        
        # Full spectrum
        h_full = HilbertSpace(lattice=small_honeycomb)
        kitaev_full = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_full,
            K=(1.0, 1.0, 1.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True
        )
        E_full = np.linalg.eigvalsh(kitaev_full.matrix.toarray())
        E_full_sorted = np.sort(E_full)
        
        # Collect eigenvalues from all particle number sectors
        E_all_N = []
        for N in range(L + 1):
            u1_sym = get_u1_sym(lat=small_honeycomb, val=N)
            h_N = HilbertSpace(lattice=small_honeycomb, global_syms=[u1_sym])
            
            # Expected dimension: binomial(L, N)
            expected_dim = comb(L, N)
            assert h_N.Nh == expected_dim, \
                f"N={N}: dimension {h_N.Nh} != expected {expected_dim}"
            
            kitaev_N = HeisenbergKitaev(
                lattice=small_honeycomb,
                hilbert_space=h_N,
                K=(1.0, 1.0, 1.0),
                J=None,
                Gamma=None,
                hx=None,
                hz=None,
                dtype=np.float64,
                use_forward=True
            )
            
            H_N = kitaev_N.matrix.toarray()
            E_N = np.linalg.eigvalsh(H_N)  # U(1) preserves Hermiticity
            E_all_N.extend(E_N)
            
            print(f"N={N}: dim={h_N.Nh} (expected {expected_dim}), "
                  f"GS energy={E_N[0] if len(E_N) > 0 else 'N/A':.6f}")
        
        # Sort and compare
        E_all_N_sorted = np.sort(E_all_N)
        
        max_error = np.max(np.abs(E_all_N_sorted - E_full_sorted))
        print(f"\nU(1) spectrum reconstruction max error: {max_error:.2e}")
        assert max_error < 1e-12, f"U(1) spectrum mismatch: {max_error:.2e}"
    
    # --------------------------------------------------------------------------------------
    # Combined Symmetries Tests
    # --------------------------------------------------------------------------------------
    
    def test_kitaev_translation_plus_u1(self, small_honeycomb):
        """
        Test Kitaev model with combined translation and U(1) symmetries.
        Each (k, N) sector is independently diagonalizable.
        """
        L = small_honeycomb.ns
        
        # Full spectrum
        h_full = HilbertSpace(lattice=small_honeycomb)
        kitaev_full = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_full,
            K=(1.0, 1.0, 1.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True
        )
        E_full = np.linalg.eigvalsh(kitaev_full.matrix.toarray())
        E_full_sorted = np.sort(E_full)
        
        # Collect from all (k, N) sectors
        E_all_sectors = []
        for N in range(L + 1):
            u1_sym = get_u1_sym(lat=small_honeycomb, val=N)
            for kx in range(small_honeycomb.lx):
                h_kN = HilbertSpace(
                    lattice=small_honeycomb,
                    sym_gen={'translation': kx},
                    global_syms=[u1_sym]
                )
                
                if h_kN.Nh == 0:
                    continue
                
                kitaev_kN = HeisenbergKitaev(
                    lattice=small_honeycomb,
                    hilbert_space=h_kN,
                    K=(1.0, 1.0, 1.0),
                    J=None,
                    Gamma=None,
                    hx=None,
                    hz=None,
                    dtype=np.float64,
                    use_forward=True
                )
                
                H_kN = kitaev_kN.matrix.toarray()
                E_kN = np.real(np.linalg.eigvals(H_kN))  # Combined symmetries, take real part
                E_all_sectors.extend(E_kN)
                
                if h_kN.Nh > 0:
                    print(f"N={N}, kx={kx}: dim={h_kN.Nh}")
        
        E_all_sorted = np.sort(E_all_sectors)
        
        max_error = np.max(np.abs(E_all_sorted - E_full_sorted))
        print(f"\nTranslation+U(1) spectrum reconstruction max error: {max_error:.2e}")
        print(f"Total sectors collected: {len(E_all_sorted)}")
        assert max_error < 1e-12, f"Translation+U(1) mismatch: {max_error:.2e}"
    
    # --------------------------------------------------------------------------------------
    # Special Cases Tests
    # --------------------------------------------------------------------------------------
    
    def test_kitaev_isotropic(self, small_honeycomb):
        """
        Test isotropic Kitaev limit: Kx = Ky = Kz.
        """
        h_full = HilbertSpace(lattice=small_honeycomb)
        
        kitaev = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_full,
            K=(1.0, 1.0, 1.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True
        )
        
        H = kitaev.matrix.toarray()
        
        # Check Hermiticity
        assert np.allclose(H, H.T.conj()), "Isotropic Kitaev Hamiltonian not Hermitian"
        
        E = np.linalg.eigvalsh(H)
        print(f"\nIsotropic Kitaev (Kx=Ky=Kz=1.0):")
        print(f"  Hilbert dimension: {h_full.Nh}")
        print(f"  GS energy: {E[0]:.6f}")
        print(f"  Energy gap: {E[1] - E[0]:.6f}")
        print(f"  Energy range: [{E[0]:.4f}, {E[-1]:.4f}]")
    
    def test_kitaev_ising_x(self, small_honeycomb):
        """
        Test Ising-X limit: Kx ≠ 0, Ky = Kz = 0.
        """
        h_full = HilbertSpace(lattice=small_honeycomb)
        
        kitaev = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_full,
            K=(1.0, 0.0, 0.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True
        )
        
        H = kitaev.matrix.toarray()
        assert np.allclose(H, H.T.conj()), "Ising-X Hamiltonian not Hermitian"
        
        E = np.linalg.eigvalsh(H)
        print(f"\nIsing-X limit (Kx=1.0, Ky=Kz=0.0):")
        print(f"  GS energy: {E[0]:.6f}")
        print(f"  First excited: {E[1]:.6f}")
    
    def test_kitaev_with_heisenberg(self, small_honeycomb):
        """
        Test Kitaev-Heisenberg model with both couplings.
        """
        h_full = HilbertSpace(lattice=small_honeycomb)
        
        kitaev_heis = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_full,
            K=(1.0, 1.0, 1.0),
            J=0.5,
            dlt=1.0,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True
        )
        
        H = kitaev_heis.matrix.toarray()
        assert np.allclose(H, H.T.conj()), "Kitaev-Heisenberg Hamiltonian not Hermitian"
        
        E = np.linalg.eigvalsh(H)
        print(f"\nKitaev-Heisenberg (K=1.0, J=0.5):")
        print(f"  GS energy: {E[0]:.6f}")
        print(f"  Energy gap: {E[1] - E[0]:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
