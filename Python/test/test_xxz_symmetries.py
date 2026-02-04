"""
Comprehensive test suite for XXZ model with all symmetries in 1D.

Tests translation, parity, U(1), and combined symmetries.
Validates spectrum reconstruction from symmetry sectors.

Author: Maksymilian Kliczkowski
Date: 2025-11-13
"""

import numpy as np
import pytest

from QES.Algebra.globals import get_u1_sym
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.xxz import XXZ
from QES.general_python.lattices import SquareLattice

##########################################################################################
#! TEST XXZ MODEL WITH SYMMETRIES
##########################################################################################


class TestXXZSymmetries:
    """Test suite for XXZ model with various symmetries in 1D."""

    @pytest.fixture
    def small_chain(self):
        """Create a small 1D chain for testing."""
        return SquareLattice(dim=1, lx=6, bc="pbc")

    @pytest.fixture
    def medium_chain(self):
        """Create a medium 1D chain for more comprehensive tests."""
        return SquareLattice(dim=1, lx=8, bc="pbc")

    # --------------------------------------------------------------------------------------
    # Translation Symmetry Tests
    # --------------------------------------------------------------------------------------

    def test_xxz_translation_all_k_sectors(self, small_chain):
        """
        Test XXZ model with translation symmetry across all k-sectors.
        Validates that concatenating all momentum sectors reproduces the full spectrum.

        This is the key test from the paper - verifying translation symmetry works correctly.
        """
        L = small_chain.lx

        # Full spectrum (no symmetry)
        h_full = HilbertSpace(lattice=small_chain)
        xxz_full = XXZ(lattice=small_chain, hilbert_space=h_full, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)
        E_full = np.linalg.eigvalsh(xxz_full.matrix.toarray())
        E_full_sorted = np.sort(E_full)

        # Collect eigenvalues from all k-sectors
        E_all_sectors = []
        for k in range(L):
            h_k = HilbertSpace(lattice=small_chain, sym_gen={"translation": k})
            xxz_k = XXZ(lattice=small_chain, hilbert_space=h_k, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)

            H_k = xxz_k.matrix.toarray()
            E_k = np.linalg.eigvalsh(H_k)
            E_all_sectors.extend(E_k)

            print(f"k={k}: Hilbert space dim = {h_k.Ns}, eigenvalues = {len(E_k)}")

        # Sort and compare
        E_all_sorted = np.sort(E_all_sectors)

        # Verify dimensions match
        assert len(E_all_sorted) == len(
            E_full_sorted
        ), f"Dimension mismatch: {len(E_all_sorted)} vs {len(E_full_sorted)}"

        # Verify eigenvalues match to machine precision
        max_error = np.max(np.abs(E_all_sorted - E_full_sorted))
        print("\nTranslation symmetry spectrum reconstruction:")
        print(f"Max error: {max_error:.2e}")
        print(f"Full spectrum size: {len(E_full_sorted)}")
        print(f"Sectors spectrum size: {len(E_all_sorted)}")

        assert max_error < 1e-12, f"Spectrum mismatch: max error = {max_error:.2e}"

    def test_xxz_translation_k0_sector(self, small_chain):
        """Test XXZ with translation symmetry at k=0 (Gamma point)."""
        h_k0 = HilbertSpace(lattice=small_chain, sym_gen={"translation": 0})
        xxz = XXZ(lattice=small_chain, hilbert_space=h_k0, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)

        H = xxz.matrix().toarray()

        # Check Hermiticity
        assert np.allclose(H, H.conj().T), "Hamiltonian not Hermitian at k=0"

        # Check eigenvalues are real
        E = np.linalg.eigvalsh(H)
        assert np.all(np.isreal(E)), "Eigenvalues not real at k=0"

        print(f"k=0 sector: dim={h_k0.Ns}, min E={E[0]:.6f}, max E={E[-1]:.6f}")

    # --------------------------------------------------------------------------------------
    # Parity Symmetry Tests
    # --------------------------------------------------------------------------------------

    def test_xxz_parity_z_even_odd(self, small_chain):
        """
        Test XXZ with ParityZ symmetry (spin-flip).
        Only valid when hx=0 (no transverse field).
        """
        # Even parity sector
        h_parity_even = HilbertSpace(lattice=small_chain, sym_gen={"parity": 1})
        xxz_even = XXZ(
            lattice=small_chain, hilbert_space=h_parity_even, jxy=1.0, jz=0.5, hx=0.0, hz=0.0
        )  # hx=0 crucial!

        H_even = xxz_even.matrix().toarray()
        E_even = np.linalg.eigvalsh(H_even)

        # Odd parity sector
        h_parity_odd = HilbertSpace(lattice=small_chain, sym_gen={"parity": -1})
        xxz_odd = XXZ(
            lattice=small_chain, hilbert_space=h_parity_odd, jxy=1.0, jz=0.5, hx=0.0, hz=0.0
        )

        H_odd = xxz_odd.matrix().toarray()
        E_odd = np.linalg.eigvalsh(H_odd)

        # Check Hermiticity
        assert np.allclose(H_even, H_even.conj().T), "Even parity Hamiltonian not Hermitian"
        assert np.allclose(H_odd, H_odd.conj().T), "Odd parity Hamiltonian not Hermitian"

        print(f"ParityZ even: dim={h_parity_even.Ns}, GS energy={E_even[0]:.6f}")
        print(f"ParityZ odd: dim={h_parity_odd.Ns}, GS energy={E_odd[0]:.6f}")

        # Combine and verify against full spectrum
        h_full = HilbertSpace(lattice=small_chain)
        xxz_full = XXZ(lattice=small_chain, hilbert_space=h_full, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)
        E_full = np.linalg.eigvalsh(xxz_full.matrix.toarray())

        E_combined = np.sort(np.concatenate([E_even, E_odd]))
        E_full_sorted = np.sort(E_full)

        max_error = np.max(np.abs(E_combined - E_full_sorted))
        print(f"ParityZ spectrum reconstruction max error: {max_error:.2e}")
        assert max_error < 1e-12, f"Parity spectrum mismatch: {max_error:.2e}"

    # --------------------------------------------------------------------------------------
    # U(1) Particle Conservation Tests
    # --------------------------------------------------------------------------------------

    def test_xxz_u1_particle_conservation(self, small_chain):
        """
        Test XXZ with U(1) particle number conservation.
        Only valid when hx=0, hz=0 (no fields breaking particle number).
        """
        L = small_chain.lx

        # Full spectrum
        h_full = HilbertSpace(lattice=small_chain)
        xxz_full = XXZ(lattice=small_chain, hilbert_space=h_full, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)
        E_full = np.linalg.eigvalsh(xxz_full.matrix().toarray())
        E_full_sorted = np.sort(E_full)

        # Collect eigenvalues from all particle number sectors
        E_all_N = []
        for N in range(L + 1):
            u1_sym = get_u1_sym(lat=small_chain, val=N)
            h_N = HilbertSpace(lattice=small_chain, global_syms=[u1_sym])

            # Expected dimension: binomial(L, N)
            from math import comb

            expected_dim = comb(L, N)
            assert h_N.Nh == expected_dim, f"N={N}: dimension {h_N.Nh} != expected {expected_dim}"

            xxz_N = XXZ(lattice=small_chain, hilbert_space=h_N, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)

            H_N = xxz_N.matrix.toarray()
            E_N = np.linalg.eigvalsh(H_N)
            E_all_N.extend(E_N)

            print(
                f"N={N}: dim={h_N.Nh} (expected {expected_dim}), "
                f"GS energy={E_N[0] if len(E_N) > 0 else 'N/A':.6f}"
            )

        # Sort and compare
        E_all_N_sorted = np.sort(E_all_N)

        max_error = np.max(np.abs(E_all_N_sorted - E_full_sorted))
        print(f"\nU(1) spectrum reconstruction max error: {max_error:.2e}")
        assert max_error < 1e-12, f"U(1) spectrum mismatch: {max_error:.2e}"

    # --------------------------------------------------------------------------------------
    # Combined Symmetries Tests
    # --------------------------------------------------------------------------------------

    def test_xxz_translation_plus_parity(self, small_chain):
        """
        Test XXZ with combined translation and parity symmetries.
        This further reduces Hilbert space dimension.
        """
        L = small_chain.lx

        # Full spectrum
        h_full = HilbertSpace(lattice=small_chain)
        xxz_full = XXZ(lattice=small_chain, hilbert_space=h_full, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)
        E_full = np.linalg.eigvalsh(xxz_full.matrix().toarray())
        E_full_sorted = np.sort(E_full)

        # Collect from all (k, parity) sectors
        E_all_sectors = []
        for k in range(L):
            for parity in [1, -1]:
                h_kp = HilbertSpace(
                    lattice=small_chain, sym_gen={"translation": k, "parity": parity}
                )
                xxz_kp = XXZ(
                    lattice=small_chain, hilbert_space=h_kp, jxy=1.0, jz=0.5, hx=0.0, hz=0.0
                )

                H_kp = xxz_kp.matrix.toarray()
                E_kp = np.linalg.eigvalsh(H_kp)
                E_all_sectors.extend(E_kp)

                print(f"k={k}, P={parity:+d}: dim={h_kp.Ns}, n_eigs={len(E_kp)}")

        E_all_sorted = np.sort(E_all_sectors)

        max_error = np.max(np.abs(E_all_sorted - E_full_sorted))
        print(f"\nTranslation+Parity spectrum reconstruction max error: {max_error:.2e}")
        print(
            f"Hilbert space reduction: {len(E_full_sorted)} -> {len(E_all_sorted)} "
            f"(factor ~{len(E_full_sorted) / len(E_all_sorted):.1f})"
        )
        assert max_error < 1e-12, f"Combined symmetry mismatch: {max_error:.2e}"

    def test_xxz_translation_plus_u1(self, small_chain):
        """
        Test XXZ with combined translation and U(1) symmetries.
        Each (k, N) sector is independently diagonalizable.
        """
        L = small_chain.lx

        # Full spectrum
        h_full = HilbertSpace(lattice=small_chain)
        xxz_full = XXZ(lattice=small_chain, hilbert_space=h_full, jxy=1.0, jz=0.5, hx=0.0, hz=0.0)
        E_full = np.linalg.eigvalsh(xxz_full.matrix().toarray())
        E_full_sorted = np.sort(E_full)

        # Collect from all (k, N) sectors
        E_all_sectors = []
        for N in range(L + 1):
            u1_sym = get_u1_sym(lat=small_chain, val=N)
            for k in range(L):
                h_kN = HilbertSpace(
                    lattice=small_chain, sym_gen={"translation": k}, global_syms=[u1_sym]
                )

                if h_kN.Nh == 0:
                    continue  # Empty sector

                xxz_kN = XXZ(
                    lattice=small_chain, hilbert_space=h_kN, jxy=1.0, jz=0.5, hx=0.0, hz=0.0
                )

                H_kN = xxz_kN.matrix.toarray()
                E_kN = np.linalg.eigvalsh(H_kN)
                E_all_sectors.extend(E_kN)

                if h_kN.Nh > 0:
                    print(f"N={N}, k={k}: dim={h_kN.Nh}")

        E_all_sorted = np.sort(E_all_sectors)

        max_error = np.max(np.abs(E_all_sorted - E_full_sorted))
        print(f"\nTranslation+U(1) spectrum reconstruction max error: {max_error:.2e}")
        print(f"Total sectors collected: {len(E_all_sorted)}")
        assert max_error < 1e-12, f"Translation+U(1) mismatch: {max_error:.2e}"

    # --------------------------------------------------------------------------------------
    # Special Cases Tests
    # --------------------------------------------------------------------------------------

    def test_xxz_heisenberg_limit(self, small_chain):
        """
        Test XXZ at Heisenberg point (Δ=1, isotropic).
        Has additional SU(2) symmetry.
        """
        h_k0 = HilbertSpace(lattice=small_chain, sym_gen={"translation": 0})
        xxz = XXZ(
            lattice=small_chain, hilbert_space=h_k0, jxy=1.0, delta=1.0, hx=0.0, hz=0.0
        )  # Delta=1 => XXX

        H = xxz.matrix().toarray()
        E = np.linalg.eigvalsh(H)

        # Check for degeneracies (SU(2) multiplets)
        print("\nHeisenberg (XXX) model at k=0:")
        print(f"First 10 eigenvalues: {E[:10]}")
        print(f"Ground state energy: {E[0]:.6f}")

        # Verify Hamiltonian is Hermitian
        assert np.allclose(H, H.conj().T), "Heisenberg Hamiltonian not Hermitian"

    def test_xxz_xy_limit(self, small_chain):
        """
        Test XXZ at XY limit (Δ=0, no Ising coupling).
        Free fermions after Jordan-Wigner transformation.
        """
        h_k0 = HilbertSpace(lattice=small_chain, sym_gen={"translation": 0})
        xxz = XXZ(
            lattice=small_chain, hilbert_space=h_k0, jxy=1.0, delta=0.0, hx=0.0, hz=0.0
        )  # Delta=0 => XY

        H = xxz.matrix().toarray()
        E = np.linalg.eigvalsh(H)

        print("\nXY model (Δ=0) at k=0:")
        print(f"Ground state energy: {E[0]:.6f}")

        assert np.allclose(H, H.conj().T), "XY Hamiltonian not Hermitian"

    def test_xxz_ising_limit(self, small_chain):
        """
        Test XXZ at Ising limit (Jxy=0, only Sz Sz interactions).
        """
        h_k0 = HilbertSpace(lattice=small_chain, sym_gen={"translation": 0})
        xxz = XXZ(
            lattice=small_chain, hilbert_space=h_k0, jxy=0.0, jz=1.0, hx=0.0, hz=0.0
        )  # Jxy=0 => Ising

        H = xxz.matrix().toarray()
        E = np.linalg.eigvalsh(H)

        print("\nIsing limit (Jxy=0) at k=0:")
        print(f"Ground state energy: {E[0]:.6f}")

        assert np.allclose(H, H.conj().T), "Ising Hamiltonian not Hermitian"


##########################################################################################
#! MAIN EXECUTION
##########################################################################################

if __name__ == "__main__":
    """Run tests with detailed output."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])
