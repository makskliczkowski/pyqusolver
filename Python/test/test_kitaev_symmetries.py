"""
Comprehensive symmetry tests for the Kitaev-Heisenberg model on honeycomb lattice.

Tests validate that symmetry-reduced Hamiltonians correctly reproduce the full spectrum
when all sectors are combined.

-------------------------------------------------------------------------------
File            : test/test_kitaev_symmetries.py
Author          : Maksymilian Kliczkowski
Email           : maksymilian.kliczkowski@pwr.edu.pl
-------------------------------------------------------------------------------
"""


import numpy as np
import pytest

try:
    from QES.Algebra.globals import get_u1_sym
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
    from QES.general_python.common.flog import get_global_logger
    from QES.general_python.lattices.honeycomb import HoneycombLattice
except ImportError as e:
    raise ImportError(f"Required QES modules not available: {e}")

logger = get_global_logger()


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
        return HoneycombLattice(dim=2, lx=2, ly=3, bc="pbc")

    @pytest.fixture
    def small_honeycomb_sym(self):
        """Small honeycomb lattice with translation symmetry for testing."""
        return HoneycombLattice(dim=2, lx=2, ly=2, bc="pbc")

    # --------------------------------------------------------------------------------------
    # Translation Symmetry Tests
    # --------------------------------------------------------------------------------------

    def test_kitaev_translation_all_k_sectors(self, small_honeycomb):
        """
        Test Kitaev model with translation symmetry across all momentum sectors.
        Verify spectrum reconstruction from all k-sectors.
        """
        L = small_honeycomb.ns

        logger.breakline(3)
        logger.info(f"{'='*80}", color="red")
        logger.info(
            f"Testing translation symmetry on honeycomb lattice with {L} sites", color="red"
        )
        logger.info(f"{'='*80}", color="red")

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
            use_forward=True,
        )
        E_full = np.linalg.eigvalsh(kitaev_full.matrix.toarray())
        E_full_sorted = np.sort(E_full)
        logger.info(f"Full Hilbert space: {h_full.Nh} states", color="red")

        # Collect eigenvalues from all (kx, ky) sectors
        E_all_k = []
        total_states = 0

        # For honeycomb there are two translation directions (kx, ky)
        for kx in range(small_honeycomb.lx):
            for ky in range(max(small_honeycomb.ly, 1)):
                sym = {"translation": {"kx": kx, "ky": ky}}
                h_k = HilbertSpace(lattice=small_honeycomb, sym_gen=sym)

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
                    use_forward=True,
                )

                H_k = kitaev_k.matrix.toarray()
                E_k = np.real(np.linalg.eigvals(H_k))  # Use eigvals, take real part
                E_all_k.extend(E_k)
                total_states += len(E_k)

                logger.info(
                    f"kx={kx}, ky={ky}: dim={h_k.Nh}, n_eigs={len(E_k)}, GS energy={E_k[0]:.6f}",
                    lvl=1,
                    color="green",
                )

        # Sort and compare
        E_all_k_sorted = np.sort(E_all_k)

        # Ensure we collected the same number of states as the full Hilbert space
        assert len(E_all_k_sorted) == len(
            E_full_sorted
        ), f"Collected states ({len(E_all_k_sorted)}) != full space ({len(E_full_sorted)})"

        max_error = np.max(np.abs(E_all_k_sorted - E_full_sorted))
        logger.info(
            f"Translation spectrum reconstruction max error: {max_error:.2e}", lvl=1, color="blue"
        )
        logger.info(
            f"Hilbert space: {len(E_full_sorted)} -> {total_states} states collected",
            lvl=1,
            color="blue",
        )
        logger.info(
            f"Reduction factor: ~{len(E_full_sorted) / total_states:.1f}x", lvl=1, color="blue"
        )

        assert max_error < 1e-12, f"Translation symmetry validation failed: {max_error:.2e}"

    def test_kitaev_translation_all_k_sectors_sym(self, small_honeycomb_sym):
        """
        Test Kitaev model with translation symmetry across all momentum sectors.
        Verify spectrum reconstruction from all k-sectors.
        """
        L = small_honeycomb_sym.ns

        logger.breakline(3)
        logger.info(f"{'='*80}", color="red")
        logger.info(
            f"Testing translation symmetry on honeycomb lattice with {L} sites", color="red"
        )
        logger.info(f"{'='*80}", color="red")

        # Full spectrum (no symmetry)
        h_full = HilbertSpace(lattice=small_honeycomb_sym)
        kitaev_full = HeisenbergKitaev(
            lattice=small_honeycomb_sym,
            hilbert_space=h_full,
            K=(1.0, 1.0, 1.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True,
        )
        E_full = np.linalg.eigvalsh(kitaev_full.matrix.toarray())
        E_full_sorted = np.sort(E_full)
        logger.info(f"Full Hilbert space: {h_full.Nh} states", color="red")

        # Collect eigenvalues from all (kx, ky) sectors
        E_all_k = []
        total_states = 0

        # For honeycomb there are two translation directions (kx, ky)
        for kx in range(small_honeycomb_sym.lx):
            for ky in range(max(small_honeycomb_sym.ly, 1)):
                sym = {"translation": {"kx": kx, "ky": ky}}
                h_k = HilbertSpace(lattice=small_honeycomb_sym, sym_gen=sym)

                if h_k.Nh == 0:
                    continue

                kitaev_k = HeisenbergKitaev(
                    lattice=small_honeycomb_sym,
                    hilbert_space=h_k,
                    K=(1.0, 1.0, 1.0),
                    J=None,
                    Gamma=None,
                    hx=None,
                    hz=None,
                    dtype=np.float64,
                    use_forward=True,
                )

                H_k = kitaev_k.matrix.toarray()
                E_k = np.real(np.linalg.eigvals(H_k))  # Use eigvals, take real part
                E_all_k.extend(E_k)
                total_states += len(E_k)

                logger.info(
                    f"kx={kx}, ky={ky}: dim={h_k.Nh}, n_eigs={len(E_k)}, GS energy={E_k[0]:.6f}",
                    lvl=1,
                    color="green",
                )

        # Sort and compare
        E_all_k_sorted = np.sort(E_all_k)

        # Ensure we collected the same number of states as the full Hilbert space
        assert len(E_all_k_sorted) == len(
            E_full_sorted
        ), f"Collected states ({len(E_all_k_sorted)}) != full space ({len(E_full_sorted)})"

        max_error = np.max(np.abs(E_all_k_sorted - E_full_sorted))
        logger.info(
            f"Translation spectrum reconstruction max error: {max_error:.2e}", lvl=1, color="blue"
        )
        logger.info(
            f"Hilbert space: {len(E_full_sorted)} -> {total_states} states collected",
            lvl=1,
            color="blue",
        )
        logger.info(
            f"Reduction factor: ~{len(E_full_sorted) / total_states:.1f}x", lvl=1, color="blue"
        )

        assert max_error < 1e-12, f"Translation symmetry validation failed: {max_error:.2e}"

    def test_kitaev_translation_k0_sector(self, small_honeycomb):
        """
        Test Kitaev model in k=0 momentum sector only.
        """

        # Explicitly request kx=0, ky=0 sector for 2D lattices
        h_k0 = HilbertSpace(lattice=small_honeycomb, sym_gen={"translation": {"kx": 0, "ky": 0}})

        kitaev_k0 = HeisenbergKitaev(
            lattice=small_honeycomb,
            hilbert_space=h_k0,
            K=(1.0, 1.0, 1.0),
            J=None,
            Gamma=None,
            hx=None,
            hz=None,
            dtype=np.float64,
            use_forward=True,
        )

        H_k0 = kitaev_k0.matrix.toarray()

        # Eigenvalues should be real even if representation is not Hermitian
        E_k0 = np.linalg.eigvals(H_k0)
        imag_part = np.max(np.abs(np.imag(E_k0)))
        logger.info(
            f"k=0 sector: {h_k0.Nh} states, GS energy = {np.real(E_k0[np.argmin(np.real(E_k0))]):.6f}",
            color="red",
        )
        logger.info(f"Max imaginary part of eigenvalues: {imag_part:.2e}", color="red")
        logger.info(
            f"Energy range: [{np.min(np.real(E_k0)):.4f}, {np.max(np.real(E_k0)):.4f}]", color="red"
        )

        assert imag_part < 1e-10, f"Eigenvalues should be real, max imag = {imag_part:.2e}"
        assert h_k0.Nh > 0, "k=0 sector should not be empty"

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
            use_forward=True,
        )
        H = kitaev.matrix.toarray()

        # Check Hermiticity
        assert np.allclose(H, H.T.conj()), "Isotropic Kitaev Hamiltonian not Hermitian"

        E = np.linalg.eigvalsh(H)
        logger.info("Isotropic Kitaev (Kx=Ky=Kz=1.0):", color="red")
        logger.info(f"  Hilbert dimension: {h_full.Nh}", color="red")
        logger.info(f"  GS energy: {E[0]:.6f}", color="red")
        logger.info(f"  Energy gap: {E[1] - E[0]:.6f}", color="red")
        logger.info(f"  Energy range: [{E[0]:.4f}, {E[-1]:.4f}]", color="red")

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
            use_forward=True,
        )

        H = kitaev.matrix.toarray()
        assert np.allclose(H, H.T.conj()), "Ising-X Hamiltonian not Hermitian"

        E = np.linalg.eigvalsh(H)
        logger.info("Ising-X limit (Kx=1.0, Ky=Kz=0.0):", color="red")
        logger.info(f"  GS energy: {E[0]:.6f}", color="red")
        logger.info(f"  First excited: {E[1]:.6f}", color="red")

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
            use_forward=True,
        )

        H = kitaev_heis.matrix.toarray()
        assert np.allclose(H, H.T.conj()), "Kitaev-Heisenberg Hamiltonian not Hermitian"

        E = np.linalg.eigvalsh(H)
        print("Kitaev-Heisenberg (K=1.0, J=0.5):")
        print(f"  GS energy: {E[0]:.6f}")
        print(f"  Energy gap: {E[1] - E[0]:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

# =============================================================================
#! EOF
# =============================================================================
