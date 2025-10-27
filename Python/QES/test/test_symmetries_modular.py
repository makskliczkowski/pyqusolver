"""
Basic tests for modular symmetry operators (Translation, Reflection, Parity).

File        : QES/test/test_symmetries_modular.py
Description : Smoke tests ensuring modular symmetries apply and compose as expected.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

import unittest
import numpy as np

try:
    from QES.general_python.lattices.square import SquareLattice as Lattice
    from QES.general_python.lattices.lattice import LatticeDirection
except Exception as e:  # pragma: no cover
    Lattice = None
    LatticeDirection = None

from QES.Algebra.Symmetries.translation import (
    TranslationSymmetry,
    build_momentum_superposition,
)
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
from QES.Algebra.Symmetries.parity import ParitySymmetry
from QES.Algebra.Symmetries.compatibility import check_compatibility, infer_momentum_sector_from_operators
from QES.Algebra.Symmetries.base import MomentumSector
from QES.Algebra.Operator.operator import SymmetryGenerators


class TestModularSymmetries(unittest.TestCase):
    def setUp(self):
        if Lattice is None:
            self.skipTest("Lattice not available")

    def _assert_ratio_constant(self, source: dict[int, complex], target: dict[int, complex]):
        self.assertEqual(set(source.keys()), set(target.keys()))
        eigenvalue = None
        for key, amplitude in source.items():
            if np.isclose(amplitude, 0.0):
                continue
            ratio = target[key] / amplitude
            if eigenvalue is None:
                eigenvalue = ratio
            else:
                self.assertTrue(np.isclose(ratio, eigenvalue))
        return eigenvalue

    def test_translation_periodicity_1d(self):
        lat = Lattice(1, 4)
        T = TranslationSymmetry(lat)
        ns = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None) or getattr(lat, 'sites', None)
        state = np.random.randint(0, 2 ** ns)
        s = state
        phase = 1.0
        for _ in range(ns):
            s, ph = T(s)
            phase *= ph
        self.assertEqual(s, state)
        self.assertTrue(np.isclose(phase, 1.0))

    def test_reflection_involution(self):
        lat = Lattice(1, 6)
        R = ReflectionSymmetry(lat)
        ns = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None) or getattr(lat, 'sites', None)
        state = np.random.randint(0, 2 ** ns)
        s1, p1 = R(state)
        s2, p2 = R(s1)
        self.assertEqual(s2, state)
        self.assertTrue(np.isclose(p1 * p2, 1.0))

    def test_parity_x_y_z_flip_shapes(self):
        lat = Lattice(1, 5)
        PX = ParitySymmetry(lat, axis='x')
        PY = ParitySymmetry(lat, axis='y')
        PZ = ParitySymmetry(lat, axis='z')
        ns = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None) or getattr(lat, 'sites', None)
        state = np.random.randint(0, 2 ** ns)

        sx, phx = PX(state)
        sy, phy = PY(state)
        sz, phz = PZ(state)

        self.assertIsInstance(sx, int)
        self.assertIsInstance(sy, int)
        self.assertIsInstance(sz, int)

        sx2, _ = PX(sx)
        sy2, _ = PY(sy)
        sz2, _ = PZ(sz)
        self.assertEqual(sx2, state)
        self.assertEqual(sy2, state)
        self.assertEqual(sz2, state)

    def test_translation_orbit_helpers(self):
        lat = Lattice(1, 8)
        T = TranslationSymmetry(lat)
        ns = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None) or getattr(lat, 'sites', None)
        state = np.random.randint(0, 2 ** ns)
        orb = T.orbit(state)
        m = len(orb)
        # Applying translation m times returns to original
        s = state
        for _ in range(m):
            s, _ = T(s)
        self.assertEqual(s, state)
        # Projector coefficients length matches orbit length and are normalized
        states, coeffs = T.projector_coefficients(k=0, state=state)
        self.assertEqual(len(states), m)
        self.assertTrue(np.isclose(np.sum(np.abs(coeffs) ** 2), 1.0))

    def test_translation_momentum_eigenvector(self):
        lat = Lattice(1, 6)
        T = TranslationSymmetry(lat)
        state = int('101011', 2)
        psi = T.momentum_projector(state, k=2)
        rotated = T.apply_superposition(psi)
        eigenvalue = self._assert_ratio_constant(psi, rotated)
        self.assertIsNotNone(eigenvalue)
        orbit_len = T.orbit_length(state)
        expected = T.apply(state)[1] * TranslationSymmetry.momentum_phase(2, orbit_len)
        self.assertTrue(np.isclose(eigenvalue, expected))

    def test_translation_momentum_projector_2d(self):
        lat = Lattice(2, 3, 2)
        state = int('101011', 2)
        Tx = TranslationSymmetry(lat, LatticeDirection.X)
        Ty = TranslationSymmetry(lat, LatticeDirection.Y)
        translations = {
            LatticeDirection.X: Tx,
            LatticeDirection.Y: Ty,
        }
        momenta = {
            LatticeDirection.X: 1,
            LatticeDirection.Y: 1,
        }
        try:
            psi = build_momentum_superposition(translations, state, momenta)
        except ValueError:
            self.skipTest("Momentum projector collapsed for this configuration.")

        rotated_x = Tx.apply_superposition(psi)
        eigen_x = self._assert_ratio_constant(psi, rotated_x)
        self.assertIsNotNone(eigen_x)
        lx = getattr(lat, 'Lx', None) or getattr(lat, 'lx', None)
        expected_x = Tx.apply(state)[1] * TranslationSymmetry.momentum_phase(momenta[LatticeDirection.X], lx)
        self.assertTrue(np.isclose(eigen_x, expected_x))

        rotated_y = Ty.apply_superposition(psi)
        eigen_y = self._assert_ratio_constant(psi, rotated_y)
        self.assertIsNotNone(eigen_y)
        ly = getattr(lat, 'Ly', None) or getattr(lat, 'ly', None)
        expected_y = Ty.apply(state)[1] * TranslationSymmetry.momentum_phase(momenta[LatticeDirection.Y], ly)
        self.assertTrue(np.isclose(eigen_y, expected_y))

    def test_hilbert_space_momentum_superposition(self):
        lat = Lattice(1, 4)
        hs = HilbertSpace(ns=4, lattice=lat, gen_mapping=False)
        hs._sym_group_modular = [
            TranslationSymmetry(lat, LatticeDirection.X, momentum_index=1)
        ]
        base_state = int('1101', 2)
        psi = hs.build_momentum_superposition(base_state)
        Tx = next(
            sym for sym in hs._sym_group_modular if isinstance(sym, TranslationSymmetry)
        )
        direct = build_momentum_superposition(
            {LatticeDirection.X: Tx}, base_state, {LatticeDirection.X: 1}
        )
        self.assertEqual(set(psi.keys()), set(direct.keys()))
        for key in psi:
            self.assertTrue(np.isclose(psi[key], direct[key]))

    def test_hilbert_space_modular_registry_multi_direction(self):
        lat = Lattice(2, 3, 2)
        sym_gen = [
            (SymmetryGenerators.Translation_x, 1),
            (SymmetryGenerators.Translation_y, 0),
            (SymmetryGenerators.Reflection, 1),
        ]
        hs = HilbertSpace(ns=lat.ns, lattice=lat, sym_gen=sym_gen, gen_mapping=False)

        translations = {
            sym.direction: sym
            for sym in hs._sym_group_modular
            if isinstance(sym, TranslationSymmetry)
        }
        self.assertEqual(set(translations.keys()), {LatticeDirection.X, LatticeDirection.Y})
        self.assertEqual(translations[LatticeDirection.X].momentum_index, 1)
        self.assertEqual(translations[LatticeDirection.Y].momentum_index, 0)
        self.assertTrue(
            any(isinstance(sym, ReflectionSymmetry) for sym in hs._sym_group_modular)
        )

    def test_compatibility_checking_translation_reflection(self):
        """Reflection should be incompatible with translation at generic k."""
        lat = Lattice(1, 4)
        
        # k=0: compatible
        T0 = TranslationSymmetry(lat, LatticeDirection.X, momentum_index=0)
        R = ReflectionSymmetry(lat)
        warnings = []
        compatible, removed = check_compatibility([T0, R], warn_callback=warnings.append)
        self.assertEqual(len(compatible), 2)
        self.assertEqual(len(removed), 0)
        
        # k=\pi: compatible
        T_pi = TranslationSymmetry(lat, LatticeDirection.X, momentum_index=2)
        compatible, removed = check_compatibility([T_pi, R], warn_callback=warnings.append)
        self.assertEqual(len(compatible), 2)
        self.assertEqual(len(removed), 0)
        
        # k=\pi/2: incompatible
        T_gen = TranslationSymmetry(lat, LatticeDirection.X, momentum_index=1)
        warnings_gen = []
        compatible, removed = check_compatibility([T_gen, R], warn_callback=warnings_gen.append)
        self.assertEqual(len(compatible), 1)
        self.assertEqual(len(removed), 1)
        self.assertIn('ReflectionSymmetry', removed)

    def test_momentum_sector_inference(self):
        """Test momentum sector inference from translation operators."""
        lat = Lattice(1, 8)
        
        T0 = TranslationSymmetry(lat, LatticeDirection.X, momentum_index=0)
        self.assertEqual(T0.get_momentum_sector(), MomentumSector.ZERO)
        
        T_pi = TranslationSymmetry(lat, LatticeDirection.X, momentum_index=4)
        self.assertEqual(T_pi.get_momentum_sector(), MomentumSector.PI)
        
        T_gen = TranslationSymmetry(lat, LatticeDirection.X, momentum_index=2)
        self.assertEqual(T_gen.get_momentum_sector(), MomentumSector.GENERIC)
        
        sector = infer_momentum_sector_from_operators([T0])
        self.assertEqual(sector, MomentumSector.ZERO)

    def test_hilbert_automatic_filtering_generic_k(self):
        """Test that HilbertSpace automatically filters reflection at generic k."""
        from QES.Algebra.hilbert import HilbertSpace
        
        # L=8 chain with k=\pi/2 translation (generic momentum)
        lat = Lattice(1, 8)
        
        # Request both translation and reflection
        sym_gen = [
            (SymmetryGenerators.Translation_x, 1),  # k = \pi/2 (generic)
            (SymmetryGenerators.Reflection, 1),
        ]
        
        # HilbertSpace should automatically filter reflection
        hs = HilbertSpace(8, lattice=lat, sym_gen=sym_gen)
        
        # Check that reflection was removed
        has_reflection = any(
            gen == SymmetryGenerators.Reflection
            for gen, _ in hs._sym_group_sec
        )
        self.assertFalse(has_reflection, r"Reflection should be filtered at k=\pi/2")
        
        # Check that translation is still there
        has_translation = any(
            gen == SymmetryGenerators.Translation_x
            for gen, _ in hs._sym_group_sec
        )
        self.assertTrue(has_translation, r"Translation should be present")

    def test_hilbert_automatic_filtering_k_zero(self):
        """Test that HilbertSpace allows reflection at k=0."""
        from QES.Algebra.hilbert import HilbertSpace
        
        # L=8 chain with k=0 translation
        lat = Lattice(1, 8)
        
        sym_gen = [
            (SymmetryGenerators.Translation_x, 0),  # k = 0
            (SymmetryGenerators.Reflection, 1),
        ]
        
        hs = HilbertSpace(8, lattice=lat, sym_gen=sym_gen)
        
        # Both should be present at k=0
        has_reflection = any(
            gen == SymmetryGenerators.Reflection
            for gen, _ in hs._sym_group_sec
        )
        has_translation = any(
            gen == SymmetryGenerators.Translation_x
            for gen, _ in hs._sym_group_sec
        )
        
        self.assertTrue(has_reflection, "Reflection should be allowed at k=0")
        self.assertTrue(has_translation, "Translation should be present at k=0")

    def test_local_space_validation_parity_on_fermions(self):
        """Verify that Parity symmetry is rejected for fermion local spaces."""
        from QES.Algebra.Hilbert.hilbert_local import LocalSpace
        
        # Create 1D lattice (dim=1, lx=4)
        lat = Lattice(1, 4)
        
        # Try to add Parity X (spin symmetry) to fermion system
        sym_gen = [
            (SymmetryGenerators.ParityX, 1),
        ]
        
        # Create fermion Hilbert space with explicit local space
        fermion_space = LocalSpace.default_fermion_spinless()
        hs = HilbertSpace(4, lattice=lat, sym_gen=sym_gen, local_space=fermion_space)
        
        # Parity should be filtered out due to local space incompatibility
        has_parity = any(
            gen == SymmetryGenerators.ParityX
            for gen, _ in hs._sym_group_sec
        )
        
        self.assertFalse(has_parity, "ParityX should be rejected for SPINLESS_FERMIONS local space")

    def test_local_space_validation_parity_on_spins(self):
        """Verify that Parity symmetry is accepted for spin local spaces."""
        from QES.Algebra.Hilbert.hilbert_local import LocalSpace
        
        # Create 1D lattice (dim=1, lx=4)
        lat = Lattice(1, 4)
        
        # Add Parity X (spin symmetry) to spin system
        sym_gen = [
            (SymmetryGenerators.ParityX, 1),
        ]
        
        # Create spin-1/2 Hilbert space with explicit local space
        spin_space = LocalSpace.default_spin_half()
        hs = HilbertSpace(4, lattice=lat, sym_gen=sym_gen, local_space=spin_space)
        
        # Parity should be present for spin systems
        has_parity = any(
            gen == SymmetryGenerators.ParityX
            for gen, _ in hs._sym_group_sec
        )
        
        self.assertTrue(has_parity, "ParityX should be accepted for SPIN_1_2 local space")

    def test_local_space_validation_translation_universal(self):
        """Verify that Translation symmetry works for all local space types."""
        from QES.Algebra.Hilbert.hilbert_local import LocalSpace
        
        # Create 1D lattice (dim=1, lx=4)
        lat = Lattice(1, 4)
        
        # Add Translation (universal symmetry)
        sym_gen = [
            (SymmetryGenerators.Translation_x, 0),
        ]
        
        # Test on fermions
        fermion_space = LocalSpace.default_fermion_spinless()
        hs_fermion = HilbertSpace(4, lattice=lat, sym_gen=sym_gen, local_space=fermion_space)
        has_translation_fermion = any(
            gen == SymmetryGenerators.Translation_x
            for gen, _ in hs_fermion._sym_group_sec
        )
        
        # Test on spins
        spin_space = LocalSpace.default_spin_half()
        hs_spin = HilbertSpace(4, lattice=lat, sym_gen=sym_gen, local_space=spin_space)
        has_translation_spin = any(
            gen == SymmetryGenerators.Translation_x
            for gen, _ in hs_spin._sym_group_sec
        )
        
        self.assertTrue(has_translation_fermion, "Translation should work for fermions")
        self.assertTrue(has_translation_spin, "Translation should work for spins")


if __name__ == "__main__":
    unittest.main()

