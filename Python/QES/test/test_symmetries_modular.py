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

from QES.Algebra.Symmetries.translation import TranslationSymmetry
from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
from QES.Algebra.Symmetries.parity import ParitySymmetry


class TestModularSymmetries(unittest.TestCase):
    def setUp(self):
        if Lattice is None:
            self.skipTest("Lattice not available")

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


if __name__ == "__main__":
    unittest.main()
