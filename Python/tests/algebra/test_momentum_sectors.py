import pytest
from QES.Algebra.Symmetries.momentum_sectors import MomentumSectorAnalyzer
from QES.general_python.lattices.lattice import LatticeDirection

class MockLattice:
    def __init__(self, lx, ly=1, lz=1):
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.ns = lx * ly * lz
        self.Ns = self.ns
        self._lx = lx
        self._ly = ly
        self._lz = lz
        self.dim = 2
        self.typek = "Mock"
        self.N_cell = 1

    def get_nn_num(self, *args, **kwargs):
        return 2

    def get_nn(self, *args, **kwargs):
        return [0, 1]

    def get_coordinates(self, site):
        return (site % self.lx, site // self.lx)

    def get_site(self, coords):
        return coords[0] + coords[1] * self.lx

    def site_index(self, x, y, z):
        return self.get_site((x, y))

    def boundary_phase(self, direction, occ_cross):
        return 1.0

class TestMomentumSectors:
    def test_analyze_1d_sectors(self):
        lattice = MockLattice(4)
        lattice.dim = 1
        analyzer = MomentumSectorAnalyzer(lattice)

        sectors = analyzer.analyze_1d_sectors(LatticeDirection.X)

        assert 0 in sectors

        found_state_1 = False
        for rep, info in sectors[0]:
            if rep == 1:
                assert info['period'] == 4
                assert info['allowed_k'] == [0, 1, 2, 3]
                found_state_1 = True
        assert found_state_1

        found_state_3 = False
        for rep, info in sectors[0]:
            if rep == 3: # 3 (0011), 6 (0110), 12 (1100), 9 (1001)
                assert info['period'] == 4
                assert info['allowed_k'] == [0, 1, 2, 3]
                assert set(info['orbit']) == {3, 6, 9, 12}
                found_state_3 = True
        assert found_state_3

    def test_analyze_2d_sectors(self):
        lattice = MockLattice(2, 2)
        analyzer = MomentumSectorAnalyzer(lattice)

        sectors = analyzer.analyze_2d_sectors((LatticeDirection.X, LatticeDirection.Y))

        assert (0, 0) in sectors

        found_state_1 = False
        for rep, info in sectors[(0, 0)]:
            if rep == 1:
                assert info['period_x'] == 2
                assert info['period_y'] == 2
                assert info['allowed_kx'] == [0, 1]
                assert info['allowed_ky'] == [0, 1]
                found_state_1 = True
        assert found_state_1
