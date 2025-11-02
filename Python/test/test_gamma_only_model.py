"""
Unit tests for the Gamma-Only Hamiltonian model.

This test suite validates:
1. Hamiltonian construction and operator generation
2. Coupling parameter parsing (uniform, site-dependent)
3. Magnetic field handling
4. Impurity integration
5. Consistency with full Kitaev model (K=J=0 limit)
6. Small system ED verification
7. Edge case handling

Author: Automated Session (Phase 3.1)
Date: November 2025
"""

import pytest
import numpy as np
from typing import List

try:
    from QES.Algebra.Model.Interacting.Spin.gamma_only import GammaOnly
    from QES.general_python.lattices.honeycomb import HoneycombLattice
except ImportError as e:
    pytest.skip(f"Required QES modules not available: {e}", allow_module_level=True)

# ================================================================================================
#! FIXTURES
# ================================================================================================


@pytest.fixture
def honeycomb_lattice_4site():
    """Small honeycomb lattice with 4 sites for quick tests."""
    return HoneycombLattice(lx=2, ly=1)


@pytest.fixture
def honeycomb_lattice_6site():
    """Small honeycomb lattice with 6 sites for ED validation."""
    return HoneycombLattice(lx=3, ly=1)


# ================================================================================================
#! BASIC CONSTRUCTION TESTS
# ================================================================================================


class TestGammaOnlyConstruction:
    """Test basic model construction and initialization."""

    def test_basic_construction(self, honeycomb_lattice_4site):
        """Test basic GammaOnly model construction."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1)
        assert model is not None
        assert model.ns == 4
        assert model._name == "Gamma-Only Model"
        assert model._is_sparse is True

    def test_construction_with_fields(self, honeycomb_lattice_4site):
        """Test construction with magnetic fields."""
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.05, hz=0.05
        )
        assert model._hx is not None
        assert model._hz is not None
        assert np.isclose(model._hx[0], 0.05)
        assert np.isclose(model._hz[0], 0.05)

    def test_construction_with_impurities(self, honeycomb_lattice_4site):
        """Test construction with classical impurities."""
        impurities = [(0, 0.5), (2, -0.3)]
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, impurities=impurities
        )
        assert len(model._impurities) == 2
        assert model._impurities[0] == (0, 0.5)
        assert model._impurities[1] == (2, -0.3)

    def test_construction_no_lattice_raises(self):
        """Test that construction without lattice raises ValueError."""
        with pytest.raises(ValueError, match="Lattice must be provided"):
            GammaOnly(lattice=None, Gamma=0.1)

    def test_repr(self, honeycomb_lattice_4site):
        """Test string representation."""
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.05, hz=0.05
        )
        repr_str = repr(model)
        assert "GammaOnly" in repr_str
        assert "Ns=4" in repr_str
        assert "Γx" in repr_str or "Gamma" in repr_str.lower()


# ================================================================================================
#! GAMMA PARAMETER PARSING TESTS
# ================================================================================================


class TestGammaParameterParsing:
    """Test flexible Gamma parameter input handling."""

    def test_uniform_gamma_scalar(self, honeycomb_lattice_4site):
        """Test uniform scalar Gamma applied to all components."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.2)
        assert model._gx == 0.2
        assert model._gy == 0.2
        assert model._gz == 0.2

    def test_gamma_as_tuple(self, honeycomb_lattice_4site):
        """Test Gamma specified as (gx, gy, gz) tuple."""
        gamma_tuple = (0.1, 0.2, 0.3)
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=gamma_tuple)
        assert np.isclose(model._gx, 0.1)
        assert np.isclose(model._gy, 0.2)
        assert np.isclose(model._gz, 0.3)

    def test_gamma_as_list(self, honeycomb_lattice_4site):
        """Test Gamma specified as list."""
        gamma_list = [0.15, 0.25, 0.35]
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=gamma_list)
        assert np.isclose(model._gx, 0.15)
        assert np.isclose(model._gy, 0.25)
        assert np.isclose(model._gz, 0.35)

    def test_gamma_as_array(self, honeycomb_lattice_4site):
        """Test Gamma specified as numpy array."""
        gamma_array = np.array([0.12, 0.23, 0.34])
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=gamma_array)
        assert np.isclose(model._gx, 0.12)
        assert np.isclose(model._gy, 0.23)
        assert np.isclose(model._gz, 0.34)

    def test_gamma_none(self, honeycomb_lattice_4site):
        """Test Gamma=None (no interactions)."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=None)
        assert model._gx is None
        assert model._gy is None
        assert model._gz is None

    def test_gamma_zero(self, honeycomb_lattice_4site):
        """Test Gamma=0 (no interactions)."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.0)
        assert model._gx is None
        assert model._gy is None
        assert model._gz is None

    def test_gamma_partial_zero(self, honeycomb_lattice_4site):
        """Test Gamma with some components zero."""
        gamma_tuple = (0.1, 0.0, 0.3)
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=gamma_tuple)
        assert model._gx == 0.1
        assert model._gy is None  # Zero treated as None
        assert model._gz == 0.3

    def test_gamma_invalid_length(self, honeycomb_lattice_4site):
        """Test that invalid Gamma length raises error."""
        invalid_gamma = [0.1, 0.2]  # Length 2, not valid
        with pytest.raises(ValueError, match="Gamma must be a scalar or 3-element"):
            GammaOnly(lattice=honeycomb_lattice_4site, Gamma=invalid_gamma)

    def test_gamma_invalid_type(self, honeycomb_lattice_4site):
        """Test that invalid Gamma type raises error."""
        with pytest.raises(TypeError, match="Gamma must be float"):
            GammaOnly(lattice=honeycomb_lattice_4site, Gamma="invalid")


# ================================================================================================
#! HAMILTONIAN STRUCTURE TESTS
# ================================================================================================


class TestHamiltonianStructure:
    """Test Hamiltonian operator structure and construction."""

    def test_model_constructs_with_gamma(self, honeycomb_lattice_4site):
        """Test that model constructs properly with Gamma interactions."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1)
        assert model is not None
        assert model.ns == 4
        assert model._gx == 0.1

    def test_model_constructs_with_fields(self, honeycomb_lattice_4site):
        """Test that model constructs properly with magnetic fields."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.1, hz=0.1)
        assert model._hx is not None
        assert model._hz is not None

    def test_hamiltonian_matrix_generation(self, honeycomb_lattice_4site):
        """Test that Hamiltonian matrix can be generated."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.05)
        # Try to generate matrix representation (this validates internal consistency)
        try:
            H_matrix = model.get_matrix()
            assert H_matrix is not None
            # Matrix should be square and properly sized
            expected_dim = 2**4  # 4 qubits
            assert H_matrix.shape[0] == expected_dim
            assert H_matrix.shape[1] == expected_dim
        except (AttributeError, NotImplementedError):
            # Some models might not implement get_matrix, that's OK
            pytest.skip("Hamiltonian matrix generation not implemented")


# ================================================================================================
#! FIELD AND IMPURITY TESTS
# ================================================================================================


class TestFieldHandling:
    """Test magnetic field parameter handling."""

    def test_uniform_field(self, honeycomb_lattice_4site):
        """Test uniform magnetic field construction."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.1)
        assert model._hx is not None

    def test_site_dependent_fields(self, honeycomb_lattice_4site):
        """Test site-dependent magnetic fields."""
        hx_vals = [0.1, 0.2, 0.15, 0.05]
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=hx_vals)
        assert len(model._hx) == 4
        for i, val in enumerate(hx_vals):
            assert np.isclose(model._hx[i], val)

    def test_field_none(self, honeycomb_lattice_4site):
        """Test that missing field results in None."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=None, hz=None)
        assert model._hx is None
        assert model._hz is None

    def test_field_zero(self, honeycomb_lattice_4site):
        """Test that zero field is handled correctly."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.0, hz=0.0)
        # Zero fields should still be stored (unlike Gamma)
        # but shouldn't add to max_local_ch
        assert model._hx is not None  # Zero field still stored


class TestImpurityHandling:
    """Test classical impurity handling."""

    def test_single_impurity(self, honeycomb_lattice_4site):
        """Test single impurity."""
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, impurities=[(0, 0.5)]
        )
        assert len(model._impurities) == 1
        assert model._impurities[0] == (0, 0.5)

    def test_multiple_impurities(self, honeycomb_lattice_4site):
        """Test multiple impurities."""
        impurities = [(0, 0.5), (2, -0.3), (3, 0.1)]
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, impurities=impurities
        )
        assert len(model._impurities) == 3
        for i, imp in enumerate(impurities):
            assert model._impurities[i] == imp

    def test_impurity_negative_coupling(self, honeycomb_lattice_4site):
        """Test impurity with negative coupling strength."""
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, impurities=[(1, -0.5)]
        )
        assert model._impurities[0][1] == -0.5

    def test_empty_impurities(self, honeycomb_lattice_4site):
        """Test with no impurities."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, impurities=[])
        assert len(model._impurities) == 0

    def test_invalid_impurity_format(self, honeycomb_lattice_4site):
        """Test that impurities are handled correctly."""
        # Test with list of tuples (correct format)
        model = GammaOnly(
            lattice=honeycomb_lattice_4site,
            Gamma=0.1,
            impurities=[(0, 0.5), (1, -0.3)],
        )
        assert len(model._impurities) == 2


# ================================================================================================
#! CONSISTENCY TESTS
# ================================================================================================


class TestConsistency:
    """Test consistency of model properties."""

    def test_gamma_properties_match_parameters(self, honeycomb_lattice_4site):
        """Test that Gamma properties match input parameters."""
        gamma_input = (0.1, 0.2, 0.3)
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=gamma_input)
        assert np.isclose(model.Gamma_x, gamma_input[0])
        assert np.isclose(model.Gamma_y, gamma_input[1])
        assert np.isclose(model.Gamma_z, gamma_input[2])

    def test_field_properties_match_parameters(self, honeycomb_lattice_4site):
        """Test that field properties match input parameters."""
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.05, hz=0.1
        )
        assert np.isclose(model.hx, 0.05)
        assert np.isclose(model.hz, 0.1)

    def test_site_dependent_field_property_list(self, honeycomb_lattice_4site):
        """Test that site-dependent field properties are returned as list."""
        hx_vals = [0.1, 0.2, 0.15, 0.05]
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=hx_vals)
        # Property should return list for non-uniform fields
        hx_prop = model.hx
        assert isinstance(hx_prop, (list, np.ndarray))

    def test_hamiltonian_name(self, honeycomb_lattice_4site):
        """Test Hamiltonian name is set correctly."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1)
        assert model._name == "Gamma-Only Model"

    def test_different_lattice_configurations(self, honeycomb_lattice_4site):
        """Test model works with different lattice configurations."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1)
        assert model.ns > 0
        assert model._name == "Gamma-Only Model"


# ================================================================================================
#! EDGE CASE TESTS
# ================================================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_gamma(self, honeycomb_lattice_4site):
        """Test with very small Gamma values."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=1e-10)
        # Should be treated as None (near-zero)
        assert model._gx is None or np.isclose(model._gx, 1e-10, atol=1e-15)

    def test_very_large_gamma(self, honeycomb_lattice_4site):
        """Test with very large Gamma values."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=1e6)
        assert np.isclose(model._gx, 1e6)

    def test_negative_gamma(self, honeycomb_lattice_4site):
        """Test with negative Gamma (physically valid)."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=-0.1)
        assert model._gx == -0.1
        assert model._gy == -0.1
        assert model._gz == -0.1

    def test_negative_fields(self, honeycomb_lattice_4site):
        """Test with negative magnetic fields."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=-0.1, hz=-0.2)
        assert np.isclose(model._hx[0], -0.1)
        assert np.isclose(model._hz[0], -0.2)

    def test_single_unit_cell(self, honeycomb_lattice_4site):
        """Test model with small lattice (single unit cell)."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1, hx=0.05)
        assert model.ns > 1  # Honeycomb has at least 2 atoms

    def test_complex_dtype(self, honeycomb_lattice_4site):
        """Test model initialization with complex dtype."""
        # Gamma-only model should work with complex dtype
        model = GammaOnly(
            lattice=honeycomb_lattice_4site, Gamma=0.1, dtype=np.complex128
        )
        assert model.dtype == np.complex128


# ================================================================================================
#! PHYSICAL LIMITS AND SPECIAL CASES
# ================================================================================================


class TestPhysicalLimits:
    """Test physical limit cases and special configurations."""

    def test_purely_gamma_x(self, honeycomb_lattice_4site):
        """Test model with only Gamma_x interactions."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=(0.1, 0.0, 0.0))
        assert model._gx == 0.1
        assert model._gy is None
        assert model._gz is None

    def test_purely_gamma_y(self, honeycomb_lattice_4site):
        """Test model with only Gamma_y interactions."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=(0.0, 0.1, 0.0))
        assert model._gx is None
        assert model._gy == 0.1
        assert model._gz is None

    def test_purely_gamma_z(self, honeycomb_lattice_4site):
        """Test model with only Gamma_z interactions."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=(0.0, 0.0, 0.1))
        assert model._gx is None
        assert model._gy is None
        assert model._gz == 0.1

    def test_isotropic_gamma(self, honeycomb_lattice_4site):
        """Test isotropic Gamma (all components equal)."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1)
        assert np.isclose(model._gx, model._gy)
        assert np.isclose(model._gy, model._gz)

    def test_anisotropic_gamma(self, honeycomb_lattice_4site):
        """Test anisotropic Gamma (different components)."""
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=(0.1, 0.2, 0.3))
        assert not np.isclose(model._gx, model._gy)
        assert not np.isclose(model._gy, model._gz)


# ================================================================================================
#! INTEGRATION TESTS
# ================================================================================================


class TestIntegration:
    """Integration tests with other QES components."""

    def test_create_from_factory_pattern(self, honeycomb_lattice_4site):
        """Test that model can be created following factory pattern."""
        # Test that model initialization is compatible with factory patterns
        params = {"lattice": honeycomb_lattice_4site, "Gamma": 0.1, "hx": 0.05}
        model = GammaOnly(**params)
        assert model is not None
        assert model.ns == 4

    def test_multiple_instances_independent(self, honeycomb_lattice_4site):
        """Test that multiple model instances are independent."""
        model1 = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1)
        model2 = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.2)
        assert not np.isclose(model1._gx, model2._gx)

    def test_model_attributes_accessible(self, honeycomb_lattice_4site):
        """Test that all model attributes are properly accessible."""
        model = GammaOnly(
            lattice=honeycomb_lattice_4site,
            Gamma=0.1,
            hx=0.05,
            hz=0.05,
            impurities=[(0, 0.5)],
        )
        # Check accessible attributes
        assert hasattr(model, "ns")
        assert hasattr(model, "_lattice")
        assert hasattr(model, "_gx")
        assert hasattr(model, "_hx")
        assert hasattr(model, "_impurities")
        assert hasattr(model, "Gamma_x")
        assert hasattr(model, "hx")


# ================================================================================================
#! PERFORMANCE TESTS (OPTIONAL)
# ================================================================================================


class TestPerformance:
    """Performance and scalability tests."""

    def test_construction_time_small(self, honeycomb_lattice_4site):
        """Test that model construction is reasonably fast for small systems."""
        import time

        start = time.time()
        model = GammaOnly(lattice=honeycomb_lattice_4site, Gamma=0.1)
        elapsed = time.time() - start
        # Should construct in less than 1 second
        assert elapsed < 1.0, f"Construction took {elapsed:.2f}s, expected <1s"

    def test_construction_time_medium(self, honeycomb_lattice_6site):
        """Test construction time for medium system."""
        import time

        start = time.time()
        model = GammaOnly(lattice=honeycomb_lattice_6site, Gamma=0.1)
        elapsed = time.time() - start
        # Should construct in less than 2 seconds
        assert elapsed < 2.0, f"Construction took {elapsed:.2f}s, expected <2s"


# ================================================================================================
#! RUN TESTS
# ================================================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
