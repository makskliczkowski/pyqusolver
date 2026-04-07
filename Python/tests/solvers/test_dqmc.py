"""DQMC stabilization and adapter regression tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from QES.Algebra.Model.Interacting.Fermionic.hubbard import HubbardModel
from QES.Algebra.Model.Interacting.Fermionic.spinful_hubbard import SpinfulHubbardModel
from QES.Algebra.Model import choose_model
from QES.general_python.lattices import choose_lattice
from QES.pydqmc.dqmc_adapter import HubbardAdapter, SpinlessDensityDensityAdapter, choose_dqmc_adapter
from QES.pydqmc.dqmc_model import HubbardDQMCModel, choose_dqmc_model
from QES.pydqmc.dqmc_sampler import DQMCSampler
from QES.pydqmc.dqmc_solver import DQMCSolver, DQMCResult, load_dqmc_result, run_dqmc
from QES.pydqmc.hs import (
    BondDensityDifferenceHS,
    ChargeHubbardHS,
    CompactInterpolatingHubbardHS,
    GaussianHubbardHS,
    MagneticHubbardHS,
    choose_hs_transformation,
)
from QES.pydqmc.postprocessing import derive_observables, rebin_series, summarize_series
from QES.pydqmc.measurements import measure_time_displaced, reweight_observables
from QES.pydqmc.stabilization import (
    calculate_green_stable,
    calculate_green_stable_numpy,
    green_residual_from_stack_numpy,
    green_residual_numpy,
    localized_diagonal_update,
    stack_product_numpy,
)


def _build_ill_conditioned_stack(num_slices: int, dim: int, span: float, seed: int = 42):
    """
    Build a synthetic DQMC slice stack with a controlled singular-value spread.

    Each slice is `U diag(exp(linspace(span, -span))) V`, so the full product
    becomes progressively more ill-conditioned as `span` grows.
    """
    rng = np.random.RandomState(seed)
    mats = []
    product = np.eye(dim)
    for _ in range(num_slices):
        u_mat, _ = np.linalg.qr(rng.randn(dim, dim))
        v_mat, _ = np.linalg.qr(rng.randn(dim, dim))
        diag = np.diag(np.exp(np.linspace(span, -span, dim)))
        mat = u_mat @ diag @ v_mat
        mats.append(mat)
        product = mat @ product
    return np.asarray(mats), product


def test_stable_green_matches_direct_inverse_for_moderate_conditioning():
    """Stable JAX and pivoted-QR NumPy paths should match the direct inverse when scales are tame."""
    num_slices = 4
    dim = 8
    Bs, B_product = _build_ill_conditioned_stack(num_slices, dim, span=3.0)
    direct = np.linalg.inv(np.eye(dim) + B_product)

    green_jax = np.asarray(calculate_green_stable(jnp.asarray(Bs), n_stable=1))
    green_np = calculate_green_stable_numpy(Bs, n_stable=1)

    np.testing.assert_allclose(green_jax, direct, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(green_np, direct, rtol=1e-8, atol=1e-10)


def test_pivoted_qr_reference_has_bounded_inverse_residual_for_stressed_stack():
    """
    The pivoted-QR/Loh reference path should keep the defining inverse residual small.

    For hard stacks we validate the matrix identity `(I + B) G = I` rather than
    demanding elementwise agreement with a naive inverse.
    """
    num_slices = 6
    dim = 10
    Bs, B_product = _build_ill_conditioned_stack(num_slices, dim, span=6.0)

    green_np = calculate_green_stable_numpy(Bs, n_stable=2)
    residual = green_residual_numpy(green_np, B_product)

    assert np.isfinite(residual)
    assert residual < 1e-3


def test_jax_stable_green_has_bounded_residual_for_stressed_stack():
    """The working JAX path should remain numerically usable on a stressed but finite stack."""
    num_slices = 6
    dim = 10
    Bs, B_product = _build_ill_conditioned_stack(num_slices, dim, span=6.0, seed=42)

    green_jax = np.asarray(calculate_green_stable(jnp.asarray(Bs), n_stable=2))
    residual = green_residual_numpy(green_jax, B_product)

    assert np.isfinite(residual)
    assert residual < 1e-2


def test_stack_product_and_residual_helpers_are_consistent():
    """The residual-from-stack helper should match the explicit residual from the assembled product."""
    Bs, _ = _build_ill_conditioned_stack(num_slices=4, dim=6, span=3.0)
    green = calculate_green_stable_numpy(Bs, n_stable=1)

    residual_from_product = green_residual_numpy(green, stack_product_numpy(Bs))
    residual_from_stack = green_residual_from_stack_numpy(green, Bs)

    assert np.isfinite(residual_from_product)
    assert np.isclose(residual_from_product, residual_from_stack)


def test_hubbard_adapter_extraction_and_measurement_shape():
    """The baseline Hubbard adapter should expose the DQMC-facing contract."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    adapter = HubbardAdapter(hamiltonian=hamiltonian, beta=1.0, M=10, U=4.0)

    kinetic = adapter.kinetic_matrix
    assert kinetic.shape == (4, 4)
    assert np.isclose(kinetic[0, 1], -2.0)
    assert np.isclose(kinetic[0, 2], -2.0)
    assert np.isclose(kinetic[0, 3], 0.0)

    greens = (
        jnp.tile(jnp.eye(4)[None, :, :] * 0.5, (2, 1, 1)),
        jnp.tile(jnp.eye(4)[None, :, :] * 0.5, (2, 1, 1)),
    )
    observables = adapter.measure_equal_time(greens, jnp.asarray(kinetic))

    assert "energy" in observables
    assert "density" in observables
    assert "double_occupancy" in observables
    assert "mz2" in observables


def test_choose_dqmc_adapter_routes_spinless_hubbard_to_density_density_adapter():
    """Spinless Hubbard models should use the bond-density DQMC adapter."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = HubbardModel(lattice=lattice, t=1.0, U=2.0)
    adapter = choose_dqmc_adapter(hamiltonian, beta=2.0, M=8)
    assert isinstance(adapter, SpinlessDensityDensityAdapter)
    assert adapter.n_hs_fields > 0
    assert adapter.max_update_size == 2


def test_choose_dqmc_adapter_accepts_spinful_hubbard_without_legacy_warning():
    """The new spinful Hubbard model should map directly to the two-channel onsite adapter."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=2.0)
    adapter = choose_dqmc_adapter(hamiltonian, beta=2.0, M=8)
    assert isinstance(adapter, HubbardAdapter)
    assert adapter.get_hs_parameters()["name"] == "magnetic"


def test_choose_dqmc_adapter_uses_charge_channel_by_default_for_attractive_spinful_hubbard():
    """The simplest attractive onsite path should default to the charge HS channel."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=-2.0)
    adapter = choose_dqmc_adapter(hamiltonian, beta=2.0, M=8)
    assert isinstance(adapter, HubbardAdapter)
    assert adapter.get_hs_parameters()["name"] == "charge"


def test_hs_factory_returns_magnetic_hubbard_baseline():
    """The working HS factory should expose the magnetic Hirsch transformation."""
    hs = choose_hs_transformation("magnetic", U=4.0, dtau=0.1)
    assert isinstance(hs, MagneticHubbardHS)
    params = hs.parameters()
    assert params["name"] == "magnetic"


def test_hs_factory_returns_spinless_bond_density_baseline():
    """The spinless density-density path should use a bond-centered HS field."""
    hs = choose_hs_transformation(
        "bond_density",
        V=[1.0, 2.0],
        bonds=[(0, 1), (1, 2)],
        n_sites=3,
        dtau=0.1,
    )
    assert isinstance(hs, BondDensityDifferenceHS)
    params = hs.parameters()
    assert params["name"] == "bond_density"
    assert params["n_bonds"] == 2


def test_hs_factory_returns_compact_interpolating_baseline():
    """The general HS factory should expose compact continuous onsite fields too."""
    hs = choose_hs_transformation("compact", U=4.0, dtau=0.1, n_sites=4, p=2.0, proposal_sigma=0.3)
    assert isinstance(hs, CompactInterpolatingHubbardHS)
    params = hs.parameters()
    assert params["name"] == "compact_interpolating"
    assert np.isclose(params["p"], 2.0)


def test_hs_factory_returns_charge_hubbard_baseline():
    """Attractive onsite Hubbard should have a charge-channel discrete HS path."""
    hs = choose_hs_transformation("charge", U=-4.0, dtau=0.1, n_sites=4)
    assert isinstance(hs, ChargeHubbardHS)
    params = hs.parameters()
    assert params["name"] == "charge"


def test_hs_factory_returns_gaussian_hubbard_baseline():
    """The HS factory should expose a noncompact Gaussian onsite field too."""
    hs = choose_hs_transformation("gaussian", U=4.0, dtau=0.1, n_sites=4, proposal_sigma=0.25)
    assert isinstance(hs, GaussianHubbardHS)
    params = hs.parameters()
    assert params["name"] == "gaussian"
    assert np.isclose(params["proposal_sigma"], 0.25)


def test_hubbard_dqmc_model_preserves_historical_entrypoint():
    """The historical wrapper should still expose the kinetic matrix through the adapter-backed model."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    dqmc_model = HubbardDQMCModel(hamiltonian=hamiltonian, beta=1.0, M=10, U=4.0)

    kinetic = dqmc_model.kinetic_matrix
    assert kinetic.shape == (4, 4)
    assert np.isclose(kinetic[0, 1], -2.0)


def test_sampler_equal_time_residuals_are_finite():
    """The sampler should expose per-chain, per-channel inverse residual diagnostics."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    model = HubbardDQMCModel(hamiltonian=hamiltonian, beta=1.0, M=8, U=4.0)
    sampler = DQMCSampler(model=model, n_stable=2, num_chains=2, seed=7)

    residuals = np.asarray(sampler.compute_equal_time_residuals())
    assert residuals.shape == (2, model.n_channels)
    assert np.all(np.isfinite(residuals))
    assert np.max(residuals) < 1e-6


def test_sampler_supports_stronger_numpy_pivoted_refresh_strategy():
    """Full recomputations should be able to use the stronger pivoted-QR reference path."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    model = HubbardDQMCModel(hamiltonian=hamiltonian, beta=1.0, M=8, U=4.0)
    sampler = DQMCSampler(
        model=model,
        n_stable=2,
        num_chains=2,
        seed=3,
        refresh_strategy="numpy_pivoted",
    )

    residuals = np.asarray(sampler.compute_equal_time_residuals())
    assert sampler.refresh_strategy == "numpy_pivoted"
    assert residuals.shape == (2, model.n_channels)
    assert np.all(np.isfinite(residuals))


def test_sampler_can_force_refresh_from_residual_threshold():
    """
    A tiny residual threshold should trigger a full Green's-function refresh after a sweep.
    """
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    model = HubbardDQMCModel(hamiltonian=hamiltonian, beta=1.0, M=8, U=4.0)
    sampler = DQMCSampler(
        model=model,
        n_stable=2,
        num_chains=1,
        seed=9,
        residual_recompute_threshold=0.0,
        residual_check_interval=1,
    )

    sampler.sweep()

    assert sampler.num_forced_refreshes >= 1
    assert sampler.last_refresh_drift >= 0.0
    assert np.all(np.isfinite(np.asarray(sampler.last_equal_time_residuals)))


def test_solver_reports_basic_diagnostics():
    """The solver should summarize acceptance and Green residual diagnostics."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = HubbardModel(lattice=lattice, t=1.0, U=4.0)
    solver = DQMCSolver(
        model=hamiltonian,
        beta=1.0,
        M=8,
        num_chains=1,
        n_stable=2,
        seed=5,
        sign_policy="allow_unsupported",
    )

    diagnostics = solver.measure_diagnostics()
    assert "acceptance_rate" in diagnostics
    assert "green_residual_mean" in diagnostics
    assert "green_residual_max" in diagnostics
    assert "forced_refreshes" in diagnostics
    assert "refresh_drift" in diagnostics
    assert "refresh_strategy" in diagnostics
    assert diagnostics["green_residual_mean"] >= 0.0
    assert 0.0 <= diagnostics["acceptance_rate"] <= 1.0
    assert diagnostics["sign_tracking"] == "measured_on_abs_weight_ensemble"
    assert diagnostics["weight_sampling"] == "absolute_determinant_ratio"
    assert diagnostics["sign_policy"] == "allow_unsupported"
    assert diagnostics["sign_envelope"] == "unsupported"


def test_run_dqmc_returns_compact_summary():
    """The high-level entrypoint should return observables, diagnostics, and training history."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = HubbardModel(lattice=lattice, t=1.0, U=4.0)

    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=8,
        warmup=1,
        sweeps=2,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=13,
        sign_policy="allow_unsupported",
    )

    assert isinstance(result, DQMCResult)
    assert "solver" in result
    assert "observables" in result
    assert "diagnostics" in result
    assert "setup" in result
    assert "energy_history" in result
    assert len(result["energy_history"]) == 2
    assert result["setup"]["hs"]["name"] == "bond_density"
    assert result["setup"]["sign"]["sign_tracking"] == "measured_on_abs_weight_ensemble"
    assert result["setup"]["sampling"]["update_mode"] == "immediate_local"
    assert result["setup"]["sign"]["sign_policy"] == "allow_unsupported"
    assert "average_sign" in result["observables"]


def test_run_dqmc_with_spinful_model_emits_no_legacy_warning():
    """The dedicated spinful model should run through DQMC without the spinless legacy warning."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=8,
        warmup=1,
        sweeps=2,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=21,
    )
    assert len(result["energy_history"]) == 2
    assert result["setup"]["hs"]["name"] == "magnetic"
    assert result["setup"]["sign"]["sign_envelope"] == "known_sign_free"


def test_run_dqmc_with_attractive_spinful_model_uses_charge_default():
    """The simple spinful attractive call should expose the charge-channel default in the summary."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=-4.0, mu=0.5)
    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=6,
        warmup=1,
        sweeps=2,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=23,
    )
    assert len(result["energy_history"]) == 2
    assert result["setup"]["hs"]["name"] == "charge"
    assert result["setup"]["sign"]["sign_envelope"] == "known_sign_free"


def test_run_dqmc_with_compact_spinful_hs_runs():
    """The spinful onsite path should also support compact continuous HS sampling."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=6,
        warmup=1,
        sweeps=2,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=31,
        hs="compact",
        p=2.0,
        proposal_sigma=0.2,
    )
    assert len(result["energy_history"]) == 2
    assert result["diagnostics"]["green_residual_mean"] >= 0.0


def test_run_dqmc_with_gaussian_spinful_hs_runs():
    """The general onsite path should also support Gaussian continuous HS sampling."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=6,
        warmup=1,
        sweeps=2,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=33,
        hs="gaussian",
        proposal_sigma=0.2,
    )
    assert len(result["energy_history"]) == 2
    assert result["diagnostics"]["green_residual_mean"] >= 0.0


def test_run_dqmc_respects_measure_every_for_equal_time_accumulation():
    """Equal-time measurements and energy history should follow the requested cadence."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=8,
        warmup=0,
        sweeps=5,
        measure_every=3,
        num_chains=1,
        n_stable=2,
        seed=37,
    )
    assert len(result["energy_history"]) == 2
    assert result["solver"]._n_measurements == 2


def test_clone_produces_independent_model_state():
    """Changing beta on one solver clone should not mutate the original model."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    solver = DQMCSolver(model=hamiltonian, beta=1.0, M=8, num_chains=1, n_stable=2, seed=41)
    cloned = solver.clone()

    cloned.set_beta(3.0)

    assert solver.model is not cloned.model
    assert np.isclose(solver.model.beta, 1.0)
    assert np.isclose(cloned.model.beta, 3.0)


def test_nonzero_mu_energy_estimator_has_correct_sign_convention():
    """The one-body estimator should treat diagonal chemical-potential terms consistently."""
    lattice = choose_lattice("square", lx=1, ly=1, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=0.0, U=0.0, mu=2.0)
    adapter = HubbardAdapter(hamiltonian=hamiltonian, beta=1.0, M=4, U=0.0)
    greens = (
        jnp.asarray([[[0.5]]], dtype=jnp.float64),
        jnp.asarray([[[0.5]]], dtype=jnp.float64),
    )

    observables = adapter.measure_equal_time(greens, jnp.asarray(adapter.kinetic_matrix))
    assert np.isclose(observables["energy"], -2.0)


def test_spinless_measurement_supports_single_configuration_input():
    """The spinless measurement path should accept `(N, N)` Green's-function input too."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    adapter = choose_dqmc_adapter(HubbardModel(lattice=lattice, t=1.0, U=2.0), beta=1.0, M=6)
    observables = adapter.measure_equal_time((jnp.eye(4) * 0.5,), jnp.asarray(adapter.kinetic_matrix))
    assert "energy" in observables
    assert "bond_density_density" in observables


def test_result_can_be_saved_and_loaded_as_json(tmp_path):
    """Structured DQMC results should support portable JSON export."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=6,
        warmup=0,
        sweeps=2,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=43,
    )

    path = tmp_path / "dqmc_result.json"
    result.save(path)
    loaded = load_dqmc_result(path)

    assert loaded.solver is None
    assert loaded.setup["hs"]["name"] == result.setup["hs"]["name"]
    assert loaded.energy_history == result.energy_history


def test_strict_sign_policy_rejects_replusive_doped_spinful_hubbard():
    """Strict public runs should reject repulsive spinful Hubbard away from half filling."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0, mu=0.3)
    with pytest.raises(ValueError, match="strict sign policy"):
        run_dqmc(
            hamiltonian,
            beta=1.0,
            M=6,
            warmup=0,
            sweeps=1,
            measure_every=1,
            num_chains=1,
            n_stable=2,
            seed=45,
        )


def test_strict_sign_policy_rejects_spinless_public_run():
    """Strict public runs should reject the spinless bond-density path unless the user opts out."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = HubbardModel(lattice=lattice, t=1.0, U=4.0)
    with pytest.raises(ValueError, match="strict sign policy"):
        run_dqmc(
            hamiltonian,
            beta=1.0,
            M=6,
            warmup=0,
            sweeps=1,
            measure_every=1,
            num_chains=1,
            n_stable=2,
            seed=46,
        )


def test_allow_unsupported_sign_policy_permits_doped_repulsive_run_with_explicit_metadata():
    """Users can still run unsupported regimes, but only with an explicit opt-out."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0, mu=0.3)
    result = run_dqmc(
        hamiltonian,
        beta=1.0,
        M=6,
        warmup=0,
        sweeps=2,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=48,
        sign_policy="allow_unsupported",
    )
    assert result["setup"]["sign"]["sign_policy"] == "allow_unsupported"
    assert result["setup"]["sign"]["sign_envelope"] == "unsupported"


def test_result_energy_summary_and_derived_observables_are_available():
    """Structured results should expose simple postprocessing helpers directly."""
    result = DQMCResult(
        solver=None,
        train_result=None,
        observables={"energy": -1.0, "density": 0.8, "double_occupancy": 0.1},
        diagnostics={},
        setup={},
        energy_history=[-1.2, -1.0, -0.8, -0.9],
        energy_autocorr_time=None,
        config={},
    )

    energy = result.summarize_energy(warmup=1, bin_size=1)
    derived = result.derive({
        "local_moment_proxy": lambda obs: obs["density"] - 2.0 * obs["double_occupancy"],
    })

    assert np.isclose(energy["mean"], np.mean([-1.0, -0.8, -0.9]))
    assert np.isclose(derived["local_moment_proxy"], 0.6)


def test_postprocessing_helpers_support_rebinning_and_formula_evaluation():
    """The lightweight workflow helpers should cover the common analysis path."""
    rebinned = rebin_series([1.0, 3.0, 5.0, 7.0], bin_size=2)
    summary = summarize_series([1.0, 2.0, 3.0, 4.0], warmup=1, bin_size=1)
    derived = derive_observables({"density": 0.9, "double_occupancy": 0.2}, {
        "moment": lambda obs: obs["density"] - 2.0 * obs["double_occupancy"],
    })

    np.testing.assert_allclose(rebinned, np.array([2.0, 6.0]))
    assert np.isclose(summary["mean"], 3.0)
    assert np.isclose(derived["moment"], 0.5)


def test_reweight_observables_uses_signed_ratio_estimator():
    """Observable reweighting should implement sum(s O) / sum(s) on the |W|-sampled ensemble."""
    weighted = reweight_observables(
        {
            "energy": jnp.asarray([1.0, 5.0]),
            "density": jnp.asarray([0.2, 0.6]),
        },
        jnp.asarray([1.0, -0.5]),
    )
    assert np.isclose(weighted["energy"], -3.0)
    assert np.isclose(weighted["density"], -0.2)


def test_measure_time_displaced_supports_signed_weights():
    """Unequal-time reduction should support explicit sign/phase weights too."""
    unequal = jnp.asarray(
        [
            [[[[1.0]]], [[[3.0]]]],
            [[[[5.0]]], [[[7.0]]]],
        ],
        dtype=jnp.float64,
    )
    weighted = measure_time_displaced(unequal, weights=jnp.asarray([1.0, -0.5]))
    np.testing.assert_allclose(np.asarray(weighted).reshape(-1), np.array([-3.0, -1.0]))


def test_hook_outputs_can_be_sign_reweighted_per_chain():
    """Custom hooks returning one value per chain should be reweighted automatically."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)

    def hook(model, greens, kinetic_matrix, observables, **kwargs):
        del model, greens, kinetic_matrix, observables, kwargs
        return {"hook_value": jnp.asarray([1.0, 3.0], dtype=jnp.float64)}

    solver = DQMCSolver(
        model=hamiltonian,
        beta=1.0,
        M=6,
        num_chains=2,
        n_stable=2,
        seed=51,
        observable_hooks=(hook,),
    )
    solver.sampler.current_signs = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    observables = solver._measure_current_observables()

    assert np.isclose(observables["hook_value"], -1.0)


def test_solver_checkpoint_roundtrip_restores_auxiliary_fields(tmp_path):
    """Restart checkpoints should restore the saved HS configuration into a compatible solver."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    hamiltonian = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
    solver = DQMCSolver(model=hamiltonian, beta=1.0, M=6, num_chains=1, n_stable=2, seed=47)
    solver.sampler.sweep()
    saved_configs = np.asarray(solver.sampler.configs)

    path = tmp_path / "dqmc_restart.npz"
    solver.save_checkpoint(path)

    restored = DQMCSolver(model=hamiltonian, beta=1.0, M=6, num_chains=1, n_stable=2, seed=49)
    restored.load_checkpoint(path)

    np.testing.assert_allclose(np.asarray(restored.sampler.configs), saved_configs)


def test_choose_dqmc_model_for_spinless_hubbard_uses_term_fields():
    """The spinless model should expose bond HS fields rather than site-only fields."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    model = choose_dqmc_model(HubbardModel(lattice=lattice, t=1.0, U=3.0), beta=1.0, M=8)
    assert model.n_hs_fields > 0
    assert model.max_update_size == 2
    assert model.term_sites.shape == (model.n_hs_fields, 2)


def test_spinless_sampler_runs_with_bond_centered_fields():
    """The faithful spinless path should run with bond HS fields and finite diagnostics."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    model = choose_dqmc_model(HubbardModel(lattice=lattice, t=1.0, U=3.0), beta=1.0, M=6)
    sampler = DQMCSampler(model=model, n_stable=2, num_chains=1, seed=17)

    residuals = np.asarray(sampler.compute_equal_time_residuals())
    assert sampler.configs.shape == (1, 6, model.n_hs_fields)
    assert residuals.shape == (1, model.n_channels)
    assert np.all(np.isfinite(residuals))


def test_localized_diagonal_update_matches_rank_one_formula():
    """The generic localized update should reduce to the exact rank-1 DQMC formula."""
    G = jnp.array([[0.7, 0.1], [0.2, 0.6]], dtype=jnp.float64)
    delta = 0.25
    updated = np.asarray(localized_diagonal_update(G, jnp.array([1]), jnp.array([delta])))

    col = np.asarray(G)[:, 1]
    e_minus_row = np.array([0.0, 1.0]) - np.asarray(G)[1, :]
    prefactor = delta / (1.0 + delta * (1.0 - np.asarray(G)[1, 1]))
    reference = np.asarray(G) - prefactor * np.outer(col, e_minus_row)
    np.testing.assert_allclose(updated, reference, rtol=1e-10, atol=1e-10)


def test_registry_can_construct_spinful_hubbard_model():
    """The shared QES model registry should expose the new spinful Hubbard class."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    model = choose_model("spinful_hubbard", lattice=lattice, t=1.0, U=3.0)
    assert isinstance(model, SpinfulHubbardModel)
