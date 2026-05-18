"""
Probe-state orchestration for NQS dynamical response functions.

This module is the bridge between TDVP trajectories and spectra. It implements
the response workflow used by the public ``NQS`` spectral methods:

1. resolve probe operators ``A`` and ``B`` and their normalization weights,
2. spawn variational probe states representing ``A|psi0>`` and ``B|psi0>``,
3. evolve one probe state, or both half-way with the symmetric two-sided
   ``<-t/2|+t/2>`` protocol,
4. evaluate normalized transition correlators by Monte Carlo or exact summation,
5. send the correlator to ``fft`` for finite-time spectral reconstruction.

The helper functions are kept small so tests can exercise each stage directly;
public users should normally enter through ``NQS.dynamic_structure_factor`` or
``NQS.compute_dynamical_correlator``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .exact import (
    ExactSummationCache,
    build_exact_sum_cache,
    exact_probe_correlator_from_model,
    exact_probe_weight,
    exact_transition_element,
)
from .fft import as_time_array, spectrum_from_correlator_impl
from .operators import (
    diagonal_operator_square,
    diagonal_probe_overlap_operator,
    identity_transition_kernel,
    materialize_apply_values,
)
from .results import NQSCorrelatorResult, NQSSpectralMapResult, NQSSpectralResult, NQSTDVPRecord
from .tdvp import (
    NQSParamView,
    REAL_TIME_TDVP_RHS_PREFACTOR,
    materialize_trajectory_params,
    time_evolve_impl,
    trajectory_views,
    validate_trajectory_times,
)


DEFAULT_NQS_TMP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tmp")
)


def estimate_expectation_value(
    nqs,
    operator,
    *,
    num_samples: int,
    num_chains: int,
) -> complex:
    """Estimate ``<O>`` from Monte Carlo local values."""
    local_values = nqs.apply(
        operator,
        num_samples=num_samples,
        num_chains=num_chains,
        return_values=True,
    )
    local_values = np.asarray(materialize_apply_values(local_values))
    if local_values.size == 0:
        return 0.0 + 0.0j
    return complex(np.mean(local_values.reshape(-1)))


def estimate_probe_weight(
    nqs,
    probe_operator,
    *,
    weight_operator=None,
    estimate_weight: bool = True,
    num_samples: int,
    num_chains: int,
) -> Optional[complex]:
    """Estimate the norm factor of a probe state."""
    if not estimate_weight:
        return None
    weight_kernel = weight_operator if weight_operator is not None else diagonal_operator_square(
        probe_operator
    )
    return estimate_expectation_value(
        nqs,
        weight_kernel,
        num_samples=num_samples,
        num_chains=num_chains,
    )


def resolve_probe_weights(
    nqs,
    *,
    bra_probe_operator,
    ket_probe_operator,
    bra_weight,
    ket_weight,
    bra_weight_operator,
    ket_weight_operator,
    estimate_weights: bool,
    exact_sum: bool,
    num_samples: int,
    num_chains: int,
    cache: Optional[ExactSummationCache] = None,
):
    """Resolve bra/ket probe weights either exactly or by Monte Carlo estimation."""
    if exact_sum:
        if bra_weight is None and estimate_weights:
            bra_weight = exact_probe_weight(
                nqs,
                bra_probe_operator,
                weight_operator=bra_weight_operator,
                cache=cache,
            )
        if ket_weight is None and estimate_weights:
            ket_weight = exact_probe_weight(
                nqs,
                ket_probe_operator,
                weight_operator=ket_weight_operator,
                cache=cache,
            )
    else:
        if bra_weight is None:
            bra_weight = estimate_probe_weight(
                nqs,
                bra_probe_operator,
                weight_operator=bra_weight_operator,
                estimate_weight=estimate_weights,
                num_samples=num_samples,
                num_chains=num_chains,
            )
        if ket_weight is None:
            ket_weight = estimate_probe_weight(
                nqs,
                ket_probe_operator,
                weight_operator=ket_weight_operator,
                estimate_weight=estimate_weights,
                num_samples=num_samples,
                num_chains=num_chains,
            )
    return bra_weight, ket_weight


def probe_seed(base_seed: Optional[int], offset: int) -> Optional[int]:
    """Derive deterministic probe seeds from a base seed."""
    if base_seed is None:
        return None
    return int(base_seed) + int(offset)


def prepare_probe_states(
    nqs,
    *,
    bra_probe_operator,
    ket_probe_operator,
    seed: Optional[int],
    spawn_directory: str,
    spawn_use_orbax: bool,
):
    """Create the bra/ket probe states used by the spectral correlator workflow."""
    base_seed = seed if seed is not None else getattr(nqs, "_seed", None)
    bra_state = nqs.spawn_like(
        modifier=bra_probe_operator,
        seed=probe_seed(base_seed, 101),
        directory=spawn_directory,
        use_orbax=spawn_use_orbax,
        verbose=False,
    )
    ket_state = nqs.spawn_like(
        modifier=ket_probe_operator,
        seed=probe_seed(base_seed, 202),
        directory=spawn_directory,
        use_orbax=spawn_use_orbax,
        verbose=False,
    )
    return bra_state, ket_state


def two_sided_half_times(times: np.ndarray) -> np.ndarray:
    """Return the symmetric half-time grid used by the two-sided protocol."""
    return times[0] + 0.5 * (times - times[0])


def resolve_probe_scale(
    bra_weight: Optional[complex],
    ket_weight: Optional[complex],
) -> complex:
    """Return the overall norm-restoration factor for the normalized correlator."""
    if bra_weight is None or ket_weight is None:
        return 1.0 + 0.0j
    return np.sqrt(complex(bra_weight) * complex(ket_weight))


def prepare_probe_trajectories(
    bra_state,
    ket_state,
    times_arr: np.ndarray,
    *,
    trajectory: Optional[NQSTDVPRecord],
    bra_trajectory: Optional[NQSTDVPRecord],
    protocol: str,
    num_samples: int,
    num_chains: int,
    time_evolve_kwargs: Dict[str, Any],
):
    """Prepare the trajectory objects required by one-sided or two-sided response protocols."""
    half_times = two_sided_half_times(times_arr)

    if protocol == "one_sided":
        if bra_trajectory is not None:
            raise ValueError("bra_trajectory is only valid for evolution_protocol='two_sided'.")
        if trajectory is None:
            trajectory = time_evolve_impl(
                ket_state,
                times_arr,
                num_samples=num_samples,
                num_chains=num_chains,
                **time_evolve_kwargs,
            )
        validate_trajectory_times(trajectory, times_arr, name="trajectory")
        return trajectory, None

    if trajectory is None and bra_trajectory is not None:
        raise ValueError("Provide both trajectories or neither for two-sided evolution.")
    if trajectory is not None and bra_trajectory is None:
        raise ValueError("bra_trajectory must be provided for two-sided evolution.")
    if trajectory is None:
        rhs_prefactor = complex(
            time_evolve_kwargs.get("rhs_prefactor", REAL_TIME_TDVP_RHS_PREFACTOR)
        )
        bra_time_kwargs = dict(time_evolve_kwargs)
        bra_time_kwargs["rhs_prefactor"] = -rhs_prefactor
        trajectory = time_evolve_impl(
            ket_state,
            half_times,
            num_samples=num_samples,
            num_chains=num_chains,
            **time_evolve_kwargs,
        )
        bra_trajectory = time_evolve_impl(
            bra_state,
            half_times,
            num_samples=num_samples,
            num_chains=num_chains,
            **bra_time_kwargs,
        )
    validate_trajectory_times(trajectory, half_times, name="ket trajectory")
    validate_trajectory_times(bra_trajectory, half_times, name="bra trajectory")
    return trajectory, bra_trajectory


def transition_correlator_exact(
    bra_views,
    ket_views,
    *,
    reference_nqs,
    operator,
) -> np.ndarray:
    """Evaluate the transition correlator exactly along a trajectory pair."""
    cache = build_exact_sum_cache(reference_nqs)
    correlator = [
        exact_transition_element(
            bra_view,
            ket_view,
            operator=operator,
            cache=cache,
        )
        for bra_view, ket_view in zip(bra_views, ket_views)
    ]
    return np.asarray(correlator, dtype=np.complex128)


def transition_correlator_mc(
    bra_views,
    ket_views,
    *,
    reference_nqs,
    operator,
    num_samples: int,
    num_chains: int,
    operator_args: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Estimate the transition correlator by Monte Carlo sampling."""
    operator_fun = operator if operator is not None else identity_transition_kernel
    op_args = operator_args or {}

    correlator = []
    for bra_view, ket_view in zip(bra_views, ket_views):
        val = reference_nqs._compute_transition_element(
            bra_view,
            ket_view,
            operator_fun,
            num_samples=num_samples,
            num_chains=num_chains,
            operator_args=op_args,
        )
        correlator.append(complex(np.asarray(val)))
    return np.asarray(correlator, dtype=np.complex128)


def transition_correlator_between_impl(
    bra_nqs,
    ket_nqs,
    trajectory: NQSTDVPRecord,
    *,
    operator=None,
    exact_sum: bool = False,
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compute the normalized transition correlator between a fixed bra and a ket trajectory."""
    operator_fun = operator if operator is not None else identity_transition_kernel
    op_args = operator_args or {}
    used_num_samples = int(num_samples if num_samples is not None else trajectory.num_samples)
    used_num_chains = int(num_chains if num_chains is not None else trajectory.num_chains)
    bra_views = (bra_nqs for _ in trajectory.param_history)
    ket_views = trajectory_views(ket_nqs, trajectory)

    if exact_sum:
        return transition_correlator_exact(
            bra_views,
            ket_views,
            reference_nqs=ket_nqs,
            operator=operator,
        )

    return transition_correlator_mc(
        bra_views,
        ket_views,
        reference_nqs=ket_nqs,
        operator=operator_fun,
        num_samples=used_num_samples,
        num_chains=used_num_chains,
        operator_args=op_args,
    )


def transition_correlator_between_trajectories_impl(
    bra_nqs,
    bra_trajectory: NQSTDVPRecord,
    ket_nqs,
    ket_trajectory: NQSTDVPRecord,
    *,
    operator=None,
    exact_sum: bool = False,
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compute the normalized overlap correlator between two TDVP trajectories."""
    validate_trajectory_times(
        bra_trajectory,
        np.asarray(ket_trajectory.times, dtype=np.float64),
        name="bra trajectory",
    )

    used_num_samples = int(num_samples if num_samples is not None else ket_trajectory.num_samples)
    used_num_chains = int(num_chains if num_chains is not None else ket_trajectory.num_chains)
    bra_views = trajectory_views(bra_nqs, bra_trajectory)
    ket_views = trajectory_views(ket_nqs, ket_trajectory)

    if exact_sum:
        return transition_correlator_exact(
            bra_views,
            ket_views,
            reference_nqs=ket_nqs,
            operator=operator,
        )

    return transition_correlator_mc(
        bra_views,
        ket_views,
        reference_nqs=ket_nqs,
        operator=operator,
        num_samples=used_num_samples,
        num_chains=used_num_chains,
        operator_args=operator_args,
    )


def transition_correlator_impl(
    nqs,
    trajectory: NQSTDVPRecord,
    *,
    operator=None,
    exact_sum: bool = False,
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compute the basic normalized transition correlator of one TDVP trajectory."""
    initial_view = NQSParamView(
        nqs,
        materialize_trajectory_params(nqs, trajectory, trajectory.param_history[0]),
        global_phase=complex(trajectory.global_phase[0]),
    )
    return transition_correlator_between_impl(
        initial_view,
        nqs,
        trajectory,
        operator=operator,
        exact_sum=exact_sum,
        num_samples=num_samples,
        num_chains=num_chains,
        operator_args=operator_args,
    )


def compute_probe_overlap_normalized(
    bra_state,
    ket_state,
    trajectory: NQSTDVPRecord,
    *,
    bra_trajectory: Optional[NQSTDVPRecord],
    protocol: str,
    exact_sum: bool,
    num_samples: int,
    num_chains: int,
    operator_args: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Evaluate the normalized bra/ket probe overlap for the chosen response protocol."""
    if protocol == "one_sided":
        return transition_correlator_between_impl(
            bra_state,
            ket_state,
            trajectory,
            operator=None,
            exact_sum=exact_sum,
            num_samples=num_samples,
            num_chains=num_chains,
            operator_args=operator_args,
        )

    return transition_correlator_between_trajectories_impl(
        bra_state,
        bra_trajectory,
        ket_state,
        trajectory,
        operator=None,
        exact_sum=exact_sum,
        num_samples=num_samples,
        num_chains=num_chains,
        operator_args=operator_args,
    )


def update_result_metadata(result: NQSSpectralResult, **extra_metadata) -> NQSSpectralResult:
    """Return a spectral result with merged metadata."""
    metadata = dict(result.metadata)
    metadata.update(extra_metadata)
    result.metadata = metadata
    return result


def probe_correlator_impl(
    nqs,
    times: Sequence[float],
    *,
    ket_probe_operator,
    bra_probe_operator=None,
    trajectory: Optional[NQSTDVPRecord] = None,
    bra_weight: Optional[complex] = None,
    ket_weight: Optional[complex] = None,
    bra_weight_operator=None,
    ket_weight_operator=None,
    estimate_weights: bool = True,
    exact_sum: bool = False,
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
    spawn_directory: str = DEFAULT_NQS_TMP_DIR,
    spawn_use_orbax: bool = False,
    seed: Optional[int] = None,
    reference_energy: Optional[float] = None,
    stitch_direct_t0: bool = False,
    evolution_protocol: str = "one_sided",
    bra_trajectory: Optional[NQSTDVPRecord] = None,
    **time_evolve_kwargs,
) -> NQSCorrelatorResult:
    """Compute the probe-state correlator ``<psi0|A^dagger exp(-iHt) B|psi0>``."""
    if ket_probe_operator is None:
        raise ValueError("ket_probe_operator must be provided.")
    if bra_probe_operator is None:
        bra_probe_operator = ket_probe_operator
    ket_probe_operator = nqs.resolve_operator(ket_probe_operator)
    bra_probe_operator = nqs.resolve_operator(bra_probe_operator)
    if bra_weight_operator is not None:
        bra_weight_operator = nqs.resolve_operator(bra_weight_operator)
    if ket_weight_operator is not None:
        ket_weight_operator = nqs.resolve_operator(ket_weight_operator)
    protocol = str(evolution_protocol).lower()
    if protocol not in ("one_sided", "two_sided"):
        raise ValueError("evolution_protocol must be 'one_sided' or 'two_sided'.")

    times_arr = as_time_array(times)
    if exact_sum and trajectory is None and bra_trajectory is None:
        correlator = exact_probe_correlator_from_model(
            nqs,
            times_arr,
            bra_probe_operator=bra_probe_operator,
            ket_probe_operator=ket_probe_operator,
            protocol=protocol,
            reference_energy=reference_energy,
        )
        return NQSCorrelatorResult(
            times=times_arr,
            correlator=correlator,
            trajectory=None,
            metadata={
                "bra_probe_operator": repr(bra_probe_operator),
                "ket_probe_operator": repr(ket_probe_operator),
                "exact_sum": True,
                "exact_unitary": True,
                "reference_energy": None if reference_energy is None else float(reference_energy),
                "evolution_protocol": protocol,
                "bra_trajectory": None,
            },
        )

    used_num_samples = int(
        num_samples if num_samples is not None else getattr(nqs.sampler, "numsamples", 0)
    )
    used_num_chains = int(
        num_chains if num_chains is not None else getattr(nqs.sampler, "numchains", 0)
    )
    exact_cache = build_exact_sum_cache(nqs) if exact_sum else None
    bra_weight, ket_weight = resolve_probe_weights(
        nqs,
        bra_probe_operator=bra_probe_operator,
        ket_probe_operator=ket_probe_operator,
        bra_weight=bra_weight,
        ket_weight=ket_weight,
        bra_weight_operator=bra_weight_operator,
        ket_weight_operator=ket_weight_operator,
        estimate_weights=estimate_weights,
        exact_sum=exact_sum,
        num_samples=used_num_samples,
        num_chains=used_num_chains,
        cache=exact_cache,
    )

    direct_t0 = None
    if stitch_direct_t0 and protocol == "one_sided" and not exact_sum:
        try:
            direct_t0 = estimate_expectation_value(
                nqs,
                diagonal_probe_overlap_operator(bra_probe_operator, ket_probe_operator),
                num_samples=used_num_samples,
                num_chains=used_num_chains,
            )
        except Exception:
            direct_t0 = None

    bra_state, ket_state = prepare_probe_states(
        nqs,
        bra_probe_operator=bra_probe_operator,
        ket_probe_operator=ket_probe_operator,
        seed=seed,
        spawn_directory=spawn_directory,
        spawn_use_orbax=spawn_use_orbax,
    )

    trajectory, bra_trajectory = prepare_probe_trajectories(
        bra_state,
        ket_state,
        times_arr,
        trajectory=trajectory,
        bra_trajectory=bra_trajectory,
        protocol=protocol,
        num_samples=used_num_samples,
        num_chains=used_num_chains,
        time_evolve_kwargs=time_evolve_kwargs,
    )

    correlator_normalized = compute_probe_overlap_normalized(
        bra_state,
        ket_state,
        trajectory,
        bra_trajectory=bra_trajectory,
        protocol=protocol,
        exact_sum=exact_sum,
        num_samples=used_num_samples,
        num_chains=used_num_chains,
        operator_args=operator_args,
    )
    scale = resolve_probe_scale(bra_weight, ket_weight)
    correlator = np.asarray(correlator_normalized, dtype=np.complex128) * scale
    if (not exact_sum) and stitch_direct_t0 and direct_t0 is not None and correlator.size > 0:
        correlator[0] = complex(direct_t0)

    if reference_energy is not None:
        correlator = correlator * np.exp(1.0j * float(reference_energy) * (times_arr - times_arr[0]))

    return NQSCorrelatorResult(
        times=times_arr,
        correlator=correlator,
        trajectory=trajectory,
        metadata={
            "bra_weight": bra_weight,
            "ket_weight": ket_weight,
            "scale": scale,
            "direct_t0": direct_t0,
            "correlator_normalized": None
            if correlator_normalized is None
            else np.asarray(correlator_normalized, dtype=np.complex128),
            "bra_probe_operator": repr(bra_probe_operator),
            "ket_probe_operator": repr(ket_probe_operator),
            "exact_sum": bool(exact_sum),
            "reference_energy": None if reference_energy is None else float(reference_energy),
            "stitch_direct_t0": bool(stitch_direct_t0),
            "evolution_protocol": protocol,
            "bra_trajectory": bra_trajectory,
        },
    )


def spectral_function_impl(
    nqs,
    times: Sequence[float],
    *,
    trajectory: Optional[NQSTDVPRecord] = None,
    operator=None,
    eta: float = 0.0,
    window: Optional[str] = "hann",
    subtract_initial: bool = False,
    positive_frequencies_only: bool = True,
    broadening_kind: str = "exponential",
    integration_rule: str = "rectangle",
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
    **time_evolve_kwargs,
) -> NQSSpectralResult:
    """Compute a spectral estimate from the basic transition correlator of one NQS."""
    if trajectory is None:
        trajectory = time_evolve_impl(
            nqs,
            times,
            num_samples=num_samples,
            num_chains=num_chains,
            **time_evolve_kwargs,
        )

    operator = nqs.resolve_operator(operator)
    correlator = transition_correlator_impl(
        nqs,
        trajectory,
        operator=operator,
        num_samples=num_samples,
        num_chains=num_chains,
        operator_args=operator_args,
    )

    result = spectrum_from_correlator_impl(
        trajectory.times,
        correlator,
        eta=eta,
        window=window,
        subtract_initial=subtract_initial,
        positive_frequencies_only=positive_frequencies_only,
        broadening_kind=broadening_kind,
        integration_rule=integration_rule,
    )
    return update_result_metadata(result, source="transition_correlator")


def dynamic_structure_factor_impl(
    nqs,
    times: Sequence[float],
    *,
    probe_operator=None,
    ket_probe_operator=None,
    bra_probe_operator=None,
    trajectory: Optional[NQSTDVPRecord] = None,
    static_weight: Optional[complex] = None,
    bra_weight: Optional[complex] = None,
    ket_weight: Optional[complex] = None,
    weight_operator=None,
    bra_weight_operator=None,
    ket_weight_operator=None,
    estimate_weight: bool = True,
    estimate_weights: Optional[bool] = None,
    eta: float = 0.0,
    window: Optional[str] = "hann",
    subtract_initial: bool = False,
    positive_frequencies_only: bool = True,
    broadening_kind: str = "exponential",
    integration_rule: str = "rectangle",
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
    spawn_directory: str = DEFAULT_NQS_TMP_DIR,
    spawn_use_orbax: bool = False,
    seed: Optional[int] = None,
    exact_sum: bool = False,
    reference_energy: Optional[float] = None,
    evolution_protocol: str = "one_sided",
    bra_trajectory: Optional[NQSTDVPRecord] = None,
    hermitian_extension: bool = True,
    **time_evolve_kwargs,
) -> NQSSpectralResult:
    """Compute a probe-state dynamical structure factor from TDVP evolution."""
    ket_probe = ket_probe_operator if ket_probe_operator is not None else probe_operator
    if ket_probe is None:
        raise ValueError("probe_operator or ket_probe_operator must be provided.")
    bra_probe = bra_probe_operator if bra_probe_operator is not None else ket_probe

    use_estimate_weights = estimate_weight if estimate_weights is None else estimate_weights
    if static_weight is not None:
        if bra_weight is None:
            bra_weight = static_weight
        if ket_weight is None:
            ket_weight = static_weight
    if weight_operator is not None:
        if bra_weight_operator is None:
            bra_weight_operator = weight_operator
        if ket_weight_operator is None:
            ket_weight_operator = weight_operator

    corr_result = probe_correlator_impl(
        nqs,
        times,
        ket_probe_operator=ket_probe,
        bra_probe_operator=bra_probe,
        trajectory=trajectory,
        bra_weight=bra_weight,
        ket_weight=ket_weight,
        bra_weight_operator=bra_weight_operator,
        ket_weight_operator=ket_weight_operator,
        estimate_weights=use_estimate_weights,
        num_samples=num_samples,
        num_chains=num_chains,
        operator_args=operator_args,
        spawn_directory=spawn_directory,
        spawn_use_orbax=spawn_use_orbax,
        seed=seed,
        exact_sum=exact_sum,
        reference_energy=reference_energy,
        evolution_protocol=evolution_protocol,
        bra_trajectory=bra_trajectory,
        **time_evolve_kwargs,
    )

    result = spectrum_from_correlator_impl(
        corr_result.times,
        corr_result.correlator,
        eta=eta,
        window=window,
        subtract_initial=subtract_initial,
        positive_frequencies_only=positive_frequencies_only,
        hermitian_extension=hermitian_extension,
        broadening_kind=broadening_kind,
        integration_rule=integration_rule,
    )
    static_weight = (
        complex(corr_result.correlator[0]) if corr_result.correlator.size else 0.0 + 0.0j
    )
    extra_metadata = dict(corr_result.metadata)
    extra_metadata.update(
        {
            "static_weight": static_weight,
            "probe_norm_scale": corr_result.metadata.get("scale", 1.0 + 0.0j),
            "probe_operator": repr(ket_probe),
            "exact_sum": bool(exact_sum),
            "source": "probe_correlator",
        }
    )
    return update_result_metadata(result, **extra_metadata)


def dynamical_correlator_impl(
    nqs,
    times: Sequence[float],
    *,
    ket_probe_operator,
    bra_probe_operator=None,
    trajectory: Optional[NQSTDVPRecord] = None,
    bra_weight: Optional[complex] = None,
    ket_weight: Optional[complex] = None,
    bra_weight_operator=None,
    ket_weight_operator=None,
    estimate_weights: bool = True,
    exact_sum: bool = False,
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
    spawn_directory: str = DEFAULT_NQS_TMP_DIR,
    spawn_use_orbax: bool = False,
    seed: Optional[int] = None,
    reference_energy: Optional[float] = None,
    evolution_protocol: str = "one_sided",
    bra_trajectory: Optional[NQSTDVPRecord] = None,
    **time_evolve_kwargs,
) -> NQSCorrelatorResult:
    """Public time-domain probe-state correlator helper."""
    return probe_correlator_impl(
        nqs,
        times,
        ket_probe_operator=ket_probe_operator,
        bra_probe_operator=bra_probe_operator,
        trajectory=trajectory,
        bra_weight=bra_weight,
        ket_weight=ket_weight,
        bra_weight_operator=bra_weight_operator,
        ket_weight_operator=ket_weight_operator,
        estimate_weights=estimate_weights,
        exact_sum=exact_sum,
        num_samples=num_samples,
        num_chains=num_chains,
        operator_args=operator_args,
        spawn_directory=spawn_directory,
        spawn_use_orbax=spawn_use_orbax,
        seed=seed,
        reference_energy=reference_energy,
        evolution_protocol=evolution_protocol,
        bra_trajectory=bra_trajectory,
        **time_evolve_kwargs,
    )


def spectral_map_impl(
    nqs,
    times: Sequence[float],
    *,
    probe_operators: Sequence[Any],
    bra_probe_operators: Optional[Sequence[Any]] = None,
    k_values: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    **spectral_kwargs,
) -> NQSSpectralMapResult:
    """Compute a probe- or momentum-resolved spectral map."""
    if probe_operators is None or len(probe_operators) == 0:
        raise ValueError("probe_operators must contain at least one operator.")

    if bra_probe_operators is None:
        bra_probe_operators = probe_operators
    if len(bra_probe_operators) != len(probe_operators):
        raise ValueError("bra_probe_operators must match probe_operators in length.")

    spectral_results = []
    per_probe_metadata = []
    for idx, (ket_probe, bra_probe) in enumerate(zip(probe_operators, bra_probe_operators)):
        result = dynamic_structure_factor_impl(
            nqs,
            times,
            ket_probe_operator=ket_probe,
            bra_probe_operator=bra_probe,
            **spectral_kwargs,
        )
        spectral_results.append(result)
        probe_meta = dict(result.metadata)
        probe_meta["probe_index"] = idx
        if labels is not None and idx < len(labels):
            probe_meta["label"] = labels[idx]
        per_probe_metadata.append(probe_meta)

    reference = spectral_results[0]
    correlator = np.stack([res.correlator for res in spectral_results], axis=0)
    spectrum = np.stack([res.spectrum for res in spectral_results], axis=0)
    spectrum_complex = np.stack([res.spectrum_complex for res in spectral_results], axis=0)

    return NQSSpectralMapResult(
        times=np.asarray(reference.times, dtype=np.float64),
        correlator=np.asarray(correlator, dtype=np.complex128),
        frequencies=np.asarray(reference.frequencies, dtype=np.float64),
        spectrum=np.asarray(spectrum, dtype=np.float64),
        spectrum_complex=np.asarray(spectrum_complex, dtype=np.complex128),
        k_values=None if k_values is None else np.asarray(k_values, dtype=np.float64),
        metadata={
            "labels": None if labels is None else list(labels),
            "per_probe_metadata": per_probe_metadata,
        },
    )


__all__ = [
    "DEFAULT_NQS_TMP_DIR",
    "dynamical_correlator_impl",
    "dynamic_structure_factor_impl",
    "estimate_expectation_value",
    "estimate_probe_weight",
    "prepare_probe_states",
    "prepare_probe_trajectories",
    "probe_correlator_impl",
    "resolve_probe_weights",
    "spectral_function_impl",
    "spectral_map_impl",
    "transition_correlator_between_impl",
    "transition_correlator_between_trajectories_impl",
    "transition_correlator_impl",
    "update_result_metadata",
]
