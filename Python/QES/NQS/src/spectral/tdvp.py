"""
TDVP trajectory helpers for NQS spectral workflows.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np

from .fft import as_time_array
from .results import NQSTDVPRecord

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


REAL_TIME_TDVP_RHS_PREFACTOR = -1.0j


def materialize_trajectory_params(reference_nqs, trajectory: NQSTDVPRecord, flat_or_tree):
    """Reconstruct a parameter pytree from flattened TDVP history when needed."""
    if (
        isinstance(flat_or_tree, np.ndarray)
        and flat_or_tree.ndim == 1
        and trajectory.shapes is not None
        and trajectory.sizes is not None
        and trajectory.is_complex is not None
    ):
        return reference_nqs.transform_flat_params(
            jnp.asarray(flat_or_tree) if JAX_AVAILABLE else flat_or_tree,
            trajectory.shapes,
            trajectory.sizes,
            trajectory.is_complex,
        )
    return flat_or_tree


class NQSParamView:
    """
    Lightweight parameter-fixed view used for TDVP correlator evaluation.
    """

    def __init__(self, nqs, params, global_phase: complex = 0.0j):
        self._nqs = nqs
        self._params = params
        self._global_phase = global_phase
        self.nvisible = nqs.nvisible
        self.hilbert = getattr(nqs, "hilbert", None)

    def sample(self, *args, **kwargs):
        kwargs.setdefault("params", self._params)
        return self._nqs.sample(*args, **kwargs)

    def ansatz(self, states, **kwargs):
        values = self._nqs.ansatz(states, params=self._params, return_values=True, **kwargs)
        if self._global_phase:
            values = values + self._global_phase
        return values

    def __call__(self, states):
        return self.ansatz(states)

    def apply(
        self,
        functions,
        *,
        states_and_psi=None,
        probabilities=None,
        batch_size=None,
        num_samples=None,
        num_chains=None,
        return_values=False,
        log_progress=False,
        args=None,
    ):
        if states_and_psi is not None:
            if isinstance(states_and_psi, tuple) and len(states_and_psi) == 2:
                states, log_psi = states_and_psi
                if log_psi is None:
                    log_psi = self.ansatz(states)
                states_and_psi = (states, log_psi)
            else:
                states = states_and_psi
                states_and_psi = (states, self.ansatz(states))

        return self._nqs.apply(
            functions,
            states_and_psi=states_and_psi,
            probabilities=probabilities,
            batch_size=batch_size,
            parameters=self._params,
            num_samples=num_samples,
            num_chains=num_chains,
            return_values=return_values,
            log_progress=log_progress,
            args=args,
        )


def trajectory_views(
    nqs,
    trajectory: NQSTDVPRecord,
    *,
    start: int = 0,
    stop: Optional[int] = None,
) -> Iterator[NQSParamView]:
    """Yield fixed-parameter NQS views along a TDVP trajectory."""
    stop = len(trajectory.param_history) if stop is None else int(stop)
    for params_t, phase_t in zip(
        trajectory.param_history[start:stop],
        trajectory.global_phase[start:stop],
    ):
        yield NQSParamView(
            nqs,
            materialize_trajectory_params(nqs, trajectory, params_t),
            global_phase=complex(phase_t),
        )


def validate_trajectory_times(
    trajectory: NQSTDVPRecord,
    expected_times: np.ndarray,
    *,
    name: str,
) -> None:
    """Validate that a stored TDVP trajectory matches the requested time grid."""
    actual = np.asarray(trajectory.times, dtype=np.float64)
    expected = np.asarray(expected_times, dtype=np.float64)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} times do not match the requested evolution grid.")


def time_evolve_impl(
    nqs,
    times: Sequence[float],
    *,
    tdvp=None,
    ode_solver: str = "Euler",
    rhs_prefactor: complex = REAL_TIME_TDVP_RHS_PREFACTOR,
    n_batch: Optional[int] = None,
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    max_dt: Optional[float] = None,
    n_substeps: Optional[int] = None,
    restore: bool = True,
    **trainer_kwargs,
) -> NQSTDVPRecord:
    """
    Evolve an NQS along a user-provided time grid using TDVP.
    """
    if not JAX_AVAILABLE or not getattr(nqs, "_isjax", False):
        raise RuntimeError("NQS time evolution currently requires the JAX backend.")

    times_arr = as_time_array(times)
    if not getattr(nqs, "_initialized", False):
        nqs.init_network()

    from ..nqs_train import NQSTrainer

    used_num_samples = int(num_samples if num_samples is not None else nqs.sampler.numsamples)
    used_num_chains = int(num_chains if num_chains is not None else nqs.sampler.numchains)
    used_n_batch = int(n_batch if n_batch is not None else nqs.batch_size)
    if max_dt is not None:
        max_dt = float(max_dt)
        if max_dt <= 0.0:
            raise ValueError("max_dt must be positive.")
    if n_substeps is not None:
        n_substeps = int(n_substeps)
        if n_substeps <= 0:
            raise ValueError("n_substeps must be a positive integer.")

    original_flat = np.array(jax.device_get(nqs.get_params(unravel=True)), copy=True)
    original_params = jax.tree_util.tree_map(
        lambda x: jnp.array(jax.device_get(x), copy=True),
        nqs.get_params(),
    )
    original_phase = complex(getattr(tdvp, "global_phase", 0.0j) if tdvp is not None else 0.0j)

    trainer = NQSTrainer(
        nqs=nqs,
        tdvp=tdvp,
        ode_solver=ode_solver,
        rhs_prefactor=rhs_prefactor,
        n_batch=used_n_batch,
        background=True,
        **trainer_kwargs,
    )

    trainer.tdvp.set_rhs_prefact(rhs_prefactor)
    trainer.tdvp.set_global_phase(original_phase)

    current_flat = original_flat
    current_time = float(times_arr[0])

    param_history = [np.asarray(current_flat)]
    phase_history = [complex(trainer.tdvp.global_phase)]
    mean_energy = [float("nan")]
    std_energy = [float("nan")]
    sigma2_hist = [float("nan")]
    rhat_hist = [float("nan")]
    shapes = sizes = is_cpx = None

    try:
        for target_time in times_arr[1:]:
            interval_dt = float(target_time - current_time)
            if interval_dt < 0.0:
                raise ValueError("times must be monotonic.")

            if interval_dt == 0.0:
                param_history.append(np.asarray(current_flat))
                phase_history.append(complex(trainer.tdvp.global_phase))
                mean_energy.append(mean_energy[-1])
                std_energy.append(std_energy[-1])
                sigma2_hist.append(sigma2_hist[-1])
                rhat_hist.append(rhat_hist[-1])
                continue

            if n_substeps is not None:
                interval_substeps = n_substeps
            elif max_dt is not None:
                interval_substeps = max(1, int(np.ceil(interval_dt / max_dt)))
            else:
                interval_substeps = 1

            stepper = trainer._step_jit if not trainer.lower_states else trainer._step_nojit
            substep_dt = interval_dt / float(interval_substeps)
            tdvp_info = None

            for substep_idx in range(interval_substeps):
                trainer.ode_solver.set_dt(substep_dt)

                (_, _), (configs, configs_ansatze), probabilities = nqs.sample(
                    params=nqs.get_params(),
                    num_samples=used_num_samples,
                    num_chains=used_num_chains,
                )

                new_params_flat, step_dt, (tdvp_info, shapes_info) = stepper(
                    f=trainer.tdvp,
                    est_fn=trainer._single_step_jit,
                    y=current_flat,
                    t=current_time,
                    configs=configs,
                    configs_ansatze=configs_ansatze,
                    probabilities=probabilities,
                    num_chains=used_num_chains,
                )

                shapes, sizes, is_cpx = shapes_info
                nqs.set_params(new_params_flat, shapes=shapes, sizes=sizes, iscpx=is_cpx)
                current_flat = new_params_flat

                step_dt = float(np.real(step_dt))
                if not np.isfinite(step_dt):
                    raise RuntimeError("TDVP ODE solver returned a non-finite step size.")
                if tdvp_info.theta0_dot is not None:
                    trainer.tdvp._theta0_dot = tdvp_info.theta0_dot
                    trainer.tdvp.update_global_phase(dt=step_dt)

                if substep_idx + 1 == interval_substeps:
                    current_time = float(target_time)
                else:
                    current_time += step_dt

            param_history.append(np.asarray(current_flat))
            phase_history.append(complex(trainer.tdvp.global_phase))
            mean_energy.append(float(np.real(tdvp_info.mean_energy)))
            std_energy.append(float(np.real(tdvp_info.std_energy)))
            sigma2_val = tdvp_info.sigma2
            sigma2_hist.append(float(np.real(sigma2_val)) if sigma2_val is not None else float("nan"))
            rhat_val = tdvp_info.r_hat
            rhat_hist.append(float(np.real(rhat_val)) if rhat_val is not None else float("nan"))

    finally:
        if restore:
            nqs.set_params(original_params)
            trainer.tdvp.set_global_phase(original_phase)

    return NQSTDVPRecord(
        times=times_arr,
        param_history=np.asarray(param_history),
        global_phase=np.asarray(phase_history, dtype=np.complex128),
        mean_energy=np.asarray(mean_energy, dtype=np.float64),
        std_energy=np.asarray(std_energy, dtype=np.float64),
        sigma2=np.asarray(sigma2_hist, dtype=np.float64),
        r_hat=np.asarray(rhat_hist, dtype=np.float64),
        num_samples=used_num_samples,
        num_chains=used_num_chains,
        shapes=shapes,
        sizes=sizes,
        is_complex=is_cpx,
        metadata={
            "rhs_prefactor": rhs_prefactor,
            "ode_solver": ode_solver,
            "n_batch": used_n_batch,
        },
    )


__all__ = [
    "NQSParamView",
    "REAL_TIME_TDVP_RHS_PREFACTOR",
    "materialize_trajectory_params",
    "time_evolve_impl",
    "trajectory_views",
    "validate_trajectory_times",
]
