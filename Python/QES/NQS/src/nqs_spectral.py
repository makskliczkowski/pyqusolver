r"""
Time-evolution and dynamical spectral helpers for NQS.

This module extends the TDVP/VMC stack with small, production-oriented tools
for real-time dynamics and dynamical response functions. The central physics
objects are:

    - TDVP trajectories of variational parameters,
    - transition correlators between variational states,
    - probe-state correlators of the form
      <psi_0| A^\dagger e^{-iHt} B |psi_0>,
    - FFT-based estimates of spectral densities such as S(q, omega).

The implementation deliberately stays close to the current NQS machinery:
probe states are built as ordinary modified NQS instances, evolved with the
existing TDVP trainer, and compared with the existing Monte Carlo transition
element estimator. This keeps the code physically transparent and compatible
with the rest of the package.
"""

from dataclasses import dataclass, field
import copy
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from QES.general_python.common.binary import BACKEND_DEF_SPIN, BACKEND_REPR, int2base

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

DEFAULT_NQS_TMP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "tmp")
)
REAL_TIME_TDVP_RHS_PREFACTOR = -2.0j


@dataclass
class NQSTDVPRecord:
    r"""
    TDVP trajectory sampled on a user-provided time grid.

    The record stores the parameter history together with the TDVP scalar
    log-amplitude offset and standard diagnostics. The ``global_phase`` field is
    complex because it carries the full additive scalar needed to reconstruct
    dynamical overlaps, not just a real angle. The record represents the
    variational approximation to
    the real-time evolution

        |psi(t)> ~= exp[-i H t] |psi(0)>

    projected onto the tangent space of the current ansatz family.
    r"""

    times: np.ndarray
    param_history: np.ndarray
    global_phase: np.ndarray
    mean_energy: np.ndarray
    std_energy: np.ndarray
    sigma2: np.ndarray
    r_hat: np.ndarray
    num_samples: int
    num_chains: int
    shapes: Optional[List[Any]] = None
    sizes: Optional[List[int]] = None
    is_complex: Optional[List[bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NQSCorrelatorResult:
    r"""
    Time-domain dynamical correlator evaluated from NQS states.

    The typical use case is a probe-state correlator

        C_AB(t) = <psi_0| A^\dagger e^{-iHt} B |psi_0>,

    where the ket probe B|psi_0> is evolved with TDVP and the bra probe
    A|psi_0> is kept fixed. The returned ``trajectory`` is the TDVP evolution of
    the ket probe state.
    """

    times: np.ndarray
    correlator: np.ndarray
    trajectory: Optional[NQSTDVPRecord] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NQSSpectralResult:
    """
    Frequency-domain spectral estimate derived from an NQS correlator.

    The spectrum is obtained from the real-time correlator by a discrete Fourier
    transform with optional exponential broadening and windowing. This is the
    natural output for dynamical response functions such as spectral functions
    and dynamical structure factors.
    """

    times: np.ndarray
    correlator: np.ndarray
    frequencies: np.ndarray
    spectrum: np.ndarray
    spectrum_complex: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NQSSpectralMapResult:
    """
    Momentum-resolved spectral map.

    This container stores a family of correlators and spectra evaluated for a
    list of probe operators, usually one operator per sampled momentum. The
    shape convention is:

        correlator.shape == (Nk, Nt)
        spectrum.shape   == (Nk, Nw)

    which matches the expected input of the k-path and grid plotting helpers in
    ``general_python.common.plotters.plot_helpers``.
    """

    times: np.ndarray
    correlator: np.ndarray
    frequencies: np.ndarray
    spectrum: np.ndarray
    spectrum_complex: np.ndarray
    k_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _identity_transition_kernel(state):
    backend = jnp if JAX_AVAILABLE else np
    x = backend.asarray(state)
    if getattr(x, "ndim", 0) != 1:
        x = x.reshape(-1)
    weights = backend.ones((1,), dtype=backend.result_type(x.dtype, backend.float32))
    return x.reshape((1, -1)), weights


def _as_time_array(times: Sequence[float]) -> np.ndarray:
    arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("times must contain at least one point.")
    if np.any(np.diff(arr) < -1e-12):
        raise ValueError("times must be monotonically non-decreasing.")
    return arr


def _window_values(name: Optional[str], n: int) -> np.ndarray:
    if name is None or str(name).lower() in ("none", "rect", "rectangular"):
        return np.ones(n, dtype=np.float64)

    key = str(name).lower()
    if key in ("hann", "hanning"):
        return np.hanning(n)
    if key == "hamming":
        return np.hamming(n)
    if key == "blackman":
        return np.blackman(n)
    raise ValueError(f"Unsupported window '{name}'.")


def _enumerate_basis_states(nqs) -> np.ndarray:
    """
    Materialize the full computational basis for small exact-summation checks.

    This helper is intended for regression tests and examples where the full
    Hilbert space is small enough that Monte Carlo noise would otherwise obscure
    whether the dynamical machinery is behaving correctly.
    """

    hilbert = getattr(nqs, "hilbert", None)
    if hilbert is None:
        raise ValueError("Exact summation requires an NQS with an attached Hilbert space.")

    sampler = getattr(nqs, "sampler", None)
    sampler_states = getattr(sampler, "_states", None)
    spin = bool(BACKEND_DEF_SPIN)
    spin_value = float(BACKEND_REPR)
    if sampler_states is not None:
        sampler_states = np.asarray(sampler_states)
        if sampler_states.size:
            flat = sampler_states.reshape(-1)
            spin = bool(np.any(flat < 0.0))
            max_abs = float(np.max(np.abs(flat)))
            if max_abs > 0.0:
                spin_value = max_abs

    basis_int = np.asarray(list(hilbert), dtype=np.int64).reshape(-1)
    ns = int(getattr(hilbert, "ns", nqs.nvisible))
    return np.stack(
        [int2base(int(state), ns, spin=spin, spin_value=spin_value, backend="np") for state in basis_int],
        axis=0,
    ).astype(np.float32, copy=False)


def _exact_wavefunction_vector(state_like, *, basis_states: np.ndarray) -> np.ndarray:
    """
    Evaluate an NQS-like object exactly on the provided basis states.

    The returned vector is unnormalized and already includes any tracked TDVP
    scalar offset carried by ``state_like.ansatz``.
    """

    log_psi = np.asarray(state_like.ansatz(basis_states), dtype=np.complex128).reshape(-1)
    return np.exp(log_psi)


def _exact_operator_matrix(nqs_like, operator) -> np.ndarray:
    """
    Build a dense operator matrix for exact small-system correlators.

    Exact summation is restricted to operators exposing ``compute_matrix`` so
    that the resulting overlaps are identical to the many-body ED definition.
    """

    if operator is None:
        dim = int(getattr(nqs_like.hilbert, "Nh", 2 ** nqs_like.nvisible))
        return np.eye(dim, dtype=np.complex128)
    if not hasattr(operator, "compute_matrix"):
        raise ValueError("Exact summation requires an operator with compute_matrix().")
    return np.asarray(
        operator.compute_matrix(
            hilbert_1=nqs_like.hilbert,
            matrix_type="dense",
            use_numpy=True,
        ),
        dtype=np.complex128,
    )


def _exact_transition_element(
    bra_nqs,
    ket_nqs,
    *,
    operator=None,
) -> complex:
    """
    Evaluate the normalized transition element by exact basis summation.

    This is the deterministic counterpart of ``_compute_transition_element`` and
    is primarily useful for tiny benchmark systems where one wants to separate
    variational errors from Monte Carlo estimator noise.
    """

    basis_states = _enumerate_basis_states(ket_nqs)
    bra_vec = _exact_wavefunction_vector(bra_nqs, basis_states=basis_states)
    ket_vec = _exact_wavefunction_vector(ket_nqs, basis_states=basis_states)
    op_mat = _exact_operator_matrix(ket_nqs, operator)

    bra_norm = np.vdot(bra_vec, bra_vec)
    ket_norm = np.vdot(ket_vec, ket_vec)
    if abs(bra_norm) < 1e-30 or abs(ket_norm) < 1e-30:
        return 0.0 + 0.0j

    numerator = np.vdot(bra_vec, op_mat @ ket_vec)
    return complex(numerator / np.sqrt(bra_norm * ket_norm))


def _exact_physical_correlator(
    bra_state,
    ket_state,
    trajectory: "NQSTDVPRecord",
) -> np.ndarray:
    """
    Evaluate the physical probe-state correlator exactly over the full basis.

    The bra and ket states are taken literally as many-body wavefunctions,
    meaning the result is the unnormalized correlator

        <phi_bra|phi_ket(t)>,

    without any Monte Carlo overlap estimator or post-hoc norm restoration.
    """

    basis_states = _enumerate_basis_states(ket_state)
    bra_vec = _exact_wavefunction_vector(bra_state, basis_states=basis_states)

    correlator = []
    for params_t, phase_t in zip(trajectory.param_history, trajectory.global_phase):
        evolved_view = _NQSParamView(
            ket_state,
            _materialize_trajectory_params(ket_state, trajectory, params_t),
            global_phase=complex(phase_t),
        )
        ket_vec = _exact_wavefunction_vector(evolved_view, basis_states=basis_states)
        correlator.append(np.vdot(bra_vec, ket_vec))
    return np.asarray(correlator, dtype=np.complex128)


def _materialize_apply_values(result):
    if hasattr(result, "values"):
        return result.values
    if isinstance(result, tuple):
        return result[0]
    return result


def _resolve_transition_kernel(operator):
    if hasattr(operator, "jax"):
        return operator.jax
    if callable(operator):
        return operator
    raise ValueError("Operator must be callable or expose a .jax transition kernel.")


def _diagonal_operator_square(operator):
    operator_fun = _resolve_transition_kernel(operator)
    backend = jnp if JAX_AVAILABLE else np

    def _square_kernel(state):
        connected_states, weights = operator_fun(state)
        weights_arr = backend.asarray(weights)
        if getattr(weights_arr, "ndim", 0) == 0:
            weights_arr = weights_arr.reshape(1)
        if getattr(weights_arr, "shape", (0,))[0] != 1:
            raise ValueError(
                "Automatic probe-weight estimation requires a diagonal or single-branch operator."
            )
        return connected_states, backend.conj(weights_arr) * weights_arr

    return _square_kernel


def _diagonal_probe_overlap_operator(bra_operator, ket_operator):
    if getattr(bra_operator, "modifies", True) or getattr(ket_operator, "modifies", True):
        raise ValueError("Direct probe-overlap estimation is only valid for diagonal probes.")

    bra_fun = _resolve_transition_kernel(bra_operator)
    ket_fun = _resolve_transition_kernel(ket_operator)
    backend = jnp if JAX_AVAILABLE else np

    def _overlap_kernel(state):
        bra_states, bra_weights = bra_fun(state)
        ket_states, ket_weights = ket_fun(state)
        bra_arr = backend.asarray(bra_weights)
        ket_arr = backend.asarray(ket_weights)
        if getattr(bra_arr, "ndim", 0) == 0:
            bra_arr = bra_arr.reshape(1)
        if getattr(ket_arr, "ndim", 0) == 0:
            ket_arr = ket_arr.reshape(1)
        if getattr(bra_arr, "shape", (0,))[0] != 1 or getattr(ket_arr, "shape", (0,))[0] != 1:
            raise ValueError(
                "Direct probe-overlap estimation requires single-branch diagonal probes."
            )
        return bra_states, backend.conj(bra_arr) * ket_arr

    return _overlap_kernel


def _estimate_expectation_value(
    nqs,
    operator,
    *,
    num_samples: int,
    num_chains: int,
) -> complex:
    local_values = nqs.apply(
        operator,
        num_samples=num_samples,
        num_chains=num_chains,
        return_values=True,
    )
    local_values = np.asarray(_materialize_apply_values(local_values))
    if local_values.size == 0:
        return 0.0 + 0.0j
    return complex(np.mean(local_values.reshape(-1)))


def _estimate_probe_weight(
    nqs,
    probe_operator,
    *,
    weight_operator=None,
    estimate_weight: bool = True,
    num_samples: int,
    num_chains: int,
) -> Optional[complex]:
    if not estimate_weight:
        return None
    weight_kernel = weight_operator if weight_operator is not None else _diagonal_operator_square(
        probe_operator
    )
    return _estimate_expectation_value(
        nqs,
        weight_kernel,
        num_samples=num_samples,
        num_chains=num_chains,
    )


def _exact_probe_weight(
    nqs,
    probe_operator,
    *,
    weight_operator=None,
) -> complex:
    r"""
    Evaluate the normalized probe weight exactly over the full basis.

    The returned object is

        <psi_0| A^\dagger A |psi_0> / <psi_0|psi_0>

    for ``probe_operator = A``. This is the deterministic exact-summation
    counterpart of ``_estimate_probe_weight`` and is required when one wants the
    exact-sum correlator path to use the same normalized-physics convention as
    the Monte Carlo estimator.
    """

    basis_states = _enumerate_basis_states(nqs)
    psi_vec = _exact_wavefunction_vector(nqs, basis_states=basis_states)
    psi_norm = np.vdot(psi_vec, psi_vec)
    if abs(psi_norm) < 1e-30:
        return 0.0 + 0.0j

    if weight_operator is None:
        probe_matrix = _exact_operator_matrix(nqs, probe_operator)
        weight_matrix = probe_matrix.conj().T @ probe_matrix
    else:
        if not hasattr(weight_operator, "compute_matrix"):
            raise ValueError(
                "Exact probe-weight evaluation requires weight_operator with compute_matrix()."
            )
        weight_matrix = _exact_operator_matrix(nqs, weight_operator)

    return complex(np.vdot(psi_vec, weight_matrix @ psi_vec) / psi_norm)


def _probe_seed(base_seed: Optional[int], offset: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + int(offset)


def _spectrum_from_correlator(
    times: Sequence[float],
    correlator: np.ndarray,
    *,
    eta: float = 0.0,
    window: Optional[str] = "hann",
    subtract_initial: bool = False,
    positive_frequencies_only: bool = True,
    hermitian_extension: bool = False,
):
    r"""
    Convert a real-time correlator into a discrete spectral estimate.

    The returned frequency grid is the FFT grid associated with the supplied
    evenly spaced time mesh. The transform applies the optional exponential
    damping and an optional apodization window before the discrete Fourier sum.
    The discrete implementation follows the physics convention

        S(\omega) = \int dt e^{+i \omega t} C(t),

    so a correlator ``C(t) ~ exp(-i omega_0 t)`` produces a peak at the
    positive excitation energy ``omega_0``.

    When ``hermitian_extension=True``, the routine uses the equilibrium
    identity ``C(-t) = C(t)^*`` to evaluate the corresponding two-sided
    transform from positive-time data only. Numerically this is implemented as
    the one-sided quadrature

        S(\omega) = 2 Re \int_0^T dt e^{+i \omega t} C(t) - C(0) dt,

    with the same damping and windowing applied on the measured ``t >= 0``
    branch. This is the physically correct finite-time construction for
    Hermitian equilibrium correlators such as dynamical structure factors.
    """
    times_arr = _as_time_array(times)
    if times_arr.size < 2:
        raise ValueError("At least two time points are required to compute a spectrum.")

    dts = np.diff(times_arr)
    dt = float(dts[0])
    if not np.allclose(dts, dt, rtol=1e-6, atol=1e-12):
        raise ValueError("The spectral FFT currently requires an evenly spaced time grid.")

    corr_work = np.asarray(correlator, dtype=np.complex128)
    t_rel = times_arr - times_arr[0]

    if subtract_initial:
        corr_work = corr_work - corr_work[0]

    if hermitian_extension and corr_work.size > 1:
        n_full = 2 * corr_work.size - 1
        freqs = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(n_full, d=dt))
        window_vals = _window_values(window, corr_work.size)
        corr_pos = corr_work * window_vals
        if eta and abs(eta) > 0.0:
            corr_pos = corr_pos * np.exp(-float(eta) * t_rel)

        phases = np.exp(1.0j * np.outer(freqs, t_rel))
        raw_fft = 2.0 * phases @ (corr_pos * dt) - corr_pos[0] * dt
    else:
        if eta and abs(eta) > 0.0:
            corr_work = corr_work * np.exp(-float(eta) * t_rel)
        window_vals = _window_values(window, corr_work.size)
        corr_work = corr_work * window_vals
        raw_fft = np.fft.fftshift(np.fft.ifft(corr_work) * corr_work.size * dt)
        freqs = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(corr_work.size, d=dt))

    spectrum = np.real(raw_fft)

    if positive_frequencies_only:
        mask = freqs >= -1e-12
        freqs = freqs[mask]
        raw_fft = raw_fft[mask]
        spectrum = spectrum[mask]

    return times_arr, freqs, spectrum, raw_fft


def spectrum_from_correlator_impl(
    times: Sequence[float],
    correlator: np.ndarray,
    *,
    eta: float = 0.0,
    window: Optional[str] = None,
    subtract_initial: bool = False,
    positive_frequencies_only: bool = True,
    hermitian_extension: bool = False,
) -> NQSSpectralResult:
    r"""
    Build a finite-time spectral estimate directly from a supplied correlator.

    Physics
    -------
    Given a sampled correlator ``C(t_n)``, this helper constructs the discrete
    approximation to

        S(\omega) = \int dt e^{i \omega t} C(t),

    on the FFT grid defined by the observation times. The result therefore
    carries the same finite-time resolution, damping, and windowing as the NQS
    dynamical-response routines and is the right comparison object for ED
    correlators sampled on the same time mesh.

    ``hermitian_extension=True`` is the physically appropriate choice for
    equilibrium two-point functions measured only for ``t >= 0`` and obeying
    ``C(-t) = C(t)^*``, for example dynamical structure factors.
    """

    times_arr, freqs, spectrum, raw_fft = _spectrum_from_correlator(
        times,
        correlator,
        eta=eta,
        window=window,
        subtract_initial=subtract_initial,
        positive_frequencies_only=positive_frequencies_only,
        hermitian_extension=hermitian_extension,
    )

    return NQSSpectralResult(
        times=times_arr,
        correlator=np.asarray(correlator, dtype=np.complex128),
        frequencies=freqs,
        spectrum=np.asarray(spectrum, dtype=np.float64),
        spectrum_complex=np.asarray(raw_fft, dtype=np.complex128),
        metadata={
            "eta": float(eta),
            "window": window,
            "positive_frequencies_only": bool(positive_frequencies_only),
            "subtract_initial": bool(subtract_initial),
            "hermitian_extension": bool(hermitian_extension),
        },
    )


class _NQSParamView:
    """
    Lightweight parameter-fixed view used for transition correlators.

    It reuses the base NQS infrastructure but injects fixed parameters and the
    separately tracked global phase from TDVP.
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
            # ``global_phase`` stores the additive complex log-amplitude offset
            # tracked by TDVP, not a standalone real angle.
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


def _materialize_trajectory_params(reference_nqs, trajectory: NQSTDVPRecord, flat_or_tree):
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
    Evolve an NQS along a user-specified time grid using TDVP.

    The returned trajectory approximates the projected Schrödinger evolution in
    the chosen variational manifold. The first point of ``times`` is recorded as
    the initial state and is not stepped.

    The entries of ``times`` are observation times in the same physical units as
    exact diagonalization. They are not rescaled by TDVP. When ``max_dt`` or
    ``n_substeps`` is provided, each interval ``times[i+1] - times[i]`` is
    integrated through smaller internal TDVP steps and only the endpoint is
    recorded. This keeps the plotted timestamps physical while reducing
    integration error from coarse Euler or Runge-Kutta updates.

    For the current complex-parameter TDVP convention the real-time RHS uses
    ``rhs_prefactor = -2j``. The factor of two is required for the projected
    parameter dynamics to reproduce the physical Schrödinger frequency of
    exactly solvable single-spin benchmarks, such as the uniform-state Rabi /
    Larmor precession test.
    """

    if not JAX_AVAILABLE or not getattr(nqs, "_isjax", False):
        raise RuntimeError("NQS time evolution currently requires the JAX backend.")

    times_arr = _as_time_array(times)
    if not getattr(nqs, "_initialized", False):
        nqs.init_network()

    from .nqs_train import NQSTrainer

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

    if JAX_AVAILABLE:
        original_flat = np.array(jax.device_get(nqs.get_params(unravel=True)), copy=True)
        original_params = jax.tree_util.tree_map(
            lambda x: jnp.array(jax.device_get(x), copy=True),
            nqs.get_params(),
        )
    else:
        original_flat = np.array(nqs.get_params(unravel=True), copy=True)
        original_params = copy.deepcopy(nqs.get_params())
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
    """
    Compute a normalized transition correlator between two NQS states.

    The returned quantity is

        C(t) = <phi_bra| O |phi_ket(t)> /
               sqrt(<phi_bra|phi_bra> <phi_ket(t)|phi_ket(t)>),

    where the ket state is taken from ``trajectory`` and the bra state is kept
    fixed. This is the Monte Carlo-stable object used internally before
    restoring the physical probe-state norm factors.
    """

    operator_fun = operator if operator is not None else _identity_transition_kernel
    op_args = operator_args or {}
    used_num_samples = int(num_samples if num_samples is not None else trajectory.num_samples)
    used_num_chains = int(num_chains if num_chains is not None else trajectory.num_chains)

    if exact_sum:
        correlator = []
        for params_t, phase_t in zip(trajectory.param_history, trajectory.global_phase):
            evolved_view = _NQSParamView(
                ket_nqs,
                _materialize_trajectory_params(ket_nqs, trajectory, params_t),
                global_phase=complex(phase_t),
            )
            correlator.append(_exact_transition_element(bra_nqs, evolved_view, operator=operator))
        return np.asarray(correlator, dtype=np.complex128)

    correlator = []
    for params_t, phase_t in zip(trajectory.param_history, trajectory.global_phase):
        evolved_view = _NQSParamView(
            ket_nqs,
            _materialize_trajectory_params(ket_nqs, trajectory, params_t),
            global_phase=complex(phase_t),
        )
        val = ket_nqs._compute_transition_element(
            bra_nqs,
            evolved_view,
            operator_fun,
            num_samples=used_num_samples,
            num_chains=used_num_chains,
            operator_args=op_args,
        )
        correlator.append(complex(np.asarray(val)))

    return np.asarray(correlator, dtype=np.complex128)


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
    """
    Compute the normalized transition correlator of one TDVP trajectory.

    This is the special case

        <psi(0)| O |psi(t)> /
        sqrt(<psi(0)|psi(0)> <psi(t)|psi(t)>)

    obtained by using the same NQS instance for both bra and ket.
    """

    initial_view = _NQSParamView(
        nqs,
        _materialize_trajectory_params(nqs, trajectory, trajectory.param_history[0]),
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


def _probe_correlator_impl(
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
    **time_evolve_kwargs,
) -> NQSCorrelatorResult:
    r"""
    Compute a probe-state correlator between independently prepared bra and ket probes.

    The physics target is

        C_AB(t) = <psi_0| A^\dagger e^{-iHt} B |psi_0>.

    The ket probe ``B|psi_0>`` is evolved with TDVP while the bra probe
    ``A|psi_0>`` remains fixed. Internally the overlap is measured in normalized
    form and multiplied back by the probe norms

        ||A|psi_0>|| * ||B|psi_0>||.

    For diagonal single-branch probes such as S_q^z the norm factors are
    estimated automatically from <psi_0|A^\dagger A|psi_0> and
    <psi_0|B^\dagger B|psi_0>.

    If ``reference_energy`` is provided, the returned correlator is rotated by
    ``exp(+i E_ref t)`` so it matches the Heisenberg-picture convention

        <psi_0| A^\dagger(t) B(0) |psi_0>,

    for a reference state ``|psi_0>`` with energy ``E_ref``.
    """

    if ket_probe_operator is None:
        raise ValueError("ket_probe_operator must be provided.")
    if bra_probe_operator is None:
        bra_probe_operator = ket_probe_operator

    used_num_samples = int(
        num_samples if num_samples is not None else getattr(nqs.sampler, "numsamples", 0)
    )
    used_num_chains = int(
        num_chains if num_chains is not None else getattr(nqs.sampler, "numchains", 0)
    )

    if exact_sum:
        if bra_weight is None and estimate_weights:
            bra_weight = _exact_probe_weight(
                nqs,
                bra_probe_operator,
                weight_operator=bra_weight_operator,
            )
        if ket_weight is None and estimate_weights:
            ket_weight = _exact_probe_weight(
                nqs,
                ket_probe_operator,
                weight_operator=ket_weight_operator,
            )
    else:
        if bra_weight is None:
            bra_weight = _estimate_probe_weight(
                nqs,
                bra_probe_operator,
                weight_operator=bra_weight_operator,
                estimate_weight=estimate_weights,
                num_samples=used_num_samples,
                num_chains=used_num_chains,
            )
        if ket_weight is None:
            ket_weight = _estimate_probe_weight(
                nqs,
                ket_probe_operator,
                weight_operator=ket_weight_operator,
                estimate_weight=estimate_weights,
                num_samples=used_num_samples,
                num_chains=used_num_chains,
            )

    direct_t0 = None
    if stitch_direct_t0:
        try:
            if exact_sum:
                overlap_matrix = _exact_operator_matrix(
                    nqs,
                    _diagonal_probe_overlap_operator(bra_probe_operator, ket_probe_operator),
                )
                basis_states = _enumerate_basis_states(nqs)
                psi_vec = _exact_wavefunction_vector(nqs, basis_states=basis_states)
                psi_norm = np.vdot(psi_vec, psi_vec)
                if abs(psi_norm) > 1e-30:
                    direct_t0 = complex(np.vdot(psi_vec, overlap_matrix @ psi_vec) / psi_norm)
            else:
                direct_t0 = _estimate_expectation_value(
                    nqs,
                    _diagonal_probe_overlap_operator(bra_probe_operator, ket_probe_operator),
                    num_samples=used_num_samples,
                    num_chains=used_num_chains,
                )
        except Exception:
            direct_t0 = None

    base_seed = seed if seed is not None else getattr(nqs, "_seed", None)
    bra_state = nqs.spawn_like(
        modifier=bra_probe_operator,
        seed=_probe_seed(base_seed, 101),
        directory=spawn_directory,
        use_orbax=spawn_use_orbax,
        verbose=False,
    )
    ket_state = nqs.spawn_like(
        modifier=ket_probe_operator,
        seed=_probe_seed(base_seed, 202),
        directory=spawn_directory,
        use_orbax=spawn_use_orbax,
        verbose=False,
    )

    if trajectory is None:
        trajectory = time_evolve_impl(
            ket_state,
            times,
            num_samples=used_num_samples,
            num_chains=used_num_chains,
            **time_evolve_kwargs,
        )

    if exact_sum:
        correlator_normalized = transition_correlator_between_impl(
            bra_state,
            ket_state,
            trajectory,
            operator=None,
            exact_sum=True,
            num_samples=used_num_samples,
            num_chains=used_num_chains,
            operator_args=operator_args,
        )
        if bra_weight is not None and ket_weight is not None:
            scale = np.sqrt(complex(bra_weight) * complex(ket_weight))
        else:
            scale = 1.0 + 0.0j
        correlator = np.asarray(correlator_normalized, dtype=np.complex128) * scale
    else:
        correlator_normalized = transition_correlator_between_impl(
            bra_state,
            ket_state,
            trajectory,
            operator=None,
            exact_sum=False,
            num_samples=used_num_samples,
            num_chains=used_num_chains,
            operator_args=operator_args,
        )

        if bra_weight is not None and ket_weight is not None:
            scale = np.sqrt(complex(bra_weight) * complex(ket_weight))
        else:
            scale = 1.0 + 0.0j

        correlator = np.asarray(correlator_normalized, dtype=np.complex128) * scale
        if stitch_direct_t0 and direct_t0 is not None and correlator.size > 0:
            correlator[0] = complex(direct_t0)

    times_arr = np.asarray(trajectory.times, dtype=np.float64)
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
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
    **time_evolve_kwargs,
) -> NQSSpectralResult:
    r"""
    Compute a spectral estimate from the basic transition correlator of one NQS.

    This is the direct frequency-domain version of

        C_O(t) = <psi(0)| O |psi(t)>.
    """

    if trajectory is None:
        trajectory = time_evolve_impl(
            nqs,
            times,
            num_samples=num_samples,
            num_chains=num_chains,
            **time_evolve_kwargs,
        )

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
    )
    result.metadata.update(
        {
            "source": "transition_correlator",
        },
    )
    return result


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
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
    operator_args: Optional[Dict[str, Any]] = None,
    spawn_directory: str = DEFAULT_NQS_TMP_DIR,
    spawn_use_orbax: bool = False,
    seed: Optional[int] = None,
    exact_sum: bool = False,
    reference_energy: Optional[float] = None,
    **time_evolve_kwargs,
) -> NQSSpectralResult:
    r"""
    Compute a probe-state dynamical structure factor from TDVP evolution.

    The general physics object is

        C_AB(t) = <psi_0| A^\dagger e^{-iHt} B |psi_0>,

    followed by a discrete Fourier transform. For Hermitian probes one can
    simply pass the same operator for bra and ket. For momentum-resolved spin
    response the physically correct choice is typically ``A = S_{-q}``,
    ``B = S_q``.

    Backward-compatible aliases
    ---------------------------
    ``probe_operator``:
        Used as the ket probe and, if no bra probe is provided, also as the
        bra probe.
    ``static_weight``:
        Legacy same-probe norm factor; when provided it is used for both bra
        and ket weights unless explicit ``bra_weight`` / ``ket_weight`` are
        given.
    ``weight_operator``:
        Legacy same-probe weight estimator; reused for both bra and ket unless
        probe-specific estimators are provided.

    ``exact_sum``:
        When True, the probe-state overlaps and norm factors are evaluated by
        deterministic full-basis summation. This is intended for tiny systems
        where one wants to diagnose the DSF machinery itself without additional
        Monte Carlo noise in the correlator estimator.

    ``reference_energy``:
        Energy of the reference state ``|psi_0>``. When provided, the correlator
        is converted from the Schr\"odinger-picture overlap into the physical
        Heisenberg-picture response by multiplying by ``exp(+i E_ref t)``.
    """

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

    corr_result = _probe_correlator_impl(
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
        **time_evolve_kwargs,
    )

    metadata = dict(corr_result.metadata)
    result = spectrum_from_correlator_impl(
        corr_result.times,
        corr_result.correlator,
        eta=eta,
        window=window,
        subtract_initial=subtract_initial,
        positive_frequencies_only=positive_frequencies_only,
        hermitian_extension=True,
    )
    metadata.update(result.metadata)
    metadata.update(
        {
            "static_weight": corr_result.metadata.get("scale", 1.0 + 0.0j),
            "probe_operator": repr(ket_probe),
            "exact_sum": bool(exact_sum),
            "source": "probe_correlator",
        }
    )
    result.metadata = metadata
    return result


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
    **time_evolve_kwargs,
) -> NQSCorrelatorResult:
    r"""
    Public helper for the time-domain probe-state correlator.

    This is the direct interface for monitoring real-time response before any
    Fourier transform or broadening is applied. When ``reference_energy`` is
    supplied, the returned correlator is in the Heisenberg-picture convention
    appropriate for observables such as ``<psi_0|A^\dagger(t) B(0)|psi_0>``.
    """

    return _probe_correlator_impl(
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
    """
    Compute a momentum- or probe-resolved spectral map.

    Each entry of ``probe_operators`` defines one channel of the response,
    usually one momentum point. The output tensor has shape ``(Nk, Nw)`` and is
    directly compatible with the spectral plotting helpers used for sum, single,
    k-path, and k-grid visualizations.
    """

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
    "NQSCorrelatorResult",
    "NQSSpectralMapResult",
    "NQSSpectralResult",
    "NQSTDVPRecord",
    "dynamical_correlator_impl",
    "dynamic_structure_factor_impl",
    "spectral_function_impl",
    "spectral_map_impl",
    "spectrum_from_correlator_impl",
    "time_evolve_impl",
    "transition_correlator_between_impl",
    "transition_correlator_impl",
]
