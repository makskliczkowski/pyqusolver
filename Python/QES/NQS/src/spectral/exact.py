"""
Exact small-system spectral helpers for QES.NQS.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    from QES.general_python.common.binary   import int2base
    from ..nqs_network_representation       import resolve_nqs_state_defaults
except ImportError as e:
    raise ImportError("Failed to import necessary utilities for NQS spectral module. Ensure general_python package is correctly installed.") from e

# ----------------------------------------------------------------------

@dataclass
class ExactSummationCache:
    """
    Small exact-summation workspace reused across correlator evaluations.
    """

    basis_states        : np.ndarray
    state_lookup        : Dict[Any, int] = field(default_factory=dict)
    operator_matrices   : Dict[Any, np.ndarray] = field(default_factory=dict)


def build_exact_sum_cache(nqs) -> ExactSummationCache:
    basis_states    = enumerate_basis_states(nqs)
    lookup          = {basis_state_key(state): idx for idx, state in enumerate(basis_states)}
    return ExactSummationCache(basis_states=basis_states, state_lookup=lookup)

def enumerate_basis_states(nqs) -> np.ndarray:
    """
    Materialize the full computational basis for small exact-summation checks.
    """
    hilbert = getattr(nqs, "hilbert", None)
    if hilbert is None:
        raise ValueError("Exact summation requires an NQS with an attached Hilbert space.")

    spin, spin_value    = resolve_nqs_state_defaults(nqs, fallback_mode_repr=0.5)
    basis_int           = np.asarray(list(hilbert), dtype=np.int64).reshape(-1)
    ns                  = int(getattr(hilbert, "ns", nqs.nvisible))
    return np.stack(
        [int2base(int(state), ns, spin=spin, spin_value=spin_value, backend="np") for state in basis_int],
        axis=0,
    ).astype(np.float32, copy=False)

def exact_wavefunction_vector(state_like, *, basis_states: np.ndarray) -> np.ndarray:
    """
    Evaluate an NQS-like object exactly on the provided basis states.
    """
    log_psi = np.asarray(state_like.ansatz(basis_states), dtype=np.complex128).reshape(-1)
    return np.exp(log_psi)

def basis_state_key(state: np.ndarray) -> tuple:
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    return tuple(np.round(arr, decimals=12).tolist())

def exact_operator_matrix_from_kernel(cache: ExactSummationCache, operator) -> np.ndarray:
    operator_fun = resolve_transition_kernel(operator)
    basis_states = cache.basis_states
    dim = basis_states.shape[0]
    matrix = np.zeros((dim, dim), dtype=np.complex128)

    for ket_idx, basis_state in enumerate(basis_states):
        connected_states, weights = operator_fun(basis_state)
        connected_arr = np.asarray(connected_states)
        if connected_arr.ndim == 1:
            connected_arr = connected_arr.reshape(1, -1)
        weights_arr = np.asarray(weights, dtype=np.complex128).reshape(-1)
        if connected_arr.shape[0] != weights_arr.shape[0]:
            raise ValueError("Operator kernel returned incompatible connected_states / weights shapes.")

        for bra_state, weight in zip(connected_arr, weights_arr):
            bra_idx = cache.state_lookup.get(basis_state_key(bra_state))
            if bra_idx is None:
                raise ValueError("Exact summation operator produced a state outside the enumerated Hilbert basis.")
            matrix[bra_idx, ket_idx] += weight

    return matrix

def exact_operator_matrix(
    nqs_like,
    operator,
    *,
    cache: Optional[ExactSummationCache] = None,
) -> np.ndarray:
    """
    Build a dense operator matrix for exact small-system correlators.
    """
    cache_key = None if operator is None else id(operator)
    if cache is not None and cache_key in cache.operator_matrices:
        return cache.operator_matrices[cache_key]

    if operator is None:
        dim = int(getattr(nqs_like.hilbert, "Nh", 2 ** nqs_like.nvisible))
        matrix = np.eye(dim, dtype=np.complex128)
    elif isinstance(operator, np.ndarray):
        matrix = np.asarray(operator, dtype=np.complex128)
    elif hasattr(operator, "compute_matrix"):
        matrix = np.asarray(
            operator.compute_matrix(hilbert_1=nqs_like.hilbert, matrix_type="dense", use_numpy=True),
            dtype=np.complex128,
        )
    else:
        if cache is None:
            cache = build_exact_sum_cache(nqs_like)
        matrix = exact_operator_matrix_from_kernel(cache, operator)

    if cache is not None:
        cache.operator_matrices[cache_key] = matrix
    return matrix

def exact_transition_element(
    bra_nqs,
    ket_nqs,
    *,
    operator=None,
    cache: Optional[ExactSummationCache] = None,
) -> complex:
    """
    Evaluate the normalized transition element by exact basis summation.
    """
    if cache is None:
        cache = build_exact_sum_cache(ket_nqs)
    basis_states = cache.basis_states
    bra_vec = exact_wavefunction_vector(bra_nqs, basis_states=basis_states)
    ket_vec = exact_wavefunction_vector(ket_nqs, basis_states=basis_states)
    op_mat = exact_operator_matrix(ket_nqs, operator, cache=cache)

    bra_norm = np.vdot(bra_vec, bra_vec)
    ket_norm = np.vdot(ket_vec, ket_vec)
    if abs(bra_norm) < 1e-30 or abs(ket_norm) < 1e-30:
        return 0.0 + 0.0j

    numerator = np.vdot(bra_vec, op_mat @ ket_vec)
    return complex(numerator / np.sqrt(bra_norm * ket_norm))


def exact_expectation_value(
    nqs,
    operator,
    *,
    cache: Optional[ExactSummationCache] = None,
) -> complex:
    if cache is None:
        cache = build_exact_sum_cache(nqs)

    psi_vec = exact_wavefunction_vector(nqs, basis_states=cache.basis_states)
    psi_norm = np.vdot(psi_vec, psi_vec)
    if abs(psi_norm) < 1e-30:
        return 0.0 + 0.0j

    op_mat = exact_operator_matrix(nqs, operator, cache=cache)
    return complex(np.vdot(psi_vec, op_mat @ psi_vec) / psi_norm)

def exact_probe_correlator_from_model(
    nqs,
    times_arr: np.ndarray,
    *,
    bra_probe_operator,
    ket_probe_operator,
    protocol: str,
    reference_energy: Optional[float] = None,
    cache: Optional[ExactSummationCache] = None,
) -> np.ndarray:
    """
    Compute probe correlators by exact unitary evolution on tiny systems.
    """
    model = getattr(nqs, "model", None)
    if model is None or not hasattr(model, "diagonalize"):
        raise ValueError("Exact probe correlators require a model exposing diagonalize().")

    from QES.Algebra.Properties.time_evo import time_evo_block

    if cache is None:
        cache = build_exact_sum_cache(nqs)

    psi0 = exact_wavefunction_vector(nqs, basis_states=cache.basis_states)
    norm0 = np.linalg.norm(psi0)
    if norm0 <= 0.0:
        raise ValueError("Exact probe correlator received a zero-norm NQS state.")
    psi0 = psi0 / norm0

    bra_mat = exact_operator_matrix(nqs, bra_probe_operator, cache=cache)
    ket_mat = exact_operator_matrix(nqs, ket_probe_operator, cache=cache)
    phi_bra0 = bra_mat @ psi0
    phi_ket0 = ket_mat @ psi0

    model.diagonalize()
    eig_vec = np.asarray(getattr(model, "_eig_vec"), dtype=np.complex128)
    eig_val = np.asarray(getattr(model, "_eig_val"), dtype=np.complex128)

    t_rel = np.asarray(times_arr, dtype=np.float64) - float(times_arr[0])
    bra_overlaps = eig_vec.conj().T @ phi_bra0
    ket_overlaps = eig_vec.conj().T @ phi_ket0

    if protocol == "two_sided":
        half_times = 0.5 * t_rel
        phi_minus = time_evo_block(eig_vec, eig_val, bra_overlaps, -half_times)
        phi_plus = time_evo_block(eig_vec, eig_val, ket_overlaps, half_times)
        correlator = np.einsum("it,it->t", np.conj(phi_minus), phi_plus)
    else:
        phi_t = time_evo_block(eig_vec, eig_val, ket_overlaps, t_rel)
        correlator = np.einsum("i,it->t", np.conj(phi_bra0), phi_t)

    correlator = np.asarray(correlator, dtype=np.complex128)
    if reference_energy is not None:
        correlator = correlator * np.exp(1.0j * float(reference_energy) * t_rel)
    return correlator


def exact_probe_weight(
    nqs,
    probe_operator,
    *,
    weight_operator=None,
    cache: Optional[ExactSummationCache] = None,
) -> complex:
    r"""Evaluate the normalized probe weight exactly over the full basis."""
    if weight_operator is None:
        if cache is None:
            cache = build_exact_sum_cache(nqs)
        probe_matrix = exact_operator_matrix(nqs, probe_operator, cache=cache)
        return exact_expectation_value(nqs, probe_matrix.conj().T @ probe_matrix, cache=cache)
    return exact_expectation_value(nqs, weight_operator, cache=cache)


def resolve_transition_kernel(operator):
    if hasattr(operator, "jax"):
        return operator.jax
    if callable(operator):
        return operator
    raise ValueError("Operator must be callable or expose a .jax transition kernel.")


__all__ = [
    "ExactSummationCache",
    "basis_state_key",
    "build_exact_sum_cache",
    "enumerate_basis_states",
    "exact_expectation_value",
    "exact_operator_matrix",
    "exact_operator_matrix_from_kernel",
    "exact_probe_correlator_from_model",
    "exact_probe_weight",
    "exact_transition_element",
    "exact_wavefunction_vector",
    "resolve_transition_kernel",
]
