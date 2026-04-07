"""
Equal-time and time-displaced measurement helpers for DQMC.

The solver should delegate observable construction here so that update logic and
physics estimators stay separate.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import jax.numpy as jnp


def ensure_chain_axis(green):
    """
    Return a Green's-function array with an explicit leading chain axis.

    Parameters
    ----------
    green : array
        Green's-function tensor with shape ``(N, N)`` or ``(n_chains, N, N)``.

    Returns
    -------
    array
        Tensor with shape ``(n_chains, N, N)``.
    """
    green = jnp.asarray(green)
    if green.ndim == 2:
        return green[None, :, :]
    return green


def measure_equal_time(model, greens: Sequence[Any], kinetic_matrix) -> Dict[str, float]:
    """
    Measure equal-time observables from channel Green's functions.

    Parameters
    ----------
    model :
        `QES.pydqmc.dqmc_model.DQMCModel` instance or any object exposing
        `measure_equal_time(...)`.
    greens :
        Sequence of channel Green's functions.  Each entry is expected to have
        shape `(n_chains, N, N)` or `(N, N)`.
    kinetic_matrix :
        Single-particle kinetic matrix `K` entering the hopping energy.

    Returns
    -------
    dict
        Model-specific equal-time estimators.  When the model does not provide
        its own measurement routine, the fallback estimator returns the mean
        density

        ``n = sum_c <1 - G_c(ii)>``.
    """
    if hasattr(model, "measure_equal_time"):
        return model.measure_equal_time(greens, kinetic_matrix)

    density = 0.0
    for green in greens:
        green = ensure_chain_axis(green)
        density += jnp.mean(1.0 - jnp.diagonal(green, axis1=-2, axis2=-1))
    return {"density": float(density)}


def measure_equal_time_by_chain(model, greens: Sequence[Any], kinetic_matrix) -> Dict[str, Any]:
    """
    Measure equal-time observables without reducing over the chain axis.

    Returns per-chain scalar estimators so solver code can perform explicit
    sign/phase reweighting before constructing final averages.
    """
    if hasattr(model, "measure_equal_time_by_chain"):
        return model.measure_equal_time_by_chain(greens, kinetic_matrix)

    density = 0.0
    for green in greens:
        green = ensure_chain_axis(green)
        density += jnp.mean(1.0 - jnp.diagonal(green, axis1=-2, axis2=-1), axis=-1)
    return {"density": density}


def reweight_observables(observables_by_chain: Mapping[str, Any], weights) -> Dict[str, float]:
    """
    Reweight per-chain observables with sign/phase factors sampled on `|W|`.

    For real-sign workflows this reduces to

        <O> = sum_i s_i O_i / sum_i s_i.
    """
    weights = jnp.asarray(weights)
    denom = jnp.sum(weights)
    if jnp.abs(denom) < 1e-14:
        return {str(name): float("nan") for name in observables_by_chain}

    out: Dict[str, float] = {}
    for name, values in observables_by_chain.items():
        vals = jnp.asarray(values)
        numer = jnp.sum(weights * vals)
        out[str(name)] = float(jnp.real(numer / denom))
    return out


def mean_observables(observables_by_chain: Mapping[str, Any]) -> Dict[str, float]:
    """Return unweighted chain means for a per-chain observable dictionary."""
    out: Dict[str, float] = {}
    for name, values in observables_by_chain.items():
        out[str(name)] = float(jnp.mean(jnp.real(jnp.asarray(values))))
    return out


def measure_time_displaced(unequal_greens, weights=None) -> Any:
    """
    Average time-displaced Green's functions over chains.

    Math:
        the sampler stores the unequal-time estimator ``G_c(\tau, 0)`` for each
        chain and channel.  The natural Monte Carlo estimator is therefore the
        chain average at fixed imaginary-time separation.
    """
    if unequal_greens is None:
        return None
    if weights is None:
        return jnp.mean(unequal_greens, axis=0)
    weights = jnp.asarray(weights)
    denom = jnp.sum(weights)
    if jnp.abs(denom) < 1e-14:
        return None
    view_shape = (weights.shape[0],) + (1,) * (jnp.asarray(unequal_greens).ndim - 1)
    return jnp.sum(jnp.asarray(unequal_greens) * weights.reshape(view_shape), axis=0) / denom
