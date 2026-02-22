"""
NumPy backend implementation for VMCSampler.

This module isolates NumPy-only sampling paths from ``vmc.py`` to keep the
main sampler focused on backend-agnostic orchestration and JAX kernels.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
import numpy as np

try:
    from QES.general_python.algebra.utils import DEFAULT_NP_INT_TYPE
except ImportError:
    DEFAULT_NP_INT_TYPE = np.int32

if TYPE_CHECKING:
    from .vmc import VMCSampler

def acceptance_probability_np(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
    log_acceptance_ratio = beta * mu * np.real(candidate_val - current_val)
    return np.exp(np.minimum(log_acceptance_ratio, 30.0))

def logprob_np(x, net_callable, net_params=None):
    return np.asarray([net_callable(net_params, y) for y in x]).reshape(x.shape[0])

def run_mcmc_steps_np(
    chain,
    logprobas,
    num_proposed,
    num_accepted,
    params,
    rng,
    steps,
    mu,
    beta,
    update_proposer,
    log_proba_fun,
    accept_config_fun,
    net_callable_fun,
):
    current_val = (
        logprobas
        if logprobas is not None
        else log_proba_fun(chain, net_callable=net_callable_fun, net_params=params)
    )
    n_chains = chain.shape[0] if chain.ndim > 1 else 1

    for _ in range(int(steps)):
        new_chain = update_proposer(chain, rng)
        new_logprobas = log_proba_fun(
            new_chain, net_callable=net_callable_fun, net_params=params
        )
        acc_probability = accept_config_fun(current_val, new_logprobas, beta, mu)

        rand_vals = rng.random(size=n_chains)
        accepted = rand_vals < acc_probability
        num_proposed += 1
        num_accepted[accepted] += 1

        current_val = np.where(accepted, new_logprobas, current_val)
        if chain.ndim == 1:
            chain = new_chain if accepted[0] else chain
        else:
            mask = accepted.reshape((n_chains,) + (1,) * (chain.ndim - 1))
            chain = np.where(mask, new_chain, chain)

    return chain, current_val, num_proposed, num_accepted

def generate_samples_np(sampler: "VMCSampler", params: Any, num_samples: int, multiple_of: int = 1):
    del multiple_of  # Reserved for compatibility.

    states, logprobas, num_proposed, num_accepted = run_mcmc_steps_np(
        chain=sampler._states,
        logprobas=sampler._logprobas,
        num_proposed=sampler._num_proposed,
        num_accepted=sampler._num_accepted,
        params=params,
        update_proposer=sampler._upd_fun,
        log_proba_fun=sampler.logprob,
        accept_config_fun=sampler.acceptance_probability,
        net_callable_fun=sampler._net_callable,
        steps=sampler._therm_steps * sampler._sweep_steps,
        rng=sampler._rng,
        mu=sampler._mu,
        beta=sampler._beta,
    )

    configs = np.empty((num_samples,) + states.shape, dtype=states.dtype)
    for i in range(num_samples):
        states, logprobas, num_proposed, num_accepted = run_mcmc_steps_np(
            chain=states,
            logprobas=logprobas,
            num_proposed=num_proposed,
            num_accepted=num_accepted,
            params=params,
            update_proposer=sampler._upd_fun,
            log_proba_fun=sampler.logprob,
            accept_config_fun=sampler.acceptance_probability,
            net_callable_fun=sampler._net_callable,
            steps=sampler._sweep_steps,
            rng=sampler._rng,
            mu=sampler._mu,
            beta=sampler._beta,
        )
        configs[i] = states

    meta = (states.copy(), np.asarray(logprobas).copy(), num_proposed, num_accepted)
    return meta, configs


def sample_np(
    sampler: "VMCSampler",
    *,
    current_params: Any,
    used_num_samples: int,
    net_callable: Any,
):
    sampler._logprobas = sampler.logprob(
        sampler._states, net_callable=net_callable, net_params=current_params
    )

    (
        sampler._states,
        sampler._logprobas,
        sampler._num_proposed,
        sampler._num_accepted,
    ), configs = generate_samples_np(sampler, current_params, used_num_samples)

    configs = configs.reshape(-1, *configs.shape[2:])
    configs_log_ansatz = sampler.logprob(
        configs, net_callable=net_callable, net_params=current_params
    )
    probs = np.exp((1.0 / sampler._logprob_fact - sampler._mu) * np.real(configs_log_ansatz))
    probs = probs.reshape(-1)
    norm = max(float(np.sum(probs)), 1e-12)
    probs = probs / norm * probs.size

    return (
        (sampler._states, sampler._logprobas),
        (configs, configs_log_ansatz.reshape(-1)),
        probs.reshape(-1),
    )


def get_sampler_np(
    sampler: "VMCSampler",
    num_samples: Optional[int] = None,
    num_chains: Optional[int] = None,
):
    if sampler._isjax:
        raise RuntimeError("NumPy sampler getter only available for NumPy backend.")

    static_num_samples = num_samples if num_samples is not None else sampler._numsamples
    static_num_chains = num_chains if num_chains is not None else sampler._numchains

    def wrapper(
        states_init,
        rng_k_init,
        param,
        num_proposed_init=None,
        num_accepted_init=None,
    ):
        sampler._states = states_init
        sampler._rng_k = rng_k_init
        sampler._num_proposed = (
            num_proposed_init
            if num_proposed_init is not None
            else np.zeros(static_num_chains, dtype=DEFAULT_NP_INT_TYPE)
        )
        sampler._num_accepted = (
            num_accepted_init
            if num_accepted_init is not None
            else np.zeros(static_num_chains, dtype=DEFAULT_NP_INT_TYPE)
        )
        return sampler.sample(
            parameters=param,
            num_samples=static_num_samples,
            num_chains=static_num_chains,
        )

    return wrapper

# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------