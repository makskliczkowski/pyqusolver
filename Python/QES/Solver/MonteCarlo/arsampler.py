"""
This module provides an implementation of Autoregressive samplers for quantum states.
Autoregressive samplers generate independent samples directly from the probability
distribution defined by an autoregressive neural network.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.11.2025
Description     : Autoregressive Sampler for quantum states using JAX.
-------------------------------------------------------
"""

from typing import Any, Callable, Optional, Tuple

try:
    from QES.Solver.MonteCarlo.sampler import JAX_AVAILABLE, Sampler, resolve_state_defaults

    if not JAX_AVAILABLE:
        raise ImportError("JAX is required to use ARSampler.")

    import jax
    import jax.nn as nn
    import jax.numpy as jnp
except ImportError:
    raise ImportError("QES package is required to use ARSampler.")

# ---------------------------------------------------------
# Autoregressive Sampler
# ---------------------------------------------------------


class ARSampler(Sampler):
    """
    Autoregressive Sampler.
    Generates independent samples directly from the probability distribution
    defined by the network P(s).

    Uses an autoregressive neural network to sample configurations site by site.
    Each conditional probability p(s_i | s_1,...,s_{i-1}) is modeled by the network (provided during init).
    """

    def __init__(
        self,
        net,
        shape: Tuple[int, ...],
        rng_k: jax.random.PRNGKey,
        *,
        dtype: Any = jnp.complex128,
        upd_fun: Optional[Callable] = None,
        backend: str = "jax",
        numchains: int = 1,
        numsamples: int = 1,
        **kwargs,
    ):
        """
        Initialize the Autoregressive Sampler.
        Parameters:
        -----------
        net         : Callable
            The autoregressive neural network. Must have a method to compute log-probabilities.
            Expected to have a signature: net.apply(params, configs) -> logits
        shape       : Tuple[int, ...]
            Shape of a single configuration (e.g., (N_sites,)).
        rng_k       : jax.random.PRNGKey
            Random number generator key for JAX.
        dtype       : Any, optional
            Data type for computations (default: jnp.complex128).
        kwargs      : Additional keyword arguments.
        """

        # AR doesn't need therm/sweep steps
        super().__init__(
            shape=shape,
            rng_k=rng_k,
            numchains=numchains,
            numsamples=numsamples,
            backend=backend,
            dtype=dtype,
            upd_fun=upd_fun,
            **kwargs,
        )

        # Ensure we have the apply function
        self._net = net
        self._network_adapter = kwargs.get("network_adapter", None)

        if hasattr(net, "_flax_module"):
            self._net_apply = net._flax_module.apply
        elif hasattr(net, "net_flax") and net.net_flax is not None:
            self._net_apply = net.net_flax.apply
        elif hasattr(net, "apply"):
            self._net_apply = net.apply
        else:
            self._net_apply = net  # Assuming it is a raw callable

        self._logits_method = None
        self._phase_method = None
        if self._network_adapter is not None and hasattr(self._network_adapter, "resolve_sampling_hooks"):
            try:
                hooks = self._network_adapter.resolve_sampling_hooks()
                self._logits_method = hooks.get("logits_method", None)
                self._phase_method = hooks.get("phase_method", None)
            except Exception:
                pass
        if self._logits_method is None:
            self._logits_method = lambda n, x: n.get_logits(x, is_binary=True)
        if self._phase_method is None:
            self._phase_method = lambda n, x: n.get_phase(x, is_binary=True)

        self._state_representation = kwargs.get("state_representation", getattr(self, "_state_representation", None))

        try:
            import QES.general_python.common.binary as Binary
            self._spin, self._mode_repr = resolve_state_defaults(
                self._state_representation,
                spin=kwargs.get("spin", None),
                mode_repr=kwargs.get("mode_repr", None),
                fallback_mode_repr=Binary.BACKEND_REPR,
            )
        except ImportError:
            self._spin, self._mode_repr = resolve_state_defaults(
                self._state_representation,
                spin=kwargs.get("spin", None),
                mode_repr=kwargs.get("mode_repr", None),
                fallback_mode_repr=0.5,
            )

        # Pre-compile the sampling kernel
        self._dtype = dtype
        self._mu = kwargs.get("mu", 2.0)  # Scaling factor for log-prob to log-psi
        self._local_dim = kwargs.get("local_dim", 2)
        self._sample_jit = jax.jit(
            self._static_sample_ar,
            static_argnames=[
                "net_apply",
                "logits_method",
                "phase_method",
                "shape",
                "total_count",
                "statetype",
                "spin",
                "local_dim",
                "state_representation",
            ],
        )
        self._name = "AR"

    @staticmethod
    def _static_sample_ar(
        net_apply,
        logits_method,
        phase_method,
        params,
        rng_key,
        shape,
        total_count,
        statetype: Any = jnp.float32,
        mu: float = 2.0,
        spin: bool = True,
        mode_repr: float = 1.0,
        local_dim: int = 2,
        state_representation: Optional[str] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-compiled sequential autoregressive sampling.
        Complexity: O(N_sites) network calls.

        1. Initialize empty configurations and log-prob accumulators.
        2. For each site i in the system:
            a. Evaluate the network on the current partial configurations.
            b. Sample the new state s_i from the conditional distribution.
            c. Update the configuration and log-prob accumulators.
        """

        N = shape[0]
        canvas = jnp.zeros((total_count, N), dtype=statetype)
        log_prob_sum = jnp.zeros((total_count,), dtype=jnp.float64)

        def scan_body(carry, site_idx):
            curr_configs, curr_log_p, key = carry
            key, subkey = jax.random.split(key)
            variables = {"params": params}

            # 1. Evaluate Network to get logits for current site
            logits = net_apply(
                variables,
                curr_configs,
                method=logits_method,
            )

            # Select logit for the CURRENT site we are filling
            if local_dim == 2 and logits.ndim == 2:
                # Shape: (Total_Samples,)
                logit_i = jnp.real(logits[:, site_idx])

                # 2. Sample (Bernoulli)
                # prob(s_i=1) = sigmoid(logit_i)
                prob_1 = nn.sigmoid(logit_i)
                new_degrees = jax.random.bernoulli(subkey, prob_1).astype(statetype)

                # 3. Update Config & Log Prob
                new_configs = curr_configs.at[:, site_idx].set(new_degrees)

                # log_p(1) = -softplus(-x), log_p(0) = -softplus(x)
                log_p_i = jnp.where(new_degrees > 0.5, -nn.softplus(-logit_i), -nn.softplus(logit_i))
            else:
                # General case: logits shape is (Total_Samples, N, local_dim)
                # Select logits for current site: (Total_Samples, local_dim)
                logit_i = jnp.real(logits[:, site_idx, :])

                # 2. Sample (Categorical)
                new_degrees = jax.random.categorical(subkey, logit_i).astype(statetype)

                # 3. Update Config & Log Prob
                new_configs = curr_configs.at[:, site_idx].set(new_degrees)

                # Compute log probability of selected classes
                log_probs = nn.log_softmax(logit_i, axis=-1)
                log_p_i = jnp.take_along_axis(log_probs, new_degrees[:, None].astype(jnp.int32), axis=-1).squeeze(-1)

            return (new_configs, curr_log_p + log_p_i, key), None

        # Run Scan over all sites
        init_val = (canvas, log_prob_sum, rng_key)
        (final_configs, final_log_psi, _), _ = jax.lax.scan(scan_body, init_val, jnp.arange(N))

        # Get Phases (Call get_phase method)
        # NOTE: final_configs is in binary (0,1) format here
        variables = {"params": params}
        phases = net_apply(
            variables,
            final_configs,
            method=phase_method,
        )

        # Combine Amplitude (log_prob / mu) and Phase
        final_log_psi = final_log_psi / mu + 1j * phases

        # Convert to requested output representation only once at the boundary.
        if local_dim == 2:
            if state_representation in {"binary_01", "occupation_binary"}:
                final_configs_phys = final_configs
            elif spin:
                final_configs_phys = (final_configs * 2.0 - 1.0) * mode_repr
            else:
                final_configs_phys = final_configs * mode_repr
        else:
            # For general dimension, we assume integer format is required.
            final_configs_phys = final_configs

        return final_configs_phys, final_log_psi

    # ---------------------------------------------------------
    # Public Sampling Method
    # ---------------------------------------------------------

    def sample(self, parameters=None, num_samples=None, num_chains=None):

        # Setup counts
        n_c = num_chains if num_chains is not None else self._numchains
        n_s = num_samples if num_samples is not None else self._numsamples
        total_samples = n_c * n_s

        if parameters is None:
            parameters = self._net.get_params()

        # Run JIT Kernel
        self._rng_k, key_sample = jax.random.split(self._rng_k)
        configs, log_psi = self._sample_jit(
            self._net_apply,
            self._logits_method,
            self._phase_method,
            parameters,
            key_sample,
            self._shape,
            total_samples,
            self._statetype,
            self._mu,
            self._spin,
            self._mode_repr,
            self._local_dim,
            self._state_representation,
        )

        # Reshape to (Chains, Samples, N)
        configs_reshaped = configs.reshape(
            (n_c, n_s) + self._shape
        )

        # Handle Amplitudes
        # AR gives P(s) = |psi(s)|^2.
        # log_psi (complex) = 0.5 * log(P(s)) + i * phase(s)
        # The AR network usually only models the Amplitude.
        configs_reshaped = configs.reshape((-1,) + self._shape)  # Flatten configs
        log_psi_reshaped = log_psi.reshape((-1,))  # Flatten log_psi too

        # Importance weights are 1.0
        weights = jnp.ones(total_samples)

        return (None, None), (configs_reshaped, log_psi_reshaped), weights

    # ---------------------------------------------------------
    # ! ABSTRACT METHOD IMPLEMENTATIONS
    # ---------------------------------------------------------

    def get_sampler(self):
        """Returns the JIT-compiled sampling kernel (Satisfies ABC)."""
        return self._sample_jit

    def _get_sampler_jax(self):
        """Returns the JAX sampling kernel (Satisfies ABC)."""
        return self._sample_jit

    def _get_sampler_np(self):
        """Numpy implementation not supported for ARSampler."""
        raise NotImplementedError("Autoregressive sampling is JAX-only.")

    def diagnose(self, samples=None) -> dict:
        """
        AR sampling produces independent samples by construction.
        Diagnostics are trivial.
        """
        metrics = {
            "ess": self.numsamples * self.numchains,
            "tau": 1.0,
            "r_hat": 1.0,
            "acceptance_rate": 1.0,
        }
        return metrics


# ---------------------------------------------------------
# End of AR Sampler
# ---------------------------------------------------------
