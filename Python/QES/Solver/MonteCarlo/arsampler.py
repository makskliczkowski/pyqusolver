'''
This module provides an implementation of Autoregressive samplers for quantum states.
Autoregressive samplers generate independent samples directly from the probability
distribution defined by an autoregressive neural network.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.11.2025
Description     : Autoregressive Sampler for quantum states using JAX.
-------------------------------------------------------
'''

from typing import Optional, Tuple, Any

try:
    from QES.Solver.MonteCarlo.sampler import Sampler, JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required to use ARSampler.")
    
    import jax
    import jax.numpy as jnp
    import jax.nn as nn
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
    def __init__(self, 
                net, 
                shape       : Tuple[int, ...], 
                rng_k       : jax.random.PRNGKey,
                *,
                dtype       : Any       = jnp.complex128,
                **kwargs):
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
        super().__init__(shape          =   shape, 
                        upd_fun         =   None, 
                        rng_k           =   rng_k, 
                        numchains       =   kwargs.get('numchains', 1),
                        numsamples      =   kwargs.get('numsamples', 1), 
                        backend         =   'jax', 
                        dtype           =   dtype,
                        **kwargs)
        
        # Ensure we have the apply function
        self._net                       = net
        if hasattr(net, 'apply'):       self._net_apply = net.apply
        elif hasattr(net, 'get_apply'): self._net_apply = net.get_apply()[0]
        else:                           self._net_apply = net # Callable
        
        # Pre-compile the sampling kernel
        self._dtype                     = dtype
        self._sample_jit                = jax.jit(self._static_sample_ar, static_argnames=['net_apply', 'shape', 'total_count', 'statetype'])
        self._mu                        = kwargs.get('mu', 2.0)  # Scaling factor for log-prob to log-psi

    @staticmethod
    def _static_sample_ar(net_apply, params, rng_key, shape, total_count, statetype: Any = jnp.float32, mu: float = 2.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-compiled sequential autoregressive sampling.
        Complexity: O(N_sites) network calls.
        
        #!TODO: Adjust for general local dimension (now assumes binary 0/1 spins)
        1. Initialize empty configurations and log-prob accumulators.
        2. For each site i in the system:
            a. Evaluate the network on the current partial configurations.
            b. Sample the new spin s_i from the conditional distribution.
            c. Update the configuration and log-prob accumulators.
        """
        
        # Number of degrees of freedom (e.g., sites)
        N = shape[0]
        
        # We start with zeros (or arbitrary, masked inputs ignore them anyway)
        # Shape: (Total_Samples, N_sites)
        canvas          = jnp.zeros((total_count, N), dtype=jnp.float32)        # Inputs are 0/1
        log_prob_sum    = jnp.zeros((total_count,), dtype=jnp.float32)          # Log prob accumulator
        
        # Sequential Scan Loop
        # We fill the lattice site by site: 0 -> 1 -> ... -> N-1
        def scan_body(carry, site_idx):
            curr_configs, curr_log_p, key   = carry
            key, subkey                     = jax.random.split(key)
            
            # Evaluate Network on current partial configurations
            # The MADE network outputs logits for ALL sites in parallel,
            # but site `i` output depends only on inputs `0...i-1`.
            # logits shape: (Total_Samples, N_sites)
            logits                          = net_apply(params, curr_configs)
            
            # Select logit for the CURRENT site we are filling
            # Shape: (Total_Samples,)
            logit_i                         = logits[:, site_idx]
            
            # Sample new spin (Bernoulli / Categorical)
            # prob(s_i=1) = sigmoid(logit_i)
            prob_1                          = nn.sigmoid(logit_i)
            
            # Sample 0 or 1
            new_degrees                     = jax.random.bernoulli(subkey, prob_1).astype(statetype)
            
            # Update Configuration
            new_configs                     = curr_configs.at[:, site_idx].set(new_degrees)
            
            # Accumulate Log Prob
            # log_p(s_i) = s_i * log(p) + (1-s_i) * log(1-p)
            # Numerical stable: 
            # log_p(1) = -softplus(-x), log_p(0) = -softplus(x) 
            # This comes from log(sigmoid(x)) and log(1-sigmoid(x))
            log_p_i                         = jnp.where(new_degrees > 0.5, -nn.softplus(-logit_i), -nn.softplus(logit_i))
            new_log_p_sum                   = curr_log_p + log_p_i
            
            return (new_configs, new_log_p_sum, key), None

        # Run Scan
        init_val                                = (canvas, log_prob_sum, rng_key)
        (final_configs, final_log_psi, _), _    = jax.lax.scan(scan_body, init_val, jnp.arange(N))

        # Get Phases if needed
        phases                                  = net_apply(params, final_configs, method=lambda n, x: n.get_phase(x))
        # if dtype is real, phases will be zero
        final_log_psi                           = final_log_psi / mu + 1j * phases
        
        # Convert 0/1 back to physical values one wants (e.g., -1/+1)
        #!TODO: General local dimension handling (if needed)
        final_configs_phys                      = 2 * final_configs - 1 
        
        return final_configs_phys, final_log_psi

    # ---------------------------------------------------------
    # Public Sampling Method
    # ---------------------------------------------------------

    def sample(self, parameters=None, num_samples=None, num_chains=None):
        
        # Setup counts
        n_c             = num_chains if num_chains is not None else self._numchains
        n_s             = num_samples if num_samples is not None else self._numsamples
        total_samples   = n_c * n_s
        
        if parameters is None: 
            # Fetch parameters from network logic
            parameters  = self._net.get_params()

        # Run JIT Kernel
        self._rng_k, key_sample = jax.random.split(self._rng_k)
        configs, log_psi        = self._sample_jit(
                                    self._net_apply, 
                                    parameters, 
                                    key_sample, 
                                    self._shape, 
                                    total_samples,
                                    self._statetype
                                )
        
        # Reshape to (Chains, Samples, N)
        configs_reshaped    = configs.reshape((n_c, n_s) + self._shape)
        
        # Handle Amplitudes
        # AR gives P(s) = |psi(s)|^2. 
        # log_psi (complex) = 0.5 * log(P(s)) + i * phase(s)
        # The AR network usually only models the Amplitude.
        log_psi_reshaped    = log_psi.reshape((n_c, n_s))
        
        # For now, we return 0.5 * log_P as the real part of log_psi.
        log_psi_reshaped    =  log_psi.reshape((n_c, n_s)) / self._mu
        
        # Weights
        # Since we sampled EXACTLY from P(s) = |psi|^2, importance weights are all 1.0
        weights             = jnp.ones((n_c, n_s))
        
        return (None, None), (configs_reshaped, log_psi_reshaped), weights
    
# ---------------------------------------------------------
# End of AR Sampler
# ---------------------------------------------------------