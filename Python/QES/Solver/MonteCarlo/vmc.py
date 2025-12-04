'''
Variational Monte Carlo Sampler Module

Extends the generic Sampler class to implement a Variational Monte Carlo (VMC)
sampler for quantum states represented by neural networks or other parameterized functions
(ansatze). Supports both JAX and NumPy backends for efficient sampling.

Example
>>> from QES.Solver.MonteCarlo.vmc import VMCSampler
>>> sampler = VMCSampler(
>>>     net             =   my_network,
>>>     shape           =   (4, 4), # Lattice shape
>>>     rng             =   np.random.default_rng(seed=42),
>>>     mu              =   2.0,    # Exponent for Born rule sampling
>>>     beta            =   1.0,    # Inverse temperature (also for parallel tempering)
>>>     therm_steps     =   100,    # 100 thermalization sweeps
>>>     sweep_steps     =   10,     # 10 MCMC updates per site
>>>     numsamples      =   1000,   # 1000 samples per chain
>>>     numchains       =   4,      # 4 parallel chains
>>> )

---------------------------------
File            : Python/QES/Solver/MonteCarlo/vmc.py
Author          : Maksymilian Kliczkowski
License         : MIT
---------------------------------
'''

#######################################################################    

import numpy as np
import numba
from typing import Tuple, Callable, Union, Optional, Any, TYPE_CHECKING
from functools import partial
from enum import Enum, auto, unique

try:
    from .sampler import (Sampler, SamplerErrors, SolverInitState,
        _propose_random_flip_jax, _propose_random_flip_np, 
        _state_distinguish, 
        _propose_random_flips_np, _propose_random_flips_jax,
    )
except ImportError as e:
    raise ImportError("Failed to import base Sampler class from MonteCarlo module.") from e

# flax for the network
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn    
except ImportError as e:
    raise ImportError("Failed to import flax module. Ensure flax is correctly installed.") from e

# from algebra
try:
    if TYPE_CHECKING:
        from QES.Algebra.hilbert import HilbertSpace
    
    from QES.general_python.algebra.utils import JAX_AVAILABLE, get_backend, DEFAULT_JP_INT_TYPE, DEFAULT_BACKEND_KEY, Array
    from QES.general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, distinguish_type
    from QES.general_python.algebra.ran_wrapper import choice, randint_np, randint_jax
except ImportError as e:
    raise ImportError("Failed to import general_python modules. Ensure QES package is correctly installed.") from e

# ---------------------------------------------------------------------------
#! Variational Monte Carlo Sampler
# ---------------------------------------------------------------------------

class VMCSampler(Sampler):
    r"""
    Variational Markov Chain Monte Carlo sampler for quantum states.

    Implements MCMC sampling from the Hilbert space based on a probability distribution
    derived from the quantum state amplitudes :math:`\psi(s)`. The target distribution is typically
    proportional to :
    
        math:`|\psi(s)|^{\mu}`, where :math:`\mu` controls the sampling bias.

    The standard Born rule distribution
        
        :math:`p(s) = |\psi(s)|^2 / \sum_{s'} |\psi(s')|^2`
    
    
    corresponds to
    
        :math:`\mu=2`. Values :math:`0 \le \mu < 2`
        
    can be used for importance sampling techniques (see [arXiv:2108.08631](https://arxiv.org/abs/2108.08631)).
    Note:
        Is there an error in the paper, where they introduce \mu?

    The sampling process uses the Metropolis-Hastings algorithm:
        1. **Initialization:**
            Set up with system shape, network/callable for
                :math:`\log\psi(s)`,
            proposal function, MCMC parameters 
                (:math:`\mu`, :math:`\beta`, steps).
        2. **Network Callable:**
            Obtain the function to compute :math:`\log\psi(s)` and its parameters.
        3. **Thermalization:** 
            Run MCMC chains for `therm_steps * sweep_steps` updates to reach equilibrium.
        4. **Sampling:**
            Run MCMC chains for `num_samples * sweep_steps` updates, collecting configurations.
        5. **Output:**
            Return final chain states, collected samples, log-ansatz values, and importance weights.

    Supports JAX backend for performance via JIT compilation of core loops.
    """
    
    def __init__(self,
                net,
                shape       : Tuple[int, ...],
                rng         : np.random.Generator           = None,
                rng_k                                       = None,
                upd_fun     : Optional[Callable]            = None,
                mu          : float                         = 2.0,
                beta        : float                         = 1.0,
                therm_steps : int                           = 100,
                sweep_steps : int                           = 100,
                seed                                        = None,
                hilbert     : 'HilbertSpace'                = None,
                numsamples  : int                           = 1,
                numchains   : int                           = 1,
                numupd      : int                           = 1,
                initstate                                   = None,
                backend     : str                           = 'default',
                logprob_fact: float                         = 0.5,
                statetype   : Union[np.dtype, jnp.dtype]    = np.float32,
                makediffer  : bool                          = True,
                **kwargs):
        r"""
        Initialize the MCSampler.

        Parameters:
            net (Callable or Flax Module):
                The network or function providing `log_psi(s)`. If a Flax module, assumes `net.apply` exists.
                If a callable, should have signature `log_psi = net(params, state)`.
            shape (Tuple[int, ...]):
                Shape of the system configuration (e.g., lattice dimensions).
            rng (np.random.Generator):
                NumPy random number generator.
            rng_k (Optional[jax.random.PRNGKey]):
                JAX random key.
            upd_fun (Optional[Callable]):
                State update proposal function. Defaults to single spin flip.
                Signature: `new_state = upd_fun(state, rng/rng_k)`
            mu (float):
                Exponent :math:`\mu` for the sampling distribution :math:`p(s) \propto |\psi(s)|^{\mu}`.
                Must be in range [0, 2]. Default is 2 (Born rule).
            beta (float):
                Inverse temperature factor :math:`\beta` for the Metropolis acceptance probability. Default is 1.0.
            therm_steps (int):
                Number of thermalization sweeps. Each sweep consists of `sweep_steps` MCMC updates per site (on average).
            sweep_steps (int):
                Number of MCMC updates per site within a single "sweep". Determines correlation between samples. Default is 1.
            seed (Optional[int]):
                Random seed for initialization if `rng`/`rng_k` not provided.
            hilbert (Optional[HilbertSpace]):
                Hilbert space object (optional).
            numsamples (int):
                Number of samples to collect per chain *after* thermalization.
            numchains (int):
                Number of parallel Markov chains.
            initstate (Array or str or SolverInitState or int):
                Initial state specification (see Sampler docs). Defaults to 'RND'.
            backend (str):
                Computational backend ('numpy', 'jax', 'default').
            logprob_fact (float):
                Factor used in calculating importance sampling weights, often :math:`1/\mu`.
                The probability is calculated as :math:`\exp((\frac{1}{\text{logprob_fact}} - \mu) \text{Re}(\log\psi(s)))`.
                Default 0.5 corresponds to Born rule :math:`|\psi|^2` when :math:`\mu=1`. Needs careful setting based on :math:`\mu`.
            statetype (Union[np.dtype, jnp.dtype]):
                Type of the state (e.g., np.float32, jnp.float32, np.int32).
            **kwargs:
                Additional arguments for the base Sampler class (e.g., `modes`, `mode_repr`).
        """
        
        super().__init__(shape, upd_fun, rng, rng_k, seed, hilbert,
                numsamples, numchains, initstate, backend, statetype, makediffer, **kwargs)
        
        if net is None:
            raise ValueError("A network (or callable) must be provided for evaluation.")

        # set the network
        self._net_callable, self._parameters    = self._set_net_callable(net)

        # set the parameters - this for modification of the distribution
        self._mu = mu
        if self._mu < 0.0 or self._mu > 2.0:
            raise ValueError(SamplerErrors.NOT_IN_RANGE_MU)
        
        self._beta                              = beta
        self._therm_steps                       = therm_steps
        self._sweep_steps                       = sweep_steps
        self._logprob_fact                      = logprob_fact
        self._logprobas                         = None
        self._total_therm_updates               = therm_steps * sweep_steps * self._size    # Total updates during thermalization
        self._total_sample_updates_per_sample   = sweep_steps * self._size                  # Updates between collected samples
        self._updates_per_sample                = self._sweep_steps                         # Steps between samples
        self._total_sample_updates_per_chain    = numsamples * self._updates_per_sample * self._numchains
        self._upd_fun                           = upd_fun
        
        # number of times a function is applied per update
        self._numupd                            = numupd
        
        if self._upd_fun is None:
            if self._isjax:
                # Bind RNG arguments to the JAX updater, already JIT-compiled.
                self._upd_fun = _propose_random_flip_jax
            else:
                # For NumPy backend, bind the RNG to the updater.
                self._upd_fun = _propose_random_flip_np
        
        # Wrap the update function if multiple updates per step are requested
        if self._numupd > 1:
            
            n_updates       = self._numupd
            base_update_fn  = self._upd_fun
            if self._isjax:
                # JAX: Use lax.scan for efficiency and split keys correctly
                def _multi_update_proposer(state, key):
                    # 1. Split the incoming key into 'n' unique sub-keys
                    keys = jax.random.split(key, n_updates)
                    
                    # 2. Define the loop body: (state, key) -> (new_state, None)
                    def body_fn(curr_state, sub_key):
                        return base_update_fn(curr_state, sub_key), None
                    
                    # 3. Run the loop efficiently on device
                    final_state, _ = jax.lax.scan(body_fn, state, keys)
                    return final_state
            else:
                def _multi_update_proposer(state, rng):
                    curr_state = state
                    for _ in range(n_updates):
                        curr_state = base_update_fn(curr_state, rng)
                    return curr_state

            self._upd_fun = _multi_update_proposer
        self._static_sample_fun                 = self.get_sampler(num_samples=self._numsamples, num_chains=self._numchains)
    
    #####################################################################
    
    def __repr__(self):
        """
        Provide a string representation of the MCSampler object.

        Returns:
            str: A formatted string containing the key attributes of the MCSampler instance,
        """
        return (f"MCSampler(shape={self._shape}, mu={self._mu}, beta={self._beta}, "
                f"therm_steps={self._therm_steps}, sweep_steps={self._sweep_steps}, "
                f"numsamples={self._numsamples}, numchains={self._numchains}, backend={self._backendstr})")

    def __str__(self):
        total_therm_updates_display     = self._total_therm_updates * self.size                     # Total updates per site
        total_sample_updates_display    = self._numsamples * self._updates_per_sample * self.size   # Total sample updates per site
        return (f"MCSampler:\n"
                f"  - State shape: {self._shape} (Size: {self.size})\n"
                f"  - Backend: {self._backendstr}\n"
                f"  - Chains: {self._numchains}, Samples/Chain: {self._numsamples}\n"
                f"  - Params: mu={self._mu:.3f}, beta={self._beta:.3f}, logprob_fact={self._logprob_fact:.3f}\n"
                f"  - Updates per sample: {self._numupd}\n"
                f"  - Thermalization: {self._therm_steps} sweeps x {self._sweep_steps} steps/sweep ({total_therm_updates_display} total site updates/chain)\n"
                f"  - Sampling: {self._updates_per_sample} steps/sample ({total_sample_updates_display} total site updates/chain)\n")
        
    ###################################################################
    
    def _set_net_callable(self, net):
        '''
        Set the network callable and extract parameters if applicable.

        Parameters:
            net (Callable or Flax Module):
                Input network object.

        Returns:
            Tuple[Callable, Any]: (network_callable, parameters)
        '''
        self._net = net
        
        # Check if it's a Flax Linen module (or similar with apply method)
        if isinstance(net, nn.Module):
            # Assume parameters are managed externally or within the module state
            # We need the apply function and the parameters.
            # This might require the user to pass parameters explicitly during sampling if they change.
            # Let's store the apply method and expect parameters at sample time.
            return net.apply, None # Parameters will be provided later
        elif hasattr(net, 'get_apply') and callable(net.get_apply):
            network_callable, parameters = net.get_apply()
            return network_callable, parameters
        elif hasattr(net, 'apply') and callable(net.apply):
            return net.apply, self.net.geta_params() if hasattr(net, 'geta_params') else self._parameters
        elif callable(net):
            return net, None # Assume no external parameters needed unless provided at sample time
        raise ValueError("Invalid network object provided. Needs to be callable or have an 'apply' method.")
    
    ###################################################################
    #! ACCEPTANCE PROBABILITY
    ###################################################################
    
    @staticmethod
    @jax.jit
    def _acceptance_probability_jax(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        r'''
        Calculate the Metropolis-Hastings acceptance probability using JAX.

        ---
        Calculates:
            :math:`\min(1, \exp(\beta \mu \cdot \text{Re}(\text{val}_{\text{cand}} - \text{val}_{\text{curr}})))`.
        ---
        Parameters:
            current_val (jnp.ndarray):
                Log-probability (or related value) of the current state(s).
            candidate_val (jnp.ndarray):
                Log-probability (or related value) of the candidate state(s).
            beta (float):
                Inverse temperature :math:`\beta`. Static argument for JIT.
            mu (float):
                Exponent :math:`\mu`. Static argument for JIT.

        Returns:
            jnp.ndarray: The acceptance probability(ies).
        '''
        delta   = jnp.real(candidate_val) - jnp.real(current_val)
        # ratio = jnp.exp(jnp.clip(beta * mu * delta, a_max=30.0)) # prevent overflow
        ratio   = jnp.exp(beta * mu * delta)
        return ratio
        log_acceptance_ratio = beta * mu * (jnp.real(candidate_val) - jnp.real(current_val))
        return jnp.exp(log_acceptance_ratio)
    
    @staticmethod
    @numba.njit
    def _acceptance_probability_np(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        '''
        Calculate the acceptance probability for the Metropolis-Hastings
        algorithm.
        Parameters:
            - current_val   : The value of the current state
            - candidate_val : The value of the candidate state
        
        Returns:
            - The acceptance probability as a float
        '''

        log_acceptance_ratio = beta * mu * np.real(candidate_val - current_val)
        return np.exp(log_acceptance_ratio)
    
    def acceptance_probability(self, current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        r'''
        Calculate the Metropolis-Hastings acceptance probability.

        Selects backend automatically. Uses instance `_beta` if `beta` is None.

        Parameters:
            current_val (array-like):
                Value (:math:`\log p(s)`) of the current state(s).
            candidate_val (array-like):
                Value (:math:`\log p(s')`) of the candidate state(s).
            beta (Optional[float]):
                Inverse temperature :math:`\beta`. Uses `self._beta` if None.
            mu (Optional[float]):
                Exponent :math:`\mu`. Uses `self._mu` if None.
        Returns:
            array-like: Acceptance probability(ies).
        '''
        use_beta = beta if beta is not None else self._beta

        if self._isjax:
            # We need beta to be static for the JIT version.
            # If beta can change dynamically, we cannot use the JIT version directly here.
            # Workaround:
            #   If beta is dynamic, call a non-jitted wrapper or re-jit if necessary.
            # Assuming beta is constant for a given sampling run for JIT benefits.
            
            if beta is not None and beta != self._beta:
                # If called with a different beta, might need a non-jitted path
                # or expect user to handle JIT compilation if beta changes often.
                # For simplicity, we assume the compiled version uses self._beta if beta is None.
                # If beta is provided *and different*, direct call might be slow if not JITted for that beta.
                # Let's use the JITted version with the provided beta if possible.
                # This requires careful handling of static arguments if beta changes frequently.
                # A simpler approach: always use self._beta in JITted functions called internally.
                # If external calls need different beta, they call this wrapper which might be slower.

                # Call JITted function with potentially non-static beta (will compile on first call with this beta)
                @partial(jax.jit, static_argnames=('beta',))
                def _accept_jax_dynamic_beta(cv, cdv, beta, mu):
                    log_acceptance_ratio = beta * jnp.real(cdv - cv) * mu
                    return jnp.exp(log_acceptance_ratio)
                return _accept_jax_dynamic_beta(current_val, candidate_val, use_beta, mu)
            else:
                return self._acceptance_probability_jax(current_val, candidate_val, beta=use_beta, mu=mu)
        else:
            return self._acceptance_probability_np(current_val, candidate_val, beta=use_beta, mu=mu)
    
    ###################################################################
    #! LOG PROBABILITY
    ###################################################################
    
    @staticmethod
    @partial(jax.jit, static_argnames=('net_callable',))
    def _logprob_jax(x, net_callable, net_params=None):
        r'''
        Calculate log probability :math:`\log \psi(s)` using JAX.
        
        This version is fully vectorized: it assumes that x is a batched
        input and applies the network callable via jax.vmap.
        
        Parameters:
            x (jnp.ndarray): Batched state configurations.
            net_callable (Callable): Function with signature: log_psi = net_callable(net_params, state)
            net_params (Any, optional): Parameters for the network callable.
        
        Returns:
            jnp.ndarray: The real part of log probabilities for the batch.
        '''
        def apply_net(s):
            return net_callable(net_params, s)
        batched_log_psi = jax.vmap(apply_net)(x)
        return jnp.real(batched_log_psi)
    
    @staticmethod
    @numba.njit
    def _logprob_np(x, net_callable, net_params = None):
        '''
        Calculate the log probability of a state using NumPy.
        Parameters:
            - x             : The state
            - net_callable  : The network callable (returns (\\log\\psi(s)))
            - net_params    : The network parameters - may be None but for flax callables
                            the parameters are necessary strictly!
        Returns:
            - The log probability as a float or complex number
        '''
        return np.array([net_callable(net_params, y) for y in x]).reshape(x.shape[0])
    
    def logprob(self, x, net_callable = None, net_params = None):
        r'''Calculate the log probability used for MCMC steps.

        Computes :math:`\text{Re}(\log \psi(s))`.
        Uses instance defaults if arguments are None. Selects backend automatically.

        Parameters:
            x (array-like):
                State configuration(s).
            net_callable (Optional[Callable]):
                Network callable. Uses `self._net_callable` if None.
            net_params (Any):
                Network parameters. Uses `self._parameters` if None.

        Returns:
            array-like: Log probabilities.
        '''
        
        use_callable    = net_callable if net_callable is not None else self._net_callable
        use_params      = net_params if net_params is not None else self._parameters
        
        if use_callable is None:
            raise ValueError("No callable provided for log probability calculation.")
    
        # if isinstance(self._net, nn.Module) and use_params is None:
        #     # Try to get params from the network instance if possible
        #     if hasattr(self._net, 'params'):
        #         use_params = self._net.params
        #     elif hasattr(self._net, 'get_params') and callable(self._net.get_params):
        #         _, use_params = self._net.get_params()
        #     else:
        #         # Cannot automatically get params, raise error or warning
        #         raise ValueError("Network seems to require parameters, but none were provided or found.")
            
        if self._isjax:
            return VMCSampler._logprob_jax(x, use_callable, use_params)
        return VMCSampler._logprob_np(x, use_callable, use_params)

    ###################################################################
    #! UPDATE CHAIN
    ###################################################################
    
    @staticmethod
    def _run_mcmc_steps_jax(chain_init,
                            current_val_init,
                            rng_k_init,
                            num_proposed_init,
                            num_accepted_init,
                            params,
                            # static args
                            steps               : int,
                            update_proposer     : Callable,
                            net_callable_fun    : Callable,
                            mu                  : float     = 2.0,
                            beta                : float     = 1.0):
        r'''
        Runs multiple MCMC steps using lax.scan. JIT-compiled.
        The single-step logic is defined internally via closure.
        Parameters:
            chain_init (jnp.ndarray):
                Initial state of the chain (shape: [numchains, *shape]).
            current_val_init (jnp.ndarray):
                Initial log-probability values for each chain (1D array).
            rng_k_init (jax.random.PRNGKey):
                Initial random key for JAX.
            num_proposed_init (jnp.ndarray):
                Initial number of proposals made so far (1D array).
            num_accepted_init (jnp.ndarray):
                Initial number of accepted proposals so far (1D array).
            params (Any):
                Network parameters.
            steps (int):
                Number of MCMC steps to perform.
            update_proposer (Callable):
                Function that proposes a new state.
            net_callable_fun (Callable):
                The network callable.
            mu (float):
                Exponent :math:`\mu` for the sampling distribution.
            beta (float):
                Inverse temperature :math:`\beta` for the Metropolis acceptance probability.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
                Final chain states, log-probabilities, updated random key,
        '''

        num_chains          = chain_init.shape[0]
        logproba_fun        = jax.vmap(lambda x: jnp.real(net_callable_fun(params, x)), in_axes=(0,))
        proposer_fn         = jax.vmap(update_proposer, in_axes=(0, 0))
        
        # Define the single-step function *inside* so it closes over the arguments
        def _sweep_chain_jax_step_inner(carry, step_idx):
            chain_in, current_val_in, current_key, num_prop, num_acc = carry
            
            # One for this step and one for the next carry
            key_step, next_key_carry    = jax.random.split(current_key)
            # Proposer and acceptance keys
            key_prop_base, key_acc      = jax.random.split(key_step)
            chain_prop_keys             = jax.random.split(key_prop_base, num_chains)
            
            #! Single MCMC update step logic
            # Propose update
            new_chain               = proposer_fn(chain_in, chain_prop_keys)    # [num_chains, state_shape]
            new_val                 = logproba_fun(new_chain)                   # [num_chains]
            if new_val.ndim > 1:    new_val = new_val[:, 0]
            
            #! Calculate the log-probability of the new state - like MCSampler._logprob_jax
            delta                   = new_val - current_val_in                  # [num_chains]
            ratio                   = jnp.exp(beta * mu * delta)                # [num_chains]

            #! Calculate acceptance probabilities (using accept_config_fun)
            u                       = jax.random.uniform(key_acc, shape=(num_chains))
            keep                    = u < ratio                                 # [num_chains,]  
            # accept/reject updates
            # broadcast accept mask over state dims
            mask                    = keep.reshape((num_chains,) + (1,) * (chain_in.ndim - 1)) 
            # Reshape mask for broadcasting: (num_chains, 1, 1...) depending on state shape
            chain_out               = jnp.where(mask, new_chain, chain_in)  
            val_out                 = jnp.where(keep, new_val, current_val_in)
            proposed_out            = num_prop + 1
            accepted_out            = num_acc + keep.astype(num_acc.dtype)
            
            new_carry               = (chain_out, val_out, next_key_carry, proposed_out, accepted_out)
            # We return None as the second element because we are not collecting 
            # intermediate states in this specific loop (this is thermalization/skipping)
            return new_carry, None
        
        # Initial carry now includes the RNG key
        initial_carry   = (chain_init, current_val_init, rng_k_init, num_proposed_init, num_accepted_init)
        
        # Use lax.scan (Efficient Loop)
        final_carry, _  = jax.lax.scan(_sweep_chain_jax_step_inner, initial_carry, None, length=steps)
        
        # Unpack final results
        final_chain, final_val, final_key, final_prop, final_acc = final_carry
        
        return final_chain, final_val, final_key, final_prop, final_acc
    
    @staticmethod
    @numba.njit
    def _run_mcmc_steps_np(chain            : np.ndarray,
                        logprobas           : np.ndarray,
                        num_proposed        : np.ndarray,
                        num_accepted        : np.ndarray,
                        params              : Any,
                        rng                 : np.random.Generator,
                        steps               : int,
                        mu                  : float,
                        beta                : float,
                        update_proposer     : Callable,
                        log_proba_fun       : Callable,
                        accept_config_fun   : Callable,
                        net_callable_fun    : Callable,
                        ):
        r"""
        NumPy version of sweeping a single chain.
        
        This function carries a tuple:
        (chain, current_logprob, rng_k, num_proposed, num_accepted)
        through a loop over a number of steps. For each step, it:
        - Proposes new candidate states using update_proposer (applied to each chain element)
        - Computes new log-probabilities via log_probability
        - Computes the acceptance probability using accept_config
        - Generates a uniform random number for each chain element to decide acceptance
        - Updates the chain and the carried current log-probability accordingly
        - Updates counters for the total proposals and acceptances.
        
        Parameters:
        chain           : Current state of the chain (NumPy array, shape (nChains, ...))
        logprobas       : Current log-probabilities for each chain element (1D NumPy array)
        rng_k           : (Not really used in the NumPy version; can be updated with a new seed)
        num_proposed    : Total number of proposals made so far (integer)
        num_accepted    : Total number of accepted proposals so far (integer)
        params          : Network parameters (passed to log_probability)
        update_proposer : Function that proposes a new state. Signature should be: new_state = update_proposer(key, state, update_proposer_arg)
        log_probability : Function to compute the log-probability; signature: new_logprob = log_probability(new_state, net_callable=..., net_params=params)
        accept_config   : Function to compute the acceptance probability from current and candidate log-probabilities.
        net_callable    : The network callable (e.g. returns Re(log\psi (s)))
        steps           : Number of update steps to perform.
        
        Returns:
        A tuple (chain, current_logprobas, rng_k, num_proposed, num_accepted)
        """
        
        current_val     = logprobas if logprobas is not None else log_proba_fun(chain, mu=mu, net_callable=net_callable_fun, net_params=params)
        n_chains        = chain.shape[0] if len(chain.shape) > 1 else 1
        
        def loop_sweep(chain, current_val, num_proposed, num_accepted, rng):
            # Loop through the number of steps
            for _ in range(steps):
                
                # use the temporary states for the proposal
                # tmpstates               = chain.copy()
                
                # For each chain element, generate a new candidate using the update proposer.
                # Here, we simulate splitting rng_k by simply generating a new random integer key.
                # also, this flips the state inplace, so we need to copy the chain
                # and then update the chain with the new state.
                # Use NumPy's vectorized operations to generate new states
                new_chain               = update_proposer(chain, rng)
                
                # Compute new log probabilities for candidates using the provided log probability function.
                new_logprobas           = log_proba_fun(new_chain, net_callable=net_callable_fun, net_params=params)
                
                # Compute acceptance probability for each chain element.
                acc_probability         = accept_config_fun(current_val, new_logprobas, beta, mu)
                
                # Decide acceptance by comparing against a random uniform sample.
                rand_vals               = rng.random(size=n_chains)
                accepted                = rand_vals < acc_probability
                num_proposed            += chain.shape[0] if len(chain.shape) > 1 else 1
                num_accepted[accepted]  += 1
                
                # Update: if accepted, take candidate, else keep old.
                new_val                 = np.where(accepted, new_logprobas, current_val)
                # Update the carry:
                # chain                   = np.array([new_chain[i] if accepted[i] else chain[i] for i in range(chain.shape[0])])
                chain                   = np.where(accepted[:, None], new_chain, chain)
                current_val             = new_val
                
                # Update rng_k with a new random integer (not used further)
            return chain, current_val, num_proposed, num_accepted
        
        # go through the number of steps, for each step: 
        # 1. propose a new state
        # 2. compute the log probability
        # 3. compute the acceptance probability
        # 4. decide with dice rule
        # 5. update the chain
        # 6. update the carry
        # 7. update rng_k with a new random integer (not used further)
        chain, current_val, num_proposed, num_accepted = loop_sweep(chain, current_val, num_proposed, num_accepted, rng)
        return chain, current_val, num_proposed, num_accepted

    def _sweep_chain(self,
                    chain               : Array,
                    logprobas           : Array,
                    rng_k               : Array,
                    num_proposed        : Array,
                    num_accepted        : Array,
                    params              : Array,
                    update_proposer     : Callable,
                    log_proba_fun       : Callable,
                    accept_config_fun   : Callable,
                    net_callable_fun    : Callable,
                    steps               : int):
        '''
        Sweep the chain for a given number of MCMC steps.
        One "step" typically involves proposing N updates, where N is the system size.

        Parameters:
            chain (array-like):
                Current states of the chains.
            logprobas (array-like):
                Log probabilities of the current states.
            rng_k_or_rng (Any):
                JAX PRNGKey or NumPy Generator.
            num_proposed (array-like):
                Array tracking proposed moves per chain.
            num_accepted (array-like):
                Array tracking accepted moves per chain.
            params (Any):
                Network parameters.
            steps (int):
                Number of MCMC update steps to perform.

        Returns:
            Tuple:
                (updated_chain, updated_logprobas, updated_rng_k/rng, updated_num_proposed, updated_num_accepted)
        '''
        use_log_proba_fun       = self.logprob if log_proba_fun is None else log_proba_fun
        use_accept_config_fun   = self.acceptance_probability if accept_config_fun is None else accept_config_fun
        use_net_callable_fun    = self._net_callable if net_callable_fun is None else net_callable_fun
        use_update_proposer     = self._upd_fun if update_proposer is None else update_proposer
        
        if logprobas is None:
            logprobas = self._logprobas
            if logprobas is None:
                logprobas = self.logprob(chain, net_callable=net_callable_fun, net_params=params)
        
        if self._isjax:
            # call the compiled version
            return self._run_mcmc_steps_jax(chain_init          =   chain,
                                            current_val_init    = logprobas,
                                            rng_k_init          = rng_k,
                                            num_proposed_init   = num_proposed,
                                            num_accepted_init   = num_accepted,
                                            params              = params,
                                            steps               = steps,
                                            update_proposer     = use_update_proposer,
                                            log_proba_fun       = use_log_proba_fun,
                                            accept_config_fun   = use_accept_config_fun,
                                            net_callable_fun    = use_net_callable_fun,
                                            mu                  = self._mu)
        # otherwise, numpy it is
        return self._run_mcmc_steps_np(chain                =   chain,
                                        logprobas           =   logprobas,
                                        rng                 =   rng_k,
                                        num_proposed        =   num_proposed,
                                        num_accepted        =   num_accepted,
                                        params              =   params,
                                        update_proposer     =   use_update_proposer,
                                        log_proba_fun       =   use_log_proba_fun,
                                        accept_config_fun   =   use_accept_config_fun,
                                        net_callable_fun    =   use_net_callable_fun,
                                        mu                  =   self._mu,
                                        beta                =   self._beta,
                                        steps               =   steps)
    
    ###################################################################
    #! SAMPLING
    ###################################################################
    
    @staticmethod
    def _generate_samples_jax(
        states_init         : Array,
        logprobas_init      : Array,
        rng_k_init          : Array,
        num_proposed_init   : Array,
        num_accepted_init   : Array,
        params              : Any,
        # static arguments:
        num_samples         : int,
        total_therm_updates : int,
        updates_per_sample  : int,
        mu                  : float,
        beta                : float,
        # Pass the actual function objects needed:
        update_proposer     : Callable,
        net_callable_fun    : Callable):
        '''
        JIT-compiled function for thermalization and sample collection using MCMC.
        Uses closures for scan bodies.
        '''

        #! Thermalization phase
        # Calls the static method _run_mcmc_steps_jax
        states_therm, logprobas_therm, rng_k_therm, num_proposed_therm, num_accepted_therm = \
            VMCSampler._run_mcmc_steps_jax(
            chain_init          =       states_init,
            current_val_init    =       logprobas_init,
            rng_k_init          =       rng_k_init,
            num_proposed_init   =       num_proposed_init,
            num_accepted_init   =       num_accepted_init,
            params              =       params,
            steps               =       total_therm_updates,
            # Pass the functions through:
            update_proposer     =       update_proposer,
            net_callable_fun    =       net_callable_fun,
            mu                  =       mu,
            beta                =       beta
            )

        #! Sampling phase (using lax.scan for collection)
        def sample_scan_body(carry, _):
            # Unpack dynamic carry state
            states_carry, logprobas_carry, rng_k_carry, num_proposed_carry, num_accepted_carry = carry

            # Run MCMC steps between samples using the static method
            states_new, logprobas_new, rng_k_new, num_proposed_new, num_accepted_new = \
                VMCSampler._run_mcmc_steps_jax(
                    chain_init          =   states_carry,
                    current_val_init    =   logprobas_carry,
                    rng_k_init          =   rng_k_carry,
                    num_proposed_init   =   num_proposed_carry,
                    num_accepted_init   =   num_accepted_carry,
                    params              =   params,
                    steps               =   updates_per_sample,
                    update_proposer     =   update_proposer,
                    net_callable_fun    =   net_callable_fun,
                    mu                  =   mu,
                    beta                =   beta
                )

            # Return updated dynamic carry and the collected sample (states_new)
            return (states_new, logprobas_new, rng_k_new, num_proposed_new, num_accepted_new), states_new

        # Run the scan to collect samples
        initial_scan_carry              = (states_therm, logprobas_therm, rng_k_therm, num_proposed_therm, num_accepted_therm)
        final_carry, collected_samples  = jax.lax.scan(
            f       = sample_scan_body, # Use the inner function defined above
            init    = initial_scan_carry,
            xs      = None,
            length  = num_samples
        )
        # final_carry contains the state after the last sample collection sweep
        return final_carry, collected_samples
    
    def _generate_samples_np(self,
                        params,
                        num_samples,
                        multiple_of = 1):
        """
        NumPy version of obtaining samples via MCMC.
        
        Parameters:
        - params         : Network parameters.
        - num_samples    : Number of sweeps (sample collection iterations) to perform.
        - multiple_of    : (Not used here, but could be used for device distribution.)
        
        Returns:
        A tuple (meta, configs) where:
            meta is a tuple of the final (states, logprobas, rng_k, num_proposed, num_accepted)
            configs is an array of shape (numSamples, -1) + self._shape containing the sampled configurations.
        """
        
        # Thermalize the chains by sweeping for (therm_steps * sweep_steps)
        states, logprobas, num_proposed, num_accepted = self._run_mcmc_steps_np(
            chain               = self._states,
            logprobas           = self._logprobas,
            num_proposed        = self._num_proposed,
            num_accepted        = self._num_accepted,
            params              = params,
            update_proposer     = self._upd_fun,
            log_proba_fun       = self.logprob,
            accept_config_fun   = self.acceptance_probability,
            net_callable_fun    = self._net_callable,
            steps               = self._therm_steps * self._sweep_steps,
            rng                 = self._rng,
            mu                  = self._mu,
            beta                = self._beta)
        
        # Now perform numSamples sweeps, collecting the resulting states
        meta            = []
        configs_list    = []
        
        # Perform the sampling sweeps - do the same as the thermalization but save the states
        for _ in range(num_samples):
            states, logprobas, num_proposed, num_accepted = self._run_mcmc_steps_np(
                chain               = states,
                logprobas           = logprobas,
                num_proposed        = num_proposed,
                num_accepted        = num_accepted,
                params              = params,
                update_proposer     = self._upd_fun,
                log_proba_fun       = self.logprob,
                accept_config_fun   = self.acceptance_probability,
                net_callable_fun    = self._net_callable,
                steps               = self._sweep_steps,
                rng                 = self._rng,
                mu                  = self._mu,
                beta                = self._beta)
                
            meta        = (states.copy(), logprobas.copy(), num_proposed, num_accepted)
            # Assume states is of shape (num_chains, *self._shape)
            configs_list.append(states.copy())
        
        # Concatenate configurations along the chain axis
        # Then reshape to (num_samples, -1) + self._shape (flattening the chain dimension per sample)
        configs = np.concatenate(configs_list, axis=0)
        return meta, configs.reshape((num_samples, -1) + self._shape)
    
    ###################################################################
    #! STATIC JAX SAMPLING KERNEL
    ###################################################################
    
    @staticmethod
    def _static_sample_jax(
            # --- Initial State (Dynamic) ---
            states_init         : jnp.ndarray,
            rng_k_init          : jnp.ndarray,
            num_proposed_init   : jnp.ndarray,
            num_accepted_init   : jnp.ndarray,
            # --- Network/Model (Dynamic) ---
            params              : Any,
            # --- Configuration (Static) ---
            num_samples         : int,
            num_chains          : int,
            total_therm_updates : int,
            updates_per_sample  : int,
            shape               : Tuple[int,...],
            mu                  : float,
            beta                : float,
            logprob_fact        : float,
            # --- Function References (Static) ---
            update_proposer     : Callable,
            net_callable_fun    : Callable
        ):
        r'''
        Static, JIT-compiled core logic for MCMC sampling in JAX. 
        
        Performs the following steps:
        1. Calculate Initial Log Probs
        2. Generate Samples via MCMC Kernel
        3. Post-processing
        4. Return Results
        Parameters:
            states_init (jnp.ndarray):
                Initial state(s) of the chain(s).
            rng_k_init (jnp.ndarray):
                Initial random key for JAX.
            num_proposed_init (jnp.ndarray):
                Initial number of proposals made so far.
            num_accepted_init (jnp.ndarray):
                Initial number of accepted proposals so far.
            params (Any):
                Network parameters.
            num_samples (int):
                Number of samples to generate.
            num_chains (int):
                Number of chains to use.
            total_therm_updates (int):
                Total number of thermalization updates.
            updates_per_sample (int):
                Number of updates per sample collection.
            shape (Tuple[int,...]):
                Shape of the state configurations.
            mu (float):
                Exponent :math:`\mu` for the sampling distribution.
            beta (float):
                Inverse temperature :math:`\beta` for the Metropolis acceptance probability.
            logprob_fact (float):
                Logarithmic factor for normalization.
            update_proposer (Callable):
                Function that proposes a new state.
            log_proba_fun_base (Callable):
                Function to compute log probabilities.
            accept_config_fun_base (Callable):
                Function to compute acceptance probabilities.
            net_callable_fun (Callable):
                The network callable.
        '''
        
        def _logproba_fun_base(params, states):
            ''' is the same as MCSampler._logprob_jax '''
            return jnp.real(net_callable_fun(params, states))

        logprobas_init = jax.vmap(_logproba_fun_base, in_axes=(None, 0))(params, states_init)
        logprobas_init = jnp.squeeze(logprobas_init, axis=-1) if logprobas_init.ndim > 1 else logprobas_init

        #! Phase 1: Thermalization
        states_therm, logprobas_therm, rng_k_therm, _, _ = VMCSampler._run_mcmc_steps_jax(
            chain_init          =   states_init,
            current_val_init    =   logprobas_init,
            rng_k_init          =   rng_k_init,
            num_proposed_init   =   num_proposed_init,
            num_accepted_init   =   num_accepted_init,
            params              =   params,
            steps               =   total_therm_updates,
            update_proposer     =   update_proposer,
            net_callable_fun    =   net_callable_fun,
            mu                  =   mu,
            beta                =   beta
        )

        #! Phase 2: Sampling    
        def sample_scan_body(carry, _):
            # Run a fixed number of updates per sample and update the chain carry.
            new_carry = VMCSampler._run_mcmc_steps_jax(
                chain_init            = carry[0],
                current_val_init      = carry[1],
                rng_k_init            = carry[2],
                num_proposed_init     = carry[3],
                num_accepted_init     = carry[4],
                params                = params,
                steps                 = updates_per_sample,
                update_proposer       = update_proposer,
                net_callable_fun      = net_callable_fun,
                mu                    = mu,
                beta                  = beta
            )
            # Return updated carry and collect the new chain state.
            return new_carry, new_carry[0]

        # Initialize the carry for the scan with the thermalized state and counters.
        initial_scan_carry = (states_therm, logprobas_therm, rng_k_therm, num_proposed_init, num_accepted_init)

        # Run the scan for the specified number of sample steps.
        final_carry, collected_samples = jax.lax.scan(sample_scan_body, initial_scan_carry, None, length=num_samples)

        # Flatten the collected states from all sample steps.
        configs_flat        = collected_samples.reshape((-1,) + shape)
        
        # Evaluate the network in a fully batched (vectorized) manner to obtain log_\psi .
        net_apply           = lambda conf: net_callable_fun(params, conf)
        batched_log_ansatz  = jax.vmap(net_apply)(configs_flat)

        # Compute the importance weights.
        log_prob_exponent    = (1.0 / logprob_fact - mu)
        probs                = jnp.exp(log_prob_exponent * jnp.real(batched_log_ansatz))
        total_samples        = num_samples * num_chains
        norm_factor          = jnp.maximum(jnp.sum(probs), 1e-10)
        probs_normalized     = probs / norm_factor * total_samples

        return final_carry, (configs_flat, batched_log_ansatz), probs_normalized
    
    ###################################################################
    
    def _sample_reinitialize(self, used_num_chains, used_num_samples):
        '''
        Reinitialize the sampler with the given parameters.
        Parameters:
            current_states:
                The current states of the chains
            current_proposed:
                The current number of proposed states
            current_accepted:
                The current number of accepted states
            used_num_chains:
                The number of chains to use
            used_num_samples:
                The number of samples to generate
        '''
        self._numchains         = used_num_chains
        self._numsamples        = used_num_samples
        self.reset()
        current_states          = self._states
        current_proposed        = self._num_proposed
        current_accepted        = self._num_accepted
        if self._isjax:
            self._static_sample_fun = self._get_sampler_jax(self._numsamples, self._numchains)
        else:
            self._static_sample_fun = self._get_sampler_np(self._numsamples, self._numchains)
        return current_states, current_proposed, current_accepted
    
    def _sample_callable(self, parameters=None):
        """
        Prepares and returns a callable network function along with the current set of parameters.

        Parameters:
            parameters (optional):
                An explicit set of parameters to use. If not provided, the method attempts to retrieve
                parameters from the internal state or network object.

        Returns:
            tuple:
                A tuple containing:
                - net_callable: The callable network function.
                - current_params: The parameters to be used with the network function.

        Notes:
            The method prioritizes the provided `parameters` argument. If not given, it attempts to retrieve parameters
            from the internal `_parameters` attribute, the network's `params` attribute, or the network's `get_params()` method.
        """
        current_params = None
        if parameters is not None:
            current_params = parameters
        elif self._parameters is not None:
            current_params = self._net.get_params()
        elif hasattr(self._net, 'params'):
            current_params = self._net.params
        elif hasattr(self._net, 'get_params'):
            current_params = self._net.get_params()
        net_callable = self._net_callable
        return net_callable, current_params
    
    def sample(self, parameters=None, num_samples=None, num_chains=None):
        r'''
        Sample the states from the Hilbert space according to the Born distribution.
        Parameters:
            parameters:
                The parameters for the network
            num_samples:
                The number of samples to generate
            num_chains:
                The number of chains to use
        Returns:
            The sampled states. Namely, we return a tuple:
            - final_state_info  : A tuple (states, logprobas) of the final states and their log probabilities
                - states            : The final states of the chains after sampling:    s
                - logprobas         : The log probabilities of the final states:        log (psi (s') / psi (s))
            - samples_tuple     : A tuple (configs, configs_log_ansatz) of the sampled configurations and their log ansatz values
                - configs           : The sampled configurations: s
                - configs_log_ansatz: The log ansatz values of the sampled configurations: log\psi (s)
            - probs             : The importance weights for each sampled configuration
        
        Example:
            >>> final_state_info, samples_tuple, probs = sampler.sample(parameters=params, num_samples=1000, num_chains=10)
            >>> states, logprobas = final_state_info
            >>> configs, configs_log_ansatz = samples_tuple
            >>> print("Final States:", states)
            >>> print("Log Probabilities:", logprobas)
            >>> print("Sampled Configurations:", configs)
            >>> print("Log Ansatz Values:", configs_log_ansatz)
            >>> print("Importance Weights:", probs)        
        '''
        
        # check the number of samples and chains
        used_num_samples        = num_samples if num_samples is not None else self._numsamples
        used_num_chains         = num_chains if num_chains is not None else self._numchains

        # Handle temporary state if num_chains differs from instance default
        current_states          = self._states
        current_proposed        = self._num_proposed
        current_accepted        = self._num_accepted
        reinitialized_for_call  = False
        
        #! check if one needs to reconfigure the sampler
        if used_num_chains != self._numchains or used_num_samples != self._numsamples:
            print(f"Warning: Running sample with {used_num_chains} chains (instance default is {self._numchains}). State reinitialized for this call.")
            reinitialized_for_call                              = True
            current_states, current_proposed, current_accepted  = self._sample_reinitialize(used_num_chains, used_num_samples)
            
        # check the parameters - if not given, use the current parameters
        net_callable, current_params = self._sample_callable(parameters)

        if self._isjax:
            if not isinstance(self._rng_k, jax.Array):
                raise TypeError(f"Sampler's RNG key is invalid: {type(self._rng_k)}")

            #! Call the function that is static and JIT-compiled
            final_state_tuple, samples_tuple, probs = self._static_sample_fun(
                states_init             = current_states,
                rng_k_init              = self._rng_k,
                params                  = current_params,
                num_proposed_init       = current_proposed,
                num_accepted_init       = current_accepted
            )
        
            #! Update Instance State (JAX)
            final_states, final_logprobas, final_rng_k, final_num_proposed, final_num_accepted = final_state_tuple
            if not reinitialized_for_call:
                self._states            = final_states
                self._logprobas         = final_logprobas
                self._num_proposed      = final_num_proposed
                self._num_accepted      = final_num_accepted
                self._rng_k             = final_rng_k
                
            final_state_info            = (self._states, self._logprobas)
            return final_state_info, samples_tuple, probs
        else:
            self._logprobas             = self.logprob(self._states,
                                            net_callable    =   net_callable,
                                            net_params      =   parameters)
            # for numpy
            (self._states, self._logprobas, self._num_proposed, self._num_accepted), configs =\
                self._generate_samples_np(parameters, num_samples, num_chains)
            configs_log_ansatz  = np.array([self._net(parameters, config) for config in configs])
            probs               = np.exp((1.0 / self._logprob_fact - self._mu) * np.real(configs_log_ansatz))
            norm                = np.sum(probs, axis=0, keepdims=True)
            probs               = probs / norm * self._numchains
            
            # flatten the configs to be of shape (num_samples * num_chains)
            configs             = configs.reshape(-1, *configs.shape[2:])
            configs_log_ansatz  = configs_log_ansatz.reshape(-1, *configs_log_ansatz.shape[2:])
            probs               = probs.reshape(-1, *probs.shape[2:])
            
            return (self._states, self._logprobas), (configs, configs_log_ansatz), probs
    
    ###################################################################
    #! SETTERS
    ###################################################################
    
    def set_mu(self, mu):
        '''
        Set the parameter mu.
        Parameters:
            - mu : The parameter mu
        '''
        self._mu = mu
        
    def set_beta(self, beta):
        '''
        Set the parameter beta.
        Parameters:
            - beta : The parameter beta
        '''
        self._beta = beta
        
    def set_therm_steps(self, therm_steps):
        '''
        Set the number of thermalization steps.
        Parameters:
            - therm_steps : The number of thermalization steps
        '''
        self._therm_steps = therm_steps

    ###################################################################
    #! GETTERS
    ###################################################################
    
    def get_mu(self):
        '''
        Get the parameter mu.
        '''
        return self._mu
    
    def get_beta(self):
        '''
        Get the parameter beta.
        '''
        return self._beta
    
    def get_therm_steps(self):
        '''
        Get the number of thermalization steps.
        '''
        return self._therm_steps
    
    def get_sweep_steps(self):
        '''
        Get the number of sweep steps.
        '''
        return self._sweep_steps

    def _get_sampler_jax(self, num_samples: Optional[int] = None, num_chains: Optional[int] = None) -> Callable:
        """
        Returns a JIT-compiled static sampling function with baked-in configuration.

        This function partially applies the core static JAX sampling logic
        (`_static_sample_jax`) with configuration parameters (like mu, beta,
        step counts, function references) taken from the current sampler instance.

        The returned function is JIT-compiled for the specific `num_samples` and
        `num_chains` provided (or taken from the instance defaults). It has a
        simplified signature suitable for repeated calls where only the initial state,
        RNG key, and network parameters might change frequently.

        Parameters:
            num_samples (Optional[int]):
                Number of samples per chain to bake into the static function.
                If None, uses `self._numsamples`.
            num_chains (Optional[int]):
                Number of chains to bake into the static function.
                If None, uses `self._numchains`.

        Returns:
            Callable: A JIT-compiled function with the signature:
            `wrapped_sampler(states_init, rng_k_init, params,
                            num_proposed_init=None, num_accepted_init=None)`
            where `num_proposed_init` and `num_accepted_init` default to zero arrays
            of the correct shape and integer dtype if not provided.
            It returns the same tuple structure as `_static_sample_jax`:
            `(final_state_tuple, samples_tuple, probs_normalized)`

        Raises:
            RuntimeError: If the sampler backend is not 'jax'.
        """
        if not self._isjax:
            raise RuntimeError("Static JAX sampler getter only available for JAX backend.")

        # Use provided sample/chain counts or instance defaults
        static_num_samples  = num_samples if num_samples is not None else self._numsamples
        static_num_chains   = num_chains if num_chains is not None else self._numchains
        
        # Gather arguments to be baked in (partial application)
        # Configuration values from self
        baked_args = {
            #! Static Config
            "num_samples"               : static_num_samples,
            "num_chains"                : static_num_chains,
            "total_therm_updates"       : self._total_therm_updates,
            "updates_per_sample"        : self._updates_per_sample,
            "shape"                     : self._shape,
            "update_proposer"           : self._upd_fun,
            "net_callable_fun"          : self._net_callable,
            "mu"                        : self._mu,
            "beta"                      : self._beta,
            "logprob_fact"              : self._logprob_fact,
        }

        # Create the partially applied function
        # Target function is MCSampler._static_sample_jax
        partial_sampler                 = partial(VMCSampler._static_sample_jax, **baked_args)

        # Define the final wrapper function with the desired signature
        # This wrapper handles default values for initial counters
        int_dtype                       = DEFAULT_JP_INT_TYPE

        @jax.jit
        def wrapped_sampler(states_init         : jax.Array,
                            rng_k_init          : jax.Array,
                            params              : Any,
                            num_proposed_init   : Optional[jax.Array] = None,
                            num_accepted_init   : Optional[jax.Array] = None):
            """
            JIT-compiled static sampler with baked-in configuration.

            Args:
                states_init: Initial states [num_chains, *shape].
                rng_k_init: Initial JAX PRNGKey.
                params: Network parameters for this run.
                num_proposed_init: Initial proposal counts [num_chains] (defaults to zeros).
                num_accepted_init: Initial acceptance counts [num_chains] (defaults to zeros).

            Returns:
                (final_state_tuple, samples_tuple, probs_normalized)
            """
            # Default initial counters to zero if not provided
            _num_proposed_init = jnp.zeros(static_num_chains, dtype=int_dtype) if num_proposed_init is None else num_proposed_init
            _num_accepted_init = jnp.zeros(static_num_chains, dtype=int_dtype) if num_accepted_init is None else num_accepted_init

            # Validate shapes of inputs relative to static_num_chains
            # if states_init.shape[0] != static_num_chains:
            #     raise ValueError(f"states_init first dimension ({states_init.shape[0]}) must match static num_chains ({static_num_chains})")
            # if _num_proposed_init.shape != (static_num_chains,):
            #     raise ValueError(f"num_proposed_init shape ({_num_proposed_init.shape}) must match ({static_num_chains},)")
            # if _num_accepted_init.shape != (static_num_chains,):
            #     raise ValueError(f"num_accepted_init shape ({_num_accepted_init.shape}) must match ({static_num_chains},)")

            # Call the partially applied core function
            final_state_tuple, samples_tuple, probs = partial_sampler(
                states_init         =   states_init,
                rng_k_init          =   rng_k_init,
                num_proposed_init   =   _num_proposed_init,
                num_accepted_init   =   _num_accepted_init,
                params              =   params
            )
            return final_state_tuple, samples_tuple, probs
        return wrapped_sampler
    
    def _get_sampler_np(self, num_samples: Optional[int] = None, num_chains: Optional[int] = None) -> Callable:
        """
        Returns a NumPy-based sampling function with baked-in configuration.

        This function partially applies the core NumPy sampling logic
        (`_generate_samples_np`) with configuration parameters (like mu, beta,
        step counts, function references) taken from the current sampler instance.

        The returned function has a simplified signature suitable for repeated
        calls where only the initial state and network parameters might change frequently.

        Parameters:
            num_samples (Optional[int]):
                Number of samples per chain to bake into the static function.
                If None, uses `self._numsamples`.
            num_chains (Optional[int]):
                Number of chains to bake into the static function.
                If None, uses `self._numchains`.

        Returns:
            Callable: A NumPy-based function with the signature:
            `wrapped_sampler(states_init, params)`
            It returns the same tuple structure as `_generate_samples_np`:
            `(meta, configs)`

        Raises:
            RuntimeError: If the sampler backend is not 'numpy'.
        """
        if self._isjax:
            raise RuntimeError("NumPy sampler getter only available for NumPy backend.")

        # Use provided sample/chain counts or instance defaults
        static_num_samples  = num_samples if num_samples is not None else self._numsamples
        static_num_chains   = num_chains if num_chains is not None else self._numchains

        # Gather arguments to be baked in (partial application)
        # Configuration values from self
        baked_args = {
            #! Static Config
            "num_samples"               : static_num_samples,
            "num_chains"                : static_num_chains,
            "therm_steps"               : self._therm_steps
        }
        
        def wrapper(states_init, rng_k_init, param, num_proposed_init=None, num_accepted_init=None):
            """
            Wrapper function for NumPy sampling.

            Args:
                states_init: Initial states [num_chains, *shape].
                rng_k_init: Initial RNG key (not used in NumPy).
                params: Network parameters for this run.

            Returns:
                (meta, configs)
            """
            # Call the partially applied core function
            self._states        = states_init
            self._rng_k         = rng_k_init
            self._num_proposed  = num_proposed_init if num_proposed_init is not None else np.zeros(static_num_chains, dtype=DEFAULT_NP_INT_TYPE)
            self._num_accepted  = num_accepted_init if num_accepted_init is not None else np.zeros(static_num_chains, dtype=DEFAULT_NP_INT_TYPE)
            return self.sample(parameters=param, num_samples=baked_args["num_samples"], num_chains=baked_args["num_chains"])
        
        return wrapper

    def get_sampler(self, num_samples: Optional[int] = None, num_chains: Optional[int] = None) -> Callable:
        """
        Get the appropriate sampler function based on the backend (JAX or NumPy).

        This method returns a callable sampling function that is configured
        according to the current instance settings. It automatically selects
        between a JAX-based or NumPy-based implementation depending on the
        backend in use.

        Parameters:
            num_samples (Optional[int]):
                Number of samples per chain to bake into the static function.
                If None, uses `self._numsamples`.
            num_chains (Optional[int]):
                Number of chains to bake into the static function.
                If None, uses `self._numchains`.
        Returns:
            Callable: A sampling function with a signature depending on the backend:
                - For JAX: `wrapped_sampler(states_init, rng_k_init, params,
                                            num_proposed_init=None, num_accepted_init=None)`
                - For NumPy: `wrapped_sampler(states_init, params)`
            The return type matches either `_static_sample_jax` or `_generate_samples_np`.
        Raises:
            RuntimeError: If the backend is neither 'jax' nor 'numpy'.
        """
        
        if self._isjax:
            return self._get_sampler_jax(num_samples, num_chains)
        elif not self._isjax:
            return self._get_sampler_np(num_samples, num_chains)
        else:
            raise RuntimeError("Sampler backend must be either 'jax' or 'numpy'.")

#############################################################################
#! EOF
#############################################################################