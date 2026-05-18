r"""
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
Version         : 2.2
---------------------------------

Change Log:
- 2026-04-29:
    - Improved cache support detection for log_psi_delta functions by inspecting their signatures.
    - Cleaned up the __init__ method and removed redundant methods
"""

#######################################################################

import inspect
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import numpy as np

#######################################################################
#! Update proposers
#######################################################################
try:
    from .sampler import Sampler, SamplerErrors, _bind_named_partial, resolve_state_defaults
    from .updates import make_hybrid_proposer, propose_local_flip, propose_local_flip_np, propose_multi_flip

    # Import _with_info variants if available (they should be in spin_jax.py now)
    try:
        from .updates.spin_jax import (
            propose_bond_flip_with_info,
            propose_exchange_with_info,
            propose_global_flip_with_info,
            propose_local_flip_with_info,
            propose_multi_flip_with_info,
            propose_worm_flip_with_info,
        )
    except ImportError:
        pass

    from .diagnostics import compute_autocorr_time, compute_rhat
    from . import vmc_numpy as vmc_np_impl
except ImportError as e:
    raise ImportError("Failed to import Sampler/Updates/Diagnostics from MonteCarlo module.") from e

# flax for the network
try:
    import  jax
    import  jax.numpy as jnp
    from    flax import linen as nn
    from    jax import lax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# from algebra
try:
    if TYPE_CHECKING:
        from QES.Algebra.hilbert import HilbertSpace

    from QES.general_python.algebra.utils import DEFAULT_JP_INT_TYPE, DEFAULT_NP_INT_TYPE
except ImportError:
    # If imports fail (e.g. in test environment), define defaults
    DEFAULT_JP_INT_TYPE = np.int32
    DEFAULT_NP_INT_TYPE = np.int32

try:
    from QES.NQS.src.network.adapters import resolve_sampling_hooks_for_network
except ImportError:
    resolve_sampling_hooks_for_network = None

# ---------------------------------------------------------------------------
#! Variational Monte Carlo Sampler
# ---------------------------------------------------------------------------


class VMCSampler(Sampler):
    r"""
    Variational Markov Chain Monte Carlo sampler for quantum states.

    This sampler performs Metropolis-Hastings sampling of a probability distribution
    defined by a quantum wavefunction ansatz $\Psi(s)$.
    The sampling probability is given by $P(s) \propto |\Psi(s)|^\mu$ (typically $\mu=2$ for Born rule).

    Expected Network Interface
    --------------------------
    The `net` object passed to the sampler must provide one of the following callables:
    1. `net.apply(params, x)`: Computes log-amplitudes for a batch of states `x`.
    2. `net(params, x)`: Callable object.
    3. `net.get_apply()`: Returns `(apply_fun, params)`.

    Invariants
    ----------
    - **State Shape**: `(num_chains, *shape)`.
    - **State Convention**: The sampler stores and returns states in one explicit
      external convention. For spin-1/2 systems the default convention follows
      the Hamiltonian/operator kernels and uses signed values `{-0.5, +0.5}`
      unless `state_representation` is set explicitly. Proposal rules, cached
      fast updates, local-energy evaluation, and returned samples all stay in
      that same external convention.
    - **Log-Probabilities**: Maintained internally to avoid re-evaluation.
      Re-evaluated only after updates or when `reset()` is called.
    - **Fast Updates**: If `net` implements `log_psi_delta`, the sampler attempts to use
      $O(1)$ or $O(N_{hidden})$ updates instead of full $O(N)$ re-evaluation.

    Supported Backends
    ------------------
    - **JAX**: Fully JIT-compiled sampling kernels (`jax.lax.scan`).
    - **NumPy**: Numba-accelerated sampling loops.
    """

    def __init__(
        self,
        net,
        shape: Tuple[int, ...],
        *,
        pt_betas: Optional[np.ndarray] = None,
        pt_replicas: Optional[int] = None,
        # RNG
        rng: np.random.Generator = None,
        rng_k=None,
        # UPD
        upd_fun: Optional[Callable] = None,
        # OTHER
        mu: float = 2.0,
        beta: float = 1.0,
        therm_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        seed=None,
        hilbert: "HilbertSpace" = None,
        numsamples: int = 1,
        numchains: int = 1,
        numupd: int = 1,
        initstate=None,
        backend: str = "default",
        logprob_fact: float = 0.5,
        statetype: Union[np.dtype, jnp.dtype] = np.float32,
        makediffer: bool = True,
        logger=None,
        dtype: Optional[Any] = None,
        sample_dtype: Optional[Any] = None,
        **kwargs,
    ):
        # Keep donation opt-in: enabled only when explicitly requested.
        donate_buffers          = bool(kwargs.pop("donate_buffers", False))
        network_eval_chunk_size = int(kwargs.pop("network_eval_chunk_size", 2048))
        if network_eval_chunk_size <= 0:
            raise ValueError("network_eval_chunk_size must be a positive integer.")

        input_convention        = getattr(net, "_input_convention", None)
        if isinstance(input_convention, dict):
            input_is_spin       = bool(input_convention.get("input_is_spin", True))
            input_value         = float(input_convention.get("input_value", 0.5 if input_is_spin else 1.0))
            kwargs.setdefault("spin", input_is_spin)
            kwargs.setdefault("mode_repr", input_value)
            kwargs.setdefault("state_representation", "spin_pm" if input_is_spin else "binary_01")

        super().__init__(shape, upd_fun, rng, rng_k, seed,
            hilbert, numsamples, numchains, initstate,
            backend, statetype, makediffer, logger, **kwargs)

        if net is None:
            raise ValueError("A network (or callable) must be provided for evaluation.")

        self._donate_buffers                    = donate_buffers
        self._network_eval_chunk_size           = network_eval_chunk_size
        self._network_adapter                   = kwargs.get("network_adapter", None)
        hooks                                   = self._resolve_network_hooks(net, self._network_adapter)
        self._net_callable                      = hooks["apply_fun"]
        self._parameters                        = hooks["parameters"]
        self._log_psi_delta_fun                 = hooks["log_psi_delta"]
        self._log_psi_delta_cache_init_fun      = hooks["log_psi_delta_cache_init"]

        # User override for delta function
        # This is some analytic update function that runs faster than re-evaluating the network
        if "log_psi_delta" in kwargs:
            self._log_psi_delta_fun = kwargs["log_psi_delta"]
        if "log_psi_delta_cache_init" in kwargs:
            self._log_psi_delta_cache_init_fun  = kwargs["log_psi_delta_cache_init"]

        declared_cache_support              = bool(hooks.get("log_psi_delta_supports_cache", False))
        inferred_cache_support              = self._callable_accepts_positional_args(self._log_psi_delta_fun, 5)
        # Gate cache-aware mode on the delta function actually being present; a declared or inferred
        # flag alone cannot enable cache-aware sampling without a real log_psi_delta callable.
        self._log_psi_delta_supports_cache  = (self._log_psi_delta_fun is not None) and bool(declared_cache_support or inferred_cache_support)

        # set the parameters - this for modification of the distribution
        self._mu = mu
        if self._mu < 0.0 or self._mu > 2.0:
            raise ValueError(SamplerErrors.NOT_IN_RANGE_MU)

        if therm_steps is None or sweep_steps is None:
            backend_name = "cpu"
            if self._isjax:
                try:
                    backend_name = jax.default_backend()
                except Exception:
                    backend_name = "cpu"

            if backend_name == "gpu":
                default_therm   = 8
                default_sweep   = 480
            else:
                # CPU-friendly fallback: fewer sequential updates than GPU profile.
                size            = max(1, int(np.prod(shape)))
                default_therm   = max(16, min(64, max(8, size // 2)))
                default_sweep   = max(24, min(240, max(16, size)))

            if therm_steps is None:
                therm_steps     = default_therm
            if sweep_steps is None:
                sweep_steps     = default_sweep

        therm_steps                     = max(1, int(therm_steps))
        sweep_steps                     = max(1, int(sweep_steps))

        self._beta                      = beta          # Inverse temperature for sampling; also used as the temperature parameter in parallel tempering if enabled.
        self._therm_steps               = therm_steps   # Number of thermalization sweeps to perform before recording samples.
        self._sweep_steps               = sweep_steps   # Number of full sweeps (updates per site) to perform during thermalization and between recorded samples.
        self._logprob_fact              = logprob_fact  # Exponent for the log-probability in the acceptance ratio; typically 0.5 for Born rule sampling, but can be tuned for performance/stability.
        self._logprobas                 = None          # Cache for log-probabilities of current states
        self._params_cache_token        = None          # Token to track when cached log-probabilities are valid based on parameter changes
        # Derived sampling schedule parameters
        self._needs_thermalization      = True
        self._static_sample_therm_steps = None
        self._static_pt_therm_steps     = None
        self._refresh_sampling_schedule()

        # -----------------------------------------------------------------
        # Sampler precision metadata
        # -----------------------------------------------------------------
        # State precision is controlled by ``statetype`` in the base sampler.
        # ``sample_dtype`` is kept as a lightweight compatibility/serialization
        # field for downstream code that records sampler precision metadata.
        # We do not implicitly cast network execution here because that would
        # add hot-path conversions without guaranteeing a real speedup.
        self._dtype         = dtype if dtype is not None else self._statetype
        self._sample_dtype  = self._statetype if sample_dtype is None else sample_dtype

        # -----------------------------------------------------------------
        # Resolve update function
        # -----------------------------------------------------------------

        self._upd_rule_name = "CUSTOM"
        if isinstance(upd_fun, (str, Enum)):
            from .sampler import get_update_function

            self._local_upd_fun = get_update_function(upd_fun, backend="jax" if self._isjax else "numpy", hilbert=hilbert, **kwargs)
            self._upd_rule_name = str(upd_fun)
        else:
            self._local_upd_fun = upd_fun
            if upd_fun is None:
                self._upd_rule_name = "LOCAL"

        # number of times a function is applied per update
        self._numupd = numupd

        if self._local_upd_fun is None:
            if self._isjax:
                self._local_upd_fun = propose_local_flip
            else:
                self._local_upd_fun = propose_local_flip_np

        # Try to switch to fast updates if available
        if self._isjax and self._log_psi_delta_fun is not None:
            self._local_upd_fun = self._resolve_fast_update_fun(self._local_upd_fun, upd_fun, **kwargs)

        # Check for global update configuration
        self._org_upd_fun = self._local_upd_fun
        self._global_name = "none"

        self.set_update_num(self._numupd)

        # -----------------------------------------------------------------
        #! Parallel Tempering (PT) Setup
        # -----------------------------------------------------------------

        self.set_replicas(pt_betas, pt_replicas)
        
        # -----------------------------------------------------------------
        #! Set up hybrid proposer if requested
        # -----------------------------------------------------------------
        self.set_hybrid_proposer(
            kwargs.get("global_p",          kwargs.get("p_global", 0.0)), # if 0.0, the proposer will just use the local update
            kwargs.get("global_fraction",   0.5),
            kwargs.get("patterns",          None),
            kwargs.get("global_update",     None),
        )

        # -----------------------------------------------------------------
        # Static sampler function (standard or PT)
        # -----------------------------------------------------------------
        self._jit_cache         = {}        # Cache for JIT-compiled samplers
        self._static_sample_fun = None
        self._static_pt_sampler = None      # Lazily created on first sample call
        self._name              = "VMC"

    # ----------------------------------------------------------------------

    def _refresh_sampling_schedule(self):
        """Recompute derived sampling counts after schedule changes."""
        self._total_therm_updates               = self._therm_steps * self._sweep_steps * self._size
        self._total_sample_updates_per_sample   = self._sweep_steps * self._size
        self._updates_per_sample                = self._sweep_steps
        self._total_sample_updates_per_chain    = self._numsamples * self._updates_per_sample * self._numchains

    def _invalidate_runtime(self, *, recompute_schedule: bool = False, refresh_layout: bool = False, require_thermalization: bool = False):
        """Refresh derived runtime state and drop cached sampler callables."""
        if recompute_schedule:
            self._refresh_sampling_schedule()
        if refresh_layout:
            self._apply_replica_layout()
        if require_thermalization:
            self._needs_thermalization = True
            
        self._static_sample_fun         = None
        self._static_pt_sampler         = None
        self._static_sample_therm_steps = None
        self._static_pt_therm_steps     = None
        self._jit_cache                 = {}

    def _set_runtime_scalar(
        self,
        attr_name: str,
        value,
        *,
        counts_changed: bool = False,
        layout_changed: bool = False,
        proposer_changed: bool = False,
        distribution_changed: bool = False,
        require_thermalization: bool = True,
    ) -> bool:
        """
        Update one runtime scalar and trigger the canonical rebuild path once.
        """
        if getattr(self, attr_name) == value:
            return False
        setattr(self, attr_name, value)
        self._on_runtime_config_change(counts_changed=counts_changed, layout_changed=layout_changed,
            proposer_changed=proposer_changed, distribution_changed=distribution_changed,
            require_thermalization=require_thermalization,
        )
        return True

    def _on_runtime_config_change(self, *, counts_changed: bool = False, layout_changed: bool = False, proposer_changed: bool = False, distribution_changed: bool = False, require_thermalization: bool = False) -> None:
        """Hook used by base sampler setters."""
        self._invalidate_runtime(
            recompute_schedule      = counts_changed,
            refresh_layout          = layout_changed,
            require_thermalization  = require_thermalization
                                    or counts_changed
                                    or layout_changed
                                    or proposer_changed
                                    or distribution_changed,
        )

    def _apply_replica_layout(self):
        """
        Normalize internal state and counter shapes after PT configuration changes.

        Re-applying ``set_replicas(...)`` must be idempotent: it should never
        introduce an extra replica axis or retain counters with incompatible
        shapes.
        """
        if self._states is None:
            return

        chain_ndim  = len(self._shape) + 1
        base_states = self._states[0] if self._states.ndim == chain_ndim + 1 else self._states
        int_dtype   = DEFAULT_JP_INT_TYPE if self._isjax else DEFAULT_NP_INT_TYPE

        if self._is_pt:
            target_shape = (self._n_replicas,) + tuple(base_states.shape)
            if self._isjax:
                self._states        = jnp.broadcast_to(base_states[None, ...], target_shape)
                self._num_proposed  = jnp.zeros((self._n_replicas, self._numchains), dtype=int_dtype)
                self._num_accepted  = jnp.zeros((self._n_replicas, self._numchains), dtype=int_dtype)
            else:
                self._states        = np.repeat(base_states[None, ...], self._n_replicas, axis=0)
                self._num_proposed  = np.zeros((self._n_replicas, self._numchains), dtype=int_dtype)
                self._num_accepted  = np.zeros((self._n_replicas, self._numchains), dtype=int_dtype)
        else:
            self._states        = base_states
            self._num_proposed  = self._backend.zeros(self._numchains, dtype=int_dtype)
            self._num_accepted  = self._backend.zeros(self._numchains, dtype=int_dtype)

        self._logprobas = None

    @staticmethod
    def _callable_accepts_positional_args(func: Optional[Callable], n_positional: int) -> bool:
        """
        Conservative signature check for cache-aware delta hooks.

        Returns True when ``func`` can be called with at least ``n_positional``
        positional arguments (or has ``*args``), False otherwise.
        """
        if func is None:
            return False
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        params            = list(sig.parameters.values())
        accepts_varargs   = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if accepts_varargs:
            return True

        positional_slots  = sum(p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in params)
        return positional_slots >= int(n_positional)

    def _resolve_fast_update_fun(self, current_fun, upd_rule, **kwargs):
        """Attempts to replace the update function with a version that returns update info."""
        spin, spin_value = resolve_state_defaults(getattr(self, "_state_representation", None), spin=kwargs.get("spin", None),
            mode_repr=kwargs.get("mode_repr", None), fallback_mode_repr=getattr(self, "_mode_repr", 1.0),
        )

        def _rule_supported(name: str) -> bool:
            supported = getattr(self._net, "fast_update_supported_rules", None)
            if supported is None:
                return True
            normalized = {str(x).upper() for x in supported}
            return name.upper() in normalized

        # Simple mapping based on function name or rule
        try:
            # If upd_rule is string/Enum, we can map directly
            if isinstance(upd_rule, (str, Enum)):
                from .sampler import UpdateRule

                if isinstance(upd_rule, str):
                    upd_rule = UpdateRule.from_str(upd_rule)

                # Try to match the update rule to a known proposer with info function
                if upd_rule == UpdateRule.LOCAL and _rule_supported("LOCAL"):
                    return _bind_named_partial(propose_local_flip_with_info, spin=spin, spin_value=spin_value)
                
                elif upd_rule == UpdateRule.EXCHANGE and _rule_supported("EXCHANGE"):
                    from .updates import get_neighbor_table

                    hilbert         = kwargs.get("hilbert")
                    order           = kwargs.get("exchange_order", 1)
                    neighbor_table  = get_neighbor_table(hilbert.lattice, order=order)
                    return partial(propose_exchange_with_info, neighbor_table=neighbor_table)
                
                elif upd_rule == UpdateRule.MULTI_FLIP and _rule_supported("MULTI_FLIP"):
                    n_flip = kwargs.get("n_flip", 1)
                    return _bind_named_partial(propose_multi_flip_with_info, n_flip=n_flip, spin=spin, spin_value=spin_value)
                
                elif upd_rule in [UpdateRule.GLOBAL, UpdateRule.PLAQUETTE, UpdateRule.WILSON] and _rule_supported("GLOBAL"):
                    if (isinstance(current_fun, partial) and current_fun.func.__name__ == "propose_global_flip"):
                        return _bind_named_partial(propose_global_flip_with_info, **current_fun.keywords)

                elif upd_rule == UpdateRule.BOND_FLIP and _rule_supported("BOND_FLIP"):
                    from .updates import get_neighbor_table

                    hilbert         = kwargs.get("hilbert")
                    order           = kwargs.get("bond_order", 1)
                    neighbor_table  = get_neighbor_table(hilbert.lattice, order=order)
                    return _bind_named_partial(
                        propose_bond_flip_with_info,
                        neighbor_table=neighbor_table,
                        spin=spin,
                        spin_value=spin_value,
                    )
                    
                elif upd_rule == UpdateRule.WORM and _rule_supported("WORM"):
                    from .updates import get_neighbor_table

                    hilbert         = kwargs.get("hilbert")
                    order           = kwargs.get("bond_order", 1)
                    neighbor_table  = get_neighbor_table(hilbert.lattice, order=order)
                    length          = kwargs.get("length", 4)
                    return _bind_named_partial(
                        propose_worm_flip_with_info,
                        neighbor_table=neighbor_table,
                        length=length,
                        spin=spin,
                        spin_value=spin_value,
                    )

            # If default was used
            if (current_fun == propose_local_flip or (isinstance(current_fun, partial) and current_fun.func == propose_local_flip)) and _rule_supported("LOCAL"):
                return _bind_named_partial(propose_local_flip_with_info, spin=spin, spin_value=spin_value)

        except ImportError:
            pass
        return current_fun

    #####################################################################
    # Public API
    #####################################################################

    def reset(self):
        """
        Reset the sampler state.
        Ensures proper state shaping for Parallel Tempering if enabled.
        """
        super().reset()
        self.invalidate_parameter_cache(require_thermalization=True)
        self._needs_thermalization = True
        self._apply_replica_layout()

    def __repr__(self):
        init_str    = str(self._initstate_t) if self._initstate_t is not None else "RND"
        glob_p      = getattr(self, "_global_p", 0.0)
        glob_str    = f",glob={self._global_name}" + (f"(p={glob_p:.2f})" if glob_p > 0 else "")
        return (
            f"VMC(s={self._shape},mu={self._mu},beta={self._beta},"
            f"Nt={self._therm_steps},Ns={self._sweep_steps},"
            f"Nsam={self._numsamples},Nch={self._numchains},"
            f"u={self._upd_rule_name},g={self._numupd}{glob_str},"
            f"initstate={init_str})"
        )

    def __str__(self):
        total_therm_updates_display     = self._total_therm_updates
        total_sample_updates_display    = self._numsamples * self._updates_per_sample
        init_str                        = str(self._initstate_t) if self._initstate_t is not None else "RND"
        glob_p                          = getattr(self, "_global_p", 0.0)

        return (
            f"MCSampler:\n"
            f"  - State shape: {self._shape} (Size: {self.size})\n"
            f"  - Backend: {self._backendstr}\n"
            f"  - Chains: {self._numchains}, Samples/Chain: {self._numsamples}\n"
            f"  - Params: mu={self._mu:.3f}, beta={self._beta:.3f}, logprob_fact={self._logprob_fact:.3f}\n"
            f"  - Update Rule: {self._upd_rule_name} (Steps/Update: {self._numupd})\n"
            f"  - Global Updates: {self._global_name} (p={glob_p:.2f})\n"
            f"  - Initial State: {init_str}\n"
            f"  - Replicas (PT): {self._pt_betas if getattr(self, '_is_pt', False) else 1}\n"
            f"  - Thermalization: {self._therm_steps} sweeps x {self._sweep_steps} steps/sweep ({total_therm_updates_display} total site updates/chain)\n"
            f"  - Sampling: {self._updates_per_sample} steps/sample ({total_sample_updates_display} total site updates/chain)\n"
        )

    def _resolve_network_hooks(self, net, adapter=None):
        """
        Resolve the sampler-facing network hooks once during setup.
        """
        self._net = net
        
        # First check if the network adapter provides sampling hooks for this network type
        # This means apply functions etc.
        if resolve_sampling_hooks_for_network is not None:
            return resolve_sampling_hooks_for_network(net, adapter=adapter)

        if adapter is not None and hasattr(adapter, "resolve_sampling_hooks"):
            hooks = adapter.resolve_sampling_hooks()
            if hooks is not None:
                return hooks

        if isinstance(net, nn.Module):
            net_callable, parameters = net.apply, None
        elif hasattr(net, "get_apply") and callable(net.get_apply):
            net_callable, parameters = net.get_apply()
        elif hasattr(net, "apply") and callable(net.apply):
            parameters = net.get_params() if hasattr(net, "get_params") and callable(net.get_params) else None
            net_callable = net.apply
        elif callable(net):
            net_callable, parameters = net, None
        else:
            raise ValueError("Invalid network object provided. Needs to be callable or have an 'apply' method.")

        log_psi_delta_fun = getattr(net, "log_psi_delta", None)
        if log_psi_delta_fun is None and hasattr(net, "get_log_psi_delta"):
            log_psi_delta_fun = net.get_log_psi_delta()

        log_psi_delta_cache_init_fun = getattr(net, "init_log_psi_delta_cache", None)
        if log_psi_delta_cache_init_fun is None and hasattr(net, "get_log_psi_delta_cache_init"):
            log_psi_delta_cache_init_fun = net.get_log_psi_delta_cache_init()

        supports_cache = bool(getattr(net, "log_psi_delta_supports_cache", False))

        return {
            "apply_fun"                     : net_callable,
            "parameters"                    : parameters,
            "log_psi_delta"                 : log_psi_delta_fun,
            "log_psi_delta_cache_init"      : log_psi_delta_cache_init_fun,
            "log_psi_delta_supports_cache"  : supports_cache,
        }

    @staticmethod
    def _parameter_cache_token(params) -> Optional[Tuple]:
        """Return a cheap identity token for the active parameter PyTree."""
        if params is None:
            return None
        try:
            leaves = jax.tree_util.tree_leaves(params) if JAX_AVAILABLE else []
        except Exception:
            leaves = []
        if not leaves:
            return (id(params),)
        return tuple(
            (
                id(leaf),
                tuple(getattr(leaf, "shape", ())),
                str(getattr(leaf, "dtype", "")),
            )
            for leaf in leaves
        )

    def invalidate_parameter_cache(self, *, require_thermalization: bool = False):
        """Discard cached log-amplitudes after the network parameters changed."""
        self._logprobas             = None
        self._params_cache_token    = None
        if require_thermalization:
            self._needs_thermalization = True

    # ---------------------------------------------------------------------

    def set_update_num(self, numupd: int):
        ''' Set the number of times the update function is applied per update step. '''
        numupd          = self._normalize_count(numupd, name="numupd", minimum=1)
        self._numupd    = numupd
        if self._org_upd_fun is None:
            raise ValueError("Original update function is not set.")

        if self._numupd > 1:
            n_updates = self._numupd
            base_update_fn = self._org_upd_fun

            if self._isjax:

                def _multi_update_proposer(state, key):
                    keys = jax.random.split(key, n_updates)

                    def body_fn(curr_state, sub_key):
                        res = base_update_fn(curr_state, sub_key)
                        # Handle potential tuple return (new_state, info)
                        if isinstance(res, tuple):
                            return res[0], None
                        return res, None

                    final_state, _ = jax.lax.scan(body_fn, state, keys)
                    return final_state

            else:

                def _multi_update_proposer(state, rng):
                    curr_state = state
                    for _ in range(n_updates):
                        curr_state = base_update_fn(curr_state, rng)
                    return curr_state

            self._upd_fun = _multi_update_proposer
        else:
            self._upd_fun = self._org_upd_fun

        if hasattr(self, "_jit_cache"):
            self._invalidate_runtime(require_thermalization=True)

    def set_hybrid_proposer(
        self,
        p_global        : float = 0.0,
        fraction        : float = 0.5,
        patterns        : Optional[Union[List, str]] = None,
        global_update   : Optional[Union[Callable, str]] = None,
    ):
        ''' 
        Configure a hybrid local/global update proposer. Global updates are applied with probability 
        p_global and can be based on predefined patterns or custom functions.

        Parameters
        ----------
        p_global : float
            Probability of applying a global update instead of a local one.
        fraction : float
            For multi-flip global updates, the fraction of sites to flip.
        patterns : list or str
            If str, can be "wilson", "plaquette", or "plaquette_all" to use lattice-based patterns.
            If list, should be a list of lists of site indices defining the patterns.
        global_update : callable or str
            A custom global update function or a string identifier for a predefined global update.
        '''
        self._org_upd_fun = self._local_upd_fun
        if p_global > 0.0 and self._isjax:

            def _fun_name(fun):
                if isinstance(fun, partial):
                    return getattr(fun.func, "__name__", "")
                return getattr(fun, "__name__", "")

            def _strip_update_info(proposer):
                def _wrapped(state, key):
                    out = proposer(state, key)
                    if isinstance(out, tuple):
                        return out[0]
                    return out

                return _wrapped

            local_returns_info = _fun_name(self._local_upd_fun).endswith("_with_info")

            def _resolve_plaquettes():
                lat = self._hilbert.lattice
                open_bc = True
                if hasattr(lat, "periodic_flags"):
                    try:
                        pbcx, pbcy, _   = lat.periodic_flags()
                        open_bc         = not (bool(pbcx) and bool(pbcy))
                    except Exception:
                        open_bc         = True
                try:
                    return lat.calculate_plaquettes(open_bc=open_bc)
                except TypeError:
                    try:
                        return lat.calculate_plaquettes(use_obc=open_bc)
                    except Exception:
                        return lat.calculate_plaquettes()
                except Exception:
                    return lat.calculate_plaquettes()

            def _resolve_patterns(name):
                if self._hilbert is None or self._hilbert.lattice is None:
                    raise ValueError(f"Hilbert space with lattice is required for '{name}' global updates.")
                
                lname = str(name).lower()
                if lname == "wilson":
                    return self._hilbert.lattice.calculate_wilson_loops()
                
                elif lname in ("plaquette", "plaquettes"):
                    return _resolve_plaquettes()
                
                elif lname in ("plaquette_all", "all_plaquettes", "plaquettes_all"):
                    all_plaq = _resolve_plaquettes()
                    if len(all_plaq) == 0:
                        raise ValueError("No plaquettes available for 'plaquette_all' update.")
                    merged = sorted({int(s) for p in all_plaq for s in p})
                    return [merged]
                else:
                    raise ValueError(f"Unknown global update pattern: {name}")

            if isinstance(global_update, str):
                patterns = _resolve_patterns(global_update)
                self._global_name = global_update
                global_update = None

            if isinstance(patterns, str):
                patterns = _resolve_patterns(patterns)

            if global_update is not None:
                if local_returns_info:

                    def _custom_global_with_info(state, key):
                        out = global_update(state, key)
                        if isinstance(out, tuple):
                            return out
                        return out, jnp.zeros((0,), dtype=jnp.int32)

                    global_flip_fn = _custom_global_with_info
                else:
                    global_flip_fn = global_update
                glob_type_str = "Custom Global Update"
                self._global_name = "custom"

            elif patterns is not None:
                try:
                    from .updates import propose_global_flip
                except ImportError as e:
                    raise ImportError(
                        "Failed to import propose_global_flip from updates module."
                    ) from e

                if len(patterns) == 0:
                    raise ValueError("Patterns list is empty.")

                max_len         = max(len(p) for p in patterns)
                n_patterns      = len(patterns)
                patterns_arr    = np.full((n_patterns, max_len), -1, dtype=np.int32)
                for i, p in enumerate(patterns):
                    patterns_arr[i, : len(p)] = p
                patterns_jax = jnp.array(patterns_arr)

                if local_returns_info:
                    try:
                        from .updates.spin_jax import propose_global_flip_with_info as _global_with_info
                    except ImportError as e:
                        raise ImportError("Local updater returns update-info but propose_global_flip_with_info is unavailable.") from e
                    
                    global_flip_fn = partial(
                        _global_with_info,
                        patterns=patterns_jax,
                        spin=bool(self._spin_repr),
                        spin_value=float(self._mode_repr),
                    )
                else:
                    global_flip_fn = partial(
                        propose_global_flip,
                        patterns=patterns_jax,
                        spin=bool(self._spin_repr),
                        spin_value=float(self._mode_repr),
                    )
                
                glob_type_str = f"Patterns (N={n_patterns})"

            else:
                n_flip_global = max(1, int(self._size * fraction))
                if local_returns_info:
                    try:
                        from .updates.spin_jax import propose_multi_flip_with_info as _multi_with_info
                    except ImportError as e:
                        raise ImportError("Local updater returns update-info but propose_multi_flip_with_info is unavailable.") from e
                    
                    global_flip_fn = partial(
                        _multi_with_info,
                        n_flip=n_flip_global,
                        spin=bool(self._spin_repr),
                        spin_value=float(self._mode_repr),
                    )
                else:
                    global_flip_fn = partial(
                        propose_multi_flip,
                        n_flip=n_flip_global,
                        spin=bool(self._spin_repr),
                        spin_value=float(self._mode_repr),
                    )
                glob_type_str       = f"Multi-Flip (frac={fraction:.2f})"
                self._global_name   = "multi-flip"

            local_for_hybrid    = self._local_upd_fun
            global_for_hybrid   = global_flip_fn
            if local_returns_info:
                # Hybrid updates combine potentially different move families.
                # Stripping update-info keeps both branches shape-compatible for JAX cond.
                local_for_hybrid    = _strip_update_info(self._local_upd_fun)
                global_for_hybrid   = _strip_update_info(global_flip_fn)

            self._org_upd_fun   = make_hybrid_proposer(local_for_hybrid, global_for_hybrid, p_global)
            self._global_name   = glob_type_str
            self._global_p      = p_global
        else:
            self._global_name   = "none"
            self._global_p      = 0.0

        self.set_update_num(self._numupd)

    def set_global_update(
        self,
        p_global            : Optional[float] = None,
        global_fraction     : float = 0.5,
        global_update       : Optional[Union[Callable, str]] = None,
        patterns            : Optional[Union[List, str]] = None,
        **kwargs,
    ):
        """
        Configure hybrid local/global proposer.

        Supports both ``p_global`` and ``global_p`` naming for compatibility
        with older and newer ``NQS.train(...)`` call paths.
        """
        if p_global is None:
            p_global = kwargs.pop("global_p", kwargs.pop("p_global", 0.0))
        
        self.set_hybrid_proposer(
            p_global        =   p_global,
            fraction        =   global_fraction,
            patterns        =   patterns,
            global_update   =   global_update,
        )

    # ---------------------------------------------------------------------
    # Jit-compiled and backend-dispatched implementations of core sampling functions
    # ---------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _acceptance_probability_jax(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        ''' Calculate the Metropolis-Hastings acceptance probability for a proposed move. '''
        delta = jnp.real(candidate_val) - jnp.real(current_val)
        ratio = jnp.exp(jnp.minimum(beta * mu * delta, 30.0))
        return ratio

    @staticmethod
    def _flatten_batch_output_jax(values, batch_size: int):
        """
        Normalize network output to shape ``(batch_size,)``.
        """
        out = jnp.asarray(values)
        if out.ndim == 0:
            return jnp.broadcast_to(out, (batch_size,))
        if out.shape[0] != batch_size:
            out = out.reshape((batch_size, -1))
            return out[:, 0]
        if out.ndim == 1:
            return out
        return out.reshape((batch_size, -1))[:, 0]

    def acceptance_probability(self, current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        ''' Public method to calculate acceptance probability, dispatching to the appropriate backend implementation. '''
        use_beta = beta if beta is not None else self._beta
        if self._isjax:
            return self._acceptance_probability_jax(current_val, candidate_val, beta=use_beta, mu=mu)
        return vmc_np_impl.acceptance_probability_np(current_val, candidate_val, beta=use_beta, mu=mu)


    @staticmethod
    @partial(jax.jit, static_argnames=("net_callable",))
    def _logprob_jax(x, net_callable, net_params=None):
        ''' Calculate log-probability (log-amplitude) for a batch of states `x` using the provided network callable and parameters. JIT-compiled. '''
        batched_log_psi = net_callable(net_params, x)
        flat_log_psi    = VMCSampler._flatten_batch_output_jax(batched_log_psi, x.shape[0])
        return jnp.real(flat_log_psi)

    def logprob(self, x, net_callable=None, net_params=None):
        ''' Calculate log-probability (log-amplitude) for a batch of states `x` 
        using the provided or default network callable and parameters.'''
        use_callable    = net_callable if net_callable is not None else self._net_callable
        if net_params is not None:
            use_params = net_params
        elif hasattr(self._net, "get_params") and callable(self._net.get_params):
            use_params = self._net.get_params()
        elif hasattr(self._net, "params"):
            use_params = self._net.params
        else:
            use_params = self._parameters

        if use_callable is None:
            raise ValueError("No callable provided for log probability calculation.")

        if self._isjax:
            return VMCSampler._logprob_jax(x, use_callable, use_params)
        return vmc_np_impl.logprob_np(x, use_callable, use_params)

    ###################################################################
    #! UPDATE CHAIN
    ###################################################################

    @staticmethod
    def _run_mcmc_steps_jax(
        chain_init,
        current_val_init,
        rng_k_init,
        num_proposed_init,
        num_accepted_init,
        delta_cache_init,
        params,
        # static args
        steps               : int,
        update_proposer     : Callable,
        net_callable_fun    : Callable,
        mu                  : float = 2.0,
        beta                : float = 1.0,
        log_psi_delta_fun   : Optional[Callable] = None,
        log_psi_delta_cache_init_fun: Optional[Callable] = None,
        log_psi_delta_supports_cache: bool = False,
    ):
        r"""
        Runs multiple MCMC steps using lax.scan. JIT-compiled.
        Supports Fast Updates if log_psi_delta_fun is provided.
        
        Parameters
        ----------
        chain_init : array
            Initial states for the MCMC chains, shape (num_chains, ...).
        current_val_init : array
            Initial log-probability values for the chains, shape (num_chains,).
        rng_k_init : PRNGKey
            Initial random key for JAX random operations.
        num_proposed_init : int
            Initial count of proposed moves for the chains.
        num_accepted_init : int 
            Initial count of accepted moves for the chains.
        delta_cache_init : any
            Initial cache for log_psi_delta calculations, if supported.
        params : any
            Parameters to pass to the network and log_psi_delta functions.
        steps : int
            Number of MCMC steps to perform.
        update_proposer : Callable
            Function that takes (chain_state, rng_key) and returns proposed new state (and optionally update info).
        net_callable_fun : Callable
            Function that takes (params, states) and returns log-probabilities for the given states 
            (should support batching over states).
        mu : float
            Exponent for acceptance probability calculation. 
        beta : float
            Inverse temperature for acceptance probability calculation.
        log_psi_delta_fun : Optional[Callable]
            If provided, a function that calculates the change in log-probability for a proposed move
            using the current state, proposed state, and update info. Should return either just the delta or a tuple of (delta, cache).
        log_psi_delta_cache_init_fun : Optional[Callable]
            If provided, a function that initializes the cache for log_psi_delta calculations based on the current chain state. Used if log_psi_delta_fun supports caching.
        log_psi_delta_supports_cache : bool
            If True, indicates that log_psi_delta_fun returns cache information that should be used to update the cache for accepted moves. If False, the cache will only be initialized once and not updated, which is suitable for cases where the cache is static or only depends on the current state and not the proposed state.  
        """

        # Optimized: assume net_callable_fun handles batches directly (no vmap)
        num_chains      = chain_init.shape[0]
        logproba_fun    = lambda x: net_callable_fun(params, x)
        proposer_fn     = jax.vmap(update_proposer, in_axes=(0, 0))

        def _sweep_chain_jax_step_inner(carry, _):
            ''' Performs a single MCMC update step for all chains in parallel. '''
            chain_in, current_val_in, current_key, num_prop, num_acc, cache_in = carry
            value_dtype = current_val_in.dtype

            key_step, next_key_carry    = jax.random.split(current_key)
            key_prop_base, key_acc      = jax.random.split(key_step)
            chain_prop_keys             = jax.random.split(key_prop_base, num_chains)

            #! Single MCMC update step logic
            # Propose update
            proposal_res = proposer_fn(chain_in, chain_prop_keys)

            # Handle fast updates / tuple return
            if isinstance(proposal_res, tuple):
                new_chain, update_info = proposal_res
            else:
                new_chain, update_info = proposal_res, None

            cache_prop = None
            # Calculate new log-probability
            if log_psi_delta_fun is not None and update_info is not None:
                if log_psi_delta_supports_cache and cache_in is not None:
                    delta_res = log_psi_delta_fun(params, current_val_in, chain_in, update_info, cache_in)
                else:
                    delta_res = log_psi_delta_fun(params, current_val_in, chain_in, update_info)

                if isinstance(delta_res, tuple):
                    delta_log_psi, cache_prop = delta_res
                else:
                    delta_log_psi = delta_res
                delta_log_psi   = jnp.asarray(delta_log_psi, dtype=value_dtype)
                new_val         = current_val_in + delta_log_psi
                delta           = jnp.real(delta_log_psi)
            else:
                # O(N_sites * N_hidden) standard update
                new_val = VMCSampler._flatten_batch_output_jax(logproba_fun(new_chain), num_chains)
                new_val = jnp.asarray(new_val, dtype=value_dtype)
                delta = jnp.real(new_val - current_val_in)

            # Work in log-space for numerical stability:
            # accept iff log(u) < min(0, log_acc).
            log_acc         = jnp.minimum(beta * mu * delta, 0.0)

            #! Calculate acceptance
            # Guard against log(0) from rare exact-zero draws.
            u               = jax.random.uniform(key_acc, shape=(num_chains,), minval=1e-12, maxval=1.0)
            log_u           = jnp.log(u)
            keep            = log_u < log_acc
            mask            = keep.reshape((num_chains,) + (1,) * (chain_in.ndim - 1))
            chain_out       = jnp.where(mask, new_chain, chain_in)
            val_out         = jnp.where(keep, new_val, current_val_in)
            proposed_out    = num_prop + 1
            accepted_out    = num_acc + keep.astype(num_acc.dtype)

            #! Update cache: keep old where rejected, update where accepted, or init if no cache.
            if cache_in is None:
                cache_out = None
            elif cache_prop is None:
                if log_psi_delta_cache_init_fun is None:
                    cache_out = cache_in
                elif not log_psi_delta_supports_cache:
                    cache_out = cache_in
                else:
                    cache_out = lax.cond(jnp.any(keep),
                        lambda _: log_psi_delta_cache_init_fun(params, chain_out),
                        lambda _: cache_in,
                        operand=None,
                    )
                # Keep the previous cache when the delta hook does not return an
                # updated cache object. Re-initializing cache from scratch on each
                # accepted move introduces a severe hot-path overhead.
                cache_out = cache_in
            else:
                def _select_cache(new_leaf, old_leaf):
                    cache_mask = keep.reshape((num_chains,) + (1,) * (new_leaf.ndim - 1))
                    return jnp.where(cache_mask, new_leaf, old_leaf)

                cache_out = jax.tree_util.tree_map(_select_cache, cache_prop, cache_in)

            new_carry = (chain_out, val_out, next_key_carry, proposed_out, accepted_out, cache_out)
            return new_carry, None

        initial_carry   = (chain_init, current_val_init, rng_k_init, num_proposed_init, num_accepted_init, delta_cache_init)
        final_carry, _  = jax.lax.scan(_sweep_chain_jax_step_inner, initial_carry, None, length=steps)
        final_chain, final_val, final_key, final_prop, final_acc, final_cache = final_carry

        return final_chain, final_val, final_key, final_prop, final_acc, final_cache

    ###################################################################
    #! SAMPLING
    ###################################################################

    ###################################################################
    #! STATIC JAX SAMPLING KERNEL
    ###################################################################

    @staticmethod
    def _batched_network_apply(params, configs, net_apply, chunk_size=2048):
        if chunk_size is None or int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        
        chunk_size  = int(chunk_size)
        n_samples   = configs.shape[0]
        if n_samples == 0:
            return jnp.empty((0,), dtype=jnp.result_type(configs.dtype, jnp.float32))

        if n_samples <= chunk_size:
            out_direct = net_apply(params, configs)
            return VMCSampler._flatten_batch_output_jax(out_direct, n_samples)

        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        pad_size = n_chunks * chunk_size - n_samples

        if pad_size > 0:
            pad_cfg         = [(0, pad_size)] + [(0, 0)] * (configs.ndim - 1)
            configs_padded  = jnp.pad(configs, pad_cfg, mode="edge")
        else:
            configs_padded  = configs

        configs_reshaped = configs_padded.reshape((n_chunks, chunk_size) + configs.shape[1:])

        def scan_fn(carry, x_chunk):
            out_chunk = VMCSampler._flatten_batch_output_jax(net_apply(params, x_chunk), chunk_size)
            return carry, out_chunk

        _, outputs = jax.lax.scan(scan_fn, None, configs_reshaped)
        outputs_flat = outputs.reshape(-1)

        if pad_size > 0:
            outputs_flat = outputs_flat[:n_samples]
        return outputs_flat

    @staticmethod
    def _init_delta_cache_jax(params, states_init, log_psi_delta_cache_init_fun=None, log_psi_delta_supports_cache=False):
        """Initialize fast-update cache when the active network supports it."""
        if log_psi_delta_supports_cache and log_psi_delta_cache_init_fun is not None:
            return log_psi_delta_cache_init_fun(params, states_init)
        return None

    @staticmethod
    def _init_logprobas_jax(params, states_init, logprobas_init, net_callable_fun, *, shape, network_eval_chunk_size=2048):
        """Initialize log-amplitudes only when they are not already cached."""
        if logprobas_init is not None:
            return logprobas_init

        states_flat     = states_init.reshape((-1,) + shape)
        logprobas_flat  = VMCSampler._batched_network_apply(params, states_flat, net_callable_fun, chunk_size=network_eval_chunk_size)
        return logprobas_flat.reshape(states_init.shape[: states_init.ndim - len(shape)])

    @staticmethod
    def _flatten_sample_outputs_jax(collected_states, collected_logprobas, shape):
        """Flatten collected states and log-amplitudes to estimator layout."""
        configs_flat    = collected_states.reshape((-1,) + shape)
        logprobas_flat  = collected_logprobas.reshape((-1,))
        return configs_flat, logprobas_flat

    @staticmethod
    def _compute_sample_probs_jax(log_ansatz, total_samples, log_prob_exponent, uniform_weights=False):
        """Normalize sample reweighting factors for estimator use."""
        prob_dtype = jnp.real(log_ansatz).dtype
        if uniform_weights:
            return None

        log_unnorm  = log_prob_exponent * jnp.real(log_ansatz)
        log_max     = jnp.max(log_unnorm)
        probs       = jnp.exp(log_unnorm - log_max)
        norm_factor = jnp.maximum(jnp.sum(probs), 1e-10)
        return probs / norm_factor * total_samples

    @staticmethod
    def _static_sample_jax(
        states_init,
        logprobas_init,
        rng_k_init,
        num_proposed_init,
        num_accepted_init,
        params,
        num_samples,
        num_chains,
        total_therm_updates,
        updates_per_sample,
        shape,
        mu,
        beta,
        logprob_fact,
        update_proposer,
        net_callable_fun,
        log_psi_delta_fun=None,
        log_psi_delta_cache_init_fun=None,
        log_psi_delta_supports_cache=False,
        uniform_weights=False,
        network_eval_chunk_size=2048,
    ):

        logprobas_init      = VMCSampler._init_logprobas_jax(params, states_init, logprobas_init, net_callable_fun, shape=shape, network_eval_chunk_size=network_eval_chunk_size)
        delta_cache_init    = VMCSampler._init_delta_cache_jax(params, states_init, log_psi_delta_cache_init_fun=log_psi_delta_cache_init_fun, log_psi_delta_supports_cache=log_psi_delta_supports_cache)

        #! Phase 1: Thermalization - run MCMC steps without collecting samples to reach equilibrium distribution.
        (
            states_therm,
            logprobas_therm,
            rng_k_therm,
            num_proposed_therm,
            num_accepted_therm,
            delta_cache_therm,
        ) = (
            VMCSampler._run_mcmc_steps_jax(
                chain_init=states_init,
                current_val_init=logprobas_init,
                rng_k_init=rng_k_init,
                num_proposed_init=num_proposed_init,
                num_accepted_init=num_accepted_init,
                delta_cache_init=delta_cache_init,
                params=params,
                steps=total_therm_updates,
                update_proposer=update_proposer,
                net_callable_fun=net_callable_fun,
                mu=mu,
                beta=beta,
                log_psi_delta_fun=log_psi_delta_fun,
                log_psi_delta_cache_init_fun=log_psi_delta_cache_init_fun,
                log_psi_delta_supports_cache=log_psi_delta_supports_cache,
            )
        )

        #! Phase 2: Sampling - run MCMC steps while collecting samples at specified intervals. Use lax.scan to efficiently collect samples across multiple steps.
        def sample_scan_body(carry, _):
            new_carry = VMCSampler._run_mcmc_steps_jax(
                chain_init=carry[0],
                current_val_init=carry[1],
                rng_k_init=carry[2],
                num_proposed_init=carry[3],
                num_accepted_init=carry[4],
                delta_cache_init=carry[5],
                params=params,
                steps=updates_per_sample,
                update_proposer=update_proposer,
                net_callable_fun=net_callable_fun,
                mu=mu,
                beta=beta,
                log_psi_delta_fun=log_psi_delta_fun,
                log_psi_delta_cache_init_fun=log_psi_delta_cache_init_fun,
                log_psi_delta_supports_cache=log_psi_delta_supports_cache,
            )
            return new_carry, (new_carry[0], new_carry[1])

        initial_scan_carry = (
            states_therm,       # states after thermalization
            logprobas_therm,    # log probabilities after thermalization - ansatz values for the thermalized states
            rng_k_therm,        # random key after thermalization
            num_proposed_therm, # number of proposed moves during thermalization
            num_accepted_therm, # number of accepted moves during thermalization
            delta_cache_therm,  # cache for log_psi_delta if supported, after thermalization
        )
        
        # Run the sampling phase, collecting states and log probabilities at each sample point. 
        final_carry, collected_samples = jax.lax.scan(sample_scan_body, initial_scan_carry, None, length=num_samples)

        collected_states, collected_logprobas   = collected_samples
        configs_flat, batched_log_ansatz        = VMCSampler._flatten_sample_outputs_jax(collected_states, collected_logprobas, shape)

        log_prob_exponent                       = 1.0 / logprob_fact - mu * beta # Reweight from sampled |psi|^(mu*beta) to target |psi|^(1/logprob_fact).
        total_samples                           = num_samples * num_chains
        probs_normalized                        = VMCSampler._compute_sample_probs_jax(batched_log_ansatz, total_samples, log_prob_exponent, uniform_weights=uniform_weights)
        fc_states, fc_lpsi, fc_key, fc_prop, fc_acc, _fc_cache = final_carry
        final_state_tuple                       = (fc_states, fc_lpsi, fc_key, fc_prop, fc_acc)
        return final_state_tuple, (configs_flat, batched_log_ansatz), probs_normalized

    ###################################################################
    #! PARALLEL TEMPERING (PT) SUPPORT
    ###################################################################

    @staticmethod
    @partial(jax.jit, static_argnames=("mu",))
    def _replica_exchange_kernel_jax_betas(log_psi, betas, key, mu):
        r"""
        Perform PT swaps by exchanging beta labels between adjacent replica slots.

        This implementation keeps updates pairwise-safe (no overlapping scatter writes),
        so chain-wise swap decisions cannot overwrite neighboring pairs.
        """
        n_replicas  = log_psi.shape[0]
        n_chains    = log_psi.shape[1]

        key, subkey = jax.random.split(key)
        swap_odd    = jax.random.bernoulli(subkey, 0.5)
        start_idx   = jnp.where(swap_odd, 1, 0)

        beta_i      = betas[:-1]
        beta_j      = betas[1:]
        log_psi_real = jnp.real(log_psi)
        lpsi_i      = log_psi_real[:-1]
        lpsi_j      = log_psi_real[1:]

        log_acc = (beta_i - beta_j) * mu * (lpsi_j - lpsi_i)

        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=(n_replicas - 1, n_chains), minval=1e-12, maxval=1.0)
        log_u           = jnp.log(u)
        accept_raw      = log_u < jnp.minimum(log_acc, 0.0)

        pair_indices    = jnp.arange(n_replicas - 1)[:, None]
        phase_mask      = (pair_indices % 2) == start_idx
        do_swap         = accept_raw & phase_mask

        # Pairwise-safe reconstruction: each slot can come from itself, slot+1, or slot-1.
        from_above      = jnp.concatenate([betas[1:], betas[-1:]], axis=0)
        from_below      = jnp.concatenate([betas[:1], betas[:-1]], axis=0)

        swap_down_mask  = jnp.zeros((n_replicas, n_chains), dtype=bool).at[:-1].set(do_swap)
        swap_up_mask    = jnp.zeros((n_replicas, n_chains), dtype=bool).at[1:].set(do_swap)

        betas_out       = jnp.where(swap_down_mask, from_above, betas)
        betas_out       = jnp.where(swap_up_mask, from_below, betas_out)

        return betas_out

    @staticmethod
    def _run_pt_round_jax(
        carry,
        *,
        params,
        updates_per_sample,
        mu,
        n_replicas,
        update_proposer,
        log_psi_delta_fun,
        log_psi_delta_cache_init_fun,
        log_psi_delta_supports_cache,
        net_callable_fun,
    ):
        """Run one PT MCMC round followed by a beta-exchange step."""
        states, lpsi, key, n_prop, n_acc, delta_cache, betas = carry
        key, subkey     = jax.random.split(key)
        replica_keys    = jax.random.split(subkey, n_replicas)

        def _single_replica_mcmc(chain, chain_lpsi, chain_key, chain_prop, chain_acc, chain_cache, beta_r):
            return VMCSampler._run_mcmc_steps_jax(
                chain_init=chain,
                current_val_init=chain_lpsi,
                rng_k_init=chain_key,
                num_proposed_init=chain_prop,
                num_accepted_init=chain_acc,
                delta_cache_init=chain_cache,
                params=params,
                steps=updates_per_sample,
                update_proposer=update_proposer,
                net_callable_fun=net_callable_fun,
                mu=mu,
                beta=beta_r,
                log_psi_delta_fun=log_psi_delta_fun,
                log_psi_delta_cache_init_fun=log_psi_delta_cache_init_fun,
                log_psi_delta_supports_cache=log_psi_delta_supports_cache,
            )

        vmapped_mcmc_step = jax.vmap(_single_replica_mcmc, in_axes=(0, 0, 0, 0, 0, 0, 0))
        states_new, lpsi_new, keys_new, n_prop_new, n_acc_new, delta_cache_new = vmapped_mcmc_step(
            states, lpsi, replica_keys, n_prop, n_acc, delta_cache, betas
        )

        key, swap_key = jax.random.split(key)
        betas_swapped = VMCSampler._replica_exchange_kernel_jax_betas(lpsi_new, betas, swap_key, mu)
        return (states_new, lpsi_new, key, n_prop_new, n_acc_new, delta_cache_new, betas_swapped)

    @staticmethod
    def _collect_pt_physical_samples_jax(states, logprobas, betas, target_beta):
        """Collect the physical-temperature sector after one PT round."""
        is_physical         = jnp.abs(betas - target_beta) < 1e-5
        physical_indices    = jnp.argmax(is_physical, axis=0)
        chain_indices       = jnp.arange(states.shape[1])
        collected_states    = states[physical_indices, chain_indices]
        collected_logprobas = logprobas[physical_indices, chain_indices]
        return collected_states, collected_logprobas

    @staticmethod
    def _static_sample_pt_jax(
        states_init,
        logprobas_init,
        rng_k_init,
        num_proposed_init,
        num_accepted_init,
        params,
        num_samples,
        num_chains,
        n_replicas,
        total_therm_updates,
        updates_per_sample,
        shape,
        mu,
        pt_betas,
        logprob_fact,
        update_proposer,
        net_callable_fun,
        log_psi_delta_fun=None,
        log_psi_delta_cache_init_fun=None,
        log_psi_delta_supports_cache=False,
        uniform_weights=False,
        network_eval_chunk_size=2048,
    ):

        # Initial betas: (n_replicas, n_chains)
        # All chains in replica r start with pt_betas[r]
        current_betas       = jnp.broadcast_to(pt_betas[:, None], (n_replicas, num_chains))
        delta_cache_init    = VMCSampler._init_delta_cache_jax(params, states_init, log_psi_delta_cache_init_fun=log_psi_delta_cache_init_fun, log_psi_delta_supports_cache=log_psi_delta_supports_cache)
        logprobas_init      = VMCSampler._init_logprobas_jax(params, states_init, logprobas_init, net_callable_fun, shape=shape, network_eval_chunk_size=network_eval_chunk_size)

        def therm_scan_body(carry, _):
            new_carry = VMCSampler._run_pt_round_jax(
                carry,
                params=params,
                updates_per_sample=updates_per_sample,
                mu=mu,
                n_replicas=n_replicas,
                update_proposer=update_proposer,
                log_psi_delta_fun=log_psi_delta_fun,
                log_psi_delta_cache_init_fun=log_psi_delta_cache_init_fun,
                log_psi_delta_supports_cache=log_psi_delta_supports_cache,
                net_callable_fun=net_callable_fun,
            )
            return new_carry, None

        # Use ceil division so requested thermalization is not truncated.
        n_therm_sweeps = max(1, (total_therm_updates + updates_per_sample - 1) // updates_per_sample)
        initial_carry = (
            states_init,
            logprobas_init,
            rng_k_init,
            num_proposed_init,
            num_accepted_init,
            delta_cache_init,
            current_betas,
        )
        therm_carry, _ = jax.lax.scan(therm_scan_body, initial_carry, None, length=n_therm_sweeps)

        def sample_scan_body(carry, _):
            new_carry = VMCSampler._run_pt_round_jax(
                carry,
                params=params,
                updates_per_sample=updates_per_sample,
                mu=mu,
                n_replicas=n_replicas,
                update_proposer=update_proposer,
                log_psi_delta_fun=log_psi_delta_fun,
                log_psi_delta_cache_init_fun=log_psi_delta_cache_init_fun,
                log_psi_delta_supports_cache=log_psi_delta_supports_cache,
                net_callable_fun=net_callable_fun,
            )
            collected_states_step, collected_lpsi_step = VMCSampler._collect_pt_physical_samples_jax(
                new_carry[0],
                new_carry[1],
                new_carry[6],
                pt_betas[0],
            )
            return new_carry, (collected_states_step, collected_lpsi_step)

        final_carry, (collected_states, collected_log_psi) = jax.lax.scan(
            sample_scan_body, therm_carry, None, length=num_samples
        )

        # Flatten
        # collected_states: (num_samples, n_chains, *shape)
        configs_flat, ansatz_flat   = VMCSampler._flatten_sample_outputs_jax(collected_states, collected_log_psi, shape)
        total_samples               = num_samples * num_chains
        phys_beta                   = pt_betas[0]
        log_prob_exponent           = 1.0 / logprob_fact - mu * phys_beta

        probs_norm                  = VMCSampler._compute_sample_probs_jax(ansatz_flat, total_samples, log_prob_exponent, uniform_weights=uniform_weights)

        # Unpack final carry
        fc_states, fc_lpsi, fc_key, fc_prop, fc_acc, _fc_cache, fc_betas = final_carry

        # Restore canonical order (sort by temperature descending to match pt_betas)
        # pt_betas is assumed sorted descending.
        # We sort current states so that row i corresponds to largest beta i.
        # sort indices by -beta (descending)
        sorted_indices = jnp.argsort(-fc_betas, axis=0)

        # Reorder states and log_psi
        # states: (n_replicas, n_chains, *shape)
        # sorted_indices: (n_replicas, n_chains)
        # Expand indices for broadcasting
        # We need to broadcast over *shape dims
        take_indices = sorted_indices
        for _ in range(fc_states.ndim - 2):
            take_indices = take_indices[..., None]

        fc_states_sorted    = jnp.take_along_axis(fc_states, take_indices, axis=0)
        fc_lpsi_sorted      = jnp.take_along_axis(fc_lpsi, sorted_indices, axis=0)
        final_state_tuple   = (fc_states_sorted, fc_lpsi_sorted, fc_key, fc_prop, fc_acc)

        return final_state_tuple, (configs_flat, ansatz_flat), probs_norm

    # -----------------------------------------------------------------

    def _sample_reinitialize(self, used_num_chains, used_num_samples, used_num_therm=None):
        self._numchains = used_num_chains if used_num_chains is not None else self._numchains
        self._numsamples = used_num_samples if used_num_samples is not None else self._numsamples
        if used_num_therm is not None:
            self._therm_steps = used_num_therm
        self._refresh_sampling_schedule()
        self._needs_thermalization = True
        self.reset()
        current_states      = self._states
        current_proposed    = self._num_proposed
        current_accepted    = self._num_accepted
        self._invalidate_runtime(require_thermalization=False)

        return current_states, current_proposed, current_accepted

    def _finalize_sample_jax(self, final_state_tuple, samples_tuple, probs, *, pt: bool = False, params_token=None):
        """Persist sampler state after one compiled JAX sampling call."""
        final_states, final_logprobas, final_rng_k, final_num_proposed, final_num_accepted = final_state_tuple

        self._states                = final_states
        self._logprobas             = final_logprobas
        self._params_cache_token    = params_token
        self._num_proposed          = final_num_proposed
        self._num_accepted          = final_num_accepted
        self._rng_k                 = final_rng_k
        self._needs_thermalization  = False

        if pt:
            final_state_info = (self._states[0], self._logprobas[0])
        else:
            final_state_info = (self._states, self._logprobas)
        return final_state_info, samples_tuple, probs

    # -----------------------------------------------------------------
    #! Public sampling interface
    # -----------------------------------------------------------------

    def sample(self, parameters=None, num_samples=None, *, num_chains=None, num_therm=None, betas=None):
        used_num_samples    = num_samples if num_samples is not None else self._numsamples
        used_num_chains     = num_chains if num_chains is not None else self._numchains
        used_num_therm      = num_therm if num_therm is not None else self._therm_steps

        if betas is not None:
            self.set_replicas(betas)

        current_states      = self._states
        current_proposed    = self._num_proposed
        current_accepted    = self._num_accepted

        if (
            used_num_chains     != self._numchains
            or used_num_samples != self._numsamples
            or used_num_therm   != self._therm_steps
        ):
            if self._logger:
                self._logger.info(f"(Warning) Running sample with {used_num_chains} chains (instance default is {self._numchains}). State reinitialized for this call.")
            current_states, current_proposed, current_accepted = self._sample_reinitialize(used_num_chains, used_num_samples, used_num_therm)

        if parameters is not None:
            current_params = parameters
        elif hasattr(self._net, "get_params") and callable(self._net.get_params):
            current_params = self._net.get_params()
        elif hasattr(self._net, "params"):
            current_params = self._net.params
        else:
            current_params = self._parameters
        params_token = self._parameter_cache_token(current_params)
        if self._params_cache_token is not None and params_token != self._params_cache_token:
            self._logprobas = None

        if self._isjax:
            if not isinstance(self._rng_k, jax.Array):
                raise TypeError(f"Sampler's RNG key is invalid: {type(self._rng_k)}")

            if self._is_pt:
                effective_therm_steps = used_num_therm if self._needs_thermalization else 0
                if (
                    self._static_pt_sampler is None
                    or self._static_pt_therm_steps != effective_therm_steps
                ):
                    self._static_pt_sampler = self._get_sampler_pt_jax(
                        num_samples=used_num_samples,
                        num_chains=used_num_chains,
                        therm_steps=effective_therm_steps,
                    )
                    self._static_pt_therm_steps = effective_therm_steps

                final_state_tuple, samples_tuple, probs = self._static_pt_sampler(
                    states_init=current_states,
                    logprobas_init=self._logprobas,
                    rng_k_init=self._rng_k,
                    params=current_params,
                    num_proposed_init=current_proposed,
                    num_accepted_init=current_accepted,
                )
                return self._finalize_sample_jax(final_state_tuple, samples_tuple, probs, pt=True, params_token=params_token)

            effective_therm_steps = used_num_therm if self._needs_thermalization else 0
            if self._static_sample_fun is None or self._static_sample_therm_steps != effective_therm_steps:
                self._static_sample_fun = self.get_sampler(num_samples=used_num_samples, num_chains=self._numchains, therm_steps=effective_therm_steps)
                self._static_sample_therm_steps = effective_therm_steps

            final_state_tuple, samples_tuple, probs = self._static_sample_fun(
                states_init=current_states,
                logprobas_init=self._logprobas,
                rng_k_init=self._rng_k,
                params=current_params,
                num_proposed_init=current_proposed,
                num_accepted_init=current_accepted,
            )
            return self._finalize_sample_jax(final_state_tuple, samples_tuple, probs, pt=False, params_token=params_token)
        else:
            return vmc_np_impl.sample_np(self, current_params=current_params, used_num_samples=used_num_samples, net_callable=self._net_callable)

    # -----------------------------------------------------------------

    def _get_sampler_pt_jax(
        self,
        num_samples: Optional[int] = None,
        num_chains: Optional[int] = None,
        therm_steps: Optional[int] = None,
    ) -> Callable:
        ''' Returns a JIT-compiled function that 
        performs parallel tempering sampling with the specified parameters. 
        
        The returned function takes initial states, RNG key, and parameters as input and returns final states, samples, and probabilities.
        '''
        
        if not self._isjax:
            raise RuntimeError("PT JAX sampler only available for JAX backend.")

        static_num_samples  = num_samples if num_samples is not None else self._numsamples
        static_num_chains   = num_chains if num_chains is not None else self._numchains
        static_therm_steps  = self._therm_steps if therm_steps is None else int(therm_steps)

        baked_args = {
            "num_samples"                   : static_num_samples,
            "num_chains"                    : static_num_chains,
            "n_replicas"                    : self._n_replicas,
            "total_therm_updates"           : static_therm_steps * self._sweep_steps * self._size,
            "updates_per_sample"            : self._updates_per_sample,
            "shape"                         : self._shape,
            "mu"                            : self._mu,
            "pt_betas"                      : self._pt_betas,
            "logprob_fact"                  : self._logprob_fact,
            "update_proposer"               : self._upd_fun,
            "net_callable_fun"              : self._net_callable,
            "log_psi_delta_fun"             : self._log_psi_delta_fun,
            "log_psi_delta_cache_init_fun"  : self._log_psi_delta_cache_init_fun,
            "log_psi_delta_supports_cache"  : self._log_psi_delta_supports_cache,
            "uniform_weights"               : self.has_uniform_weights,
            "network_eval_chunk_size"       : self._network_eval_chunk_size,
        }

        partial_sampler = partial(VMCSampler._static_sample_pt_jax, **baked_args)
        int_dtype = DEFAULT_JP_INT_TYPE
        n_replicas = self._n_replicas

        def wrapped_pt_sampler_impl(
            states_init: jax.Array,
            logprobas_init: Optional[jax.Array],
            rng_k_init: jax.Array,
            params: Any,
            num_proposed_init: Optional[jax.Array] = None,
            num_accepted_init: Optional[jax.Array] = None,
        ):

            _num_proposed = (
                jnp.zeros((n_replicas, static_num_chains), dtype=int_dtype)
                if num_proposed_init is None
                else num_proposed_init
            )
            _num_accepted = (
                jnp.zeros((n_replicas, static_num_chains), dtype=int_dtype)
                if num_accepted_init is None
                else num_accepted_init
            )

            return partial_sampler(
                states_init=states_init,
                logprobas_init=logprobas_init,
                rng_k_init=rng_k_init,
                num_proposed_init=_num_proposed,
                num_accepted_init=_num_accepted,
                params=params,
            )

        if self._donate_buffers:
            return jax.jit(wrapped_pt_sampler_impl, donate_argnums=(0, 1))
        return jax.jit(wrapped_pt_sampler_impl)

    def _get_sampler_jax(
        self,
        num_samples: Optional[int] = None,
        num_chains: Optional[int] = None,
        therm_steps: Optional[int] = None,
    ) -> Callable:
        if not self._isjax:
            raise RuntimeError("Static JAX sampler getter only available for JAX backend.")

        static_num_samples  = num_samples if num_samples is not None else self._numsamples
        static_num_chains   = num_chains if num_chains is not None else self._numchains
        static_therm_steps  = self._therm_steps if therm_steps is None else int(therm_steps)

        baked_args = {
            "num_samples"                   : static_num_samples,
            "num_chains"                    : static_num_chains,
            "total_therm_updates"           : static_therm_steps * self._sweep_steps * self._size,
            "updates_per_sample"            : self._updates_per_sample,
            "shape"                         : self._shape,
            "update_proposer"               : self._upd_fun,
            "net_callable_fun"              : self._net_callable,
            "mu"                            : self._mu,
            "beta"                          : self._beta,
            "logprob_fact"                  : self._logprob_fact,
            "log_psi_delta_fun"             : self._log_psi_delta_fun,
            "log_psi_delta_cache_init_fun"  : self._log_psi_delta_cache_init_fun,
            "log_psi_delta_supports_cache"  : self._log_psi_delta_supports_cache,
            "uniform_weights"               : self.has_uniform_weights,
            "network_eval_chunk_size"       : self._network_eval_chunk_size,
        }

        partial_sampler = partial(VMCSampler._static_sample_jax, **baked_args)
        int_dtype       = DEFAULT_JP_INT_TYPE

        def wrapped_sampler_impl(
            states_init         : jax.Array,
            logprobas_init      : Optional[jax.Array],
            rng_k_init          : jax.Array,
            params              : Any,
            num_proposed_init   : Optional[jax.Array] = None,
            num_accepted_init   : Optional[jax.Array] = None,
        ):

            _num_proposed_init = (
                jnp.zeros(static_num_chains, dtype=int_dtype)
                if num_proposed_init is None
                else num_proposed_init
            )
            _num_accepted_init = (
                jnp.zeros(static_num_chains, dtype=int_dtype)
                if num_accepted_init is None
                else num_accepted_init
            )

            final_state_tuple, samples_tuple, probs = partial_sampler(
                states_init=states_init,
                logprobas_init=logprobas_init,
                rng_k_init=rng_k_init,
                num_proposed_init=_num_proposed_init,
                num_accepted_init=_num_accepted_init,
                params=params,
            )
            return final_state_tuple, samples_tuple, probs

        if self._donate_buffers:
            return jax.jit(wrapped_sampler_impl, donate_argnums=(0, 1))
        return jax.jit(wrapped_sampler_impl)

    def _get_sampler_np(
        self, num_samples: Optional[int] = None, num_chains: Optional[int] = None
    ) -> Callable:
        return vmc_np_impl.get_sampler_np(sampler=self, num_samples=num_samples, num_chains=num_chains)

    def get_sampler(self, num_samples: Optional[int] = None, num_chains: Optional[int] = None, therm_steps: Optional[int] = None) -> Callable:
        '''
        Get a sampler function with the given number of samples and chains baked in as static arguments.
        '''
        effective_therm_steps = self._therm_steps if therm_steps is None else int(therm_steps)
        cache_key = (
            num_samples if num_samples is not None else self._numsamples,
            num_chains  if num_chains  is not None else self._numchains,
            effective_therm_steps,
            self._isjax,
            self._is_pt,
            self._donate_buffers if self._isjax else False,
            self._updates_per_sample,
            self._mu,
            self._beta if not self._is_pt else None,
            tuple(np.asarray(self._pt_betas).tolist()) if self._is_pt and self._pt_betas is not None else None,
            id(self._upd_fun),
            id(self._net_callable),
            id(self._log_psi_delta_fun),
            id(self._log_psi_delta_cache_init_fun),
            bool(self._log_psi_delta_supports_cache),
            self._network_eval_chunk_size,
        )

        if hasattr(self, "_jit_cache") and cache_key in self._jit_cache:
            return self._jit_cache[cache_key]

        if self._isjax:
            if self._is_pt:
                sampler = self._get_sampler_pt_jax(num_samples, num_chains, therm_steps=effective_therm_steps)
            else:
                sampler = self._get_sampler_jax(num_samples, num_chains, therm_steps=effective_therm_steps)
        else:
            sampler = self._get_sampler_np(num_samples, num_chains)

        if hasattr(self, "_jit_cache"):
            self._jit_cache[cache_key] = sampler

        return sampler

    def set_update_function(self, upd_fun: Union[str, "Enum", Callable], **kwargs):
        if isinstance(upd_fun, (str, Enum)):
            from .sampler import get_update_function

            if "hilbert" not in kwargs and self._hilbert is not None:
                kwargs["hilbert"] = self._hilbert
            if "lattice" not in kwargs:
                if kwargs.get("hilbert"):
                    kwargs["lattice"] = kwargs["hilbert"].lattice

            self._local_upd_fun = get_update_function(
                upd_fun, backend="jax" if self._isjax else "numpy", **kwargs
            )
        else:
            self._local_upd_fun = upd_fun

        if self._isjax and self._log_psi_delta_fun is not None:
            self._local_upd_fun = self._resolve_fast_update_fun(
                self._local_upd_fun, upd_fun, **kwargs
            )

        self._org_upd_fun = self._local_upd_fun
        self.set_update_num(self._numupd)

    def set_apply_fun(
        self,
        apply_fun: Callable,
        *,
        log_psi_delta_fun: Optional[Callable] = None,
        log_psi_delta_cache_init_fun: Optional[Callable] = None,
        log_psi_delta_supports_cache: bool = False,
    ):
        """Replace the sampler log-amplitude callable and reset dependent caches."""
        self._net_callable                 = apply_fun
        self._log_psi_delta_fun            = log_psi_delta_fun
        self._log_psi_delta_cache_init_fun = log_psi_delta_cache_init_fun
        inferred_cache_support             = self._callable_accepts_positional_args(log_psi_delta_fun, 5)
        self._log_psi_delta_supports_cache = bool(log_psi_delta_fun is not None and (log_psi_delta_supports_cache or inferred_cache_support))
        self.invalidate_parameter_cache(require_thermalization=True)
        self._invalidate_runtime(require_thermalization=True)

    ###################################################################
    #! SETTERS
    ###################################################################

    def set_mu(self, mu):
        ''' Set the exponent mu for the sampler. This will trigger a refresh of the sampling schedule and recompilation of JIT functions if applicable.'''
        mu = float(mu)
        self._set_runtime_scalar("_mu", mu, distribution_changed=True)

    def set_beta(self, beta):
        ''' Set the inverse temperature beta for the sampler. This will trigger a refresh of the sampling schedule and recompilation of JIT functions if applicable.'''
        beta = float(beta)
        self._set_runtime_scalar("_beta", beta, distribution_changed=True)

    def set_therm_steps(self, therm_steps):
        ''' Set the number of thermalization steps. This will trigger a refresh of the sampling schedule and recompilation of JIT functions if applicable.'''
        therm_steps = self._normalize_count(therm_steps, name="therm_steps", minimum=0)
        self._set_runtime_scalar("_therm_steps", therm_steps, counts_changed=True)

    def set_sweep_steps(self, sweep_steps):
        ''' Set the number of MCMC sweeps per sample. This will trigger a refresh of the sampling schedule and recompilation of JIT functions if applicable.'''
        sweep_steps = self._normalize_count(sweep_steps, name="sweep_steps", minimum=1)
        self._set_runtime_scalar("_sweep_steps", sweep_steps, counts_changed=True)

    def set_replicas(self, betas: Optional[Union[List[float], np.ndarray, jnp.ndarray]], n_replicas: Optional[int] = None, min_beta: Optional[float] = None):
        if betas is None and (n_replicas is not None and n_replicas > 1):
            max_beta = float(self._beta)
            min_beta = float(min_beta if min_beta is not None else max_beta / 10.0)
            if max_beta <= 0.0 or min_beta <= 0.0:
                raise ValueError("PT beta ladder values must be positive.")
            if self._isjax:
                betas = jnp.logspace(np.log10(max_beta), np.log10(min_beta), n_replicas)
            else:
                betas = np.logspace(np.log10(max_beta), np.log10(min_beta), n_replicas)
        elif betas is not None:
            betas_np = np.asarray(betas, dtype=np.float64)
            if betas_np.ndim != 1:
                raise ValueError("PT beta ladder must be a one-dimensional array.")
            if np.any(betas_np <= 0.0):
                raise ValueError("PT beta ladder values must be positive.")
            if betas_np.shape[0] <= 1:
                betas = None
            elif self._isjax:
                betas = jnp.asarray(betas_np, dtype=jnp.result_type(jnp.float32, float(self._beta)))
            else:
                betas = betas_np
        current_betas = getattr(self, "_pt_betas", None)
        same_schedule = hasattr(self, "_pt_betas") and current_betas is betas
        if current_betas is not None and betas is not None and len(current_betas) == len(betas):
            same_schedule = bool(jnp.allclose(jnp.asarray(current_betas), jnp.asarray(betas))) if self._isjax else bool(np.allclose(np.asarray(current_betas), np.asarray(betas)))
        if same_schedule:
            return
        self._pt_betas      = betas
        self._is_pt         = self._pt_betas is not None
        self._n_replicas    = len(self._pt_betas) if self._is_pt else 1
        self._invalidate_runtime(refresh_layout=True, require_thermalization=True)

    def set_pt_betas(self, betas: Optional[Union[List[float], np.ndarray, jnp.ndarray]], n_replicas: Optional[int] = None, min_beta: Optional[float] = None):
        """Compatibility alias used by NQS.train(..., pt_betas=...)."""
        self.set_replicas(betas, n_replicas=n_replicas, min_beta=min_beta)

    def diagnose(self, samples: Optional[Union[np.ndarray, Any]] = None) -> dict:
        metrics = {"acceptance_rate": float(np.mean(np.array(self.accepted_ratio)))}

        if samples is None:
            return metrics

        if JAX_AVAILABLE and isinstance(samples, jax.Array):
            samples_np = np.array(samples)
        else:
            samples_np = np.array(samples)

        if np.iscomplexobj(samples_np):
            samples_np = np.real(samples_np)

        if samples_np.ndim == 1:
            n_chains = self.numchains
            n_total = samples_np.shape[0]
            if n_total % n_chains != 0:
                metrics["error"] = "Sample count not divisible by num_chains"
                return metrics
            n_samples = n_total // n_chains
            chains = samples_np.reshape((n_chains, n_samples))
        else:
            chains = samples_np

        metrics["r_hat"] = float(compute_rhat(chains))

        taus = []
        for c in chains:
            taus.append(compute_autocorr_time(c))
        avg_tau = np.mean(taus)
        metrics["tau"] = float(avg_tau)

        ess_total = np.sum([len(c) / t for c, t in zip(chains, taus)])
        metrics["ess"] = float(ess_total)

        return metrics

    @property
    def is_pt(self):
        return self._is_pt

    @property
    def pt_betas(self):
        return self._pt_betas

    @property
    def n_replicas(self):
        return self._n_replicas

    @property
    def total_therm_updates(self):
        return self._total_therm_updates

    @property
    def updates_per_sample(self):
        return self._updates_per_sample

    @property
    def logprob_fact(self):
        return self._logprob_fact

    @property
    def mu(self):
        return self._mu

    @property
    def beta(self):
        return self._beta

    @property
    def therm_steps(self):
        return self._therm_steps

    @property
    def sweep_steps(self):
        return self._sweep_steps

    @property
    def has_uniform_weights(self):
        """
        True when returned sample weights are analytically uniform.

        For the single-temperature sampler this happens when the exponent in
        ``exp((1/logprob_fact - mu*beta) * Re(log psi))`` is zero.
        """
        try:
            beta        = float(np.asarray(self._pt_betas)[0]) if self._is_pt and self._pt_betas is not None else float(self._beta)
            exponent    = 1.0 / float(self._logprob_fact) - float(self._mu) * beta
        except Exception:
            return False
        return abs(exponent) < 1e-12


#########################################
#! EOF
#########################################
