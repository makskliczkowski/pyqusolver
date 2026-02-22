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
Version         : 2.0
---------------------------------
"""

#######################################################################

from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import numpy as np

try:
    from .sampler import Sampler, SamplerErrors, SolverInitState
    from .updates import (
        make_hybrid_proposer,
        propose_local_flip,
        propose_local_flip_np,
        propose_multi_flip,
    )

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
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# from algebra
try:
    if TYPE_CHECKING:
        from QES.Algebra.hilbert import HilbertSpace
        from QES.general_python.algebra.utils import Array

    from QES.general_python.algebra.utils import (
        DEFAULT_BACKEND_KEY,
        DEFAULT_JP_INT_TYPE,
        DEFAULT_NP_INT_TYPE,
    )
except ImportError:
    # If imports fail (e.g. in test environment), define defaults
    DEFAULT_JP_INT_TYPE = np.int32
    DEFAULT_NP_INT_TYPE = np.int32
    DEFAULT_BACKEND_KEY = None

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
        donate_buffers = bool(kwargs.pop("donate_buffers", False))

        super().__init__(
            shape,
            upd_fun,
            rng,
            rng_k,
            seed,
            hilbert,
            numsamples,
            numchains,
            initstate,
            backend,
            statetype,
            makediffer,
            logger,
            **kwargs,
        )

        if net is None:
            raise ValueError("A network (or callable) must be provided for evaluation.")

        self._donate_buffers                    = donate_buffers
        self._net_callable, self._parameters    = self._set_net_callable(net)

        # Check for fast updates support
        self._log_psi_delta_fun = None
        if hasattr(net, "log_psi_delta"):
            self._log_psi_delta_fun = net.log_psi_delta
        elif hasattr(net, "get_log_psi_delta"):
            self._log_psi_delta_fun = net.get_log_psi_delta()

        self._log_psi_delta_cache_init_fun = None
        if hasattr(net, "init_log_psi_delta_cache"):
            self._log_psi_delta_cache_init_fun = net.init_log_psi_delta_cache
        elif hasattr(net, "get_log_psi_delta_cache_init"):
            self._log_psi_delta_cache_init_fun = net.get_log_psi_delta_cache_init()

        # User override for delta function
        if "log_psi_delta" in kwargs:
            self._log_psi_delta_fun = kwargs["log_psi_delta"]
        if "log_psi_delta_cache_init" in kwargs:
            self._log_psi_delta_cache_init_fun = kwargs["log_psi_delta_cache_init"]

        self._log_psi_delta_supports_cache = self._log_psi_delta_cache_init_fun is not None

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
                # Table S2-inspired defaults for GPU runs.
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

        therm_steps = max(1, int(therm_steps))
        sweep_steps = max(1, int(sweep_steps))

        self._beta                  = beta
        self._therm_steps           = therm_steps
        self._sweep_steps           = sweep_steps
        self._logprob_fact          = logprob_fact
        self._logprobas             = None
        self._total_therm_updates   = therm_steps * sweep_steps * self._size    # Total updates during thermalization
        self._total_sample_updates_per_sample = sweep_steps * self._size        # Updates between collected samples
        self._updates_per_sample    = self._sweep_steps  # Steps between samples
        self._total_sample_updates_per_chain = numsamples * self._updates_per_sample * self._numchains

        # -----------------------------------------------------------------
        # Mixed Precision Setup
        # -----------------------------------------------------------------
        self._dtype = (
            dtype if dtype is not None else (jnp.complex128 if JAX_AVAILABLE else np.complex128)
        )
        self._sample_dtype = sample_dtype

        if self._isjax and self._sample_dtype is None:
            # Auto-enable mixed precision for double precision models
            if self._dtype == jnp.complex128:
                self._sample_dtype = jnp.complex64
            elif self._dtype == jnp.float64:
                self._sample_dtype = jnp.float32
            else:
                self._sample_dtype = self._dtype

        if self._sample_dtype is None:
            self._sample_dtype = self._dtype

        # -----------------------------------------------------------------
        # Resolve update function
        # -----------------------------------------------------------------

        self._upd_rule_name = "CUSTOM"
        if isinstance(upd_fun, (str, Enum)):
            from .sampler import get_update_function

            self._local_upd_fun = get_update_function(
                upd_fun, backend="jax" if self._isjax else "numpy", hilbert=hilbert, **kwargs
            )
            self._upd_rule_name = str(upd_fun)
        else:
            self._local_upd_fun = upd_fun
            if upd_fun is None:
                self._upd_rule_name = "LOCAL (Default)"

        # number of times a function is applied per update
        self._numupd = numupd

        if self._local_upd_fun is None:
            if self._isjax:
                self._local_upd_fun = propose_local_flip
            else:
                self._local_upd_fun = propose_local_flip_np

        # Try to switch to fast updates if available
        if self._isjax and self._log_psi_delta_fun is not None:
            self._local_upd_fun = self._resolve_fast_update_fun(
                self._local_upd_fun, upd_fun, **kwargs
            )

        # Check for global update configuration
        self._org_upd_fun = self._local_upd_fun
        self._global_name = "none"

        self.set_update_num(self._numupd)

        # -----------------------------------------------------------------
        #! Parallel Tempering (PT) Setup
        # -----------------------------------------------------------------

        self.set_replicas(pt_betas, pt_replicas)
        self.set_hybrid_proposer(
            kwargs.get("global_p", kwargs.get("p_global", 0.0)),
            kwargs.get("global_fraction", 0.5),
            kwargs.get("patterns", None),
            kwargs.get("global_update", None),
        )

        # -----------------------------------------------------------------
        # Static sampler function (standard or PT)
        # -----------------------------------------------------------------
        self._jit_cache = {}  # Cache for JIT-compiled samplers
        self._static_sample_fun = self.get_sampler(
            num_samples=self._numsamples, num_chains=self._numchains
        )
        self._static_pt_sampler = None  # Lazily created on first PT sample call
        self._name = "VMC"

    def _resolve_fast_update_fun(self, current_fun, upd_rule, **kwargs):
        """Attempts to replace the update function with a version that returns update info."""
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

                if upd_rule == UpdateRule.LOCAL and _rule_supported("LOCAL"):
                    return propose_local_flip_with_info
                elif upd_rule == UpdateRule.EXCHANGE and _rule_supported("EXCHANGE"):
                    from .updates import get_neighbor_table

                    hilbert = kwargs.get("hilbert")
                    order = kwargs.get("exchange_order", 1)
                    neighbor_table = get_neighbor_table(hilbert.lattice, order=order)
                    return partial(propose_exchange_with_info, neighbor_table=neighbor_table)
                elif upd_rule == UpdateRule.MULTI_FLIP and _rule_supported("MULTI_FLIP"):
                    n_flip = kwargs.get("n_flip", 1)
                    return partial(propose_multi_flip_with_info, n_flip=n_flip)
                elif (
                    upd_rule in [UpdateRule.GLOBAL, UpdateRule.PLAQUETTE, UpdateRule.WILSON]
                    and _rule_supported("GLOBAL")
                ):
                    if (
                        isinstance(current_fun, partial)
                        and current_fun.func.__name__ == "propose_global_flip"
                    ):
                        return partial(propose_global_flip_with_info, **current_fun.keywords)
                    pass
                elif upd_rule == UpdateRule.BOND_FLIP and _rule_supported("BOND_FLIP"):
                    from .updates import get_neighbor_table

                    hilbert = kwargs.get("hilbert")
                    order = kwargs.get("bond_order", 1)
                    neighbor_table = get_neighbor_table(hilbert.lattice, order=order)
                    return partial(propose_bond_flip_with_info, neighbor_table=neighbor_table)
                elif upd_rule == UpdateRule.WORM and _rule_supported("WORM"):
                    from .updates import get_neighbor_table

                    hilbert = kwargs.get("hilbert")
                    order = kwargs.get("bond_order", 1)
                    neighbor_table = get_neighbor_table(hilbert.lattice, order=order)
                    length = kwargs.get("length", 4)
                    return partial(
                        propose_worm_flip_with_info, neighbor_table=neighbor_table, length=length
                    )

            # If default was used
            if current_fun == propose_local_flip and _rule_supported("LOCAL"):
                return propose_local_flip_with_info

        except ImportError:
            pass
        return current_fun

    #####################################################################

    def reset(self):
        """
        Reset the sampler state.
        Ensures proper state shaping for Parallel Tempering if enabled.
        """
        super().reset()

        if getattr(self, "_is_pt", False) and self._states.ndim == len(self._shape) + 1:
            if self._isjax:
                self._states = jnp.tile(
                    self._states[None, ...], (self._n_replicas,) + (1,) * self._states.ndim
                )
                self._num_proposed = jnp.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_proposed.dtype
                )
                self._num_accepted = jnp.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_accepted.dtype
                )
            else:
                self._states = np.tile(
                    self._states[None, ...], (self._n_replicas,) + (1,) * self._states.ndim
                )
                self._num_proposed = np.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_proposed.dtype
                )
                self._num_accepted = np.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_accepted.dtype
                )

    def set_numchains(self, numchains):
        super().set_numchains(numchains)
        if getattr(self, "_is_pt", False) and self._states.ndim == len(self._shape) + 1:
            if self._isjax:
                self._states = jnp.tile(
                    self._states[None, ...], (self._n_replicas,) + (1,) * self._states.ndim
                )
                self._num_proposed = jnp.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_proposed.dtype
                )
                self._num_accepted = jnp.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_accepted.dtype
                )
            else:
                self._states = np.tile(
                    self._states[None, ...], (self._n_replicas,) + (1,) * self._states.ndim
                )
                self._num_proposed = np.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_proposed.dtype
                )
                self._num_accepted = np.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_accepted.dtype
                )

        self._static_sample_fun = None
        self._static_pt_sampler = None
        self._jit_cache = {}

    def autotune_chains(self, max_memory_gb: float = 12.0, safety_factor: float = 0.8):
        """
        Auto-tune the number of chains based on available memory.
        """
        if not self._isjax:
            return

        bytes_per_site = 16
        state_mem = self.size * bytes_per_site
        overhead_factor = 100
        mem_per_chain = state_mem * overhead_factor

        target_mem = max_memory_gb * 1e9 * safety_factor
        optimal_chains = int(target_mem / mem_per_chain)

        optimal_chains = max(1, (optimal_chains // 64) * 64)

        self.set_numchains(optimal_chains)
        if self._logger:
            self._logger.info(f"Auto-tuned num_chains to {optimal_chains}")

    def __repr__(self):
        init_str = str(self._initstate_t) if self._initstate_t is not None else "RND"
        glob_p = getattr(self, "_global_p", 0.0)
        glob_str = f",glob={self._global_name}" + (f"(p={glob_p:.2f})" if glob_p > 0 else "")
        return (
            f"VMC(s={self._shape},mu={self._mu},beta={self._beta},"
            f"Nt={self._therm_steps},Ns={self._sweep_steps},"
            f"Nsam={self._numsamples},Nch={self._numchains},"
            f"u={self._upd_rule_name},g={self._numupd}{glob_str},"
            f"initstate={init_str})"
        )

    def __str__(self):
        total_therm_updates_display = (
            self._total_therm_updates * self.size
        )  # Total updates per site
        total_sample_updates_display = (
            self._numsamples * self._updates_per_sample * self.size
        )  # Total sample updates per site
        init_str = str(self._initstate_t) if self._initstate_t is not None else "RND"
        glob_p = getattr(self, "_global_p", 0.0)

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

    def _set_net_callable(self, net):
        self._net = net
        if isinstance(net, nn.Module):
            return net.apply, None
        elif hasattr(net, "get_apply") and callable(net.get_apply):
            network_callable, parameters = net.get_apply()
            return network_callable, parameters
        elif hasattr(net, "apply") and callable(net.apply):
            params = (
                net.get_params()
                if hasattr(net, "get_params") and callable(net.get_params)
                else None
            )
            return net.apply, params
        elif callable(net):
            return net, None
        raise ValueError(
            "Invalid network object provided. Needs to be callable or have an 'apply' method."
        )

    def _set_replicas(self, betas: np.ndarray):
        self._pt_betas = betas
        self._is_pt = self._pt_betas is not None
        self._n_replicas = len(self._pt_betas) if self._is_pt else 1

        if self._is_pt:
            if self._isjax:
                self._states = jnp.tile(
                    self._states[None, ...], (self._n_replicas,) + (1,) * self._states.ndim
                )
                self._num_proposed = jnp.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_proposed.dtype
                )
                self._num_accepted = jnp.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_accepted.dtype
                )
            else:
                self._states = np.tile(
                    self._states[None, ...], (self._n_replicas,) + (1,) * self._states.ndim
                )
                self._num_proposed = np.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_proposed.dtype
                )
                self._num_accepted = np.zeros(
                    (self._n_replicas, self._numchains), dtype=self._num_accepted.dtype
                )

    def set_update_num(self, numupd: int):
        self._numupd = numupd
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

    def set_hybrid_proposer(
        self,
        p_global        : float = 0.0,
        fraction        : float = 0.5,
        patterns        : Optional[Union[List, str]] = None,
        global_update   : Optional[Union[Callable, str]] = None,
    ):
        self._org_upd_fun = self._local_upd_fun
        if p_global > 0.0 and self._isjax:

            def _fun_name(fun):
                if isinstance(fun, partial):
                    return getattr(fun.func, "__name__", "")
                return getattr(fun, "__name__", "")

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

                max_len = max(len(p) for p in patterns)
                n_patterns = len(patterns)
                patterns_arr = np.full((n_patterns, max_len), -1, dtype=np.int32)
                for i, p in enumerate(patterns):
                    patterns_arr[i, : len(p)] = p
                patterns_jax = jnp.array(patterns_arr)

                if local_returns_info:
                    try:
                        from .updates.spin_jax import propose_global_flip_with_info as _global_with_info
                    except ImportError as e:
                        raise ImportError("Local updater returns update-info but propose_global_flip_with_info is unavailable.") from e
                    
                    global_flip_fn = partial(_global_with_info, patterns=patterns_jax)
                else:
                    global_flip_fn = partial(propose_global_flip, patterns=patterns_jax)
                
                glob_type_str = f"Patterns (N={n_patterns})"

            else:
                n_flip_global = max(1, int(self._size * fraction))
                if local_returns_info:
                    try:
                        from .updates.spin_jax import propose_multi_flip_with_info as _multi_with_info
                    except ImportError as e:
                        raise ImportError("Local updater returns update-info but propose_multi_flip_with_info is unavailable.") from e
                    
                    global_flip_fn = partial(_multi_with_info, n_flip=n_flip_global)
                else:
                    global_flip_fn = partial(propose_multi_flip, n_flip=n_flip_global)
                glob_type_str = f"Multi-Flip (frac={fraction:.2f})"
                self._global_name = "multi-flip"

            self._org_upd_fun = make_hybrid_proposer(self._local_upd_fun, global_flip_fn, p_global)
            self._global_name = glob_type_str
            self._global_p = p_global
        else:
            self._global_name = "none"
            self._global_p = 0.0

        self.set_update_num(self._numupd)
        self._static_sample_fun = None
        self._static_pt_sampler = None
        self._jit_cache = {}

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

    @staticmethod
    @jax.jit
    def _acceptance_probability_jax(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        delta = jnp.real(candidate_val) - jnp.real(current_val)
        ratio = jnp.exp(jnp.minimum(beta * mu * delta, 30.0))
        return ratio

    @staticmethod
    def _acceptance_probability_np(current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        return vmc_np_impl.acceptance_probability_np(current_val=current_val, candidate_val=candidate_val, beta=beta, mu=mu)

    def acceptance_probability(self, current_val, candidate_val, beta: float = 1.0, mu: float = 2.0):
        use_beta = beta if beta is not None else self._beta
        if self._isjax:
            return self._acceptance_probability_jax(current_val, candidate_val, beta=use_beta, mu=mu)
        else:
            return self._acceptance_probability_np(current_val, candidate_val, beta=use_beta, mu=mu)

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

    @staticmethod
    @partial(jax.jit, static_argnames=("net_callable",))
    def _logprob_jax(x, net_callable, net_params=None):
        def apply_net(s):
            return net_callable(net_params, s)

        batched_log_psi = jax.vmap(apply_net)(x)
        flat_log_psi = VMCSampler._flatten_batch_output_jax(batched_log_psi, x.shape[0])
        return jnp.real(flat_log_psi)

    @staticmethod
    def _logprob_np(x, net_callable, net_params=None):
        return vmc_np_impl.logprob_np(x, net_callable, net_params)

    def logprob(self, x, net_callable=None, net_params=None):
        ''' Calculate log-probability (log-amplitude) for a batch of states `x` using the provided or default network callable and parameters.'''
        use_callable    = net_callable if net_callable is not None else self._net_callable
        use_params      = net_params if net_params is not None else self._parameters

        if use_callable is None:
            raise ValueError("No callable provided for log probability calculation.")

        if self._isjax:
            return VMCSampler._logprob_jax(x, use_callable, use_params)
        return VMCSampler._logprob_np(x, use_callable, use_params)

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
        """

        num_chains = chain_init.shape[0]
        # Optimized: assume net_callable_fun handles batches directly (no vmap)
        logproba_fun = lambda x: net_callable_fun(params, x)
        proposer_fn = jax.vmap(update_proposer, in_axes=(0, 0))

        def _sweep_chain_jax_step_inner(carry, _):
            chain_in, current_val_in, current_key, num_prop, num_acc, cache_in = carry

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
                    delta_res = log_psi_delta_fun(
                        params, current_val_in, chain_in, update_info, cache_in
                    )
                else:
                    delta_res = log_psi_delta_fun(params, current_val_in, chain_in, update_info)

                if isinstance(delta_res, tuple):
                    delta_log_psi, cache_prop = delta_res
                else:
                    delta_log_psi = delta_res
                new_val         = current_val_in + delta_log_psi
                delta           = jnp.real(delta_log_psi)

                if (
                    cache_in is not None
                    and cache_prop is None
                    and log_psi_delta_cache_init_fun is not None
                ):
                    cache_prop = log_psi_delta_cache_init_fun(params, new_chain)
            else:
                # O(N_sites * N_hidden) standard update
                new_val = VMCSampler._flatten_batch_output_jax(logproba_fun(new_chain), num_chains)
                delta = jnp.real(new_val - current_val_in)
                if cache_in is not None and log_psi_delta_cache_init_fun is not None:
                    cache_prop = log_psi_delta_cache_init_fun(params, new_chain)

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

            if cache_in is None:
                cache_out = None
            elif cache_prop is None:
                cache_out = cache_in
            else:
                def _select_cache(new_leaf, old_leaf):
                    cache_mask = keep.reshape((num_chains,) + (1,) * (new_leaf.ndim - 1))
                    return jnp.where(cache_mask, new_leaf, old_leaf)

                cache_out = jax.tree_util.tree_map(_select_cache, cache_prop, cache_in)

            new_carry       = (
                chain_out,
                val_out,
                next_key_carry,
                proposed_out,
                accepted_out,
                cache_out,
            )
            return new_carry, None

        initial_carry = (
            chain_init,
            current_val_init,
            rng_k_init,
            num_proposed_init,
            num_accepted_init,
            delta_cache_init,
        )
        final_carry, _ = jax.lax.scan(
            _sweep_chain_jax_step_inner, initial_carry, None, length=steps
        )
        final_chain, final_val, final_key, final_prop, final_acc, final_cache = final_carry

        return final_chain, final_val, final_key, final_prop, final_acc, final_cache

    @staticmethod
    def _run_mcmc_steps_np(
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
        return vmc_np_impl.run_mcmc_steps_np(
            chain=chain,
            logprobas=logprobas,
            num_proposed=num_proposed,
            num_accepted=num_accepted,
            params=params,
            rng=rng,
            steps=steps,
            mu=mu,
            beta=beta,
            update_proposer=update_proposer,
            log_proba_fun=log_proba_fun,
            accept_config_fun=accept_config_fun,
            net_callable_fun=net_callable_fun,
        )

    def _sweep_chain(
        self,
        chain,
        logprobas,
        rng_k,
        num_proposed,
        num_accepted,
        params,
        update_proposer,
        log_proba_fun,
        accept_config_fun,
        net_callable_fun,
        steps,
    ):
        use_log_proba_fun = self.logprob if log_proba_fun is None else log_proba_fun
        use_accept_config_fun = (
            self.acceptance_probability if accept_config_fun is None else accept_config_fun
        )
        use_net_callable_fun = self._net_callable if net_callable_fun is None else net_callable_fun
        use_update_proposer = self._upd_fun if update_proposer is None else update_proposer

        if logprobas is None:
            logprobas = self._logprobas
            if logprobas is None:
                logprobas = self.logprob(chain, net_callable=net_callable_fun, net_params=params)

        if self._isjax:
            return self._run_mcmc_steps_jax(
                chain_init=chain,
                current_val_init=logprobas,
                rng_k_init=rng_k,
                num_proposed_init=num_proposed,
                num_accepted_init=num_accepted,
                delta_cache_init=None,
                params=params,
                steps=steps,
                update_proposer=use_update_proposer,
                net_callable_fun=use_net_callable_fun,
                mu=self._mu,
                beta=self._beta,
                log_psi_delta_fun=self._log_psi_delta_fun,
                log_psi_delta_cache_init_fun=self._log_psi_delta_cache_init_fun,
                log_psi_delta_supports_cache=self._log_psi_delta_supports_cache,
            )
        return self._run_mcmc_steps_np(
            chain,
            logprobas,
            num_proposed,
            num_accepted,
            params,
            rng_k,
            steps,
            self._mu,
            self._beta,
            use_update_proposer,
            use_log_proba_fun,
            use_accept_config_fun,
            use_net_callable_fun,
        )

    ###################################################################
    #! SAMPLING
    ###################################################################

    @staticmethod
    def _generate_samples_jax(
        states_init,
        logprobas_init,
        rng_k_init,
        num_proposed_init,
        num_accepted_init,
        delta_cache_init,
        params,
        num_samples,
        total_therm_updates,
        updates_per_sample,
        mu,
        beta,
        update_proposer,
        net_callable_fun,
        log_psi_delta_fun=None,
        log_psi_delta_cache_init_fun=None,
        log_psi_delta_supports_cache=False,
    ):

        #! Thermalization phase
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

        #! Sampling phase (using lax.scan for collection)
        def sample_scan_body(carry, _):
            (
                states_carry,
                logprobas_carry,
                rng_k_carry,
                num_proposed_carry,
                num_accepted_carry,
                delta_cache_carry,
            ) = (
                carry
            )
            (
                states_new,
                logprobas_new,
                rng_k_new,
                num_proposed_new,
                num_accepted_new,
                delta_cache_new,
            ) = (
                VMCSampler._run_mcmc_steps_jax(
                    chain_init=states_carry,
                    current_val_init=logprobas_carry,
                    rng_k_init=rng_k_carry,
                    num_proposed_init=num_proposed_carry,
                    num_accepted_init=num_accepted_carry,
                    delta_cache_init=delta_cache_carry,
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
            )
            return (
                states_new,
                logprobas_new,
                rng_k_new,
                num_proposed_new,
                num_accepted_new,
                delta_cache_new,
            ), states_new

        initial_scan_carry = (
            states_therm,
            logprobas_therm,
            rng_k_therm,
            num_proposed_therm,
            num_accepted_therm,
            delta_cache_therm,
        )
        final_carry, collected_samples = jax.lax.scan(
            f=sample_scan_body, init=initial_scan_carry, xs=None, length=num_samples
        )
        return final_carry, collected_samples

    def _generate_samples_np(self, params, num_samples, multiple_of=1):
        return vmc_np_impl.generate_samples_np(sampler=self, params=params, num_samples=num_samples, multiple_of=multiple_of)

    ###################################################################
    #! STATIC JAX SAMPLING KERNEL
    ###################################################################

    @staticmethod
    def _batched_network_apply(params, configs, net_apply, chunk_size=2048):
        n_samples = configs.shape[0]
        if n_samples == 0:
            return jnp.empty((0,), dtype=jnp.result_type(configs.dtype, jnp.float32))

        if n_samples <= chunk_size:
            out_direct = net_apply(params, configs)
            return VMCSampler._flatten_batch_output_jax(out_direct, n_samples)

        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        pad_size = n_chunks * chunk_size - n_samples

        if pad_size > 0:
            pad_cfg = [(0, pad_size)] + [(0, 0)] * (configs.ndim - 1)
            configs_padded = jnp.pad(configs, pad_cfg, mode="edge")
        else:
            configs_padded = configs

        configs_reshaped = configs_padded.reshape((n_chunks, chunk_size) + configs.shape[1:])

        def scan_fn(carry, x_chunk):
            out_chunk = VMCSampler._flatten_batch_output_jax(
                net_apply(params, x_chunk), chunk_size
            )
            return carry, out_chunk

        _, outputs = jax.lax.scan(scan_fn, None, configs_reshaped)
        outputs_flat = outputs.reshape(-1)

        if pad_size > 0:
            outputs_flat = outputs_flat[:n_samples]
        return outputs_flat

    @staticmethod
    def _static_sample_jax(
        states_init,
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
    ):

        logprobas_init = net_callable_fun(params, states_init)
        logprobas_init = VMCSampler._flatten_batch_output_jax(logprobas_init, num_chains)
        if log_psi_delta_cache_init_fun is not None:
            delta_cache_init = log_psi_delta_cache_init_fun(params, states_init)
        else:
            delta_cache_init = None

        #! Phase 1: Thermalization
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

        #! Phase 2: Sampling
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
            states_therm,
            logprobas_therm,
            rng_k_therm,
            num_proposed_therm,
            num_accepted_therm,
            delta_cache_therm,
        )
        final_carry, collected_samples = jax.lax.scan(
            sample_scan_body, initial_scan_carry, None, length=num_samples
        )

        collected_states, collected_logprobas = collected_samples
        configs_flat = collected_states.reshape((-1,) + shape)
        logprobas_flat = collected_logprobas.reshape((-1,))
        batched_log_ansatz = logprobas_flat

        log_prob_exponent = 1.0 / logprob_fact - mu
        log_unnorm = log_prob_exponent * jnp.real(batched_log_ansatz)
        log_max = jnp.max(log_unnorm)
        probs = jnp.exp(log_unnorm - log_max)

        total_samples = num_samples * num_chains
        norm_factor = jnp.maximum(jnp.sum(probs), 1e-10)
        probs_normalized = probs / norm_factor * total_samples

        fc_states, fc_lpsi, fc_key, fc_prop, fc_acc, _fc_cache = final_carry
        final_state_tuple = (fc_states, fc_lpsi, fc_key, fc_prop, fc_acc)
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
    def _static_sample_pt_jax(
        states_init,
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
    ):

        # Initial betas: (n_replicas, n_chains)
        # All chains in replica r start with pt_betas[r]
        current_betas = jnp.tile(pt_betas[:, None], (1, num_chains))

        if log_psi_delta_cache_init_fun is not None:
            delta_cache_init = log_psi_delta_cache_init_fun(params, states_init)
        else:
            delta_cache_init = None

        def _single_replica_mcmc(chain, lpsi, key, n_prop, n_acc, delta_cache_r, beta_r):
            # beta_r is now (n_chains,) - per-chain beta
            # But _run_mcmc_steps_jax takes scalar beta or we need to modify it?
            # _run_mcmc_steps_jax takes 'beta'. If we pass array, it propagates?
            # log_acc = beta * mu * delta. If beta is array (num_chains,), it broadcasts. Yes.
            return VMCSampler._run_mcmc_steps_jax(
                chain_init=chain,
                current_val_init=lpsi,
                rng_k_init=key,
                num_proposed_init=n_prop,
                num_accepted_init=n_acc,
                delta_cache_init=delta_cache_r,
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

        states_flat = states_init.reshape((-1,) + shape)
        logprobas_flat = VMCSampler._batched_network_apply(params, states_flat, net_callable_fun)
        logprobas_init = logprobas_flat.reshape((states_init.shape[0], states_init.shape[1]))

        def therm_scan_body(carry, _):
            states, lpsi, key, n_prop, n_acc, delta_cache, betas = carry
            key, subkey = jax.random.split(key)
            replica_keys = jax.random.split(subkey, n_replicas)

            states_new, lpsi_new, keys_new, n_prop_new, n_acc_new, delta_cache_new = vmapped_mcmc_step(
                states, lpsi, replica_keys, n_prop, n_acc, delta_cache, betas
            )

            key, swap_key = jax.random.split(key)
            betas_swapped = VMCSampler._replica_exchange_kernel_jax_betas(
                lpsi_new, betas, swap_key, mu
            )

            return (
                states_new,
                lpsi_new,
                key,
                n_prop_new,
                n_acc_new,
                delta_cache_new,
                betas_swapped,
            ), None

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
            states, lpsi, key, n_prop, n_acc, delta_cache, betas = carry
            key, subkey = jax.random.split(key)
            replica_keys = jax.random.split(subkey, n_replicas)

            states_new, lpsi_new, keys_new, n_prop_new, n_acc_new, delta_cache_new = vmapped_mcmc_step(
                states, lpsi, replica_keys, n_prop, n_acc, delta_cache, betas
            )

            key, swap_key = jax.random.split(key)
            betas_swapped = VMCSampler._replica_exchange_kernel_jax_betas(
                lpsi_new, betas, swap_key, mu
            )
            new_carry = (
                states_new,
                lpsi_new,
                key,
                n_prop_new,
                n_acc_new,
                delta_cache_new,
                betas_swapped,
            )

            # Collect Physical Samples
            # We want samples where beta == pt_betas[0] (1.0)
            # betas_swapped is (n_replicas, n_chains).
            # We need to gather states where beta matches target.
            target_beta = pt_betas[0]

            # Mask of physical chains: (n_replicas, n_chains)
            # Use approx equality for floats
            is_physical = jnp.abs(betas_swapped - target_beta) < 1e-5

            # We need to return a fixed shape.
            # But number of physical samples per step is exactly 'num_chains' (one per column)
            # because we permuted betas column-wise.
            # So for each chain column c, exactly one row r has beta=1.0.
            # We can use argmax to find the row index for each chain.
            physical_indices = jnp.argmax(is_physical, axis=0)  # (n_chains,)

            # Gather:
            # states_new: (n_replicas, n_chains, *shape)
            # We want to select states_new[physical_indices[c], c]

            # Advanced indexing:
            # dim 0: physical_indices
            # dim 1: arange(n_chains)
            chain_indices = jnp.arange(num_chains)

            # If shape is 1D (L,):
            # collected: (n_chains, L)
            # states_new[physical_indices, chain_indices]
            collected_states_step = states_new[physical_indices, chain_indices]
            collected_lpsi_step = lpsi_new[physical_indices, chain_indices]

            return new_carry, (collected_states_step, collected_lpsi_step)

        final_carry, (collected_states, collected_log_psi) = jax.lax.scan(
            sample_scan_body, therm_carry, None, length=num_samples
        )

        # Flatten
        # collected_states: (num_samples, n_chains, *shape)
        configs_flat = collected_states.reshape((-1,) + shape)
        log_psi_flat = collected_log_psi.reshape((-1,))
        ansatz_flat = log_psi_flat
        log_psi_real = jnp.real(ansatz_flat)

        phys_beta = pt_betas[0]
        log_prob_exponent = 1.0 / logprob_fact - mu * phys_beta

        log_unnorm = log_prob_exponent * log_psi_real
        log_unnorm_max = jnp.max(log_unnorm)
        probs = jnp.exp(log_unnorm - log_unnorm_max)
        probs_norm = probs / jnp.maximum(jnp.sum(probs), 1e-10) * (num_samples * num_chains)

        # We should remove the betas from final_carry before returning to match expected signature if needed?
        # But final_carry is internal.
        # However, _get_sampler_pt_jax unpacks it? No, it returns final_state_tuple.
        # We need to make sure the unpacking in wrapper matches.
        # The wrapper returns (final_state_tuple, samples_tuple, probs).
        # final_state_tuple corresponds to (states, lpsi, key, prop, acc).
        # Our final_carry has betas at the end. We should strip it.

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

        fc_states_sorted = jnp.take_along_axis(fc_states, take_indices, axis=0)
        fc_lpsi_sorted = jnp.take_along_axis(fc_lpsi, sorted_indices, axis=0)

        # We generally do NOT sort keys or counters, they stay with the replica slot.
        # (Or should counters move with the chain? Usually counters track acceptance at a given T).

        final_state_tuple = (fc_states_sorted, fc_lpsi_sorted, fc_key, fc_prop, fc_acc)

        return final_state_tuple, (configs_flat, ansatz_flat), probs_norm

    # -----------------------------------------------------------------

    def _sample_reinitialize(self, used_num_chains, used_num_samples, used_num_therm=None):
        self._numchains = used_num_chains if used_num_chains is not None else self._numchains
        self._numsamples = used_num_samples if used_num_samples is not None else self._numsamples
        if used_num_therm is not None:
            self._therm_steps = used_num_therm
            self._total_therm_updates = self._therm_steps * self._sweep_steps * self._size
        self._total_sample_updates_per_chain = (
            self._numsamples * self._updates_per_sample * self._numchains
        )
        self.reset()
        current_states = self._states
        current_proposed = self._num_proposed
        current_accepted = self._num_accepted
        if self._isjax:
            self._static_sample_fun = self._get_sampler_jax(self._numsamples, self._numchains)
            self._static_pt_sampler = self._get_sampler_pt_jax(self._numsamples, self._numchains)
        else:
            self._static_sample_fun = self._get_sampler_np(self._numsamples, self._numchains)
            self._static_pt_sampler = None

        return current_states, current_proposed, current_accepted

    def _sample_callable(self, parameters=None):
        current_params = None
        if parameters is not None:
            current_params = parameters
        elif self._parameters is not None:
            current_params = self._parameters
        elif hasattr(self._net, "get_params") and callable(self._net.get_params):
            current_params = self._net.get_params()
        elif hasattr(self._net, "params"):
            current_params = self._net.params
        net_callable = self._net_callable
        return net_callable, current_params

    def sample(
        self, parameters=None, num_samples=None, *, num_chains=None, num_therm=None, betas=None
    ):
        used_num_samples = num_samples if num_samples is not None else self._numsamples
        used_num_chains = num_chains if num_chains is not None else self._numchains
        used_num_therm = num_therm if num_therm is not None else self._therm_steps

        if betas is not None:
            self._set_replicas(betas)

        reinitialized_for_call = False
        current_states = self._states
        current_proposed = self._num_proposed
        current_accepted = self._num_accepted

        if (
            used_num_chains != self._numchains
            or used_num_samples != self._numsamples
            or used_num_therm != self._therm_steps
        ):
            if self._logger:
                self._logger.info(
                    f"(Warning) Running sample with {used_num_chains} chains (instance default is {self._numchains}). State reinitialized for this call."
                )
            reinitialized_for_call = True
            current_states, current_proposed, current_accepted = self._sample_reinitialize(
                used_num_chains, used_num_samples, used_num_therm
            )

        net_callable, current_params = self._sample_callable(parameters)

        if self._isjax:
            if not isinstance(self._rng_k, jax.Array):
                raise TypeError(f"Sampler's RNG key is invalid: {type(self._rng_k)}")

            if self._is_pt:
                if self._static_pt_sampler is None:
                    self._static_pt_sampler = self._get_sampler_pt_jax(
                        num_samples=used_num_samples, num_chains=used_num_chains
                    )

                final_state_tuple, samples_tuple, probs = self._static_pt_sampler(
                    states_init=current_states,
                    rng_k_init=self._rng_k,
                    params=current_params,
                    num_proposed_init=current_proposed,
                    num_accepted_init=current_accepted,
                )

                (
                    final_states,
                    final_logprobas,
                    final_rng_k,
                    final_num_proposed,
                    final_num_accepted,
                ) = final_state_tuple

                self._states = final_states
                self._logprobas = final_logprobas
                self._num_proposed = final_num_proposed
                self._num_accepted = final_num_accepted
                self._rng_k = final_rng_k

                final_state_info = (self._states[0], self._logprobas[0])
                return final_state_info, samples_tuple, probs

            if self._static_sample_fun is None:
                self._static_sample_fun = self.get_sampler(
                    num_samples=used_num_samples, num_chains=self._numchains
                )

            final_state_tuple, samples_tuple, probs = self._static_sample_fun(
                states_init=current_states,
                rng_k_init=self._rng_k,
                params=current_params,
                num_proposed_init=current_proposed,
                num_accepted_init=current_accepted,
            )

            final_states, final_logprobas, final_rng_k, final_num_proposed, final_num_accepted = (
                final_state_tuple
            )

            self._states = final_states
            self._logprobas = final_logprobas
            self._num_proposed = final_num_proposed
            self._num_accepted = final_num_accepted
            self._rng_k = final_rng_k

            final_state_info = (self._states, self._logprobas)
            return final_state_info, samples_tuple, probs
        else:
            return vmc_np_impl.sample_np(
                self,
                current_params=current_params,
                used_num_samples=used_num_samples,
                net_callable=net_callable,
            )

    # -----------------------------------------------------------------

    def _get_sampler_pt_jax(
        self, num_samples: Optional[int] = None, num_chains: Optional[int] = None
    ) -> Callable:
        if not self._isjax:
            raise RuntimeError("PT JAX sampler only available for JAX backend.")

        static_num_samples = num_samples if num_samples is not None else self._numsamples
        static_num_chains = num_chains if num_chains is not None else self._numchains

        baked_args = {
            "num_samples": static_num_samples,
            "num_chains": static_num_chains,
            "n_replicas": self._n_replicas,
            "total_therm_updates": self._total_therm_updates,
            "updates_per_sample": self._updates_per_sample,
            "shape": self._shape,
            "mu": self._mu,
            "pt_betas": self._pt_betas,
            "logprob_fact": self._logprob_fact,
            "update_proposer": self._upd_fun,
            "net_callable_fun": self._net_callable,
            "log_psi_delta_fun": self._log_psi_delta_fun,
            "log_psi_delta_cache_init_fun": self._log_psi_delta_cache_init_fun,
            "log_psi_delta_supports_cache": self._log_psi_delta_supports_cache,
        }

        partial_sampler = partial(VMCSampler._static_sample_pt_jax, **baked_args)
        int_dtype = DEFAULT_JP_INT_TYPE
        n_replicas = self._n_replicas

        def wrapped_pt_sampler_impl(
            states_init: jax.Array,
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
                rng_k_init=rng_k_init,
                num_proposed_init=_num_proposed,
                num_accepted_init=_num_accepted,
                params=params,
            )

        if self._donate_buffers:
            return jax.jit(wrapped_pt_sampler_impl, donate_argnums=(0, 1))
        return jax.jit(wrapped_pt_sampler_impl)

    def _get_sampler_jax(
        self, num_samples: Optional[int] = None, num_chains: Optional[int] = None
    ) -> Callable:
        if not self._isjax:
            raise RuntimeError("Static JAX sampler getter only available for JAX backend.")

        static_num_samples  = num_samples if num_samples is not None else self._numsamples
        static_num_chains   = num_chains if num_chains is not None else self._numchains

        baked_args = {
            "num_samples"           : static_num_samples,
            "num_chains"            : static_num_chains,
            "total_therm_updates"   : self._total_therm_updates,
            "updates_per_sample"    : self._updates_per_sample,
            "shape"                 : self._shape,
            "update_proposer"       : self._upd_fun,
            "net_callable_fun"      : self._net_callable,
            "mu"                    : self._mu,
            "beta"                  : self._beta,
            "logprob_fact"          : self._logprob_fact,
            "log_psi_delta_fun"     : self._log_psi_delta_fun,
            "log_psi_delta_cache_init_fun": self._log_psi_delta_cache_init_fun,
            "log_psi_delta_supports_cache": self._log_psi_delta_supports_cache,
        }

        partial_sampler = partial(VMCSampler._static_sample_jax, **baked_args)
        int_dtype       = DEFAULT_JP_INT_TYPE

        def wrapped_sampler_impl(
            states_init         : jax.Array,
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

    def get_sampler(self, num_samples: Optional[int] = None, num_chains: Optional[int] = None) -> Callable:
        '''
        Get a sampler function with the given number of samples and chains baked in as static arguments.
        '''
        cache_key = (
            num_samples if num_samples is not None else self._numsamples,
            num_chains  if num_chains  is not None else self._numchains,
            self._isjax,
            self._donate_buffers if self._isjax else False,
        )

        if hasattr(self, "_jit_cache") and cache_key in self._jit_cache:
            return self._jit_cache[cache_key]

        if self._isjax:
            sampler = self._get_sampler_jax(num_samples, num_chains)
        elif not self._isjax:
            sampler = self._get_sampler_np(num_samples, num_chains)
        else:
            raise RuntimeError("Sampler backend must be either 'jax' or 'numpy'.")

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

        self._static_sample_fun = None
        self._static_pt_sampler = None
        self._jit_cache = {}

    ###################################################################
    #! SETTERS
    ###################################################################

    def set_mu(self, mu):
        self._mu = mu

    def set_beta(self, beta):
        self._beta = beta

    def set_therm_steps(self, therm_steps):
        self._therm_steps = therm_steps

    def set_sweep_steps(self, sweep_steps):
        self._sweep_steps = sweep_steps

    def set_replicas(
        self,
        betas: Optional[Union[List[float], np.ndarray, jnp.ndarray]],
        n_replicas: Optional[int] = None,
        min_beta: Optional[float] = None,
    ):
        if betas is None and (n_replicas is not None and n_replicas > 1):
            min_beta = min_beta if min_beta is not None else self._beta / 10.0
            if self._isjax:
                betas = jnp.logspace(0, np.log10(min_beta), n_replicas)
            else:
                betas = np.logspace(0, np.log10(min_beta), n_replicas)
        self._set_replicas(betas)

    ###################################################################
    #! GETTERS
    ###################################################################

    def get_mu(self):
        return self._mu

    def get_beta(self):
        return self._beta

    def get_therm_steps(self):
        return self._therm_steps

    def get_sweep_steps(self):
        return self._sweep_steps

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

    def get_pt_betas(self):
        return self._pt_betas

    def get_pt_n_replicas(self):
        return self._n_replicas

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
        if self._is_pt:
            return False
        try:
            exponent = 1.0 / float(self._logprob_fact) - float(self._mu) * float(self._beta)
        except Exception:
            return False
        return abs(exponent) < 1e-12


#########################################
#! EOF
#########################################
