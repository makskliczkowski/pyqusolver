"""
DQMC Sampler Module using JAX.
Optimized kernels for Determinant Quantum Monte Carlo simulations.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple, Union, List, TYPE_CHECKING

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax import config
    config.update("jax_enable_x64", True)
except ImportError:
    raise ImportError("JAX is required for DQMCSampler. Please install it via 'pip install jax jaxlib'.")

from    functools import partial
import  numpy as np

try:
    from ..general_python.algebra.utilities import udt
except ImportError:
    raise ImportError("UDT utilities are required for DQMCSampler. Please ensure they are available in the specified path.")

if TYPE_CHECKING:
    from .dqmc_model import DQMCModel

# ---------------------------------------------------------------------------
# JAX KERNELS FOR DQMC
# ---------------------------------------------------------------------------

@jax.jit
def sherman_morrison_update(G, site, delta):
    """
    Fast update of the Green's function using the Sherman-Morrison formula.
    G_new = G - ( (G_{:,site} * delta) @ G_{site,:} ) / (1 + delta * (1 - G_{site,site}))
    """
    col = G[:, site]
    row = G[site, :]
    g_ii = G[site, site]
    prefactor = delta / (1.0 + delta * (1.0 - g_ii))
    return G - prefactor * jnp.outer(col, row)

@jax.jit
def propagate_green(G, B, iB):
    """
    Propagate Green's function to the next Trotter slice: G(tau+1) = B_tau G(tau) B_tau^-1.
    """
    return B @ G @ iB

@partial(jax.jit, static_argnames=("n_stable",))
def calculate_green_stable(Bs, n_stable):
    """
    Calculate Green's function stably using UDT decomposition.
    Bs: B matrices [M, N, N]
    n_stable: number of slices between stable regrouping.
    """
    n_stable = max(1, int(n_stable))
    M = Bs.shape[0]
    dim = Bs.shape[1]
    
    # Pre-multiply Bs in groups of n_stable
    num_groups  = M // n_stable
    rem         = M % n_stable
    
    def group_step(carry, g_idx):
        group_start = g_idx * n_stable
        group_Bs = lax.dynamic_slice(Bs, (group_start, 0, 0), (n_stable, dim, dim))
        
        # Multiply group_Bs: B_n @ ... @ B_1
        def mult_step(mat, i):
            return group_Bs[i] @ mat, None
        B_group, _ = lax.scan(mult_step, jnp.eye(dim, dtype=Bs.dtype), jnp.arange(n_stable))
        
        return udt.udt_fact_mult(B_group, carry, backend="jax"), None

    initial_state = udt.UDTState(
        U=jnp.eye(dim, dtype=Bs.dtype),
        D=jnp.ones((dim,), dtype=Bs.dtype),
        T=jnp.eye(dim, dtype=Bs.dtype),
    )
    final_state, _ = lax.scan(group_step, initial_state, jnp.arange(num_groups))
    
    # Handle remainder
    if rem > 0:
        rem_Bs = lax.dynamic_slice(Bs, (num_groups * n_stable, 0, 0), (rem, dim, dim))
        def rem_mult_step(mat, i):
            return rem_Bs[i] @ mat, None
        B_rem, _ = lax.scan(rem_mult_step, jnp.eye(dim, dtype=Bs.dtype), jnp.arange(rem))
        final_state = udt.udt_fact_mult(B_rem, final_state, backend="jax")
    
    # Fully stable inverse in UDT space (avoid explicit reconstruction).
    return udt.udt_inv_1p(final_state, backend="jax")

# ---------------------------------------------------------------------------
# CORE SWEEP KERNEL
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_sites", "n_channels", "n_stable", "calculate_deltas_fn"))
def dqmc_sweep_jax(
    configs, 
    Gs, 
    Bs_all, 
    iBs_all, 
    model_params,
    key, 
    n_sites, 
    n_channels,
    n_stable,
    calculate_deltas_fn
):
    """
    One full sweep (forward) over all Trotter slices.
    Returns: final Gs at tau=0, updated configs, and stats.
    """
    M = configs.shape[0]
    
    def tau_step(carry, tau_idx):
        Gs_curr, configs_curr, k, acc = carry
        
        # Periodic refresh
        def refresh_Gs(gs_in):
            def shift_B(c):
                # G(tau) = (I + B_{tau-1}...B_0 B_M...B_tau)^-1
                shifted = jnp.roll(Bs_all[c], -tau_idx, axis=0)
                return calculate_green_stable(shifted, n_stable)
            return tuple(shift_B(c) for c in range(n_channels))

        Gs_ready = lax.cond(
            (tau_idx % n_stable == 0) & (tau_idx > 0),
            refresh_Gs,
            lambda _gs: _gs,
            Gs_curr
        )

        def site_loop(site_carry, site_idx):
            Gs_s, config_s, ks, acc_s = site_carry
            ks, subk = jax.random.split(ks)
            
            s_old = config_s[tau_idx, site_idx]
            s_new = -s_old
            
            deltas = calculate_deltas_fn(s_old, s_new, site_idx)
            
            ratio = 1.0
            for c in range(n_channels):
                ratio *= (1.0 + deltas[c] * (1.0 - Gs_s[c][site_idx, site_idx]))
            
            accept = jax.random.uniform(subk) < jnp.minimum(1.0, jnp.abs(ratio))
            
            Gs_next = tuple(
                lax.cond(
                    accept,
                    lambda _g: sherman_morrison_update(_g, site_idx, deltas[c]),
                    lambda _g: _g,
                    Gs_s[c]
                ) for c in range(n_channels)
            )
            
            config_next = config_s.at[tau_idx, site_idx].set(
                lax.cond(accept, lambda _: s_new, lambda _: s_old, None)
            )
            
            return (Gs_next, config_next, ks, acc_s + jnp.where(accept, 1, 0)), None

        final_site_carry, _ = lax.scan(site_loop, (Gs_ready, configs_curr, k, acc), jnp.arange(n_sites))
        Gs_tau, configs_tau, key_tau, acc_tau = final_site_carry
        
        # Propagate to next slice: G(tau+1) = B_tau G(tau) B_tau^-1
        Gs_next_tau = tuple(
            propagate_green(Gs_tau[c], Bs_all[c, tau_idx], iBs_all[c, tau_idx]) 
            for c in range(n_channels)
        )
        
        return (Gs_next_tau, configs_tau, key_tau, acc_tau), None

    final_carry, _ = lax.scan(tau_step, (Gs, configs, key, 0), jnp.arange(M))
    return final_carry

# ---------------------------------------------------------------------------
# SAMPLER CLASS
# ---------------------------------------------------------------------------

class DQMCSampler:
    """
    Determinant Quantum Monte Carlo Sampler.
    Wraps JAX kernels for efficient sampling.
    """
    def __init__(
        self, 
        model: DQMCModel, 
        n_stable: int = 10,
        num_chains: int = 1,
        seed: int = 42
    ):
        self.model = model
        self.n_sites = model.n_sites
        self.M = model.M
        self.n_channels = model.n_channels
        self.n_stable = n_stable
        self.num_chains = num_chains
        self.dtau = model.dtau
        
        self.key = jax.random.PRNGKey(seed)
        
        self.configs = jax.random.choice(
            self.key, 
            jnp.array([-1.0, 1.0]), 
            shape=(self.num_chains, self.M, self.n_sites)
        )
        
        self.K_jax = jnp.array(model.kinetic_matrix)
        
        self._vmapped_sweep = jax.vmap(
            partial(dqmc_sweep_jax, 
                    n_sites=self.n_sites, 
                    n_channels=self.n_channels, 
                    n_stable=self.n_stable,
                    calculate_deltas_fn=model.calculate_update_deltas),
            in_axes=(0, 0, 0, 0, None, 0)
        )
        
        self.recompute_everything()

    def _compute_matrices_for_configs(self, configs):
        """Calculate Gs, Bs, iBs for given configurations efficiently."""
        exp_K = jax.scipy.linalg.expm(-self.dtau * self.K_jax)
        exp_invK = jax.scipy.linalg.expm(self.dtau * self.K_jax)
        
        def compute_single_chain(config):
            Bs, iBs = jax.vmap(lambda c_t: self.model.get_propagators(c_t, exp_K, exp_invK))(config)
            Bs = jnp.transpose(Bs, (1, 0, 2, 3))
            iBs = jnp.transpose(iBs, (1, 0, 2, 3))
            
            Gs = tuple(calculate_green_stable(Bs[c], self.n_stable) for c in range(self.n_channels))
            return Gs, Bs, iBs

        return jax.vmap(compute_single_chain)(configs)

    def recompute_everything(self):
        """Full recomputation of Green's functions and propagators."""
        (self.Gs, self.Bs_all, self.iBs_all) = self._compute_matrices_for_configs(self.configs)
        # Gs_avg initialization for measurements (accumulated during solver.train)
        self.Gs_avg = self.Gs 

    def sweep(self):
        """Perform one full sweep over all chains using JAX."""
        self.key, subkey = jax.random.split(self.key)
        keys = jax.random.split(subkey, self.num_chains)
        
        (Gs_new, configs_new, keys_new, acc_total) = self._vmapped_sweep(
            self.configs,
            self.Gs,
            self.Bs_all,
            self.iBs_all,
            None, 
            keys
        )
        
        self.configs = configs_new
        self.Gs = Gs_new
        self.Gs_avg = Gs_new # Current equal-time Gs
        
        self.Bs_all, self.iBs_all = self._update_Bs_only(self.configs)
        return jnp.mean(acc_total)

    def _update_Bs_only(self, configs):
        """Update Bs and iBs without recomputing Green's functions."""
        exp_K = jax.scipy.linalg.expm(-self.dtau * self.K_jax)
        exp_invK = jax.scipy.linalg.expm(self.dtau * self.K_jax)
        
        @jax.vmap
        def compute_single_chain(config):
            Bs, iBs = jax.vmap(lambda c_t: self.model.get_propagators(c_t, exp_K, exp_invK))(config)
            Bs = jnp.transpose(Bs, (1, 0, 2, 3))
            iBs = jnp.transpose(iBs, (1, 0, 2, 3))
            return Bs, iBs

        return compute_single_chain(configs)

    def compute_unequal_time_Gs(self):
        """
        Compute G(tau, 0) = B_tau ... B_1 G(0) for all tau.
        Returns: (num_chains, n_channels, M, N, N)
        """
        @jax.vmap
        def compute_chain(Bs, G0):
            def step(carry, tau):
                res = []
                for c in range(self.n_channels):
                    res.append(Bs[c, tau] @ carry[c])
                res_stack = jnp.stack(res)
                return res_stack, res_stack
            
            initial_G = jnp.stack(list(G0))
            _, G_history = jax.lax.scan(step, initial_G, jnp.arange(self.M))
            return G_history
            
        res = compute_chain(self.Bs_all, self.Gs)
        return jnp.transpose(res, (0, 2, 1, 3, 4))
