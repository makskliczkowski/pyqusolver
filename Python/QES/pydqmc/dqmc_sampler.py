"""
DQMC Sampler Module using JAX.

The sampler owns the mutable DQMC state:

- auxiliary fields,
- equal-time Green's functions,
- slice propagators and their inverses,
- per-chain RNG keys.

The numerically delicate Green's-function identities live in
`QES.pydqmc.stabilization` so that the sampler itself stays focused on update
flow rather than low-level linear algebra.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax import config
    config.update("jax_enable_x64", True)
except ImportError:
    raise ImportError("JAX is required for DQMCSampler. Please install it via 'pip install jax jaxlib'.")

from functools import partial

if TYPE_CHECKING:
    from .dqmc_model import DQMCModel

from .stabilization import (
    calculate_green_stable,
    calculate_green_stable_numpy,
    green_residual_from_stack,
    localized_diagonal_update,
    localized_diagonal_update_ratio,
    propagate_green,
)

# ---------------------------------------------------------------------------
# CORE SWEEP KERNEL
# ---------------------------------------------------------------------------

@partial(
    jax.jit,
    static_argnames=(
        "n_hs_fields",
        "n_channels",
        "n_stable",
        "calculate_deltas_fn",
        "propose_field_value_fn",
        "field_weight_ratio_fn",
        "proposal_ratio_fn",
    ),
)
def dqmc_sweep_jax(
    configs, 
    Gs, 
    Bs_all, 
    iBs_all, 
    term_sites,
    key, 
    n_hs_fields, 
    n_channels,
    n_stable,
    calculate_deltas_fn,
    propose_field_value_fn,
    field_weight_ratio_fn,
    proposal_ratio_fn,
):
    """
    Perform one full forward DQMC sweep for one Markov chain.

    Math:
        the kernel walks through all imaginary-time slices and all local HS
        terms.  For each proposed local move it evaluates the channel product
        of determinant ratios and, if accepted, applies the exact low-rank
        Green's-function update.

    Returns
    -------
    tuple
        Final equal-time Green's functions, updated auxiliary fields, updated
        RNG key, and accepted-move count.
    """
    M = configs.shape[0]
    
    def tau_step(carry, tau_idx):
        """Advance one Trotter slice, including optional stabilization refresh."""
        Gs_curr, configs_curr, k, acc = carry
        
        # Periodic refresh
        def refresh_Gs(gs_in):
            """Rebuild equal-time Green's functions at the current imaginary-time origin."""
            def shift_B(c):
                """Rotate the slice stack so stable reconstruction starts at `tau_idx`."""
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

        def term_loop(term_carry, term_idx):
            """Attempt one local HS update inside the current time slice."""
            Gs_s, config_s, ks, acc_s = term_carry
            ks, proposal_key, accept_key = jax.random.split(ks, 3)
            
            s_old = config_s[tau_idx, term_idx]
            s_new = propose_field_value_fn(s_old, proposal_key, term_idx)

            indices = term_sites[term_idx]
            deltas = calculate_deltas_fn(s_old, s_new, term_idx)

            ratio = 1.0
            for c in range(n_channels):
                ratio *= localized_diagonal_update_ratio(Gs_s[c], indices, deltas[c])
            ratio *= field_weight_ratio_fn(s_old, s_new, term_idx)
            ratio *= proposal_ratio_fn(s_old, s_new, term_idx)
            
            accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, jnp.abs(ratio))
            
            Gs_next = tuple(
                lax.cond(
                    accept,
                    lambda _g: localized_diagonal_update(_g, indices, deltas[c]),
                    lambda _g: _g,
                    Gs_s[c]
                ) for c in range(n_channels)
            )
            
            config_next = config_s.at[tau_idx, term_idx].set(
                lax.cond(accept, lambda _: s_new, lambda _: s_old, None)
            )
            
            return (Gs_next, config_next, ks, acc_s + jnp.where(accept, 1, 0)), None

        final_term_carry, _ = lax.scan(term_loop, (Gs_ready, configs_curr, k, acc), jnp.arange(n_hs_fields))
        Gs_tau, configs_tau, key_tau, acc_tau = final_term_carry
        
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
    Low-level sampler for DQMC auxiliary-field configurations.

    The sampler owns the mutable Monte Carlo state:
    auxiliary fields, equal-time Green's functions, slice propagators, and RNG
    state.  Solver-level accumulation and observable interpretation live one
    layer above this class.
    """
    def __init__(
        self, 
        model: DQMCModel, 
        n_stable: int = 10,
        num_chains: int = 1,
        seed: int = 42,
        residual_recompute_threshold: float | None = 1e-6,
        refresh_strategy: str = "jax_udt",
        residual_check_interval: int | None = None,
    ):
        """
        Initialize the DQMC sampler state for one model and one inverse temperature.

        Parameters
        ----------
        model : DQMCModel
            Model wrapper providing kinetic and HS information.
        n_stable : int
            Number of slices between stabilization checkpoints in the fast
            equal-time reconstruction path.
        num_chains : int
            Number of independent Markov chains evolved in parallel.
        seed : int
            Random seed for the JAX PRNG.
        residual_recompute_threshold : float or None
            Threshold on ``||(I + B_M ... B_1)G - I||_F`` above which the
            equal-time Green's functions are rebuilt from scratch.
        refresh_strategy : str
            Full-refresh backend.  ``"numpy_pivoted"`` favors robustness,
            ``"jax_udt"`` keeps the entire path in JAX.
        residual_check_interval : int or None
            Number of sweeps between full residual diagnostics.  When omitted,
            the sampler reuses ``n_stable`` as a conservative default interval.
        """
        self.model = model
        self.n_sites = model.n_sites
        self.n_hs_fields = model.n_hs_fields
        self.M = model.M
        self.n_channels = model.n_channels
        self.n_stable = n_stable
        self.num_chains = num_chains
        self.dtau = model.dtau
        self.term_sites = jnp.asarray(model.term_sites, dtype=jnp.int32)
        self.exp_K = None
        self.exp_invK = None
        self.residual_recompute_threshold = residual_recompute_threshold
        self.refresh_strategy = str(refresh_strategy).strip().lower()
        self.residual_check_interval = max(1, int(residual_check_interval or n_stable))
        self.last_equal_time_residuals = None
        self.last_refresh_drift = 0.0
        self.num_forced_refreshes = 0
        self._sweep_count = 0
        
        self.key = jax.random.PRNGKey(seed)
        
        self.configs = jnp.asarray(
            self.model.initial_fields(
                self.key,
                (self.num_chains, self.M, self.n_hs_fields),
            )
        )
        
        self.K_jax = jnp.array(model.kinetic_matrix)
        
        self._vmapped_sweep = jax.vmap(
            partial(dqmc_sweep_jax, 
                    n_hs_fields=self.n_hs_fields, 
                    n_channels=self.n_channels, 
                    n_stable=self.n_stable,
                    calculate_deltas_fn=model.calculate_update_deltas,
                    propose_field_value_fn=model.propose_field_value,
                    field_weight_ratio_fn=model.field_weight_ratio,
                    proposal_ratio_fn=model.proposal_ratio),
            in_axes=(0, 0, 0, 0, None, 0)
        )
        
        self.recompute_everything()

    def _refresh_kinetic_cache(self):
        """
        Cache the kinetic Trotter factors shared by all chains and slices.

        Math:
            for a fixed model and `dtau`, the kinetic part of each slice is the
            same matrix exponential `exp(-dtau K)` and its inverse.  Caching
            them avoids repeated dense exponentiation inside every sweep.
        """
        self.exp_K = jax.scipy.linalg.expm(-self.dtau * self.K_jax)
        self.exp_invK = jax.scipy.linalg.expm(self.dtau * self.K_jax)

    def _compute_single_chain_matrices(self, config):
        """
        Build slice propagators and equal-time Green's functions for one chain.

        The refresh strategy only affects full recomputations.  Local sweep
        updates remain on the fast JAX path.
        """
        Bs, iBs = jax.vmap(lambda c_t: self.model.get_propagators(c_t, self.exp_K, self.exp_invK))(config)
        Bs = jnp.transpose(Bs, (1, 0, 2, 3))
        iBs = jnp.transpose(iBs, (1, 0, 2, 3))

        if self.refresh_strategy == "numpy_pivoted":
            # Full refreshes are infrequent, so we allow a slower but stronger
            # reference factorization here to reduce long-time drift.
            greens = tuple(
                jnp.asarray(calculate_green_stable_numpy(np.asarray(Bs[c]), self.n_stable))
                for c in range(self.n_channels)
            )
        elif self.refresh_strategy == "jax_udt":
            greens = tuple(calculate_green_stable(Bs[c], self.n_stable) for c in range(self.n_channels))
        else:
            raise ValueError(
                f"Unknown refresh_strategy={self.refresh_strategy!r}. "
                f"Expected 'numpy_pivoted' or 'jax_udt'."
            )
        return greens, Bs, iBs

    def _compute_slice_propagators(self, config_slices):
        """Build propagators for one chain over the provided slice subset."""
        Bs, iBs = jax.vmap(lambda c_t: self.model.get_propagators(c_t, self.exp_K, self.exp_invK))(config_slices)
        return jnp.transpose(Bs, (1, 0, 2, 3)), jnp.transpose(iBs, (1, 0, 2, 3))

    def _compute_matrices_for_configs(self, configs):
        """Build equal-time Green's functions and slice propagators for all chains."""
        if self.exp_K is None or self.exp_invK is None:
            self._refresh_kinetic_cache()

        if self.refresh_strategy == "jax_udt":
            def compute_single_chain(config):
                """Compute all matrices for one chain on the pure-JAX refresh path."""
                return self._compute_single_chain_matrices(config)

            return jax.vmap(compute_single_chain)(configs)

        greens_by_channel = [[] for _ in range(self.n_channels)]
        Bs_all = []
        iBs_all = []
        for chain_idx in range(configs.shape[0]):
            greens, Bs, iBs = self._compute_single_chain_matrices(configs[chain_idx])
            for c in range(self.n_channels):
                greens_by_channel[c].append(greens[c])
            Bs_all.append(Bs)
            iBs_all.append(iBs)

        Gs = tuple(jnp.stack(channel_greens, axis=0) for channel_greens in greens_by_channel)
        return Gs, jnp.stack(Bs_all, axis=0), jnp.stack(iBs_all, axis=0)

    def recompute_everything(self):
        """Rebuild Green's functions and propagators from the current HS fields."""
        self._refresh_kinetic_cache()
        (self.Gs, self.Bs_all, self.iBs_all) = self._compute_matrices_for_configs(self.configs)
        # Gs_avg initialization for measurements (accumulated during solver.train)
        self.Gs_avg = self.Gs 
        self.current_signs = self.compute_configuration_signs()
        self.last_equal_time_residuals = self.compute_equal_time_residuals()

    def get_capabilities(self):
        """Return runtime metadata for the active local-update implementation."""
        return {
            **self.model.get_sampling_metadata(),
            "refresh_strategy": self.refresh_strategy,
            "residual_check_interval": int(self.residual_check_interval),
            "n_stable": int(self.n_stable),
            "num_chains": int(self.num_chains),
        }

    def sweep(self):
        """Perform one full Monte Carlo sweep over all chains."""
        self.key, subkey = jax.random.split(self.key)
        keys = jax.random.split(subkey, self.num_chains)
        previous_configs = self.configs
        previous_Bs_all = self.Bs_all
        previous_iBs_all = self.iBs_all
        
        (Gs_new, configs_new, keys_new, acc_total) = self._vmapped_sweep(
            self.configs,
            self.Gs,
            self.Bs_all,
            self.iBs_all,
            self.term_sites,
            keys
        )
        
        self.configs = configs_new
        self.Gs = Gs_new
        self.Gs_avg = Gs_new # Current equal-time Gs
        self.current_signs = self.compute_configuration_signs()

        self.Bs_all, self.iBs_all = self._update_Bs_only(
            self.configs,
            previous_configs=previous_configs,
            previous_Bs_all=previous_Bs_all,
            previous_iBs_all=previous_iBs_all,
        )
        self._sweep_count += 1
        if self._sweep_count % self.residual_check_interval == 0:
            self.last_equal_time_residuals = self.compute_equal_time_residuals()
            self._force_refresh_if_needed()
        # One sweep proposes exactly `M * n_hs_fields` local updates per chain,
        # so the acceptance rate is the accepted-count average divided by that.
        return jnp.mean(acc_total) / float(self.M * self.n_hs_fields)

    def _update_Bs_only(self, configs, previous_configs=None, previous_Bs_all=None, previous_iBs_all=None):
        """Refresh only the slice propagators after accepted field updates."""
        if self.exp_K is None or self.exp_invK is None:
            self._refresh_kinetic_cache()

        if previous_configs is None or previous_Bs_all is None or previous_iBs_all is None:
            @jax.vmap
            def compute_single_chain(config):
                """Rebuild slice propagators for one chain without reusing prior slice data."""
                return self._compute_slice_propagators(config)

            return compute_single_chain(configs)

        Bs_all = previous_Bs_all
        iBs_all = previous_iBs_all
        for chain_idx in range(self.num_chains):
            changed_taus = np.flatnonzero(
                np.any(
                    np.asarray(configs[chain_idx]) != np.asarray(previous_configs[chain_idx]),
                    axis=1,
                )
            )
            if changed_taus.size == 0:
                continue
            if changed_taus.size == self.M:
                Bs_chain, iBs_chain = self._compute_slice_propagators(configs[chain_idx])
                Bs_all = Bs_all.at[chain_idx].set(Bs_chain)
                iBs_all = iBs_all.at[chain_idx].set(iBs_chain)
                continue

            Bs_sel, iBs_sel = self._compute_slice_propagators(configs[chain_idx, changed_taus])
            for local_idx, tau_idx in enumerate(changed_taus):
                Bs_all = Bs_all.at[chain_idx, :, int(tau_idx)].set(Bs_sel[:, local_idx])
                iBs_all = iBs_all.at[chain_idx, :, int(tau_idx)].set(iBs_sel[:, local_idx])
        return Bs_all, iBs_all

    def compute_unequal_time_Gs(self):
        """
        Compute the unequal-time Green's-function history ``G(\tau, 0)``.

        Returns
        -------
        array
            Tensor with shape ``(num_chains, n_channels, M, N, N)``.
        """
        @jax.vmap
        def compute_chain(Bs, G0):
            """Propagate one chain's equal-time Green's function through all slices."""
            def step(carry, tau):
                """Advance the unequal-time correlator by one slice."""
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

    def compute_equal_time_residuals(self):
        """
        Return inverse residuals for the current equal-time Green's functions.

        Math:
            for each chain and channel we evaluate

                || (I + B_M ... B_1) G - I ||_F,

            which directly measures how well the stored `G` satisfies the
            defining equal-time inverse relation.
        """
        def per_chain(Bs_chain, Gs_chain):
            """Compute residual diagnostics for one chain across all fermion channels."""
            vals = []
            for c in range(self.n_channels):
                vals.append(green_residual_from_stack(Gs_chain[c], Bs_chain[c]))
            return jnp.stack(vals)

        return jax.vmap(per_chain)(self.Bs_all, self.Gs)

    def compute_configuration_signs(self):
        """
        Return the per-chain fermion sign/phase on the sampled `|W|` ensemble.

        For real weights this is a sign in `{-1, +1}`. The implementation uses
        the determinant phase of the equal-time Green's functions, exploiting

            G = (I + B_M ... B_1)^(-1).
        """
        g_stack = jnp.stack(self.Gs, axis=1)

        def per_chain(chain_greens):
            signs = []
            for c in range(self.n_channels):
                sign_c, _ = jnp.linalg.slogdet(chain_greens[c])
                signs.append(jnp.conj(sign_c))
            return jnp.prod(jnp.stack(signs))

        phases = jax.vmap(per_chain)(g_stack)
        if jnp.max(jnp.abs(jnp.imag(phases))) < 1e-12:
            return jnp.sign(jnp.real(phases))
        return phases

    def _force_refresh_if_needed(self):
        """
        Recompute `G` from scratch when the inverse residual becomes too large.

        Math:
            fast Sherman-Morrison updates are only exact in exact arithmetic.
            Over many accepted local moves, roundoff error accumulates.  The
            residual

                || (I + B_M ... B_1) G - I ||_F

            detects this drift.  When it crosses a threshold, we rebuild the
            equal-time Green's functions from the current slice stack.
        """
        if self.residual_recompute_threshold is None:
            return

        residuals = self.last_equal_time_residuals
        max_residual = float(jnp.max(residuals))
        if max_residual <= float(self.residual_recompute_threshold):
            return

        old_gs = self.Gs
        self.recompute_everything()
        drift_terms = []
        for c in range(self.n_channels):
            drift_terms.append(jnp.linalg.norm(self.Gs[c] - old_gs[c]))
        self.last_refresh_drift = float(jnp.max(jnp.stack(drift_terms)))
        self.num_forced_refreshes += 1
