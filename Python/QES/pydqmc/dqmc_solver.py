"""
DQMC Solver Module.
Integrates DQMC scheme with the QES Solver framework.
"""

from __future__ import annotations
from typing import Optional, Union, Any, List, Dict
import numpy as np
import jax
import jax.numpy as jnp
from QES.Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain, McsReturn
from QES.pydqmc.dqmc_sampler import DQMCSampler
from QES.pydqmc.dqmc_model import DQMCModel, choose_dqmc_model

class DQMCSolver(MonteCarloSolver):
    """
    Solver for Determinant Quantum Monte Carlo simulations.
    Supports single and multiple chains, fast JAX-based updates, and stability regrouping.
    """
    def __init__(
        self,
        model: Union[DQMCModel, 'Hamiltonian'],
        beta: Optional[float] = None,
        M: Optional[int] = None,
        n_stable: int = 10,
        num_chains: int = 1,
        seed: Optional[int] = None,
        directory: Optional[str] = "dqmc_results",
        **kwargs
    ):
        if not isinstance(model, DQMCModel):
            if beta is None or M is None:
                raise ValueError("beta and M must be provided if model is a Hamiltonian.")
            model = choose_dqmc_model(model, beta, M, **kwargs)
            
        super().__init__(
            seed=seed,
            shape=(model.M, model.n_sites),
            hilbert=model.hamiltonian.hilbert_space,
            directory=directory,
            **kwargs
        )
        self.model = model
        self.sampler = DQMCSampler(model, n_stable=n_stable, num_chains=num_chains, seed=seed if seed else 42)
        self._info = f"DQMC Solver for {model.hamiltonian._name}"
        self._last_acc_rate = 0.0
        self._energies = []
        
        # Accumulators for measurement
        self._gs_accum      = None
        self._n_measurements = 0

    @property
    def lastloss(self):
        """Used for parallel tempering swap probability."""
        if len(self._energies) > 0:
            return self._energies[-1]
        return 0.0

    def set_beta(self, beta: float):
        """Update beta and recompute matrices if needed."""
        self.model.beta = beta
        self.sampler.dtau = self.model.dtau
        self.sampler.recompute_everything()

    def train_step(self, i: int, par: McsTrain, verbose: bool = False, **kwargs) -> McsReturn:
        """Perform one sweep of DQMC and record observables."""
        acc_rate = self.sampler.sweep()
        self._last_acc_rate = float(acc_rate)
        
        # Accumulate Green's functions for measurement
        if self._gs_accum is None:
            self._gs_accum = [jnp.zeros_like(g) for g in self.sampler.Gs_avg]
            
        for c in range(self.model.n_channels):
            self._gs_accum[c] += self.sampler.Gs_avg[c]
        
        self._n_measurements += 1
        
        # Collect observables using current average
        obs = self.measure_observables()
        self._energies.append(obs["energy"])
        
        # Optional: collection of unequal time Gs
        collect_unequal = kwargs.get("collect_unequal", False)
        if collect_unequal and i % par.mc_corr == 0:
            self._unequal_gs = self.sampler.compute_unequal_time_Gs()
        
        return McsReturn(losses=[obs["energy"]], finished=False)

    def train(self, par: McsTrain, verbose: bool = False, **kwargs):
        """Execute the full DQMC simulation."""
        # Reset accumulators
        self._gs_accum = None
        self._n_measurements = 0
        
        # Warmup
        for _ in range(par.mcth):
            self.sampler.sweep()
            
        # Sampling
        self._energies = []
        for step in range(par.mcsam):
            self.train_step(step, par, verbose=verbose, **kwargs)
            
        return McsReturn(
            losses=self._energies, 
            losses_mean=[np.mean(self._energies)], 
            losses_std=[np.std(self._energies)], 
            finished=True
        )

    def get_gs_avg(self):
        """Returns the average Green's functions across all measurement steps."""
        if self._gs_accum is None or self._n_measurements == 0:
            return self.sampler.Gs_avg
        return tuple(g / self._n_measurements for g in self._gs_accum)

    def measure_energy(self):
        """
        Measure the average energy across all chains.
        E = <H> = <K> + <V>
        """
        Gs = self.get_gs_avg()
        K = self.sampler.K_jax
        
        # E_kin per chain (sum over channels)
        e_kin = 0.0
        for c in range(self.model.n_channels):
            e_kin -= jnp.trace(K @ Gs[c], axis1=-2, axis2=-1)
        
        # E_int calculation depends on the model
        if hasattr(self.model, 'U'):
            # Hubbard model specific measurement
            n_up = 1.0 - jnp.diagonal(Gs[0], axis1=-2, axis2=-1)
            n_dn = 1.0 - jnp.diagonal(Gs[1], axis1=-2, axis2=-1)
            e_int = self.model.U * jnp.sum(n_up * n_dn, axis=-1)
        else:
            e_int = 0.0 
        
        total_e = e_kin + e_int
        return float(jnp.mean(total_e))

    def measure_observables(self) -> Dict[str, Any]:
        """
        Measure equal-time observables.
        """
        Gs = self.get_gs_avg()
        
        # Densities per channel
        densities = []
        for c in range(self.model.n_channels):
            n_c = 1.0 - jnp.diagonal(Gs[c], axis1=-2, axis2=-1)
            densities.append(jnp.mean(n_c))
        
        density = sum(densities)
        
        # Local moment calculation
        if self.model.n_channels >= 2:
            n_up = 1.0 - jnp.diagonal(Gs[0], axis1=-2, axis2=-1)
            n_dn = 1.0 - jnp.diagonal(Gs[1], axis1=-2, axis2=-1)
            mz2 = jnp.mean(n_up + n_dn - 2 * n_up * n_dn)
        else:
            mz2 = 0.0

        return {
            "energy": self.measure_energy(),
            "density": float(density),
            "mz2": float(mz2)
        }

    def measure_unequal_time(self):
        """
        Collect time-displaced observables like Sz(tau)Sz(0).
        """
        if not hasattr(self, "_unequal_gs"):
            return None
        return jnp.mean(self._unequal_gs, axis=0)

    def clone(self) -> DQMCSolver:
        """Create a copy of the solver."""
        return DQMCSolver(
            model       =   self.model,
            n_stable    =   self.sampler.n_stable,
            num_chains  =   self.sampler.num_chains,
            seed        =   None 
        )

    def swap(self, other: DQMCSolver):
        """Swap configurations with another solver."""
        self.sampler.configs, other.sampler.configs = other.sampler.configs, self.sampler.configs
        self.sampler.recompute_everything()
        other.sampler.recompute_everything()

    def save_weights(self, directory=None, name="configs"):
        if directory:
            path = f"{directory}/{name}.npy"
            np.save(path, np.array(self.sampler.configs))

    def load_weights(self, directory=None, name="configs"):
        if directory:
            path = f"{directory}/{name}.npy"
            self.sampler.configs = jnp.array(np.load(path))
            self.sampler.recompute_everything()
