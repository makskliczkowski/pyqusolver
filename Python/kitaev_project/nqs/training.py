"""
High-level trainers that orchestrate QES.NQS components.

The implementation wires up network factories, TDVP, samplers, and training
infrastructure while delegating heavy variational work to QES.NQS.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, List, TYPE_CHECKING
import numpy as np

from .types import NeuralAnsatz, NQSTrainingConfig, TrainingArtifact

if TYPE_CHECKING:
    from QES.NQS.nqs import NQS
    from QES.NQS.src.tdvp import TDVP
    from QES.Solver.MonteCarlo.sampler import VMCSampler


@dataclass
class StepInfo:
    """Information about a single training step."""
    
    @dataclass
    class TimeInfo:
        time_sample         : float = 0.0
        time_energies       : float = 0.0
        time_optimization   : float = 0.0
        time_update         : float = 0.0
        
        @property
        def time_total(self):
            return self.time_sample + self.time_energies + self.time_optimization + self.time_update
    
    mean_energy : float = None
    std_energy  : float = None
    success     : bool = True
    message     : str = ''
    time_info   : TimeInfo = field(default_factory=TimeInfo)


class NQSTrainer:
    """
    Coordinator for QES.NQS training stack.

    Responsibilities:
        1. Instantiate ansatz/sampler/TDVP stack
        2. Run variational optimization loop
        3. Compute observables using operator.jax
        4. Gather metadata and push to IO layer
    """

    def __init__(self, hamiltonian, hilbert_space, lattice, writer=None):
        self.hamiltonian = hamiltonian
        self.hilbert_space = hilbert_space
        self.lattice = lattice
        self.writer = writer
        
        # Backend setup
        self.backend = None
        self.jnp_mod = None
        self.rng = None
        self.rng_key = None

    # ------------------------------------------------------------------

    def _setup_backend(self, config: NQSTrainingConfig):
        """Initialize JAX/numpy backend."""
        if self.backend is not None:
            return
        
        from QES.general_python.algebra.utils import get_backend
        
        backend_modules = get_backend(config.backend, random=True, seed=config.seed, scipy=True)
        if isinstance(backend_modules, tuple):
            self.jnp_mod, (self.rng, self.rng_key), scipy_mod = backend_modules
        else:
            self.jnp_mod = backend_modules
            self.rng, self.rng_key = None, None
        
        self.backend = config.backend

    # ------------------------------------------------------------------

    def _create_sampler(self, network: Any, config: NQSTrainingConfig):
        """Create Monte Carlo sampler."""
        from QES.Solver.MonteCarlo.sampler import VMCSampler
        
        st_shape = (self.lattice.Ns,)
        
        sampler = VMCSampler(
            net         = network,
            shape       = st_shape,
            rng         = self.rng,
            rng_k       = self.rng_key,
            numchains   = config.num_chains,
            numsamples  = config.num_samples,
            sweep_steps = min(config.num_sweeps, 20),
            therm_steps = config.num_burnin,
            backend     = self.jnp_mod,
            mu          = 2.0,
            seed        = config.seed,
            dtype       = np.complex128,
            statetype   = np.float32,
            makediffer  = True,
        )
        
        return sampler

    # ------------------------------------------------------------------

    def _create_tdvp(self, config: NQSTrainingConfig):
        """Create TDVP integrator."""
        from QES.NQS.src.tdvp import TDVP
        import QES.general_python.algebra.solvers as solvermodule
        import QES.general_python.algebra.preconditioners as precondmodule
        
        precond = precondmodule.choose_precond(precond_id=0, backend=self.jnp_mod)
        
        tdvp = TDVP(
            use_sr          = True,
            use_minsr       = config.use_minsr,
            rhs_prefactor   = 1.0,
            sr_lin_solver   = solvermodule.SolverType.SCIPY_CG,
            sr_precond      = precond,
            sr_pinv_tol     = config.solver_tol,
            sr_pinv_cutoff  = config.solver_cutoff,
            sr_snr_tol      = config.solver_tol,
            sr_diag_shift   = config.reg_strength,
            sr_lin_solver_t = solvermodule.SolverForm.GRAM,
            sr_maxiter      = config.solver_maxiter,
            backend         = self.jnp_mod
        )
        
        return tdvp

    # ------------------------------------------------------------------

    def prepare_solver(self, ansatz: NeuralAnsatz, config: NQSTrainingConfig):
        """
        Instantiate the low-level QES solver for a given ansatz.
        """
        from QES.NQS.nqs import NQS
        
        self._setup_backend(config)
        
        # Create network from ansatz
        network = ansatz.create_network(
            input_shape=(self.lattice.Ns,),
            backend=self.jnp_mod,
            seed=config.seed
        )
        
        # Create sampler and TDVP
        sampler = self._create_sampler(network, config)
        tdvp = self._create_tdvp(config)
        
        # Create NQS system
        nqs = NQS(
            net         = network,
            sampler     = sampler,
            model       = self.hamiltonian,
            seed        = config.seed,
            beta        = 1.0,
            mu          = sampler.get_mu(),
            shape       = (self.lattice.Ns,),
            backend     = self.jnp_mod,
            batch_size  = config.batch_size,
            beta_penalty= 0.0,
        )
        
        return nqs

    # ------------------------------------------------------------------

    def _train_single_step(self, nqs, lr: float, reset: bool = False) -> StepInfo:
        """Execute a single training step."""
        step_info = StepInfo()
        
        try:
            # Sample configurations
            (_, _), (configs, configs_ansatze), probabilities = nqs.sample(reset=reset)
            
            # Compute gradients and energy
            single_step_par = nqs.step(
                params=nqs.get_params(),
                configs=configs,
                configs_ansatze=configs_ansatze,
                probabilities=probabilities
            )
            
            # Update parameters (placeholder - actual SR update would go here)
            # nqs.update_parameters(gradients, -lr, ...)
            
            step_info.mean_energy = single_step_par.loss_mean
            step_info.std_energy = single_step_par.loss_std
            step_info.message = "ok"
            
        except Exception as e:
            step_info.success = False
            step_info.message = str(e)
            step_info.mean_energy = float('nan')
            step_info.std_energy = float('nan')
        
        return step_info

    # ------------------------------------------------------------------

    def train(
        self,
        ansatz: NeuralAnsatz,
        config: NQSTrainingConfig,
        stage: str = "ground_state",
    ) -> TrainingArtifact:
        """
        Run variational training loop.
        
        Uses QES.NQS infrastructure with operator.jax for local energy computation.
        """
        solver = self.prepare_solver(ansatz, config)
        
        # Training history
        history = np.zeros(config.n_epochs)
        history_std = np.zeros(config.n_epochs)
        
        # Training loop
        for epoch in range(config.n_epochs):
            lr = max(5e-3, config.learning_rate * (0.999 ** epoch))
            step_info = self._train_single_step(solver, lr, reset=(epoch == 0))
            
            if not step_info.success:
                print(f"Training failed at epoch {epoch}: {step_info.message}")
                break
            
            history[epoch] = step_info.mean_energy
            history_std[epoch] = step_info.std_energy
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{config.n_epochs}: E = {step_info.mean_energy:.6f} ± {step_info.std_energy:.6f}")
        
        # Compute final observables using operator.jax
        observables = self._compute_observables(solver, config)
        
        metrics = {
            "planned_epochs": config.n_epochs,
            "learning_rate": config.learning_rate,
            "reg_strength": config.reg_strength,
            "backend": config.backend,
            "final_energy": float(history[-1]),
            "final_energy_std": float(history_std[-1]),
        }

        artifact = TrainingArtifact(
            ansatz_name=ansatz.name,
            stage=stage,
            hdf5_path="",
            metrics=metrics,
            observables=observables,
            network_state=ansatz.snapshot(),
            extras={"config": asdict(config), "history": history.tolist()},
        )

        if self.writer is not None:
            artifact.hdf5_path = self.writer.write_training_run(artifact)

        return artifact

    # ------------------------------------------------------------------

    def _compute_observables(self, nqs: NQS, config: NQSTrainingConfig) -> Dict[str, Any]:
        """Compute observables using operator.jax sampling."""
        try:
            from .observables import compute_observables_nqs_jax
            
            observables = compute_observables_nqs_jax(
                nqs_model=nqs,
                hamiltonian=self.hamiltonian,
                lattice=self.lattice,
                hilbert_space=self.hilbert_space,
                observables_dict=None,  # Use default spin observables
                n_samples=config.num_samples * 10
            )
            
            return observables
            
        except Exception as e:
            print(f"Warning: Observable computation failed: {e}")
            return {
                "energy": float('nan'),
                "energy_std": float('nan'),
            }
