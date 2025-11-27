'''
NQS Training Module.
Integrates High-Level Phase Scheduling with Low-Level TDVP Physics.

State-of-the-Art features:
1. Automated Phase Scheduling       (Pre-train -> Main -> Refine)
2. Global Phase Evolution tracking  (theta_0)
3. Architecture-aware checkpointing (saves network metadata)
4. Dynamic ODE time-step adjustment (also via learning rate scheduler)

------------------------------------------------------------------------
File                : NQS/nqs_train.py
Author              : Maksymilian Kliczkowski
Email               : maksymilian.kliczkowski@pwr.edu.pl
Copyright           : (c) 2024-2026 Maksymilian Kliczkowski
------------------------------------------------------------------------
'''

import json
import numpy as np
from contextlib     import contextmanager
from dataclasses    import dataclass, field, asdict
from typing         import Any, Callable, List, Optional, Union, Dict

# TQDM for progress bars
try:
    # If running in a notebook, use tqdm.notebook.trange for better display
    from IPython import get_ipython
    if get_ipython() is not None:
        from tqdm.notebook  import trange
    else:
        from tqdm           import trange
except ImportError:
    from tqdm import trange

# QES General Imports
try:
    from QES.general_python.common.flog         import Logger
    from QES.general_python.common.timer        import timeit
except ImportError:
    import logging
    Logger = logging.getLogger 

#! NQS Imports
try:
    from QES.NQS.nqs                            import NQS
    from QES.NQS.src.tdvp                       import TDVP, TDVPLowerPenalty
    from QES.general_python.algebra.ode         import IVP
    from QES.general_python.ml.training_phases  import create_phase_schedulers, PhaseScheduler
except ImportError as e:
    raise ImportError("QES core modules missing.") from e

# ------------------------------------------------------

@dataclass
class NQSTrainTime:
    """Performance timers for profiling."""
    n_steps         : int   = 0
    step            : list  = field(default_factory=list)
    sample          : list  = field(default_factory=list)
    update          : list  = field(default_factory=list)
    total           : list  = field(default_factory=list)
    
    def reset(self):
        self.step, self.sample, self.update, self.total = [], [], [], []
        self.n_steps = 0

# ------------------------------------------------------

@dataclass
class NQSTrainStats:
    """Training statistics container."""
    history         : List[float]         = field(default_factory=list)
    history_std     : List[float]         = field(default_factory=list)
    lr_history      : List[float]         = field(default_factory=list)
    reg_history     : List[float]         = field(default_factory=list)
    global_phase    : List[complex]       = field(default_factory=list) # New: Track theta0
    timings         : NQSTrainTime        = field(default_factory=NQSTrainTime)

    def to_dict(self):
        # Helper for JSON serialization
        return {
            "energy_mean"   : self.history,
            "energy_std"    : self.history_std,
            "lr"            : self.lr_history,
            "reg"           : self.reg_history
        }

# ------------------------------------------------------

class NQSTrainer:
    '''
    Trainer for Neural Quantum States.
    It orchestrates the interaction between the Neural Network, the Physics Engine (TDVP),
    and the Optimization Schedule.
    
    Features:
    - Phase Scheduling (learning rate + regularization)
    - Global Phase Evolution Tracking
    - Checkpointing with architecture metadata
    - Dynamic ODE time-step adjustment
    - Excited State Penalty Terms
    - JIT Compilation for performance
    '''
    
    def __init__( self,
                nqs             : NQS,
                # Solvers
                ode_solver      : Union[IVP, str]   = 'rk4',
                tdvp            : Optional[TDVP]    = None,
                # Configuration
                n_batch         : int               = 1000,         # Batch size for sampling
                phases          : Union[str, tuple] = 'default',    # e.g., "kitaev" or (lr_sched, reg_sched)
                # Utilities
                early_stopper   : Any               = None,         # Callable or EarlyStopping
                logger          : Optional[Logger]  = None,         # Logger instance
                lower_states    : List[NQS]         = None,         # For excited states - list of lower NQS
                **kwargs
            ):
        
        self.nqs                = nqs
        self.logger             = logger if logger else Logger()
        self.n_batch            = n_batch
        self.lower_states       = lower_states
        self.early_stopper      = early_stopper

        # 1. Setup Schedulers (The Integrated Part)
        if isinstance(phases, str):
            self.logger.info(f"Initializing training phases with preset: '{phases}'")
            self.lr_scheduler, self.reg_scheduler = create_phase_schedulers(phases, self.logger)
        elif isinstance(phases, (tuple, list)) and len(phases) == 2:
            self.lr_scheduler, self.reg_scheduler = phases
        else:
            # Fallback for manual injection
            self.lr_scheduler   = kwargs.get('lr_scheduler')
            self.reg_scheduler  = kwargs.get('reg_scheduler')

        # 2. Setup TDVP (Physics Engine)
        if tdvp is None:
            init_reg            = self.reg_scheduler(0) if self.reg_scheduler else 1e-3
            self.tdvp           = TDVP(sr_diag_shift=init_reg)
            self.logger.info(f"Created default TDVP engine with initial reg={init_reg:.1e}")
        else:
            self.tdvp           = tdvp

        # 3. Setup ODE Solver
        if isinstance(ode_solver, str):
            init_dt             = self.lr_scheduler(0) if self.lr_scheduler else 1e-2
            self.ode_solver     = IVP(method=ode_solver, dt=init_dt)
        else:
            self.ode_solver     = ode_solver

        # 4. JIT Compile Critical Paths
        # We pre-compile the sampling and step functions to avoid runtime overhead
        self._single_step_jit   = nqs.wrap_single_step_jax(batch_size = n_batch)
        
        # 5. State
        self.stats              = NQSTrainStats()

    # ------------------------------------------------------
    #! Private Helpers
    # ------------------------------------------------------

    @contextmanager
    def _time(self, phase, fn, *args, **kwargs):
        """Context manager for timing phases."""
        result, elapsed = timeit(fn, *args, **kwargs)
        self.stats.timings[phase].append(elapsed)
        yield result
    
    def _timed_execute(self, phase: str, fn: Callable, *args, **kwargs):
        """
        Executes a function, measures wall time (syncing JAX), and records statistics.
        """
        
        # 1. Execute with robust timing (handles JAX synchronization)
        result, dt                  = timeit(fn, *args, **kwargs)
        
        # 2. Store the duration in the stats object
        # Assumes self.stats.timings has attributes like 'sample', 'step', etc.
        timer_list                  = getattr(self.stats.timings, phase, None)
        if timer_list is not None:  timer_list.append(dt)
            
        return result    
        
    def _update_hyperparameters(self, epoch: int, last_loss: float):
        """Syncs schedulers with solvers."""
        
        # 1. Learning Rate  -> ODE Time Step
        if self.lr_scheduler:
            new_lr = self.lr_scheduler(epoch, last_loss)
            self.ode_solver.set_dt(float(new_lr))
            self.stats.lr_history.append(new_lr)
        
        # 2. Regularization -> TDVP Diagonal Shift or something else
        if self.reg_scheduler:
            new_reg = self.reg_scheduler(epoch, last_loss)
            self.tdvp.set_diag_shift(float(new_reg))
            self.stats.reg_history.append(new_reg)

        return self.ode_solver.dt, self.tdvp.sr_diag_shift

    # ------------------------------------------------------

    def train(self, n_epochs: int = None, save_path: str = "./checkpoints"):
        """
        Main training loop.
        """
        
        # Auto-detect epochs from scheduler if not provided
        if n_epochs is None and isinstance(self.lr_scheduler, PhaseScheduler):
            n_epochs = sum(p.epochs for p in self.lr_scheduler.phases)
            self.logger.info(f"Auto-detected total epochs from phases: {n_epochs}")
        
        n_epochs    = n_epochs or 100
        self.stats  = NQSTrainStats() # Reset stats
        pbar        = trange(n_epochs, desc="NQS Training", leave=True)
        
        for epoch in pbar:
            # 1. Scheduling 
            last_E                          = self.stats.history[-1] if self.stats.history else 0.0
            lr, reg                         = self._update_hyperparameters(epoch, last_E)

            # 2. Sampling (Timed)
            # Returns: ((keys), (configs, ansatz_vals), probs)
            sample_out                      = self._timed_execute("sample", self.nqs.sample, reset=(epoch==0))
            (_, _), (cfgs, cfgs_psi), probs = sample_out
            
            # Handle Excited States (Penalty Terms)
            lower_contr                     = self._prepare_lower_states(cfgs, cfgs_psi)

            # 3. Physics Step (TDVP + ODE) (Timed)
            params_flat                     = self.nqs.get_params(unravel=True)
            step_out                        = self._timed_execute(
                                                "step", 
                                                self.ode_solver.step,
                                                f               = self.tdvp,
                                                y               = params_flat,
                                                t               = 0.0,
                                                est_fn          = self._single_step_jit,
                                                configs         = cfgs,
                                                configs_ansatze = cfgs_psi,
                                                probabilities   = probs,
                                                lower_states    = lower_contr
                                            )
            dparams, _, (info, meta)        = step_out

            # 4. Update Weights (Timed)
            # Wraps self.nqs.set_params()
            self._timed_execute("update", self.nqs.set_params, dparams, 
                                shapes=meta[0], sizes=meta[1], iscpx=meta[2])

            # 5. Global Phase Integration
            # No heavy computation here, simple scalar update
            self.tdvp.update_global_phase(dt=lr)
            self.stats.global_phase.append(self.tdvp.global_phase)

            # 6. Logging & Storage
            mean_E = np.real(info.mean_energy)
            self.stats.history.append(mean_E)
            self.stats.history_std.append(np.real(info.std_energy))

            # Calculate total time for this epoch
            t_sample    = self.stats.timings.sample[-1]
            t_step      = self.stats.timings.step[-1]
            t_update    = self.stats.timings.update[-1]
            
            pbar.set_postfix({
                "E"     : f"{mean_E:.4f}",
                "lr"    : f"{lr:.1e}",
                "t_step": f"{t_step:.2f}s" # Live performance monitoring
            })

            # 7. Checkpointing
            if epoch % 50 == 0 or epoch == n_epochs - 1:
                self.save_checkpoint(epoch, save_path)

            # 8. Early Stopping
            if np.isnan(mean_E):
                self.logger.error("Energy is NaN. Stopping training.")
                break

            if self.early_stopper and self.early_stopper(mean_E):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        return self.stats

    def _prepare_lower_states(self, cfgs_current, excited_on_excited):
        """Helper to prepare penalty terms for excited states."""
        
        if not self.lower_states: return None
        
        lower_contr = []
        for nqs_lower in self.lower_states:
            
            # Sample from the lower state
            (_, _), (cfgs_lower, cfgs_psi_lower), _ = nqs_lower.sample()
            
            if excited_on_excited is None:
                excited_on_excited  = self.nqs.ansatz(cfgs_current)
            
            # Calculate cross-ratios
            # Note: This requires O(N_samples) network evaluations
            penalty = TDVPLowerPenalty(
                excited_on_lower    = self.nqs.ansatz(cfgs_lower),
                lower_on_excited    = nqs_lower.ansatz(cfgs_current),
                excited_on_excited  = excited_on_excited, #! check if correct
                lower_on_lower      = cfgs_psi_lower.flatten(),
                params_j            = nqs_lower.get_params(),
                configs_j           = cfgs_lower,
                beta_j              = nqs_lower.beta_penalty,
                backend_np          = self.nqs.backend
            )
            lower_contr.append(penalty)
        return lower_contr

    def save_checkpoint(self, step: int, path: str):
        """
        Saves weights + Architecture Metadata.
        """
        # 1. Get architecture info
        net     = self.nqs.ansatz
        meta    = {
                    "network_class" : net.__class__.__name__,
                    "input_shape"   : net.input_shape if hasattr(net, 'input_shape') else None,
                    "step"          : step,
                    "last_energy"   : self.stats.history[-1]
                }
        
        # 2. Save via NQS (which handles the actual weight serialization)
        self.nqs.save_weights(step=step, extra_meta=meta)
        
# ------------------------------------------------------
#! EOF
# ------------------------------------------------------