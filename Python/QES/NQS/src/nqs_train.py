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

import numpy as np
from contextlib     import contextmanager
from dataclasses    import dataclass, field, asdict
from typing         import Any, Callable, List, Optional, Union, Dict
from pathlib        import Path
from enum           import Enum

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
    from QES.NQS.nqs                            import NQS, VMCSampler
    from QES.NQS.src.tdvp                       import TDVP, TDVPLowerPenalty
    # solvers
    from QES.general_python.algebra.ode         import choose_ode, IVP
    from QES.general_python.algebra.solvers     import SolverType, SolverForm, choose_solver, choose_precond
    # training phases
    from QES.general_python.ml.training_phases  import create_phase_schedulers, PhaseScheduler
except ImportError as e:
    raise ImportError("QES core modules missing.") from e

# ------------------------------------------------------

class NQSTimeModes(Enum):
    """Timing modes for profiling."""
    OFF         = 0   # No timing
    BASIC       = 1   # Basic timing (total time per epoch)
    FIRST       = 2   # Time only the first epoch
    LAST        = 4   # Time only the last epoch
    DETAILED    = 8   # Detailed timing (per phase: sampling, step, update)

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
            "history"       : self.history,
            "history_std"   : self.history_std,
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
    
    _ERR_INVALID_SCHEDULER      = "Invalid scheduler provided. Must be a PhaseScheduler or callable."
    _ERR_NO_NQS                 = "NQS instance must be provided."
    _ERR_INVALID_SOLVER         = "Invalid ODE solver specified."
    _ERR_INVALID_TIMING_MODE    = "Invalid timing mode specified."
    _ERR_INVALID_LOWER_STATES   = "Lower states must be a list of NQS instances."
    
    def __init__( self,
                nqs             : NQS,
                *,
                # Solvers
                lin_solver      : Union[str, Callable]  = SolverType.SCIPY_CG,      # Linear Solver
                pre_solver      : Union[str, Callable]  = None,                     # Preconditioner
                ode_solver      : Union[IVP, str]       = 'Euler',                  # ODE Solver or preset
                tdvp            : Optional[TDVP]        = None,                     # Setup TDVP engine
                # Configuration
                n_batch         : int                   = 1000,                     # Batch size for sampling
                phases          : Union[str, tuple]     = 'default',                # e.g., "kitaev" or (lr_sched, reg_sched)
                # Utilities
                timing_mode     : NQSTimeModes          = NQSTimeModes.LAST,        # Timing mode
                early_stopper   : Any                   = None,                     # Callable or EarlyStopping
                logger          : Optional[Logger]      = None,                     # Logger instance
                lower_states    : List[NQS]             = None,                     # For excited states - list of lower NQS
                # --------------------------------------------------------------
                lr_scheduler    : Optional[Callable]    = None,                     # Direct LR scheduler injection
                reg_scheduler   : Optional[Callable]    = None,                     # Direct Reg scheduler injection
                # --------------------------------------------------------------
                lr              : Optional[float]       = None,                     # Direct LR injection (bypass scheduler)
                reg             : Optional[float]       = None,                     # Direct Reg injection (bypass scheduler)
                # --------------------------------------------------------------
                # Linear Solver + Preconditioner
                # --------------------------------------------------------------
                lin_sigma       : float                 = None,                     # Linear solver sigma, inferred from diag_shift if None
                lin_is_gram     : bool                  = True,                     # Is Gram matrix solver [(S^dagger S) x = b]
                lin_type        : SolverForm           = SolverForm.GRAM,           # Solver form (gram, matvec, matrix)
                # --------------------------------------------------------------
                # TDVP arguments, if TDVP is created internally
                # --------------------------------------------------------------
                use_sr          : bool                  = True,                     # Whether to use SR
                use_minsr       : bool                  = False,                    # Whether to use MinSR
                rhs_prefactor   : float                 = -1.0,                     # RHS prefactor
                diag_shift      : float                 = 1e-5,                     # Initial diagonal shift
                # --------------------------------------------------------------
                **kwargs
            ):
        '''
        Initializes the NQS Trainer.
        '''
        
        if nqs is None:         raise ValueError(self._ERR_NO_NQS)
        self.nqs                = nqs                                           # Most important component
        self.logger             = logger if logger else Logger("nqs_trainer")   # Logger
        self.n_batch            = n_batch
        
        # Validate lower states
        if lower_states is not None:
            if not isinstance(lower_states, list) or not all(isinstance(s, NQS) for s in lower_states):
                raise ValueError(self._ERR_INVALID_LOWER_STATES)
            
        # Utilities
        self.lower_states               = lower_states
        self.early_stopper              = early_stopper
        if isinstance(timing_mode, str):
            try:
                timing_mode             = NQSTimeModes[timing_mode.upper()]
                self.timing_mode        = timing_mode
            except KeyError:
                raise ValueError(self._ERR_INVALID_TIMING_MODE)
            self.timing_mode = timing_mode if isinstance(timing_mode, NQSTimeModes) else NQSTimeModes.BASIC

        # Setup Schedulers (The Integrated Part)
        self._init_reg, self._init_lr   = None, None
        self._set_phases(phases, lr, reg, lr_scheduler=lr_scheduler, reg_scheduler=reg_scheduler)
        
        # Setup Linear Solver + Preconditioner
        try:
            self.lin_solver = choose_solver(solver_id=lin_solver, sigma=lin_sigma, is_gram=lin_is_gram, backend=nqs.backend, **kwargs)
            self.pre_solver = choose_precond(precond_id = pre_solver, **kwargs) if pre_solver else None
        except Exception as e:
            raise ValueError(self._ERR_INVALID_SOLVER) from e
        
        # Setup TDVP (Physics Engine)
        self.tdvp = tdvp
        if self.tdvp is None:
            self.logger.warning("No TDVP engine provided. Creating default TDVP instance.", lvl=0, color='yellow')
            self.tdvp       = TDVP(
                                use_sr          =   use_sr, 
                                use_minsr       =   use_minsr,
                                rhs_prefactor   =   rhs_prefactor,
                                sr_diag_shift   =   diag_shift,
                                sr_lin_solver   =   self.lin_solver,
                                sr_lin_solver_t =   lin_type,
                                sr_precond      =   self.pre_solver,
                                backend         =   nqs.backend
                            )                    
            self.logger.info(f"Created default TDVP engine {self.tdvp}", lvl=1, color='yellow')
            
        # Setup ODE Solver
        try:
            self.ode_solver = ode_solver
            if isinstance(ode_solver, str):
                init_dt             = self.lr_scheduler(0) if self.lr_scheduler else 1e-2
                self.ode_solver     = choose_ode(ode_type=ode_solver, dt=init_dt, backend=nqs.backend)
                
        except Exception as e:
            raise ValueError(self._ERR_INVALID_SOLVER) from e
        
        # JIT Compile Critical Paths
        # We pre-compile the sampling and step functions to avoid runtime overhead
        self._single_step_jit   = nqs.wrap_single_step_jax(batch_size = n_batch)
        
        # State
        self.stats              = NQSTrainStats()

    # ------------------------------------------------------
    #! Private Helpers
    # ------------------------------------------------------

    def _set_phases(self, phases, lr, reg, lr_scheduler=None, reg_scheduler=None, **kwargs):
        ''' Initializes the learning rate and regularization schedulers. '''
        
        if isinstance(phases, str):
            self.logger.info(f"Initializing training phases with preset: '{phases}'")
            self.lr_scheduler, self.reg_scheduler = create_phase_schedulers(phases, self.logger)
        elif isinstance(phases, (tuple, list)) and len(phases) == 2:
            self.lr_scheduler, self.reg_scheduler = phases
            if not (callable(self.lr_scheduler) or isinstance(self.lr_scheduler, PhaseScheduler)):
                raise ValueError(self._ERR_INVALID_SCHEDULER)
            if not (callable(self.reg_scheduler) or isinstance(self.reg_scheduler, PhaseScheduler)):
                raise ValueError(self._ERR_INVALID_SCHEDULER)
        else:
            # Fallback for manual injection
            self.lr_scheduler   = lr_scheduler
            self.reg_scheduler  = reg_scheduler
            
        self._init_lr       = self.lr_scheduler(0)  if self.lr_scheduler    else lr  or 1e-2
        self._init_reg      = self.reg_scheduler(0) if self.reg_scheduler   else reg or 1e-3
    
    # ------------------------------------------------------
    #! Timing Helpers
    # ------------------------------------------------------
        
    @contextmanager
    def _time(self, phase, fn, *args, **kwargs):
        """Context manager for timing phases."""
        result, elapsed = timeit(fn, *args, **kwargs)
        self.stats.timings[phase].append(elapsed)
        yield result
    
    def _timed_execute(self, phase: str, fn: Callable, *args, **kwargs):
        """
        Conditionally executes with timing based on mode.
        If timing is OFF/BASIC, it avoids synchronization barriers.
        
        Parameters:
        -----------
        phase: str
            Phase name for timing ('sample', 'step', 'update').
        fn: Callable
            Function to execute.
        *args, **kwargs:
            Arguments to pass to the function.
        Returns:
        --------
        Result of the function execution.
        """
        
        # Check if we should time this specific phase
        total_steps     = kwargs.get('epoch', None)
        should_time     = (self.timing_mode == NQSTimeModes.DETAILED)
        should_time    |= (self.timing_mode == NQSTimeModes.FIRST and self.stats.timings.n_steps == 0)
        should_time    |= (self.timing_mode == NQSTimeModes.LAST and total_steps is not None and self.stats.timings.n_steps == total_steps - 1)
        kwargs.pop('epoch', None)       # Remove epoch from kwargs to avoid passing it to fn
        
        if should_time:
            result, dt                  = timeit(fn, *args, **kwargs)
            timer_list                  = getattr(self.stats.timings, phase, None)
            if timer_list is not None:  timer_list.append(dt)
            return result
        else:
            timer_list                  = getattr(self.stats.timings, phase, None)
            if timer_list is not None:  timer_list.append(np.nan) # Placeholder for no timing
            return fn(*args, **kwargs)
        
    # ------------------------------------------------------    
        
    def _update_hyperparameters(self, epoch: int, last_loss: float):
        """Syncs schedulers with solvers."""
        
        # 1. Learning Rate -> ODE Time Step
        if self.lr_scheduler:
            new_lr      = self.lr_scheduler(epoch, last_loss)
            self.ode_solver.set_dt(float(new_lr))
            self.stats.lr_history.append(new_lr)
            
            # Use this value directly
            current_dt  = new_lr
        else:
            raw_dt      = getattr(self.ode_solver, 'dt', 1e-3)
            current_dt  = raw_dt() if callable(raw_dt) else raw_dt
        
        # 2. Regularization -> TDVP Diagonal Shift or something else
        if self.reg_scheduler:
            new_reg     = self.reg_scheduler(epoch, last_loss)
            self.tdvp.set_diag_shift(float(new_reg))
            self.stats.reg_history.append(new_reg)
            
            current_reg = new_reg
        else:
            current_reg = self.tdvp.sr_diag_shift

        return current_dt, current_reg

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

    # ------------------------------------------------------

    def train(self, 
            n_epochs            : int   = None, 
            checkpoint_every    : int   = 50,
            *,
            save_path           : str   = None,
            reset_stats         : bool  = True,
            use_pbar            : bool  = True,
            absolute_path       : bool  = False,
            **kwargs) -> NQSTrainStats:
        """
        Main training loop.
        """
        # Timer for the WHOLE epoch (BASIC mode)
        import time
        t0                  = time.time()
        checkpoint_every    = max(1, checkpoint_every)
        
        # Auto-detect epochs from scheduler if not provided
        if n_epochs is None and isinstance(self.lr_scheduler, PhaseScheduler):
            n_epochs = sum(p.epochs for p in self.lr_scheduler.phases)
            self.logger.info(f"Auto-detected total epochs from phases: {n_epochs}")
        
        # Reset stats if needed, if not we continue accumulating
        if reset_stats:
            self.stats  = NQSTrainStats() # Reset stats
            
        n_epochs    = n_epochs or 100
        pbar        = trange(n_epochs, desc="NQS Training", leave=True) if use_pbar else range(n_epochs)
        
        try:
            for epoch in pbar:
                # 1. Scheduling 
                last_E                          = self.stats.history[-1] if self.stats.history else 0.0
                lr, reg                         = self._update_hyperparameters(epoch, last_E)

                # 2. Sampling (Timed)
                # Returns: ((keys), (configs, ansatz_vals), probs)
                sample_out                      = self._timed_execute("sample", self.nqs.sample, reset=(epoch==0), epoch=epoch)
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
                                                    lower_states    = lower_contr,
                                                    epoch           = epoch
                                                )
                dparams, _, (info, meta)        = step_out

                # 4. Update Weights (Timed)
                self._timed_execute("update", self.nqs.set_params, dparams, shapes=meta[0], sizes=meta[1], iscpx=meta[2], epoch=epoch)

                # 5. Global Phase Integration
                # No heavy computation here, simple scalar update
                self.tdvp.update_global_phase(dt=lr)
                self.stats.global_phase.append(self.tdvp.global_phase)

                # 6. Logging & Storage
                mean_loss   = np.real(info.mean_energy)
                self.stats.history.append(mean_loss)
                self.stats.history_std.append(np.real(info.std_energy))

                # Calculate total time for this epoch
                t_sample    = self.stats.timings.sample[-1]
                t_update    = self.stats.timings.update[-1]
                t_step      = self.stats.timings.step[-1]
                acc_ratio   = self.nqs.sampler.accepted_ratio
                
                if use_pbar:
                    postfix = {
                            "loss" : f"{mean_loss:.4f}",
                            "lr"   : f"{lr:.1e}",
                            "acc"  : f"{acc_ratio:.2%}",
                        }
                        
                    # Only add detailed timings to the bar if they are real numbers
                    if not np.isnan(t_step):
                        postfix["t_step"]  = f"{t_step:.2f}s"
                    
                    if self.timing_mode.value < NQSTimeModes.DETAILED.value:
                        postfix["t_epoch"] = f"{(time.time() - t0) / (epoch + 1):.2f}s"
                else:
                    self.logger.info(f"Epoch {epoch}: loss={mean_loss:.4f}, lr={lr:.1e}, acc={acc_ratio:.2%}, t_step={t_step:.2f}s, t_samp={t_sample:.2f}s, t_upd={t_update:.2f}s")

                # Checkpointing
                if epoch % checkpoint_every == 0 or epoch == n_epochs - 1:
                    self.save_checkpoint(epoch, save_path, fmt="h5", overwrite=True, absolute=absolute_path)

                # Early Stopping
                if np.isnan(mean_loss):
                    self.logger.error("Energy is NaN. Stopping training.")
                    break

                if self.early_stopper and self.early_stopper(mean_loss):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user.")
        except StopIteration as e:
            self.logger.info(f"Training stopped: {e}")
        except Exception as e:
            self.logger.error(f"An error occurred during training: {e}")
            raise e
        
        # Finalize
        self.save_checkpoint(epoch, save_path, fmt="h5", overwrite=True, absolute=absolute_path)
        self.stats.history      = np.array(self.stats.history).flatten().tolist()
        self.stats.history_std  = np.array(self.stats.history_std).flatten().tolist()
        self.logger.info(f"Training completed in {time.time() - t0:.2f} seconds over {len(self.stats.history)}/{n_epochs} epochs.", lvl=0, color='green')
        return self.stats

    # ------------------------------------------------------
    #! Checkpointing
    # ------------------------------------------------------

    def save_checkpoint(self, step: int, path: Union[str, Path], fmt: str = "h5", overwrite: bool = True, absolute = False):
        """
        Saves weights + Architecture Metadata.
        
        Parameters:
        -----------
        step: int
            Current training step or epoch. Used for versioning.
        path: str
            File path to save the weights.
        fmt: str
            Format to save weights ('h5', 'json', etc.)
        overwrite: bool
            Whether to overwrite existing files.
        """
        net         = self.nqs.net
        path        = path if path is not None else self.nqs.defdir
        meta        = {
                        "network_class" : net.__class__.__name__,
                        "step"          : step,
                        "last_loss"     : float(self.stats.history[-1]) if self.stats.history else 0.0
                    }
        
        # If path is a directory (ends with /), append filename.
        # If path is a filename, use it directly.
        import os
        if path.endswith(os.sep) or os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            final_path  = os.path.join(path, f"checkpoint_{step}.{fmt}")
            absolute    = True
        else:
            final_path  = path
            absolute    = True # Treat user input as absolute path usually

        return self.nqs.save_weights(
            filename        =   final_path, 
            step            =   step, 
            save_metadata   =   meta, 
            fmt             =   fmt, 
            overwrite       =   overwrite, 
            absolute        =   absolute
        )
        
# ------------------------------------------------------
#! EOF
# ------------------------------------------------------