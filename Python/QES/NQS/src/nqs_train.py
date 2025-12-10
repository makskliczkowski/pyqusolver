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

import jax
import numpy as np
from contextlib     import contextmanager
from dataclasses    import dataclass, field, asdict
from typing         import Any, Callable, List, Optional, Union, Dict
from pathlib        import Path
from enum           import Enum
from functools      import partial

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
    from QES.general_python.common.flog         import Logger, get_global_logger
    from QES.general_python.common.timer        import timeit
    from QES.general_python.common.hdf5man      import HDF5Manager
except ImportError:
    raise ImportError("QES general modules missing.")

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
    """
    Training statistics container.
    
    Attributes:
    -----------
    history : List[float]
        Loss/energy values per epoch.
    history_std : List[float]
        Standard deviation of loss per epoch.
    lr_history : List[float]
        Learning rate schedule history.
    reg_history : List[float]
        Regularization schedule history.
    global_phase : List[complex]
        Global phase evolution (theta_0) for wavefunction tracking.
    timings : NQSTrainTime
        Performance timing data.
    seed : Optional[int]
        Random seed used for reproducibility.
    exact_predictions : Optional[np.ndarray]
        Reference values from exact methods (e.g., ED eigenvalues).
        Can be a single value or array of values (ground state, excited states, etc.)
    exact_method : Optional[str]
        Method used to compute exact predictions (e.g., 'lanczos', 'full_diag', 'dmrg').
    """
    history             : List[float]               = field(default_factory=list)
    history_std         : List[float]               = field(default_factory=list)
    lr_history          : List[float]               = field(default_factory=list)
    diag_history        : List[float]               = field(default_factory=list)
    reg_history         : List[float]               = field(default_factory=list)
    global_phase        : List[complex]             = field(default_factory=list)
    timings             : NQSTrainTime              = field(default_factory=NQSTrainTime)
    seed                : Optional[int]             = None
    exact_predictions   : Optional[np.ndarray]      = None  # e.g., ED eigenvalues
    exact_method        : Optional[str]             = None  # e.g., 'lanczos', 'full_diag'

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for HDF5 serialization."""
        result = {
            "history/val"           : self.history,
            "history/std"           : self.history_std,
            "history/lr"            : self.lr_history,
            "history/reg"           : self.reg_history,
            "history/diag_shift"    : self.diag_history,
            "history/theta0"        : np.array(self.global_phase),
            "timings/n_steps"       : self.timings.n_steps,
            "timings/step"          : self.timings.step,
            "timings/sample"        : self.timings.sample,
            "timings/update"        : self.timings.update,
            "timings/total"         : self.timings.total,
        }
        if self.seed is not None:
            result["seed"]              = self.seed
        if self.exact_predictions is not None:
            result["exact/predictions"] = np.asarray(self.exact_predictions)
        if self.exact_method is not None:
            result["exact/method"]      = self.exact_method
        return result
    
    def set_exact(self, predictions: Union[float, List[float], np.ndarray], method: str = None):
        """
        Set exact/reference predictions (e.g., from ED).
        
        Parameters:
        -----------
        predictions : Union[float, List[float], np.ndarray]
            Reference values. Can be:
            - Single value (ground state energy)
            - Array of values (eigenvalues, multiple observables, etc.)
        method : str, optional
            Method used to compute these values (e.g., 'lanczos', 'full_diag', 'dmrg').
        """
        self.exact_predictions  = np.atleast_1d(np.asarray(predictions))
        self.exact_method       = method
        
    @property
    def exact_gs(self) -> Optional[float]:
        """Get the ground state energy from exact predictions (first element)."""
        if self.exact_predictions is not None and len(self.exact_predictions) > 0:
            return float(self.exact_predictions[0])
        return None
    
    @property
    def has_exact(self) -> bool:
        """Check if exact predictions are available."""
        return self.exact_predictions is not None and len(self.exact_predictions) > 0

# ------------------------------------------------------

class NQSTrainer:
    '''
    Trainer for Neural Quantum States.
    
    Orchestrates the interaction between the Neural Network, the Physics Engine (TDVP),
    and the Optimization Schedule. Supports multiple scheduler configuration patterns.
    
    Features
    --------
    - **Phase Scheduling**: Multi-phase training with different LR/Reg per phase
    - **Global Phase Evolution**:   Tracking theta_0 for wavefunction analysis  
    - **Checkpointing**:            Architecture-aware saving with metadata
    - **Dynamic Time-Step**:        ODE dt controlled via learning rate scheduler
    - **Excited States**:           Penalty terms for targeting excited states
    - **JIT Compilation**:          Pre-compiled critical paths for performance
    
    Scheduler Configuration Patterns
    ---------------------------------
    There are multiple ways to configure learning rate (LR) and regularization (Reg) schedules:
    
    **1. Preset Phases (Recommended for beginners)**
    
    Use a named preset that defines multi-phase training:
    
    >>> trainer = NQSTrainer(nqs, phases='default')  # Pre-train -> Main -> Refine
    >>> trainer = NQSTrainer(nqs, phases='kitaev')   # Specialized for Kitaev models
    
    **2. Custom Phase List**
    
    Define your own phases with `LearningPhase` dataclass:
    
    >>> from QES.general_python.ml.training_phases import LearningPhase, PhaseType, PhaseScheduler
    >>> 
    >>> my_phases = [
    ...     LearningPhase(
    ...         name="warmup", epochs=50, phase_type=PhaseType.PRE_TRAINING,
    ...         lr=0.1, lr_schedule="exponential", lr_kwargs={'lr_decay': 0.05},
    ...         reg=0.01, reg_schedule="constant"
    ...     ),
    ...     LearningPhase(
    ...         name="main", epochs=200, phase_type=PhaseType.MAIN,
    ...         lr=0.02, lr_schedule="adaptive", lr_kwargs={'patience': 15, 'lr_decay': 0.5},
    ...         reg=0.001, reg_schedule="constant"
    ...     ),
    ... ]
    >>> lr_sched = PhaseScheduler(my_phases, param_type='lr')
    >>> reg_sched = PhaseScheduler(my_phases, param_type='reg')
    >>> trainer = NQSTrainer(nqs, phases=(lr_sched, reg_sched))
    
    **3. Direct Scheduler Injection**
    
    Pass any callable `(epoch, loss) -> float` as a scheduler:
    
    >>> # Using built-in schedulers
    >>> from QES.general_python.ml.schedulers import choose_scheduler
    >>> lr_sched    = choose_scheduler('cosine', initial_lr=0.01, max_epochs=300, min_lr=1e-5)
    >>> reg_sched   = choose_scheduler('constant', initial_lr=0.001, max_epochs=300)
    >>> trainer     = NQSTrainer(nqs, lr_scheduler=lr_sched, reg_scheduler=reg_sched, phases=None)
    >>> 
    >>> # Using custom lambda
    >>> custom_lr = lambda epoch, loss: 0.01 * (0.99 ** epoch)
    >>> trainer = NQSTrainer(nqs, lr_scheduler=custom_lr, reg_scheduler=lambda e, l: 0.001, phases=None)
    
    **4. Constant Values (Bypass Scheduling)**
    
    Use fixed LR/Reg throughout training:
    
    >>> trainer = NQSTrainer(nqs, lr=0.01, reg=0.001, phases=None)
    
    Available Scheduler Types
    -------------------------
    - ``'constant'``        : Fixed value throughout training
    - ``'exponential'``     : lr * exp(-decay_rate * epoch)
    - ``'step'``            : lr * decay^floor(epoch/step_size)
    - ``'cosine'``          : Cosine annealing from initial_lr to min_lr
    - ``'linear'``          : Linear decay from initial_lr to min_lr
    - ``'adaptive'``        : ReduceLROnPlateau (requires loss metric)
    
    See Also
    --------
    - `QES.general_python.ml.training_phases`   : Phase definitions and presets
    - `QES.general_python.ml.schedulers`        : Low-level scheduler implementations
    '''
    
    _ERR_INVALID_SCHEDULER      = "Invalid scheduler provided. Must be a PhaseScheduler or callable."
    _ERR_NO_NQS                 = "NQS instance must be provided."
    _ERR_INVALID_SOLVER         = "Invalid solver specified."
    _ERR_INVALID_TIMING_MODE    = "Invalid timing mode specified."
    _ERR_INVALID_LOWER_STATES   = "Lower states must be a list of NQS instances."
    
    def _log(self, message: str, lvl: int = 1, log: str = 'info', color: str = 'white', verbose: bool = True):
        """Helper for logging messages."""
        if not verbose:
            return
        if self.logger:
            self.logger.say(message, lvl=lvl, log=log, color=color)
    
    def __init__( self,
                nqs             : NQS,
                *,
                # Solvers
                lin_solver      : Union[str, Callable]  = SolverType.SCIPY_CG,      # Linear Solver
                lin_force_mat   : bool                  = False,                    # Force forming full matrix
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
                reg_scheduler   : Optional[Callable]    = None,                     # Direct Reg scheduler injection (for future L2 reg)
                diag_scheduler  : Optional[Callable]    = None,                     # Direct diag_shift scheduler injection
                # --------------------------------------------------------------
                lr              : Optional[float]       = None,                     # Direct LR injection       (bypass scheduler)
                reg             : Optional[float]       = None,                     # Direct Reg injection      (bypass scheduler)
                diag_shift      : float                 = 1e-5,                     # Initial diagonal shift    (bypass scheduler)
                grad_clip       : Optional[float]       = None,                     # Gradient clipping threshold (None = no clipping)
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
                # --------------------------------------------------------------
                **kwargs
            ):
        '''
        Initialize the NQS Trainer.
        
        Parameters
        ----------
        nqs : NQS
            Neural Quantum State instance to train.
            
        Scheduler Configuration (flexible, mix and match)
        -------------------------------------------------
        phases : Union[str, Tuple], default='default'
            Phase scheduling configuration. Can be:
            - ``str``   : Preset name ('default', 'kitaev')
            - ``tuple`` : (lr_scheduler, reg_scheduler) - each can be string, callable, or PhaseScheduler
            - ``None``  : Use lr_scheduler/reg_scheduler/diag_scheduler params or constant lr/reg/diag values
            
        lr_scheduler : str or Callable, optional
            Learning rate scheduler. Can be:
            - Scheduler type string: 'constant', 'exponential', 'step', 'cosine', 'linear', 'adaptive'
            - Callable with signature: ``(epoch: int, loss: float) -> float``
            - PhaseScheduler instance
            
        reg_scheduler : str or Callable, optional  
            Regularization scheduler (for future L2 weight regularization).
            Same options as lr_scheduler.

        diag_scheduler : str or Callable, optional
            Diagonal shift (SR regularization) scheduler. Controls the diagonal
            shift added to the SR matrix: S -> S + diag_shift * I.
            Same options as lr_scheduler. If provided, diag_shift is the initial value.
            
        lr : float, optional
            Initial learning rate. When used alone (phases=None, no lr_scheduler), creates
            a constant scheduler. Can also override the initial value of other schedulers.
            
        reg : float, optional
            Initial regularization. Same behavior as lr.

        diag_shift : float, optional
            Initial diagonal shift value. When used alone (phases=None, no diag_scheduler), creates
            a constant scheduler. Can also override the initial value of other schedulers.
            
        Scheduler kwargs (passed to factory when using string types):
            - lr_decay : float - Decay rate for exponential/step/adaptive
            - step_size : int - Steps between decays (step scheduler)
            - min_lr : float - Minimum value (cosine, linear, adaptive)
            - patience : int - Epochs before reduction (adaptive)
            - min_delta : float - Minimum improvement (adaptive)
            - lr_initial : float - Alternative key for initial learning rate (overrides initial_lr)
            - lr_max_epochs : int - Alternative key for max_epochs (overrides max_epochs)
            - lr_es : int - Early stopping patience (sets early_stopping_patience)
            
            **kwargs : dict
            Extra args passed to scheduler factory (lr_decay, min_lr, patience, etc.)
            *LR*
            - lr_decay      : float - Decay rate for exponential/step/adaptive
            - lr_step_size  : int - Step size for step scheduler
            - lr_min        : float - Minimum learning rate for cosine/linear/adaptive
            - lr_patience   : int - Patience for adaptive scheduler
            - lr_min_delta  : float - Min improvement for adaptive scheduler
            - lr_initial    : float - Alternative to initial_lr (overrides initial_lr)
            - lr_max_epochs : int - Alternative to max_epochs (overrides max_epochs)
            - lr_es         : int - Early stopping patience (sets early_stopping_patience)
            *Diagonal*
            - diag_decay    : float - Decay rate for diag schedulers
            - diag_step_size: int - Step size for diag schedulers
            - diag_min      : float - Minimum diag shift for diag schedulers
            - diag_patience : int - Patience for diag schedulers
            - diag_min_delta: float - Min improvement for diag schedulers
            - diag_initial  : float - Alternative to initial diag_shift (overrides diag_shift)
            - diag_max_epochs : int - Alternative to max_epochs for diag scheduler
            *Reg*
            - reg_decay     : float - Decay rate for reg schedulers
            - reg_step_size : int - Step size for reg schedulers
            - reg_min       : float - Minimum reg for reg schedulers
            - reg_patience  : int - Patience for reg schedulers
            - reg_min_delta : float - Min improvement for reg schedulers
            - reg_initial   : float - Alternative to initial reg (overrides reg)
            - reg_max_epochs : int - Alternative to max_epochs for reg scheduler
            
        Solvers
        -------
        lin_solver : Union[str, Callable], default=SolverType.SCIPY_CG
            Linear solver for SR equations ('cg', 'gmres', 'scipy_cg', etc.).
            
        pre_solver : Union[str, Callable], optional
            Preconditioner for linear solver ('jacobi', 'ilu', etc.).
            
        ode_solver : Union[IVP, str], default='Euler'
            ODE integrator ('Euler', 'RK4', 'Heun', or IVP instance).
            
        tdvp : TDVP, optional
            Pre-configured TDVP engine. If None, created internally.
            
        TDVP Configuration (when tdvp=None)
        ------------------------------------
        use_sr : bool, default=True
            Enable Stochastic Reconfiguration.
            
        use_minsr : bool, default=False  
            Use MinSR (scales O(N_samples) instead of O(N_params)).
            
        rhs_prefactor : float, default=-1.0
            Prefactor for gradient RHS (imaginary time: -1, real time: -1j).
            
        diag_shift : float, default=1e-5
            Initial diagonal regularization for SR matrix.
            
        grad_clip : float, optional
            Maximum L2 norm for gradient clipping. If None, no clipping is applied.
            Helps stabilize training for CNNs and complex networks with occasional
            large gradients. Typical values: 1.0-10.0.
            
        Other Parameters
        ----------------
        n_batch : int, default=1000
            Batch size for VMC sampling.
            
        timing_mode : NQSTimeModes, default=NQSTimeModes.LAST
            Profiling mode (OFF, BASIC, FIRST, LAST, DETAILED).
            
        early_stopper : Callable, optional
            Early stopping criterion.
            
        lower_states : List[NQS], optional
            Lower energy states for excited state targeting.
            
        Examples
        --------
        >>> # 1. Preset phases (recommended for beginners)
        >>> trainer     = NQSTrainer(nqs, phases='default')
        >>> trainer     = NQSTrainer(nqs, phases='kitaev')
        
        >>> # 2. String scheduler types (simple and flexible)
        >>> trainer     = NQSTrainer(nqs, lr_scheduler='cosine', lr=0.01, reg=0.001, diag_shift=1e-5, phases=None)
        >>> trainer     = NQSTrainer(nqs, lr_scheduler='adaptive', lr=0.02, 
        ...                      patience=20, lr_decay=0.5, phases=None)  # kwargs passed to scheduler
        
        >>> # 3. Constant values (simplest)
        >>> trainer     = NQSTrainer(nqs, lr=0.01, reg=0.001, diag_shift=1e-5, phases=None)
        
        >>> # 4. Tuple of string schedulers
        >>> trainer     = NQSTrainer(nqs, phases=('exponential', 'constant'), lr=0.05, lr_decay=0.02)
        
        >>> # 5. Custom callable scheduler
        >>> from QES.general_python.ml.schedulers import choose_scheduler
        >>> lr          = choose_scheduler('cosine', initial_lr=0.01, max_epochs=500, min_lr=1e-5)
        >>> trainer     = NQSTrainer(nqs, lr_scheduler=lr, reg=0.001, diag_shift=1e-5, phases=None)
        '''
        
        if nqs is None:         raise ValueError(self._ERR_NO_NQS)
        self.nqs                = nqs                                               # Most important component
        self.logger             = logger
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
                self._log(f"Timing mode set to: {self.timing_mode.name}", lvl=1, color='green')
            except KeyError:
                raise ValueError(self._ERR_INVALID_TIMING_MODE)
            self.timing_mode = timing_mode if isinstance(timing_mode, NQSTimeModes) else NQSTimeModes.BASIC

        # Setup Schedulers (The Integrated Part)
        self._init_reg, self._init_lr, self._init_diag = reg, lr, diag_shift
        self._set_phases(phases, lr, reg, lr_scheduler=lr_scheduler, reg_scheduler=reg_scheduler, diag_scheduler=diag_scheduler, diag_shift=diag_shift, **kwargs)
        
        # Setup Linear Solver + Preconditioner
        try:
            self.lin_solver = choose_solver(solver_id=lin_solver, sigma=lin_sigma, is_gram=lin_is_gram, backend=nqs.backend_str, **kwargs)
            self.pre_solver = choose_precond(precond_id = pre_solver, **kwargs) if pre_solver else None
        except Exception as e:
            raise ValueError(self._ERR_INVALID_SOLVER) from e
        
        # Setup TDVP (Physics Engine)
        self.tdvp       = tdvp
        self.grad_clip  = grad_clip  # Store for later use
        if self.tdvp is None:
            self._log("No TDVP engine provided. Creating default TDVP instance.", lvl=0, color='yellow')
            self.tdvp = TDVP(
                            use_sr          =   use_sr, 
                            use_minsr       =   use_minsr,
                            rhs_prefactor   =   rhs_prefactor,
                            sr_diag_shift   =   diag_shift,
                            sr_lin_solver   =   self.lin_solver,
                            sr_lin_solver_t =   lin_type,
                            sr_precond      =   self.pre_solver,
                            backend         =   nqs.backend,
                            logger          =   logger,
                            sr_form_matrix  =   lin_force_mat,
                            sr_snr_tol      =   kwargs.get('sr_snr_tol', 1e-12),
                            sr_lin_x0       =   kwargs.get('sr_lin_x0', None),
                            sr_maxiter      =   kwargs.get('sr_maxiter', 1000),
                            sr_pinv_cutoff  =   kwargs.get('sr_pinv_cutoff', 1e-8),
                            use_timing      =   kwargs.get('use_timing', False)
                        )                    
            self._log(f"Created default TDVP engine {self.tdvp}", lvl=1, color='yellow')
        
        # Set gradient clipping on TDVP
        if grad_clip is not None:
            self.tdvp.set_grad_clip(grad_clip)
            self._log(f"Gradient clipping enabled: max_norm={grad_clip:.2e}", lvl=1, color='cyan')
            
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
        # self._single_step_jit   = nqs.wrap_single_step_jax(batch_size = n_batch)
        self._single_step_jit   = jax.jit(nqs.wrap_single_step_jax(batch_size = n_batch))
        
        # Define a function that runs the WHOLE step (ODE + TDVP + Energy)
        def train_step_logic(f, est_fn, y, t, configs, configs_ansatze, probabilities, lower_states):
            # We pass them down to the solver's **rhs_args
            new_params, new_t, (info, meta) = self.ode_solver.step(
                f               =   f, 
                t               =   t, 
                y               =   y, 
                # These pass into **rhs_args of the solver:
                est_fn          =   est_fn,
                configs         =   configs,
                configs_ansatze =   configs_ansatze,
                probabilities   =   probabilities,
                lower_states    =   lower_states
            )
            loss_info = (info.mean_energy, info.std_energy)
            return new_params, new_t, (loss_info, meta)
        
        # NOTE: Do NOT JIT train_step_logic because TDVP has side effects (stores global phase).
        # TODO: Investigate if there's a way to JIT this without side effects.
        # JIT compilation would cause tracer leaks from compute_global_phase_evolution.
        self._step_jit          = train_step_logic
        # self._step_jit          = jax.jit(train_step_logic, static_argnames=['f', 'est_fn', 'lower_states'])
                    
        # State
        self.stats              = NQSTrainStats()

    # ------------------------------------------------------
    #! Private Helpers
    # ------------------------------------------------------

    def _set_phases(self, phases, lr, reg, lr_scheduler=None, reg_scheduler=None, diag_scheduler=None, diag_shift=None, n_epochs=500, **kwargs):
        '''
        Initialize learning rate, regularization, and diagonal shift schedulers.
        
        Supports multiple configuration patterns:
        
        1. Preset string: phases='default' or phases='kitaev'
        2. Tuple of schedulers: phases=(lr_sched, reg_sched)
        3. String scheduler types: lr_scheduler='cosine', reg_scheduler='constant'
        4. Direct float values: lr=0.01, reg=0.001 (constant schedulers)
        5. Mix: lr_scheduler='adaptive' with lr=0.05 (sets initial value)
        
        Parameters
        ----------
        phases : str or tuple, optional
            Preset name or (lr_scheduler, reg_scheduler) tuple.
        lr : float, optional
            Initial learning rate. Overrides preset/scheduler initial value.
        reg : float, optional
            Initial regularization. Overrides preset/scheduler initial value.
        lr_scheduler : str, Callable, or PhaseScheduler, optional
            LR scheduler. Can be scheduler type string ('cosine', 'exponential', etc.)
        reg_scheduler : str, Callable, or PhaseScheduler, optional
            Reg scheduler (for future L2 regularization).
        diag_scheduler : str, Callable, or PhaseScheduler, optional
            Diagonal shift (SR regularization) scheduler.
        diag_shift : float, optional
            Initial diagonal shift value.
        n_epochs : int, default=500
            Max epochs for auto-created schedulers (when using string types).
        **kwargs : dict
            Extra args passed to scheduler factory (lr_decay, min_lr, patience, etc.)
            *LR*
            - lr_decay      : float - Decay rate for exponential/step/adaptive
            - lr_step_size  : int - Step size for step scheduler
            - lr_min        : float - Minimum learning rate for cosine/linear/adaptive
            - lr_patience   : int - Patience for adaptive scheduler
            - lr_min_delta  : float - Min improvement for adaptive scheduler
            *Diagonal*
            - diag_decay    : float - Decay rate for diag schedulers
            - diag_step_size: int - Step size for diag schedulers
            - diag_min      : float - Minimum diag shift for diag schedulers
            - diag_patience : int - Patience for diag schedulers
            - diag_min_delta: float - Min improvement for diag schedulers
            *Reg*
            - reg_decay     : float - Decay rate for reg schedulers
            - reg_step_size : int - Step size for reg schedulers
            - reg_min       : float - Minimum reg for reg schedulers
            - reg_patience  : int - Patience for reg schedulers
            - reg_min_delta : float - Min improvement for reg schedulers
        -----------
        '''
        from QES.general_python.ml.schedulers import choose_scheduler
        
        diag_kwargs = {k: v for k, v in kwargs.items() if k.startswith('diag_')         }
        reg_kwargs  = {k: v for k, v in kwargs.items() if k.startswith('reg_')          }
        lr_kwargs   = {k: v for k, v in kwargs.items() if k.startswith('lr_')           }
        # transform keys to be accepted by scheduler factory
        
        diag_kwargs = {k.replace('diag_',   'lr_'): v for k, v in diag_kwargs.items()   }
        if self._init_diag is not None: diag_kwargs['lr_initial'] = self._init_diag
        reg_kwargs  = {k.replace('reg_',    'lr_'): v for k, v in reg_kwargs.items()    }
        if self._init_reg is not None:  reg_kwargs['lr_initial']  = self._init_reg
        lr_kwargs   = {k.replace('lr_',     'lr_'): v for k, v in lr_kwargs.items()     }
        if self._init_lr is not None:   lr_kwargs['lr_initial']   = self._init_lr
        
        # Helper to create scheduler from string or passthrough
        def _resolve_scheduler(sched, init_val, param_name, max_epochs, **kwargs):
            if sched is None:           
                if init_val is not None: # Constant scheduler from float value
                    return choose_scheduler('constant', initial_lr=init_val, max_epochs=max_epochs, logger=self.logger)
                return None
            
            elif isinstance(sched, str):
                # String scheduler type -> create via factory
                init = init_val if init_val is not None else (1e-2 if param_name == 'lr' else 1e-3)
                self._log(f"Creating '{sched}' scheduler for {param_name} (init={init:.2e})", lvl=2, color='blue')
                return choose_scheduler(sched, initial_lr=init, max_epochs=max_epochs, logger=self.logger, **kwargs)
            
            elif callable(sched) or isinstance(sched, PhaseScheduler):
                return sched
            else:
                raise ValueError(f"Invalid {param_name}_scheduler type: {type(sched)}")
        
        # Check if user explicitly provided schedulers or lr/reg values
        # If so, these take precedence over the default 'phases' preset
        user_provided_scheduler = (lr_scheduler is not None or reg_scheduler is not None or lr is not None or reg is not None)
        
        # User provided lr_scheduler/reg_scheduler or lr/reg directly -> use them
        if user_provided_scheduler and (phases == 'default' or phases is None):
            self.lr_scheduler   = _resolve_scheduler(lr_scheduler,      lr,         'lr',   n_epochs, **lr_kwargs)
            self.reg_scheduler  = _resolve_scheduler(reg_scheduler,     reg,        'reg',  n_epochs, **reg_kwargs)
            self.diag_scheduler = _resolve_scheduler(diag_scheduler,    diag_shift, 'diag', n_epochs, **diag_kwargs)
        
        # Preset string (e.g., 'default', 'kitaev') - only if no direct scheduler provided
        elif isinstance(phases, str):
            self._log(f"Initializing training phases with preset: '{phases}'")
            self.lr_scheduler, self.reg_scheduler   = create_phase_schedulers(phases, self.logger)
            # diag_scheduler is separate from phase presets
            self.diag_scheduler                     = _resolve_scheduler(diag_scheduler, diag_shift, 'diag', n_epochs, **diag_kwargs)
            
        # Tuple of schedulers
        elif isinstance(phases, (tuple, list)) and len(phases) == 2:
            # Validate and resolve if strings
            self.lr_scheduler, self.reg_scheduler   = phases
            self.lr_scheduler                       = _resolve_scheduler(self.lr_scheduler,     lr,         'lr',   n_epochs, **lr_kwargs)
            self.reg_scheduler                      = _resolve_scheduler(self.reg_scheduler,    reg,        'reg',  n_epochs, **reg_kwargs)
            self.diag_scheduler                     = _resolve_scheduler(diag_scheduler,        diag_shift, 'diag', n_epochs, **diag_kwargs)
            
        # No phases -> use injected schedulers or create from lr/reg
        else:
            self.lr_scheduler                       = _resolve_scheduler(lr_scheduler,          lr,         'lr',   n_epochs, **lr_kwargs)
            self.reg_scheduler                      = _resolve_scheduler(reg_scheduler,         reg,        'reg',  n_epochs, **reg_kwargs)
            self.diag_scheduler                     = _resolve_scheduler(diag_scheduler,        diag_shift, 'diag', n_epochs, **diag_kwargs)
        
        # Compute initial values (override with explicit lr/reg if provided)
        if self.lr_scheduler:
            self._init_lr = lr if lr is not None else self.lr_scheduler(0)
        else:
            self._init_lr = lr or 1e-2
            
        if self.reg_scheduler:
            self._init_reg = reg if reg is not None else self.reg_scheduler(0)
        else:
            self._init_reg = reg or 1e-3
        
        # Initial diagonal shift
        if self.diag_scheduler:
            self._init_diag = diag_shift if diag_shift is not None else self.diag_scheduler(0)
        else:
            self._init_diag = diag_shift or 1e-5
    
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
    #! Hyperparameter Updates
    # ------------------------------------------------------
        
    def _update_hyperparameters(self, epoch: int, last_loss: float):
        """Syncs schedulers with solvers."""
        
        # 1. Learning Rate -> ODE Time Step
        if self.lr_scheduler:
            new_lr          = self.lr_scheduler(epoch, last_loss)
            self.ode_solver.set_dt(float(new_lr))
            self.stats.lr_history.append(new_lr)
            current_dt      = new_lr
        else:
            raw_dt          = getattr(self.ode_solver, 'dt', 1e-3)
            current_dt      = raw_dt() if callable(raw_dt) else raw_dt
        
        # 2. Diagonal Shift (SR regularization) -> TDVP
        if self.diag_scheduler:
            new_diag        = self.diag_scheduler(epoch, last_loss)
            self.tdvp.set_diag_shift(float(new_diag))
            self.stats.diag_history.append(new_diag)
            current_diag    = new_diag
        else:
            current_diag    = self.tdvp.sr_diag_shift

        # 3. Other Regularization
        if self.reg_scheduler:
            new_reg         = self.reg_scheduler(epoch, last_loss)
            self.tdvp.set_regularization(float(new_reg))
            self.stats.reg_history.append(new_reg)
            current_reg     = new_reg
        else:
            current_reg     = self.tdvp.regularization

        return current_dt, current_reg, current_diag

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
    #! Main Training Loop
    # ------------------------------------------------------

    def train(self, 
            n_epochs            : int   = None, 
            checkpoint_every    : int   = 50,
            *,
            save_path           : str   = None,
            reset_stats         : bool  = True,
            use_pbar            : bool  = True,
            **kwargs) -> NQSTrainStats:
        """
        Main training loop.
        
        Parameters:
        -----------
        n_epochs: int
            Total number of training epochs for this call. If None, defaults to 100 or inferred from scheduler.
        checkpoint_every: int
            Frequency of saving checkpoints (in epochs).
        save_path: str
            Path to save checkpoints.
        reset_stats: bool
            Whether to reset training statistics at the start. If False, training continues
            from the previous epoch count and history is appended. Schedulers also continue
            from the correct global epoch (e.g., if you trained 100 epochs before, calling
            train(n_epochs=50, reset_stats=False) will run epochs 100-149).
        use_pbar: bool
            Whether to use a progress bar during training.
        **kwargs:
            Additional arguments passed to the checkpoint saving function.
            - exact_predictions : Union[float, List[float], np.ndarray]
                Reference values for exact predictions (e.g., from ED).
            - exact_method : str, optional
                Method used to compute exact predictions (e.g., 'lanczos', 'full_diag'), informative only.
        
        Returns:
        -----------
        NQSTrainStats
            Training statistics collected during the run.
            
        Example:
        --------
            >>> trainer = NQSTrainer(nqs)
            >>> stats = trainer.train(n_epochs=500, checkpoint_every=100, save_path="./checkpoints/")
            ... Training completed in 1234.56 seconds over 500 epochs.
            
            >>> # Continue training from where we left off
            >>> stats = trainer.train(n_epochs=200, reset_stats=False)
            ... Continuing training from epoch 500 (reset_stats=False)
            ... Training completed. Total epochs: 700 (this run: 200, started at: 500).
        """
        # Timer for the WHOLE epoch (BASIC mode)
        import time
        checkpoint_every    = max(1, checkpoint_every)
        
        # Auto-detect epochs from scheduler if not provided
        if n_epochs is None and isinstance(self.lr_scheduler, PhaseScheduler):
            n_epochs = sum(p.epochs for p in self.lr_scheduler.phases)
            self._log(f"Auto-detected total epochs from phases: {n_epochs}")
        
        # Set exact information on NQS object if provided
        if 'exact_predictions' in kwargs:
            ex_pred = kwargs['exact_predictions']
            if ex_pred is not None:
                self.nqs.exact  = {
                    'exact_predictions' : ex_pred,
                    'exact_method'      : kwargs.get('exact_method', None),
                    'exact_energy'      : float(ex_pred) if np.ndim(ex_pred) == 0 else ex_pred[len(self.lower_states)] if self.lower_states else ex_pred[0]
                }

        # Reset stats if needed, if not we continue accumulating
        if reset_stats:
            self.stats  = NQSTrainStats() # Reset stats
            start_epoch = 0
        else:
            # Continue from where we left off
            start_epoch = len(self.stats.history)
            if start_epoch > 0:
                self._log(f"Continuing training from epoch {start_epoch} (reset_stats=False)", lvl=1, color='cyan')
        
        t0          = (time.time() + self.stats.timings.total[-1]) if len(self.stats.timings.total) > 0 else time.time()
        n_epochs    = n_epochs or 100
        pbar        = trange(n_epochs, desc="NQS Training", leave=True) if use_pbar else range(n_epochs)
        
        try:
            for epoch in pbar:
                # Global epoch accounts for previous training when continuing
                global_epoch                    = start_epoch + epoch
                
                # Scheduling (use global_epoch so schedulers continue properly)
                last_E                          = self.stats.history[-1] if len(self.stats.history) > 0 else 0.0
                lr, reg, diag                   = self._update_hyperparameters(epoch, last_E) # use local epoch for schedulers as we are in a single phase

                # Sampling (Timed)
                # Returns: ((keys), (configs, ansatz_vals), probs)
                # Note: reset=(epoch==0) resets sampler only at start of this train() call
                sample_out                      = self._timed_execute("sample", self.nqs.sample, reset=(epoch==0), epoch=global_epoch, num_samples=kwargs.get('num_samples', None), num_chains=kwargs.get('num_chains', None))
                (_, _), (cfgs, cfgs_psi), probs = sample_out
                if epoch == 0:                  self._log(f"Sampled {cfgs.shape[0]} configurations with batch size {self.n_batch}", lvl=1, color='green')
                
                # Handle Excited States (Penalty Terms)
                lower_contr                     = self._prepare_lower_states(cfgs, cfgs_psi)
                if epoch == 0:                  self._log(f"Prepared lower states for excited state handling", lvl=1, color='green')
                # Physics Step (TDVP / ODE) (Timed)
                params_flat                     = self.nqs.get_params(unravel=True)
                step_out                        = self._timed_execute(
                                                    "step", 
                                                    self._step_jit,
                                                    f               = self.tdvp,
                                                    y               = params_flat,
                                                    t               = 0.0,
                                                    est_fn          = self._single_step_jit,
                                                    configs         = cfgs,
                                                    configs_ansatze = cfgs_psi,
                                                    probabilities   = probs,
                                                    lower_states    = lower_contr,
                                                    epoch           = global_epoch
                                                )
                dparams, _, ((mean_loss, std_loss), meta) = step_out

                # 4. Update Weights (Timed)
                self._timed_execute("update", self.nqs.set_params, dparams, shapes=meta[0], sizes=meta[1], iscpx=meta[2], epoch=global_epoch)

                # 5. Global Phase Integration
                # No heavy computation here, simple scalar update
                self.tdvp.update_global_phase(dt=lr)
                self.stats.global_phase.append(self.tdvp.global_phase)

                # 6. Logging & Storage
                mean_loss                   = np.real(mean_loss)
                self.stats.timings.n_steps += 1
                self.stats.history.append(mean_loss)
                self.stats.history_std.append(np.real(std_loss))
                self.stats.timings.total.append(time.time() - t0)

                # Calculate total time for this epoch
                t_sample    = self.stats.timings.sample[-1]
                t_update    = self.stats.timings.update[-1]
                t_step      = self.stats.timings.step[-1]
                acc_ratio   = self.nqs.sampler.accepted_ratio
                
                if use_pbar:
                    postfix = {
                            "epoch": f"G:{global_epoch},L:{epoch}",
                            "loss" : f"{mean_loss:.4f}",
                            "acc"  : f"{acc_ratio:.2%}",
                        }
                    if np.real(self.tdvp.rhs_prefactor) != 0:
                        postfix["lr"]       = f"{lr:.2e}"
                    else:
                        postfix["dt"]       = f"{lr:.2e}"
                        
                    if reg is not None and reg > 0:
                        postfix["reg"]      = f"{reg:.1e}"
                    if diag is not None and diag > 0:
                        postfix["diag"]     = f"{diag:.1e}"
                    
                    # Only add detailed timings to the bar if they are real numbers
                    if not np.isnan(t_step):
                        postfix["t_step"]   = f"{t_step:.2f}s"
                        postfix["t_samp"]   = f"{t_sample:.2f}s"
                        postfix["t_upd"]    = f"{t_update:.2f}s"
                    
                    if self.timing_mode.value <= NQSTimeModes.DETAILED.value:
                        postfix["t_epoch"] = f"{(time.time() - t0) / (epoch + 1):.2f}s"
                        
                    pbar.set_postfix(postfix)
                else:
                    self._log(f"Epoch G:{global_epoch},L:{epoch}: loss={mean_loss:.4f}, lr={lr:.1e}, acc={acc_ratio:.2%}, t_step={t_step:.2f}s, t_samp={t_sample:.2f}s, t_upd={t_update:.2f}s")

                # Checkpointing (use global_epoch for consistent checkpoint naming)
                if epoch % checkpoint_every == 0 or epoch == n_epochs - 1:
                    self.save_checkpoint(global_epoch, save_path, fmt="h5", overwrite=True, **kwargs)

                # Early Stopping
                if np.isnan(mean_loss):
                    self._log("Energy is NaN. Stopping training.", lvl=0, color='red')
                    break

                if self.early_stopper and self.early_stopper(mean_loss):
                    self._log(f"Early stopping triggered at epoch G:{global_epoch},L:{epoch}", lvl=0, color='yellow')
                    break
                
        except KeyboardInterrupt:
            self._log("Training interrupted by user.", lvl=0, color='red')
        except StopIteration as e:
            self._log(f"Training stopped: {e}", lvl=0, color='yellow')
        except Exception as e:
            self._log(f"An error occurred during training: {e}", lvl=0, color='red')
            if use_pbar: pbar.close()
            raise e
        
        # Finalize
        if use_pbar:            pbar.close()
        self.stats.history      = np.array(self.stats.history).flatten().tolist()
        self.stats.history_std  = np.array(self.stats.history_std).flatten().tolist()
        final_epoch             = start_epoch + epoch + 1  # Total epochs trained so far
        self.save_checkpoint(final_epoch, save_path, fmt="h5", overwrite=True, verbose=True, **kwargs)
        self._log(f"Training completed in {time.time() - t0:.2f} seconds. Total epochs: {len(self.stats.history)} (this run: {n_epochs}, started at: {start_epoch}).", lvl=0, color='green')
        return self.stats

    # ------------------------------------------------------
    #! Checkpointing
    # ------------------------------------------------------

    def save_checkpoint(self, step: int, path: Union[str, Path] = None, *, fmt: str = "h5", overwrite: bool = True, **kwargs) -> Union[Path, None]:
        """
        Saves weights + Architecture Metadata + Training Stats.
        
        The checkpoint manager handles path resolution:
        - If path is a directory: saves to {path}/checkpoint_{step}.{fmt}
        - If path is a file: saves directly to that file
        - If path is None: uses default directory from NQS
        
        Parameters:
        -----------
        step : Union[int, str]
            Current training step or epoch. Used for versioning.
        path : Union[str, Path], optional
            File path or directory to save the weights.
            If None, uses NQS default directory.
        fmt : str
            Format to save weights ('h5', 'json', etc.) - used for HDF5 fallback.
        overwrite : bool
            Whether to overwrite existing files.
            
        Returns:
        --------
        Path : The path where the checkpoint was saved.
        """
        import os
        
        net         = self.nqs.net
        path        = str(path) if path is not None else str(self.nqs.defdir)
        seed        = getattr(self.nqs, '_net_seed', getattr(self.nqs, '_seed', None))
        
        # Update stats with seed before saving
        self.stats.seed                 = seed
        self.stats.exact_predictions    = kwargs.get('exact_predictions',   None) if self.stats.exact_predictions is None else self.stats.exact_predictions
        
        # Build comprehensive metadata
        meta                = {
                                "network_class" : net.__class__.__name__,
                                "step"          : step,
                                "seed"          : seed,
                                "last_loss"     : float(self.stats.history[-1]) if self.stats.history else 0.0,
                                "n_epochs"      : len(self.stats.history),
                            }
        
        # Resolve path for stats file
        if path.endswith(os.sep) or os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            final_path          = os.path.join(path, f"checkpoint_{step}.{fmt}")
            final_path_stats    = os.path.join(path, "stats.h5")
        else:
            # Path is a specific file
            final_path          = path
            final_path_stats    = os.path.join(str(Path(path).parent), "stats.h5")
            os.makedirs(str(Path(path).parent), exist_ok=True)

        if not overwrite and os.path.exists(final_path):
            self._log(f"Checkpoint file {final_path} already exists and overwrite is False. Skipping save.", lvl=1, color='yellow')
            return None
        
        # Save training history/stats
        HDF5Manager.save_hdf5(
            directory   = os.path.dirname(final_path_stats), 
            filename    = os.path.basename(final_path_stats), 
            data        = self.stats.to_dict()
        )
        self._log(f"Saved training stats to {final_path_stats}", lvl=2, verbose=kwargs.get('verbose', False))

        # Delegate weight saving to NQS (which uses checkpoint manager)
        return self.nqs.save_weights(
            filename    = final_path,
            step        = step,
            metadata    = meta, 
        )
        
    def load_checkpoint(self, step: Optional[Union[int, str]] = None, path: Optional[Union[str, Path]] = None, *, fmt: str = "h5", load_weights: bool = True, fallback_latest: bool = True) -> NQSTrainStats:
        """
        Loads training history from checkpoint.
        
        Parameters:
        -----------
        step : Optional[Union[int, str]]
            Training step or epoch to load. Can be:
            - int: 
                specific step number
            - 'latest': 
                load the most recent checkpoint
            - None: 
                same as 'latest' for Orbax, or finds latest for HDF5
        path : Union[str, Path], optional
            Path to the checkpoint directory or file.
            If None, uses NQS default directory.
        fmt : str
            Format of the saved weights ('h5', 'json', etc.)
        load_weights : bool
            Whether to load the weights from the checkpoint.
        fallback_latest : bool
            If True and the requested step is not found, try to load the latest 
            available checkpoint instead of raising an error.
            
        Returns:
        --------
        NQSTrainStats
            Loaded training statistics.
            
        Examples:
        ---------
        >>> stats = trainer.load_checkpoint(step=100)        # Load specific step
        >>> stats = trainer.load_checkpoint(step='latest')   # Load most recent
        >>> stats = trainer.load_checkpoint()                # Same as 'latest'
        """
        import os, re
        
        # Resolve the search directory
        if path is None:
            search_dir      = self.nqs.ckpt_manager.directory
            path_str        = str(self.nqs.defdir) # Fallback for stats file
        else:
            path_obj        = Path(path)
            if path_obj.is_file():
                search_dir  = path_obj.parent
                path_str    = str(path)
            else:
                search_dir  = path_obj
                path_str    = str(path)

        # Scan directory for available steps (Independent of Manager)
        available_steps     = []
        if os.path.exists(search_dir):
            # Scan for explicit checkpoint folders or files (e.g. checkpoint_100.h5 or /100/)
            for item in os.listdir(search_dir):
                
                # Match "checkpoint_123.h5" or directory names that are integers "123"
                if item.isdigit():
                    available_steps.append(int(item))
                    
                elif item.startswith("checkpoint_") and (item.endswith(f".{fmt}") or os.path.isdir(os.path.join(search_dir, item))):
                    try:
                        # Extract number from "checkpoint_123.h5"
                        num = re.search(r'(\d+)', item)
                        if num: 
                            available_steps.append(int(num.group(1)))
                    except: 
                        pass
        
        # Determine the requested_step_for_manager
        requested_step_for_manager = None
        if step is None or (isinstance(step, str) and step.lower() == 'latest'):
            requested_step_for_manager = None # Let Orbax's native latest resolver handle it
        else:
            requested_step_for_manager = int(step) # Convert explicit step to int

        # Load Statistics (History)
        # Determine stats filename
        if os.path.isfile(path_str) and "stats" in path_str:
            stats_file = path_str
        elif os.path.isdir(path_str):
            stats_file = os.path.join(path_str, "stats.h5")
        else:
            stats_file = os.path.join(os.path.dirname(path_str), "stats.h5")

        try:
            # Assuming HDF5Manager is available in context
            # Reconstruct NQSTrainStats object (abbreviated for clarity, use your full constructor)
            stats_data = HDF5Manager.read_hdf5(file_path=Path(stats_file))
            self.stats = NQSTrainStats(
                            history             = list(stats_data.get("/history/val",        [])),
                            history_std         = list(stats_data.get("/history/std",        [])),
                            lr_history          = list(stats_data.get("/history/lr",         [])),
                            reg_history         = list(stats_data.get("/history/reg",        [])),
                            diag_history        = list(stats_data.get('/history/diag_shift', [])),
                            global_phase        = list(stats_data.get("/history/theta0",     [])),
                            seed                = stats_data.get("/seed",               None),
                            exact_predictions   = stats_data.get("/exact/predictions",  None),
                            exact_method        = stats_data.get("/exact/method",       None),
                            timings             = NQSTrainTime(
                                                    n_steps = stats_data.get("/timings/n_steps", 0),
                                                    step    = list(stats_data.get("/timings/step",      [])),
                                                    sample  = list(stats_data.get("/timings/sample",    [])),
                                                    update  = list(stats_data.get("/timings/update",    [])),
                                                    total   = list(stats_data.get("/timings/total",     [])),
                                                    )
                        )
            self._log(f"Loaded stats from {stats_file}", lvl=1, color='green')
        except Exception as e:
            self._log(f"Stats load failed ({e}). Starting fresh stats.", lvl=1, color='yellow')

        # Load Weights Logic
        if load_weights: # No need for target_step here, it's passed below
            try:
                ckpt_filename = None
                
                # If path is to a specific .h5 file, use it directly as filename
                if path is not None and str(path).endswith('.h5'):
                    ckpt_filename = str(path)

                # NQS.load_weights will pass this to Manager. 
                self.nqs.load_weights(step=requested_step_for_manager, filename=ckpt_filename)
                
                # After successful load, if requested_step_for_manager was None, resolve what step was actually loaded
                loaded_step_info = requested_step_for_manager
                if loaded_step_info is None:
                    # Get the step that Orbax actually loaded (use manager's latest_step)
                    loaded_step_info = self.nqs.ckpt_manager.latest_step if self.nqs.ckpt_manager.use_orbax else "latest (unknown)"


                self._log(f"Successfully loaded weights for step {loaded_step_info}", lvl=1, color='green')
                return self.stats
                
            except Exception as e:
                self._log(f"Loading weights failed: {e}", lvl=0, color='red')
                
                # Handle fallback_latest if native Orbax load failed for a specific step
                if requested_step_for_manager is not None and fallback_latest:
                     self._log(f"Failed to load specific step {requested_step_for_manager}. Attempting to load latest as fallback...", lvl=1, color='yellow')
                     try:
                         self.nqs.load_weights(step=None, filename=ckpt_filename) # Try loading latest
                         loaded_step_info = self.nqs.ckpt_manager.latest_step if self.nqs.ckpt_manager.use_orbax else "latest (fallback)"
                         self._log(f"Successfully loaded latest weights (step {loaded_step_info}) after fallback.", lvl=1, color='green')
                         return self.stats
                     except Exception as e_fallback:
                         self._log(f"Critical: Failed to load latest weights after fallback: {e_fallback}", lvl=0, color='red')
                         raise e_fallback
                else:
                    self._log(f"Critical: Failed to load weights for step {requested_step_for_manager if requested_step_for_manager is not None else 'latest'}: {e}", lvl=0, color='red')
                    raise e
            
        return self.stats
    
# ------------------------------------------------------
#! EOF
# ------------------------------------------------------