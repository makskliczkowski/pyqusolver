"""
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
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jax
import numpy as np

# TQDM for progress bars
try:
    # If running in a notebook, use tqdm.notebook.trange for better display
    from IPython import get_ipython

    if get_ipython() is not None:
        from tqdm.notebook import trange
    else:
        from tqdm import trange
except ImportError:
    from tqdm import trange

# QES General Imports
try:
    from QES.general_python.common.flog import Logger
    from QES.general_python.common.hdf5man import HDF5Manager
    from QES.general_python.common.timer import timeit
    from QES.qes_globals import get_logger as get_global_logger
except ImportError:
    raise ImportError("QES general modules missing.")

#! NQS Imports
try:
    # solvers
    from QES.general_python.algebra.ode import IVP, choose_ode
    from QES.general_python.algebra.solvers import (
        SolverForm,
        SolverType,
        choose_precond,
        choose_solver,
    )

    # training phases
    from QES.general_python.ml.schedulers       import EarlyStopping, choose_scheduler
    from QES.general_python.ml.training_phases  import PhaseScheduler, create_phase_schedulers
    from QES.NQS.nqs                            import NQS, VMCSampler
    from QES.NQS.src.auto_tuner                 import AutoTunerConfig, TDVPAutoTuner
    from QES.NQS.src.tdvp                       import TDVP, TDVPLowerPenalty
except ImportError as e:
    raise ImportError("QES core modules missing.") from e

# ------------------------------------------------------


class NQSTimeModes(Enum):
    """Timing modes for profiling."""

    OFF = 0  # No timing
    BASIC = 1  # Basic timing (total time per epoch)
    FIRST = 2  # Time only the first epoch
    LAST = 4  # Time only the last epoch
    DETAILED = 8  # Detailed timing (per phase: sampling, step, update)


# ------------------------------------------------------


@dataclass
class NQSTrainTime:
    """Performance timers for profiling."""

    n_steps: int = 0
    step            : list = field(default_factory=list)
    sample          : list = field(default_factory=list)
    update          : list = field(default_factory=list)
    tdvp_prepare    : list = field(default_factory=list)
    tdvp_gradient   : list = field(default_factory=list)
    tdvp_covariance : list = field(default_factory=list)
    tdvp_x0         : list = field(default_factory=list)
    tdvp_solve      : list = field(default_factory=list)
    tdvp_phase      : list = field(default_factory=list)
    total           : list = field(default_factory=list)

    def reset(self):
        self.step, self.sample, self.update, self.total             = [], [], [], []
        self.tdvp_prepare, self.tdvp_gradient, self.tdvp_covariance = [], [], []
        self.tdvp_x0, self.tdvp_solve, self.tdvp_phase              = [], [], []
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

    history             : List[float] = field(default_factory=list)
    history_std         : List[float] = field(default_factory=list)
    lr_history          : List[float] = field(default_factory=list)
    diag_history        : List[float] = field(default_factory=list)
    sigma2_history      : List[float] = field(default_factory=list)
    rhat_history        : List[float] = field(default_factory=list)
    reg_history         : List[float] = field(default_factory=list)
    global_phase        : List[complex] = field(default_factory=list)
    timings             : NQSTrainTime = field(default_factory=NQSTrainTime)
    seed                : Optional[int] = None
    exact_predictions   : Optional[np.ndarray] = None  # e.g., ED eigenvalues
    exact_method        : Optional[str] = None  # e.g., 'lanczos', 'full_diag'

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for HDF5 serialization."""
        result = {  
            "history/val"               : self.history,
            "history/std"               : self.history_std,
            "history/lr"                : self.lr_history,
            "history/reg"               : self.reg_history,
            "history/diag_shift"        : self.diag_history,
            "history/sigma2"            : self.sigma2_history,
            "history/r_hat"             : self.rhat_history,
            "history/theta0"            : np.array(self.global_phase),
            "timings/n_steps"           : self.timings.n_steps,
            "timings/step"              : self.timings.step,
            "timings/sample"            : self.timings.sample,
            "timings/update"            : self.timings.update,
            "timings/tdvp/prepare"      : self.timings.tdvp_prepare,
            "timings/tdvp/gradient"     : self.timings.tdvp_gradient,
            "timings/tdvp/covariance"   : self.timings.tdvp_covariance,
            "timings/tdvp/x0"           : self.timings.tdvp_x0,
            "timings/tdvp/solve"        : self.timings.tdvp_solve,
            "timings/tdvp/phase"        : self.timings.tdvp_phase,
            "timings/total"             : self.timings.total,
        }
        if self.seed is not None:
            result["seed"] = self.seed
        if self.exact_predictions is not None:
            result["exact/predictions"] = np.asarray(self.exact_predictions)
        if self.exact_method is not None:
            result["exact/method"] = self.exact_method
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
        self.exact_predictions = np.atleast_1d(np.asarray(predictions))
        self.exact_method = method

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
    """
    Trainer for Neural Quantum States.

    Orchestrates the interaction between the Neural Network, the Physics Engine (TDVP),
    and the Optimization Schedule. Supports multiple scheduler configuration patterns.

    Features
    --------
    - **Phase Scheduling**: Multi-phase training with different LR/Reg per phase
    - **Global Phase Evolution**:   Tracking theta_0 for wavefunction analysis
    - **Checkpointing**:            Architecture-aware saving with metadata
    - **Dynamic Time-Step**:        ODE dt controlled via learning rate scheduler
    - **Optimizers**:               Support for JAX/Optax optimizers (Adam, SGD, etc.)
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
    """

    _ERR_INVALID_SCHEDULER      = "Invalid scheduler provided. Must be a PhaseScheduler or callable."
    _ERR_NO_NQS                 = "NQS instance must be provided."
    _ERR_INVALID_SOLVER         = "Invalid solver specified."
    _ERR_INVALID_TIMING_MODE    = "Invalid timing mode specified."
    _ERR_INVALID_LOWER_STATES   = "Lower states must be a list of NQS instances."

    def _log(
        self,
        message: str,
        lvl: int = 1,
        log: str = "info",
        color: str = "white",
        verbose: bool = True,
    ):
        """Helper for logging messages. In background mode, logs sparsely."""
        if not verbose:
            return

        # In background mode, check if we should log this step
        if self.background and hasattr(self, "_current_epoch"):
            # Log first epoch, every Nth epoch, and last epoch
            epoch       = self._current_epoch
            n_epochs    = getattr(self, "_total_epochs", 1)
            interval    = getattr(self, "background_log_interval", 20)
            should_log  = (epoch == 1) or (epoch % interval == 0) or (epoch == n_epochs)
            if not should_log:
                return

        if self.logger:
            self.logger.say(message, lvl=lvl, log=log, color=color)

    def __init__(
        self,
        nqs: NQS,
        *,
        # Solvers
        lin_solver          : Union[str, Callable] = SolverType.SCIPY_CG,  # Linear Solver
        lin_force_mat       : bool = False,                     # Force forming full matrix
        pre_solver          : Union[str, Callable] = None,      # Preconditioner
        ode_solver          : Union[IVP, str] = "Euler",        # ODE Solver or preset
        tdvp                : Optional[TDVP] = None,            # Setup TDVP engine
        # Configuration
        n_batch             : int = 256,                        # Batch size for sampling
        phases              : Union[str, tuple] = None,         # e.g., "kitaev" or (lr_sched, reg_sched)
        # Utilities
        timing_mode         : NQSTimeModes = NQSTimeModes.LAST, # Timing mode
        early_stopper       : Any = None,                       # Callable or EarlyStopping
        optimizer           : Optional[Any] = None,             # Optimizer instance or identifier
        logger              : Optional[Logger] = None,          # Logger instance
        lower_states        : List[NQS] = None,                 # For excited states - list of lower NQS
        background          : bool = False,                     # Quiet/background mode (no pbar/log spam)
        # --------------------------------------------------------------
        lr_scheduler        : Optional[Callable] = None,        # Direct LR scheduler injection
        reg_scheduler       : Optional[Callable] = None,        # Direct Reg scheduler injection (for future L2 reg)
        diag_scheduler      : Optional[Callable] = None,        # Direct diag_shift scheduler injection
        # --------------------------------------------------------------
        lr                  : Optional[float] = 7e-3,           # Direct LR injection       (bypass scheduler)
        reg                 : Optional[float] = None,           # Direct Reg injection      (bypass scheduler)
        diag_shift          : float = 5e-5,                     # Initial diagonal shift    (bypass scheduler)
        grad_clip           : Optional[float] = None,           # Gradient clipping threshold (None = no clipping)
        # --------------------------------------------------------------
        # Linear Solver + Preconditioner
        # --------------------------------------------------------------
        lin_sigma           : float = None,                     # Linear solver sigma, inferred from diag_shift if None
        lin_is_gram         : bool = True,                      # Is Gram matrix solver [(S^dagger S) x = b]
        lin_type            : SolverForm = SolverForm.GRAM,     # Solver form (gram, matvec, matrix)
        # --------------------------------------------------------------
        # Auto-Tuner
        # --------------------------------------------------------------
        auto_tune           : bool = False,                     # Enable adaptive auto-tuner
        auto_tuner          : Optional[Any] = None,             # Custom auto-tuner instance
        # --------------------------------------------------------------
        # TDVP arguments, if TDVP is created internally
        # --------------------------------------------------------------
        use_sr              : bool = True,                      # Whether to use SR
        use_minsr           : bool = False,                     # Whether to use MinSR
        rhs_prefactor       : float = -1.0,                     # RHS prefactor
        # --------------------------------------------------------------
        dtype               : Any = None,                       # Data type for internal computations
        **kwargs,
    ):
        r"""
        Initialize the NQS Trainer.

        Parameters
        ----------
        nqs : NQS
            Neural Quantum State instance to train.

        Scheduler Configuration (flexible, mix and match)
        -------------------------------------------------
        phases : Union[str, Tuple, None], default=None
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
        """

        if nqs is None:
            raise ValueError(self._ERR_NO_NQS)
        self.nqs = nqs  # Most important component
        self.logger = logger
        self.n_batch = n_batch
        self.background = background or bool(int(os.getenv("NQS_BACKGROUND", "0") or "0"))
        self.background_log_interval = kwargs.pop(
            "background_log_interval", 20
        )  # Log every Nth epoch in background mode
        self.verbose = nqs.verbose and not self.background
        self.dtype = dtype if dtype is not None else nqs.dtype
        if dtype is None and hasattr(nqs, "_precision_policy"):
            if getattr(nqs, "_iscpx", False):
                self.dtype = nqs._precision_policy.accum_complex_dtype
            else:
                self.dtype = nqs._precision_policy.accum_real_dtype

        # Validate lower states
        if lower_states is not None:
            if not isinstance(lower_states, list) or not all(
                isinstance(s, NQS) for s in lower_states
            ):
                raise ValueError(self._ERR_INVALID_LOWER_STATES)

        # Utilities
        self.lower_states = lower_states
        
        # Setup Early Stopping
        self.early_stopper = early_stopper
        if self.early_stopper is None:
            # Build explicit early-stopping kwargs to avoid colliding with
            # scheduler keys such as lr_patience/reg_patience/diag_patience.
            es_kwargs = {k: v for k, v in kwargs.items() if k.startswith(("es_", "early_"))}
            if "patience" in kwargs and "early_stopping_patience" not in es_kwargs:
                es_kwargs["early_stopping_patience"] = kwargs["patience"]
            if "min_delta" in kwargs and "early_stopping_min_delta" not in es_kwargs:
                es_kwargs["early_stopping_min_delta"] = kwargs["min_delta"]

            # Prefer unified factory when available, but keep compatibility with
            # older/stale EarlyStopping implementations that may not expose it.
            if hasattr(EarlyStopping, "from_kwargs"):
                self.early_stopper = EarlyStopping.from_kwargs(logger=self.logger, **es_kwargs)
            else:
                es_patience = es_kwargs.get(
                    "patience",
                    es_kwargs.get(
                        "early_stopping_patience",
                        es_kwargs.get("es_patience", es_kwargs.get("early_patience", 0)),
                    ),
                )
                es_min_delta = es_kwargs.get(
                    "min_delta",
                    es_kwargs.get(
                        "early_stopping_min_delta",
                        es_kwargs.get("es_min_delta", es_kwargs.get("early_min_delta", 1e-3)),
                    ),
                )
                self.early_stopper = EarlyStopping(
                    patience=es_patience, min_delta=es_min_delta, logger=self.logger
                )

        if isinstance(timing_mode, str):
            try:
                timing_mode = NQSTimeModes[timing_mode.upper()]
                self.timing_mode = timing_mode
                self._log(f"Timing mode set to: {self.timing_mode.name}", lvl=1, color="green")
            except KeyError:
                raise ValueError(self._ERR_INVALID_TIMING_MODE)
        self.timing_mode = (
            timing_mode if isinstance(timing_mode, NQSTimeModes) else NQSTimeModes.BASIC
        )

        # Setup Schedulers (The Integrated Part)
        self._init_reg, self._init_lr, self._init_diag = reg, lr, diag_shift

        # Prefer the Kitaev preset for Kitaev/Gamma models when caller left default preset.
        if isinstance(phases, str) and phases.lower() == "default":
            model_name = str(getattr(nqs, "model", getattr(nqs, "_model", ""))).lower()
            if "kitaev" in model_name or "gamma" in model_name:
                phases = "kitaev"
        
        # Determine max epochs for schedulers
        max_ep_val = kwargs.get("n_epochs", kwargs.get("epochs", 500))
        
        self._set_phases(
            phases,
            lr,
            reg,
            lr_scheduler=lr_scheduler,
            reg_scheduler=reg_scheduler,
            diag_scheduler=diag_scheduler,
            diag_shift=diag_shift,
            n_epochs=max_ep_val,
            **kwargs,
        )

        # Setup Linear Solver + Preconditioner
        try:
            self.lin_solver = choose_solver(
                solver_id=lin_solver,
                sigma=lin_sigma,
                is_gram=lin_is_gram,
                backend=nqs.backend_str,
                **kwargs,
            )
            self.pre_solver = (
                choose_precond(precond_id=pre_solver, **kwargs) if pre_solver else None
            )
        except Exception as e:
            raise ValueError(self._ERR_INVALID_SOLVER) from e

        # Setup TDVP (Physics Engine)
        self.tdvp = tdvp
        self.grad_clip = grad_clip  # Store for later use
        if self.tdvp is None:
            self._log(
                "No TDVP engine provided. Creating default TDVP instance.", lvl=0, color="yellow"
            )
            self.tdvp = TDVP(
                use_sr=use_sr,
                use_minsr=use_minsr,
                rhs_prefactor=rhs_prefactor,
                # SR Solver
                sr_diag_shift=diag_shift,
                sr_lin_solver=self.lin_solver,
                sr_lin_solver_t=lin_type,
                sr_precond=self.pre_solver,
                sr_form_matrix=lin_force_mat,
                sr_snr_tol=kwargs.get("sr_snr_tol", 1e-12),
                sr_lin_x0=kwargs.get("sr_lin_x0", None),
                sr_maxiter=kwargs.get("sr_maxiter", 1000),
                sr_pinv_cutoff=kwargs.get("sr_pinv_cutoff", 1e-8),
                # other
                backend=nqs.backend,
                logger=logger,
                use_timing=kwargs.get("use_timing", False),
                verbose=nqs.verbose,
                dtype=self.dtype,
            )
            self._log(
                f"Created default TDVP engine {self.tdvp}",
                lvl=1,
                color="yellow",
                verbose=self.verbose,
            )

        # Set gradient clipping on TDVP
        if grad_clip is not None:
            self.tdvp.set_grad_clip(grad_clip)
            self._log(
                f"Gradient clipping enabled: max_norm={grad_clip:.2e}",
                lvl=1,
                color="cyan",
                verbose=self.verbose,
            )

        # Setup ODE Solver
        try:
            self.ode_solver = ode_solver
            if isinstance(ode_solver, str):
                init_dt = self.lr_scheduler(0) if self.lr_scheduler else 1e-2
                self.ode_solver = choose_ode(ode_type=ode_solver, dt=init_dt, backend=nqs.backend)

        except Exception as e:
            raise ValueError(self._ERR_INVALID_SOLVER) from e

        # JIT Compile Critical Paths
        # We pre-compile the sampling and step functions to avoid runtime overhead
        self._single_step_jit = nqs.wrap_single_step_jax(batch_size=n_batch)

        # Define a function that runs the WHOLE step (ODE + TDVP + Energy)
        def train_step_logic(
            f, est_fn, y, t, configs, configs_ansatze, probabilities, lower_states, num_chains=1
        ):
            # We pass them down to the solver's **rhs_args
            new_params, new_t, (info, meta) = self.ode_solver.step(
                f=f,
                t=t,
                y=y,
                # These pass into **rhs_args of the solver:
                est_fn=est_fn,
                configs=configs,
                configs_ansatze=configs_ansatze,
                probabilities=probabilities,
                lower_states=lower_states,
                num_chains=num_chains,
            )
            # info is TDVPStepInfo, meta is (shapes, sizes, iscpx)
            return new_params, new_t, (info, meta)

        self._step_nojit = train_step_logic

        def train_step_logic_no_lower(f, est_fn, y, t, configs, configs_ansatze, probabilities, num_chains=1):
            return train_step_logic(f, est_fn, y, t, configs, configs_ansatze, probabilities, lower_states=None, num_chains=num_chains,)

        # JIT compile the training step for maximum performance (no excited-state penalties).
        # Mark 'f' (TDVP) and 'est_fn' (estimation function) as static.
        # The lower_states path stays non-jitted because it carries Python objects.
        if self.lower_states:
            self._step_jit = self._step_nojit
        else:
            self._step_jit = jax.jit(
                train_step_logic_no_lower, static_argnames=["f", "est_fn", "num_chains"]
            )

        # State
        self.stats = NQSTrainStats()
        self.best_energy = float("inf")

        # Setup Optimizer (Experimental)
        self.optimizer = optimizer
        self.opt_state = None
        if self.optimizer is not None and nqs._isjax:
            try:
                import optax
                # Handle string identifiers for common optimizers
                if isinstance(self.optimizer, str):
                    opt_id = self.optimizer.lower()
                    lr_val = self.lr_scheduler if self.lr_scheduler else self._init_lr
                    if opt_id == "adam":
                        self.optimizer = optax.adam(learning_rate=lr_val)
                    elif opt_id == "sgd":
                        self.optimizer = optax.sgd(learning_rate=lr_val)
                    elif opt_id == "adamw":
                        self.optimizer = optax.adamw(learning_rate=lr_val)
                    else:
                        raise ValueError(f"Unsupported optimizer string: {self.optimizer}")
                
                # Initialize optimizer state with flat parameters
                self.opt_state = self.optimizer.init(self.nqs.get_params(unravel=True))
                self._log(f"Initialized optimizer: {self.optimizer}", lvl=1, color="cyan")
            except ImportError:
                self._log("Optax not found. Optimizer disabled.", lvl=0, color="red")
                self.optimizer = None

        # Setup Auto-Tuner
        self.auto_tuner = auto_tuner
        if auto_tune and self.auto_tuner is None:
            self.auto_tuner = TDVPAutoTuner(logger=self.logger)
            self._log("TDVP Auto-Tuner enabled.", lvl=1, color="cyan", verbose=self.verbose)

    # ------------------------------------------------------
    #! Private Helpers
    # ------------------------------------------------------

    def _set_phases(
        self,
        phases,
        lr,
        reg,
        lr_scheduler=None,
        reg_scheduler=None,
        diag_scheduler=None,
        diag_shift=None,
        n_epochs=500,
        **kwargs,
    ):
        """
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
        """
        from QES.general_python.ml.schedulers import choose_scheduler

        def _normalize_scheduler_aliases(sched_kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize short aliases (e.g. ``lr_init``) to scheduler factory keys."""
            out = dict(sched_kwargs)
            if "lr_init" in out and "lr_initial" not in out:
                out["lr_initial"] = out.pop("lr_init")
            return out

        diag_kwargs = {k: v for k, v in kwargs.items() if k.startswith("diag_")}
        reg_kwargs = {k: v for k, v in kwargs.items() if k.startswith("reg_")}
        lr_kwargs = {k: v for k, v in kwargs.items() if k.startswith("lr_")}
        # transform keys to be accepted by scheduler factory

        diag_kwargs = {k.replace("diag_", "lr_"): v for k, v in diag_kwargs.items()}
        if self._init_diag is not None:
            diag_kwargs["lr_initial"] = self._init_diag
        diag_kwargs = _normalize_scheduler_aliases(diag_kwargs)

        reg_kwargs = {k.replace("reg_", "lr_"): v for k, v in reg_kwargs.items()}
        if self._init_reg is not None:
            reg_kwargs["lr_initial"] = self._init_reg
        reg_kwargs = _normalize_scheduler_aliases(reg_kwargs)

        lr_kwargs = {k.replace("lr_", "lr_"): v for k, v in lr_kwargs.items()}
        if self._init_lr is not None:
            lr_kwargs["lr_initial"] = self._init_lr
        lr_kwargs = _normalize_scheduler_aliases(lr_kwargs)

        # Helper to create scheduler from string or passthrough
        def _resolve_scheduler(sched, init_val, param_name, max_epochs, **kwargs):
            if sched is None:
                if init_val is not None:  # Constant scheduler from float value
                    return choose_scheduler(
                        "constant", initial_lr=init_val, max_epochs=max_epochs, logger=self.logger
                    )
                return None

            elif isinstance(sched, str):
                # String scheduler type -> create via factory
                init = init_val if init_val is not None else (1e-2 if param_name == "lr" else 1e-3)
                return choose_scheduler(
                    sched, initial_lr=init, max_epochs=max_epochs, logger=self.logger, **kwargs
                )

            elif callable(sched) or isinstance(sched, PhaseScheduler):
                return sched
            else:
                raise ValueError(f"Invalid {param_name}_scheduler type: {type(sched)}")

        # Check if user explicitly provided schedulers or lr/reg values
        # If so, these take precedence over the default 'phases' preset
        user_provided_scheduler = (
            lr_scheduler is not None
            or reg_scheduler is not None
            or lr is not None
            or reg is not None
        )

        # User provided lr_scheduler/reg_scheduler or lr/reg directly -> use them
        if user_provided_scheduler and (phases == "default" or phases is None):
            self.lr_scheduler = _resolve_scheduler(lr_scheduler, lr, "lr", n_epochs, **lr_kwargs)
            self.reg_scheduler = _resolve_scheduler(
                reg_scheduler, reg, "reg", n_epochs, **reg_kwargs
            )
            self.diag_scheduler = _resolve_scheduler(
                diag_scheduler, diag_shift, "diag", n_epochs, **diag_kwargs
            )

        # Preset string (e.g., 'default', 'kitaev') - only if no direct scheduler provided
        elif isinstance(phases, str):
            self._log(
                f"Initializing training phases with preset: '{phases}'",
                lvl=1,
                color="green",
                verbose=self.verbose,
            )
            self.lr_scheduler, self.reg_scheduler = create_phase_schedulers(phases, self.logger)
            # diag_scheduler is separate from phase presets
            self.diag_scheduler = _resolve_scheduler(
                diag_scheduler, diag_shift, "diag", n_epochs, **diag_kwargs
            )

        # Tuple of schedulers
        elif isinstance(phases, (tuple, list)) and len(phases) == 2:
            # Validate and resolve if strings
            self.lr_scheduler, self.reg_scheduler = phases
            self.lr_scheduler = _resolve_scheduler(
                self.lr_scheduler, lr, "lr", n_epochs, **lr_kwargs
            )
            self.reg_scheduler = _resolve_scheduler(
                self.reg_scheduler, reg, "reg", n_epochs, **reg_kwargs
            )
            self.diag_scheduler = _resolve_scheduler(
                diag_scheduler, diag_shift, "diag", n_epochs, **diag_kwargs
            )

        # No phases -> use injected schedulers or create from lr/reg
        else:
            self.lr_scheduler = _resolve_scheduler(lr_scheduler, lr, "lr", n_epochs, **lr_kwargs)
            self.reg_scheduler = _resolve_scheduler(
                reg_scheduler, reg, "reg", n_epochs, **reg_kwargs
            )
            self.diag_scheduler = _resolve_scheduler(
                diag_scheduler, diag_shift, "diag", n_epochs, **diag_kwargs
            )

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
        total_steps = kwargs.get("epoch", None)
        should_time = self.timing_mode == NQSTimeModes.DETAILED
        should_time |= self.timing_mode == NQSTimeModes.FIRST and self.stats.timings.n_steps == 0
        should_time |= (
            self.timing_mode == NQSTimeModes.LAST
            and total_steps is not None
            and self.stats.timings.n_steps == total_steps - 1
        )
        kwargs.pop("epoch", None)  # Remove epoch from kwargs to avoid passing it to fn

        if should_time:
            result, dt = timeit(fn, *args, **kwargs)
            timer_list = getattr(self.stats.timings, phase, None)
            if timer_list is not None:
                timer_list.append(dt)
            return result
        else:
            timer_list = getattr(self.stats.timings, phase, None)
            if timer_list is not None:
                timer_list.append(np.nan)  # Placeholder for no timing
            return fn(*args, **kwargs)

    def _record_tdvp_timings(self, tdvp_info: Any):
        """
        Persist TDVP internal timings in per-epoch arrays.
        """
        phase_map = (
            ("tdvp_prepare",    "prepare"),
            ("tdvp_gradient",   "gradient"),
            ("tdvp_covariance", "covariance"),
            ("tdvp_x0",         "x0"),
            ("tdvp_solve",      "solve"),
            ("tdvp_phase",      "phase"),
        )

        tdvp_timings = getattr(tdvp_info, "timings", None)
        tdvp_enabled = bool(getattr(self.tdvp, "use_timing", False))

        for stats_name, tdvp_name in phase_map:
            value = np.nan
            if tdvp_enabled and tdvp_timings is not None:
                raw = getattr(tdvp_timings, tdvp_name, None)
                if raw is not None:
                    try:
                        candidate = float(np.real(raw))
                        if np.isfinite(candidate):
                            value = candidate
                    except Exception:
                        value = np.nan
            getattr(self.stats.timings, stats_name).append(value)

    @staticmethod
    def _finite_mean(values: List[float]) -> float:
        if values is None or len(values) == 0:
            return float("nan")
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))

    def _timing_window(self, values: List[float], last_n: int) -> List[float]:
        if values is None or len(values) == 0:
            return []
        if last_n is None or last_n <= 0:
            return values
        return values[-int(last_n):]

    def _log_timing_breakdown(self, *, last_n: int = 20):
        """
        Log compact timing breakdown for the recent timed epochs.
        """
        st      = self.stats.timings
        samp    = self._finite_mean(self._timing_window(st.sample, last_n))
        step    = self._finite_mean(self._timing_window(st.step, last_n))
        upd     = self._finite_mean(self._timing_window(st.update, last_n))

        comp_total = 0.0
        for v in (samp, step, upd):
            if np.isfinite(v):
                comp_total += v

        if comp_total <= 0.0:
            return

        def _pct(value: float, base: float) -> str:
            if (not np.isfinite(value)) or base <= 0.0:
                return "nan"
            return f"{100.0 * value / base:.1f}%"

        # TDVP internals are a subset of "step" when TDVP timing is enabled.
        t_prepare   = self._finite_mean(self._timing_window(st.tdvp_prepare, last_n))
        t_grad      = self._finite_mean(self._timing_window(st.tdvp_gradient, last_n))
        t_cov       = self._finite_mean(self._timing_window(st.tdvp_covariance, last_n))
        t_x0        = self._finite_mean(self._timing_window(st.tdvp_x0, last_n))
        t_solve     = self._finite_mean(self._timing_window(st.tdvp_solve, last_n))
        t_phase     = self._finite_mean(self._timing_window(st.tdvp_phase, last_n))

        has_tdvp    = any(np.isfinite(v) for v in (t_prepare, t_grad, t_cov, t_x0, t_solve, t_phase))

        msg = (
            f"Timing breakdown (last {last_n} timed epochs): "
            f"sample={samp:.3f}s ({_pct(samp, comp_total)}), "
            f"step={step:.3f}s ({_pct(step, comp_total)}), "
            f"update={upd:.3f}s ({_pct(upd, comp_total)})"
        )
        if has_tdvp:
            msg += (
                f"; TDVP⊂step: prep={t_prepare:.3f}s ({_pct(t_prepare, step)}), "
                f"grad={t_grad:.3f}s ({_pct(t_grad, step)}), "
                f"cov={t_cov:.3f}s ({_pct(t_cov, step)}), "
                f"x0={t_x0:.3f}s ({_pct(t_x0, step)}), "
                f"solve={t_solve:.3f}s ({_pct(t_solve, step)}), "
                f"phase={t_phase:.3f}s ({_pct(t_phase, step)})"
            )
        self._log(msg, lvl=1, color="cyan", verbose=True)

    # ------------------------------------------------------
    #! Hyperparameter Updates
    # ------------------------------------------------------

    def _update_hyperparameters(self, epoch: int, last_loss: float):
        """Syncs schedulers with solvers."""

        # 1. Learning Rate -> ODE Time Step
        if self.lr_scheduler:
            new_lr = self.lr_scheduler(epoch, last_loss)
            self.ode_solver.set_dt(float(new_lr))
            self.stats.lr_history.append(new_lr)
            current_dt = new_lr
        else:
            raw_dt = getattr(self.ode_solver, "dt", 1e-3)
            current_dt = raw_dt() if callable(raw_dt) else raw_dt

        # 2. Diagonal Shift (SR regularization) -> TDVP
        if self.diag_scheduler:
            new_diag = self.diag_scheduler(epoch, last_loss)
            self.tdvp.set_diag_shift(float(new_diag))
            self.stats.diag_history.append(new_diag)
            current_diag = new_diag
        else:
            current_diag = self.tdvp.sr_diag_shift

        # 3. Other Regularization
        if self.reg_scheduler:
            new_reg = self.reg_scheduler(epoch, last_loss)
            self.tdvp.set_regularization(float(new_reg))
            self.stats.reg_history.append(new_reg)
            current_reg = new_reg
        else:
            current_reg = self.tdvp.regularization

        return current_dt, current_reg, current_diag

    def _prepare_lower_states(self, cfgs_current, excited_on_excited):
        """Helper to prepare penalty terms for excited states."""

        if not self.lower_states:
            return None

        lower_contr = []
        for nqs_lower in self.lower_states:

            # Sample from the lower state
            (_, _), (cfgs_lower, cfgs_psi_lower), _ = nqs_lower.sample()

            if excited_on_excited is None:
                excited_on_excited = self.nqs.ansatz(cfgs_current)

            # Calculate cross-ratios
            # Note: This requires O(N_samples) network evaluations
            penalty = TDVPLowerPenalty(
                excited_on_lower=self.nqs.ansatz(cfgs_lower),
                lower_on_excited=nqs_lower.ansatz(cfgs_current),
                excited_on_excited=excited_on_excited,  #! check if correct
                lower_on_lower=cfgs_psi_lower.flatten(),
                params_j=nqs_lower.get_params(),
                configs_j=cfgs_lower,
                beta_j=nqs_lower.beta_penalty,
                backend_np=self.nqs.backend,
                dtype=self.tdvp.dtype,
            )
            lower_contr.append(penalty)
        return lower_contr

    # ------------------------------------------------------
    #! Main Training Loop
    # ------------------------------------------------------

    def _train_set_exact_info(
        self, exact_predictions: Union[float, List[float], np.ndarray], exact_method: str = None
    ):
        """Sets exact prediction info on the NQS object."""
        if exact_predictions is not None:
            self.nqs.exact = {
                "exact_predictions": exact_predictions,
                "exact_method": exact_method,
                # Choose appropriate exact energy for this state
                "exact_energy": (
                    float(exact_predictions)
                    if np.ndim(exact_predictions) == 0
                    else (
                        exact_predictions[len(self.lower_states)]
                        if self.lower_states
                        else exact_predictions[0]
                    )
                ),
            }

    def _train_update_pbar(
        self,
        pbar,
        epoch: int,
        global_epoch: int,
        n_epochs: int,
        stats: NQSTrainStats,
        mean_loss: float,
        acc_ratio: float,
        lr: float,
        reg: float,
        diag: float,
        sigma2: Optional[float],
        r_hat: Optional[float],
        t0: float,
        exact_loss: float = None,
    ):
        """Updates the progress bar with current stats."""

        def _as_finite_float(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            try:
                val = float(np.real(value))
            except Exception:
                return None
            if not np.isfinite(val):
                return None
            return val

        def _fmt(value: Optional[float], spec: str, default: str = "nan") -> str:
            val = _as_finite_float(value)
            if val is None:
                return default
            return format(val, spec)

        # Calculate total time for this epoch
        t_sample = self.stats.timings.sample[-1]
        t_update = self.stats.timings.update[-1]
        t_step = self.stats.timings.step[-1]
        acc_ratio = self.nqs.sampler.accepted_ratio

        if pbar is not None:
            postfix = {
                "epoch": f"G:{global_epoch},L:{epoch}",
                "loss": f"{mean_loss:.4f}",
                "acc": f"{acc_ratio:.2%}",
            }

            if exact_loss is not None:
                postfix["error"] = f"{np.abs(exact_loss - mean_loss):.4e}"

            # Get last timings (if available)
            if np.real(self.tdvp.rhs_prefactor) != 0:
                postfix["lr"] = f"{lr:.2e}"
            else:
                postfix["dt"] = f"{lr:.2e}"

            if reg is not None and reg > 0:
                postfix["reg"] = f"{reg:.1e}"
            if diag is not None and diag > 0:
                postfix["diag"] = f"{diag:.1e}"
            sigma2_val = _as_finite_float(sigma2)
            rhat_val = _as_finite_float(r_hat)
            if sigma2_val is not None:
                postfix["sigma2"] = f"{sigma2_val:.2e}"
            if rhat_val is not None:
                postfix["R"] = f"{rhat_val:.3f}"

            # Only add detailed timings to the bar if they are real numbers
            if not np.isnan(t_step):
                postfix["t_step"] = f"{t_step:.2f}s"
                postfix["t_samp"] = f"{t_sample:.2f}s"
                postfix["t_upd"] = f"{t_update:.2f}s"

            if self.timing_mode.value <= NQSTimeModes.DETAILED.value:
                postfix["t_epoch"] = f"{(time.time() - t0) / (epoch + 1):.2f}s"

            pbar.set_postfix(postfix)

        else:
            lr_str = _fmt(lr, ".1e")
            diag_str = _fmt(diag, ".1e")
            sigma2_str = _fmt(sigma2, ".2e")
            rhat_str = _fmt(r_hat, ".3f")
            if exact_loss is not None:
                self._log(
                    f"Epoch G:{global_epoch},L:{epoch}: loss={mean_loss:.4f} (exact={exact_loss:.4f}, d={mean_loss - exact_loss:.4f}), lr={lr_str}, diag={diag_str}, sigma2={sigma2_str}, R={rhat_str}, acc={acc_ratio:.2%}, t_step={t_step:.2f}s, t_samp={t_sample:.2f}s, t_upd={t_update:.2f}s"
                )
            else:
                self._log(
                    f"Epoch G:{global_epoch},L:{epoch}: loss={mean_loss:.4f}, lr={lr_str}, diag={diag_str}, sigma2={sigma2_str}, R={rhat_str}, acc={acc_ratio:.2%}, t_step={t_step:.2f}s, t_samp={t_sample:.2f}s, t_upd={t_update:.2f}s"
                )

    def train(
        self,
        n_epochs            : int = None,
        checkpoint_every    : int = 50,
        *,
        save_path           : str = None,
        reset_stats         : bool = True,
        use_pbar            : bool = True,
        **kwargs,
    ) -> NQSTrainStats:
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
        checkpoint_every                = max(1, checkpoint_every)
        # Best-checkpoint policy (Orbax best tag reuses a fixed step key).
        # Throttle frequency by default to avoid repeated expensive overwrites.
        save_best_checkpoint            = bool(kwargs.pop("save_best_checkpoint", True))
        best_checkpoint_every           = max(1, int(kwargs.pop("best_checkpoint_every", checkpoint_every)))
        best_checkpoint_min_delta       = float(kwargs.pop("best_checkpoint_min_delta", 0.0))
        save_best_stats                 = bool(kwargs.pop("save_best_stats", False))
        last_best_save_global_epoch     = -best_checkpoint_every
        timing_report_every             = int(kwargs.pop("timing_report_every", 0) or 0)
        timing_report_last_n            = int(kwargs.pop("timing_report_last_n", 20) or 20)
        timing_report_final             = bool(kwargs.pop("timing_report_final", True))

        background_flag = (self.background or bool(int(os.getenv("NQS_BACKGROUND", "0") or "0")) or bool(kwargs.pop("background", False)))
        use_pbar        = use_pbar and not background_flag

        # Auto-detect epochs from scheduler if not provided
        if n_epochs is None and isinstance(self.lr_scheduler, PhaseScheduler):
            n_epochs = sum(p.epochs for p in self.lr_scheduler.phases)
            self._log(f"Auto-detected total epochs from phases: {n_epochs}", lvl=1, color="green", verbose=self.verbose,)

        # Set total epochs for sparse logging in background mode
        self._total_epochs = n_epochs or 100
        # Set exact info if provided
        num_samples         = kwargs.get("num_samples", None)
        num_chains          = kwargs.get("num_chains", None)
        exact_predictions   = kwargs.get("exact_predictions", None)
        exact_method        = kwargs.get("exact_method", None)
        self._train_set_exact_info(exact_predictions, exact_method)

        # Reset stats if needed, if not we continue accumulating
        if reset_stats:
            self.stats          = NQSTrainStats()
            self.best_energy    = float("inf")
            start_epoch         = 0
        else:
            start_epoch = len(self.stats.history)
            if start_epoch > 0:
                self._log(f"Continuing training from epoch {start_epoch} (reset_stats=False)", lvl=1, color="cyan", verbose=self.verbose,)

        t0          = ((time.time() + self.stats.timings.total[-1]) if len(self.stats.timings.total) > 0 else time.time())
        n_epochs    = n_epochs or 100
        pbar        = trange(n_epochs, desc="NQS Training", leave=True) if use_pbar else range(n_epochs)

        try:
            for epoch in pbar:
                # Global epoch accounts for previous training when continuing
                global_epoch        = start_epoch + epoch
                self._current_epoch = global_epoch  # Track for sparse logging in background mode

                # Scheduling (use global_epoch so schedulers continue properly)
                last_E          = self.stats.history[-1] if len(self.stats.history) > 0 else 0.0
                lr, reg, diag   = self._update_hyperparameters(global_epoch, last_E)  # use global epoch so resumed training continues schedules

                # Sampling (Timed)
                # Returns: ((keys), (configs, ansatz_vals), probs)
                # Note: reset=(epoch==0) resets sampler only at start of this train() call
                # If auto-tuner active, it might request more samples
                current_num_samples = num_samples
                if self.auto_tuner:
                    current_num_samples = self.auto_tuner.n_samples

                sample_out = self._timed_execute(
                    "sample",
                    self.nqs.sample,
                    reset=(global_epoch == 0),
                    epoch=global_epoch,
                    num_samples=current_num_samples,
                    num_chains=num_chains,
                )
                (_, _), (cfgs, cfgs_psi), probs = sample_out
                if epoch == 0:
                    self._log(
                        f"Sampled {cfgs.shape[0]} configurations with batch size {self.n_batch}",
                        lvl=1,
                        color="green",
                        verbose=self.verbose,
                    )

                # Handle Excited States (Penalty Terms)
                lower_contr = self._prepare_lower_states(cfgs, cfgs_psi)
                if epoch == 0:
                    self._log(
                        "Prepared lower states for excited state handling",
                        lvl=1,
                        color="green",
                        verbose=self.verbose,
                    )
                # Physics Step (TDVP / ODE) (Timed)
                params_flat = self.nqs.get_params(unravel=True)
                step_kwargs = {
                    "f": self.tdvp,
                    "y": params_flat,
                    "t": 0.0,
                    "est_fn": self._single_step_jit,
                    "configs": cfgs,
                    "configs_ansatze": cfgs_psi,
                    "probabilities": probs,
                    "num_chains": int(num_chains if num_chains is not None else self.nqs.sampler.numchains),
                }
                step_fn = self._step_jit
                if lower_contr is not None:
                    step_fn = self._step_nojit
                    step_kwargs["lower_states"] = lower_contr

                step_out = self._timed_execute(
                    "step",
                    step_fn,
                    epoch=global_epoch,
                    **step_kwargs,
                )
                dparams, _, (tdvp_info, shapes_info)    = step_out
                mean_loss                               = tdvp_info.mean_energy
                std_loss                                = tdvp_info.std_energy
                sigma2                                  = tdvp_info.sigma2
                if sigma2 is None:
                    sigma2 = self.nqs.backend.real(std_loss) ** 2
                r_hat = tdvp_info.r_hat
                self._record_tdvp_timings(tdvp_info)

                # Auto-Tuner: Update Logic
                accepted = True
                if self.auto_tuner:
                    diagnostics                     = self.nqs.sampler.diagnose(cfgs_psi)
                    new_params, accepted, warnings  = self.auto_tuner.update(tdvp_info, diagnostics, np.real(mean_loss))

                    if warnings:
                        for w in warnings:
                            self._log(f"AutoTuner: {w}", lvl=2, color="yellow", verbose=self.verbose)

                    # Apply new parameters
                    # 1. LR -> ODE dt
                    self.ode_solver.set_dt(new_params["lr"])
                    # 2. Diag Shift -> TDVP
                    self.tdvp.set_diag_shift(new_params["diag_shift"])
                    # 3. N Samples -> Next iteration
                    # (Handled at start of loop)

                    if not accepted:
                        self._log("AutoTuner rejected step. Skipping parameter update.", lvl=1, color="red", verbose=self.verbose,)

                # Update Weights (Timed) - Only if accepted
                if accepted:
                    if self.optimizer is not None and self.nqs._isjax:
                        # Use optax update logic
                        import optax
                        
                        # Direction from TDVP (dot_theta)
                        # ODE solver returns new parameters dparams. 
                        # direction = (new - old) / dt
                        params_flat = self.nqs.get_params(unravel=True)
                        dt_step     = float(lr)
                        direction   = (dparams - params_flat) / dt_step if dt_step > 1e-12 else (dparams - params_flat)
                        
                        # Direction is -grad for standard gradient descent
                        # optax expects gradients to subtract (update = -lr * grad)
                        # So we pass -direction as the gradient
                        updates, self.opt_state = self.optimizer.update(-direction, self.opt_state, params_flat)
                        new_params_flat         = optax.apply_updates(params_flat, updates)
                        
                        self._timed_execute(
                            "update",
                            self.nqs.set_params,
                            new_params_flat,
                            shapes=shapes_info[0],
                            sizes=shapes_info[1],
                            iscpx=shapes_info[2],
                            epoch=global_epoch,
                        )
                    else:
                        self._timed_execute(
                            "update",
                            self.nqs.set_params,
                            dparams,
                            shapes=shapes_info[0],
                            sizes=shapes_info[1],
                            iscpx=shapes_info[2],
                            epoch=global_epoch,
                        )

                # Global Phase Integration
                # Extract theta0_dot from JIT output and update TDVP state
                if tdvp_info.theta0_dot is not None:
                    self.tdvp._theta0_dot = tdvp_info.theta0_dot
                    self.tdvp.update_global_phase(dt=lr)

                # Record global phase
                self.stats.global_phase.append(self.tdvp.global_phase)

                # Logging & Storage
                mean_loss = np.real(mean_loss)
                self.stats.timings.n_steps += 1
                self.stats.history.append(mean_loss)
                self.stats.history_std.append(np.real(std_loss))
                self.stats.sigma2_history.append(float(np.real(sigma2)))
                self.stats.rhat_history.append(float(np.real(r_hat)) if r_hat is not None else float("nan"))
                self.stats.timings.total.append(time.time() - t0)

                # Track best
                if mean_loss < (self.best_energy - best_checkpoint_min_delta):
                    self.best_energy = mean_loss
                    if save_best_checkpoint and (global_epoch - last_best_save_global_epoch >= best_checkpoint_every):
                        self.save_checkpoint(
                            "best",
                            save_path,
                            fmt="h5",
                            overwrite=True,
                            save_stats=save_best_stats,
                            **kwargs,
                        )
                        last_best_save_global_epoch = global_epoch

                self._train_update_pbar(
                    pbar if use_pbar else None,
                    epoch,
                    global_epoch,
                    n_epochs,
                    self.stats,
                    mean_loss,
                    self.nqs.sampler.accepted_ratio,
                    lr,
                    reg,
                    diag,
                    float(np.real(sigma2)),
                    float(np.real(r_hat)) if r_hat is not None else float("nan"),
                    t0,
                    exact_loss=(self.nqs.exact["exact_energy"] if self.nqs.exact is not None else None),
                )

                # Checkpointing (use global_epoch for consistent checkpoint naming)
                if epoch % checkpoint_every == 0 and epoch > 0:
                    self.save_checkpoint(
                        global_epoch, save_path, fmt="h5", overwrite=True, **kwargs
                    )

                # Early Stopping
                if np.isnan(mean_loss):
                    self._log("Energy is NaN. Stopping training.", lvl=0, color="red")
                    break

                if self.early_stopper and self.early_stopper(mean_loss):
                    self._log(
                        f"Early stopping triggered at epoch G:{global_epoch},L:{epoch}", lvl=0, color="yellow",)
                    break

                if timing_report_every > 0 and ((epoch + 1) % timing_report_every == 0):
                    self._log_timing_breakdown(last_n=timing_report_last_n)

        except KeyboardInterrupt:
            self._log("Training interrupted by user.", lvl=0, color="red")
        except StopIteration as e:
            self._log(f"Training stopped: {e}", lvl=0, color="yellow")
        except Exception as e:
            self._log(f"An error occurred during training: {e}", lvl=0, color="red")
            if use_pbar:
                pbar.close()
            raise e

        # Finalize
        if use_pbar:
            pbar.close()
        self.stats.history          = np.array(self.stats.history).flatten().tolist()
        self.stats.history_std      = np.array(self.stats.history_std).flatten().tolist()
        if timing_report_final:
            self._log_timing_breakdown(last_n=timing_report_last_n)
            
        self.save_checkpoint("final", save_path, fmt="h5", overwrite=True, verbose=True, **kwargs)
        self._log(f"Training completed in {time.time() - t0:.2f} seconds. Total epochs: {len(self.stats.history)} (this run: {n_epochs}, started at: {start_epoch}).", lvl=0, color="green",)
        return self.stats

    # ------------------------------------------------------
    #! Checkpointing
    # ------------------------------------------------------

    def save_checkpoint(
        self,
        step: Union[int, str],
        path: Union[str, Path] = None,
        *,
        fmt: str = "h5",
        overwrite: bool = True,
        save_stats: bool = True,
        **kwargs,
    ) -> Union[Path, None]:
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
        save_stats : bool
            Whether to save/update ``stats.h5`` together with weights.
            For frequent tag checkpoints (e.g. ``best``), disabling this can
            reduce I/O overhead.

        Returns:
        --------
        Path : The path where the checkpoint was saved.
        """
        import os

        net = self.nqs.net
        path = str(path) if path is not None else str(self.nqs.defdir)
        seed = getattr(self.nqs, "_net_seed", getattr(self.nqs, "_seed", None))

        # Update stats with seed before saving
        self.stats.seed = seed
        self.stats.exact_predictions = kwargs.get("exact_predictions", None)

        # Build comprehensive metadata
        meta = {
            "network_class": net.__class__.__name__,
            "step": step,
            "seed": seed,
            "last_loss": float(self.stats.history[-1]) if self.stats.history else 0.0,
            "n_epochs": len(self.stats.history),
        }

        # Resolve path for stats file
        if path.endswith(os.sep) or os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            final_path = os.path.join(path, f"checkpoint_{step}.{fmt}")
            final_path_stats = os.path.join(path, "stats.h5")
        else:
            # Path is a specific file
            final_path = path
            final_path_stats = os.path.join(str(Path(path).parent), "stats.h5")
            os.makedirs(str(Path(path).parent), exist_ok=True)

        if not overwrite and os.path.exists(final_path):
            self._log(
                f"Checkpoint file {final_path} already exists and overwrite is False. Skipping save.",
                lvl=1,
                color="yellow",
            )
            return None

        # Save training history/stats
        if save_stats:
            HDF5Manager.save_hdf5(
                directory=os.path.dirname(final_path_stats),
                filename=os.path.basename(final_path_stats),
                data=self.stats.to_dict(),
            )
            self._log(
                f"Saved training stats to {final_path_stats}",
                lvl=2,
                verbose=kwargs.get("verbose", False),
            )

        # Delegate weight saving to NQS (which uses checkpoint manager)
        return self.nqs.save_weights(
            filename=final_path,
            step=step,
            metadata=meta,
        )

    def load_checkpoint(
        self,
        step: Optional[Union[int, str]] = None,
        path: Optional[Union[str, Path]] = None,
        *,
        fmt: str = "h5",
        load_weights: bool = True,
        fallback_latest: bool = True,
    ) -> NQSTrainStats:
        """
        Loads training history from checkpoint.

        Parameters:
        -----------
        step : Optional[Union[int, str]]
            Training step or epoch to load. Can be:
            - int:
                specific step number
            - 'latest', 'best', 'final':
                load tagged checkpoint
            - None:
                same as 'latest'
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
        >>> stats = trainer.load_checkpoint(step='best')     # Load best energy so far
        """
        import os
        import re

        # Resolve the search directory
        if path is None:
            search_dir = self.nqs.ckpt_manager.directory
            path_str = str(self.nqs.defdir)  # Fallback for stats file
        else:
            path_obj = Path(path)
            if path_obj.is_file():
                search_dir = path_obj.parent
                path_str = str(path)
            else:
                search_dir = path_obj
                path_str = str(path)

        # Determine step identifier
        requested_step_for_manager = step
        if step is not None and isinstance(step, str) and step.lower() == "latest":
            requested_step_for_manager = None # Manager handles latest if None

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
            stats_data = HDF5Manager.read_hdf5(file_path=Path(stats_file))
            self.stats = NQSTrainStats(
                        history             =list(stats_data.get("/history/val", [])),
                        history_std         =list(stats_data.get("/history/std", [])),
                        lr_history          =list(stats_data.get("/history/lr", [])),
                        reg_history         =list(stats_data.get("/history/reg", [])),
                        diag_history        =list(stats_data.get("/history/diag_shift", [])),
                        sigma2_history      =list(stats_data.get("/history/sigma2", [])),
                        rhat_history        =list(stats_data.get("/history/r_hat", [])),
                        global_phase        =list(stats_data.get("/history/theta0", [])),
                        seed                =stats_data.get("/seed", None),
                        exact_predictions   =stats_data.get("/exact/predictions", None),
                        exact_method        =stats_data.get("/exact/method", None),
                        timings             =NQSTrainTime(
                                                n_steps             =stats_data.get("/timings/n_steps", 0),
                                                step                =list(stats_data.get("/timings/step", [])),
                                                sample              =list(stats_data.get("/timings/sample", [])),
                                                update              =list(stats_data.get("/timings/update", [])),
                                                tdvp_prepare        =list(stats_data.get("/timings/tdvp/prepare", [])),
                                                tdvp_gradient       =list(stats_data.get("/timings/tdvp/gradient", [])),
                                                tdvp_covariance     =list(stats_data.get("/timings/tdvp/covariance", [])),
                                                tdvp_x0             =list(stats_data.get("/timings/tdvp/x0", [])),
                                                tdvp_solve          =list(stats_data.get("/timings/tdvp/solve", [])),
                                                tdvp_phase          =list(stats_data.get("/timings/tdvp/phase", [])),
                                                total               =list(stats_data.get("/timings/total", [])),
                                            ),
            )
            if self.stats.history:
                self.best_energy = min(self.stats.history)
            self._log(f"Loaded stats from {stats_file}", lvl=1, color="green")
        except Exception as e:
            self._log(f"Stats load failed ({e}). Starting fresh stats.", lvl=1, color="yellow")

        # Load Weights Logic
        if load_weights:
            try:
                ckpt_filename = None

                # If path is to a specific .h5 file, use it directly as filename
                if path is not None and str(path).endswith(".h5"):
                    ckpt_filename = str(path)

                # NQS.load_weights will pass this to Manager.
                self.nqs.load_weights(step=requested_step_for_manager, filename=ckpt_filename)

                # After successful load, resolve what step was actually loaded
                loaded_step_info = requested_step_for_manager or "latest"
                if requested_step_for_manager is None and self.nqs.ckpt_manager.use_orbax:
                    loaded_step_info = self.nqs.ckpt_manager.latest_step

                self._log(
                    f"Successfully loaded weights for step {loaded_step_info}", lvl=1, color="green"
                )
                return self.stats

            except Exception as e:
                self._log(f"Loading weights failed: {e}", lvl=0, color="red")

                # Handle fallback_latest
                if requested_step_for_manager is not None and fallback_latest:
                    self._log(
                        f"Failed to load specific step {requested_step_for_manager}. Attempting to load latest as fallback...",
                        lvl=1,
                        color="yellow",
                    )
                    try:
                        self.nqs.load_weights(step=None, filename=ckpt_filename)
                        return self.stats
                    except Exception as e_fallback:
                        self._log(f"Critical: Failed to load fallback: {e_fallback}", lvl=0, color="red")
                        raise e_fallback
                else:
                    raise e

        return self.stats


# ------------------------------------------------------
#! EOF
# ------------------------------------------------------
