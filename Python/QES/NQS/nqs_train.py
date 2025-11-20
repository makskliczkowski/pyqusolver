'''
NQS training class. It implements the training loop, learning rate scheduling, regularization scheduling, and early stopping.
It works with JAX backend and is compatible with Flax networks.

Author: Maksymilian Kliczkowski
'''

import os
import json
import numpy as np
from contextlib import contextmanager
from tqdm import trange
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

try:
    # from QES.general_python.common.plot import Plotter, colorsCycle, linestylesCycle
    from QES.general_python.common.flog import Logger
    from QES.general_python.common.timer import timeit
except ImportError as e:
    raise ImportError("Could not import general_python modules. Make sure QES is installed and the PYTHONPATH is set correctly.") from e

#! NQS
try:
    from QES.NQS.nqs import NQS
    from QES.NQS.tdvp import TDVP, TDVPLowerPenalty
    from QES.general_python.algebra.ode import IVP
    from QES.general_python.ml.schedulers import Parameters, EarlyStopping
except ImportError as e:
    raise ImportError("Could not import NQS modules. Make sure QES is installed and the PYTHONPATH is set correctly.") from e

# ------------------------------------------------------

@dataclass
class NQSTrainTime:
    n_steps         : int   = None
    step            : list  = field(default_factory=list)
    sample          : list  = field(default_factory=list)
    gradient        : list  = field(default_factory=list)
    update          : list  = field(default_factory=list)
    prepare         : list  = field(default_factory=list)
    solve           : list  = field(default_factory=list)
    lower_states    : list  = field(default_factory=list) # time for lower states sampling
    total_times     : list  = field(default_factory=list)
    total_time      : float = None  # Added total_time attribute
    
    def todict(self):
        return {
            "step"              : self.step,
            "sample"            : self.sample,
            "gradient"          : self.gradient,
            "update"            : self.update,
            "prepare"           : self.prepare,
            "solve"             : self.solve
        }
    
    # --------------------------------------------------
    
    def __post_init__(self):
        self.total_times    = np.sum([self.step, self.sample, self.gradient, self.update, self.prepare, self.solve], axis=0)
        self.total_time     = np.sum(self.total_times)

    def __setitem__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            raise KeyError(f"Key {key} not found in NQSTrainTime.")
    
    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"Key {key} not found in NQSTrainTime.")
    
    # --------------------------------------------------
    
    def items(self):
        return self.__dict__.items()
    
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()
    
    def reset(self, n_steps: int = 0):
        self.n_steps         = n_steps
        self.total_time      = None
        for key in self.__dict__.keys():
            if key != "n_steps" and key != "total_time":
                self.__dict__[key] = []

# ------------------------------------------------------

@dataclass
class NQSTrainStats:
    '''
    Dataclass to store training statistics.
    '''
    # ------------------------------------------------------
    nqs             : 'NQS'               =   None
    # ------------------------------------------------------
    nqs_train       : 'NQSTrainer'        =   None
    # visible nodes
    n_visible       : int                 =   None
    n_params        : int                 =   None
    n_samples       : int                 =   None
    # ------------------------------------------------------
    # history of energies and their std
    history         : np.ndarray          =   None
    history_std     : np.ndarray          =   None
    # timing breakdown of different parts of the algorithm
    timings         : NQSTrainTime        =   None
    # ------------------------------------------------------
    # history of learning rates and regularizers
    lr_history      : list                =   None
    reg_history     : list                =   None
    # mean and std of the last n_for_mean points
    last_mean       : float               =   None
    last_std        : float               =   None
    last_time       : float               =   None
    last_n_time     : int                 =   15
    total_times     : np.ndarray          =   None
    # ------------------------------------------------------
    true_loss       : float               =   None # if known, e.g. from ED
    true_error      : float               =   None # if loss is known, the error from it
    true_rel_error  : float               =   None # relative error if loss is known
    
    def __post_init__(self):
        self.calculate_stats(n_for_mean=self.last_n_time)
    
    def calculate_stats(self, n_for_mean: int = 15):
        ''' Calculate mean and std of the last n_for_mean points '''
        if self.history is None or self.history_std is None or self.nqs is None:
            raise ValueError("Missing history or nqs for calculating stats.")
        if self.timings is None:
            raise ValueError("Missing timings for calculating stats.")
        if n_for_mean <= 0:
            raise ValueError("n_for_mean must be positive.")
        if n_for_mean > len(self.history):
            raise ValueError("n_for_mean must be less than or equal to the length of history.")
        
        # visible nodes
        self.n_visible      = self.nqs.nvis
        self.n_params       = self.nqs.npar
        
        # convert to numpy arrays
        self.timings        = {k: np.array(v) for k, v in self.timings.items()}
        self.history        = np.array(self.history)
        self.history_std    = np.array(self.history_std)

        # calculate mean and std of the last n_for_mean points
        energies            = self.history[~np.isnan(self.history)][:-2] / self.nqs.size
        self.last_mean      = np.nanmean(energies[-n_for_mean:])
        self.last_std       = np.nanstd(energies[-n_for_mean:])
        # self.total_times    = np.sum([self.timings[k] for k in self.timings.keys()], axis=0) / self.nqs.size
        # self.last_time      = np.nanmean(self.total_times[-n_for_mean:]) / self.nqs.size
        self.lr_history     = np.array(self.nqs_train.lr_scheduler.history)
        self.reg_history    = np.array(self.nqs_train.reg_scheduler.history)
        self.last_n_time    = n_for_mean
        
        # if true loss is known, calculate the error
        if self.true_loss is not None:
            self.true_error = np.abs(self.last_mean - self.true_loss)
            self.true_rel_error = self.true_error / np.abs(self.true_loss) if self.true_loss != 0 else 0

# ------------------------------------------------------

class NQSTrainer:
    '''
    Trainer class for Neural Quantum States (NQS).
    Implements training loop, learning rate scheduling, regularization scheduling, and early stopping.
    '''
    
    def __init__( self,
                nqs             : NQS,
                ode_solver      : IVP,
                tdvp            : TDVP,
                n_batch         : int,
                lr_scheduler    : Parameters,
                reg_scheduler   : Parameters,
                early_stopper   : EarlyStopping,
                logger          : Logger,
                *args,
                lower_states    : List[NQS] = None, # list of NQS for lower states in excited state calculations
                **kwargs
            ):
        #! set the objects
        self.nqs                = nqs
        self.tdvp               = tdvp
        self.ode_solver         = ode_solver
        
        # schedulers and early stopping
        self.lr_scheduler       = lr_scheduler
        self.reg_scheduler      = reg_scheduler
        self.early_stopper      = early_stopper
        
        # logger
        self.logger             = logger

        #? should I jit compile the ansatz - it is compiled but we can move it till the very end
        self.ansatz             = nqs.ansatz
        self.loc_energy         = nqs.local_energy
        self.flat_grad          = nqs.flat_grad
        
        #! JIT-compiled single step (y, t, *, configs, configs_ansatze, probabilities, int_step)
        self._single_step       = nqs.wrap_single_step_jax(batch_size = n_batch)

        #! storage for history
        self.history            = []
        self.history_std        = []
        self.lr_history         = []
        self.reg_history        = []
        self.timings            = NQSTrainTime(n_steps = 0)
        
        #! other parameters
        self.n_batch            = n_batch
        self.lower_states       = lower_states # list of NQS for lower states in excited state calculations

    # ------------------------------------------------------

    @contextmanager
    def _time(self, phase: str, fn, *args, **kwargs):
        """
        Context manager to time a function call and store elapsed time.

        Yields:
            result of fn(*args, **kwargs)
        """
        result, elapsed = timeit(fn, *args, **kwargs)
        self.timings[phase].append(elapsed)
        yield result

    # ------------------------------------------------------

    def _update_lr(self, epoch: int, last_loss: float, update_lr: bool = True):
        """
        Update the learning rate based on the epoch and last loss.
        Parameters
        ----------
        epoch : int
            Current epoch number.
        last_loss : float
            Last recorded loss.
        """
        if update_lr and self.lr_scheduler is not None:
            lr = self.lr_scheduler(epoch, last_loss)
            self.ode_solver.set_dt(float(lr))
            return lr
        return self.ode_solver.dt
    
    def _update_reg(self, epoch: int, last_loss: float, update_reg: bool = True):
        """
        Update the regularization based on the epoch and last loss.
        Parameters
        ----------
        epoch : int
            Current epoch number.
        last_loss : float
            Last recorded loss.
        update_reg : bool
            Whether to update the regularization.
        """
        if update_reg and self.reg_scheduler is not None:
            reg = self.reg_scheduler(epoch, last_loss)
            self.tdvp.set_diag_shift(reg)
            return reg
        return self.tdvp.sr_diag_shift
    
    # ------------------------------------------------------

    def _reset_history(self, reset: bool = False):
        """
        Reset the history of the training.
        Parameters
        ----------
        reset : bool
            Whether to reset the history.
        """
        if reset:
            self.history        = []
            self.history_std    = []
            self.timings.reset(n_steps=0)
        else:
            self.history        = list(self.history)
            self.history_std    = list(self.history_std)
    
    # ------------------------------------------------------

    def train(self,
            n_epochs            : int,
            reset               : bool = False,
            use_lr_scheduler    : bool = True,
            use_reg_scheduler   : bool = True,
            **kwargs):
        
        #! reset the history
        self._reset_history(reset=reset)
        
        #! reset the early stopping
        self.early_stopper.reset()

        #! get the current parameters
        last_params     = self.nqs.get_params()
        
        #! create the progress bar
        pbar            = trange(n_epochs, desc="Training NQS", leave=True)
        
        #! go through the epochs
        for epoch in pbar:
            
            #! schedulers
            lr          = self._update_lr(epoch, np.real(self.history[-1]) if self.history else None, use_lr_scheduler)
            reg         = self._update_reg(epoch, np.real(self.history[-1]) if self.history else None, use_reg_scheduler)

            #! sampling
            with self._time("sample", self.nqs.sample, reset=reset) as sample_out:
                (_, _), (cfgs, cfgs_psi), probs = sample_out

            #! sampling for lower states, if provided
            lower_contr = None
            if self.lower_states is not None:
                lower_contr = []
                # sample values for lower states
                for nqs_lower in self.lower_states:
                    with self._time("lower_states", nqs_lower.sample, reset=reset) as lower_sample_out:
                        (_, _), (cfgs_lower, cfgs_psi_lower), _ = lower_sample_out
                        lower_contr_class = TDVPLowerPenalty(
                            excited_on_lower        = self.nqs.ansatz(cfgs_lower),
                            lower_on_excited        = nqs_lower.ansatz(cfgs),
                            excited_on_excited      = cfgs_psi.flatten(),
                            lower_on_lower          = cfgs_psi_lower.flatten(),
                            params_j                = nqs_lower.get_params(),
                            configs_j               = cfgs_lower,
                            beta_j                  = nqs_lower.beta_penalty,
                            backend_np              = self.nqs.backend
                        )
                        lower_contr.append(lower_contr_class)

            #! energy + gradient
            params = self.nqs.get_params(unravel=True) # gives a vector instead of a dict
            with self._time("step", 
                            self.ode_solver.step, 
                            f               = self.tdvp,
                            y               = params,
                            t               = 0.0,
                            est_fn          = self._single_step,
                            configs         = cfgs,
                            configs_ansatze = cfgs_psi,
                            probabilities   = probs,
                            # for excited states
                            lower_states    = lower_contr
                            ) as step_out:
                dparams, _, (info, meta) = step_out

            #! update
            with self._time("update", self.nqs.set_params, dparams,
                                    shapes  = meta[0],
                                    sizes   = meta[1],
                                    iscpx   = meta[2]) as update_out:
                pass

            #! record
            mean_E      = info.mean_energy
            std_E       = info.std_energy
            std_E_real  = np.real(std_E)
            mean_E_real = np.real(mean_E)
            self.history.append(mean_E_real)
            self.history_std.append(std_E_real)

            #! add other times
            self.timings["gradient"].append(info.timings['gradient'])
            self.timings["prepare"].append(info.timings['prepare'])
            self.timings["solve"].append(info.timings['solve'])

            #! progress bar
            times       = {p: self.timings[p][-1] for p in ("sample","step","update","gradient","prepare","solve")}
            total       = sum(times.values())
            pbar.set_postfix({
                "E/N"           :   f"{mean_E / self.nqs.size:.4e} +/-  {std_E_real / self.nqs.size:.4e}",
                "lr"            :   f"{lr:.1e}",
                "sig"           :   f"{reg:.1e}",
                **{f"t_{k}"     :   f"{v:.2e}s" for k,v in times.items()},
                "t_total"       :   f"{total:.2e}s",
                }, refresh=True)

            #! check for NaN
            if np.isnan(mean_E) or np.isnan(std_E):
                self.logger.warning(f"NaN at epoch {epoch}, stopping.")
                break
            
            #! update the last parameters
            last_params         = self.nqs.get_params()
            
            #! save the checkpoint
            self.nqs.save_weights(step = epoch, save_metadata = True)
            
            if self.early_stopper(mean_E_real):
                self.logger.info(f"Early stopping at epoch {epoch}.")
                break
        
        #! set the last parameters in the end
        self.nqs.set_params(last_params)

        #! convert to arrays
        self.lr_history     = np.array(self.lr_scheduler.history)
        self.reg_history    = np.array(self.reg_scheduler.history)
        
        for k in self.timings.keys():
            self.timings[k] = np.array(self.timings[k])

        return np.array(self.history), np.array(self.history_std), self.timings

    # ------------------------------------------------------

    def _save_json(self, params: dict, filename: str):
        """
        Save the parameters to a json file.
        Parameters
        ----------
        params : dict
            Parameters to save.
        filename : str
            Filename to save the parameters to.
        """
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        self.logger.info(f"Saved parameters to {filename}", color="green", lvl=1)

    # ------------------------------------------------------
    
    def __repr__(self):
        return f"NQSTrainer(nqs={self.nqs}, ode_solver={self.ode_solver}, tdvp={self.tdvp}"
    
    def __str__(self):
        return f"NQSTrainer(nqs={self.nqs}, ode_solver={self.ode_solver}, tdvp={self.tdvp}"
    
    def __call__(self, *args, **kwargs):
        """
        Call the train method with the given arguments.
        """
        return self.train(*args, **kwargs)

# ------------------------------------------------------
#! EOF

        # betas           = kwargs.get('betas', None)     # penalties for the excited states - Array [n_lower_states,]
        # nqs_lower       = kwargs.get('nqs_lower', None) # ansatze for the lower states - list [f(s) -> log_psi_j(s)]
        # if betas is not None and nqs_lower is not None:
        #     '''
        #     For the energies, calculate the modified local energies:
        #     .. math::
        #         E_{loc}^k(s) = E_{loc}(s) + \\sum_{j=1}^{k} \\beta_j <\\psi | w_j >  < w_j | \\psi> / [|\\psi|^2 * <w_j|w_j>]
            
        #     For the derivatives, calculate the modified derivatives:
        #     .. math::
        #         O_k(s) = O(s) + \\sum_{j=1}^{k}
            
        #     '''
            
        #     if isinstance(nqs_lower, list) or isinstance(nqs_lower, tuple):
                
        #         # r_psi_low_ov_exc    = self.backend.stack([nqs_lower[j](net_params, t, configs) for j in range(len(nqs_lower))], axis=0)  # [n_lower_states, n_samples]
        #         # r_psi_exc_ov_low    = self.backend.stack([nqs_lower[j](net_params, t, configs_ansatze) for j in range(len(nqs_lower))], axis=0)  # [n_lower_states, n_samples]
        