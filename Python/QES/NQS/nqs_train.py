'''
NQS training class.
'''


import os
import json
import time
import numpy as np
from contextlib import contextmanager
from tqdm import trange
from typing import Union

from QES.general_python.common.plot import Plotter, colorsCycle, linestylesCycle
from QES.general_python.common.flog import Logger
from QES.general_python.common.timer import timeit

#! NQS
from NQS.nqs import NQS
from NQS.tdvp import TDVP
from QES.general_python.algebra.ode import IVP
from QES.general_python.ml.schedulers import Parameters, EarlyStopping

class NQSTrainer:
    '''
    Trainer class for Neural Quantum States (NQS).
    Implements training loop, learning rate scheduling, regularization scheduling, and early stopping.'
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
                **kwargs
            ):
        #! set the objects
        self.nqs                = nqs
        self.ode_solver         = ode_solver
        self.tdvp               = tdvp
        self.lr_scheduler       = lr_scheduler
        self.reg_scheduler      = reg_scheduler
        self.early_stopper      = early_stopper
        self.logger             = logger

        #! should I jit compile the ansatz - it is compiled but we can move it till the very end
        self.ansatz             = nqs.ansatz
        self.loc_energy         = nqs.local_energy
        self.flat_grad          = nqs.flat_grad
        
        #! JIT-compiled single step (y, t, *, configs, configs_ansatze, probabilities, int_step)
        self._single_step       = nqs.wrap_single_step_jax(batch_size = n_batch)

        # storage for history
        self.history            = []
        self.history_std        = []
        self.lr_history         = []
        self.reg_history        = []
        self.timings            = {"sample": [], "step": [], "update": [], "gradient": [], "prepare": [], "solve": []}

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
            self.timings        = {"sample": [], "step": [], "update": []}
        else:
            self.history        = list(self.history)
            self.history_std    = list(self.history_std)
            self.timings        = {k: list(v) for k, v in self.timings.items()}
    
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
        last_params = self.nqs.get_params()
        
        #! create the progress bar
        pbar = trange(n_epochs, desc="Training", leave=True)
        
        #! go through the epochs
        for epoch in pbar:
            
            #! schedulers
            lr          = self._update_lr(epoch, np.real(self.history[-1]) if self.history else None, use_lr_scheduler)
            reg         = self._update_reg(epoch, np.real(self.history[-1]) if self.history else None, use_reg_scheduler)

            #! sampling
            with self._time("sample", self.nqs.sample, reset=reset) as sample_out:
                (_, _), (cfgs, cfgs_psi), probs = sample_out

            #! energy + gradient
            params = self.nqs.get_params(unravel=True) # gives a vector instead of a dict
            with self._time("step", self.ode_solver.step, 
                                    f               = self.tdvp,
                                    y               = params,
                                    t               = 0.0,
                                    est_fn          = self._single_step,
                                    configs         = cfgs,
                                    configs_ansatze = cfgs_psi,
                                    probabilities   = probs) as step_out:
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
            self.timings["gradient"].append(info.times['gradient'])
            self.timings["prepare"].append(info.times['prepare'])
            self.timings["solve"].append(info.times['solve'])
            
            #! progress bar
            times       = {p: self.timings[p][-1] for p in ("sample","step","update","gradient","prepare","solve")}
            total       = sum(times.values())
            pbar.set_postfix({
                "E/N"           :   f"{mean_E / self.nqs.size:.4e} ± {std_E_real / self.nqs.size:.4e}",
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
            last_params = self.nqs.get_params()
            
            #! save the checkpoint
            self.nqs.save_weights(
                step = epoch,
                save_metadata = True,
            )
            
            if self.early_stopper(mean_E_real):
                self.logger.info(f"Early stopping at epoch {epoch}.")
                break
        
        #! set the last parameters in the end
        self.nqs.set_params(last_params)

        #! convert to arrays
        self.history        = np.array(self.history)
        self.history_std    = np.array(self.history_std)
        self.lr_history     = np.array(self.lr_scheduler.history)
        self.reg_history    = np.array(self.reg_scheduler.history)
        
        for k in self.timings:
            self.timings[k] = np.array(self.timings[k])

        return self.history, self.history_std, self.timings

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