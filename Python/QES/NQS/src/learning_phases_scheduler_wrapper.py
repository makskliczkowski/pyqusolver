"""
file        : NQS/src/learning_phases_scheduler_wrapper.py
author      : Maksymilian Kliczkowski
date        : November 1, 2025

Wrapper adapter to make LearningPhaseScheduler compatible with NQSTrainer.

This module provides adapters that make LearningPhaseScheduler work as a
drop-in replacement for ExponentialDecayScheduler and ConstantScheduler
in the NQSTrainer class, without requiring any changes to NQSTrainer code.

The wrapper converts LearningPhaseScheduler into callable schedulers that
follow the standard signature: scheduler(epoch, loss) -> float
"""

from typing import Optional, Callable, List, Dict, Any
from .learning_phases import LearningPhaseScheduler, LearningPhase, create_learning_phases


class LearningPhaseParameterScheduler:
    """
    Wrapper adapter for LearningPhaseScheduler to work with NQSTrainer.
    
    This class wraps a LearningPhaseScheduler and extracts either learning_rate
    or regularization parameters on-demand, making it compatible with the
    standard scheduler interface used by NQSTrainer.
    
    Usage with NQSTrainer:
    ```python
    # Create learning phases
    phases = create_learning_phases('default')
    
    # Wrap for learning rate
    lr_scheduler = LearningPhaseParameterScheduler(phases, param_type='learning_rate')
    
    # Wrap for regularization
    reg_scheduler = LearningPhaseParameterScheduler(phases, param_type='regularization')
    
    # Use in NQSTrainer (no changes to NQSTrainer code needed!)
    nqs_train = NQSTrainer(
        nqs=nqs,
        ode_solver=params.ode_solver,
        tdvp=params.tdvp,
        n_batch=params.numbatch,
        lr_scheduler=lr_scheduler,
        reg_scheduler=reg_scheduler,
        early_stopper=early_stopper,
        logger=logger
    )
    ```
    
    Attributes:
        scheduler (LearningPhaseScheduler): Underlying phase scheduler
        param_type (str): Either 'learning_rate' or 'regularization'
        logger (Optional[Callable]): Optional logging function
        history (List[float]): History of returned values (for compatibility)
    """
    
    def __init__(self, 
                 phases: List[LearningPhase],
                 param_type: str = 'learning_rate',
                 logger: Optional[Callable] = None):
        """
        Initialize the parameter scheduler wrapper.
        
        Parameters:
            phases (List[LearningPhase]): List of learning phases
            param_type (str): Parameter to extract: 'learning_rate' or 'regularization'
            logger (Optional[Callable]): Optional logging function
        """
        if param_type not in ['learning_rate', 'regularization']:
            raise ValueError(f"param_type must be 'learning_rate' or 'regularization', got '{param_type}'")
        
        self.scheduler = LearningPhaseScheduler(phases, logger=logger)
        self.param_type = param_type
        self.logger = logger if logger is not None else lambda msg, **kw: None
        self.history: List[float] = []
        
        self.log(f"Initialized {self.__class__.__name__} for extracting '{param_type}'", color='blue')
    
    def log(self, msg: str, **kwargs):
        """Log a message using the logger."""
        self.logger(msg, **kwargs)
    
    def __call__(self, epoch: int, loss: Optional[float] = None) -> float:
        """
        Get the hyperparameter value for the given epoch.
        
        This is the standard scheduler interface used by NQSTrainer.
        The `loss` parameter is accepted for compatibility but not used by
        learning phases (they use epoch-based scheduling).
        
        Parameters:
            epoch (int): Global training epoch (across all phases)
            loss (Optional[float]): Loss value (unused for epoch-based scheduling)
        
        Returns:
            float: The learning rate or regularization value for this epoch
        """
        # Map global epoch to phase-specific parameters
        phase_epoch = self._get_phase_epoch(epoch)
        hyperparams = self.scheduler.get_current_hyperparameters(phase_epoch)
        
        value = hyperparams[self.param_type]
        self.history.append(value)
        
        return value
    
    def _get_phase_epoch(self, global_epoch: int) -> int:
        """
        Convert global epoch to phase-specific epoch.
        
        Parameters:
            global_epoch (int): Epoch across all phases
        
        Returns:
            int: Epoch within the current phase
        """
        # Calculate epochs completed in previous phases
        completed_epochs = sum(
            p.epochs for p in self.scheduler.phases[:self.scheduler.current_phase_idx]
        )
        
        # Phase-specific epoch is relative to current phase start
        phase_epoch = global_epoch - completed_epochs
        
        # Check if we need to advance to next phase
        current_phase = self.scheduler.current_phase
        while phase_epoch >= current_phase.epochs and not self.scheduler.is_finished:
            self.scheduler.on_phase_end()
            self.scheduler.current_phase_idx += 1
            
            if not self.scheduler.is_finished:
                self.scheduler.on_phase_start()
                completed_epochs = sum(
                    p.epochs for p in self.scheduler.phases[:self.scheduler.current_phase_idx]
                )
                phase_epoch = global_epoch - completed_epochs
            else:
                phase_epoch = current_phase.epochs - 1
        
        return max(0, min(phase_epoch, current_phase.epochs - 1))
    
    def reset(self):
        """Reset the scheduler state and history."""
        self.scheduler = LearningPhaseScheduler(self.scheduler.phases, logger=self.logger)
        self.history = []
        self.log(f"Reset {self.__class__.__name__}", color='blue')
    
    def get_schedule_description(self) -> Dict[str, Any]:
        """
        Get a description of the learning schedule.
        
        Returns:
            Dict containing schedule metadata
        """
        phases_info = []
        total_epochs = 0
        
        for phase in self.scheduler.phases:
            total_epochs += phase.epochs
            phases_info.append({
                'name': phase.name,
                'epochs': phase.epochs,
                'type': phase.phase_type.name,
                f'{self.param_type}': phase.learning_rate if self.param_type == 'learning_rate' else phase.regularization,
                f'{self.param_type}_schedule': (
                    phase.lr_decay if self.param_type == 'learning_rate' 
                    else phase.reg_schedule
                )
            })
        
        return {
            'param_type': self.param_type,
            'total_epochs': total_epochs,
            'num_phases': len(self.scheduler.phases),
            'phases': phases_info,
            'history_length': len(self.history)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"LearningPhaseParameterScheduler("
                f"param_type='{self.param_type}', "
                f"phases={len(self.scheduler.phases)}, "
                f"total_epochs={sum(p.epochs for p in self.scheduler.phases)})")


def create_learning_phase_schedulers(
        preset: str = 'default',
        logger: Optional[Callable] = None
) -> tuple[LearningPhaseParameterScheduler, LearningPhaseParameterScheduler]:
    """
    Factory function to create both learning rate and regularization schedulers
    from a single learning phases preset.
    
    This is the recommended way to set up learning phases for NQSTrainer.
    
    Parameters:
        preset (str): Preset name: 'default', 'fast', 'thorough', 'kitaev'
        logger (Optional[Callable]): Optional logging function
    
    Returns:
        Tuple of (lr_scheduler, reg_scheduler) ready for NQSTrainer
    
    Example:
        ```python
        from QES.NQS.src.learning_phases_scheduler_wrapper import create_learning_phase_schedulers
        
        # Create both schedulers from preset
        lr_scheduler, reg_scheduler = create_learning_phase_schedulers('kitaev', logger=logger)
        
        # Use directly with NQSTrainer
        nqs_train = NQSTrainer(
            ...
            lr_scheduler=lr_scheduler,
            reg_scheduler=reg_scheduler,
            ...
        )
        ```
    """
    phases = create_learning_phases(preset)
    
    lr_scheduler = LearningPhaseParameterScheduler(
        phases, 
        param_type='learning_rate',
        logger=logger
    )
    
    reg_scheduler = LearningPhaseParameterScheduler(
        phases,
        param_type='regularization',
        logger=logger
    )
    
    return lr_scheduler, reg_scheduler


# Backwards compatibility: alias for common usage
LearningPhaseSchedulerLR = LearningPhaseParameterScheduler
LearningPhaseSchedulerReg = LearningPhaseParameterScheduler
