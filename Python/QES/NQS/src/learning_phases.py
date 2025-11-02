"""
file        : NQS/src/learning_phases.py
author      : Maksymilian Kliczkowski
date        : November 1, 2025

Learning phase framework for Neural Quantum State training.

This module implements a multi-phase training system for NQS, allowing:
- Phase transitions with configurable parameters
- Phase-specific callbacks and callbacks
- Adaptive learning rates per phase
- Regularization scheduling per phase
- Progress tracking and reporting

Learning phases represent different stages of optimization:
1. Pre-training: Initialize network with simple loss, high learning rate
2. Main Optimization: Full Hamiltonian, adaptive learning rate
3. Refinement: Fine-tune observables, low learning rate, high regularization
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum, auto
import numpy as np


class PhaseType(Enum):
    """Enumeration of learning phase types."""
    PRE_TRAINING = auto()
    MAIN = auto()
    REFINEMENT = auto()
    CUSTOM = auto()


@dataclass
class LearningPhase:
    """
    Represents a single learning phase in the NQS training process.
    
    A learning phase consists of:
    - Number of epochs
    - Learning rate and decay schedule
    - Regularization strength and schedule
    - Loss function type
    - Callbacks for phase events
    
    Attributes:
        name (str): Descriptive name of the phase (e.g., "pre-training", "main")
        epochs (int): Number of training epochs in this phase
        phase_type (PhaseType): Type of learning phase
        
        learning_rate (float): Initial learning rate for this phase
        lr_decay (float): Learning rate decay factor per epoch (multiplicative)
        lr_min (float): Minimum learning rate (clamp lower bound)
        
        regularization (float): Regularization strength for this phase
        reg_schedule (str): Regularization schedule ('constant', 'exponential', 'linear', 'adaptive')
        
        loss_type (str): Type of loss function to use ('energy', 'variance', 'combined', 'custom')
        loss_weight_energy (float): Weight of energy term in combined loss
        loss_weight_variance (float): Weight of variance term in combined loss
        
        beta_penalty (float): Beta penalty for excited state orthogonality
        
        on_phase_start (Optional[Callable]): Callback when phase begins
        on_phase_end (Optional[Callable]): Callback when phase ends
        on_epoch_start (Optional[Callable]): Callback at epoch start
        on_epoch_end (Optional[Callable]): Callback at epoch end
        
        description (str): Description of what this phase does
    """
    
    name: str = "learning_phase"
    epochs: int = 100
    phase_type: PhaseType = PhaseType.MAIN
    
    # Learning rate parameters
    learning_rate: float = 1e-2
    lr_decay: float = 0.999
    lr_min: float = 1e-4
    
    # Regularization parameters
    regularization: float = 1e-3
    reg_schedule: str = "constant"  # 'constant', 'exponential', 'linear', 'adaptive'
    
    # Loss function parameters
    loss_type: str = "energy"  # 'energy', 'variance', 'combined', 'custom'
    loss_weight_energy: float = 1.0
    loss_weight_variance: float = 0.0
    
    # Additional parameters
    beta_penalty: float = 0.0
    
    # Callbacks
    on_phase_start: Optional[Callable] = None
    on_phase_end: Optional[Callable] = None
    on_epoch_start: Optional[Callable] = None
    on_epoch_end: Optional[Callable] = None
    
    # Description
    description: str = ""
    
    def __post_init__(self):
        """Validate phase parameters after initialization."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.lr_min <= 0:
            raise ValueError(f"lr_min must be positive, got {self.lr_min}")
        if self.lr_decay <= 0 or self.lr_decay > 1:
            raise ValueError(f"lr_decay must be in (0, 1], got {self.lr_decay}")
        if self.regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {self.regularization}")
        
        if self.reg_schedule not in ['constant', 'exponential', 'linear', 'adaptive']:
            raise ValueError(f"Unknown reg_schedule: {self.reg_schedule}")
        
        if self.loss_type not in ['energy', 'variance', 'combined', 'custom']:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        if self.loss_weight_energy < 0 or self.loss_weight_variance < 0:
            raise ValueError("Loss weights must be non-negative")
    
    def get_learning_rate(self, epoch: int) -> float:
        """
        Get the learning rate for a given epoch within this phase.
        
        Parameters:
            epoch (int): Epoch number within this phase (0-indexed)
        
        Returns:
            float: Learning rate for this epoch
        """
        lr = self.learning_rate * (self.lr_decay ** epoch)
        return max(lr, self.lr_min)
    
    def get_regularization(self, epoch: int) -> float:
        """
        Get the regularization strength for a given epoch within this phase.
        
        Parameters:
            epoch (int): Epoch number within this phase (0-indexed)
        
        Returns:
            float: Regularization strength for this epoch
        """
        if self.reg_schedule == 'constant':
            return self.regularization
        
        elif self.reg_schedule == 'exponential':
            # Increase regularization exponentially
            decay = np.exp(-5 * epoch / self.epochs)  # Smooth decay
            return self.regularization * (1 + decay)
        
        elif self.reg_schedule == 'linear':
            # Linearly increase regularization
            factor = epoch / max(self.epochs - 1, 1)
            return self.regularization * (1 + factor)
        
        elif self.reg_schedule == 'adaptive':
            # Adaptive: high initially, decay over time
            return self.regularization * np.exp(-3 * epoch / self.epochs)
        
        else:
            return self.regularization
    
    def __repr__(self) -> str:
        """String representation of the learning phase."""
        return (f"LearningPhase(name={self.name}, epochs={self.epochs}, "
                f"lr={self.learning_rate:.2e}, reg={self.regularization:.2e}, "
                f"loss={self.loss_type})")


# Standard phase configurations
DEFAULT_PRE_TRAINING = LearningPhase(
    name="pre_training",
    epochs=50,
    phase_type=PhaseType.PRE_TRAINING,
    learning_rate=1e-1,
    lr_decay=0.995,
    lr_min=1e-3,
    regularization=5e-2,
    reg_schedule="exponential",
    loss_type="energy",
    description="Pre-training phase: Initialize network with high learning rate"
)

DEFAULT_MAIN = LearningPhase(
    name="main",
    epochs=200,
    phase_type=PhaseType.MAIN,
    learning_rate=3e-2,
    lr_decay=0.999,
    lr_min=1e-4,
    regularization=1e-3,
    reg_schedule="constant",
    loss_type="energy",
    description="Main optimization phase: Full Hamiltonian with adaptive learning rate"
)

DEFAULT_REFINEMENT = LearningPhase(
    name="refinement",
    epochs=100,
    phase_type=PhaseType.REFINEMENT,
    learning_rate=1e-2,
    lr_decay=0.9999,
    lr_min=1e-5,
    regularization=5e-3,
    reg_schedule="linear",
    loss_type="combined",
    loss_weight_energy=0.8,
    loss_weight_variance=0.2,
    description="Refinement phase: Fine-tune observables with low learning rate"
)

# Preset configurations
PRESETS = {
    "default": [DEFAULT_PRE_TRAINING, DEFAULT_MAIN, DEFAULT_REFINEMENT],
    "fast": [
        LearningPhase(name="pre_training", epochs=20, phase_type=PhaseType.PRE_TRAINING,
                     learning_rate=1e-1, regularization=1e-2),
        LearningPhase(name="main", epochs=50, phase_type=PhaseType.MAIN,
                     learning_rate=5e-2, regularization=1e-3),
    ],
    "thorough": [
        LearningPhase(name="pre_training", epochs=100, phase_type=PhaseType.PRE_TRAINING,
                     learning_rate=1e-1, regularization=1e-1),
        LearningPhase(name="main", epochs=500, phase_type=PhaseType.MAIN,
                     learning_rate=1e-2, regularization=5e-4),
        LearningPhase(name="refinement", epochs=200, phase_type=PhaseType.REFINEMENT,
                     learning_rate=5e-3, regularization=1e-3),
    ],
    "kitaev": [
        # Specialized for frustrated systems like Kitaev model
        LearningPhase(name="pre_training", epochs=75, phase_type=PhaseType.PRE_TRAINING,
                     learning_rate=1e-1, lr_decay=0.99, regularization=5e-2),
        LearningPhase(name="main", epochs=300, phase_type=PhaseType.MAIN,
                     learning_rate=2e-2, lr_decay=0.9995, regularization=1e-3),
        LearningPhase(name="refinement", epochs=150, phase_type=PhaseType.REFINEMENT,
                     learning_rate=5e-3, lr_decay=0.99995, regularization=5e-3),
    ],
}


class LearningPhaseScheduler:
    """
    Manages the training schedule across multiple learning phases.
    
    Handles:
    - Phase transitions
    - Learning rate and regularization updates
    - Callback execution
    - Progress tracking
    """
    
    def __init__(self, phases: List[LearningPhase], logger: Optional[Callable] = None):
        """
        Initialize the learning phase scheduler.
        
        Parameters:
            phases (List[LearningPhase]): List of learning phases in order
            logger (Optional[Callable]): Logging function for progress reporting
        """
        if not phases:
            raise ValueError("At least one learning phase is required")
        
        self.phases = phases
        self.logger = logger if logger is not None else lambda msg, **kw: None
        
        # Current phase tracking
        self.current_phase_idx = 0
        self.global_epoch = 0
        self.total_epochs = sum(p.epochs for p in phases)
        
        self._history = {
            'phases': [],
            'learning_rates': [],
            'regularizations': [],
            'epoch_losses': []
        }
    
    @property
    def current_phase(self) -> LearningPhase:
        """Get the current learning phase."""
        return self.phases[self.current_phase_idx]
    
    @property
    def is_finished(self) -> bool:
        """Check if all phases have been completed."""
        return self.current_phase_idx >= len(self.phases)
    
    def get_current_hyperparameters(self, phase_epoch: int) -> Dict[str, float]:
        """
        Get current hyperparameters based on phase and epoch.
        
        Parameters:
            phase_epoch (int): Epoch within current phase (0-indexed)
        
        Returns:
            Dict with 'learning_rate' and 'regularization' keys
        """
        phase = self.current_phase
        return {
            'learning_rate': phase.get_learning_rate(phase_epoch),
            'regularization': phase.get_regularization(phase_epoch)
        }
    
    def on_phase_start(self):
        """Execute callbacks when a new phase begins."""
        phase = self.current_phase
        self.logger(
            f"Starting {phase.phase_type.name} phase: '{phase.name}' "
            f"({phase.epochs} epochs, lr={phase.learning_rate:.2e})",
            color='green'
        )
        
        if phase.on_phase_start is not None:
            phase.on_phase_start()
        
        self._history['phases'].append({
            'name': phase.name,
            'start_epoch': self.global_epoch,
            'type': phase.phase_type.name
        })
    
    def on_phase_end(self):
        """Execute callbacks when a phase ends."""
        phase = self.current_phase
        self.logger(
            f"Completed {phase.phase_type.name} phase: '{phase.name}'",
            color='cyan'
        )
        
        if phase.on_phase_end is not None:
            phase.on_phase_end()
    
    def on_epoch_start(self, phase_epoch: int):
        """Execute callbacks at the start of each epoch."""
        phase = self.current_phase
        if phase.on_epoch_start is not None:
            phase.on_epoch_start(phase_epoch)
    
    def on_epoch_end(self, phase_epoch: int, loss: float):
        """Execute callbacks at the end of each epoch."""
        phase = self.current_phase
        hyperparams = self.get_current_hyperparameters(phase_epoch)
        
        if phase.on_epoch_end is not None:
            phase.on_epoch_end(phase_epoch, loss, hyperparams)
        
        # Track history
        self._history['learning_rates'].append(hyperparams['learning_rate'])
        self._history['regularizations'].append(hyperparams['regularization'])
        self._history['epoch_losses'].append(loss)
    
    def advance_epoch(self) -> bool:
        """
        Advance to the next epoch and handle phase transitions.
        
        Returns:
            bool: True if training should continue, False if all phases are complete
        """
        phase_epoch = self.global_epoch - sum(p.epochs for p in self.phases[:self.current_phase_idx])
        
        if phase_epoch + 1 >= self.current_phase.epochs:
            # Current phase is complete, move to next
            self.on_phase_end()
            self.current_phase_idx += 1
            
            if self.is_finished:
                self.logger("All learning phases completed", color='green')
                return False
            else:
                self.on_phase_start()
        
        self.global_epoch += 1
        return True
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of training progress across all phases."""
        return {
            'current_phase': self.current_phase.name,
            'current_phase_index': self.current_phase_idx,
            'total_phases': len(self.phases),
            'global_epoch': self.global_epoch,
            'total_epochs': self.total_epochs,
            'progress_pct': 100 * self.global_epoch / self.total_epochs,
            'history': self._history
        }


def create_learning_phases(preset: str = 'default') -> List[LearningPhase]:
    """
    Create a standard set of learning phases.
    
    Parameters:
        preset (str): Name of preset ('default', 'fast', 'thorough', 'kitaev')
    
    Returns:
        List[LearningPhase]: List of learning phases
    
    Example:
        >>> phases = create_learning_phases('default')
        >>> scheduler = LearningPhaseScheduler(phases)
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    return PRESETS[preset]
