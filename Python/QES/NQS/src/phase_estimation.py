"""
file        : NQS/src/phase_estimation.py
author      : Maksymilian Kliczkowski
date        : November 1, 2025

Quantum phase estimation for Neural Quantum States.

Based on quantum phase estimation (QPE) principles, this module implements:
1. Phase estimation via controlled operators (phase kickback)
2. Amplitude estimation techniques
3. Integration with NQS for variational phase extraction

Key references:
- arXiv:2506.03124: Phase estimation techniques for quantum states
- Quantum Phase Estimation Algorithm (QPEA)
- Variational Quantum Eigensolver phase extraction

The module provides tools to:
- Extract phases of quantum amplitudes
- Estimate geometric phases
- Use phase information for adaptive learning control
- Improve convergence via phase-aware optimization
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from enum import Enum, auto
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = np
    JAX_AVAILABLE = False


class PhaseEstimationType(Enum):
    """Types of phase estimation methods."""
    AMPLITUDE_PHASE = auto()          # Extract phase from amplitude
    GEOMETRIC_PHASE = auto()          # Berry/geometric phase
    RELATIVE_PHASE = auto()           # Phase between two states
    CONTROLLED_UNITARY = auto()       # Phase kickback from controlled-U
    SWAP_TEST = auto()                # Phase via SWAP test
    CUSTOM = auto()                   # Custom phase estimation


@dataclass
class PhaseEstimationConfig:
    """
    Configuration for phase estimation.
    
    Attributes:
        method (PhaseEstimationType): Phase estimation method
        num_shots (int): Number of samples for phase estimation
        phase_precision (float): Precision of phase measurement (radians)
        regularization (float): Regularization for phase extraction
        use_jax (bool): Whether to use JAX for computations
        callback (Optional[Callable]): Callback for phase updates
        description (str): Description of the configuration
    """
    
    method: PhaseEstimationType = PhaseEstimationType.AMPLITUDE_PHASE
    num_shots: int = 1000
    phase_precision: float = 1e-2
    regularization: float = 1e-4
    use_jax: bool = True
    callback: Optional[Callable] = None
    description: str = "Standard phase estimation"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_shots <= 0:
            raise ValueError(f"num_shots must be positive, got {self.num_shots}")
        if self.phase_precision <= 0:
            raise ValueError(f"phase_precision must be positive, got {self.phase_precision}")
        if self.regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {self.regularization}")


@dataclass
class PhaseEstimationResult:
    """
    Result from phase estimation.
    
    Attributes:
        phase (np.ndarray or float): Estimated phase(s)
        phase_std (Optional[float]): Standard deviation of phase estimate
        confidence (float): Confidence of the estimate (0-1)
        method (PhaseEstimationType): Method used
        samples_used (int): Number of samples used
        converged (bool): Whether estimation converged
        metadata (Dict): Additional information
    """
    
    phase: Union[np.ndarray, float]
    phase_std: Optional[float] = None
    confidence: float = 1.0
    method: PhaseEstimationType = PhaseEstimationType.AMPLITUDE_PHASE
    samples_used: int = 0
    converged: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """String representation."""
        phase_str = f"{self.phase:.4f}" if isinstance(self.phase, float) else f"shape{self.phase.shape}"
        return (f"PhaseEstimationResult("
                f"phase={phase_str}, "
                f"std={self.phase_std:.2e if self.phase_std else None}, "
                f"confidence={self.confidence:.3f}, "
                f"method={self.method.name})")


class PhaseExtractor:
    """
    Main class for quantum phase estimation from NQS amplitudes.
    
    Provides methods to:
    1. Extract phases from wavefunction amplitudes
    2. Estimate geometric phases along learning trajectories
    3. Compute relative phases between quantum states
    4. Use controlled unitaries for phase measurement
    5. Integrate phase information into learning
    """
    
    def __init__(self, config: Optional[PhaseEstimationConfig] = None):
        """
        Initialize phase extractor.
        
        Parameters:
            config (Optional[PhaseEstimationConfig]): Configuration for phase estimation
        """
        self.config = config if config is not None else PhaseEstimationConfig()
        self.history: List[PhaseEstimationResult] = []
        self._jax_available = JAX_AVAILABLE and self.config.use_jax
    
    def estimate_amplitude_phase(self,
                                amplitudes: np.ndarray,
                                states: Optional[np.ndarray] = None) -> PhaseEstimationResult:
        """
        Extract phase from complex amplitude.
        
        For a complex amplitude A = |A| * e^(iφ), extract φ.
        
        Parameters:
            amplitudes (np.ndarray): Complex amplitudes
            states (Optional[np.ndarray]): Quantum states (for context)
        
        Returns:
            PhaseEstimationResult with estimated phase
        """
        amplitudes = np.asarray(amplitudes)
        
        # Handle single amplitude
        if amplitudes.ndim == 0:
            amplitudes = amplitudes.reshape(1)
        
        # Extract phases using complex argument
        phases = np.angle(amplitudes)
        
        # Compute confidence from amplitude magnitude
        magnitudes = np.abs(amplitudes)
        confidence = np.mean(magnitudes) / (np.max(magnitudes) + 1e-10)
        
        result = PhaseEstimationResult(
            phase=phases if len(phases) > 1 else phases[0],
            phase_std=np.std(phases) if len(phases) > 1 else None,
            confidence=float(confidence),
            method=PhaseEstimationType.AMPLITUDE_PHASE,
            samples_used=len(amplitudes),
            converged=True
        )
        
        self.history.append(result)
        return result
    
    def estimate_geometric_phase(self,
                                phases_trajectory: List[np.ndarray]) -> PhaseEstimationResult:
        """
        Estimate geometric (Berry) phase along a learning trajectory.
        
        The geometric phase is accumulated as the system evolves along a closed path
        in parameter space.
        
        Parameters:
            phases_trajectory (List[np.ndarray]): Phases at each step of trajectory
        
        Returns:
            PhaseEstimationResult with geometric phase
        """
        if len(phases_trajectory) < 2:
            raise ValueError("Need at least 2 trajectory points")
        
        phases_trajectory = [np.asarray(p) for p in phases_trajectory]
        
        # Compute accumulated phase difference (unwrapped)
        phase_diff = np.diff([p.mean() if hasattr(p, 'mean') else p for p in phases_trajectory])
        geometric_phase = np.sum(phase_diff)
        
        # Normalize to [-π, π]
        geometric_phase = np.angle(np.exp(1j * geometric_phase))
        
        result = PhaseEstimationResult(
            phase=geometric_phase,
            phase_std=np.std(phase_diff),
            confidence=0.8,  # Moderate confidence for geometric phase
            method=PhaseEstimationType.GEOMETRIC_PHASE,
            samples_used=len(phases_trajectory),
            converged=True,
            metadata={'trajectory_length': len(phases_trajectory)}
        )
        
        self.history.append(result)
        return result
    
    def estimate_relative_phase(self,
                               psi1: np.ndarray,
                               psi2: np.ndarray) -> PhaseEstimationResult:
        """
        Estimate relative phase between two quantum states.
        
        Computes phase of <ψ₁|ψ₂> / |<ψ₁|ψ₂>|
        
        Parameters:
            psi1 (np.ndarray): First state (log amplitudes)
            psi2 (np.ndarray): Second state (log amplitudes)
        
        Returns:
            PhaseEstimationResult with relative phase
        """
        psi1 = np.asarray(psi1)
        psi2 = np.asarray(psi2)
        
        # Compute overlap (as complex numbers in log space)
        # <ψ₁|ψ₂> = sum(exp(psi1_i + conj(psi2_i)))
        if psi1.dtype == np.float64 or psi1.dtype == np.float32:
            # Real amplitudes - no phase
            relative_phase = 0.0
            confidence = 0.5
        else:
            # Complex amplitudes
            overlap = np.sum(np.exp(psi1 + np.conj(psi2)))
            relative_phase = np.angle(overlap)
            confidence = np.abs(overlap) / (len(psi1) + 1e-10)
        
        result = PhaseEstimationResult(
            phase=relative_phase,
            confidence=float(confidence),
            method=PhaseEstimationType.RELATIVE_PHASE,
            samples_used=len(psi1),
            converged=True
        )
        
        self.history.append(result)
        return result
    
    def estimate_controlled_phase(self,
                                 state: np.ndarray,
                                 control_operator: Callable,
                                 num_iterations: int = 10) -> PhaseEstimationResult:
        """
        Estimate phase using controlled unitary (phase kickback).
        
        Applies control operator repeatedly and extracts phase from interference.
        
        Parameters:
            state (np.ndarray): Quantum state
            control_operator (Callable): Controlled unitary operation
            num_iterations (int): Number of controlled operation iterations
        
        Returns:
            PhaseEstimationResult with estimated phase
        """
        state = np.asarray(state)
        phases = []
        
        # Apply controlled operator iteratively
        current_state = state.copy()
        for i in range(num_iterations):
            # Apply controlled operation
            next_state = control_operator(current_state)
            
            # Extract phase from overlap
            overlap = np.vdot(state, next_state)
            phase = np.angle(overlap)
            phases.append(phase)
            
            current_state = next_state
        
        # Average phase across iterations
        avg_phase = np.angle(np.mean([np.exp(1j * p) for p in phases]))
        
        result = PhaseEstimationResult(
            phase=avg_phase,
            phase_std=np.std(phases),
            confidence=0.7,
            method=PhaseEstimationType.CONTROLLED_UNITARY,
            samples_used=num_iterations,
            converged=True
        )
        
        self.history.append(result)
        return result
    
    def compute_phase_gradient(self,
                              phases: np.ndarray,
                              epoch: int) -> float:
        """
        Compute gradient of phase with respect to learning evolution.
        
        Can be used to adapt learning rates based on phase evolution.
        
        Parameters:
            phases (np.ndarray): Phases at different epochs
            epoch (int): Current epoch
        
        Returns:
            float: Phase gradient
        """
        if len(self.history) < 2:
            return 0.0
        
        recent_results = self.history[-10:]  # Last 10 measurements
        
        if len(recent_results) < 2:
            return 0.0
        
        # Extract phases
        phase_sequence = [r.phase for r in recent_results]
        if isinstance(phase_sequence[0], np.ndarray):
            phase_sequence = [p.mean() for p in phase_sequence]
        
        # Compute finite difference
        phase_gradient = np.diff(phase_sequence)[-1] if len(phase_sequence) > 1 else 0.0
        
        return float(phase_gradient)
    
    def should_update_learning_rate(self,
                                   phase_gradient: float,
                                   threshold: float = 0.1) -> bool:
        """
        Determine if learning rate should be updated based on phase evolution.
        
        Parameters:
            phase_gradient (float): Phase gradient from compute_phase_gradient
            threshold (float): Threshold for decision
        
        Returns:
            bool: True if learning rate should be increased
        """
        return abs(phase_gradient) < threshold
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get summary of phase estimation history.
        
        Returns:
            Dict with history statistics
        """
        if not self.history:
            return {'num_estimations': 0}
        
        phases = []
        confidences = []
        
        for result in self.history:
            phase = result.phase
            if isinstance(phase, np.ndarray):
                phases.extend(phase.flatten().tolist())
            else:
                phases.append(float(phase))
            confidences.append(result.confidence)
        
        return {
            'num_estimations': len(self.history),
            'num_phases': len(phases),
            'mean_phase': float(np.mean(phases)) if phases else 0.0,
            'std_phase': float(np.std(phases)) if phases else 0.0,
            'mean_confidence': float(np.mean(confidences)),
            'methods_used': list(set(r.method.name for r in self.history))
        }
    
    def reset(self):
        """Reset history."""
        self.history = []
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"PhaseExtractor("
                f"method={self.config.method.name}, "
                f"num_shots={self.config.num_shots}, "
                f"estimations={len(self.history)})")


# Convenience functions for quick phase estimation

def extract_phase_from_amplitude(amplitude: complex) -> float:
    """Quick phase extraction from a single complex amplitude."""
    return float(np.angle(amplitude))


def compute_phase_difference(phi1: float, phi2: float) -> float:
    """Compute wrapped phase difference."""
    diff = phi2 - phi1
    return float(np.angle(np.exp(1j * diff)))
