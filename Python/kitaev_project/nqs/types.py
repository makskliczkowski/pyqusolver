"""
Shared typing utilities for the NQS workflows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class NQSTrainingConfig:
    """
    High-level configuration for a single training run.
    """

    architecture: str = "autoregressive"
    backend: str = "jax"
    sampler: str = "metropolis_local"
    batch_size: int = 1024
    n_epochs: int = 400
    learning_rate: float = 5e-3
    reg_strength: float = 1e-3
    orthogonality_beta: float = 0.0
    excited_states: int = 0
    integrator_dt: float = 0.05
    patience: int = 50
    min_delta: float = 1e-4
    seed: Optional[int] = None
    checkpoint_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Sampler parameters
    num_chains: int = 10
    num_samples: int = 200
    num_sweeps: int = 16
    num_burnin: int = 25
    
    # SR parameters
    use_minsr: bool = False
    solver_maxiter: int = 600
    solver_tol: float = 1e-8
    solver_cutoff: float = 1e-8


@dataclass
class TrainingArtifact:
    """
    Container storing everything we persist after a run.
    """

    ansatz_name: str
    stage: str
    hdf5_path: str
    metrics: Dict[str, Any]
    observables: Dict[str, Any]
    network_state: Dict[str, Any]
    extras: Dict[str, Any] = field(default_factory=dict)


class NeuralAnsatz(Protocol):
    """
    Minimal protocol that every ansatz implementation must satisfy.
    """

    name: str

    def create_solver(self, hamiltonian, hilbert_space, config: NQSTrainingConfig):
        ...

    def snapshot(self) -> Dict[str, Any]:
        ...
