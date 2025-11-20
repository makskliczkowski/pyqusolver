"""
Neural-quantum-state utilities for the Kitaev workflow.
"""

from .types import NeuralAnsatz, NQSTrainingConfig, TrainingArtifact

# Lazy imports to avoid circular dependencies and heavy QES.NQS imports
def _get_trainer():
    from .training import NQSTrainer
    return NQSTrainer

def _get_pipelines():
    from .pipelines import (
        train_ground_state,
        train_excited_states,
        fine_tune_with_impurities,
        sweep_parameter,
    )
    return train_ground_state, train_excited_states, fine_tune_with_impurities, sweep_parameter

# Expose lazy-loaded classes via __getattr__
def __getattr__(name):
    if name == "NQSTrainer":
        return _get_trainer()
    elif name in ("train_ground_state", "train_excited_states", "fine_tune_with_impurities", "sweep_parameter"):
        pipelines = _get_pipelines()
        mapping = {
            "train_ground_state": pipelines[0],
            "train_excited_states": pipelines[1],
            "fine_tune_with_impurities": pipelines[2],
            "sweep_parameter": pipelines[3],
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "NeuralAnsatz",
    "NQSTrainingConfig",
    "TrainingArtifact",
    "NQSTrainer",
    "train_ground_state",
    "train_excited_states",
    "fine_tune_with_impurities",
    "sweep_parameter",
]

