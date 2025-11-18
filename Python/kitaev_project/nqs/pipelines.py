"""
Workflow helpers for different NQS study modes.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .training import NQSTrainer
from .types import NQSTrainingConfig, TrainingArtifact, NeuralAnsatz


def train_ground_state(
    trainer: NQSTrainer,
    ansatz: NeuralAnsatz,
    config: NQSTrainingConfig,
) -> TrainingArtifact:
    return trainer.train(ansatz, config, stage="ground_state")


def train_excited_states(
    trainer: NQSTrainer,
    ansatz: NeuralAnsatz,
    config: NQSTrainingConfig,
    n_states: int = 4,
) -> List[TrainingArtifact]:
    artifacts = []
    for idx in range(n_states):
        excited_cfg = NQSTrainingConfig(
            **{
                **config.__dict__,
                "excited_states": idx + 1,
                "orthogonality_beta": config.orthogonality_beta or 0.1,
            }
        )
        artifacts.append(
            trainer.train(ansatz, excited_cfg, stage=f"excited_{idx+1}")
        )
    return artifacts


def fine_tune_with_impurities(
    trainer_factory,
    ansatz: NeuralAnsatz,
    base_config: NQSTrainingConfig,
    impurity_configs: Sequence[Dict],
):
    """
    Convenience wrapper that rebuilds trainer/model per impurity setup.
    """
    tuned_runs = []
    for imp_cfg in impurity_configs:
        trainer = trainer_factory(imp_cfg)
        tuned_runs.append(
            trainer.train(ansatz, base_config, stage="impurity_finetune")
        )
    return tuned_runs


def sweep_parameter(
    trainer_factory,
    ansatz: NeuralAnsatz,
    base_config: NQSTrainingConfig,
    parameter: str,
    values: Iterable[float],
) -> Dict[float, TrainingArtifact]:
    """
    Perform a simple parameter sweep by re-running the trainer with updated configs.
    """
    results = {}
    for value in values:
        cfg_dict = base_config.__dict__.copy()
        cfg_dict["metadata"] = {**cfg_dict.get("metadata", {}), f"sweep_{parameter}": value}
        cfg = NQSTrainingConfig(**cfg_dict)
        trainer = trainer_factory(value)
        results[value] = trainer.train(
            ansatz, cfg, stage=f"sweep_{parameter}"
        )
    return results
