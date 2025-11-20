"""
Command-line utility demonstrating the recommended workflow wiring.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..models import KitaevModelBuilder, ModelConfig
from ..nqs import NQSTrainer, NQSTrainingConfig
from ..nqs.architectures import (
    AutoregressiveAnsatz,
    RBMAnsatz,
    SimpleConvAnsatz,
)
from ..nqs.pipelines import train_ground_state
from ..ed import LanczosSolver, LanczosConfig
from ..io import KitaevResultWriter


def available_ansatze():
    return {
        "autoregressive": AutoregressiveAnsatz(),
        "rbm": RBMAnsatz(),
        "simple_conv": SimpleConvAnsatz(),
    }


def run_pipeline(
    model_cfg: ModelConfig,
    training_cfg: NQSTrainingConfig,
    output_path: Path,
):
    builder = KitaevModelBuilder()
    lattice = builder.build_lattice(model_cfg)
    hilbert = builder.build_hilbert_space(lattice)
    hamiltonian = builder.build_hamiltonian(model_cfg, hilbert_space=hilbert)

    writer = KitaevResultWriter(path=output_path)
    trainer = NQSTrainer(hamiltonian, hilbert, writer=writer)

    ansatz = available_ansatze()[training_cfg.architecture]
    artifact = train_ground_state(trainer, ansatz, training_cfg)

    ed_solver = LanczosSolver(hamiltonian, LanczosConfig(lattice_label=f"{model_cfg.lx}x{model_cfg.ly}"))
    ed_result = ed_solver.run()
    writer.write_ed_result("benchmark", ed_result.energies, ed_result.metadata)

    writer.write_metadata(
        {
            "model": model_cfg.summary(),
            "training": training_cfg.metadata,
            "artifact_path": artifact.hdf5_path,
        }
    )
    return artifact.hdf5_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Demo pipeline for Kitaev workflow.")
    parser.add_argument("--output", type=Path, default=Path("kitaev_results.h5"))
    parser.add_argument("--architecture", choices=list(available_ansatze().keys()), default="autoregressive")
    parser.add_argument("--lx", type=int, default=3)
    parser.add_argument("--ly", type=int, default=2)
    parser.add_argument("--lz", type=int, default=1)
    parser.add_argument("--Kz", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def main():
    args = _parse_args()
    model_cfg = ModelConfig(lx=args.lx, ly=args.ly, lz=args.lz, K={"x": 1.0, "y": 1.0, "z": args.Kz})
    training_cfg = NQSTrainingConfig(architecture=args.architecture, n_epochs=args.epochs)
    run_pipeline(model_cfg, training_cfg, args.output)


if __name__ == "__main__":
    main()
