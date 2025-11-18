"""
Main workflow orchestrator for Kitaev honeycomb impurity studies.

This script coordinates:
1. Ground state computation via NQS and ED
2. Excited states (lowest 4 states)
3. Transfer learning with impurities
4. Parameter sweeps (e.g., varying K_z)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..models import KitaevModelBuilder, ModelConfig
from ..nqs import NQSTrainer, NQSTrainingConfig
from ..nqs.architectures import AutoregressiveAnsatz, RBMAnsatz, SimpleConvAnsatz
from ..nqs.pipelines import (
    train_ground_state,
    train_excited_states,
    fine_tune_with_impurities,
    sweep_parameter,
)
from ..ed import LanczosSolver, LanczosConfig
from ..io import KitaevResultWriter


def available_architectures():
    """Return dictionary of available NQS architectures."""
    return {
        "autoregressive": AutoregressiveAnsatz(),
        "rbm": RBMAnsatz(),
        "simple_conv": SimpleConvAnsatz(),
    }


class KitaevWorkflow:
    """
    Complete workflow orchestrator for Kitaev model studies.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: NQSTrainingConfig,
        output_dir: Path,
        run_ed: bool = True,
        ed_max_sites: int = 24,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_ed = run_ed
        self.ed_max_sites = ed_max_sites

        self.builder = KitaevModelBuilder()
        self.writer = KitaevResultWriter(path=self.output_dir / "results.h5")

        # Build clean model
        self.lattice = self.builder.build_lattice(model_config)
        self.hilbert = self.builder.build_hilbert_space(self.lattice)
        self.hamiltonian = self.builder.build_hamiltonian(
            model_config, hilbert_space=self.hilbert
        )

        print(f"Initialized Kitaev workflow:")
        print(f"  Lattice: {model_config.lx}×{model_config.ly} ({self.lattice.Ns} sites)")
        print(f"  Output: {self.output_dir}")
        print(f"  ED enabled: {self.run_ed and self.lattice.Ns <= self.ed_max_sites}")

    def run_ground_state(self, architecture: str = "rbm") -> Dict:
        """
        Compute ground state using both NQS and ED (if enabled).
        
        Args:
            architecture: NQS architecture to use
            
        Returns:
            Dictionary with NQS and ED results
        """
        print("\n" + "=" * 60)
        print("TASK 1: Ground State Computation")
        print("=" * 60)

        results = {}

        # NQS ground state
        print(f"\n→ Training NQS ({architecture}) for ground state...")
        ansatz = available_architectures()[architecture]
        trainer = NQSTrainer(self.hamiltonian, self.hilbert, writer=self.writer)
        nqs_artifact = train_ground_state(trainer, ansatz, self.training_config)
        results["nqs"] = nqs_artifact
        print(f"  Energy: {nqs_artifact.observables.get('energy', 'N/A')}")

        # ED benchmark
        if self.run_ed and self.lattice.Ns <= self.ed_max_sites:
            print(f"\n→ Running ED/Lanczos for benchmark...")
            ed_config = LanczosConfig(
                lattice_label=f"{self.model_config.lx}x{self.model_config.ly}",
                n_states=1,
                compute_observables=True,
            )
            ed_solver = LanczosSolver(
                self.hamiltonian, self.lattice, self.hilbert, ed_config
            )
            ed_result = ed_solver.run()
            results["ed"] = ed_result
            
            # Write ED results
            self.writer.write_ed_result(
                "ground_state",
                ed_result.energies,
                {
                    **ed_result.metadata,
                    "observables": ed_result.observables,
                },
            )
            print(f"  ED Energy: {ed_result.energies[0]:.6f}")
            
            # Compare
            if not np.isnan(nqs_artifact.observables.get("energy", np.nan)):
                nqs_energy = nqs_artifact.observables["energy"]
                ed_energy = ed_result.energies[0]
                rel_error = abs(nqs_energy - ed_energy) / abs(ed_energy)
                print(f"  Relative error: {rel_error:.2e}")

        print("\n✓ Ground state computation complete")
        return results

    def run_excited_states(self, architecture: str = "rbm", n_states: int = 4) -> Dict:
        """
        Compute lowest excited states using NQS and ED.
        
        Args:
            architecture: NQS architecture to use
            n_states: Number of excited states to compute (beyond ground state)
            
        Returns:
            Dictionary with NQS and ED excited state results
        """
        print("\n" + "=" * 60)
        print(f"TASK 2: Excited States (lowest {n_states})")
        print("=" * 60)

        results = {}

        # NQS excited states
        print(f"\n→ Training NQS for {n_states} excited states...")
        ansatz = available_architectures()[architecture]
        trainer = NQSTrainer(self.hamiltonian, self.hilbert, writer=self.writer)
        nqs_artifacts = train_excited_states(
            trainer, ansatz, self.training_config, n_states=n_states
        )
        results["nqs"] = nqs_artifacts
        
        for idx, artifact in enumerate(nqs_artifacts):
            energy = artifact.observables.get("energy", np.nan)
            print(f"  State {idx+1}: E = {energy}")

        # ED excited states
        if self.run_ed and self.lattice.Ns <= self.ed_max_sites:
            print(f"\n→ Computing {n_states} excited states via ED...")
            ed_config = LanczosConfig(
                lattice_label=f"{self.model_config.lx}x{self.model_config.ly}",
                n_states=n_states + 1,  # Include ground state
                compute_observables=True,
            )
            ed_solver = LanczosSolver(
                self.hamiltonian, self.lattice, self.hilbert, ed_config
            )
            ed_result = ed_solver.run()
            results["ed"] = ed_result
            
            self.writer.write_ed_result(
                "excited_states",
                ed_result.energies,
                {
                    **ed_result.metadata,
                    "observables": ed_result.observables,
                },
            )
            
            for idx, energy in enumerate(ed_result.energies[1:], 1):
                print(f"  ED State {idx}: E = {energy:.6f}")

        print("\n✓ Excited states computation complete")
        return results

    def run_transfer_learning(
        self,
        architecture: str = "rbm",
        impurity_sites: List[int] = None,
        impurity_strengths: List[float] = None,
    ) -> Dict:
        """
        Transfer learning: train on clean model, then fine-tune with impurities.
        
        Args:
            architecture: NQS architecture to use
            impurity_sites: List of site indices for impurities
            impurity_strengths: List of impurity strengths (same length as sites)
            
        Returns:
            Dictionary with clean and impurity-tuned results
        """
        print("\n" + "=" * 60)
        print("TASK 3: Transfer Learning with Impurities")
        print("=" * 60)

        if impurity_sites is None:
            impurity_sites = [0, self.lattice.Ns // 2]
        if impurity_strengths is None:
            impurity_strengths = [1.0, -1.0]

        results = {}

        # First train on clean model
        print("\n→ Training on clean Kitaev model...")
        ansatz = available_architectures()[architecture]
        trainer = NQSTrainer(self.hamiltonian, self.hilbert, writer=self.writer)
        clean_artifact = train_ground_state(trainer, ansatz, self.training_config)
        results["clean"] = clean_artifact
        print(f"  Clean model energy: {clean_artifact.observables.get('energy', 'N/A')}")

        # Fine-tune with impurities
        print(f"\n→ Fine-tuning with impurities at sites {impurity_sites}...")
        impurity_configs = []
        
        for site, strength in zip(impurity_sites, impurity_strengths):
            # Create model config with impurity
            imp_config = ModelConfig(
                **{
                    **self.model_config.__dict__,
                    "impurities": [(site, strength)],
                }
            )
            impurity_configs.append(imp_config)
            
            # Build impurity Hamiltonian
            imp_hamiltonian = self.builder.build_hamiltonian(
                imp_config, hilbert_space=self.hilbert
            )
            
            # Fine-tune (use reduced epochs)
            finetune_config = NQSTrainingConfig(
                **{
                    **self.training_config.__dict__,
                    "n_epochs": self.training_config.n_epochs // 2,
                }
            )
            
            imp_trainer = NQSTrainer(imp_hamiltonian, self.hilbert, writer=self.writer)
            imp_artifact = imp_trainer.train(
                ansatz, finetune_config, stage=f"impurity_site_{site}"
            )
            
            key = f"impurity_{site}_{strength:+.1f}"
            results[key] = imp_artifact
            print(f"  Impurity at site {site} (strength {strength:+.1f}): "
                  f"E = {imp_artifact.observables.get('energy', 'N/A')}")

        print("\n✓ Transfer learning complete")
        return results

    def run_parameter_sweep(
        self,
        architecture: str = "rbm",
        parameter: str = "Kz",
        values: List[float] = None,
    ) -> Dict:
        """
        Sweep a coupling parameter and track observables.
        
        Args:
            architecture: NQS architecture to use
            parameter: Parameter to sweep ('Kz', 'J', 'hz', etc.)
            values: List of parameter values to sweep
            
        Returns:
            Dictionary mapping parameter values to results
        """
        print("\n" + "=" * 60)
        print(f"TASK 4: Parameter Sweep ({parameter})")
        print("=" * 60)

        if values is None:
            values = np.linspace(0.0, 2.0, 5).tolist()

        results = {}
        ansatz = available_architectures()[architecture]

        print(f"\n→ Sweeping {parameter} over {len(values)} values...")
        
        for value in values:
            # Update model config
            sweep_config = ModelConfig(**self.model_config.__dict__)
            
            if parameter == "Kz":
                sweep_config.K["z"] = value
            elif parameter == "Kx":
                sweep_config.K["x"] = value
            elif parameter == "Ky":
                sweep_config.K["y"] = value
            elif parameter == "J":
                sweep_config.J = value
            elif parameter == "hz":
                sweep_config.hz = value
            elif parameter == "hx":
                sweep_config.hx = value
            else:
                print(f"  Warning: Unknown parameter {parameter}")
                continue

            # Build Hamiltonian for this parameter value
            sweep_hamiltonian = self.builder.build_hamiltonian(
                sweep_config, hilbert_space=self.hilbert
            )
            
            # Train
            trainer = NQSTrainer(sweep_hamiltonian, self.hilbert, writer=self.writer)
            artifact = trainer.train(
                ansatz, self.training_config, stage=f"sweep_{parameter}_{value:.3f}"
            )
            
            results[value] = artifact
            print(f"  {parameter} = {value:.3f}: E = {artifact.observables.get('energy', 'N/A')}")
            
            # Also run ED for small systems
            if self.run_ed and self.lattice.Ns <= self.ed_max_sites:
                ed_config = LanczosConfig(
                    lattice_label=f"{parameter}={value:.3f}",
                    n_states=1,
                    compute_observables=True,
                )
                ed_solver = LanczosSolver(
                    sweep_hamiltonian, self.lattice, self.hilbert, ed_config
                )
                ed_result = ed_solver.run()
                
                self.writer.write_ed_result(
                    f"sweep_{parameter}_{value:.3f}",
                    ed_result.energies,
                    ed_result.metadata,
                )

        print("\n✓ Parameter sweep complete")
        return results

    def run_full_workflow(
        self,
        architecture: str = "rbm",
        n_excited: int = 4,
        sweep_param: Optional[str] = "Kz",
        sweep_values: Optional[List[float]] = None,
    ):
        """
        Run the complete workflow: ground state, excited states, 
        transfer learning, and parameter sweeps.
        """
        print("\n" + "=" * 70)
        print(" KITAEV HONEYCOMB IMPURITY WORKFLOW ".center(70, "="))
        print("=" * 70)
        
        # Save model configuration
        self.writer.write_metadata({
            "model_config": self.model_config.summary(),
            "training_config": self.training_config.__dict__,
            "architecture": architecture,
        })

        all_results = {}

        # Task 1: Ground state
        all_results["ground_state"] = self.run_ground_state(architecture)

        # Task 2: Excited states
        all_results["excited_states"] = self.run_excited_states(architecture, n_excited)

        # Task 3: Transfer learning with impurities
        all_results["transfer_learning"] = self.run_transfer_learning(architecture)

        # Task 4: Parameter sweep
        if sweep_param is not None:
            all_results["parameter_sweep"] = self.run_parameter_sweep(
                architecture, sweep_param, sweep_values
            )

        print("\n" + "=" * 70)
        print(" WORKFLOW COMPLETE ".center(70, "="))
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"HDF5 file: {self.writer.path}")
        
        return all_results


def main():
    """Command-line interface for the workflow."""
    parser = argparse.ArgumentParser(
        description="Kitaev honeycomb impurity workflow"
    )
    
    # Lattice parameters
    parser.add_argument("--lx", type=int, default=3, help="Lattice size in x")
    parser.add_argument("--ly", type=int, default=2, help="Lattice size in y")
    parser.add_argument("--bc", type=str, default="pbc", help="Boundary conditions")
    
    # Model parameters
    parser.add_argument("--Kx", type=float, default=1.0, help="Kitaev coupling K_x")
    parser.add_argument("--Ky", type=float, default=1.0, help="Kitaev coupling K_y")
    parser.add_argument("--Kz", type=float, default=1.0, help="Kitaev coupling K_z")
    parser.add_argument("--J", type=float, default=None, help="Heisenberg coupling")
    
    # Training parameters
    parser.add_argument("--architecture", type=str, default="rbm",
                       choices=["rbm", "autoregressive", "simple_conv"],
                       help="NQS architecture")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    
    # Workflow options
    parser.add_argument("--output", type=str, default="kitaev_results",
                       help="Output directory")
    parser.add_argument("--skip-ed", action="store_true",
                       help="Skip ED benchmark")
    parser.add_argument("--n-excited", type=int, default=4,
                       help="Number of excited states")
    parser.add_argument("--sweep-param", type=str, default="Kz",
                       help="Parameter to sweep")
    parser.add_argument("--sweep-values", type=str, default=None,
                       help="Sweep values as JSON list, e.g. '[0.0, 0.5, 1.0]'")
    
    args = parser.parse_args()
    
    # Build configurations
    model_config = ModelConfig(
        lx=args.lx,
        ly=args.ly,
        bc=args.bc,
        K={"x": args.Kx, "y": args.Ky, "z": args.Kz},
        J=args.J,
    )
    
    training_config = NQSTrainingConfig(
        architecture=args.architecture,
        n_epochs=args.epochs,
        learning_rate=args.lr,
    )
    
    # Parse sweep values
    sweep_values = None
    if args.sweep_values:
        sweep_values = json.loads(args.sweep_values)
    
    # Run workflow
    workflow = KitaevWorkflow(
        model_config=model_config,
        training_config=training_config,
        output_dir=Path(args.output),
        run_ed=not args.skip_ed,
    )
    
    workflow.run_full_workflow(
        architecture=args.architecture,
        n_excited=args.n_excited,
        sweep_param=args.sweep_param,
        sweep_values=sweep_values,
    )


if __name__ == "__main__":
    main()
