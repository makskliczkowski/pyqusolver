#!/usr/bin/env python3
"""
Simple example demonstrating the Kitaev workflow.

This script runs a minimal calculation on a 2×2 lattice to verify
everything is working correctly.

Usage:
    python simple_example.py
"""

from pathlib import Path
from kitaev_project.models import ModelConfig, KitaevModelBuilder
from kitaev_project.nqs import NQSTrainingConfig
from kitaev_project.workflows.main_workflow import KitaevWorkflow

def main():
    print("\n" + "="*70)
    print(" KITAEV HONEYCOMB - SIMPLE EXAMPLE ".center(70))
    print("="*70 + "\n")

    # Configuration
    print("⚙️  Configuration:")
    
    model_config = ModelConfig(
        lx=2,
        ly=2,
        bc="pbc",
        K={"x": 1.0, "y": 1.0, "z": 1.0},  # Isotropic Kitaev
        J=None,  # No Heisenberg
    )
    
    training_config = NQSTrainingConfig(
        architecture="rbm",
        n_epochs=50,  # Short run for demo
        learning_rate=0.01,
    )
    
    output_dir = Path("example_output")
    
    print(f"  Lattice: {model_config.lx}×{model_config.ly} ({2*model_config.lx*model_config.ly} spins)")
    print(f"  Couplings: Kx={model_config.K['x']}, Ky={model_config.K['y']}, Kz={model_config.K['z']}")
    print(f"  Architecture: {training_config.architecture}")
    print(f"  Epochs: {training_config.n_epochs}")
    print(f"  Output: {output_dir}/")
    
    # Initialize workflow
    print(f"\n🔧 Initializing workflow...")
    workflow = KitaevWorkflow(
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
        run_ed=True,
    )
    
    # Run ground state calculation
    print("\n🚀 Running ground state calculation...")
    print("   (This will take ~30 seconds)\n")
    
    results = workflow.run_ground_state(architecture="rbm")
    
    # Display results
    print("\n" + "="*70)
    print(" RESULTS ".center(70))
    print("="*70 + "\n")
    
    if "nqs" in results:
        nqs_energy = results["nqs"].observables.get("energy", float("nan"))
        print(f"NQS Energy:    {nqs_energy}")
    
    if "ed" in results:
        ed_energy = results["ed"].energies[0]
        print(f"ED Energy:     {ed_energy:.10f}")
        
        if not isinstance(nqs_energy, str) and nqs_energy == nqs_energy:  # Check if not NaN
            import numpy as np
            rel_error = abs(nqs_energy - ed_energy) / abs(ed_energy)
            print(f"Relative Err:  {rel_error:.2e}")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   - results.h5  (HDF5 data)")
    
    print("\n" + "="*70)
    print(" EXAMPLE COMPLETE ✓ ".center(70))
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Check GETTING_STARTED.md for more examples")
    print("  2. Run the Jupyter notebook: notebooks/kitaev_nqs_vs_ed.ipynb")
    print("  3. Try different architectures: --architecture autoregressive")
    print("  4. Increase system size: --lx 3 --ly 2")
    print("  5. Add impurities or sweep parameters\n")

if __name__ == "__main__":
    main()
