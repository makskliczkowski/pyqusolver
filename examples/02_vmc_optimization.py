"""
VMC Optimization Example
========================

This example demonstrates how to train a Neural Quantum State (NQS)
to find the ground state of a Hamiltonian using Variational Monte Carlo.

Model: Restricted Boltzmann Machine (RBM)
System: Heisenberg-Kitaev model on Honeycomb lattice

To run:
    python examples/02_vmc_optimization.py
"""

import QES
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.NQS.nqs import NQS
from QES.general_python.ml.net_impl.networks.net_rbm import RBM
import numpy as np

def main():
    # Initialize logger
    logger = QES.get_logger()
    logger.info("Starting VMC Optimization Example...")

    # 1. Setup Physical System
    # Small 2x2 Honeycomb lattice (8 sites)
    lattice = HoneycombLattice(dim=2, lx=2, ly=2, bc="pbc")

    # Use complex128
    hilbert = HilbertSpace(lattice=lattice, is_manybody=True, dtype=np.complex128)

    # Heisenberg-Kitaev Hamiltonian
    hamiltonian = HeisenbergKitaev(
        lattice=lattice,
        hilbert_space=hilbert,
        J=0.1,
        K=1.0,
        hz=0.0,
        dtype=np.complex128
    )

    # 2. Define Neural Network Ansatz (RBM)
    # Input shape corresponds to number of sites
    n_visible = lattice.ns
    # Simple RBM with alpha=2 (n_hidden = 2 * n_visible)
    rbm_net = RBM(
        input_shape=(n_visible,),
        n_hidden=n_visible * 2,
        param_dtype="complex128", # Use complex parameters for general wavefunction
        dtype="complex128"
    )

    # 3. Initialize NQS Solver
    # Uses default VMCSampler
    # Must pass hilbert explicitly as NQS requires it for initialization checks
    # RBM requires JAX backend
    nqs = NQS(
        logansatz=rbm_net,
        model=hamiltonian,
        hilbert=hilbert,
        batch_size=1000,
        seed=42,
        backend='jax',
        # Optimization settings
        use_orbax=False # Disable checkpointing for simple example
    )

    logger.info(f"NQS initialized with {nqs.num_params} parameters.")

    # 4. Train
    # Using Stochastic Reconfiguration (SR) which is standard for VMC
    print("\nStarting training loop (10 epochs)...")
    stats = nqs.train(
        n_epochs=10,
        lr=0.05,            # Learning rate
        n_batch=1000,       # Samples per step
        use_sr=True,        # Use Stochastic Reconfiguration
        diag_shift=0.01,    # SR regularization
        use_pbar=True       # Show progress bar
    )

    # 5. Results
    final_energy = stats.history[-1]
    print(f"\nFinal Energy: {final_energy:.6f}")

    # Optional: Compare with exact diagonalization (if system small enough)
    if lattice.ns <= 12:
        print("Computing exact ground state energy for comparison...")
        hamiltonian.diagonalize(k=1, method='exact')
        exact_E = hamiltonian.eigenvalues[0]
        print(f"Exact Energy: {exact_E:.6f}")
        print(f"Error: {abs(final_energy - exact_E):.6f}")

if __name__ == "__main__":
    main()
