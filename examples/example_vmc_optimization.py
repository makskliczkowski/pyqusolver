"""
Example: Variational Monte Carlo (VMC) Optimization
===================================================

This example demonstrates how to:
1. Define a Lattice and Hilbert Space.
2. Define a Hamiltonian (Heisenberg model).
3. Create a Neural Quantum State (NQS) ansatz (RBM).
4. Set up the VMC Sampler.
5. Initialize the VMC driver and run an optimization loop.
"""
import sys
import os
import numpy as np

# Ensure QES is in path if running from root
current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))

import QES
from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice
try:
    from QES.general_python.qes_globals import QES_GLOBALS
except ImportError:
    try:
        from QES.qes_globals import QES_GLOBALS
    except ImportError:
        pass

# Import NQS components
try:
    from QES.NQS.nqs import NQS
except ImportError as e:
    print(f"Skipping VMC example: Required NQS modules not found. ({e})")
    sys.exit(0)

def main():
    QES.qes_reseed(42)
    print("="*60)
    print("VMC Optimization: Heisenberg Chain (N=10)")
    print("="*60)

    # 1. Setup System
    ns = 10
    lattice = SquareLattice(dim=1, lx=ns, bc="pbc")
    # VMC usually works with spin-1/2 (local dim 2)
    hilbert = HilbertSpace(lattice=lattice, ns=ns, is_manybody=True)

    print(f"System: {ns} sites")

    # 2. Hamiltonian (Heisenberg)
    # Using real parameters for simplicity
    ham = Hamiltonian(hilbert_space=hilbert, name="Heisenberg", dtype=np.complex128)
    ops = ham.operators

    # Standard Heisenberg: H = J sum S_i . S_{i+1} = J/4 sum sigma_i . sigma_{i+1}
    J = 1.0
    J_eff = 4.0 * J # Effective coupling for Pauli matrices

    sx = ops.sig_x(ns=ns, type_act='correlation')
    sy = ops.sig_y(ns=ns, type_act='correlation')
    sz = ops.sig_z(ns=ns, type_act='correlation')

    for i in range(ns):
        j = (i + 1) % ns
        coeff = 0.25 * J_eff
        ham.add(sx, sites=[i, j], multiplier=coeff)
        ham.add(sy, sites=[i, j], multiplier=coeff)
        ham.add(sz, sites=[i, j], multiplier=coeff)

    ham.build() # Build matrix/operator structure

    # 3. Initialize NQS (Driver + Ansatz + Sampler)
    # NQS class handles network creation ('rbm'), sampler ('vmc'), and optimization loop.
    print("Initializing NQS (RBM + VMC)...")

    try:
        nqs = NQS(
            logansatz='rbm',                # Use built-in RBM
            model=ham,                      # Physical model
            sampler='vmc',                  # Use VMC sampler
        hilbert=hilbert,                # Explicitly pass Hilbert space
            alpha=2.0,                      # RBM hidden density
            n_chains=10,                    # Number of MCMC chains
            n_samples=1000,                 # Samples per step
            lr=0.01,                        # Learning rate
        dtype=np.complex128,            # Complex wavefunction
        seed=42,                        # Explicit seed for JAX RNG
        backend='jax'                   # Force JAX backend
        )

        # 4. Run Optimization
        print("\nStarting optimization loop (5 epochs)...")

        # Train for 5 epochs
        # Disable checkpointing to avoid async I/O issues in example
        stats = nqs.train(n_epochs=5, checkpoint_every=100)

        # Explicitly close/wait for checkpoint manager if needed
        if hasattr(nqs, 'ckpt_manager') and hasattr(nqs.ckpt_manager, 'wait_until_finished'):
             nqs.ckpt_manager.wait_until_finished()

        print("\nOptimization finished.")

        # Access history from stats object
        if hasattr(stats, 'history') and 'Energy' in stats.history:
            final_E = stats.history['Energy'][-1]
            final_Var = stats.history['EnergyVariance'][-1]
            print(f"Final Energy: {final_E:.6f} +/- {np.sqrt(final_Var/1000):.6f}")
        else:
            print("Training completed (stats format not automatically parsed).")

        # Ground state energy for N=10 Heisenberg is approx -1.8019 per site * 10 = -18.019 (if J=4 scaling used)
        # Since we scaled J_eff=4, we expect approx -18.0.

    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
