"""
Workflow: Variational Ground State Search (VMC vs ED)
=====================================================

This example demonstrates a complete variational workflow:
1. Define a quantum system (1D Transverse Field Ising Model).
2. Calculate the EXACT ground state energy (ED) for reference.
3. Define a Neural Quantum State (RBM) ansatz.
4. Optimize the ansatz using Variational Monte Carlo (VMC).
5. Compare the variational energy to the exact result.

This serves as a correctness sanity check for the VMC implementation.
"""
import sys
import os
import numpy as np
import jax

# Add Python paths
current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir, "Python")):
    sys.path.append(os.path.join(current_dir, "Python"))
    sys.path.append(os.path.join(current_dir, "Python", "QES"))

import QES
from QES.Algebra.hamil import Hamiltonian
from QES.Algebra.hilbert import HilbertSpace
from QES.general_python.lattices import SquareLattice

# Import NQS
try:
    from QES.NQS.nqs import NQS
except ImportError:
    print("NQS module not found. Skipping VMC part.")
    sys.exit(0)

def main():
    QES.qes_reseed(42)
    print("="*60)
    print("VMC vs ED Sanity Check: TFIM Chain (N=10)")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. System Definition
    # ------------------------------------------------------------------
    N = 10
    h = 1.0  # Critical point
    print(f"\n[1] Defining System (N={N}, h={h})...")

    lattice = SquareLattice(dim=1, lx=N, bc="pbc")
    hilbert = HilbertSpace(lattice=lattice, ns=N, is_manybody=True)
    ham = Hamiltonian(hilbert_space=hilbert, name="TFIM", dtype=np.complex128)
    ops = ham.operators

    # H = - sum Z_i Z_{i+1} - h sum X_i
    sz_corr = ops.sig_z(ns=N, type_act='correlation')
    sx = ops.sig_x(ns=N, type_act='local')

    for i in range(N):
        j = (i + 1) % N
        ham.add(sz_corr, sites=[i, j], multiplier=-1.0)
        ham.add(sx, sites=[i], multiplier=-h)

    ham.build() # Ensure matrix is built for ED

    # ------------------------------------------------------------------
    # 2. Exact Diagonalization (Reference)
    # ------------------------------------------------------------------
    print("\n[2] Computing Exact Ground State (ED)...")
    ham.diagonalize(k=1)
    E_exact = ham.eigenvalues[0].real
    print(f"    E_exact = {E_exact:.6f}")

    # ------------------------------------------------------------------
    # 3. NQS Setup & Optimization
    # ------------------------------------------------------------------
    print("\n[3] Initializing NQS (RBM) and Optimizing...")

    # Initialize NQS
    nqs = NQS(
        logansatz='rbm',
        model=ham,
        sampler='vmc',
        hilbert=hilbert,
        alpha=2.0,           # Hidden unit density
        n_chains=8,          # Parallel chains
        n_samples=500,       # Samples per step
        lr=0.05,             # Learning rate
        dtype=np.complex128,
        seed=42,
        backend='jax',
        use_orbax=False      # Disable checkpointing for example
    )

    print(f"    Ansatz: RBM with {nqs.num_params} parameters")

    # Train
    print("    Starting training (10 epochs)...")
    # Disable checkpointing (set larger than n_epochs) to avoid async issues in example
    stats = nqs.train(
        n_epochs=10,
        checkpoint_every=100,
        use_pbar=True,
        lr_scheduler='exponential', # Decay LR
        use_sr=True,                # Use Stochastic Reconfiguration
        diag_shift=0.01             # SR regularization
    )

    # ------------------------------------------------------------------
    # 4. Results Analysis
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Extract final energy from stats history
    # stats.history is a list of energy values per epoch
    if hasattr(stats, 'history') and len(stats.history) > 0:
        E_vmc = stats.history[-1]

        # stats.history_std contains standard deviation of local energies
        # We want standard error of the mean = std / sqrt(N_samples)
        # Note: stats.history_std might be empty or contain 0.0
        if hasattr(stats, 'history_std') and len(stats.history_std) > 0:
            std_dev = stats.history_std[-1]
            n_total = 8 * 500 # n_chains * n_samples
            E_err = std_dev / np.sqrt(n_total)
        else:
            E_err = 0.0
    else:
        E_vmc = -999.0
        E_err = 0.0
        print("Warning: Could not extract final energy from stats.")

    rel_error = abs(E_vmc - E_exact) / abs(E_exact)

    print(f"System: N={N} TFIM (h={h})")
    print(f"Exact Energy:       {E_exact:.6f}")
    print(f"Variational Energy: {E_vmc:.6f} +/- {E_err:.6f}")
    print(f"Relative Error:     {rel_error:.2e}")

    if rel_error < 1e-3:
        print("\nSUCCESS: VMC converged to within 0.1% of exact result.")
    elif rel_error < 1e-2:
        print("\nPASS: VMC converged to within 1% of exact result.")
    else:
        print("\nWARNING: VMC convergence was poor. Try more epochs/samples.")

if __name__ == "__main__":
    main()
