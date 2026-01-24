# Quickstart

This guide covers the two most common workflows: Exact Diagonalization (ED) and Variational Monte Carlo (VMC) with Neural Quantum States (NQS).

## 1. Exact Diagonalization (ED)

Use ED for small systems or non-interacting (quadratic) systems where you need the exact ground state energy and spectrum.

Here we solve a **free fermion chain** (Tight-Binding Model) which can be solved efficiently ($O(N^3)$) even for large $N$.

```python
import numpy as np
import QES
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian

# 1. Start a session (handles backend/precision)
with QES.run(backend='numpy'):

    # 2. Define a Quadratic Hamiltonian
    # ns=20 sites
    H = QuadraticHamiltonian(ns=20)

    # Add hopping: H = -t sum (c_i^dag c_{i+1} + h.c.)
    for i in range(19):
        H.add_hopping(i, i+1, -1.0)

    # 3. Diagonalize
    # This computes single-particle eigenvalues
    H.diagonalize()

    # Ground state energy is sum of negative eigenvalues (filled Fermi sea)
    E_gs = sum(val for val in H.eig_val if val < 0)
    print(f"Ground State Energy: {E_gs:.6f}")
```

## 2. Neural Quantum States (VMC)

Use VMC/NQS for large interacting systems where ED is intractable. This example trains an RBM ground state for the **Transverse Field Ising Model**.

```python
import QES
from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
from QES.general_python.lattices.square import SquareLattice
from QES.NQS import NQS

# 1. Use JAX backend for training (GPU acceleration)
with QES.run(backend='jax', seed=42):

    # 2. Define Model
    # 1D Chain with 20 sites, Periodic Boundary Conditions
    lat = SquareLattice(dim=1, lx=20, bc='pbc')
    H = TransverseFieldIsing(lattice=lat, j=1.0, hx=0.5)

    # 3. Initialize NQS Solver
    # ansatz='rbm' creates a Restricted Boltzmann Machine
    # alpha=1 sets hidden density (nh = alpha * ns)
    psi = NQS(
        logansatz='rbm',
        model=H,
        alpha=1,
        sampler='vmc',  # Standard Metropolis
        batch_size=1024
    )

    # 4. Train
    # Uses Stochastic Reconfiguration (SR) by default
    stats = psi.train(
        n_epochs=100,
        lr=0.01,
        n_batch=1000
    )

    print(f"Final Energy: {stats.loss_mean[-1]:.6f}")
```

## Next Steps

- **Concepts**: Learn about the [Global State and Backends](concepts.md).
- **Modules**: Explore [Project Structure](modules.md).
- **Examples**: Check the `examples/` folder for advanced scripts (symmetries, custom networks, TDVP).
