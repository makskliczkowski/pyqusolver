# Quick Start

Welcome to the fast track! This guide will help you solve your first quantum problem with QES in minutes. We'll focus on a **Quadratic Hamiltonian** (free fermions), which is a great starting point because it's fast and exact.

## Your First Simulation

Let's calculate the ground state energy of a simple 1D chain of fermions with hopping and onsite energy.

### 1. Import QES

First, bring in the necessary tools.

```python
import numpy as np
from QES.Algebra import QuadraticHamiltonian
```

### 2. Define the Hamiltonian

We'll create a system with 4 sites. We will add:
*   **Hopping** ($t = -1.0$) between neighboring sites.
*   **Onsite energy** ($\mu = 0.5$) on the first site.

```python
# Create a Hamiltonian for 4 sites, conserving particle number
qh = QuadraticHamiltonian(ns=4, particle_conserving=True)

# Add nearest-neighbor hopping (periodic boundary if we closed the loop, but let's do open chain)
# Site 0 <-> 1, 1 <-> 2, 2 <-> 3
qh.add_hopping(0, 1, -1.0)
qh.add_hopping(1, 2, -1.0)
qh.add_hopping(2, 3, -1.0)

# Add a potential to site 0
qh.add_onsite(0, 0.5)
```

### 3. Solve It!

Now, we diagonalize the Hamiltonian to find its single-particle energy levels.

```python
qh.diagonalize()

print("Single-particle eigenvalues:")
print(qh.eig_val)
```

### 4. Calculate Many-Body Energy

With the single-particle spectrum known, we can calculate the energy of a specific many-body state. Let's fill the two lowest energy levels (half-filling).

```python
# Indices of the occupied orbitals (0 and 1 are the lowest energy states)
occupied_orbitals = [0, 1]

energy = qh.many_body_energy(occupied_orbitals)
print(f"Ground state energy (half-filling): {energy}")
```

## Going Further: Superconductivity

QES easily handles non-particle-conserving systems like superconductors using the Bogoliubov-de Gennes (BdG) formalism.

```python
# Set particle_conserving=False to enable pairing terms
bdg_ham = QuadraticHamiltonian(ns=4, particle_conserving=False)

# Add hopping
for i in range(3):
    bdg_ham.add_hopping(i, i+1, -1.0)

# Add superconducting pairing (Delta = 0.5)
for i in range(3):
    bdg_ham.add_pairing(i, i+1, 0.5)

bdg_ham.diagonalize()
print("BdG Quasiparticle Spectrum:")
print(bdg_ham.eig_val)
```

## What Just Happened?

1.  **Object Creation**: You instantiated a `QuadraticHamiltonian`. This object manages the matrix representation of your physics problem.
2.  **Model Definition**: You used helper methods like `add_hopping` to describe the physics term-by-term.
3.  **Diagonalization**: QES solved the eigenvalue problem under the hood using efficient linear algebra routines (NumPy or JAX).
4.  **Analysis**: You queried the results to get real physical quantities.

## Next Steps

Now that you've got the basics, check out the :doc:`Examples <examples>` section for more complex scenarios, including interacting spin models and neural network optimizations!
