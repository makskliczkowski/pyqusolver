# Physics Background

This section provides a brief overview of the physical concepts and mathematical formalisms used within QES.

## The Quantum Eigenvalue Problem

The central task of QES is to solve the time-independent Schrödinger equation:

$$ \hat{H} |\Psi\rangle = E |\Psi\rangle $$

where:
*   $\hat{H}$ is the **Hamiltonian operator** representing the total energy of the system.
*   $|\Psi\rangle$ is the **wavefunction** (eigenvector).
*   $E$ is the **energy** (eigenvalue).

Usually, we are interested in the **ground state** (the state with the lowest energy, $E_0$) and low-lying excited states.

## Computational Bases

QES supports multiple ways to represent the wavefunction, depending on the physics:

### 1. Particle Number Basis (Fock Space)
Used for fermionic or bosonic systems where particle number is conserved. A state is represented by the occupation numbers of orbitals:
$$ |n_1, n_2, \dots, n_L\rangle $$

### 2. Spin Basis
Used for magnetic systems (e.g., Heisenberg model). For spin-1/2, each site is either up ($\uparrow$) or down ($\downarrow$).

### 3. Bogoliubov Basis
Used for non-interacting (quadratic) systems, including superconductors. The problem is mapped to a basis of independent "quasiparticles" via a Bogoliubov transformation.

## Solution Methods

### Exact Diagonalization (ED)
For small systems, we can construct the full matrix representation of $\hat{H}$ and diagonalize it numerically.
*   **Pros**: Exact results, access to all eigenstates.
*   **Cons**: Exponential scaling. Memory limits usually restrict this to $N \approx 20-30$ spins.

### Quadratic Solvers
For non-interacting systems (Hamiltonians quadratic in creation/annihilation operators), the problem simplifies drastically. We only need to diagonalize an $N \times N$ matrix (or $2N \times 2N$ for BdG) instead of $2^N \times 2^N$.
*   **Pros**: Extremely fast, handles $N \sim 1000s$.
*   **Cons**: Only applies to non-interacting models (or mean-field approximations).

### Neural Quantum States (NQS)
For large interacting systems, we use a Variational Monte Carlo approach. We approximate the wavefunction $|\Psi\rangle$ with a neural network parameterized by weights $W$:
$$ \Psi_W(x) = \langle x | \Psi_W \rangle $$

We minimize the variational energy:
$$ E(W) = \frac{\langle \Psi_W | \hat{H} | \Psi_W \rangle}{\langle \Psi_W | \Psi_W \rangle} $$

*   **Pros**: Can handle very large systems ($N > 100$), flexible ansatz.
*   **Cons**: Approximate solution, optimization can be tricky.

## Symmetries

Symmetries are crucial for reducing the complexity of the problem. If an operator $\hat{S}$ commutes with the Hamiltonian ($[\hat{H}, \hat{S}] = 0$), we can block-diagonalize $\hat{H}$ and solve the problem in smaller "symmetry sectors."

QES supports:
*   **Translation**: Momentum sectors ($k$).
*   **Point Groups**: Reflection, Inversion, Rotation.
*   **Internal Symmetries**: Spin parity ($Z_2$), Particle conservation ($U(1)$).
