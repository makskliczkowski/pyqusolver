# QES Capabilities Checklist

## Verified & Working Features ✅

### Exact Diagonalization

- [x] Sparse Hamiltonian construction for spin systems
- [x] Eigenvalue/eigenvector computation via SciPy (Lanczos, ARPACK)
- [x] Numba-optimized matrix-vector products
- [x] Hamiltonian caching for repeated structures
- [x] Multiple solver backends (ARPACK, dense solvers)

### Lattice Geometries

- [x] 1D Chain (nearest-neighbor, periodic/open)
- [x] 2D Square lattice (4 neighbors, PBC/OBC)
- [x] 2D Hexagonal lattice (3 neighbors)
- [x] 2D Triangular lattice (6 neighbors)
- [x] 2D Honeycomb lattice (3 neighbors with sublattices)
- [x] Efficient neighbor finding
- [x] Boundary condition handling

### Quantum Spin Models

- [x] Transverse Field Ising Model (TFIM)
- [x] XXZ Model with anisotropy parameter Δ
- [x] Heisenberg Model (isotropic XXX)
- [x] J1-J2 Model (first + second nearest-neighbor)
- [x] Kitaev-Heisenberg hybrid model
- [x] Custom model framework (manual Hamiltonian construction)
- [x] Site-dependent coupling constants
- [x] Multiple fields (h_x, h_z) support

### Neural Quantum States

- [x] Restricted Boltzmann Machine (RBM) ansatz
- [x] Convolutional Neural Network (CNN) ansatz
- [x] Dense fully-connected networks
- [x] Custom Flax network integration
- [x] Complex-valued wavefunctions
- [x] Log-amplitude parameterization

### Sampling & Monte Carlo

- [x] Metropolis-Hastings Markov chain sampling
- [x] Gibbs sampling for RBM states
- [x] Configurable chain length, thermalization steps
- [x] Local energy estimation via samples
- [x] Error bars via sample variance
- [x] Parallel MCMC chains

### Training Methods

- [x] Stochastic gradient descent (SGD)
- [x] Adam optimizer
- [x] RMSprop optimizer
- [x] Learning rate scheduling (exponential decay, step-wise)
- [x] Early stopping with patience
- [x] Time-Dependent Variational Principle (TDVP)
- [x] Natural gradient descent via Fisher metric

### Observable Estimation

- [x] Local energy from samples
- [x] Expectation values via sampling
- [x] Two-point correlators
- [x] Magnetization (individual sites and total)
- [x] Structure factors
- [x] Custom observable evaluation

### Physics Utilities

- [x] Von Neumann entropy: $S(\rho) = -\text{Tr}(\rho \ln \rho)$
- [x] Shannon entropy: $H(p) = -\sum p_i \ln p_i$
- [x] Rényi entropy: $S_\alpha(\rho)$
- [x] Purity: $\text{Tr}(\rho^2)$
- [x] Partial trace (reduced density matrix)
- [x] Entanglement entropy via SVD
- [x] Correlation functions
- [x] Density matrix operations
- [x] Spectral properties

### Computational Backends

- [x] NumPy backend (CPU, exact arithmetic)
- [x] JAX backend (GPU, automatic differentiation)
- [x] Transparent backend switching
- [x] Backend-agnostic library code
- [x] Session management (seed, precision, backend)

### Development & Testing

- [x] Pytest test suite
- [x] Import hygiene checking
- [x] Type hints (partial)
- [x] Black code formatting
- [x] Sphinx documentation generation
- [x] Example notebooks/scripts

---

## Verified Working Features (by Test Suite)

### Symmetry Support

- [x] Translation symmetry (k-sectors)
- [x] Parity symmetry (ParityZ)
- [x] U(1) Particle conservation
- [x] Combined symmetries (translation + parity, translation + U(1), etc.)
- [x] Symmetry-reduced Hilbert space construction
- [x] Spectrum reconstruction from symmetry sectors

### Fermionic & Quadratic Systems

- [x] Fermionic systems via QuadraticHamiltonian class
- [x] Bogoliubov-de Gennes (BdG) systems (non-particle-conserving fermions)
- [x] Particle-conserving fermionic systems
- [x] Free fermion Hamiltonians with hopping and pairing terms
- [x] Slater determinant wavefunction calculations
- [x] Bogoliubov decomposition for diagonalization

### Parallelization

- ❌ No multi-GPU support
- ❌ Single GPU or CPU only
- ❌ No MPI parallelization across nodes
- ⚠️ Batch parallelization limited by GPU memory

### Optimization & Training

- ⚠️ TDVP matrix inversion numerically sensitive (documented)
- ⚠️ RBM limited expressibility for highly entangled states
- ⚠️ No excited state targeting via orthogonalization
- ⚠️ No variational optimization over multiple ansätze
- ⚠️ No automatic hyperparameter tuning

### Advanced Methods

- ❌ No Green's function computation via NQS
- ⚠️ Symmetry detection: manual specification (automatic detection not implemented)
- ⚠️ Custom symmetries require manual Hilbert space restriction

### Lattice Flexibility

- ⚠️ No automatic 3D lattice generation
- ⚠️ Custom lattice geometries require manual implementation
- ❌ No quasi-periodic or incommensurate structures

---

**Version**: 0.1.0 (Alpha) – API may evolve; core functionality stable.
