# Neural Quantum States (NQS) Module

Machine learning-based variational approach for finding quantum ground states and excited states through neural network ansätze.

## Overview

Neural Quantum States represent quantum wavefunctions using neural networks and optimize them via variational Monte Carlo (VMC) or Time-Dependent Variational Principle (TDVP). This module provides:

- **Network Architectures**: Restricted Boltzmann Machines (RBMs), Convolutional Neural Networks (CNNs), fully-connected dense networks
- **Sampling Methods**: Markov chain Monte Carlo (Metropolis-Hastings, Gibbs), importance sampling
- **Training Methods**: Stochastic gradient descent, natural gradient descent via TDVP
- **Observable Estimation**: Local energy, two-point correlators, magnetic moments

## Physical Background

### Variational Principle

For a parameterized wavefunction $|\psi(\theta)\rangle$:

$$E_{\text{var}}(\theta) = \frac{\langle \psi(\theta) | \hat{H} | \psi(\theta) \rangle}{\langle \psi(\theta) | \psi(\theta) \rangle} \geq E_0$$

where $E_0$ is the true ground state energy. Optimizing $\theta$ minimizes $E_{\text{var}}(\theta)$.

### Local Energy Estimator

For Monte Carlo sampling from $|\psi(\theta)|^2$:

$$\langle E \rangle = \frac{1}{M} \sum_{i=1}^M E_{\text{loc}}(s_i), \quad E_{\text{loc}}(s) = \frac{\langle s | \hat{H} | \psi(\theta) \rangle}{\langle s | \psi(\theta) \rangle}$$

### Neural Network Ansatz

The network computes the log-amplitude of the wavefunction:

$$\log \psi(s_1, \ldots, s_N; \theta) = \text{Network}(s_1, \ldots, s_N; \theta)$$

This ensures automatic normalizability and real-valued output.

## Core Components

### 1. Network Architectures

#### Restricted Boltzmann Machine (RBM)
- **Visible units**: Correspond to physical spins/qubits ($N$ sites)
- **Hidden units**: Learned feature detectors ($\alpha N$ units, $\alpha = 2$ typical)
- **Energy**: $E(v, h) = -a^T v - b^T h - v^T W h$
- **Advantages**: Analytical gradient computation, proven effective for many quantum states
- **Limitations**: Limited capacity for entangled states with $\alpha \lesssim 2$

```python
from QES.NQS.src.network_factory import NetworkFactory
import jax.numpy as jnp

# Create RBM with 16 visible, 32 hidden units
net = NetworkFactory.create(
    'rbm',
    input_shape=(16,),
    alpha=2.0,  # Hidden units = 2 × visible units
    dtype=jnp.complex64
)
```

#### Convolutional Neural Network (CNN)
- Exploits spatial structure of lattices
- Layers: Conv → ReLU/ELU → Pooling → Dense → Output
- Effective for translationally invariant systems
- Supports arbitrary spatial dimensions

```python
net = NetworkFactory.create(
    'cnn',
    input_shape=(64,),  # 8×8 lattice flattened
    reshape_dims=(8, 8),
    features=(16, 32),
    kernel_sizes=[(3, 3), (3, 3)],
    activations=['relu', 'relu'],
    dtype=jnp.float32
)
```

#### Fully-Connected Dense Network
- General-purpose, highest expressibility
- Many parameters → longer training, overfitting risk
- Suitable for small systems ($N \lesssim 12$ sites)

```python
net = NetworkFactory.create(
    'dense',
    input_shape=(16,),
    hidden_features=(64, 128, 64),
    output_features=1,
    activations=['relu', 'relu', 'relu']
)
```

### 2. Sampling & State Estimation

#### Markov Chain Monte Carlo
Sample quantum states from $|\psi(\theta)|^2$ via random walks:

```python
from QES.Solver.MonteCarlo.sampler import VMCSampler

sampler = VMCSampler(
    net=net,
    shape=(16,),  # System size
    n_chains=4,   # Parallel chains
    n_samples=1000,  # Samples per update
    therm_steps=500,  # Thermalization steps
    sweep_steps=1,    # Sweeps before accepting
    backend='jax'
)

# Sample states (shape: [n_chains × n_samples, 16])
states, log_psi = sampler.sample(params)
```

#### Local Energy Computation

$$E_{\text{loc}}(s) = \frac{\langle s | \hat{H} | \psi(\theta) \rangle}{\langle s | \psi(\theta) \rangle}$$

Estimated from sampled states:

```python
local_energies = model.local_energy(states, log_psi, params)
E_var = jnp.mean(local_energies)
E_std = jnp.std(local_energies) / jnp.sqrt(n_samples)
```

### 3. Training Methods

#### Stochastic Gradient Descent (SGD)
Standard update: $\theta_{t+1} = \theta_t - \eta \nabla E(\theta_t)$

```python
from QES.NQS.nqs_train import NQSTrainer
from optax import adam, exponential_decay

# Learning rate schedule
lr_schedule = exponential_decay(
    init_value=0.01,
    transition_steps=100,
    decay_rate=0.95
)

trainer = NQSTrainer(
    nqs=nqs,
    optimizer=adam(learning_rate=lr_schedule),
    batch_size=64,
    n_epochs=500
)

history = trainer.train()
print(f"Final energy: {history['energy'][-1]:.6f}")
```

#### Time-Dependent Variational Principle (TDVP)
Natural gradient descent using metric tensor $\mathcal{F}$:

$$\theta_{t+dt} = \theta_t - \mathcal{F}^{-1} \nabla E(\theta_t) \, dt$$

where $\mathcal{F}$ is the Quantum Fisher Information Matrix (QFIM):

$$\mathcal{F}_{ij} = \left\langle \frac{\partial \log \psi}{\partial \theta_i} \frac{\partial \log \psi}{\partial \theta_j} \right\rangle - \left\langle \frac{\partial \log \psi}{\partial \theta_i} \right\rangle \left\langle \frac{\partial \log \psi}{\partial \theta_j} \right\rangle$$

```python
from QES.NQS.tdvp import TDVP

tdvp = TDVP(
    nqs=nqs,
    S_matrix_threshold=1e-4,  # Regularization
    use_natural_gradient=True
)

# TDVP updates automatically use Fisher metric
history = trainer.train(method='tdvp', tdvp_solver=tdvp)
```

## Quick Start Examples

### Example 1: Basic NQS Training (SGD)

```python
import QES
from QES.Algebra.Model.Interacting.Spin import TransverseFieldIsing
from QES.general_python.lattices import SquareLattice
from QES.NQS import NQS

with QES.run(backend='jax', seed=42):
    # 1D chain: 10 spins
    lattice = SquareLattice(lx=10, ly=1)
    
    # Transverse Field Ising Model
    H = TransverseFieldIsing(
        lattice=lattice,
        j=1.0,
        hx=0.5,
        hz=0.0
    )
    
    # Create NQS with RBM ansatz
    nqs = NQS(
        model=H,
        ansatz='rbm',
        alpha=2.0,  # 20 hidden units
        seed=42
    )
    
    # Train with SGD
    results = nqs.train(
        n_epochs=300,
        batch_size=64,
        learning_rate=0.01,
        optimizer='adam',
        early_stopping_patience=30,
        verbose=True
    )
    
    print(f"Final variational energy: {results['energy_final']:.6f}")
    print(f"Standard error: {results['energy_std']:.6f}")
```

### Example 2: NQS with TDVP (Natural Gradient)

```python
import jax.numpy as jnp
from QES.NQS.nqs_train import NQSTrainer
from QES.NQS.tdvp import TDVP

with QES.run(backend='jax', seed=42):
    # ... Setup model and NQS as above ...
    
    # Initialize TDVP solver
    tdvp = TDVP(
        nqs=nqs,
        S_matrix_threshold=1e-3,  # Tikhonov regularization
        method='moore_penrose'
    )
    
    # Create trainer with TDVP
    trainer = NQSTrainer(
        nqs=nqs,
        tdvp=tdvp,
        batch_size=128,
        learning_rate=0.001,
    )
    
    # Train with natural gradient (typically faster convergence)
    history = trainer.train(n_epochs=200)
    
    print(f"Best energy found: {min(history['energy']):.6f}")
```

### Example 3: Custom Network Architecture

```python
import flax.linen as nn
from QES.general_python.ml.net_impl import FlaxInterface

# Define custom Flax network
class CustomNet(nn.Module):
    """Hybrid RBM-CNN network"""
    hidden_features: int = 32
    
    @nn.compact
    def __call__(self, x):
        # RBM layer
        x = nn.Dense(self.hidden_features, use_bias=True)(x)
        x = nn.sigmoid(x)
        
        # Global pooling and output
        x = jnp.mean(x, axis=-1, keepdims=True)
        return x.squeeze(-1)

# Wrap and use
net = FlaxInterface(
    net_module=CustomNet,
    input_shape=(16,),
    backend='jax',
    dtype=jnp.complex64
)

nqs = NQS(
    model=H,
    ansatz=net,
    seed=42
)

results = nqs.train(n_epochs=500)
```

### Example 4: Observable Estimation

```python
import jax.numpy as jnp

# After training...
# Compute expectation values

# Magnetization (z-component)
Sz_expectation = nqs.expectation_value(
    observable='Sz',
    samples=sampled_states
)

# Two-point correlations: ⟨σ^z_i σ^z_j⟩
correlation_matrix = nqs.correlation_function(
    operator='Sz',
    operator_j='Sz',
    sample_size=5000
)
print(f"Nearest-neighbor correlation: {correlation_matrix[0, 1]:.6f}")
```

## Configuration & Tuning

### Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| **Learning Rate** | $10^{-3}$ – $10^{-1}$ | Start high, decay over time |
| **Batch Size** | 32–512 | Larger = smoother, slower per-update |
| **Hidden Units (α)** | 1–4 | α=1–2 for 1D chains; α≥2 for 2D |
| **Thermalization Steps** | 500–5000 | Ensure Markov chain convergence |
| **Number of Samples** | 1000–10000 | More samples → lower variance |

### Debugging Convergence Issues

```python
# 1. Check if local energy has large variance
print(f"Energy variance: {E_std**2:.6f}")
if E_std**2 > 1.0:
    # Increase sample size or improve network capacity
    sampler.n_samples *= 2

# 2. Monitor gradient norms
grad_norms = [jnp.linalg.norm(grad) for grad in grads]
print(f"Max gradient: {jnp.max(grad_norms):.6f}")

# 3. Check for NaN/Inf
if jnp.isnan(E_var) or jnp.isinf(E_var):
    # Reduce learning rate
    lr *= 0.5
```

## Limitations & Current Capabilities

**Verified Working**:
- TFIM and XXZ on 1D/2D lattices (6–16 sites)
- RBM and CNN ansätze
- VMC with Metropolis-Hastings and Gibbs sampling
- Early stopping, learning rate scheduling

**Known Limitations**:
- No multi-GPU support (single GPU only)
- TDVP convergence sensitive to regularization parameter
- RBM expressibility limited for highly entangled states
- Fermion systems not yet supported

**Not Yet Implemented**:
- Excited state targeting via orthogonalization
- Imaginary-time evolution
- Multi-flavor fermion systems

## References

1. Carleo & Troyer (2017): "Solving the quantum many-body problem with artificial neural networks" – *Nature Physics* 13, 435–441
2. Choo et al. (2020): "Efficient neural quantum states on loops" – *Physical Review X* 10, 021014  
3. Stokes et al. (2020): "Quantum Fisher information and natural gradient learning" – *Physical Review A* 85, 062315
4. Melko et al. (2019): "Restricted Boltzmann machines as effective descriptions of quantum many-body states" – *Nature Reviews Physics* 3, 856–880

## Contributing

Contributions welcome. Please ensure:
1. Code follows PEP 8 (Black formatter)
2. All training converges or includes warnings
3. New architectures validated on benchmark systems
4. Unit tests added for new features

## License

CC-BY-4.0 – See root [LICENSE.md](../../../../LICENSE.md)
