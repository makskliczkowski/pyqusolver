# NQS Configuration System

## Overview

The QES NQS module uses a centralized configuration system based on dataclasses to manage physical models and solver parameters. This ensures consistency across different components (scripts, notebooks, demos) and provides a high-level interface for complex tasks like network parameter estimation.

## 1. Physical Configuration: `NQSPhysicsConfig`

`NQSPhysicsConfig` defines the physical system, including the lattice and Hamiltonian parameters. It behaves like a dictionary, allowing dynamic argument addition.

### Key Fields

- `model_type`: The name of the quantum model (e.g., "kitaev", "xxz", "tfi").
- `lattice_type`: The lattice geometry ("honeycomb", "square", etc.).
- `lx`, `ly`, `lz`: Lattice dimensions.
- `bc`: Boundary conditions ("pbc" or "obc").
- `hx`, `hy`, `hz`: External magnetic fields.
- `impurities`: List of magnetic impurities.
- `args`: Dictionary for model-specific couplings.

### Methods

- `make_hamiltonian()`: Automatically constructs the `Lattice`, `HilbertSpace`, and `Hamiltonian` objects.
- `to_dict()`: Converts the config to a flat dictionary.

---

## 2. Solver Configuration: `NQSSolverConfig`

`NQSSolverConfig` manages hyperparameters for the NQS ansatz and the Variational Monte Carlo (VMC) optimization.

### Key Fields

- `ansatz`: The network architecture (e.g., "rbm", "cnn", "resnet", "ar").
- `n_chains`, `n_samples`: MCMC sampling parameters.
- `lr`, `epochs`: Training parameters.
- `dtype`: Numerical precision ("complex128", "float64").
- `backend`: Computation backend ("jax" or "numpy").

### SOTA Parameter Estimation

One of the most powerful features is the ability to automatically estimate state-of-the-art (SOTA) hyperparameters based on the physical system.

```python
s_cfg = NQSSolverConfig(ansatz='resnet')
# Estimate ResNet depth/width for a specific Kitaev system
s_cfg.estimate(physics_config) 
# Create the actual network
net = s_cfg.make_net(physics_config)
```

---

## 3. Usage Example

```python
from QES.NQS import NQSPhysicsConfig, NQSSolverConfig

# Define Physics
p_cfg = NQSPhysicsConfig(
    model_type='kitaev', 
    lx=4, ly=3, 
    hx=0.1, hy=0.1, hz=0.1
)
p_cfg.args['kz'] = 1.0

# Define Solver
s_cfg = NQSSolverConfig(ansatz='rbm', epochs=500)

# Create objects
hamil, hilbert, lattice = p_cfg.make_hamiltonian()
net = s_cfg.make_net(p_cfg)

# Initialize NQS
psi = NQS(logansatz=net, model=hamil)
psi.train()
```

## 4. Integration

These classes are designed to be imported and extended in research projects (e.g., `impurity_solver.py`), ensuring that only general configuration logic stays in the core QES module while specific experiment logic remains in the project directory.
