"""
QES Neural Quantum States (NQS)
===============================

Library for Variational Quantum Monte Carlo using 
Neural Networks, JAX, and Time-Dependent Variational Principles (TDVP).

Key Components:
---------------
1. NQS:         The quantum state wrapper connecting Neural Networks to Physics.
2. TDVP:        The physics engine solving the flow equations.
3. NQSTrainer:  The orchestrator managing phases, scheduling, and optimization.

Usage Example:
--------------
    >>> from QES.NQS import NQS, NQSTrainer, NetworkFactory
    >>> 
    >>> # 1. Create Network (RBM)
    >>> net = NetworkFactory.create('rbm', input_shape=(100,), alpha=2.0)
    >>> 
    >>> # 2. Define Physics
    >>> psi = NQS(ansatz=net, hamiltonian=my_hamiltonian)
    >>> 
    >>> # 3. Train (using 'kitaev' preset for frustrated systems)
    >>> trainer = NQSTrainer(psi, phases="kitaev")
    >>> trainer.train()

---------------------------------------------------------------------
File        : QES/NQS/__init__.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-01
---------------------------------------------------------------------
"""

import sys

# 1. Core Imports (Expose the API)
# We use try-except blocks to handle partial installations or lazy loading issues
try:
    from .nqs import NQS
except ImportError as e:
    raise ImportError(f"Could not import NQS module. Ensure QES is installed correctly.\nOriginal error: {e}")

try:
    # Direct access to the Trainer (The main entry point for users)
    from .src.nqs_train import NQSTrainer, NQSTrainStats
except ImportError:
    pass

try:
    from .src.nqs_network_integration import NetworkFactory, NetworkSelector
except ImportError:
    pass

try:
    # Direct access to the Physics Engine
    from .src.tdvp import TDVP, TDVPStepInfo
except ImportError:
    pass

# 2. Metadata
MODULE_DESCRIPTION  = "Neural Quantum States (NQS) with TDVP and Adaptive Scheduling."
__version__         = "2.0.0"

# 3. Helper Functions

def help():
    """Prints the library capability summary."""
    print(__doc__)

def quick_start(mode: str = 'ground'):
    """
    Prints a runnable boilerplate script to the console.
    Usage: QES.NQS.quick_start(mode='ground'|'excited')
    
    Parameters:
    -----------
    mode : str
        'ground' for ground state optimization (default)
    """
    boilerplate = """
# ==========================================
# QES NQS Quick Start Script
# ==========================================
import jax
import jax.numpy as jnp
from QES.NQS import NQS, NQSTrainer, NetworkFactory

# 0. Define Hamiltonian (Placeholder)
model = (...)  # Define your model here, it should expose function local_energy_fn
"""

    ground_body = """
# 1. Create Network
#    'alpha' is density of hidden units (N_hidden = alpha * N_visible)
net = NetworkFactory.create('rbm', input_shape=(16,), alpha=1.0)

# 2. Initialize NQS
psi = NQS(
    ansatz=net, 
    hamiltonian=MockHamiltonian(),
    n_visible=16
)

# 3. Initialize Trainer
#    'phases' automates Pre-training -> Main -> Refinement
trainer = NQSTrainer(
    nqs=psi,
    phases="kitaev", 
    n_batch=1024,
    ode_solver="rk4"
)

# 4. Run
print("Starting Ground State Optimization...")
stats = trainer.train()
print(f"Final Energy: {stats.history[-1]:.5f}")
"""

    excited_body = """
# 1. Load/Create Ground State (The Lower State)
#    In practice, you would load weights: psi_ground.load("ground_weights.h5")
net_g = NetworkFactory.create('rbm', input_shape=(16,), alpha=1.0)
psi_ground = NQS(net_g, MockHamiltonian(), n_visible=16)

# 2. Define Penalty Strength
#    This repels the new state from the ground state.
psi_ground.beta_penalty = 10.0 

# 3. Create New State (The Excited State)
#    Usually a different architecture or larger alpha
net_e = NetworkFactory.create('cnn', input_shape=(16,), features=(8, 8))
psi_excited = NQS(net_e, MockHamiltonian(), n_visible=16)

# 4. Initialize Trainer with Lower States
#    The trainer will automatically compute overlaps and add penalty forces.
trainer = NQSTrainer(
    nqs=psi_excited,
    lower_states=[psi_ground],  # <--- Critical for excited states
    phases="kitaev",
    n_batch=2048
)

# 5. Run
print("Starting Excited State Optimization...")
stats = trainer.train()
print(f"Final Energy (Excited): {stats.history[-1]:.5f}")
"""

    if mode.lower() == 'excited':
        print(boilerplate + excited_body)
    else:
        print(boilerplate + ground_body)

# --------------------------------------------------------------

# 5. Export
__all__ = [
    "NQS",
    "NQSTrainer",
    "NQSTrainStats",
    "TDVP",
    "TDVPStepInfo",
    "NetworkFactory",
    "NetworkSelector",
    "quick_start",
    "help"
]

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------