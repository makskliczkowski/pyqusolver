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
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from QES.NQS import NQS, NQSTrainer, NetworkFactory, VMCSampler
    >>> 
    >>> # Mock Hamiltonian (replace with your actual model)
    >>> class MockHamiltonian:
    ...     def __init__(self, ns): self.ns = ns
    ...     def local_energy_fn(self, state, params, log_psi_func): return jnp.array(0.0) # Placeholder
    ...     @property
    ...     def shape(self): return (16,)
    >>> model = MockHamiltonian(ns=16)
    >>> 
    >>> # 1. Create Network (RBM)
    >>> net = NetworkFactory.create('rbm', input_shape=(model.ns,), alpha=2.0)
    >>> 
    >>> # 2. Create Sampler
    >>> sampler = VMCSampler(
    ...     net         =   net, 
    ...     shape       =   (model.ns,), 
    ...     rng         =   jax.random, 
    ...     rng_k       =   jax.random.PRNGKey(0), 
    ...     numchains   =   1, 
    ...     numsamples  =   10, 
    ...     therm_steps =   100, 
    ...     sweep_steps =   1, 
    ...     backend     =   'jax'
    ... )
    >>> 
    >>> # 3. Define NQS Physics
    >>> psi = NQS(net=net, sampler=sampler, model=model)
    >>> 
    # 4. Train (using 'kitaev' preset for frustrated systems)
    >>> trainer = NQSTrainer(nqs=psi, phases="kitaev")
    >>> # stats = trainer.train() # Uncomment to run actual training
    >>>
    >>> # Or use the convenient train() method directly on NQS:
    >>> # stats = psi.train(n_epochs=300, lr=1e-2, lr_scheduler='cosine')
    >>> # stats = psi.train(n_epochs=100, override=False)  # Continue training

Ground State Optimization Example:
----------------------------------
    import jax
    import jax.numpy as jnp
    from QES.NQS import NQS, NQSTrainer, NetworkFactory, VMCSampler

    # Define Hamiltonian (must provide local_energy_fn)
    model = Hamiltonian(...)  # Replace with actual Hamiltonian

    # 1. Create Network
    net = NetworkFactory.create('rbm', input_shape=(16,), alpha=1.0)

    # 2. Create Sampler
    sampler = VMCSampler(
        net         =   net,
        shape       =   (16,),
        rng         =   jax.random,
        rng_k       =   jax.random.PRNGKey(0),
        numchains   =   1,
        numsamples  =   10,
        therm_steps =   100,
        sweep_steps =   1,
        backend     =   'jax'
    )

    # 3. Initialize NQS
    psi = NQS(net=net, sampler=sampler, model=model, shape=(16,)) # Sampler is now included

    # 4. Initialize Trainer
    trainer = NQSTrainer(nqs=psi, phases="kitaev", n_batch=1024, ode_solver="rk4")

    # 5. Run
    print("Starting Ground State Optimization...")
    # stats = trainer.train() # Uncomment to run actual training
    # print(f"Final Energy: {stats.history[-1]:.5f}")
    
    # Alternative: Use train() method directly on NQS
    # stats = psi.train(n_epochs=300, lr=1e-2, lr_scheduler='cosine', use_sr=True)
    # stats = psi.train(n_epochs=100, override=False)  # Continue with same trainer
    
    # 6. Save/Load Checkpoints
    # psi.save_weights("ground_state.h5")  # Save network weights
    # psi.load_weights("ground_state.h5")  # Load network weights
    # Trainer checkpoints are saved automatically every checkpoint_every epochs

Excited State Optimization Example:
-----------------------------------
    import jax
    import jax.numpy as jnp
    from QES.NQS import NQS, NQSTrainer, NetworkFactory, VMCSampler

    # Mock Hamiltonian (replace with your actual model)
    model = Hamiltonian(...)  # Replace with actual Hamiltonian
    
    # 1. Load/Create Ground State (The Lower State)
    #    In practice, you would load weights: psi_ground.load("ground_weights.h5")
    net_g       = NetworkFactory.create('rbm', input_shape=(16,), alpha=1.0)
    sampler_g   = VMCSampler(
        net         =   net_g,
        shape       =   (16,),
        rng         =   jax.random,
        rng_k       =   jax.random.PRNGKey(0),
        numchains   =   1,
        numsamples  =   10,
        therm_steps =   100,
        sweep_steps =   1,
        backend     =   'jax'
    )
    psi_ground              = NQS(net=net_g, sampler=sampler_g, model=model, shape=(16,))
    psi_ground.beta_penalty = 10.0 

    # 2. Create New State (The Excited State)
    #    Usually a different architecture or larger alpha
    net_e       = NetworkFactory.create('cnn', input_shape=(16,), features=(8, 8))
    sampler_e   = VMCSampler(
        net         =   net_e,
        shape       =   (16,),
        rng         =   jax.random,
        rng_k       =   jax.random.PRNGKey(1), # Use a different key for the excited state sampler
        numchains   =   1,
        numsamples  =   10,
        therm_steps =   100,
        sweep_steps =   1,
        backend     =   'jax'
    )
    psi_excited = NQS(net=net_e, sampler=sampler_e, model=model, shape=(16,))

    # 3. Initialize Trainer with Lower States
    #    The trainer will automatically compute overlaps and add penalty forces.
    trainer     = NQSTrainer(
                nqs             =   psi_excited,
                lower_states    =   [psi_ground],  # <--- Critical for excited states
                phases          =   "kitaev",
                n_batch         =   2048
                )

    # 4. Run
    print("Starting Excited State Optimization...")
    # stats = trainer.train() # Uncomment to run actual training
    # print(f"Final Energy (Excited): {stats.history[-1]:.5f}")
    
    # Alternative: Use train() method with lower_states
    # stats = psi_excited.train(
    #     n_epochs=300,
    #     lower_states=[psi_ground],
    #     lr=1e-2,
    #     lr_scheduler='cosine'
    # )
    
    # 5. Save/Load
    # psi_excited.save_weights("excited_state.h5")

Checkpoint Management:
----------------------
    # Manual Save/Load (weights only)
    psi.save_weights("checkpoint.h5")
    psi.load_weights("checkpoint.h5")
    
    # Automatic checkpoints during training
    stats = psi.train(
        n_epochs=300,
        checkpoint_every=50,      # Save every 50 epochs
        save_path="./checkpoints" # Directory for checkpoints
    )
    
    # Training creates files like:
    # ./checkpoints/epoch_50.h5, epoch_100.h5, ...
    
    # Resume training from checkpoint
    psi.load_weights("./checkpoints/epoch_200.h5")
    stats = psi.train(n_epochs=100)  # Continue from loaded state

---------------------------------------------------------------------
File        : QES/NQS/__init__.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-01
---------------------------------------------------------------------
"""

import sys

# Core Imports (Expose the API)
# We use try-except blocks to handle partial installations or lazy loading issues
try:
    from .nqs import NQS, NQSEvalEngine, NQSSingleStepResult, NQSObservable, NQSLoss
except ImportError as e:
    raise ImportError(f"Could not import NQS module. Ensure QES is installed correctly.\nOriginal error: {e}")

try:
    from QES.Solver.MonteCarlo.vmc import VMCSampler
except ImportError:
    pass

try:
    # Direct access to the Trainer (The main entry point for users)
    from .src.nqs_train import NQSTrainer, NQSTrainStats
except ImportError:
    pass

try:
    from .src.nqs_network_integration import NetworkFactory
except ImportError:
    pass

try:
    # Direct access to the Physics Engine
    from .src.tdvp import TDVP, TDVPStepInfo
except ImportError:
    pass

# Metadata
MODULE_DESCRIPTION  = "Neural Quantum States (NQS) with TDVP and Adaptive Scheduling."
__version__         = "2.0.0"

# Helper Functions

def info():
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

# Alternative: Use psi.train() directly
# stats = psi.train(n_epochs=300, lr=1e-2, lr_scheduler='cosine')
# psi.save_weights("ground_state.h5")  # Save weights
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

# Alternative: Use psi.train() with lower_states
# stats = psi_excited.train(n_epochs=300, lower_states=[psi_ground], lr=1e-2)
# psi_excited.save_weights("excited_state.h5")
"""

    if mode.lower() == 'excited':
        print(boilerplate + excited_body)
    else:
        print(boilerplate + ground_body)

# --------------------------------------------------------------

# Export
__all__ = [
    "NQS",
    "NQSTrainer",
    "NQSTrainStats",
    "TDVP",
    "TDVPStepInfo",
    "NetworkFactory",
    "quick_start",
    "info"
]

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------