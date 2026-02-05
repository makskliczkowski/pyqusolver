"""
QES Neural Quantum States (NQS)
===============================

Variational methods using Neural Networks and JAX.

This module implements the Variational Monte Carlo (VMC) method using
neural network ansatzes. It supports ground state search, time evolution
(TDVP), and excited state search.

Entry Points
------------
- :class:`NQS`: The main solver class combining network, sampler, and model.
- :class:`NQSTrainer`: Advanced training loop manager.

Flow
----
::

    Hamiltonian (from Algebra)
        |
        v
       NQS (Ansatz + Sampler)
        |
        v
    Optimizer (SR / TDVP)
        |
        v
     Results (Energy, Observables)

Key Features
------------
- **JAX-based**: Fully differentiable and GPU-accelerated.
- **Flexible Architectures**: RBMs, CNNs, Transformers, and custom Flax modules.
- **Stochastic Reconfiguration (SR)**: Natural gradient descent implementation.
- **Time Evolution**: Time-Dependent Variational Principle (TDVP).

Invariants & Constraints
------------------------
1.  **Ansatz Signature**: Custom networks must implement `__call__(self, x)` where
    `x` has shape `(batch, n_sites)` (integer or float). The output must be
    complex log-amplitudes of shape `(batch,)` (scalar per sample).
2.  **Stability**: For deep networks, use `log_cosh` activation instead of `relu`
    or `tanh` to prevent numerical overflow in VMC.
3.  **Data Types**: Prefer `complex128` for high-precision physics, though `complex64`
    is supported for performance.
4.  **Sampling**: Samplers must be reset via `sampler.reset()` if the number of
    chains is changed.
"""

import importlib
from typing import TYPE_CHECKING, Any

# ----------------------------------------------------------------------------
# Lazy Import Configuration
# ----------------------------------------------------------------------------

_LAZY_IMPORTS = {
    "NQS": (".nqs", "NQS"),
    "VMCSampler": ("QES.Solver.MonteCarlo.vmc", "VMCSampler"),
    "NQSTrainer": (".src.nqs_train", "NQSTrainer"),
    "NQSTrainStats": (".src.nqs_train", "NQSTrainStats"),
    "NQSObservable": (".src.nqs_engine", "NQSObservable"),
    "NQSLoss": (".src.nqs_engine", "NQSLoss"),
    "NQSEvalEngine": (".src.nqs_engine", "NQSEvalEngine"),
    "EvaluationResult": (".src.nqs_engine", "EvaluationResult"),
    "NetworkFactory": (".src.nqs_network_integration", "NetworkFactory"),
    "TDVP": (".src.tdvp", "TDVP"),
    "TDVPStepInfo": (".src.tdvp", "TDVPStepInfo"),
}

_LAZY_CACHE = {}

if TYPE_CHECKING:
    from QES.Solver.MonteCarlo.vmc import VMCSampler

    from .nqs import NQS
    from .src.nqs_engine import EvaluationResult, NQSEvalEngine, NQSLoss, NQSObservable
    from .src.nqs_network_integration import NetworkFactory
    from .src.nqs_train import NQSTrainer, NQSTrainStats
    from .src.tdvp import TDVP, TDVPStepInfo


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy imports (PEP 562)."""
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            if module_path.startswith("."):
                module = importlib.import_module(module_path, package=__name__)
            else:
                module = importlib.import_module(module_path)

            result = getattr(module, attr_name)
            _LAZY_CACHE[name] = result
            return result
        except (ImportError, AttributeError) as e:
            # Special handling for NQS as it's critical
            if name == "NQS":
                raise ImportError(
                    f"Could not import NQS module. Ensure QES is installed correctly.\nOriginal error: {e}"
                ) from e
            # VMCSampler might be optional or handle gracefully
            if name == "VMCSampler":
                return None
            raise ImportError(
                f"Failed to import lazy attribute '{name}' from '{module_path}': {e}"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Support for dir() and tab completion."""
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


# Metadata
MODULE_DESCRIPTION = "Neural Quantum States (NQS) with TDVP and Adaptive Scheduling."
__version__ = "2.0.0"

# Helper Functions


def info():
    """Prints the library capability summary."""
    print(__doc__)


def quick_start(mode: str = "ground"):
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

    if mode.lower() == "excited":
        print(boilerplate + excited_body)
    else:
        print(boilerplate + ground_body)


# --------------------------------------------------------------

# Export
__all__ = [
    "NQS",
    "NQSTrainer",
    "NQSTrainStats",
    "NQSObservable",
    "NQSLoss",
    "NQSEvalEngine",
    "EvaluationResult",
    "TDVP",
    "TDVPStepInfo",
    "NetworkFactory",
    "VMCSampler",
    "quick_start",
    "info",
]

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------
