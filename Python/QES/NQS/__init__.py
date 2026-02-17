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
- **JAX-based**: 
    Fully differentiable and GPU-accelerated.
- **Flexible Architectures**: 
    RBMs, CNNs, Transformers, and custom Flax modules.
- **Stochastic Reconfiguration (SR)**: 
    Natural gradient descent implementation.
- **Time Evolution**: 
    Time-Dependent Variational Principle (TDVP).
"""

import  importlib
from    dataclasses import dataclass, field, asdict
from    typing      import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# ----------------------------------------------------------------------------
#! Configuration dataclasses
# ----------------------------------------------------------------------------

@dataclass
class NQSPhysicsConfig:
    """
    Configuration for the physical model in NQS.
    Behaves like a dictionary and allows dynamic argument addition.
    """
    model_type      : str   = "kitaev"
    lattice_type    : str   = "honeycomb"
    lx              : int   = 3
    ly              : int   = 3
    lz              : int   = 1
    bc              : str   = "pbc"
    flux            : Optional[Union[str, Dict]] = None # Boundary fluxes, see Lattice class for details
    # magnetic field
    hx              : float = 0.0
    hy              : float = 0.0
    hz              : float = 0.0
    # Impurities: list of (site, amplitude) or (site, phi, theta, amplitude)
    impurities      : List[Tuple] = field(default_factory=list)
    # General couplings and other dynamic arguments
    args            : Dict[str, Any] = field(default_factory=dict)

    # ---------------
    # Lazy properties for derived quantities can be added here if needed
    # ---------------

    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        return self.args.get(key)

    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            self.args[key] = value

    def get(self, key, default=None):
        try:
            val = self[key]
            return val if val is not None else default
        except (AttributeError, KeyError):
            return default

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def to_dict(self):
        d       = asdict(self)
        extra   = d.pop("args")
        d.update(extra)
        return d

    @property
    def num_sites(self) -> int:
        mult = 2 if self.lattice_type.lower() == 'honeycomb' else 1
        return self.lx * self.ly * mult

    def make_hamiltonian(self, **kwargs):
        """
        Creates a Hamiltonian instance based on this configuration.
        """
        from QES.Algebra.Model              import choose_model
        from QES.general_python.lattices    import choose_lattice
        from QES.Algebra.hilbert            import HilbertSpace

        # Use local imports to avoid circular dependencies
        lattice = choose_lattice(
            typek   =   self.lattice_type,
            lx      =   self.lx,
            ly      =   self.ly,
            bc      =   self.bc,
            flux    =   self.flux,
            **self.args # Pass any additional lattice-related args
        )

        # Allow overriding hilbert kwargs from args, but also support direct kwargs
        hilbert_kwargs = self.get("hilbert_kwargs", {})
        if "dtype" in kwargs and "dtype" not in hilbert_kwargs:
            hilbert_kwargs["dtype"] = kwargs["dtype"]
            
        hilbert         = HilbertSpace(lattice=lattice, **hilbert_kwargs)
        model_kwargs    = self.args.copy()
        model_kwargs.update({
            "lattice"       : lattice,
            "hilbert_space" : hilbert,
            "hx"            : self.hx if self.hx != 0 else None,
            "hy"            : self.hy if self.hy != 0 else None,
            "hz"            : self.hz if self.hz != 0 else None,
            "impurities"    : self.impurities,
        })
        
        # Special mapping for Kitaev couplings if provided in args
        if self.model_type == 'kitaev':
            
            if 'K' not in model_kwargs and ('kxy' in self.args or 'kz' in self.args):
                model_kwargs['K']       = (self.get('kxy', 0.0), self.get('kxy', 0.0), self.get('kz', 1.0))
                
            if 'Gamma' not in model_kwargs and ('gamma_xy' in self.args or 'gamma_z' in self.args):
                model_kwargs['Gamma']   = (self.get('gamma_xy', 0.0), self.get('gamma_xy', 0.0), self.get('gamma_z', 0.0))
        # Update with any additional model-related args
        model_kwargs.update(kwargs)

        return choose_model(self.model_type, **model_kwargs), hilbert, lattice

@dataclass
class NQSSolverConfig:
    """
    Configuration for the NQS solver and training.
    """
    ansatz      : str   = "rbm"
    sampler     : str   = "MCSampler"
    n_chains    : int   = 16
    n_samples   : int   = 1000
    n_sweep     : int   = 10
    n_therm     : int   = 100
    lr          : float = 0.01
    epochs      : int   = 300
    dtype       : str   = "complex128"
    backend     : str   = "jax"
    
    # SOTA configuration estimated from physics
    sota_config : Any   = field(default=None, init=False)

    def __post_init__(self):
        """
        Lazy estimation can be triggered here if enough information is present.
        """
        pass

    def estimate(self, physics_config: NQSPhysicsConfig, **kwargs):
        """
        Automatically estimates network parameters based on physics configuration.
        """
        from QES.NQS.src.nqs_network_integration import estimate_network_params
        
        self.sota_config = estimate_network_params(
            net_type=self.ansatz,
            num_sites=physics_config.num_sites,
            lattice_dims=(physics_config.lx, physics_config.ly),
            lattice_type=physics_config.lattice_type,
            model_type=physics_config.model_type,
            dtype=self.dtype,
            **kwargs
        )
        return self.sota_config

    def make_net(self, physics_config: NQSPhysicsConfig, **kwargs):
        """
        Creates a network instance based on this configuration and a physics configuration.
        """
        from QES.NQS.src.nqs_network_integration import NetworkFactory
        
        if self.sota_config is None:
            self.estimate(physics_config, **kwargs)
            
        factory_kwargs = self.sota_config.to_factory_kwargs()
        factory_kwargs.update(kwargs)
        
        return NetworkFactory.create(
            network_type=self.ansatz,
            backend=self.backend,
            **factory_kwargs
        )

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
    "NQSPhysicsConfig",
    "NQSSolverConfig",
    "quick_start",
    "info",
]

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------
