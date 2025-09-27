'''
Implementation of various physical problem layers for Neural Quantum States (NQS).
'''
import inspect
import numpy as np
from abc import ABC, abstractmethod

#! Import the backend interface
from nqs_backend import *

# --------------------------------------------------------------
#! Interface for physical problems
# --------------------------------------------------------------

class PhysicsInterface(ABC):
    """Abstract physical problem layer: wavefunctions, density matrices, tomography, etc."""

    def __init__(self, backend: BackendInterface):
        self.backend    = backend
        self.net        = None # Neural network ansatz

    @abstractmethod
    def setup(self, model, net):
        """Initialize physics object given model and network."""
        pass

    @abstractmethod
    def loss(self, params, data):
        """Define training loss (local energy, likelihood, etc.)."""
        pass

# --------------------------------------------------------------
#! Wavefunction physics
# --------------------------------------------------------------

class WavefunctionPhysics(PhysicsInterface):
    '''
    Wavefunction physics interface for NQS.
    Supports Hamiltonian operators and local energy computations.
    '''
    
    def setup(self, hamiltonian: Hamiltonian, net: Callable):
        '''
        Setup the wavefunction physics with a Hamiltonian and neural network.
        Parameters:
        -----------
        hamiltonian : Hamiltonian or Callable
            The Hamiltonian operator or a function to compute local energies.
        net : Callable
            The neural network ansatz representing the wavefunction.
        '''
        
        if hamiltonian is None:
            raise ValueError("Wavefunction mode requires a Hamiltonian.")
        if not isinstance(self._hamiltonian, Hamiltonian):
            if not callable(self._hamiltonian):
                raise ValueError(self._ERROR_HAMILTONIAN_TYPE)
            # check if it accepts a state vector only
            elif len(inspect.signature(self._hamiltonian).parameters) != 1:
                raise ValueError(self._ERROR_HAMILTONIAN_ARGS)
            else:
                self.local_energy_fn = hamiltonian
        else:    
            self.local_energy_fn = self.backend.local_energy(hamiltonian)

    def loss(self, params, states):
        return self.local_energy_fn(states, params)

# --------------------------------------------------------------

class DensityMatrixPhysics(PhysicsInterface):
    def setup(self, model, net):
        # Model could be Lindbladian or other operator
        self.operator = model
        self.net      = net

    def loss(self, params, states):
        pass

# --------------------------------------------------------------

class TomographyPhysics(PhysicsInterface):
    def setup(self, model, net):
        # Model is experimental data (probabilities)
        self.data = model
        self.net = net

    def loss(self, params, measurements):
        # Negative log-likelihood for observed measurements
        predicted = self.net(params, measurements)
        return -np.sum(self.data * np.log(predicted + 1e-9))

# --------------------------------------------------------------

class PhysicsInterfaces(Enum):
    ''' Factory for creating physics interfaces. '''
    WAVEFUNCTION    = WavefunctionPhysics
    DENSITY_MATRIX  = DensityMatrixPhysics
    TOMOGRAPHY      = TomographyPhysics
    
    def create(self, backend: BackendInterface):
        return self.value(backend)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"PhysicsInterfaces.{self.name}"
    
    def __call__(self, backend: BackendInterface):
        return self.create(backend)

def nqs_choose_physics(physics_type: Union[str, PhysicsInterfaces], backend: BackendInterface) -> PhysicsInterface:
    """Factory function to select and instantiate a physics interface."""
    if isinstance(physics_type, PhysicsInterfaces):
        return physics_type(backend)
    
    try:
        physics_enum = PhysicsInterfaces[physics_type.upper()]
        return physics_enum(backend)
    except KeyError:
        raise ValueError(f"Unsupported physics type: {physics_type}")

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------