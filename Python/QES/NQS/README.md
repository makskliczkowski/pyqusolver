# Neural Quantum State (NQS) Solver

This folder contains the implementation of the Neural Quantum State (NQS) solver for quantum many-body systems. The NQS solver uses Monte Carlo methods to optimize neural network representations of quantum states. It supports both NumPy and JAX backends for efficient computation.

## Contents

- `nqs.py`  : Main NQS solver class and interface.
- `src/`    : Contains submodules for physics models, neural network architectures, and evaluation engines.
- Example scripts and tests for NQS usage.

## Features

- Monte Carlo-based training and sampling for quantum states.
- Flexible backend support (NumPy or JAX).
- Modular design for custom networks, samplers, and physical models.
- Functions to evaluate the neural network ansatz and apply custom physical functions (e.g., local energy, observables).

## Environment Setup

Set the following environment variables before running:

```bash
export QES_PYPATH=/path/to/QES
export PY_BACKEND=jax # or numpy
```

## Installation

Install dependencies from the main requirements file:

```bash
pip install -r ../../requirements/requirements.txt
```

---

## Usage Example

```python
import jax
import jax.numpy as jnp
from QES.NQS.nqs import NQS
from QES.NQS.src.nqs_network_integration import NetworkFactory
from QES.Solver.MonteCarlo.sampler import VMCSampler # Import the sampler

# Mock Hamiltonian (replace with your actual model)
class MockHamiltonian:
    def __init__(self, ns): self.ns = ns
    def local_energy_fn(self, state, params, log_psi_func): return jnp.array(0.0) # Placeholder
    @property
    def shape(self): return (16,)
model = MockHamiltonian(ns=16)

# Define your network
net = NetworkFactory.create('rbm', input_shape=(model.ns,), alpha=2.0)

# Define your sampler
sampler = VMCSampler(
    net=net,
    shape=(model.ns,),
    rng=jax.random,
    rng_k=jax.random.PRNGKey(0),
    numchains=1,
    numsamples=10,
    therm_steps=100,
    sweep_steps=1,
    backend='jax'
)

# Finally, create the NQS solver instance
nqs = NQS(net=net, sampler=sampler, model=model, batch_size=32)

# Evaluate the ansatz
states = jnp.array([[-0.5, 0.5], [0.5, -0.5]], dtype=jnp.float32) # Example states
log_psi = nqs.evaluate(states)

# Apply a custom function (e.g., local energy)
def local_energy(states, psi, params):
    # This is a placeholder. Your actual local energy calculation would go here.
    return jnp.array([1.0, 2.0]) # Example energy values

energy = nqs.apply(local_energy, states_and_psi=(states, log_psi))
```

---

## Training and TDVP

The NQS package includes a training module (`nqs_train.py`) and support for the Time-Dependent Variational Principle (TDVP) method for optimizing neural quantum states.

- **Training**: The `NQSTrainer` class implements the training loop, learning rate scheduling, regularization scheduling, and early stopping. It works with JAX backend and is compatible with Flax networks. Training is performed by repeatedly sampling states, computing gradients, and updating network parameters.

- **TDVP**: The TDVP (Time-Dependent Variational Principle) is a method for evolving variational wavefunctions in time or optimizing them for ground/excited states. The package provides `TDVP` and `TDVPLowerPenalty` classes for these tasks. TDVP is used within the training loop to compute parameter updates.

### Example: Training with TDVP

```python
from QES.NQS.nqs                        import NQS
from QES.NQS.nqs_train                  import NQSTrainer
from QES.NQS.tdvp                       import TDVP
from QES.general_python.algebra.ode     import IVP
from QES.general_python.ml.schedulers   import Parameters, EarlyStopping

# Setup NQS, TDVP, ODE solver, and schedulers
nqs             = NQS(...)              # as defined earlier
tdvp            = TDVP(...)             # TDVP instance -> to compute parameter updates
ode_solver      = IVP(...)              # ODE solver for TDVP equations
lr_scheduler    = Parameters(...)       # Learning rate scheduler
reg_scheduler   = Parameters(...)       # Regularization scheduler
early_stopper   = EarlyStopping(...)    # Early stopping

# Create the trainer
trainer         = NQSTrainer(
                    nqs             = nqs,
                    ode_solver      = ode_solver,
                    tdvp            = tdvp,
                    n_batch         = 32,
                    lr_scheduler    = lr_scheduler,
                    reg_scheduler   = reg_scheduler,
                    early_stopper   = early_stopper,
                    logger          = None
                )

# Train for a number of epochs
history, history_std, timings = trainer.train(n_epochs=100)
```

The training loop will sample states, compute gradients using TDVP, update parameters, and record statistics. Early stopping and learning rate/regularization scheduling are supported.

- See the main package README for more details.
- Example scripts are available in the `examples/` directory.

---

## Custom Networks and Activation Functions

You can easily create your own neural network in Flax and use it with NQS. The only requirements are:

- The network must accept input arrays of shape `(batch_size, n_visible)` (where `n_visible` is the number of sites/spins/qubits).
- The output must be a 1D or 2D array of shape `(batch_size,)` or `(batch_size, 1)` representing the log-amplitude (or amplitude) of the quantum state.

### Plugging in a Custom Flax Network

Suppose you have a Flax module:

```python
import flax.linen   as nn
import jax.numpy    as jnp

class MyCustomNet(nn.Module):
    features: int = 32
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)

# Wrap with QES FlaxInterface
from QES.general_python.ml.net_impl.interface_net_flax import FlaxInterface
net = FlaxInterface(net_module=MyCustomNet, input_shape=(n_visible,), backend='jax', dtype=jnp.float32)

# Use with NQS
nqs = NQS(net=net, sampler=sampler, model=model, batch_size=32)
```

### Customizing Activation Functions

All QES networks accept an `activations` argument (or `act_fun`) to specify activation functions for each layer. You can use built-in JAX/NumPy functions or those provided in `QES.general_python.ml.net_impl.activation_functions`.

For example:

```python
from QES.general_python.ml.net_impl.activation_functions import elu_jnp, relu_jnp

net = CNN(
    input_shape     =   (n_visible,),
    reshape_dims    =   (lx, ly),
    features        =   (8, 8),
    kernel_sizes    =   [(2, 2), (2, 2)],
    activations     =   [elu_jnp, relu_jnp],
    output_shape    =   (1,),
    dtype           =   jnp.float32,
    param_dtype     =   jnp.float32,
    seed            =   42
)
```

You can also pass custom activation functions as callables. The activation function must accept a JAX array and return a JAX array of the same shape.

### Arguments for Networks

- **RBM**: `input_shape`, `n_hidden`, `dtype`, `param_dtype`, `seed`, `visible_bias`, `bias`
- **CNN**: `input_shape`, `reshape_dims`, `features`, `kernel_sizes`, `strides`, `activations`, `output_shape`, `dtype`, `param_dtype`, `final_activation`, `seed`
- **Autoregressive**: `input_shape`, `hidden_layers`, `activation`, `output_activation`, `use_bias`, `dtype`, `param_dtype`, `seed`

See the source code and docstrings for more details on each network's arguments.
