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
from QES.NQS.nqs import NQS

# Define your network, sampler, and Hamiltonian
net         = ... # Neural network (e.g., RBM, CNN)
sampler     = ... # Monte Carlo sampler
model       = ... # Hamiltonian or physical model

# Finally, create the NQS solver instance
nqs         = NQS(net=net, sampler=sampler, model=model, batch_size=32)

# Evaluate the ansatz
states      = np.array([[-1, 1], [1, -1]])
log_psi     = nqs.evaluate(states)

# Apply a custom function (e.g., local energy)
def local_energy(states, psi, params):
    return energy

energy     = nqs.apply(local_energy, states_and_psi=(states, log_psi))
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
