# Learning Phases Implementation Guide

**Date**: November 1, 2025  
**Feature**: Multi-Phase Learning for Neural Quantum States  
**Status**: ✅ IMPLEMENTED

---

## Overview

The Learning Phases feature enables structured, multi-phase training for NQS systems. It provides:

- **Pre-training Phase**: Initialize network with high learning rate
- **Main Optimization Phase**: Full Hamiltonian optimization with adaptive learning rate
- **Refinement Phase**: Fine-tune observables with low learning rate and regularization
- **Custom Phases**: Define your own phase configurations
- **Phase Callbacks**: Execute code at phase transitions
- **Adaptive Hyperparameters**: Learning rate and regularization schedules

---

## Architecture

### Key Components

#### 1. `LearningPhase` Dataclass

Represents a single training phase with parameters:

```python
@dataclass
class LearningPhase:
    name: str                           # Phase identifier
    epochs: int                         # Number of epochs
    phase_type: PhaseType              # PRE_TRAINING, MAIN, REFINEMENT, CUSTOM
    
    learning_rate: float               # Initial learning rate
    lr_decay: float                    # Multiplicative decay per epoch
    lr_min: float                      # Lower bound on learning rate
    
    regularization: float              # Regularization strength
    reg_schedule: str                  # 'constant', 'exponential', 'linear', 'adaptive'
    
    loss_type: str                     # 'energy', 'variance', 'combined'
    loss_weight_energy: float          # Weight for energy term
    loss_weight_variance: float        # Weight for variance term
    
    beta_penalty: float                # Penalty for excited state orthogonality
    
    # Callbacks
    on_phase_start: Optional[Callable]
    on_phase_end: Optional[Callable]
    on_epoch_start: Optional[Callable]
    on_epoch_end: Optional[Callable]
```

#### 2. `LearningPhaseScheduler`

Manages training across multiple phases:

```python
class LearningPhaseScheduler:
    def __init__(self, phases: List[LearningPhase], logger=None)
    
    @property
    def current_phase(self) -> LearningPhase
    
    @property
    def is_finished(self) -> bool
    
    def get_current_hyperparameters(epoch: int) -> Dict[str, float]
    def on_phase_start(self)
    def on_phase_end(self)
    def advance_epoch(self) -> bool
    def get_progress_summary(self) -> Dict[str, Any]
```

#### 3. NQS Integration

Updated `NQS.train()` method:

```python
def train(self,
         nsteps: int = 1,
         verbose: bool = False,
         use_sr: bool = True,
         use_learning_phases: bool = True,
         **kwargs) -> list:
    """Train with or without learning phases"""
```

---

## Usage Examples

### Example 1: Simple Multi-Phase Training

```python
from QES.NQS.nqs import NQS
from QES.NQS.src.learning_phases import create_learning_phases

# Create NQS with default 3-phase training
nqs = NQS(
    net=network,
    sampler=sampler,
    model=hamiltonian,
    learning_phases='default'  # Uses pre-training, main, refinement
)

# Train with learning phases
history = nqs.train(use_learning_phases=True, verbose=True)

# Access results
print(f"Final energy: {history['phase_energies'][-1]}")
print(f"Phase transitions: {history['phase_transitions']}")
```

### Example 2: Custom Learning Phases

```python
from QES.NQS.src.learning_phases import LearningPhase, PhaseType

# Define custom phases
phases = [
    LearningPhase(
        name="initialization",
        epochs=30,
        phase_type=PhaseType.PRE_TRAINING,
        learning_rate=5e-2,
        lr_decay=0.99,
        regularization=1e-1,
        reg_schedule='exponential',
        description="Initialize network quickly"
    ),
    LearningPhase(
        name="optimization",
        epochs=300,
        phase_type=PhaseType.MAIN,
        learning_rate=1e-2,
        lr_decay=0.9995,
        regularization=1e-3,
        reg_schedule='constant',
        description="Main optimization with adaptive learning rate"
    ),
]

nqs = NQS(
    net=network,
    sampler=sampler,
    model=hamiltonian,
    learning_phases=phases
)

history = nqs.train(use_learning_phases=True)
```

### Example 3: Kitaev Model with Learning Phases

```python
from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
from QES.NQS.src.learning_phases import create_learning_phases

# Create Kitaev model
lattice = HoneycombLattice(lx=4, ly=3, bc='pbc')
kitaev_model = HeisenbergKitaev(
    lattice=lattice,
    K=[1.0, 0.8, 0.5],  # Anisotropic Kitaev
    J=0.3,              # Heisenberg coupling
    impurities=[(0, 0.5)]
)

# Use Kitaev-optimized phases
nqs = NQS(
    net=network,
    sampler=sampler,
    model=kitaev_model,
    learning_phases='kitaev'  # Preset for frustrated systems
)

history = nqs.train(use_learning_phases=True, verbose=True)

print(f"Ground state energy: {history['phase_energies'][-1]:.6f}")
```

### Example 4: Using Callbacks

```python
def on_phase_start_callback():
    print(f"Starting new training phase")

def on_phase_end_callback():
    print(f"Phase completed!")

def on_epoch_end_callback(epoch, loss, hyperparams):
    print(f"Epoch {epoch}: E={loss:.6f}, lr={hyperparams['learning_rate']:.2e}")

phases = [
    LearningPhase(
        name="main",
        epochs=100,
        learning_rate=1e-2,
        on_phase_start=on_phase_start_callback,
        on_phase_end=on_phase_end_callback,
        on_epoch_end=on_epoch_end_callback
    )
]

nqs = NQS(
    net=network,
    sampler=sampler,
    model=hamiltonian,
    learning_phases=phases
)

history = nqs.train(use_learning_phases=True)
```

### Example 5: Backwards Compatibility - Traditional Training

```python
# Old-style training still works
history = nqs.train(nsteps=1000, use_learning_phases=False)

# Or with single phase
nqs2 = NQS(
    net=network,
    sampler=sampler,
    model=hamiltonian,
    learning_phases=[
        LearningPhase(
            name="single_phase",
            epochs=1000,
            learning_rate=1e-2
        )
    ]
)

history = nqs2.train(use_learning_phases=True)
```

---

## Available Presets

### 1. `'default'` (Recommended)

Three-phase training optimized for general use:

```python
Pre-training: 50 epochs, lr=0.1 → 0.001, reg=0.05 (exponential decay)
Main:         200 epochs, lr=0.03 → 0.0001, reg=0.001 (constant)
Refinement:   100 epochs, lr=0.01 → 0.00001, reg=0.005 (linear)
```

### 2. `'fast'`

Quick training for prototyping:

```python
Pre-training: 20 epochs, high learning rate
Main:         50 epochs, medium learning rate
```

### 3. `'thorough'`

Extended training for best results:

```python
Pre-training: 100 epochs, high regularization
Main:         500 epochs, low regularization
Refinement:   200 epochs, very low learning rate
```

### 4. `'kitaev'`

Optimized for frustrated systems (Kitaev, Heisenberg-Kitaev):

```python
Pre-training: 75 epochs, optimized for spin frustration
Main:         300 epochs, adaptive decay
Refinement:   150 epochs, fine-tune phase
```

---

## Learning Rate Schedules

### Supported Schedules

1. **Exponential** (default for pre-training)
   ```
   lr(epoch) = initial_lr * (decay)^epoch
   ```

2. **Linear** (refinement phase)
   ```
   lr(epoch) = initial_lr * (1 - epoch/total_epochs)
   ```

3. **Constant**
   ```
   lr(epoch) = initial_lr
   ```

---

## Regularization Schedules

### Supported Schedules

1. **Constant** (default for main optimization)
   ```
   reg(epoch) = regularization
   ```

2. **Exponential** (pre-training)
   ```
   reg(epoch) = regularization * (1 + exp(-5*epoch/epochs))
   ```

3. **Linear** (refinement)
   ```
   reg(epoch) = regularization * (1 + epoch/epochs)
   ```

4. **Adaptive**
   ```
   reg(epoch) = regularization * exp(-3*epoch/epochs)
   ```

---

## Hyperparameter Tuning Guide

### For Simple Systems

```python
phases = [
    LearningPhase(
        name="quick",
        epochs=50,
        learning_rate=1e-2,
        regularization=1e-4
    )
]
```

### For Frustrated Systems (Kitaev, Gamma)

```python
use_learning_phases='kitaev'
# Or customize:
phases = [
    LearningPhase(
        name="pre",
        epochs=100,
        learning_rate=5e-2,
        lr_decay=0.98,
        regularization=1e-1,
        reg_schedule='exponential'
    ),
    LearningPhase(
        name="main",
        epochs=400,
        learning_rate=1e-2,
        lr_decay=0.999,
        regularization=5e-4
    )
]
```

### For Large Systems

```python
phases = [
    LearningPhase(
        name="initialization",
        epochs=200,
        learning_rate=1e-1,
        regularization=5e-2
    ),
    LearningPhase(
        name="optimization",
        epochs=1000,
        learning_rate=5e-3,
        lr_decay=0.9995,
        regularization=1e-3
    ),
    LearningPhase(
        name="refinement",
        epochs=300,
        learning_rate=1e-3,
        regularization=5e-3
    )
]
```

---

## Advanced Features

### Phase-Specific Loss Functions

```python
phases = [
    LearningPhase(
        name="main",
        epochs=100,
        loss_type='combined',
        loss_weight_energy=0.8,
        loss_weight_variance=0.2  # Multi-objective optimization
    )
]
```

### Excited State Training

```python
phases = [
    LearningPhase(
        name="excited_state",
        epochs=100,
        beta_penalty=1.0  # Penalty for orthogonality to ground state
    )
]
```

### Progress Monitoring

```python
history = nqs.train(use_learning_phases=True)

# Access detailed history
print(f"Total epochs: {history['summary']['total_epochs']}")
print(f"Progress: {history['summary']['progress_pct']:.1f}%")
print(f"Phase sequence: {[p['phase_name'] for p in history['phase_transitions']]}")

# Plot learning curves
import matplotlib.pyplot as plt
plt.plot(history['global_epochs'], history['phase_energies'])
plt.xlabel('Global Epoch')
plt.ylabel('Energy')
plt.show()
```

---

## Performance Considerations

### Memory Usage

- Each phase stores minimal additional state
- Phase scheduler is lightweight
- Callbacks optional

### Computational Efficiency

- Learning rate decay reduces per-epoch computation
- Adaptive schedules optimize convergence
- Phase transitions allow strategy changes

### Recommended Epoch Counts

| System Type | Total Epochs | Pre-training | Main | Refinement |
|-------------|--------------|--------------|------|------------|
| Small (N<8) | 150-300      | 10-20%       | 50-70% | 10-20%    |
| Medium (N=8-12) | 300-600  | 10-15%       | 70-80% | 10-15%    |
| Large (N>12) | 600-2000    | 5-10%        | 70-85% | 5-10%     |
| Frustrated  | 400-1000     | 15-25%       | 60-75% | 10-20%    |

---

## Troubleshooting

### Energy diverging?

**Solution**: Reduce learning rate in pre-training phase

```python
phase.learning_rate = 5e-2  # Reduce from 0.1
```

### Slow convergence?

**Solution**: Increase learning rate or reduce regularization

```python
phase.learning_rate = 5e-2
phase.regularization = 5e-4
```

### Early stopping?

**Solution**: Add more refinement epochs or reduce decay

```python
phase.lr_decay = 0.9998  # Slower decay
phase.epochs = 200  # More epochs
```

---

## Migration Guide

### From Old-Style Training

**Before:**
```python
for epoch in range(1000):
    params, energy, _ = nqs.train_step_np(...)
    lr = initial_lr * (0.999 ** epoch)
```

**After:**
```python
history = nqs.train(use_learning_phases=True)
```

---

## References

- `QES/NQS/src/learning_phases.py` - Learning phase implementation
- `QES/NQS/nqs.py` - NQS integration (lines ~1245-1380)
- `NQS_WORKING_DOCUMENT.md` - Design documentation

---

## File Locations

- **Learning Phases Module**: `QES/NQS/src/learning_phases.py`
- **NQS Integration**: `QES/NQS/nqs.py` (train methods)
- **Examples**: This document

---

**Status**: ✅ COMPLETE AND TESTED  
**Version**: 1.0  
**Last Updated**: November 1, 2025
