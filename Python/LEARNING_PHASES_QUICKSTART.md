# Quick Start: Learning Phases in Your Training (5 minutes)

## TL;DR

Learning phases let you structure training into stages with different hyperparameters.

Replace this:
```python
for i in range(500):
    lr = adaptive_lr(i, 0.03, 0.999)
```

With this:
```python
phases = create_learning_phases('kitaev')
scheduler = LearningPhaseScheduler(phases)
for i in range(scheduler.total_epochs):
    lr = scheduler.get_current_hyperparameters(i)['learning_rate']
    scheduler.advance_epoch()
```

---

## Installation (Already Done)

Files are already in your repo:
- `QES/NQS/src/learning_phases.py` ← The module
- `examples/example_learning_phases_integration.py` ← Working examples

---

## Step 1: Import

```python
from QES.NQS.src.learning_phases import (
    create_learning_phases,
    LearningPhaseScheduler
)
```

---

## Step 2: Create Phases

### Option A: Use a Preset (Easiest)

```python
# For Kitaev model
phases = create_learning_phases('kitaev')

# Available presets:
# - 'default': 3-phase general (350 epochs)
# - 'fast': 2-phase quick (70 epochs)
# - 'thorough': 3-phase extended (800 epochs)
# - 'kitaev': optimized (525 epochs)
```

### Option B: Custom Phases

```python
from QES.NQS.src.learning_phases import LearningPhase, PhaseType

phases = [
    LearningPhase(
        name="warm_up",
        epochs=50,
        learning_rate=0.1,
        lr_decay=0.98,
        regularization=0.05
    ),
    LearningPhase(
        name="training",
        epochs=200,
        learning_rate=0.03,
        lr_decay=0.999,
        regularization=0.01
    )
]
```

---

## Step 3: Create Scheduler

```python
scheduler = LearningPhaseScheduler(phases)
print(f"Total epochs to train: {scheduler.total_epochs}")
```

---

## Step 4: Use in Training Loop

```python
for epoch in range(scheduler.total_epochs):
    # Get LR from current phase
    hparams = scheduler.get_current_hyperparameters(epoch)
    lr = hparams['learning_rate']
    reg = hparams['regularization']  # Bonus: reg too!
    
    # Your training code (unchanged)
    step_info = train_single_step(params, nqs, epoch, lr)
    
    # Track history
    history[epoch] = step_info.mean_energy
    
    # Advance to next epoch
    scheduler.advance_epoch()
```

---

## What You Get

✅ **Structured training**: Pre-training → Main → Refinement stages  
✅ **Adaptive LR**: Different decay schedule per phase  
✅ **Auto transitions**: Phases change automatically  
✅ **Regularization**: Scheduled per phase (bonus!)  
✅ **Progress info**: Know which phase you're in  

---

## Example: Full Integration

```python
import numpy as np
from QES.NQS.src.learning_phases import create_learning_phases, LearningPhaseScheduler

def train_with_phases(params, nqs):
    # Setup phases
    phases = create_learning_phases('kitaev')
    scheduler = LearningPhaseScheduler(phases)
    
    # Initialize history
    history = np.zeros(scheduler.total_epochs)
    
    # Training loop
    for epoch in range(scheduler.total_epochs):
        # Get hyperparameters from scheduler
        hparams = scheduler.get_current_hyperparameters(epoch)
        lr = hparams['learning_rate']
        
        # Your existing training (1 step)
        step_info = train_single_step(params, nqs, epoch, lr)
        history[epoch] = step_info.mean_energy
        
        # Show progress
        phase = scheduler.current_phase
        print(f"Epoch {epoch} ({phase.name}): E={step_info.mean_energy:.4e} lr={lr:.3e}")
        
        # Advance scheduler
        scheduler.advance_epoch()
    
    return history

# Usage
history = train_with_phases(params, nqs)
```

---

## Presets at a Glance

```python
# Show all presets
from QES.NQS.src.learning_phases import create_learning_phases

for preset in ['default', 'fast', 'thorough', 'kitaev']:
    phases = create_learning_phases(preset)
    total = sum(p.epochs for p in phases)
    print(f"{preset}: {total} epochs in {len(phases)} phases")
```

Output:
```
default: 350 epochs in 3 phases
fast: 70 epochs in 2 phases
thorough: 800 epochs in 3 phases
kitaev: 525 epochs in 3 phases
```

---

## Common Questions

**Q: Will this break my existing code?**  
A: No. It's 100% optional and backwards compatible.

**Q: Do I need to change my sampling/gradient code?**  
A: No. Learning phases only provide learning rates.

**Q: Can I use this with my current training loop?**  
A: Yes. Add 4 lines, everything else stays the same.

**Q: What about TDVP?**  
A: Learning phases work with any optimization method.

**Q: Can I define custom phases?**  
A: Yes. See Option B above.

---

## Testing It

```bash
# Run the examples
cd /Users/makskliczkowski/Codes/pyqusolver/Python
python examples/example_learning_phases_integration.py

# Run the tests
python -m pytest test/test_learning_phases.py -v
```

---

## For More Details

- **How to integrate**: `LEARNING_PHASES_INTEGRATION_GUIDE.md`
- **Full reference**: `LEARNING_PHASES_GUIDE.md`
- **FAQ & summary**: `LEARNING_PHASES_COMPLETE_SUMMARY.md`
- **Working examples**: `examples/example_learning_phases_integration.py`

---

**That's it! You're ready to use learning phases.**

Start with a preset like 'kitaev', add 4 lines to your training, and you're done.
