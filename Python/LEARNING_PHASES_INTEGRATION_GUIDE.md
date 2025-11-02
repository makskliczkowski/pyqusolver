# Learning Phases Integration Guide: Consolidating Your Training Style

**TL;DR**: Learning phases are a tool for managing hyperparameter schedules across multiple training epochs. Your current training loop already does this manually—learning phases just automate it.

---

## What Are Learning Phases? (Simple Version)

Your current training does this:
```python
for epoch in range(nepo):
    lr = adaptive_lr(epoch, initial_lr, decay_rate)  # ← Different LR each epoch
    # Train one step with this LR
    # Collect energy history
```

Learning phases formalize this into named stages:
```
Phase 1 (Pre-training):   lr: 0.1 → 0.001,  epochs: 50
Phase 2 (Main):           lr: 0.03 → 0.0001, epochs: 200
Phase 3 (Refinement):     lr: 0.01 → 0.00001, epochs: 100
```

Instead of one exponential decay across 500 epochs, you get three different decay schedules optimized for each stage.

---

## Your Current Training Pattern

Your training function (`train_function`) does:

```python
for i in pbar:
    # 1. Get adaptive LR for this epoch
    current_lr = adaptive_lr(i, params.lr, decay_rate)
    
    # 2. Train one step
    step_info = train_single_step(params, nqs, i, current_lr, reset)
    
    # 3. Track history
    history[i] = step_info.mean_energy
    history_std[i] = step_info.std_energy
    epoch_times[i] = step_info.time_info.time_total
```

**No TDVP is used here** - you're doing manual gradient computation and parameter updates.

---

## How Learning Phases Fit In

Instead of a single `adaptive_lr()` function, learning phases provide:

### Without Learning Phases (Your Current Way)
```python
def adaptive_lr(epoch, initial_lr, decay_rate):
    return max(5.0e-3, initial_lr * (decay_rate ** epoch))

# Applied globally to all 500 epochs
for i in range(500):
    lr = adaptive_lr(i, 0.03, 0.999)  # Same formula the whole time
```

### With Learning Phases (Structured Way)
```python
from QES.NQS.src.learning_phases import create_learning_phases

phases = create_learning_phases('default')
# Phase 1: different decay schedule
# Phase 2: different decay schedule  
# Phase 3: different decay schedule

scheduler = LearningPhaseScheduler(phases)
for global_epoch in range(500):
    hparams = scheduler.get_current_hyperparameters(global_epoch)
    lr = hparams['learning_rate']      # ← Automatically from current phase
    reg = hparams['regularization']    # ← Bonus: regularization schedule too
    
    scheduler.advance_epoch()
```

---

## Integration Option 1: Minimal Change (Recommended)

Keep your training loop mostly unchanged, just use learning phases for hyperparameter scheduling:

```python
from QES.NQS.src.learning_phases import create_learning_phases, LearningPhaseScheduler
import numpy as np

def train_function_with_phases(params: SimulationParams, 
                               nqs: nqsmodule.NQS,
                               reset: bool = False,
                               phase_preset: str = 'default'):
    """Your training function enhanced with learning phase scheduling."""
    
    # Create phase scheduler
    phases = create_learning_phases(phase_preset)
    scheduler = LearningPhaseScheduler(phases)
    
    history = []
    history_std = []
    epoch_times = []
    
    pbar = trange(scheduler.total_epochs, desc="Training with Phases...", leave=True)
    timer = Timer("Total Training Time")
    
    for global_epoch in pbar:
        # Get hyperparameters from current phase
        hparams = scheduler.get_current_hyperparameters(global_epoch)
        current_lr = hparams['learning_rate']
        
        # Your existing training step
        step_info = train_single_step(
            params=params, 
            nqs=nqs, 
            i=global_epoch, 
            lr=current_lr,  # ← From phase scheduler
            reset=reset, 
            timer=timer
        )
        
        # Phase transition callback (optional)
        if scheduler.phase_changed:
            phase = scheduler.current_phase
            logger.info(f"Transitioning to {phase.phase_type.name} phase: {phase.name}")
            if phase.on_phase_start:
                phase.on_phase_start()
        
        # Your existing history tracking
        history.append(step_info.mean_energy)
        history_std.append(step_info.std_energy)
        epoch_times.append(step_info.time_info.time_total)
        
        # Progress display
        phase_name = scheduler.current_phase.name
        phase_epoch = scheduler.phase_epoch
        pbar.set_description(f"Phase: {phase_name} ({phase_epoch}/{scheduler.current_phase.epochs})")
        pbar.set_postfix({
            "E_mean": f"{step_info.mean_energy:.4e}",
            "E_std": f"{step_info.std_energy:.4e}",
            "lr": f"{current_lr:.3e}",
            "t_epoch": f"{step_info.time_info.time_total:.3e}s"
        })
        
        # Advance scheduler
        scheduler.advance_epoch()
    
    return np.array(history), np.array(history_std), np.array(epoch_times)
```

**Usage**:
```python
# Using default phases
history, history_std, times = train_function_with_phases(params, nqs)

# Using Kitaev-optimized phases
history, history_std, times = train_function_with_phases(params, nqs, phase_preset='kitaev')

# Or fast prototyping
history, history_std, times = train_function_with_phases(params, nqs, phase_preset='fast')
```

---

## Integration Option 2: Custom Phases for Your System

Define your training schedule as phases:

```python
from QES.NQS.src.learning_phases import LearningPhase, PhaseType

# Define your custom phases matching your training philosophy
my_phases = [
    LearningPhase(
        name="initialization",
        epochs=50,
        phase_type=PhaseType.PRE_TRAINING,
        learning_rate=0.1,      # Start high to explore
        lr_decay=0.98,
        lr_min=1e-5,
        regularization=0.05,
        reg_schedule='exponential'
    ),
    LearningPhase(
        name="convergence",
        epochs=200,
        phase_type=PhaseType.MAIN,
        learning_rate=0.03,     # Medium for steady progress
        lr_decay=0.999,
        lr_min=1e-5,
        regularization=0.01,
        reg_schedule='linear'
    ),
    LearningPhase(
        name="fine_tuning",
        epochs=100,
        phase_type=PhaseType.REFINEMENT,
        learning_rate=0.01,     # Low for precision
        lr_decay=0.995,
        lr_min=5e-5,
        regularization=0.005,
        reg_schedule='adaptive'
    )
]

# Use in your training
scheduler = LearningPhaseScheduler(my_phases)
history, history_std, times = train_function_with_phases(
    params, nqs, 
    phase_preset=my_phases  # Pass custom phases directly
)
```

---

## Integration Option 3: If You Want to Use NQS.train()

The NQS class HAS a built-in train method (though your manual loop is more explicit):

```python
# WITHOUT learning phases (backwards compatible)
history = nqs.train(nsteps=500, use_learning_phases=False)

# WITH learning phases (using our system)
nqs = nqsmodule.NQS(
    net=network,
    sampler=sampler,
    model=model,
    learning_phases='kitaev'  # ← Specify phases at init
)

history = nqs.train(nsteps=500, use_learning_phases=True)
```

---

## Key Differences: Learning Phases vs Your Manual Approach

| Aspect | Your Current | With Learning Phases |
|--------|-------------|----------------------|
| LR Schedule | One formula (decay_rate) | Per-phase formulas |
| Regularization | Not tracked | Adaptive per phase |
| Phase transitions | Manual (implicitly) | Automatic with callbacks |
| Flexibility | High (you control everything) | Structured (presets available) |
| Lines of code | Minimal | Same (if using Option 1) |
| Debugging | Easy (explicit loop) | Easier (structured phases) |

---

## Important Notes

### ✅ What Learning Phases Handle
- Hyperparameter scheduling (LR, regularization)
- Phase transition callbacks
- Progress tracking across phases
- Preset configurations (default, fast, thorough, kitaev)

### ❌ What Learning Phases DON'T Change
- Your sampling logic (`nqs.sample()`)
- Your gradient computation (`nqs.step()`)
- Your parameter updates
- Your history tracking
- **TDVP is optional** - you don't need to use it

### 🔄 TDVP Status
Your code prepares TDVP but doesn't use it in `train_single_step()`. This is fine! TDVP is an alternative optimization method (second-order). Your current approach uses stochastic gradient descent, which is valid. Learning phases work with BOTH.

---

## Recommendation

**Use Option 1** - it's the minimal change that gives you structure without sacrificing your control:

```python
# Your existing training loop stays mostly the same
# Just add 3 lines at the top to create the scheduler
# And 1 line to get the learning rate from it each epoch

phases = create_learning_phases('kitaev')
scheduler = LearningPhaseScheduler(phases)

for epoch in range(total_epochs):
    lr = scheduler.get_current_hyperparameters(epoch)['learning_rate']
    # Rest of your code stays the same
    scheduler.advance_epoch()
```

This way you get:
- ✅ Structured, reproducible training
- ✅ Easy phase management
- ✅ Callbacks for monitoring
- ✅ Preset configurations
- ✅ Your existing code mostly unchanged

---

## Why This Matters

Your current approach works fine. Learning phases add value when you want to:
1. **Share training configs** - Define "kitaev_setup" once, use everywhere
2. **Reproduce results** - "I used preset 'kitaev'" is clear and reproducible
3. **Experiment systematically** - Change one preset, see effect on all experiments
4. **Track phase transitions** - Know exactly when phases change and why
5. **Add callbacks** - Run code at phase boundaries (e.g., checkpoint saving)

---

## Quick Start: Your Setup

```python
from QES.NQS.src.learning_phases import create_learning_phases, LearningPhaseScheduler

# In your main training call:
def train_nqs_with_phases():
    params = SimulationParams(nepo=500, lr=0.03)  # Your config
    nqs = prepare_nqs(params)
    
    # Create phases for Kitaev model (your interest)
    phases = create_learning_phases('kitaev')
    scheduler = LearningPhaseScheduler(phases)
    
    # Your training loop with phase scheduling
    for global_epoch in range(scheduler.total_epochs):
        lr = scheduler.get_current_hyperparameters(global_epoch)['learning_rate']
        step_info = train_single_step(params, nqs, global_epoch, lr)
        scheduler.advance_epoch()
        
    return history
```

That's it! No rewrites needed.
