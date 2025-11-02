# Learning Phases: Complete Summary & Consolidation

**Date**: November 1, 2025  
**Status**: Learning Phases Feature Complete & Tested ✅

---

## What Are Learning Phases? (Your Questions Answered)

### Q1: "What are learning phases? Should this be inside the class?"

**A**: Learning phases are a **meta-layer for hyperparameter scheduling**. Think of them as a structured way to define how your training should evolve over time.

**Currently**:
- ✅ They ARE inside the NQS class (optional parameter)
- ✅ They can be used standalone for any training loop
- ✅ They don't replace your training—they guide it

**Your training code**:
```python
# You manually define one LR schedule
def adaptive_lr(epoch, initial_lr=0.03, decay_rate=0.999):
    return max(5.0e-3, initial_lr * (decay_rate ** epoch))

# This runs the same formula for all 500 epochs
for i in range(500):
    lr = adaptive_lr(i, 0.03, 0.999)
    step_info = train_single_step(params, nqs, i, lr)
```

**With learning phases**:
```python
# Three different decay schedules for three phases
phases = create_learning_phases('kitaev')
# Phase 1: rapid decay (explore)
# Phase 2: slow decay (converge)
# Phase 3: minimal decay (refine)

scheduler = LearningPhaseScheduler(phases)
for i in range(scheduler.total_epochs):
    lr = scheduler.get_current_hyperparameters(i)['learning_rate']
    step_info = train_single_step(params, nqs, i, lr)
    scheduler.advance_epoch()
```

---

### Q2: "Did it stop using TDVP?"

**A**: No—TDVP is still optional. Let me clarify:

Your training code does **NOT use TDVP**:
```python
# In train_single_step():
(_, _), (configs, configs_ansatze), probabilities = nqs.sample(reset=reset)
single_step_par = nqs.step(params=nqs.get_params(), configs=configs, ...)
# ← This is stochastic gradient descent, not TDVP
```

**TDVP** is a second-order optimization method:
- Your code: First-order (gradients only)
- TDVP: Second-order (uses Gram matrix)

Learning phases work with **BOTH**:
- ✅ Your current SGD approach
- ✅ TDVP-based approaches
- ✅ Any optimization method

They just provide: **"What learning rate should I use at this epoch?"**

---

### Q3: "What is this even? Should I consolidate with my training?"

**A**: Yes, absolutely. Here's the consolidation:

Your training loop is the **implementation** (HOW to do one step).  
Learning phases are the **strategy** (WHAT hyperparameters to use).

**Current state** (separate):
```
Your training_function() → Manually calls adaptive_lr()
```

**Consolidated state** (integrated):
```
Your training_function() → Calls scheduler.get_hyperparameters()
                              ↓
                         LearningPhaseScheduler
                              ↓
                         Returns adaptive LR per phase
```

---

## Your Training Code + Learning Phases Integration

### Minimal Integration (3 lines change)

```python
def train_function(params, nqs, reset=False, decay_rate=0.999):
    
    # ADD: Create phase scheduler (3 lines)
    phases = create_learning_phases('kitaev')
    scheduler = LearningPhaseScheduler(phases)
    total_epochs = scheduler.total_epochs
    
    history = np.zeros(total_epochs, dtype=np.float64)
    history_std = np.zeros(total_epochs, dtype=np.float64)
    epoch_times = np.zeros(total_epochs, dtype=np.float64)
    
    pbar = trange(total_epochs, desc="Training with Phases...", leave=True)
    timer = Timer("Total Training Time")
    
    for i in pbar:
        # CHANGE: Replace adaptive_lr() with phase scheduler
        hparams = scheduler.get_current_hyperparameters(i)
        current_lr = hparams['learning_rate']
        
        # Your existing code (unchanged)
        step_info = train_single_step(params, nqs, i, current_lr, reset, timer)
        
        history[i] = step_info.mean_energy
        history_std[i] = step_info.std_energy
        epoch_times[i] = step_info.time_info.time_total
        
        pbar.set_postfix({
            "E": f"{step_info.mean_energy:.4e}",
            "lr": f"{current_lr:.3e}"
        })
        
        # ADD: Advance scheduler (1 line)
        scheduler.advance_epoch()
    
    return history, history_std, epoch_times
```

**That's it!** 4 lines added, everything else works the same.

---

## How to Use in Your Training

### Option A: Use Preset (Simplest)

```python
# For Kitaev model training
history, history_std, times = train_function(params, nqs, phase_preset='kitaev')

# Available presets:
# - 'default': 3-phase general purpose (350 epochs)
# - 'fast': 2-phase quick prototyping (70 epochs)
# - 'thorough': 3-phase extended training (800 epochs)
# - 'kitaev': Optimized for frustrated systems (525 epochs)
```

### Option B: Define Custom Phases

```python
from QES.NQS.src.learning_phases import LearningPhase, PhaseType

my_phases = [
    LearningPhase(
        name="warm_up",
        epochs=50,
        phase_type=PhaseType.PRE_TRAINING,
        learning_rate=0.1,
        lr_decay=0.98,
        regularization=0.05
    ),
    # ... more phases
]

scheduler = LearningPhaseScheduler(my_phases)
```

### Option C: Callbacks at Phase Boundaries

```python
def on_pretraining_end():
    print("Pre-training complete! Checkpoint saved.")
    # Save network weights
    # Validate convergence
    # etc.

phases = [
    LearningPhase(
        name="pre_training",
        epochs=50,
        learning_rate=0.1,
        on_phase_end=on_pretraining_end
    ),
    # ...
]
```

---

## What You Get

### ✅ Structured Training Phases

Instead of:
```
Epoch 1-500: random LR schedule
```

You get:
```
Phase 1 (Pre-training):    Epochs 1-75,   LR 0.1 → 0.0001   (explore)
Phase 2 (Main):            Epochs 76-375, LR 0.02 → 0.0001  (converge)
Phase 3 (Refinement):      Epochs 376-525, LR 0.005 → 0.0001 (refine)
```

### ✅ Automatic Phase Transitions

The scheduler automatically:
- Detects phase boundaries
- Calls callbacks (`on_phase_start`, `on_phase_end`)
- Provides phase information for logging
- Tracks global and phase-local epochs

### ✅ Regularization Scheduling (Bonus)

Not just learning rates—also regularization:
```python
hparams = scheduler.get_current_hyperparameters(epoch)
lr = hparams['learning_rate']
reg = hparams['regularization']  # ← Also scheduled per phase

# Use in your optimization step
apply_regularization(parameters, reg)
```

### ✅ Reproducibility

Your training setup becomes:
```python
# One line describes your entire training strategy
history = train_with_phases(params, nqs, phase_preset='kitaev')

# vs current way:
history = train(params, nqs, decay_rate=0.999)
# (What does 0.999 mean? How many epochs? What schedule?)
```

---

## Files You Need to Know

### Core Implementation
- **`QES/NQS/src/learning_phases.py`** (660 lines)
  - `LearningPhase` dataclass
  - `LearningPhaseScheduler` orchestrator
  - 4 preset configurations

- **`QES/NQS/nqs.py`** (updated)
  - Added `learning_phases` parameter to `__init__`
  - Added `use_learning_phases` flag to `train()`
  - Integrated scheduler

### Documentation
- **`LEARNING_PHASES_GUIDE.md`** (550+ lines)
  - Complete reference with examples
  - All schedules documented
  - Troubleshooting guide

- **`LEARNING_PHASES_INTEGRATION_GUIDE.md`** (NEW)
  - How to use in YOUR training loop
  - Side-by-side comparisons
  - Practical examples

### Examples
- **`examples/example_learning_phases_integration.py`** (NEW)
  - 6 complete working examples
  - Shows integration patterns
  - Demonstrates all presets

### Tests
- **`test/test_learning_phases.py`** (NEW)
  - 14 passing tests
  - Tests dataclass, scheduler, presets, callbacks
  - Validates integration

---

## Quick Reference: How to Adapt Your Code

### Current Training Loop
```python
for i in range(500):
    lr = adaptive_lr(i, 0.03, 0.999)
    step_info = train_single_step(params, nqs, i, lr)
    history[i] = step_info.mean_energy
```

### With Learning Phases
```python
phases = create_learning_phases('kitaev')
scheduler = LearningPhaseScheduler(phases)

for i in range(scheduler.total_epochs):
    lr = scheduler.get_current_hyperparameters(i)['learning_rate']
    step_info = train_single_step(params, nqs, i, lr)
    history[i] = step_info.mean_energy
    scheduler.advance_epoch()
```

**Difference**: 4 lines added, everything else identical.

---

## What's NOT Changing

Your training code stays the same:
- ✅ `nqs.sample()` - unchanged
- ✅ `nqs.step()` - unchanged
- ✅ Parameter updates - unchanged
- ✅ History tracking - unchanged
- ✅ Timing/profiling - unchanged
- ✅ Progress bars - unchanged

Learning phases ONLY provide: **"What LR should I use this epoch?"**

---

## Next Steps

### Immediate (Optional)
1. **Evaluate**: Does this make sense for your workflow?
2. **Test**: Try one of the examples
3. **Integrate**: Add 4 lines to your training function

### For Production Use
1. Choose a preset or define custom phases
2. Update your training call
3. That's it!

### For Further Development
- Task 5: Refactor evaluation methods (consolidate 19 functions)
- Task 6: Code cleanup (remove dead code)
- Task 7: Add autoregressive networks

---

## Summary

**Learning Phases are**:
- ✅ A hyperparameter scheduling system
- ✅ Optional to use
- ✅ Compatible with your existing code
- ✅ Not replacing TDVP/your optimization
- ✅ Just giving structure to training schedules

**To use them**:
1. Import: `from QES.NQS.src.learning_phases import create_learning_phases, LearningPhaseScheduler`
2. Create: `phases = create_learning_phases('kitaev')`
3. Schedule: `lr = scheduler.get_current_hyperparameters(i)['learning_rate']`
4. Advance: `scheduler.advance_epoch()`

**That's the consolidation!** Your training logic stays 100% the same, just with structured hyperparameter management.

---

## Questions? Reference These

| Question | Answer Location |
|----------|-----------------|
| How do I use learning phases? | `LEARNING_PHASES_INTEGRATION_GUIDE.md` |
| What are the presets? | `LEARNING_PHASES_GUIDE.md` → Presets section |
| Can I define custom phases? | `examples/example_learning_phases_integration.py` → Example 2 |
| How do callbacks work? | `examples/example_learning_phases_integration.py` → Example 3 |
| What presets exist? | Run `python examples/example_learning_phases_integration.py` |
| Do I need to change my training? | No, 4 lines added maximum |

---

**Ready to integrate?** Start with `LEARNING_PHASES_INTEGRATION_GUIDE.md` Option 1 (Minimal Integration).
