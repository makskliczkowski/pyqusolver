"""
Practical example: Using Learning Phases with your training loop.

This shows how to integrate learning phases into your existing training code
with minimal modifications.
"""

import numpy as np
from tqdm import trange
from pathlib import Path
import sys

# Your existing imports would go here
# from your_setup import SimulationParams, train_single_step, prepare_nqs, etc.

# Add learning phases import
from QES.NQS.src.learning_phases import (
    LearningPhase, 
    PhaseType, 
    LearningPhaseScheduler,
    create_learning_phases
)


# ==============================================================================
# EXAMPLE 1: Minimal Integration (Recommended)
# ==============================================================================

def train_function_enhanced(params,
                           nqs,
                           reset=False,
                           phase_preset='kitaev',
                           **kwargs):
    """
    Your existing training function, enhanced with learning phases.
    
    Changes: 
    - Get LR from phase scheduler instead of adaptive_lr()
    - Optional: Display phase information
    - Optional: Call phase callbacks
    """
    
    # Setup phase scheduler (3 new lines)
    phases = create_learning_phases(phase_preset)
    scheduler = LearningPhaseScheduler(phases)
    total_epochs = scheduler.total_epochs
    
    # Initialize history (unchanged)
    history = np.zeros(total_epochs, dtype=np.float64)
    history_std = np.zeros(total_epochs, dtype=np.float64)
    epoch_times = np.zeros(total_epochs, dtype=np.float64)
    phase_info = []
    
    # Training loop (with 2 modifications marked below)
    pbar = trange(total_epochs, desc="Training with Learning Phases...", leave=True)
    
    for i in pbar:
        # CHANGE 1: Get LR from phase scheduler instead of adaptive_lr()
        hparams = scheduler.get_current_hyperparameters(i)
        current_lr = hparams['learning_rate']
        
        # Your existing step function (unchanged)
        # step_info = train_single_step(params=params, nqs=nqs, i=i, 
        #                               lr=current_lr, reset=reset)
        
        # For this example, we'll simulate it:
        step_info_mean = np.random.randn() * 0.01  # Simulate energy
        step_info_std = np.abs(np.random.randn() * 0.005)
        
        # CHANGE 2: Update progress bar with phase information
        phase = scheduler.current_phase
        phase_name = phase.name
        phase_epoch = scheduler.phase_epoch  # Epoch within current phase
        phase_total = phase.epochs
        
        pbar.set_description(
            f"Phase: {phase_name} ({phase_epoch}/{phase_total})"
        )
        pbar.set_postfix({
            "E": f"{step_info_mean:.4e}",
            "\sigmaE": f"{step_info_std:.4e}",
            "lr": f"{current_lr:.3e}",
            "reg": f"{hparams['regularization']:.3e}"
        })
        
        # Track history (unchanged)
        history[i] = step_info_mean
        history_std[i] = step_info_std
        
        # Optional: Track phase transitions
        if scheduler.phase_changed and i > 0:
            phase_info.append({
                'epoch': i,
                'phase_name': phase_name,
                'phase_type': phase.phase_type.name,
                'initial_lr': phase.learning_rate
            })
            print(f"\n-> Transitioned to {phase.phase_type.name} phase: {phase_name}")
            
            # Optional: Call phase callbacks
            if phase.on_phase_start is not None:
                phase.on_phase_start()
        
        # Advance scheduler to next epoch (1 new line)
        scheduler.advance_epoch()
    
    return history, history_std, phase_info


# ==============================================================================
# EXAMPLE 2: Define Custom Phases for Your System
# ==============================================================================

def create_custom_phases_for_kitaev():
    """
    Create custom phases tailored to Kitaev model training.
    
    This lets you define your training strategy as structured phases.
    """
    
    phases = [
        LearningPhase(
            name="rapid_exploration",
            epochs=50,
            phase_type=PhaseType.PRE_TRAINING,
            learning_rate=0.15,      # High: explore phase space
            lr_decay=0.98,           # Fast decay: settle quickly
            lr_min=1e-5,
            regularization=0.08,     # High regularization: stabilize
            reg_schedule='exponential',
            loss_type='energy'
        ),
        
        LearningPhase(
            name="steady_convergence",
            epochs=250,
            phase_type=PhaseType.MAIN,
            learning_rate=0.03,      # Medium: balanced progress
            lr_decay=0.9995,         # Slow decay: steady improvement
            lr_min=1e-5,
            regularization=0.01,     # Lower regularization
            reg_schedule='linear',
            loss_type='combined'     # Could use energy+variance
        ),
        
        LearningPhase(
            name="precision_refinement",
            epochs=150,
            phase_type=PhaseType.REFINEMENT,
            learning_rate=0.008,     # Low: fine-tune
            lr_decay=0.998,          # Very slow decay
            lr_min=5e-5,
            regularization=0.002,    # Minimal regularization
            reg_schedule='adaptive',
            loss_type='energy'
        )
    ]
    
    return phases


# ==============================================================================
# EXAMPLE 3: Phase-Based Training with Callbacks
# ==============================================================================

def create_phases_with_callbacks():
    """
    Create phases with callbacks for monitoring and checkpointing.
    """
    
    def on_pretraining_start():
        print("-> Pre-training phase started: exploring phase space")
    
    def on_pretraining_end():
        print("-> Pre-training phase complete: checking convergence")
    
    def on_main_start():
        print("-> Main training phase: focus on energy minimization")
    
    def on_main_end():
        print("-> Main training phase complete: achieved convergence")
    
    phases = [
        LearningPhase(
            name="pre_training",
            epochs=50,
            phase_type=PhaseType.PRE_TRAINING,
            learning_rate=0.1,
            lr_decay=0.98,
            regularization=0.05,
            on_phase_start=on_pretraining_start,
            on_phase_end=on_pretraining_end
        ),
        
        LearningPhase(
            name="main_training",
            epochs=200,
            phase_type=PhaseType.MAIN,
            learning_rate=0.03,
            lr_decay=0.999,
            regularization=0.01,
            on_phase_start=on_main_start,
            on_phase_end=on_main_end
        ),
        
        LearningPhase(
            name="refinement",
            epochs=100,
            phase_type=PhaseType.REFINEMENT,
            learning_rate=0.01,
            lr_decay=0.995,
            regularization=0.005
        )
    ]
    
    return phases


# ==============================================================================
# EXAMPLE 4: Compare Presets
# ==============================================================================

def compare_phase_presets():
    """
    Show the different preset configurations available.
    """
    
    presets = ['default', 'fast', 'thorough', 'kitaev']
    
    for preset_name in presets:
        phases = create_learning_phases(preset_name)
        total_epochs = sum(p.epochs for p in phases)
        
        print(f"\n{preset_name.upper()} Preset:")
        print(f"  Total epochs: {total_epochs}")
        for i, phase in enumerate(phases, 1):
            print(f"  Phase {i}: {phase.name}")
            print(f"    - Type: {phase.phase_type.name}")
            print(f"    - Epochs: {phase.epochs}")
            print(f"    - LR: {phase.learning_rate:.3e} -> {phase.lr_min:.3e}")
            print(f"    - Reg schedule: {phase.reg_schedule}")


# ==============================================================================
# EXAMPLE 5: Advanced - Mixed Phases
# ==============================================================================

def create_mixed_phases():
    """
    Mix preset phases with custom ones for maximum flexibility.
    """
    
    # Start with default preset
    phases_list = create_learning_phases('default')
    
    # Extend with a custom continuation phase
    extended_phase = LearningPhase(
        name="extended_refinement",
        epochs=50,
        phase_type=PhaseType.CUSTOM,
        learning_rate=0.001,
        lr_decay=0.99,
        regularization=0.001,
        reg_schedule='constant'
    )
    
    phases_list.append(extended_phase)
    
    return phases_list


# ==============================================================================
# EXAMPLE 6: Training Loop Comparison
# ==============================================================================

def show_training_comparison():
    """
    Show side-by-side comparison of old vs new training approach.
    """
    
    print("\n" + "="*70)
    print("TRAINING APPROACH COMPARISON")
    print("="*70)
    
    # OLD WAY (your current approach)
    print("\nOLD WAY (Single adaptive LR):")
    print("""
    def adaptive_lr(epoch, initial_lr, decay_rate):
        return max(5.0e-3, initial_lr * (decay_rate ** epoch))
    
    for i in range(500):
        lr = adaptive_lr(i, 0.03, 0.999)  # Same formula all 500 epochs
        step_info = train_single_step(params, nqs, i, lr)
        history[i] = step_info.mean_energy
    """)
    
    # NEW WAY (with learning phases)
    print("\nNEW WAY (Structured phases):")
    print("""
    phases = create_learning_phases('kitaev')
    scheduler = LearningPhaseScheduler(phases)
    
    for i in range(scheduler.total_epochs):
        lr = scheduler.get_current_hyperparameters(i)['learning_rate']
        step_info = train_single_step(params, nqs, i, lr)
        history[i] = step_info.mean_energy
        scheduler.advance_epoch()  # Handles phase transitions automatically
    """)
    
    print("\nKEY DIFFERENCES:")
    print("  • Old: Single global decay schedule")
    print("  • New: Phase-specific decay schedules")
    print("  • Old: Manual tracking of what's happening")
    print("  • New: Automatic phase transition detection")
    print("  • Old: Hard-coded 'adaptive_lr' function")
    print("  • New: Flexible preset configurations")


# ==============================================================================
# MAIN: Usage Examples
# ==============================================================================

if __name__ == "__main__":
    print("Learning Phases Integration Examples")
    print("=" * 70)
    
    # Example 1: Show preset comparison
    print("\n1. Available Presets:")
    compare_phase_presets()
    
    # Example 2: Show custom phases
    print("\n2. Custom Phases for Kitaev:")
    custom_phases = create_custom_phases_for_kitaev()
    total = sum(p.epochs for p in custom_phases)
    print(f"   Total epochs: {total}")
    for phase in custom_phases:
        print(f"   - {phase.name}: {phase.epochs} epochs, LR schedule: {phase.lr_decay}")
    
    # Example 3: Show phase-based training comparison
    print("\n3. Training Approach Comparison:")
    show_training_comparison()
    
    # Example 4: Instantiate scheduler
    print("\n4. Instantiating Scheduler with 'kitaev' preset:")
    phases = create_learning_phases('kitaev')
    scheduler = LearningPhaseScheduler(phases)
    print(f"   Scheduler created with {len(scheduler.phases)} phases")
    print(f"   Total epochs to train: {scheduler.total_epochs}")
    print(f"   Current phase: {scheduler.current_phase.name}")
    
    # Simulate a few epoch transitions
    print("\n   Simulating 100 epoch iterations:")
    prev_phase = scheduler.current_phase.name
    for epoch in range(100):
        current_phase_name = scheduler.current_phase.name
        if current_phase_name != prev_phase and epoch > 0:
            print(f"   -> Phase transition at epoch {epoch}: {current_phase_name}")
            prev_phase = current_phase_name
        scheduler.advance_epoch()
    
    if not scheduler.is_finished:
        print(f"\n   Final phase: {scheduler.current_phase.name}")
        print(f"   Epochs completed: {scheduler.global_epoch}")
    
    print("\n" + "=" * 70)
    print("Integration complete! Use these patterns in your training code.")
    print("=" * 70)
