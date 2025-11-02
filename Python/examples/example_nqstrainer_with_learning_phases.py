#!/usr/bin/env python3
"""
file        :   example_nqstrainer_with_learning_phases.py
author      :   Maksymilian Kliczkowski
date        :   November 1, 2025

Example: Using Learning Phases with NQSTrainer

This example shows how to:
1. Use LearningPhaseParameterScheduler with NQSTrainer (Option 2 integration)
2. Integrate phase estimation for adaptive learning
3. Create a complete training pipeline with multi-phase optimization

Key points:
- Learning phases are now external to NQS class
- NQSTrainer orchestrates training with learning phases
- Phase estimation can monitor convergence and adapt learning rates
- Fully backwards compatible with existing NQSTrainer code

Recommended usage pattern:
```python
# Create learning phase schedulers
lr_scheduler, reg_scheduler = create_learning_phase_schedulers('kitaev', logger=logger)

# Create NQSTrainer with learning phases
trainer = NQSTrainer(
    nqs=nqs,
    ode_solver=params.ode_solver,
    tdvp=params.tdvp,
    n_batch=params.numbatch,
    lr_scheduler=lr_scheduler,
    reg_scheduler=reg_scheduler,
    early_stopper=early_stopper,
    logger=logger
)

# Train with multi-phase optimization
history, history_std, timings = trainer.train(
    n_epochs=525,  # Total across all phases
    reset=False,
    use_lr_scheduler=True,
    use_reg_scheduler=True
)
```
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import learning phases components
from QES.NQS.src.learning_phases import (
    LearningPhase, LearningPhaseScheduler,
    create_learning_phases, PRESETS, PhaseType
)
from QES.NQS.src.learning_phases_scheduler_wrapper import (
    LearningPhaseParameterScheduler,
    create_learning_phase_schedulers
)
from QES.NQS.src.phase_estimation import (
    PhaseExtractor, PhaseEstimationConfig, PhaseEstimationType
)

# Import NQS and training components (would be used in actual code)
# from QES.NQS.nqs import NQS
# from QES.NQS.tdvp import TDVP
# from nqstrainer import NQSTrainer  # Your trainer class


def example_1_basic_learning_phase_scheduler():
    """
    Example 1: Create and use learning phase schedulers directly
    
    Shows how LearningPhaseParameterScheduler provides the standard
    scheduler interface that NQSTrainer expects.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Learning Phase Scheduler")
    print("="*70)
    
    # Create learning phases for Kitaev model (frustrated system)
    phases = create_learning_phases('kitaev')
    
    # Wrap for learning rate extraction
    lr_scheduler = LearningPhaseParameterScheduler(phases, param_type='learning_rate')
    
    # Wrap for regularization extraction
    reg_scheduler = LearningPhaseParameterScheduler(phases, param_type='regularization')
    
    print(f"\nLearning phase scheduler created:")
    print(f"  Total phases: {len(phases)}")
    print(f"  Total epochs: {sum(p.epochs for p in phases)}")
    print(f"  Total learning rate schedule: {lr_scheduler.get_schedule_description()}")
    
    # Simulate training loop
    print(f"\nSimulating 100 epochs:")
    for epoch in range(100):
        lr = lr_scheduler(epoch)
        reg = reg_scheduler(epoch)
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: lr={lr:.3e}, reg={reg:.3e}")
    
    print(f"  ✓ Scheduler history length: {len(lr_scheduler.history)}")
    return lr_scheduler, reg_scheduler


def example_2_factory_function():
    """
    Example 2: Use factory function for convenience
    
    Shows the recommended way to set up learning phases for NQSTrainer.
    """
    print("\n" + "="*70)
    print("Example 2: Factory Function for Learning Phase Schedulers")
    print("="*70)
    
    # One-liner to create both schedulers
    lr_scheduler, reg_scheduler = create_learning_phase_schedulers('default')
    
    print(f"\n✓ Created learning phase schedulers from 'default' preset")
    print(f"  LR Scheduler: {lr_scheduler}")
    print(f"  Reg Scheduler: {reg_scheduler}")
    
    return lr_scheduler, reg_scheduler


def example_3_all_presets():
    """
    Example 3: Explore all available presets
    
    Shows the different training strategies available.
    """
    print("\n" + "="*70)
    print("Example 3: Available Learning Phase Presets")
    print("="*70)
    
    print(f"\nAvailable presets:")
    for preset_name, phases in PRESETS.items():
        total_epochs = sum(p.epochs for p in phases)
        print(f"\n  {preset_name.upper()}:")
        print(f"    Total epochs: {total_epochs}")
        for i, phase in enumerate(phases, 1):
            print(f"    Phase {i}: {phase.name}")
            print(f"      - Epochs: {phase.epochs}")
            print(f"      - LR: {phase.learning_rate:.2e} (decay: {phase.lr_decay})")
            print(f"      - Reg: {phase.regularization:.2e} ({phase.reg_schedule})")


def example_4_phase_estimation():
    """
    Example 4: Phase estimation for adaptive learning
    
    Shows how to use phase estimation to monitor convergence
    and adapt learning rates.
    """
    print("\n" + "="*70)
    print("Example 4: Phase Estimation for Adaptive Learning")
    print("="*70)
    
    # Create phase extractor
    config = PhaseEstimationConfig(
        method=PhaseEstimationType.AMPLITUDE_PHASE,
        num_shots=1000,
        phase_precision=1e-2,
        regularization=1e-4
    )
    phase_extractor = PhaseExtractor(config)
    
    # Simulate amplitude evolution during training
    print(f"\nSimulating phase estimation during training:")
    phases_history = []
    
    for epoch in range(50):
        # Simulate complex amplitudes (would come from NQS evaluation)
        # Amplitude drifts over time as network learns
        phase_angle = epoch * 0.05  # Linear phase drift
        amplitude = np.exp(1j * phase_angle) * (1.0 - 0.01 * epoch)  # Decreasing magnitude
        
        # Estimate phase
        result = phase_extractor.estimate_amplitude_phase(np.array([amplitude]))
        phases_history.append(result.phase if isinstance(result.phase, float) else result.phase.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: phase={result.phase:.4f}, confidence={result.confidence:.3f}")
    
    # Analyze convergence
    phase_gradient = phase_extractor.compute_phase_gradient(
        np.array(phases_history), epoch=50
    )
    print(f"\nPhase gradient: {phase_gradient:.4f}")
    print(f"Should increase LR: {phase_extractor.should_update_learning_rate(phase_gradient)}")
    
    # Summary
    summary = phase_extractor.get_history_summary()
    print(f"\nPhase estimation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def example_5_nqstrainer_pattern():
    """
    Example 5: NQSTrainer usage pattern with learning phases
    
    Shows the complete pattern for using learning phases with NQSTrainer.
    This is NOT runnable without actual NQS/TDVP setup, but shows the structure.
    """
    print("\n" + "="*70)
    print("Example 5: NQSTrainer Usage Pattern (Pseudocode)")
    print("="*70)
    
    code = '''
# Step 1: Create learning phase schedulers
from QES.NQS.src.learning_phases_scheduler_wrapper import create_learning_phase_schedulers
from QES.general_python.common.flog import Logger

logger = Logger()

# Create both lr_scheduler and reg_scheduler from preset
lr_scheduler, reg_scheduler = create_learning_phase_schedulers(
    preset='kitaev',  # Optimized for frustrated systems
    logger=logger
)

# Step 2: Initialize NQSTrainer with learning phase schedulers
# (No changes to NQSTrainer code needed!)
from nqstrainer import NQSTrainer  # Your trainer module
from QES.general_python.common.stopping import EarlyStopping

early_stopper = EarlyStopping(
    patience=100,
    min_delta=1e-4,
    logger=logger
)

trainer = NQSTrainer(
    nqs=nqs,
    ode_solver=params.ode_solver,
    tdvp=params.tdvp,
    n_batch=params.numbatch,
    lr_scheduler=lr_scheduler,      # ← Learning phase scheduler
    reg_scheduler=reg_scheduler,    # ← Learning phase scheduler
    early_stopper=early_stopper,
    logger=logger
)

# Step 3: Train with multi-phase optimization
print("Starting multi-phase training with 525 epochs (Kitaev preset)")
print("  Phase 1: Pre-training (75 epochs, high LR)")
print("  Phase 2: Main optimization (300 epochs, adaptive LR)")
print("  Phase 3: Refinement (150 epochs, low LR)")

history, history_std, timings = trainer.train(
    n_epochs=525,
    reset=False,
    use_lr_scheduler=True,
    use_reg_scheduler=True
)

# Step 4: Analyze results
print(f"Training complete:")
print(f"  Final energy: {history[-1]:.6f}")
print(f"  Energy std: {history_std[-1]:.6f}")
print(f"  Total training time: {sum(timings.values()):.2f}s")

# Step 5 (Optional): Integrate phase estimation
from QES.NQS.src.phase_estimation import PhaseExtractor, PhaseEstimationConfig

phase_extractor = PhaseExtractor(PhaseEstimationConfig())

# During training, monitor phase evolution:
# for state, amplitude in samples:
#     phase_result = phase_extractor.estimate_amplitude_phase(amplitude)
#     if phase_result.confidence < 0.5:
#         print(f"Low confidence phase estimate - may need different LR schedule")
'''
    
    print(code)
    
    print("\n✓ Pattern summary:")
    print("  1. Create schedulers: create_learning_phase_schedulers(preset)")
    print("  2. Pass to NQSTrainer: lr_scheduler=..., reg_scheduler=...")
    print("  3. No changes to NQSTrainer code needed!")
    print("  4. Training automatically uses multi-phase optimization")
    print("  5. Optional: Monitor phase evolution for adaptive control")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("NQSTrainer + Learning Phases Integration Examples")
    print("="*70)
    
    # Example 1: Basic scheduler
    lr_sched, reg_sched = example_1_basic_learning_phase_scheduler()
    
    # Example 2: Factory function
    lr_sched2, reg_sched2 = example_2_factory_function()
    
    # Example 3: All presets
    example_3_all_presets()
    
    # Example 4: Phase estimation
    example_4_phase_estimation()
    
    # Example 5: NQSTrainer pattern (pseudocode)
    example_5_nqstrainer_pattern()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nKey takeaways:")
    print("  ✓ Learning phases are now external to NQS")
    print("  ✓ NQSTrainer is the main training orchestrator")
    print("  ✓ LearningPhaseParameterScheduler works as drop-in replacement")
    print("  ✓ Phase estimation enables adaptive optimization")
    print("  ✓ Backwards compatible with existing code")


if __name__ == "__main__":
    main()
