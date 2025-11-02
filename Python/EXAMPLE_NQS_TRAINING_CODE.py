# Example NQS Training Code
# This is a reference implementation showing:
# 1. Complete simulation parameter setup
# 2. Component preparation (lattice, model, network, sampler, TDVP)
# 3. Training loop with excited states support
# 4. Single-step training procedure

# Note: This is provided as context for refactoring into Learning Phases framework

"""
Key features in this code:
- MultiState support (ground state + excited states)
- Adaptive learning rate scheduling
- Timer-based performance tracking
- Modular component preparation
- SR (Stochastic Reconfiguration) integration
- Early stopping and regularization schedulers

Potential improvements:
- Consolidate 19 energy functions (currently scattered)
- Implement multi-phase training structure
- Add learning phase callbacks and transitions
- Refactor long methods (train_single_step could be split)
- Add more comprehensive logging
"""

# CODE LOCATION: User provided in request
# TO BE INTEGRATED WITH: NQS_WORKING_DOCUMENT.md Task 2.2 (Learning Phases)
