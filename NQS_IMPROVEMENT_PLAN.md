# NQS Improvement Plan - Kitaev Model & Quantum Spin Liquids

**Date:** November 1, 2025  
**Goal:** Enhance Neural Quantum State (NQS) implementation for Kitaev models, add learning phases, improve evaluation, and extend model support.

---

## Executive Summary

This document outlines a structured plan to:
1. **Learn & Study Phase**: Understand QSLs and Kitaev physics (leverage 3-day learning plan)
2. **Analyze & Audit**: Review current NQS code for design patterns, bottlenecks, and missing features
3. **Extend Models**: Implement Gamma-only model and improve site impurity handling
4. **Improve Architecture**: Add learning phases, better evaluation, autoregressive networks
5. **Optimize & Test**: Performance improvements, comprehensive testing, and documentation

---

## Phase 1: Analysis & Audit (Todo #1, #2, #3)

### 1.1 Current NQS Structure

**Key Files:**
- `nqs.py` (2122 lines) - Main NQS solver class
- `src/nqs_backend.py` - Backend abstraction (NumPy/JAX)
- `src/nqs_networks.py` - Network selection and initialization
- `src/nqs_physics.py` - Physics problems and interfaces

**Main Classes:**
- `NQS(MonteCarloSolver)` - Core solver
- `NQSSingleStepResult` - Single step result dataclass
- `MonteCarloSolver` - Parent class (handles sampling, etc.)

### 1.2 Questions to Answer

#### Learning Phases
- [ ] Are there explicit learning phases implemented?
- [ ] How is the training currently structured?
- [ ] What callbacks/hooks exist for phase transitions?
- [ ] Is there adaptive learning rate scheduling?

#### Evaluation Functions
- [ ] Which functions compute local energy?
- [ ] How many separate methods do this? (check for duplication)
- [ ] Are there bottlenecks in observable computation?
- [ ] Is evaluation properly batched?

#### Unnecessary Code
- [ ] Are there dead imports or unused methods?
- [ ] Are there duplicate implementations?
- [ ] What's the complexity of configuration parameters?
- [ ] Are there old commented-out blocks?

#### Model Support
- [ ] How is the Hamiltonian integrated?
- [ ] Can we easily add new interaction terms?
- [ ] Are site impurities working end-to-end in training?
- [ ] How flexible are the bond interactions?

---

## Phase 2: Implementation Roadmap

### 2.1 Learning Phases Feature (Todo #4)

**Goal:** Add structured training phases with clear transitions.

```python
@dataclass
class LearningPhase:
    name: str
    duration: int
    learning_rate: float
    batch_size: int
    num_samples: int
    regularization: Optional[float] = None
    callbacks: List[Callable] = field(default_factory=list)

class NQS:
    def __init__(self, ..., learning_phases: List[LearningPhase] = None):
        # Pre-training, main, refinement phases
        self.learning_phases = learning_phases or self._default_phases()
        self.current_phase_idx = 0
    
    def train(self, nsteps, use_learning_phases=True, **kwargs):
        # Support phase-based training
        if use_learning_phases:
            for phase in self.learning_phases:
                self._train_phase(phase)
        else:
            # Original training
            pass
    
    def _train_phase(self, phase: LearningPhase):
        """Execute a single training phase with callbacks."""
        for step in range(phase.duration):
            result = self.step(...)
            for callback in phase.callbacks:
                callback(phase, step, result)
            self._update_learning_params(phase)
```

### 2.2 Improved Evaluation (Todo #5)

**Goal:** Consolidate evaluation functions with caching and better documentation.

```python
class NQS:
    @cached_property
    def _local_energy_fn(self):
        """Compute local energy for a state."""
        # Consolidate existing implementations
        pass
    
    def compute_local_energy(self, states: Array, 
                           batch_size: Optional[int] = None,
                           params: Optional[Any] = None) -> Array:
        """
        Compute local energy for states.
        
        Parameters:
            states: Configuration array
            batch_size: Batch size for computation
            params: Network parameters
        
        Returns:
            Local energy array [num_samples]
        """
        pass
    
    def compute_observables(self, states: Array, operators: List[Operator],
                          **kwargs) -> Dict[str, Array]:
        """Compute multiple observables efficiently."""
        pass
```

### 2.3 Model Extensions (Todo #8)

**Goal:** Implement Gamma-only model class.

```python
# File: QES/Algebra/Model/Interacting/Spin/gamma_only.py
class GammaOnly(Hamiltonian):
    """
    Pure Gamma interaction model on honeycomb lattice.
    
    H = -Σ Γ (σᵢˣσⱼʸ + σᵢʸσⱼˣ)  [and cyclic permutations]
    
    Studies: Gapless QSL with fractionalized excitations.
    """
    def __init__(self, lattice: Lattice, Gamma: Union[float, List[float]], ...):
        pass
```

### 2.4 Autoregressive Networks (Todo #7)

**Goal:** Add autoregressive architectures for frustrated systems.

```python
# File: QES/general_python/ml/net_impl/networks/net_autoregressive.py
class AutoregressiveNetwork(Networks.GeneralNet):
    """
    Autoregressive network for quantum states.
    
    Represents ψ(s) = Π_i f(s_i | s_{1..i-1})
    Good for frustrated systems like Kitaev models.
    """
    def __init__(self, input_shape: tuple, hidden_dim: int = 64, ...):
        pass
```

---

## Phase 3: Testing Strategy (Todo #2, #9, #10)

### Current Tests
- `test_nqs_gs.ipynb` - Ground state training
- `test_nqs_time_evo.ipynb` - Time evolution
- `nqs_solver.py` - Solver tests
- `nqs_train.py` - Training profiling

### New Tests to Add
- [ ] Learning phase transitions
- [ ] Site impurity integration
- [ ] Gamma-only model validation
- [ ] Autoregressive network training
- [ ] Performance benchmarks (speed, memory)
- [ ] Kitaev model exact diagonalization comparison

---

## Phase 4: Documentation (Todo #11)

### Tutorials
1. **Learning Phases Guide** - How to structure multi-phase training
2. **Kitaev Model Training** - End-to-end example with HeisenbergKitaev
3. **Gamma-Only Model** - Studying gapless QSLs with pure Gamma interactions
4. **Custom Models** - Adding new interaction terms
5. **Performance Tips** - Optimization and profiling

---

## Phase 5: Cleanup & Optimization (Todo #6, #12, #13)

### Code Quality
- [ ] Remove unused imports and dead code
- [ ] Add comprehensive type hints
- [ ] Improve docstring coverage
- [ ] Simplify complex methods (>50 lines)

### Performance
- [ ] Profile memory usage
- [ ] Optimize JAX JIT compilation
- [ ] Cache expensive computations
- [ ] Parallelize where possible

---

## Implementation Order

**Week 1:**
1. Todo #1: Analyze current implementation
2. Todo #2: Run all tests, identify failures
3. Todo #3: Validate Kitaev model integration

**Week 2:**
4. Todo #4: Implement learning phases
5. Todo #5: Refactor evaluation methods

**Week 3:**
6. Todo #7: Add autoregressive networks
7. Todo #8: Implement Gamma-only model

**Week 4:**
8. Todo #9: Verify site impurities
9. Todo #10: Build comprehensive test suite
10. Todo #11: Create documentation

**Week 5:**
11. Todo #6: Cleanup code
12. Todo #12: Optimize performance
13. Todo #13: Improve clarity

---

## Key Metrics to Track

- **Correctness**: Test coverage (target: >90%)
- **Performance**: Training time/memory per step
- **Clarity**: Docstring completeness, type hint coverage
- **Maintainability**: Lines per method, cyclomatic complexity

---

## Resources

### Kitaev Physics (3-Day Learning Plan)
- **Day 1**: QSL concepts + Kitaev model intro
  - Trebst (2023) - Kitaev Magnets lecture notes
  - Adhikary et al. (2025) - Primer on Kitaev Model
  
- **Day 2**: Exact solution + excitations
  - Kitaev (2006) - Original paper
  - Zschocke (2016) - PhD thesis Chapter 2
  
- **Day 3**: Gamma-only model + ED implementation
  - Rousochatzakis & Perkins (2017)
  - Luo et al. (2021) - Gapless QSL in Gamma model
  - Matsuda et al. (2025) - Kitaev review (in RMP)

### Code References
- Current: `/Python/QES/NQS/`
- Model: `/Python/QES/Algebra/Model/Interacting/Spin/`
- Sampler: `/Python/QES/Solver/MonteCarlo/`
- Networks: `/Python/QES/general_python/ml/net_impl/networks/`

---

## Success Criteria

✓ Learning phases fully integrated with configurable transitions  
✓ All evaluation functions consolidated and optimized  
✓ Site impurities work end-to-end with NQS training  
✓ Gamma-only model implemented and tested  
✓ Autoregressive networks available for frustrated systems  
✓ Comprehensive test suite (>90% coverage)  
✓ Documentation with working tutorials  
✓ 20-30% speed improvement from optimization  
✓ Code clarity metrics: <50 LOC per method, >95% type hints, >90% docstring coverage  

---

## Notes

- Prioritize correctness over performance initially
- Maintain backward compatibility where possible
- Document assumptions and design decisions
- Consider computational cost (Kitaev systems grow as 2^N)
- Use small clusters for testing (N ≤ 24 sites typical for ED)
