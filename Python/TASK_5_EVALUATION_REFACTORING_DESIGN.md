# Task 5: Evaluation Refactoring Design Document

## Executive Summary

This document outlines the refactoring strategy for consolidating 19 scattered evaluation/energy computation methods into a unified, modular interface. The goal is to improve code maintainability, reduce redundancy, and enable better integration with the operator framework.

**Status**: Planning Phase (Task 5.1 - Analysis)
**Priority**: High
**Expected Timeline**: 2-3 hours
**Risk Level**: Low (backwards compatible wrapper design)

---

## 1. Current State Analysis

### 1.1 Scattered Evaluation Methods in nqs.py

**Category 1: Ansatz Evaluation (3 methods)**
```
1. _eval_jax()          - Evaluates network using JAX backend
2. _eval_np()           - Evaluates network using NumPy backend  
3. ansatz()             - Public method dispatching to _eval_jax/_eval_np
```
**Purpose**: Compute log-wavefunction amplitudes $\log|\psi(s)|$ for given states
**Common Issues**: Duplicated backend dispatch logic, no unified interface

---

**Category 2: Function Application on Batches (5 methods)**
```
4. _apply_fun_jax()     - Apply function to JAX arrays with batching
5. _apply_fun_np()      - Apply function to NumPy arrays with batching
6. _apply_fun()         - Dispatcher for _apply_fun_jax/_apply_fun_np
7. _apply_fun_s()       - Apply functions using sampler (special case)
8. apply()              - Public method with states or sampler
```
**Purpose**: Evaluate arbitrary functions (e.g., local energy) on state batches
**Common Issues**: Multiple dispatch layers, sampler coupling, complex batching logic

---

**Category 3: Energy Evaluation (1 primary + implicit)**
```
9. step()               - Single training step (implicit energy eval)
10. _single_step_groundstate() - Core energy + gradient computation
```
**Purpose**: Compute local energies and gradients for optimization
**Common Issues**: Tightly coupled with gradient computation, hard to test independently

---

**Category 4: Utility/Wrapper Methods (3 methods)**
```
11. wrap_single_step_jax()    - JIT wrapper for single-step
12. log_prob_ratio()          - Compute probability ratios
13. apply_func                - Property returning apply function
14. eval_func                 - Property returning eval function
```
**Purpose**: Facilitate advanced use cases (JIT compilation, comparisons)
**Common Issues**: Limited reusability, scattered implementation logic

---

**Category 5: Special/Advanced Evaluation (7+ methods)**
```
15. eval_observables()       - Evaluate multiple observables
16. transform_flat_params()  - Parameter transformation
17. ansatz_modified          - Modified ansatz with operators
18. update_parameters()      - Parameter updates
19. set_modifier/unset_modifier - State modifier management
20. log_derivative()          - Gradient computation
21. (implicit) sampling functions
```
**Purpose**: High-level evaluation capabilities and parameterization
**Common Issues**: Scattered across class, not systematized

---

### 1.2 Dependencies on Operator Framework

**operator.py** provides:
- `OperatorFunction`: Base class for state-independent operators
- `Operator`: Full operator with Hamiltonian/observable support
- Backend dispatch: `.npy`, `.jax` properties
- JAX compilation support via `make_jax_operator_closure()`

**Current Usage**: Not used in NQS evaluation (opportunity for consolidation)

---

### 1.3 Backend Complexity

Current pattern (repeated 5+ times):
```python
if JAX_AVAILABLE and isinstance(s, jnp.ndarray) and self._fun_jax is not None:
    return self._fun_jax(s)
elif isinstance(s, np.ndarray) and self._fun_np is not None:
    return self._fun_np(s)
else:
    return self._fun_int(s)
```

**Problem**: Violates DRY principle, error-prone, hard to extend

---

## 2. Proposed Architecture

### 2.1 New Structure: Unified Evaluation Engine

```
ComputeLocalEnergy (NEW)
    ├── EvaluationEngine (backend abstraction)
    │   ├── JAXBackend (JIT compilation, vectorization)
    │   ├── NumpyBackend (efficient batch processing)
    │   └── MixedBackend (hybrid dispatch)
    ├── BatchProcessor (handles batching logic)
    ├── AnswerComputation (ansatz evaluation)
    └── ObservableEvaluation (operator-based observables)

NQS class (REFACTORED)
    ├── evaluate()     [NEW unified interface]
    ├── local_energy() [simplified wrapper]
    ├── step()         [unchanged externally]
    └── apply()        [delegates to ComputeLocalEnergy]
```

### 2.2 New ComputeLocalEnergy Interface

```python
class ComputeLocalEnergy:
    """Unified interface for all energy/observable computations."""
    
    def __init__(self, nqs: NQS, backend: str = 'auto'):
        """Initialize with NQS reference and backend selection."""
        
    def evaluate_ansatz(self, states, batch_size=None, params=None):
        """Compute log-wavefunction: log|ψ(s)|"""
        
    def evaluate_observable(self, operator, states, batch_size=None, 
                          params=None, as_expectation=True):
        """Evaluate <ψ|O|ψ> or local values O(s)"""
        
    def evaluate_energy(self, states, batch_size=None, params=None):
        """Compute local energies E_loc(s) = H*ψ / ψ"""
        
    def evaluate_function(self, func, states, batch_size=None, params=None):
        """General-purpose function evaluation framework"""
        
    def batch_evaluate(self, states, functions, batch_size=None, params=None):
        """Evaluate multiple functions efficiently"""
```

### 2.3 Benefits of New Design

| Aspect | Current | New | Improvement |
|--------|---------|-----|-------------|
| Code Duplication | 5+ backends | 1 unified | -80% redundancy |
| Testability | 19 scattered | 5 core methods | Better unit testing |
| Maintainability | Hard to extend | Plugin backend | Easy to add backends |
| Performance | Per-method JIT | Global optimization | 10-20% faster |
| Documentation | Scattered | Centralized | Single source of truth |
| API Surface | Inconsistent | Consistent | Easier for users |

---

## 3. Implementation Plan

### Phase 3.1: Create EvaluationEngine (File 1: ~400 lines)

**File**: `unified_evaluation_engine.py`

**Components**:
1. `EvaluationBackend` (abstract base)
2. `JAXBackend`, `NumpyBackend` implementations
3. `EvaluationConfig` dataclass
4. `EvaluationResult` dataclass
5. Batch processing utilities

**Key Features**:
- Automatic backend selection based on state type
- Unified dispatcher pattern
- JIT compilation support
- Error handling and validation

### Phase 3.2: Create ComputeLocalEnergy (File 2: ~300 lines)

**File**: `compute_local_energy.py`

**Components**:
1. `ComputeLocalEnergy` main class
2. Integration with EvaluationEngine
3. Operator framework compatibility
4. Backwards-compatible wrappers

### Phase 3.3: Refactor NQS to use ComputeLocalEnergy (File 3: nqs.py changes)

**Changes**:
1. Add ComputeLocalEnergy instance to NQS.__init__()
2. Delegate _eval_jax/_eval_np to engine
3. Delegate _apply_fun* to engine
4. Create unified evaluate() method
5. Maintain backwards compatibility via wrapper methods

**Maintenance**: ~200 lines modified, 0 lines deleted (backwards compat)

### Phase 3.4: Create Examples (File 4: ~300 lines)

**File**: `example_unified_evaluation.py`

**Sections**:
1. Basic ansatz evaluation
2. Observable evaluation
3. Energy computation
4. Batch processing patterns
5. Performance benchmarking
6. Backend switching

### Phase 3.5: Testing & Validation (Changes to test/)

**Tests**:
1. Unit tests for EvaluationEngine backends
2. Integration tests with NQS
3. Backwards compatibility verification
4. Performance benchmarks
5. Edge cases (empty batches, single states, etc.)

### Phase 3.6: Documentation (File 5: ~400 lines)

**File**: `EVALUATION_REFACTORING.md`

**Sections**:
1. Architecture overview with diagrams
2. Before/after comparisons
3. Migration guide for users
4. Performance characteristics
5. Future enhancement opportunities
6. Troubleshooting guide

---

## 4. Backwards Compatibility Strategy

### 4.1 Non-Breaking Changes

**Old Method** → **New Implementation**
```python
# OLD: nqs.ansatz(states)
# NEW: Delegates to ComputeLocalEnergy.evaluate_ansatz()
# Signature unchanged ✓

# OLD: nqs.apply(functions, states)
# NEW: Delegates to ComputeLocalEnergy.evaluate_function()
# Signature unchanged ✓

# OLD: nqs.step()
# NEW: Uses ComputeLocalEnergy internally
# Behavior unchanged ✓
```

### 4.2 Deprecation Path (Optional, Future)

```python
@deprecated("Use nqs.evaluate_ansatz() instead")
def ansatz(self, states, **kwargs):
    return self._eval_engine.evaluate_ansatz(states, **kwargs)
```

---

## 5. Key Design Decisions

### 5.1 Why ComputeLocalEnergy is Separate from NQS

**Rationale**:
- Single Responsibility Principle: NQS handles sampling/parameters, ComputeLocalEnergy handles evaluation
- Testability: Can test evaluation logic independently
- Reusability: Other classes can use ComputeLocalEnergy
- Separation of Concerns: Keeps NQS focused on network management

### 5.2 Why Unified Backend Abstraction

**Rationale**:
- DRY: Eliminate duplicated dispatch logic (5+ occurrences)
- Extensibility: Easy to add new backends (GPU, TPU, etc.)
- Performance: Global optimization opportunity
- Maintenance: Single point of backend logic

### 5.3 Why Integrate with Operator Framework

**Rationale**:
- operator.py provides `.npy` and `.jax` interface (standard pattern)
- Can leverage JAX compilation support from OperatorFunction
- Consistent with codebase conventions
- Enables better observable evaluation

---

## 6. Risk Analysis

### 6.1 Low Risk

- **Backwards compatibility**: Wrapper methods preserve old API
- **Testing**: Extensive examples and unit tests
- **Performance**: New design enables optimization, not regression

### 6.2 Medium Risk

- **Integration complexity**: Need to validate operator framework integration
- **Edge cases**: Batch edge cases (empty, single element)
- **Documentation**: Users need clear migration guide

### 6.3 Mitigation Strategies

- Create comprehensive test suite before refactoring NQS
- Use feature flags for gradual rollout
- Document all API changes clearly
- Provide multiple working examples

---

## 7. Success Criteria

- ✓ All 19 methods consolidated into 5 core methods
- ✓ 80%+ code duplication eliminated
- ✓ Zero breaking changes to public API
- ✓ 10+ integration examples provided
- ✓ Unit tests with 90%+ coverage
- ✓ Performance benchmarks documented
- ✓ Comprehensive architecture documentation
- ✓ Git commits with clear messages

---

## 8. File Structure After Refactoring

```
Python/QES/NQS/
├── src/
│   ├── __init__.py
│   ├── nqs.py (modified: +200 lines for integration)
│   ├── unified_evaluation_engine.py (new: 400 lines)
│   ├── compute_local_energy.py (new: 300 lines)
│   ├── learning_phases_scheduler_wrapper.py (existing)
│   ├── phase_estimation.py (existing)
│   └── [other files]
│
├── examples/
│   └── example_unified_evaluation.py (new: 300 lines)
│
└── docs/
    └── EVALUATION_REFACTORING.md (new: 400 lines)
```

---

## 9. Timeline & Checkpoints

| Phase | Task | Est. Time | Checkpoint |
|-------|------|-----------|------------|
| 5.1 | Analysis & Design | 30 min | ✓ This document |
| 5.2 | EvaluationEngine | 45 min | Unit tests pass |
| 5.3 | ComputeLocalEnergy | 30 min | Integration tests pass |
| 5.4 | NQS Integration | 30 min | All public methods work |
| 5.5 | Examples | 20 min | All examples run |
| 5.6 | Documentation | 20 min | Doc complete |
| 5.7 | Git Commit | 10 min | Commit to main |
| **Total** | **Task 5 Complete** | **2.5 hours** | **73% → 80%** |

---

## 10. Next Steps

1. **Immediate** (Next 5 min): Review this design with stakeholder
2. **Phase 5.2** (45 min): Implement EvaluationEngine
3. **Phase 5.3** (30 min): Implement ComputeLocalEnergy
4. **Phase 5.4** (30 min): Integrate with NQS
5. **Phase 5.5-5.7** (50 min): Examples, docs, commit

**Target Completion**: ~2.5 hours from now
**Projected Project Status**: 73% → 80% (Session 5)

---

## Appendix: Evaluation Method Inventory

### Complete List of 19+ Methods to Consolidate

1. `_eval_jax(states, batch_size, params)` - JAX ansatz eval
2. `_eval_np(states, batch_size, params)` - NumPy ansatz eval
3. `ansatz(states, batch_size, params)` - Ansatz dispatcher
4. `_apply_fun_jax(func, states, ...)` - JAX func application
5. `_apply_fun_np(func, states, ...)` - NumPy func application
6. `_apply_fun(func, states, ...)` - Func app dispatcher
7. `_apply_fun_s(func, sampler, ...)` - Sampler-based eval
8. `apply(functions, states_and_psi, ...)` - Public apply interface
9. `step(problem, configs, ...)` - Single training step
10. `_single_step_groundstate(params, configs, ...)` - Ground state step
11. `wrap_single_step_jax(batch_size)` - JIT wrapper
12. `log_prob_ratio(...)` - Probability ratio calculation
13. `apply_func` (property) - Apply function property
14. `eval_func` (property) - Eval function property
15. `eval_observables(operators, ...)` - Observable evaluation
16. `transform_flat_params(...)` - Parameter transformation
17. `ansatz_modified` (property) - Modified ansatz
18. `update_parameters(...)` - Parameter updates
19. `set_modifier/unset_modifier(...)` - State modifier management
20. `log_derivative(states, ...)` - Gradient computation
21. Additional: `sample()`, `sampler_func`, backend initialization

**Consolidation Strategy**: Group into 5 core methods:
1. `evaluate_ansatz()` - Consolidated from 1,2,3
2. `evaluate_function()` - Consolidated from 4,5,6,7,8
3. `evaluate_energy()` - Consolidated from 9,10,11
4. `batch_evaluate()` - New unified multi-function interface
5. `evaluate_observable()` - Consolidated from 15, new operator integration

---

**Document Status**: FINAL - Ready for Phase 5.2
**Last Updated**: Session 5, Task 5.1
**Next Phase**: Implementation of EvaluationEngine
