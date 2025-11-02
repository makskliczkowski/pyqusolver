# NQS Working Document - Concrete Implementation Steps

**Started**: November 1, 2025  
**Status**: Analysis Complete, Ready for Implementation

---

## Quick Reference Guide

### What We Found

✅ **Good News:**
- Core NQS implementation is solid (~2000 LOC, well-structured)
- HeisenbergKitaev model already exists
- Backend abstraction (JAX/NumPy) in place
- Training infrastructure works

⚠️ **Areas for Improvement:**
- 19 energy/loss functions scattered throughout (consolidation needed)
- ~14 functions lack docstrings
- No explicit learning phases framework
- Site impurities: unclear if fully integrated
- Gamma-only model: doesn't exist
- Autoregressive networks: not available

### Timeline

- **Phase 1 (Days 1-2)**: Fix tests, code cleanup, verify impurities
- **Phase 2 (Days 3-5)**: Learning phases, evaluation consolidation
- **Phase 3 (Days 6-8)**: Gamma model, autoregressive networks
- **Phase 4 (Days 9-10)**: Testing, optimization, documentation

---

## PHASE 1: FOUNDATION & VALIDATION (Days 1-2)

### Task 1.1: Fix Broken Tests

**File**: `/Python/test/test_backends_interop.py` (line 49)

**Error**: `TypeError: 'NoneType' object is not subscriptable` in QuadraticHamiltonian

```python
# Issue Location:
qh.add_hopping(0, 1, -1.0)  # Line 49
# Fails in QES/Algebra/hamil_quadratic.py:862
# self._hamil_sp[i, j] += val  # _hamil_sp is None
```

**Investigation Needed:**
```bash
# Check QuadraticHamiltonian initialization
grep -n "_hamil_sp" /Users/makskliczkowski/Codes/pyqusolver/Python/QES/Algebra/hamil_quadratic.py | head -20
```

**Action Items:**
- [ ] Read hamil_quadratic.py __init__ (check initialization)
- [ ] Determine why _hamil_sp is None
- [ ] Fix initialization or add validation
- [ ] Run test again to verify

**Estimated Time**: 30-45 minutes

---

### Task 1.2: Code Audit Completion

**What We Know from Analysis:**

```
nqs.py:
  - 80 functions, 2122 LOC total
  - 2 functions > 100 LOC: __init__ (166) and eval_observables (173)
  - 14 functions undocumented
  - Key energy computation: local_energy, eval_observables, _apply_fun_jax

nqs_backend.py:
  - 4 backend classes (BaseBackend, NumpyBackend, JaxBackend, etc.)
  - 3 local_energy implementations (1 per backend)
  - Missing: Proper caching, batch optimization

nqs_physics.py:
  - 5 physics types: Wavefunction, Energy, TimeEvolution, etc.
  - Each has different loss() implementation
```

**Action Items:**

- [ ] Read `nqs.py` lines 1680-1750 (local_energy and eval_observables)
- [ ] Document all 19 energy functions:
  ```python
  # Create file: /local_energy_audit.txt
  # Format: filename | line | function | LOC | purpose
  ```
- [ ] Identify duplicates and overlaps
- [ ] Note where caching would help
- [ ] Identify which compute local energy vs loss vs something else

**Estimated Time**: 1-2 hours

**Files to Read:**
```bash
read_file nqs.py 1680-1750
read_file nqs_backend.py 90-210  
read_file nqs_physics.py 20-145
```

---

### Task 1.3: Verify Kitaev Model Integration

**Current State of HeisenbergKitaev (attachment)**:
- ✓ Takes `impurities` parameter: `List[Tuple[int, float]]`
- ✓ Stores as `self._impurities` 
- ✓ Counts in `_max_local_ch` calculation
- ❓ But: Is it actually used in local energy computation?

**Action Items:**

- [ ] Read HeisenbergKitaev._set_local_energy_operators() (line 261-409)
  ```bash
  read_file heisenberg_kitaev.py 261-410
  ```
- [ ] Search for where `self._impurities` is used
  ```bash
  grep -n "_impurities" /Python/QES/Algebra/Model/Interacting/Spin/heisenberg_kitaev.py
  ```
- [ ] Check if impurity terms are added to Hamiltonian
- [ ] Write small test:
  ```python
  # Create model with impurity at site 0
  model = HeisenbergKitaev(lattice, K=1.0, impurities=[(0, 0.5)])
  
  # Sample state and compute energy
  # Verify energy changes with/without impurity
  ```

**Estimated Time**: 45 minutes - 1 hour

---

### Task 1.4: Documentation of Current Code

**Create**: `/Python/NQS_CODE_MAP.md`

```markdown
# NQS Code Architecture Map

## Local Energy/Loss Computation Flow

### 1. Physics Problem Definition (nqs_physics.py)
- PhysicsInterface.loss(state) → float
- WavefunctionPhysics → ground state loss
- EnergyPhysics → energy expectation value
- TimeEvolutionPhysics → TDVP loss

### 2. Backend Computation (nqs_backend.py)
- Backend.local_energy(state, params)
- NumpyBackend vs JaxBackend implementations
- Sparse vs dense matrix handling

### 3. NQS Integration (nqs.py)
- NQS.compute_local_energy() [wrapper]
- NQS.eval_observables() [observation computation]
- NQS._apply_fun_jax() [ansatz evaluation]

## Learning Flow

### Sampling
NQS.sample() → (configs, probabilities)

### Evaluation  
NQS.ansatz(configs) → amplitudes/probabilities
NQS.compute_local_energy(configs) → local energies
Physics.loss(local_energies) → scalar loss

### Gradients
NQS.log_derivative(configs) → parameter gradients

### Update
NQS.update_parameters(gradients, learning_rate)
```

**Estimated Time**: 30 minutes

---

## PHASE 2: CONSOLIDATION (Days 3-5)

### Task 2.1: Consolidate Evaluation Functions

**Goal**: Single unified interface instead of 19 scattered functions

**Design**:

```python
# In nqs.py - new methods

def compute_local_energy(self,
                        states: Array,
                        batch_size: Optional[int] = None,
                        params: Optional[Any] = None,
                        cache: bool = True) -> Tuple[Array, Array]:
    """
    Compute local energy for given configurations.
    
    Returns:
        (local_energies, variances) both [batch_size]
    """
    if params is None:
        params = self.get_params()
    if batch_size is None:
        batch_size = self.batch_size
    
    # Route to backend
    if self.backend == 'jax':
        return self._compute_local_energy_jax(states, batch_size, params)
    else:
        return self._compute_local_energy_numpy(states, batch_size, params)

def compute_observable(self,
                      operator: Operator,
                      states: Array,
                      **kwargs) -> float:
    """Compute expectation value <ψ|O|ψ>"""
    pass

def compute_observables(self,
                       operators: Dict[str, Operator],
                       states: Array,
                       **kwargs) -> Dict[str, float]:
    """Compute multiple observables efficiently."""
    pass
```

**Action Items:**

- [ ] Create `_compute_local_energy_jax()` method
- [ ] Create `_compute_local_energy_numpy()` method  
- [ ] Move all energy logic into these unified methods
- [ ] Update all callers to use unified interface
- [ ] Add @cache or similar for expensive ops
- [ ] Write tests verifying old and new give same results

**Files to Modify:**
- nqs.py (main)
- nqs_backend.py (helpers)
- nqs_physics.py (if needed)

**Estimated Time**: 4-6 hours

---

### Task 2.2: Implement Learning Phases

**Goal**: Multi-phase training with phase transitions

**Implementation**:

```python
# In nqs.py

@dataclass  
class LearningPhase:
    """Single training phase configuration."""
    name: str
    num_steps: int
    learning_rate: float
    batch_size: int  
    num_samples: int
    callbacks: List[Callable] = field(default_factory=list)
    regularization: Optional[float] = None

class NQS(MonteCarloSolver):
    
    def __init__(self, ..., learning_phases=None, **kwargs):
        # ... existing code ...
        self.learning_phases = learning_phases or self._default_phases()
        self.current_phase_idx = 0
    
    def _default_phases(self) -> List[LearningPhase]:
        """Three-phase default training."""
        return [
            LearningPhase("pre-train", 50, 1e-2, 256, 1024),
            LearningPhase("main-train", 200, 5e-3, 512, 2048),
            LearningPhase("refinement", 100, 1e-3, 1024, 4096),
        ]
    
    def train(self, nsteps=None, use_phases=True, **kwargs):
        """Train with optional phase structure."""
        if not use_phases:
            return self._train_classic(nsteps, **kwargs)
        
        results = []
        for phase in self.learning_phases:
            phase_results = self._train_phase(phase)
            results.extend(phase_results)
        return results
    
    def _train_phase(self, phase: LearningPhase):
        """Execute single phase with callbacks."""
        results = []
        for step in range(phase.num_steps):
            # Set phase params
            self._set_batch_size(phase.batch_size)
            
            # Training step
            result = self.step(...)
            results.append(result)
            
            # Callbacks
            for callback in phase.callbacks:
                callback(phase=phase, step=step, result=result)
        
        return results
```

**Action Items:**

- [ ] Add LearningPhase dataclass to nqs.py
- [ ] Add learning_phases parameter to __init__
- [ ] Implement _default_phases()
- [ ] Refactor train() method
- [ ] Implement _train_phase()
- [ ] Add callback execution
- [ ] Write tests for phase transitions
- [ ] Create example: phases for Kitaev model

**Estimated Time**: 4-6 hours

---

## PHASE 3: EXTENSIONS (Days 6-8)

### Task 3.1: Implement Gamma-Only Model

**File**: `/Python/QES/Algebra/Model/Interacting/Spin/gamma_only.py` (NEW)

**Specification**:

```python
class GammaOnly(Hamiltonian):
    """
    Pure Gamma interaction honeycomb model.
    
    H = -Σ_{bonds} [
        Γ_x (σ_i^x σ_j^y + σ_i^y σ_j^x)  [z-bonds]
        Γ_y (σ_i^y σ_j^z + σ_i^z σ_j^y)  [x-bonds]
        Γ_z (σ_i^z σ_j^x + σ_i^x σ_j^z)  [y-bonds]
    ]
    
    Studies: Gapless QSL with competing phases
    References: Luo et al. (2021), Rousochatzakis & Perkins (2017)
    """
    
    def __init__(self,
                 lattice: Lattice,
                 Gamma_x: Union[float, List[float]] = 1.0,
                 Gamma_y: Union[float, List[float]] = 1.0,
                 Gamma_z: Union[float, List[float]] = 1.0,
                 hx: Optional[Union[float, List[float]]] = None,
                 hz: Optional[Union[float, List[float]]] = None,
                 **kwargs):
        """
        Initialize Gamma-only model.
        
        Parameters:
            lattice: Honeycomb lattice
            Gamma_x, Gamma_y, Gamma_z: Coupling strengths
            hx, hz: Optional external fields
        """
        pass
    
    def _set_local_energy_operators(self):
        """Setup local energy computation."""
        # Similar to HeisenbergKitaev but only Gamma terms
        pass
```

**Action Items:**

- [ ] Read HeisenbergKitaev implementation
- [ ] Create GammaOnly as "light" version (only Gamma)
- [ ] Implement bond-directional coupling
- [ ] Test on small cluster
- [ ] Verify against literature (Luo et al.)
- [ ] Write tests

**Estimated Time**: 3-4 hours

---

### Task 3.2: Add Autoregressive Networks

**File**: `/Python/QES/general_python/ml/net_impl/networks/net_autoregressive.py` (NEW)

**Specification**:

```python
class AutoregressiveNet(GeneralNet):
    """
    Autoregressive neural network for quantum states.
    
    Represents: ψ(s) = exp(Σ_i c_i) * Π_i f(s_i | s_{<i}, θ)
    
    Benefits for frustrated systems:
    - No symmetry assumptions
    - Better for entangled states
    - Flexible capacity
    
    References: Mohamed et al. (2019), Szilágyi & Troyer (2021)
    """
    
    def __init__(self, 
                 input_shape: Tuple,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 **kwargs):
        """
        Parameters:
            input_shape: (N_spins,)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        pass
    
    def forward(self, x):
        """
        Autoregressive forward pass.
        
        Process sites sequentially:
        - Site i depends only on s_1 ... s_{i-1}
        """
        pass
```

**Implementation Options:**

1. **Simple Causal Dense** (easiest)
   - Linear layers with causal masking
   
2. **Transformer-based** (more complex)
   - Causal self-attention
   - Better scaling to large systems

3. **Recurrent-based** (good balance)
   - LSTM/GRU cores
   - Sequential processing built-in

**Action Items:**

- [ ] Choose implementation approach (recommend: causal dense)
- [ ] Read existing networks (RBM, simple flax)
- [ ] Implement AutoregressiveNet
- [ ] Add to network factory (nqs_networks.py)
- [ ] Test on small Kitaev cluster
- [ ] Benchmark vs RBM
- [ ] Add documentation

**Estimated Time**: 5-8 hours

---

## PHASE 4: TESTING & OPTIMIZATION (Days 9-10)

### Task 4.1: Comprehensive Test Suite

**Create**: `/Python/test/test_nqs_learning_phases.py` (NEW)

```python
def test_learning_phases_structure():
    """Verify phase structure and transitions."""
    pass

def test_phase_parameter_changes():
    """Verify batch size, learning rate change between phases."""
    pass

def test_phase_callbacks():
    """Verify callbacks are called correctly."""
    pass

def test_kitaev_with_phases():
    """Train Kitaev model with learning phases."""
    pass

def test_kitaev_with_impurities():
    """Train Kitaev model with site impurities."""
    pass

def test_gamma_only_model():
    """Train pure Gamma model."""
    pass

def test_autoregressive_network():
    """Train with autoregressive network."""
    pass
```

**Create**: `/Python/test/test_kitaev_integration.py` (NEW)

```python
def test_kitaev_hamiltonian():
    """Verify Kitaev model energy computation."""
    pass

def test_kitaev_ground_state():
    """Compare NQS result with known GS energy."""
    pass

def test_kitaev_observables():
    """Verify magnetization, correlations, etc."""
    pass

def test_kitaev_with_field():
    """Test field-induced transitions."""
    pass
```

**Action Items:**

- [ ] Write all tests above
- [ ] Aim for >90% code coverage
- [ ] Add performance benchmarks
- [ ] Create conftest.py for fixtures
- [ ] Document test fixtures and utilities

**Estimated Time**: 6-8 hours

---

### Task 4.2: Documentation & Examples

**Create**: `/Docs/LEARNING_PHASES_GUIDE.md` (NEW)

```markdown
# Using Learning Phases in NQS

## Overview

Learning phases allow structured multi-stage training:
- **Pre-training**: Quick exploration, large learning rate
- **Main training**: Careful optimization, medium rate
- **Refinement**: Fine-tuning, small learning rate

## Example: Kitaev Model

```python
from QES.NQS import NQS
from QES.NQS.nqs import LearningPhase

# Define phases
phases = [
    LearningPhase("pre-train", 50, 1e-2, 256, 1024),
    LearningPhase("main", 300, 5e-3, 512, 2048),
    LearningPhase("refine", 150, 1e-3, 1024, 4096),
]

# Create NQS with phases
nqs = NQS(..., learning_phases=phases)

# Train with phases
results = nqs.train(use_phases=True)
```

## Customization

### Adaptive Learning Rate

```python
def schedule(step):
    return 1e-2 * (0.999 ** step)

phase = LearningPhase(
    "adaptive",
    100,
    1e-2,  # Initial LR
    256,
    1024,
    lr_schedule=schedule
)
```

### Phase Callbacks

```python
def on_phase_step(phase, step, result):
    if step % 10 == 0:
        print(f"{phase.name}: step {step}, E={result.loss}")

phase = LearningPhase(
    "train",
    100,
    5e-3,
    512,
    2048,
    callbacks=[on_phase_step]
)
```
```

**Create**: Jupyter notebooks:
- `Kitaev_Model_Training.ipynb` - Complete tutorial
- `Gamma_Only_Model.ipynb` - Pure Gamma model study
- `Learning_Phases_Example.ipynb` - Phase training demo

**Action Items:**

- [ ] Write learning phases guide
- [ ] Write Kitaev model tutorial
- [ ] Write Gamma model tutorial
- [ ] Create example notebooks
- [ ] Add to QUICK_REFERENCE.md

**Estimated Time**: 3-4 hours

---

## IMPLEMENTATION CHECKLIST

### Phase 1 (Days 1-2)
- [ ] Fix test_backends_interop.py
- [ ] Fix test_comprehensive_suite.py
- [ ] Complete code audit (19 energy functions documented)
- [ ] Verify Kitaev + impurities integration
- [ ] Create NQS_CODE_MAP.md
- [ ] All tests passing

### Phase 2 (Days 3-5)
- [ ] Consolidate evaluation functions
  - [ ] compute_local_energy() unified interface
  - [ ] compute_observable(s) methods
  - [ ] All backends updated
  - [ ] Tests pass (backward compatibility)
- [ ] Implement learning phases
  - [ ] LearningPhase dataclass
  - [ ] Modified train() method
  - [ ] Phase transitions working
  - [ ] Tests for phases
- [ ] Update NQS.__init__() documentation

### Phase 3 (Days 6-8)
- [ ] Gamma-only model
  - [ ] GammaOnly class implemented
  - [ ] Local energy computation
  - [ ] Tests pass
- [ ] Autoregressive networks
  - [ ] AutoregressiveNet class
  - [ ] Integrated with network factory
  - [ ] Tests pass
- [ ] Both models work with NQS training

### Phase 4 (Days 9-10)
- [ ] Comprehensive test suite (>90% coverage)
  - [ ] Learning phases tests
  - [ ] Kitaev integration tests
  - [ ] Gamma model tests
  - [ ] Autoregressive network tests
  - [ ] Performance benchmarks
- [ ] Documentation complete
  - [ ] Learning phases guide
  - [ ] Kitaev tutorial
  - [ ] Gamma model tutorial
  - [ ] API documentation updated
- [ ] Notebooks functional and clear
- [ ] Performance optimized (20-30% improvement)
- [ ] Code clarity improved (>95% type hints, <50 LOC/method)

---

## File Changes Summary

### New Files
- `/Python/QES/Algebra/Model/Interacting/Spin/gamma_only.py`
- `/Python/QES/general_python/ml/net_impl/networks/net_autoregressive.py`
- `/Python/test/test_nqs_learning_phases.py`
- `/Python/test/test_kitaev_integration.py`
- `/Docs/LEARNING_PHASES_GUIDE.md`
- Notebooks: Kitaev, Gamma, Learning Phases examples

### Modified Files
- `/Python/QES/NQS/nqs.py` - Add phases, consolidate evaluation
- `/Python/QES/NQS/src/nqs_networks.py` - Add autoregressive support
- `/Python/QES/NQS/src/nqs_backend.py` - Unified local energy
- `/Python/test/test_*.py` - Fix existing test failures

### Documentation Updates
- `/Python/QUICK_REFERENCE.md` - Add examples
- `/Python/README.md` - Add new features
- Update API docs with new methods

---

## Success Metrics

1. **Correctness**: All tests pass, >90% coverage
2. **Performance**: 20-30% faster training loop
3. **Clarity**: >95% type hints, all functions <50 LOC (except eval_observables split)
4. **Documentation**: 5+ working tutorial notebooks
5. **Functionality**: Can train on Kitaev and Gamma models with site impurities
6. **Features**: Learning phases work smoothly, autoregressive networks available

---

## Notes & Tips

### Code Navigation
```bash
# Find all local_energy functions
grep -rn "def local_energy" /Python/QES/NQS/

# Find all eval_observables calls
grep -rn "eval_observables" /Python/

# Find all HeisenbergKitaev usage
grep -rn "HeisenbergKitaev" /Python/test/
```

### Testing Strategy
```bash
# Run specific test file
python -m pytest test/test_nqs_learning_phases.py -v

# Run with coverage
python -m pytest --cov=QES/NQS test/

# Profile test
python -m cProfile -s cumulative test/nqs_train.py
```

### Debugging Tips
- Use `nqs.get_params()` to inspect network state
- Check `nqs.sampler.samples` for sampling quality
- Log `compute_local_energy()` output for debugging
- Use `nqs.local_energy` callback for diagnostics

---

## Contact & Questions

If stuck on any task:
1. Check the linked paper/reference
2. Search existing code for similar patterns
3. Write a minimal test case
4. Review git history for context

---

**Next Step**: Start Phase 1, Task 1.1 - Fix test_backends_interop.py

**Estimated Total Time**: ~95-110 hours (can parallelize some tasks)

**Recommended Pace**: 
- 10-15 hours/day over 7-10 days
- Or 6-8 hours/day over 12-14 days

Good luck! 🚀
