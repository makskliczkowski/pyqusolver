# QES (Quantum EigenSolver) Development Guide

## Project Overview

QES is a modular Python framework for quantum many-body physics: exact diagonalization, Neural Quantum States (NQS), variational Monte Carlo, and symmetry-enhanced simulations. The codebase supports dual backend computation (NumPy/JAX) for CPU/GPU acceleration.

## Architecture: Core Components

### 1. Global Singleton Pattern (`QES.qes_globals`)
**CRITICAL**: Never create new loggers, backend managers, or RNGs. Always import from `qes_globals`:

```python
from QES.qes_globals import get_logger, get_backend_manager, get_numpy_rng, next_jax_key

log         = get_logger()
backend_mgr = get_backend_manager()
xp          = backend_mgr.np  # NumPy or JAX numpy
rng         = get_numpy_rng()
```

- `qes_globals` prevents double initialization and race conditions across modules
- Backend switching happens once per process via `backend_mgr`
- Use `backend_mgr.seed_scope(seed)` for deterministic blocks

### 2. Registry-Driven Declarative API
The package uses registries for Hamiltonians, operators, and Hilbert spaces. Remember to that one can use declarative `Config` + `from_config()`:

```python
from QES.Algebra import HilbertConfig, HamiltonianConfig, Hamiltonian

# Define via config
hilbert_cfg = HilbertConfig(ns=4, is_manybody=False, part_conserv=True)
quad_cfg    = HamiltonianConfig(
              kind        = "quadratic",
              hilbert     = hilbert_cfg,
              parameters  = {"ns": 4, "particle_conserving": True}
          )

# Instantiate via registry
ham         = Hamiltonian.from_config(quad_cfg)
```

- Registry keys: `HAMILTONIAN_REGISTRY`, operator catalogs in [QES/Algebra/Operator/catalog.py](Python/QES/Algebra/Operator/catalog.py)
- Check available models: `HAMILTONIAN_REGISTRY.list_available()`
- Register new types via `@dataclass` specs (see [hamil_config.py](Python/QES/Algebra/hamil_config.py))

### 3. Backend Abstraction (`QES.general_python.algebra.utils`)
Switch backends via environment variables **before** importing QES:

```bash
export PY_BACKEND=jax  # or 'numpy' (default)
export PY_FLOATING_POINT=float64  # or 'float32'
```

Code backend-agnostically:
```python
backend_mgr = get_backend_manager()
xp          = backend_mgr.np  # Use 'xp' for backend-neutral code
dtype       = backend_mgr.default_dtype
```

- JAX backend enables `@jit`, automatic differentiation, GPU execution
- See [QES/Algebra/backends/__init__.py](Python/QES/Algebra/backends/__init__.py) for backend interface

### 4. Lazy Import System
Top-level `QES.__init__.py` uses lazy imports to minimize startup overhead:

```python
import QES  # Lightweight
from QES import Algebra, NQS, Solver        # Deferred until access
from QES import HilbertSpace, Hamiltonian   # Direct lazy facade
```

- Explore modules: `QES.list_modules()`, `QES.describe_module('NQS')`
- Implementation: [QES/registry.py](Python/QES/registry.py)

## Development Workflows

### Environment Setup
```bash
cd Python/
pip install -e ".[dev]"     # Development mode
pip install -e ".[jax,ml]"  # With JAX and ML extras
make dev-setup              # Installs dev deps + pre-commit hooks
```

### Testing
```bash
make test          # Run pytest on test/ and general_python/tests/
make test-all      # Test across Python versions (tox)
pytest test/test_comprehensive_suite.py::TestQuadratic::test_bdg_simple -v
```

### Code Quality
```bash
make format        # black + isort
make lint          # flake8
make type-check    # mypy
make check         # format + lint + type-check + test
```

### Build & Release
```bash
make build         # Create dist/ packages
make upload        # Upload to TestPyPI
```

## Coding Conventions

### 1. Performance-Critical Paths
Use `@numba.njit` for NumPy hot loops, `@jit` for JAX:

```python
import numba

@numba.njit(cache=True, fastmath=True)
def _compute_matrix_element(state_idx, hilbert_dim):
    # Pure NumPy code, no Python objects
    ...
```

- Examples: [QES/Algebra/Operator/operator.py](Python/QES/Algebra/Operator/operator.py) lines 1013+, 1921+
- JAX JIT: See [general_python/algebra/utilities/pfaffian_jax.py](Python/QES/general_python/algebra/utilities/pfaffian_jax.py)

### 2. Configuration Dataclasses
Use frozen dataclasses for immutable config objects:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MyConfig:
    ns: int
    parameters: dict = field(default_factory=dict)
```

- See [hamil_config.py](Python/QES/Algebra/hamil_config.py), [hilbert_config.py](Python/QES/Algebra/hilbert_config.py)

### 3. Basis Transformations
Hamiltonians and operators support multiple bases (e.g., real-space ↔ k-space). Register transformations:

```python
# In Hamiltonian subclass
@classmethod
def register_basis_transform(cls, from_basis: str, to_basis: str, handler):
    cls._basis_transforms[(from_basis, to_basis)] = handler

# Use
ham.to_basis("k-space")  # Dispatches registered handler
```

- Example: [example_basis_transformations.py](Python/examples/example_basis_transformations.py)
- Implementation: [hamil.py](Python/QES/Algebra/hamil.py) lines 292+

### 4. Symmetry Integration
Symmetry generators reduce Hilbert space dimensions. Pass `sym_gen` to `HilbertSpace`:

```python
from QES.Algebra.Symmetries import TranslationSymmetry

sym_gen = [TranslationSymmetry(lattice, sector=(0, 0))]
hilbert = HilbertSpace(lattice=lattice, sym_gen=sym_gen, ...)
```

- See [test_kitaev_symmetries.py](Python/test/test_kitaev_symmetries.py), [test_xxz_symmetries.py](Python/test/test_xxz_symmetries.py)

## Package Structure Reference

```
QES/
├── Algebra/          # Hilbert spaces, Hamiltonians, operators, symmetries
│   ├── hamil.py      # Base Hamiltonian class (1629 lines)
│   ├── hilbert.py    # HilbertSpace orchestrator
│   ├── Operator/     # Operator implementations (spin, fermion)
│   ├── Hamil/        # Hamiltonian specializations
│   ├── Symmetries/   # Symmetry generators and containers
│   └── backends/     # Backend abstraction layer
├── NQS/              # Neural Quantum States, TDVP, training
├── Solver/           # Monte Carlo samplers, optimization
└── general_python/   # Scientific utilities (backends, lattices, physics, ML)
```

## Key Files for New Contributors

- [PACKAGE_STRUCTURE.md](Python/PACKAGE_STRUCTURE.md) - Complete package layout
- [EXAMPLES.md](Python/EXAMPLES.md) - Runnable usage snippets
- [INSTALL.md](Python/INSTALL.md) - Installation guide with extras
- [Makefile](Python/Makefile) - Development automation commands
- [pyproject.toml](Python/pyproject.toml) - Package metadata and dependencies

## Common Pitfalls

1. **Avoid double initialization**: Import globals from `qes_globals`, not from individual modules
2. **Backend consistency**: Use `xp = backend_mgr.np`, never mix `numpy` and `jax.numpy` directly
3. **Config over constructors**: Prefer `Hamiltonian.from_config()` to enable registry lookups
4. **Lazy imports**: Don't import entire submodules at package init; see [__init__.py](Python/QES/__init__.py)
5. **Numba typing**: Numba JIT requires statically typed functions; avoid Python objects in `@njit` code
6. **Remember to register**: New Hamiltonians, operators, and symmetries must be registered to be discoverable
7. **Testing coverage**: Add tests for new features in appropriate test files under `test/`
8. **Documentation**: Update docstrings and examples when adding new functionality
9. **Memory efficiency**: Be mindful of large array allocations, especially with JAX on GPU ir HPC clusters
10. **Version control**: Follow branching and commit message conventions outlined in CONTRIBUTING.md
11. **Fast paths**: Profile critical code sections to ensure performance targets are met - use `@numba.njit` or JAX `@jit` as needed - try to match state-of-the-art implementations
for algorithms implemented (fast Pfaffian, Lanczos, etc.)
12. **Symmetry correctness**: When implementing new symmetries, validate sector reductions against full Hilbert space results
13. **Use provided utilities**: Leverage existing utility functions in `general_python` for common tasks (e.g., lattice generation, matrix operations). New common functions should be added here.
Never duplicate functionality. Never make general python depend on QES. Never make QES depend on examples. Never make QES depend on tests. Make sure dependencies are one-way only. Make lazy imports work properly.

## Examples for Common Tasks

### Add a new Hamiltonian model
1. Subclass `Hamiltonian` in `QES/Algebra/Model/`
2. Register in `hamil_config.py`: `HAMILTONIAN_REGISTRY.register(key="my_model", builder=...)`
3. Add tests in `test/test_comprehensive_suite.py`

### Extend backend support
1. Add backend-specific code in `general_python/algebra/backends/`
2. Register in `BackendManager` ([utils.py](Python/QES/general_python/algebra/utils.py))
3. Test with `PY_BACKEND=jax pytest test/test_backend_integration.py`

### Implement a new symmetry
1. Subclass `SymmetryBase` in `QES/Algebra/Symmetries/`
2. Implement `generate()`, `sector_labels()`, `reduce_hilbert()`
3. Test reduction correctness vs full space (see [test_kitaev_symmetries.py](Python/test/test_kitaev_symmetries.py))

---

**Contact**: Maksymilian Kliczkowski (maksymilian.kliczkowski@pwr.edu.pl)  
**License**: CC-BY-4.0
