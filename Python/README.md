# QES Python Package

**QES** (Quantum EigenSolver) is a comprehensive Python framework for quantum many-body eigenvalue problems, Neural Quantum States (NQS), and variational methods.

---

## Quick Import Examples

### Lightweight top-level import
```python
import QES

# Access core globals
logger = QES.get_logger()
backend_mgr = QES.get_backend_manager()

# Discover available modules
for mod in QES.list_modules():
    print(f"{mod['name']:<20} - {mod['description']}")

# Lazy subpackage access (only imports on first use)
from QES import Algebra, NQS, Solver
```

### Direct class imports
```python
# Lazy facade: import the main classes directly from QES
from QES import HilbertSpace, Hamiltonian

hs = HilbertSpace(ns=4, nhl=2)
```

### Modular subpackage imports
```python
# Explicit subpackage import
from QES.Algebra import HilbertSpace, Hamiltonian
from QES.Algebra.Operator import operators_spin

# Low-level access
from QES.Algebra.Hilbert import hilbert_jit_methods
```

---

## Module Discovery

Use `QES.list_modules()` and `QES.describe_module()` to explore the library:

```python
import QES

# List all top-level modules
modules = QES.list_modules(include_submodules=True)
for m in modules:
    print(f"{m['name']:<25} {m['description']}")

# Get description for a specific module
print(QES.describe_module('NQS'))
```

**Sample Output:**
```
Algebra                   Algebra for quantum many-body: Hilbert spaces, Hamiltonians, operators, symmetries.
Algebra.Hamil             High-level Hamiltonian class for quantum many-body systems.
Algebra.Hilbert           High-level Hilbert space class for quantum many-body systems.
Algebra.Operator          Operator classes and concrete operators (spin, fermions) with matrix builders.
NQS                       Neural Quantum States (models, training, and TDVP methods).
Solver                    Core solver interfaces and Monte Carlo-based solvers.
general_python            Shared scientific utilities: algebra backends, logging, lattices, maths, ML, physics.
```

---

## Package Structure

```markdown
QES/
  __init__.py             # Lazy top-level API, globals, registry
  qes_globals.py          # Singleton logger and backend manager
  registry.py             # Module discovery utilities
  Algebra/                # Hilbert spaces, Hamiltonians, operators, symmetries
    Hamil/
    Hilbert/
    Model/
    Operator/
    Properties/
  NQS/                    # Neural Quantum States, training, TDVP
  Solver/                 # Solvers (exact diag, Monte Carlo, etc.)
  general_python/         # Reusable scientific utilities (backend, lattices, maths, ML, physics)


---

## Documentation for General Utilities and Physics

- [General Python Utilities README](QES/general_python/README.md): Overview of all scientific and numerical tools (algebra, lattices, maths, ML, etc.)
- [Physics Utilities README](QES/general_python/physics/README.md): Advanced quantum/statistical physics tools (thermal, spectral, response, etc.)

See these for subpackage-specific details and links to further documentation.
```

## Physics Module Documentation

The `general_python/physics` module is a comprehensive toolkit for condensed matter and quantum statistical physics, with advanced features for thermal, spectral, and response calculations.

- [Physics Module Structure & API](./PHYSICS_MODULE.md): Directory structure, submodules, and capabilities
- [Physics Module: Mathematical Background](./PHYSICS_MATH.md): Mathematical descriptions of implemented algorithms and quantities
- [Physics Module: Usage Examples](./PHYSICS_EXAMPLES.md): Extended code examples for all major features

Additional walkthroughs (including interacting/quadratic Hamiltonian tutorials) live in [EXAMPLES.md](./EXAMPLES.md).

See these documents for detailed usage, mathematical background, and advanced examples.

## Installation

**Standard:**

```bash
pip install -e .
```

**With JAX support:**

```bash
pip install -e ".[jax]"
```

**With all optional dependencies:**

```bash
pip install -e ".[all]"
```

---

## Development

### Run Tests

```bash
pytest
# or
python -m unittest discover -s test
```

### Check Imports

```bash
python test/test_imports_lightweight.py
```

### Build Documentation

```bash
make docs
```

---

## Design Principles

1. **Modularity**: Each subpackage can be imported independently with minimal overhead.
2. **Lazy loading**: Top-level `QES` imports are lightweight; heavy modules load on first use.
3. **Discoverability**: `QES.list_modules()` and `MODULE_DESCRIPTION` strings help users navigate the library.
4. **Global singleton management**: Logger and backend manager are initialized once and shared across the package.
5. **Unified import paths**: All imports use the `QES.*` namespace to avoid ambiguity.

---

## License

CC-BY-4.0 (see [LICENSE](../LICENSE))

## Author

(C) Maksymilian Kliczkowski 2025
Date    : 2025
Email   : maxgrom97@gmail.com

---