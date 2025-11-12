# Symmetry Implementation Guide

## Table of Contents

1. [Overview](#overview)
2. [Theory Background](#theory-background)
3. [Architecture](#architecture)
4. [How to Add a New Symmetry](#how-to-add-a-new-symmetry)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

---

## Overview

The Quantum Eigensolver (QES) implements a unified, extensible framework for exploiting symmetries in quantum many-body systems. This guide explains:

- **Theoretical foundation** (group theory, irreps, representatives)
- **Software architecture** (containers, operators, compatibility)
- **Step-by-step instructions** for adding new symmetries
- **Examples** from existing implementations

**Key Design Principles:**

1. **Uniform Treatment**: All symmetries use the same interface
2. **Automatic Group Construction**: Build full symmetry group from generators
3. **Multi-Backend Support**: Integer, NumPy, and JAX representations
4. **Compatibility Checking**: Automatic validation of symmetry combinations
5. **Memory Efficiency**: On-the-fly representative finding with optional caching

---

## Theory Background

### General Formalism

For a Hamiltonian $\hat{H}$ with symmetry group $G$, we have:

$$[\hat{H}, U(g)]_- = 0, \quad \forall g \in G$$

The Hilbert space decomposes into symmetry sectors:

$$\mathcal{H} \simeq \bigoplus_\alpha \mathbb{K}^{n_\alpha} \otimes \mathbb{K}^{d_\alpha}, \quad \hat{H} \simeq \bigoplus_\alpha I_{n_\alpha} \otimes \hat{H}_\alpha$$

where:

- $\alpha$ labels irreducible representations (irreps)
- $d_\alpha$ is the dimension of the irrep
- $n_\alpha$ is its multiplicity

### Abelian vs Non-Abelian Symmetries

**Abelian Symmetries** (all irreps 1D):

- Examples: $U(1)$ particle conservation, discrete translations $C_L$, parity $\mathbb{Z}_2$
- Matrix blocks labeled uniquely by quantum numbers
- Can be diagonalized independently
- **Current Implementation Focus**

**Non-Abelian Symmetries** (irreps $d_\alpha > 1$):

- Examples: $SU(2)$ spin rotation, point groups
- Internal degeneracies within blocks (Schur's lemma)
- Requires Clebsch-Gordan coefficients
- **Planned for future development**

### Representative States and Orbits

Given a reference state $|r\rangle$, the symmetry-adapted orbit is:

$$|\tilde{r}_\alpha\rangle = \frac{d_\alpha}{\mathcal{N}_\alpha\sqrt{|G|}} \sum_{g \in G} \chi_\alpha(g) |g(r)\rangle$$

Normalization:

$$\mathcal{N}_\alpha = \sqrt{\sum_{g \in G: g(r) = r} \chi_\alpha(g)}$$

**Computational Strategy:**

- Store only representative configurations $\{\bar{r}\}$
- Reconstruct full orbit on-the-fly
- If $\mathcal{N} = 0$, state cannot generate non-zero state in that sector

### Example: XXZ Hamiltonian

For the XXZ chain with periodic boundary conditions:

$$\hat{H} = -\sum_{\ell=1}^L \left[\hat{b}^\dagger_\ell \hat{b}_{\ell+1} + \hat{b}^\dagger_{\ell+1} \hat{b}_\ell + \frac{\Delta}{2}(2\hat{N}_\ell - 1)(2\hat{N}_{\ell+1} - 1)\right]$$

**Available Symmetries:**

- Translation: $T$, momentum $k = 2\pi m / L$
- Reflection: $R$ (only at $k=0, \pi$)
- Parity: $P_x, P_y, P_z$
- $U(1)$: Particle number $N$

**Dimension Reduction:**

- Full space: $D = 2^L$
- With translation: $\sim 2^L / L$
- With translation + parity: $\sim 2^L / (2L)$ to $2^L / (4L)$
- With $U(1)$ at $N$ particles: $\binom{L}{N}$ per momentum sector

---

## Architecture

### Component Overview

```
QES/Algebra/Symmetries/
├── base.py                    # Base classes, enums, registry
├── symmetry_container.py      # Main container, group building
├── translation.py             # Translation symmetry
├── reflection.py              # Reflection symmetry
├── parity.py                  # Parity (spin-flip) symmetries
└── compatibility.py           # Compatibility checking utilities
```

### Class Hierarchy

```
SymmetryOperator (base.py)
    ├── apply_int(state: int) -> (int, complex)
    ├── apply_numpy(state: ndarray) -> (ndarray, complex)
    └── apply_jax(state: jnp.ndarray) -> (jnp.ndarray, complex)
    
SymmetryContainer (symmetry_container.py)
    ├── generators: List[SymmetryOperator]
    ├── global_symmetries: List[GlobalSymmetry]
    ├── symmetry_group: List[GroupElement]
    ├── build_group()
    ├── find_representative(state) -> (rep, phase)
    └── compute_normalization(state) -> norm

HilbertSpace (hilbert.py)
    ├── _sym_container: SymmetryContainer
    ├── mapping_: np.ndarray
    ├── normalization_: np.ndarray
    └── Uses container for all symmetry operations
```

### Data Flow

1. **Initialization**:

   ```python
   HilbertSpace(ns=10, sym_gen=[(SymmetryGenerators.T, 0), ...])
   ```

2. **Container Creation**:
   - Instantiates `SymmetryContainer`
   - Checks compatibility of generators
   - Creates symmetry operator instances
   - Builds full symmetry group

3. **Mapping Generation**:
   - Iterate all states in full Hilbert space
   - Check global symmetries (filter)
   - Find representative using container
   - Compute normalization
   - Store mapping: state_idx -> (representative_idx, norm)

4. **Matrix Element Computation**:
   - Operator acts: $|r\rangle \to |m\rangle$
   - Find representative: $|m\rangle \to |\bar{m}\rangle$
   - Look up in mapping
   - Return: $(\bar{m}_{\text{idx}}, \text{norm}_{\bar{m}} / \text{norm}_r \cdot \overline{\chi})$

---

## How to Add a New Symmetry

### Step-by-Step Guide

#### 1. Define the Symmetry Class

Create a new file in `QES/Algebra/Symmetries/` (e.g., `my_symmetry.py`):

```python
"""
My custom symmetry operation.

Description of what this symmetry does physically.
"""

import numpy as np
from typing import Tuple, Optional
from QES.Algebra.Symmetries.base import (
    SymmetryOperator, 
    SymmetryClass, 
    MomentumSector,
    LocalSpaceTypes
)
from QES.general_python.common.binary import popcount, int2binstr  # utility functions

try:
    from QES.general_python.algebra.utils import JAX_AVAILABLE
    if JAX_AVAILABLE:
        import jax.numpy as jnp
        from jax import jit
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

class MySymmetry(SymmetryOperator):
    """
    Implementation of My Symmetry.
    
    Physical Meaning
    ----------------
    [Explain what this symmetry represents physically]
    
    Quantum Numbers
    ---------------
    sector : [type]
        [Explain the quantum number, e.g., eigenvalue, momentum, etc.]
    
    Compatible Symmetries
    ---------------------
    - Always commutes with: [list]
    - Conditionally commutes with: [list + conditions]
    - Never commutes with: [list]
    
    Examples
    --------
    >>> sym = MySymmetry(sector=1, ns=10)
    >>> new_state, phase = sym.apply_int(5, ns=10)
    """
    
    # Class attributes for compatibility
    symmetry_class = SymmetryClass.GENERIC  # Change to appropriate class
    compatible_with = {
        SymmetryClass.TRANSLATION,
        SymmetryClass.U1_PARTICLE,
        # Add others that always commute
    }
    momentum_dependent = {
        MomentumSector.ZERO: {SymmetryClass.REFLECTION},
        MomentumSector.PI: {SymmetryClass.REFLECTION},
        # Add momentum-dependent compatibilities
    }
    supported_local_spaces = set()  # Empty = universal; or specify {LocalSpaceTypes.SPIN_1_2, ...}
    
    def __init__(
        self, 
        sector: int,
        ns: Optional[int] = None,
        lattice: Optional['Lattice'] = None,
        **kwargs
    ):
        """
        Initialize the symmetry operator.
        
        Parameters
        ----------
        sector : int
            Quantum number for this symmetry sector
        ns : int, optional
            Number of sites (if not using lattice)
        lattice : Lattice, optional
            Lattice structure
        **kwargs
            Additional parameters
        """
        self.sector = sector
        self.ns = ns if ns is not None else (lattice.ns if lattice else None)
        self.lattice = lattice
        
        # Validate
        if self.ns is None:
            raise ValueError("Must provide either ns or lattice")
    
    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """
        Apply symmetry to integer state.
        
        Parameters
        ----------
        state : int
            Integer representation of state (bitstring)
        ns : int
            Number of sites
        **kwargs
            nhl : int
                Local Hilbert space dimension (default: 2)
        
        Returns
        -------
        new_state : int
            Transformed state
        phase : complex
            Symmetry eigenvalue (typically e^{i*sector*something})
        
        Algorithm
        ---------
        1. [Describe transformation]
        2. [Compute phase]
        3. [Return result]
        
        Examples
        --------
        >>> sym = MySymmetry(sector=1, ns=4)
        >>> sym.apply_int(0b1010, ns=4)  # |1010>
        (6, (1+0j))  # |0110>, phase = 1
        """
        nhl = kwargs.get('nhl', 2)
        
        # Your transformation logic here
        # Example: bit operations, permutations, etc.
        new_state = state  # Replace with actual transformation
        
        # Compute phase based on sector and transformation
        phase = np.exp(1j * 2 * np.pi * self.sector / ns)  # Example
        
        return new_state, phase
    
    def apply_numpy(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, complex]:
        """
        Apply symmetry to numpy state vector.
        
        Parameters
        ----------
        state : np.ndarray
            State vector (shape: (nh,) or (ns,))
        **kwargs
            Additional parameters
        
        Returns
        -------
        new_state : np.ndarray
            Permuted/transformed state vector
        phase : complex
            Symmetry eigenvalue
        
        Notes
        -----
        For vector representation, you typically:
        1. Create permutation matrix or index mapping
        2. Apply permutation
        3. Multiply by phase
        """
        # Your vector transformation
        # Example: permutation, sign flips, etc.
        new_state = state.copy()  # Replace with transformation
        phase = 1.0  # Replace with actual phase
        
        return new_state, phase
    
    def apply_jax(self, state: 'jnp.ndarray', **kwargs) -> Tuple['jnp.ndarray', complex]:
        """
        Apply symmetry to JAX state vector.
        
        Should be decorated with @jax.jit for performance.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available")
        
        # JAX implementation (similar to numpy but JIT-compatible)
        new_state = state  # Replace with transformation
        phase = 1.0
        
        return new_state, phase
```

#### 2. Register in SymmetryGenerators Enum

Edit `QES/Algebra/Operator/operator.py`:

```python
class SymmetryGenerators(Enum):
    # Existing symmetries
    T = "translation"
    R = "reflection"
    PX = "parity_x"
    # ... others ...
    
    # Add your new symmetry
    MY_SYMMETRY = "my_symmetry"
```

#### 3. Add to Container Factory

Edit `QES/Algebra/Symmetries/symmetry_container.py` in `_create_symmetry_operator()`:

```python
def _create_symmetry_operator(...):
    """Factory to create symmetry operator instances."""
    try:
        # ... existing cases ...
        
        elif gen_type == SymmetryGenerators.MY_SYMMETRY:
            from QES.Algebra.Symmetries.my_symmetry import MySymmetry
            return MySymmetry(sector=sector, ns=ns, lattice=lattice)
        
        else:
            ...
```

#### 4. (Optional) Add Compatibility Rules

If your symmetry has special compatibility rules beyond what's in the class attributes, add them to `SymmetryCompatibility.check_pair_compatibility()` in `symmetry_container.py`:

```python
def check_pair_compatibility(self, gen1, gen2, sector1, sector2):
    """Check if two generators are compatible."""
    
    # ... existing rules ...
    
    # Special case: My Symmetry and Other Symmetry
    if {gen1, gen2} == {SymmetryGenerators.MY_SYMMETRY, SymmetryGenerators.OTHER}:
        # Check some condition
        if some_condition(sector1, sector2):
            return False, "Explanation why incompatible"
    
    return True, "Compatible"
```

#### 5. Write Tests

Create `test/symmetries/test_my_symmetry.py`:

```python
"""Tests for my symmetry implementation."""

import pytest
import numpy as np
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Operator.operator import SymmetryGenerators

def test_my_symmetry_basic():
    """Test basic application of my symmetry."""
    hilbert = HilbertSpace(
        ns=4,
        sym_gen=[(SymmetryGenerators.MY_SYMMETRY, 0)],
        gen_mapping=True
    )
    
    assert hilbert.Nh > 0
    assert hilbert.Nh <= hilbert.Nhfull

def test_my_symmetry_representatives():
    """Test representative finding."""
    # ... test logic ...

def test_my_symmetry_compatibility():
    """Test compatibility with other symmetries."""
    # ... test logic ...
```

#### 6. Document

Add documentation:

- **Docstrings** in your class
- **Examples** in docstrings
- **Entry in this guide** with physical meaning

---

## Examples

### Example 1: Custom Rotation Symmetry

Suppose you want to implement a discrete rotation symmetry $C_n$ (rotation by $2\pi/n$):

```python
class RotationSymmetry(SymmetryOperator):
    """Discrete rotation symmetry C_n."""
    
    symmetry_class = SymmetryClass.POINT_GROUP
    compatible_with = {SymmetryClass.U1_PARTICLE, SymmetryClass.U1_SPIN}
    
    def __init__(self, n_fold: int, sector: int, lattice: 'Lattice'):
        """
        Parameters
        ----------
        n_fold : int
            Order of rotation (e.g., 4 for C_4)
        sector : int
            Momentum index (0, 1, ..., n_fold-1)
        lattice : Lattice
            Must have appropriate symmetry
        """
        self.n_fold = n_fold
        self.sector = sector
        self.lattice = lattice
        self.ns = lattice.ns
        
        # Pre-compute site permutation under rotation
        self.permutation = self._compute_permutation()
    
    def _compute_permutation(self) -> np.ndarray:
        """Compute how sites permute under rotation."""
        # Use lattice geometry to determine permutation
        perm = np.zeros(self.ns, dtype=int)
        for i in range(self.ns):
            # Map site i to rotated position
            perm[i] = self.lattice.rotate_site(i, self.n_fold)
        return perm
    
    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """Apply C_n rotation to integer state."""
        nhl = kwargs.get('nhl', 2)
        
        # Extract occupations
        occupations = [(state >> i) & 1 for i in range(ns)]
        
        # Apply permutation
        new_occupations = [occupations[self.permutation[i]] for i in range(ns)]
        
        # Reconstruct state
        new_state = sum(occ << i for i, occ in enumerate(new_occupations))
        
        # Phase: e^{2pi i * sector / n_fold}
        phase = np.exp(2j * np.pi * self.sector / self.n_fold)
        
        return new_state, phase
```

### Example 2: Time-Reversal Symmetry

Time-reversal is anti-unitary (complex conjugation):

```python
class TimeReversalSymmetry(SymmetryOperator):
    """Time-reversal symmetry T (anti-unitary)."""
    
    symmetry_class = SymmetryClass.TIME_REVERSAL
    compatible_with = {SymmetryClass.TRANSLATION, SymmetryClass.REFLECTION}
    
    def __init__(self, sector: int, **kwargs):
        """
        Parameters
        ----------
        sector : int
            +/- 1 (even/odd under T)
        """
        self.sector = sector
    
    def apply_int(self, state: int, ns: int, **kwargs) -> Tuple[int, complex]:
        """
        For spin-1/2, T flips all spins and complex conjugates.
        
        In integer representation, we return the state and indicate
        complex conjugation via a special flag in kwargs.
        """
        # Flip all spins: NOT operation
        nhl = kwargs.get('nhl', 2)
        max_state = (nhl ** ns) - 1
        new_state = max_state - state  # Bit-flip all
        
        # Phase (sector eigenvalue)
        phase = self.sector  # +/- 1
        
        # Note: Anti-unitary nature handled at higher level
        # (when building matrix elements, take complex conjugate)
        
        return new_state, phase
    
    def apply_numpy(self, state: np.ndarray, **kwargs) -> Tuple[np.ndarray, complex]:
        """For vectors, flip spins and complex conjugate."""
        # Reverse ordering and conjugate
        new_state = np.conj(state[::-1])
        phase = self.sector
        return new_state, phase
```

---

## Best Practices

### Performance Tips

1. **JIT Compilation**: Decorate `apply_int()` with `@numba.njit` for speed:

   ```python
   from numba import njit
   
   @staticmethod
   @njit(cache=True)
   def _apply_int_core(state, ns, nhl, sector):
       # Core logic here (must be Numba-compatible)
       ...
       return new_state, phase
   
   def apply_int(self, state, ns, **kwargs):
       nhl = kwargs.get('nhl', 2)
       return self._apply_int_core(state, ns, nhl, self.sector)
   ```

2. **Pre-compute Tables**: For expensive operations (factorials, permutations), compute once in `__init__`:

   ```python
   def __init__(self, ...):
       ...
       self.lookup_table = self._build_lookup_table()
   ```

3. **Avoid Python Loops**: Use NumPy vectorization when possible in `apply_numpy()`.

4. **Cache Representatives**: For repeated use, build representative map:

   ```python
   hilbert = HilbertSpace(..., gen_mapping=True)  # builds mapping
   container.build_representative_map(nh_full)    # builds cache
   ```

### Memory Efficiency

1. **Don't Store Full Group**: `SymmetryContainer` stores group elements as tuples of operators, not explicit matrices.

2. **On-Demand Computation**: Representatives found on-the-fly during matrix construction, only cached if requested.

3. **Use Global Symmetries**: Filter states before representative finding:

   ```python
   # Good: U(1) filters states before expensive symmetry operations
   sym_gen = [...],
   global_syms = [get_u1_sym(n_particles)]
   ```

### Testing

1. **Test Identity**: Applying $g^{|G|}$ should return identity (up to phase).

2. **Test Commutation**: Verify compatibility claims:

   ```python
   state_1 = sym1.apply_int(sym2.apply_int(state, ns)[0], ns)[0]
   state_2 = sym2.apply_int(sym1.apply_int(state, ns)[0], ns)[0]
   assert state_1 == state_2  # If claimed to commute
   ```

3. **Test Representative Uniqueness**: Each orbit should have exactly one representative.

4. **Test Normalization**: $\mathcal{N} > 0$ for representatives, $\mathcal{N} = 0$ for disallowed states.

### Documentation

1. **Physical Meaning**: Explain what the symmetry represents physically.

2. **Quantum Numbers**: Clearly document sector values and their meaning.

3. **Compatibility**: List which symmetries it commutes/doesn't commute with.

4. **Examples**: Provide worked examples in docstrings.

5. **References**: Cite papers/textbooks for complex symmetries.

---

## Summary Checklist

When adding a new symmetry:

- [ ] Create class inheriting from `SymmetryOperator`
- [ ] Implement `apply_int()`, `apply_numpy()`, `apply_jax()`
- [ ] Set `symmetry_class`, `compatible_with`, `momentum_dependent`
- [ ] Add to `SymmetryGenerators` enum
- [ ] Add to `_create_symmetry_operator()` factory
- [ ] (Optional) Add special compatibility rules
- [ ] Write comprehensive tests
- [ ] Document with examples
- [ ] Run full test suite
- [ ] Update this guide with new symmetry

For questions or issues, contact: <maksymilian.kliczkowski@pwr.edu.pl>

---

**Version**: 2.0.0  
**Last Updated**: October 28, 2025  
**Author**: Maksymilian Kliczkowski
