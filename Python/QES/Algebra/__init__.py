"""QES algebra package.

Purpose
-------
Provide core mathematical objects used across QES exact-diagonalization and
variational workflows:

- Hilbert-space definitions,
- operator algebra,
- Hamiltonian construction,
- symmetry-aware basis handling.

Input/output contracts
----------------------
- Inputs are expected to be numerically typed arrays or scalar parameters
  consistent with the target Hilbert space.
- Basis-dependent APIs assume states/operators are aligned to the same ordering.
- Matrix-like outputs are typically NumPy/JAX arrays or sparse-compatible
  structures depending on backend path.

Dtype and shape expectations
----------------------------
- Complex-valued dtypes are common for quantum operators/states.
- Vector states are typically shape ``(dim,)`` and operators are shape
  ``(dim, dim)`` after basis expansion.
- Mixed backend pipelines should avoid implicit dtype promotion when combining
  NumPy and JAX objects.

Numerical stability notes
-------------------------
- Near-degenerate spectra can amplify solver tolerance effects.
- Large Hilbert spaces may require sparse or iterative methods to reduce memory
  pressure and floating-point accumulation error.

Determinism notes
-----------------
- Determinism depends on backend behavior and seeded stochastic components.
- For Monte Carlo or randomized routines, users should set explicit seeds using
  the public QES seed helpers.
"""

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = (
    "Algebra for quantum many-body: Hilbert spaces, Hamiltonians, operators, symmetries."
)

# Import main classes with explicit relative imports to avoid ambiguity
try:
    from . import Symmetries as _sym  # type: ignore
    from .backends import identity, inner, kron, outer, overlap, trace  # type: ignore
    from .hamil import Hamiltonian  # type: ignore
    from .hamil_config import (  # type: ignore
        HAMILTONIAN_REGISTRY,
        HamiltonianConfig,
        register_hamiltonian,
    )
    from .hilbert import HilbertSpace  # type: ignore
    from .hilbert_config import HilbertConfig, SymmetrySpec  # type: ignore

    # Curate exported symmetry names (skip private/dunder)
    _sym_exports = [n for n in dir(_sym) if not n.startswith("_")]
    __all__ = [
        "HilbertSpace",
        "HilbertConfig",
        "SymmetrySpec",
        "Hamiltonian",
        "HamiltonianConfig",
        "HAMILTONIAN_REGISTRY",
        "register_hamiltonian",
        *_sym_exports,
    ]
except Exception as e:  # Broad except to keep package import resilient
    import warnings
    warnings.warn(f"QES.Algebra import failed: {e}")
    __all__ = [
        "HilbertSpace",
        "Hamiltonian",
        "HilbertConfig",
        "SymmetrySpec",
        "HamiltonianConfig",
        "HAMILTONIAN_REGISTRY",
        "register_hamiltonian",
    ]  # Minimal exports if imports fail

# ----------------------------------------------------------------------------
#! EOF
# ---------------------------------------------------------------------------
