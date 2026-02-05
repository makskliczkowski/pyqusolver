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

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

MODULE_DESCRIPTION = (
    "Algebra for quantum many-body: Hilbert spaces, Hamiltonians, operators, symmetries."
)

_LAZY_IMPORTS = {
    "HilbertSpace": (".hilbert", "HilbertSpace"),
    "HilbertConfig": (".hilbert_config", "HilbertConfig"),
    "SymmetrySpec": (".hilbert_config", "SymmetrySpec"),
    "Hamiltonian": (".hamil", "Hamiltonian"),
    "HamiltonianConfig": (".hamil_config", "HamiltonianConfig"),
    "HAMILTONIAN_REGISTRY": (".hamil_config", "HAMILTONIAN_REGISTRY"),
    "register_hamiltonian": (".hamil_config", "register_hamiltonian"),
    # backend helper re-exports kept for compatibility
    "identity": (".backends", "identity"),
    "inner": (".backends", "inner"),
    "kron": (".backends", "kron"),
    "outer": (".backends", "outer"),
    "overlap": (".backends", "overlap"),
    "trace": (".backends", "trace"),
    # package-level modules
    "Symmetries": (".Symmetries", None),
    "Operator": (".Operator", None),
    "Hilbert": (".Hilbert", None),
    "Hamil": (".Hamil", None),
}

_SYMMETRY_MODULE = ".Symmetries"
_LAZY_CACHE: dict[str, Any] = {}

if TYPE_CHECKING:
    from . import Hamil, Hilbert, Operator, Symmetries
    from .backends import identity, inner, kron, outer, overlap, trace
    from .hamil import Hamiltonian
    from .hamil_config import HAMILTONIAN_REGISTRY, HamiltonianConfig, register_hamiltonian
    from .hilbert import HilbertSpace
    from .hilbert_config import HilbertConfig, SymmetrySpec


__all__ = [
    "HilbertSpace",
    "HilbertConfig",
    "SymmetrySpec",
    "Hamiltonian",
    "HamiltonianConfig",
    "HAMILTONIAN_REGISTRY",
    "register_hamiltonian",
    "identity",
    "inner",
    "kron",
    "outer",
    "overlap",
    "trace",
    "Symmetries",
    "Operator",
    "Hilbert",
    "Hamil",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        result = module if attr_name is None else getattr(module, attr_name)
        _LAZY_CACHE[name] = result
        return result

    # Compatibility path: historical exports pulled from QES.Algebra.Symmetries.
    symm = importlib.import_module(_SYMMETRY_MODULE, package=__name__)
    if hasattr(symm, name):
        result = getattr(symm, name)
        _LAZY_CACHE[name] = result
        return result

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    symm = importlib.import_module(_SYMMETRY_MODULE, package=__name__)
    symmetry_names = [n for n in dir(symm) if not n.startswith("_")]
    return sorted(set(list(globals().keys()) + __all__ + list(_LAZY_IMPORTS.keys()) + symmetry_names))
