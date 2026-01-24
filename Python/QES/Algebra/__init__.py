"""
QES Algebra Module
==================

Core components for defining quantum systems.

This module provides the building blocks for quantum mechanics simulations:
Hilbert spaces, Operators, and Hamiltonians. It handles basis management,
symmetries, and matrix construction.

Entry Points
------------
- :class:`Hamiltonian`: The main object for defining physical models.
- :class:`HilbertSpace`: Defines the state space (spins, fermions) and symmetries.
- :class:`Operator`: General operators (observables).

Flow
----
::

    Lattice (geometry)
        |
        v
    HilbertSpace (basis & symmetries)
        |
        v
    Hamiltonian (interactions)
        |
        +---> Matrix Building (ED)
        |
        +---> NQS Training (VMC)

Submodules
----------
- ``hamil``: Hamiltonian implementations.
- ``hilbert``: Hilbert space logic.
- ``symmetries``: Group theory and symmetry operations.
- ``Operator``: Operator definitions.
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
