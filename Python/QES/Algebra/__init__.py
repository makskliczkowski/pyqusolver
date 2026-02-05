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

import importlib
from typing import TYPE_CHECKING, Any

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = (
    "Algebra for quantum many-body: Hilbert spaces, Hamiltonians, operators, symmetries."
)

_LAZY_IMPORTS = {
    # Core Classes
    "HilbertSpace": (".hilbert", "HilbertSpace"),
    "Hamiltonian": (".hamil", "Hamiltonian"),
    "Operator": (".Operator", None),

    # Config
    "HilbertConfig": (".hilbert_config", "HilbertConfig"),
    "SymmetrySpec": (".hilbert_config", "SymmetrySpec"),
    "HamiltonianConfig": (".hamil_config", "HamiltonianConfig"),
    "HAMILTONIAN_REGISTRY": (".hamil_config", "HAMILTONIAN_REGISTRY"),
    "register_hamiltonian": (".hamil_config", "register_hamiltonian"),

    # Submodules
    "Hamil": (".Hamil", None),
    "Hilbert": (".Hilbert", None),
    "Symmetries": (".Symmetries", None),
    "Model": (".Model", None),
    "backends": (".backends", None),

    # Backends utilities
    "identity": (".backends", "identity"),
    "inner": (".backends", "inner"),
    "kron": (".backends", "kron"),
    "outer": (".backends", "outer"),
    "overlap": (".backends", "overlap"),
    "trace": (".backends", "trace"),

    # Symmetries re-exports (from Symmetries)
    "TranslationSymmetry": (".Symmetries", "TranslationSymmetry"),
    "ReflectionSymmetry": (".Symmetries", "ReflectionSymmetry"),
    "InversionSymmetry": (".Symmetries", "InversionSymmetry"),
    "ParitySymmetry": (".Symmetries", "ParitySymmetry"),
    "SymmetryOperator": (".Symmetries", "SymmetryOperator"),
    "SymmetryClass": (".Symmetries", "SymmetryClass"),
    "MomentumSector": (".Symmetries", "MomentumSector"),
    "SymmetryRegistry": (".Symmetries", "SymmetryRegistry"),
    "choose": (".Symmetries", "choose"),
    "get_available_symmetries": (".Symmetries", "get_available_symmetries"),
    "build_momentum_superposition": (".Symmetries", "build_momentum_superposition"),
    "MomentumSectorAnalyzer": (".Symmetries", "MomentumSectorAnalyzer"),
    "CompactSymmetryData": (".Symmetries", "CompactSymmetryData"),
    "SymmetryContainer": (".Symmetries", "SymmetryContainer"),
}

if TYPE_CHECKING:
    from . import Hamil, Hilbert, Model, Operator, Symmetries, backends
    from .backends import identity, inner, kron, outer, overlap, trace
    from .hamil import Hamiltonian
    from .hamil_config import (
        HAMILTONIAN_REGISTRY,
        HamiltonianConfig,
        register_hamiltonian,
    )
    from .hilbert import HilbertSpace
    from .hilbert_config import HilbertConfig, SymmetrySpec
    from .Symmetries import (
        CompactSymmetryData,
        InversionSymmetry,
        MomentumSector,
        MomentumSectorAnalyzer,
        ParitySymmetry,
        ReflectionSymmetry,
        SymmetryClass,
        SymmetryContainer,
        SymmetryOperator,
        SymmetryRegistry,
        TranslationSymmetry,
        build_momentum_superposition,
        choose,
        get_available_symmetries,
    )


def __getattr__(name: str) -> Any:
    """Lazily import submodules and classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        if attr_name:
            return getattr(module, attr_name)
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


__all__ = [
    # Core
    "HilbertSpace",
    "Hamiltonian",
    "Operator",
    # Config
    "HilbertConfig",
    "SymmetrySpec",
    "HamiltonianConfig",
    "HAMILTONIAN_REGISTRY",
    "register_hamiltonian",
    # Submodules
    "Hamil",
    "Hilbert",
    "Symmetries",
    "Model",
    "backends",
    # Backends
    "identity",
    "inner",
    "kron",
    "outer",
    "overlap",
    "trace",
    # Symmetries
    "TranslationSymmetry",
    "ReflectionSymmetry",
    "InversionSymmetry",
    "ParitySymmetry",
    "SymmetryOperator",
    "SymmetryClass",
    "MomentumSector",
    "SymmetryRegistry",
    "choose",
    "get_available_symmetries",
    "build_momentum_superposition",
    "MomentumSectorAnalyzer",
    "CompactSymmetryData",
    "SymmetryContainer",
]
