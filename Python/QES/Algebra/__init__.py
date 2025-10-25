"""
QES Algebra Module
==================

This package contains modules for algebraic operations in the Quantum EigenSolver project.

Modules:
--------
- hilbert       : High-level Hilbert space class for quantum many-body systems
- hamil         : Hamiltonian construction and manipulation
- symmetries    : Symmetry operations and group theory
- Operator      : General operators in quantum mechanics
- Model         : Predefined quantum models (interacting and non-interacting)
- Hilbert       : Hilbert space utilities
- Properties    : Physical properties calculations
And more...

Classes:
--------
- HilbertSpace  : Main class for quantum many-body Hilbert spaces
- Hamiltonian   : Hamiltonian matrix construction and operations
- Operator      : General quantum mechanical operators

File    : QES/Algebra/__init__.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = "Algebra for quantum many-body: Hilbert spaces, Hamiltonians, operators, symmetries."

# Import main classes with explicit relative imports to avoid ambiguity
try:
    from .hilbert import HilbertSpace                           # type: ignore
    from .hilbert_config import HilbertConfig, SymmetrySpec     # type: ignore
    from .hamil import Hamiltonian                              # type: ignore
    from .hamil_config import (                                 # type: ignore
        HamiltonianConfig,
        HAMILTONIAN_REGISTRY,
        register_hamiltonian,
    )
    from . import symmetries as _sym    # type: ignore
    # Curate exported symmetry names (skip private/dunder)
    _sym_exports    = [n for n in dir(_sym) if not n.startswith("_")]
    __all__         = [
        'HilbertSpace',
        'HilbertConfig',
        'SymmetrySpec',
        'Hamiltonian',
        'HamiltonianConfig',
        'HAMILTONIAN_REGISTRY',
        'register_hamiltonian',
        *_sym_exports,
    ]
except Exception:                                       # Broad except to keep package import resilient
    __all__         = [
        'HilbertSpace',
        'Hamiltonian',
        'HilbertConfig',
        'SymmetrySpec',
        'HamiltonianConfig',
        'HAMILTONIAN_REGISTRY',
        'register_hamiltonian',
    ]   # Minimal exports if imports fail
    
# ----------------------------------------------------------------------------
