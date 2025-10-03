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
- HilbertSpace: Main class for quantum many-body Hilbert spaces
- Hamiltonian: Hamiltonian matrix construction and operations
- Operator: General quantum mechanical operators

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

# Import main classes with explicit relative imports to avoid ambiguity
try:
    from .hilbert import HilbertSpace  # type: ignore
    from .hamil import Hamiltonian     # type: ignore
    from . import symmetries as _sym
    # Curate exported symmetry names (skip private/dunder)
    _sym_exports    = [n for n in dir(_sym) if not n.startswith("_")]
    __all__         = ['HilbertSpace', 'Hamiltonian', *_sym_exports]
except Exception:                                       # Broad except to keep package import resilient
    __all__         = ['HilbertSpace', 'Hamiltonian']   # Minimal exports if imports fail