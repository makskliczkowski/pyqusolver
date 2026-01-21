"""
QES Solver Module
=================

This module contains various quantum eigenvalue solvers and computational methods.

Modules:
--------
- solver        : Base solver classes and implementations
- MonteCarlo    : Monte Carlo methods for quantum systems

Classes:
--------
- Solver        : Abstract base class for quantum eigenvalue solvers

File    : QES/Solver/__init__.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

from .solver import Solver

__all__ = ["Solver"]

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION = "Core solver interfaces and Monte Carlo-based solvers."

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
