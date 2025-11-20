"""
QES Neural Quantum States (NQS) Module
======================================

This module provides implementation of Neural Quantum States for 
variational quantum many-body calculations.

Modules:
--------
- nqs: Core Neural Quantum State implementations
- nqs_train: Training algorithms for NQS
- tdvp: Time-Dependent Variational Principle methods
- REF: Reference implementations and examples

File    : QES/NQS/__init__.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION  = "Neural Quantum States (models, training, and TDVP methods)."

# Expose submodules without wildcard imports to keep import time light
_submods            = []

try:
    from . import nqs as nqs
    _submods.append('nqs')
except Exception:
    pass

try:
    from . import nqs_train as nqs_train
    _submods.append('nqs_train')
except Exception:
    pass

try:
    from . import tdvp as tdvp
    _submods.append('tdvp')
except Exception:
    pass

__all__ = _submods

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------