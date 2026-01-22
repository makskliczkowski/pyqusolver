"""
Hilbert Space Utilities Module
==============================

This module contains utilities for Hilbert space operations and optimizations.

Modules:
--------
- hilbert_jit_methods       : JIT-compiled methods for Hilbert space operations
- hilbert_jit_states        : State manipulation with JIT compilation
- hilbert_jit_states_jax    : JAX-based implementations for GPU acceleration
- matrix_builder            : Optimized matrix construction for symmetry-reduced spaces

---------------------------------------------------
File        : QES/Algebra/Hilbert/__init__.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Date        : 2025-10-29
---------------------------------------------------
"""

try:
    from .hilbert_jit_methods import *
    from .hilbert_jit_states import *
    from .matrix_builder import build_operator_matrix, get_symmetry_rotation_matrix

    __all__ = ["build_operator_matrix", "get_symmetry_rotation_matrix"]
except ImportError:
    __all__ = []

# --------------------------------------------------
#! EOF
# --------------------------------------------------
