"""
Hilbert Space Utilities Module
==============================

This module contains utilities for Hilbert space operations and optimizations.

Modules:
--------
- hilbert_jit_states        : State manipulation with JIT compilation
- hilbert_jit_states_jax    : JAX-based implementations for GPU acceleration
- matrix_builder            : Optimized matrix construction for symmetry-reduced spaces

---------------------------------------------------
File        : QES/Algebra/Hilbert/__init__.py
Author      : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date        : 2025-10-29
---------------------------------------------------
"""

import importlib
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Lazy Import Configuration
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # matrix_builder
    "build_operator_matrix": (".matrix_builder", "build_operator_matrix"),
    "get_symmetry_rotation_matrix": (".matrix_builder", "get_symmetry_rotation_matrix"),
    # hilbert_jit_states
    "calculate_slater_det": (".hilbert_jit_states", "calculate_slater_det"),
    "calculate_permanent": (".hilbert_jit_states", "calculate_permanent"),
    "bogolubov_decompose": (".hilbert_jit_states", "bogolubov_decompose"),
    "pairing_matrix": (".hilbert_jit_states", "pairing_matrix"),
    "calculate_bogoliubov_amp": (".hilbert_jit_states", "calculate_bogoliubov_amp"),
    "calculate_bosonic_gaussian_amp": (".hilbert_jit_states", "calculate_bosonic_gaussian_amp"),
    "many_body_state_full": (".hilbert_jit_states", "many_body_state_full"),
    "nrg_particle_conserving": (".hilbert_jit_states", "nrg_particle_conserving"),
    "nrg_bdg": (".hilbert_jit_states", "nrg_bdg"),
}

_SUBMODULES = {
    "hilbert_jit_states": "State manipulation with JIT compilation",
    "hilbert_jit_states_jax": "JAX-based implementations",
    "matrix_builder": "Optimized matrix construction",
}

if TYPE_CHECKING:
    from .hilbert_jit_states import (
        bogolubov_decompose,
        calculate_bogoliubov_amp,
        calculate_bosonic_gaussian_amp,
        calculate_permanent,
        calculate_slater_det,
        many_body_state_full,
        nrg_bdg,
        nrg_particle_conserving,
        pairing_matrix,
    )
    from .matrix_builder import build_operator_matrix, get_symmetry_rotation_matrix


def __getattr__(name: str) -> Any:
    """Lazily import submodules and classes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", package=__name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()) + list(_SUBMODULES.keys()))


__all__ = [
    # matrix_builder
    "build_operator_matrix",
    "get_symmetry_rotation_matrix",
    # hilbert_jit_states
    "calculate_slater_det",
    "calculate_permanent",
    "bogolubov_decompose",
    "pairing_matrix",
    "calculate_bogoliubov_amp",
    "calculate_bosonic_gaussian_amp",
    "many_body_state_full",
    "nrg_particle_conserving",
    "nrg_bdg",
]
