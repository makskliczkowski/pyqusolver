"""
Base classes and registry for symmetry operations in Hilbert space.

File        : QES/Algebra/Symmetries/base.py
Description : Base classes and registry for symmetry operations in Hilbert space.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
"""

from typing import Tuple, Any, Dict, Type
####################################################################################################
# Abstract base class for symmetry operations
####################################################################################################

class SymmetryOperator:
    """
    Abstract base class for a symmetry operation.
    """
    def apply(self, state: int) -> Tuple[int, complex]:
        """
        Apply the symmetry to a basis state (integer representation).
        Returns (new_state, phase).
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

####################################################################################################
# Registry for symmetry operator classes
####################################################################################################

class SymmetryRegistry:
    """
    Registry for symmetry operator classes, for extensibility.
    """
    _registry: Dict[str, Type[SymmetryOperator]] = {}

    @classmethod
    def register(cls, name: str, sym_cls: Type[SymmetryOperator]):
        cls._registry[name] = sym_cls

    @classmethod
    def get(cls, name: str) -> Type[SymmetryOperator]:
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys())
