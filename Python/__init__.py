"""
This module initializes the Quantum EigenSolver package.

Imports:
    test as Test:
        A module for testing functionalities.
    QES:
        The main module containing the Quantum EigenSolver implementation.
        
----------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
License         : MIT
Copyright       : (c) 2021-2026 Maksymilian Kliczkowski
----------------------------------------------------------------------------
"""

import  importlib
from    typing import Any, TYPE_CHECKING

# ----------------------------------------------------------------------------
# Lazy Import Configuration
# ----------------------------------------------------------------------------

# Mapping of attribute names to (module_relative_path, attribute_name_in_module)
_LAZY_IMPORTS = {
    'Test'  :   ('.test', None),
    'QES'   :   ('.QES',  None),
}

# Cache for lazily loaded modules/attributes
_LAZY_CACHE = {}

if TYPE_CHECKING:
    import test as Test
    import QES  as QES


def __getattr__(name: str) -> Any:
    """
    Module-level __getattr__ for lazy imports (PEP 562).
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            # Import the module relative to this package
            module      = importlib.import_module(module_path, package=__name__)
            
            # If attr_name is None, we want the module itself
            if attr_name is None:
                result  = module
            else:
                result  = getattr(module, attr_name)
            
            _LAZY_CACHE[name] = result
            return result
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import lazy module '{name}' from '{module_path}': {e}") from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Support for dir() and tab completion.
    """
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

__all__ = ["Test", "QES"]

# ----------------------------------------------------------------------------
#! END OF FILE
# ----------------------------------------------------------------------------
