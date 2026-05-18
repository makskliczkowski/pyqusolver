"""
Compatibility package for source-tree imports.

The installable Python package is ``QES``. This wrapper exists only so tools
that put the repository root on ``PYTHONPATH`` can still resolve
``Python.QES`` without importing the full solver stack eagerly.
----------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
License         : MIT
Copyright       : (c) 2023-2026 Maksymilian Kliczkowski
----------------------------------------------------------------------------
"""

from    __future__ import annotations

import  importlib as _importlib
from    typing import TYPE_CHECKING, Any

_LAZY_IMPORTS = {
    "QES": (".QES", None),
}

_LAZY_CACHE: dict[str, Any] = {}

if TYPE_CHECKING:
    from . import QES as QES


def __getattr__(name: str) -> Any:
    """Resolve compatibility exports lazily."""
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name]
    module = _importlib.import_module(module_path, package=__name__)
    result = module if attr_name is None else getattr(module, attr_name)
    _LAZY_CACHE[name] = result
    return result


def __dir__() -> list[str]:
    """Return public compatibility exports."""
    return sorted(set(globals()) | set(__all__))


__all__ = ["QES"]
