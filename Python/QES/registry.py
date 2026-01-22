"""
Module discovery and descriptions for QES.

This module provides functions to introspect available QES subpackages and
modules, returning a concise description for each. Descriptions are sourced from
`MODULE_DESCRIPTION` variables when present, falling back to the first
non-empty line of a module's docstring.

Typical usage
-------------
    import QES
    for m in QES.list_modules():
        print(m['name'], '-', m['description'])

    print(QES.describe_module('QES.Algebra'))

Implementation notes
--------------------
- We keep a curated list of top-level packages for stability and predictable
  ordering. We also optionally enumerate a few commonly used submodules.
- Import errors are handled gracefully; missing optional dependencies won't
  break discovery.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class ModuleInfo:
    name: str  # short name, e.g. "Algebra.Hilbert"
    path: str  # dotted path, e.g. "QES.Algebra.hilbert"
    description: str  # one-liner

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# Curated index of key modules: name -> dotted path
_TOP_LEVEL: Dict[str, str] = {
    "Algebra": "QES.Algebra",
    "NQS": "QES.NQS",
    "Solver": "QES.Solver",
    "general_python": "QES.general_python",
}

_COMMON_SUBMODULES: Dict[str, str] = {
    # Algebra submodules
    "Algebra.Hilbert": "QES.Algebra.hilbert",
    "Algebra.Hamil": "QES.Algebra.hamil",
    "Algebra.Operator": "QES.Algebra.Operator",
    "Algebra.Symmetries": "QES.Algebra.symmetries",
    # NQS submodules
    "NQS.core": "QES.NQS.nqs",
    "NQS.train": "QES.NQS.nqs_train",
    "NQS.tdvp": "QES.NQS.tdvp",
    # Solver submodules
    "Solver.core": "QES.Solver.solver",
}

_DEF_FALLBACK = "No description available."

# ----------------------------------------------------------------------------
# Helper functions


def _first_line(s: Optional[str]) -> str:
    if not s:
        return _DEF_FALLBACK
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return _DEF_FALLBACK


# ----------------------------------------------------------------------------


def _desc_from_module(mod) -> str:
    # Prefer explicit MODULE_DESCRIPTION if present
    desc = getattr(mod, "MODULE_DESCRIPTION", None)
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    # Fallback to first non-empty docstring line
    return _first_line(getattr(mod, "__doc__", None))


# ---------------------------------------------------------------------------


def _safe_import(path: str):
    try:
        return importlib.import_module(path)
    except Exception:
        return None


# ----------------------------------------------------------------------------


def _collect(entries: Dict[str, str]) -> List[ModuleInfo]:
    out: List[ModuleInfo] = []
    for name, path in entries.items():
        mod = _safe_import(path)
        desc = _desc_from_module(mod) if mod is not None else _DEF_FALLBACK
        out.append(ModuleInfo(name=name, path=path, description=desc))
    return out


# ---------------------------------------------------------------------------


def list_modules(include_submodules: bool = True) -> List[Dict[str, str]]:
    """Return a list of dicts with name, path, and description of QES modules.

    Parameters
    ----------
    include_submodules : bool
        If True, include commonly used submodules; otherwise only top-level
        QES packages are listed.
    """
    items: List[ModuleInfo] = _collect(_TOP_LEVEL)
    if include_submodules:
        items += _collect(_COMMON_SUBMODULES)
    # stable ordering by name
    items.sort(key=lambda x: x.name)
    return [mi.to_dict() for mi in items]


# ----------------------------------------------------------------------------


def describe_module(name_or_path: str) -> str:
    """Return a short description for a given module name or dotted path.

    Accepts either the curated short name (e.g., "Algebra.Hilbert") or a
    fully qualified path (e.g., "QES.Algebra.hilbert").
    """
    # Try direct path import first
    mod = _safe_import(name_or_path)
    if mod is not None:
        return _desc_from_module(mod)

    # Try to resolve via curated tables
    if name_or_path in _TOP_LEVEL:
        mod = _safe_import(_TOP_LEVEL[name_or_path])
        return _desc_from_module(mod) if mod else _DEF_FALLBACK
    if name_or_path in _COMMON_SUBMODULES:
        mod = _safe_import(_COMMON_SUBMODULES[name_or_path])
        return _desc_from_module(mod) if mod else _DEF_FALLBACK

    # Try importing as QES.<name>
    if not name_or_path.startswith("QES."):
        mod = _safe_import("QES." + name_or_path)
        if mod is not None:
            return _desc_from_module(mod)

    return _DEF_FALLBACK


# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
