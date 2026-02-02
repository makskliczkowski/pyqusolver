"""Non-interacting quantum model definitions (placeholder).

See higher level factory interfaces in ``QES.Algebra.Model``.

----------------
File            : Algebra/Model/Noninteracting/__init__.py
Author          : Maksymilian Kliczkowski
----------------
"""

from    __future__ import annotations

import  importlib
from    typing import TYPE_CHECKING

from QES.Algebra.Model.Noninteracting import plrb

__all__: list[str] = [
    'conserving',
    'aubry_andre',
    'free_fermions',
    # general
    'syk', 'plrb', 'rpm',
]

_LAZY_MODULES: dict[str, str] = {
    'nonconserving'     : '.nonconserving',
    'conserving'        : '.conserving',
    'aubry_andre'       : '.conserving.aubry_andre',
    'free_fermions'     : '.conserving.free_fermions',
    'syk'               : '.syk',
    'plrb'              : '.plrb',
    'rpm'               : '.rpm',
}

def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module          = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))

if TYPE_CHECKING:
    from .              import conserving, nonconserving
    from .              import plrb, rpm, syk
    from .conserving    import aubry_andre, free_fermions
    
# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
