"""Non-interacting quantum model definitions.

Modules:
--------
- conserving: Particle-conserving models (Free Fermions, Aubry-Andre)
- syk: Sachdev-Ye-Kitaev model
- plrb: Power-Law Random Banded model
- rpm: Rosenzweig-Porter model

----------------
File            : Algebra/Model/Noninteracting/__init__.py
Author          : Maksymilian Kliczkowski
----------------
"""

from    __future__ import annotations

import  importlib
from    typing import TYPE_CHECKING

try:
    from .._registry import create_model as _create_model, get_model_export_names as _get_model_export_names, resolve_model_export as _resolve_model_export
except ImportError:
    raise ImportError("Failed to import model registry utilities. Ensure the general_python package is correctly installed.")

__all__: list[str] = [
    'conserving',
    'aubry_andre',
    'free_fermions',
    'syk', 'plrb', 'rpm',
    'PLRB_SP', 'PLRB_MB', 'RPM_SP', 'RPM_MB',
    'AubryAndre', 'FreeFermions', 'SYK2', 'PowerLawRandomBanded', 'RosenzweigPorter',
    'choose_model'
]

if TYPE_CHECKING:
    from .conserving import free_fermions, aubry_andre
    import plrb
    import syk
    import rpm
    # aliases
    PLRB_SP                 = plrb.PLRB_SP
    PLRB_MB                 = plrb.PLRB_MB
    PLRB                    = plrb.PowerLawRandomBanded
    PowerLawRandomBanded    = plrb.PowerLawRandomBanded
    RPM_SP                  = rpm.RPM_SP
    RPM_MB                  = rpm.RPM_MB
    RPM                     = rpm.RosenzweigPorter
    RosenzweigPorter        = rpm.RosenzweigPorter
    SYK2                    = syk.SYK2
    FreeFermions            = free_fermions.FreeFermions
    AubryAndre              = aubry_andre.AubryAndre

_LAZY_MODULES: dict[str, str] = {
    'nonconserving'         : '.nonconserving',
    'conserving'            : '.conserving',
    'aubry_andre'           : '.conserving.aubry_andre',
    'free_fermions'         : '.conserving.free_fermions',
    'syk'                   : '.syk',
    'plrb'                  : '.plrb',
    'rpm'                   : '.rpm',   
}

_CLASS_MAP: dict[str, tuple[str, str]] = {
    'AubryAndre'            : ('.conserving.aubry_andre', 'AubryAndre'),
    'FreeFermions'          : ('.conserving.free_fermions', 'FreeFermions'),
    'SYK2'                  : ('.syk', 'SYK2'),
    'PLRB_SP'               : ('.plrb', 'PLRB_SP'),
    'PLRB_MB'               : ('.plrb', 'PLRB_MB'),
    'PowerLawRandomBanded'  : ('.plrb', 'PowerLawRandomBanded'),
    'PLRB'                  : ('.plrb', 'PowerLawRandomBanded'),
    'RPM_SP'                : ('.rpm', 'RPM_SP'),
    'RPM_MB'                : ('.rpm', 'RPM_MB'),
    'RosenzweigPorter'      : ('.rpm', 'RosenzweigPorter'),
    'RPM'                   : ('.rpm', 'RosenzweigPorter'),
}

def __getattr__(name: str):
    if name in _CLASS_MAP:
        module_name, class_name = _CLASS_MAP[name]
        module                  = importlib.import_module(module_name, __name__)
        attr                    = getattr(module, class_name)
        globals()[name]         = attr
        return attr
    if name in _LAZY_MODULES:
        module                  = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name]         = module
        return module
    try:
        return _resolve_model_export(name, family="noninteracting")
    except ValueError:
        pass
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODULES.keys() | _CLASS_MAP.keys() | set(_get_model_export_names(family="noninteracting")))

def choose_model(model_name: str, **kwargs):
    """
    Returns an instance of a non-interacting model of the desired type.

    Args:
        model_name (str):
            Type of model (e.g. "aubry_andre", "syk2", "rpm")
        **kwargs:
            Parameters for the model constructor.

    Returns:
        Hamiltonian: An instance of the desired quantum non-interacting model.
    """
    return _create_model(model_name, family="noninteracting", **kwargs)

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
