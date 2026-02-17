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

__all__: list[str] = [
    'conserving',
    'aubry_andre',
    'free_fermions',
    'syk', 'plrb', 'rpm',
    'AubryAndre', 'FreeFermions', 'SYK2', 'PowerLawRandomBanded', 'RosenzweigPorter',
    'choose_model'
]

_LAZY_MODULES: dict[str, str] = {
    'nonconserving'         : '.nonconserving',
    'conserving'            : '.conserving',
    'aubry_andre'           : '.conserving.aubry_andre',
    'free_fermions'         : '.conserving.free_fermions',
    'syk'                   : '.syk',
    'plrb'                  : '.plrb',
    'rpm'                   : '.rpm',
}

_CLASS_MAP: dict[str, str] = {
    'AubryAndre'            : '.conserving.aubry_andre',
    'FreeFermions'          : '.conserving.free_fermions',
    'SYK2'                  : '.syk',
    'PowerLawRandomBanded'  : '.plrb',
    'PLRB'                  : '.plrb',
    'RosenzweigPorter'      : '.rpm',
    'RPM'                   : '.rpm',
}

def __getattr__(name: str):
    if name in _CLASS_MAP:
        module = importlib.import_module(_CLASS_MAP[name], __name__)
        return getattr(module, name)
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()) + list(_CLASS_MAP.keys()))

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
    model_name_map = {
        "aubry_andre"           : "AubryAndre",
        "free_fermions"         : "FreeFermions",
        "syk2"                  : "SYK2",
        "syk"                   : "SYK2",
        "plrb"                  : "PowerLawRandomBanded",
        "power_law_random_banded" : "PowerLawRandomBanded",
        "rpm"                   : "RosenzweigPorter",
        "rosenzweig_porter"     : "RosenzweigPorter",
    }

    # Normalize model name
    lookup = model_name.lower().replace(" ", "_").replace("-", "_")

    cls_name = None
    if lookup in model_name_map:
        cls_name = model_name_map[lookup]
    elif model_name in _CLASS_MAP:
        cls_name = model_name

    if cls_name is None:
        raise ValueError(f"Unknown non-interacting model '{model_name}'. Available: {list(model_name_map.keys())}")

    cls = __getattr__(cls_name)
    return cls(**kwargs)

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
