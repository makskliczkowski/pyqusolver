"""
QES model registry and factory helpers.

This module is the single source of truth for model discovery, aliases,
constructor kwargs introspection, and shared impurity utilities.Submodules:
-----------
- Interacting: 
    Models with particle interactions
- Noninteracting: 
    Free particle and non-interacting models

Classes:
--------
Various quantum model implementations including:
- Spin models (Heisenberg, Ising, XY, etc.)
- Fermionic models (Hubbard, t-J, etc.)
- Bosonic models

-------------------------------
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
-------------------------------

"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ._registry import (
    create_model            as _create_model,
    get_model_aliases       as _get_model_aliases,
    get_model_export_names  as _get_model_export_names,
    model_kwargs            as _model_kwargs,
    resolve_model_export    as _resolve_model_export,
)


__all__ = [
    "intr",
    "nintr",
    "choose_model",
    "qes_model_aliases",
    "qes_models_kwargs",
    # Impurity utilities
    "SiteGeometry",
    "make_impurity",
    "impurity_site_amplitude",
    "compute_impurity_distances",
    "compute_site_geometries",
    "group_site_geometries",
    "pick_site_geometries_by_angle",
    "impurity"
] + list(_get_model_export_names())

_MODULE_ALIASES = {
    "intr"      : ".Interacting",
    "nintr"     : ".Noninteracting",
    "impurity"  : "._impurity",
    "dummy"     : ".dummy",
}

_IMPURITY_EXPORTS = {
    "SiteGeometry",
    "make_impurity",
    "impurity_site_amplitude",
    "compute_impurity_distances",
    "compute_site_geometries",
    "group_site_geometries",
    "pick_site_geometries_by_angle",
}

if TYPE_CHECKING:
    # Impurity utilities
    import _impurity                                                as impurity
    # Fermionic
    from .Interacting.Fermionic.free_fermion_manybody               import ManyBodyFreeFermions
    from .Interacting.Fermionic.hubbard                             import HubbardModel
    from .Interacting.Fermionic.spinful_hubbard                     import SpinfulHubbardModel
    # Spin-1/2
    from .Interacting.Spin.heisenberg_kitaev                        import HeisenbergKitaev
    from .Interacting.Spin.j1j2                                     import J1J2Model
    from .Interacting.Spin.qsm                                      import QSM
    from .Interacting.Spin.transverse_ising                         import TransverseFieldIsing
    from .Interacting.Spin.ultrametric                              import UltrametricModel
    from .Interacting.Spin.xxz                                      import XXZ
    # Non-interacting
    from .Noninteracting.conserving.Majorana.kitaev_gamma_majorana  import KitaevGammaMajorana
    from .Noninteracting.conserving.aubry_andre                     import AubryAndre
    from .Noninteracting.conserving.free_fermions                   import FreeFermions
    from .Noninteracting.plrb                                       import PLRB, PLRB_MB, PLRB_SP, PowerLawRandomBanded
    from .Noninteracting.rpm                                        import RPM, RPM_MB, RPM_SP, RosenzweigPorter
    from .Noninteracting.syk                                        import SYK2
    # Dummy
    from .dummy                                                     import DummyHamiltonian


def qes_model_aliases() -> dict[str, str]:
    """
    Return ``alias -> class_name`` model mappings for the public model registry.
    """
    return _get_model_aliases()


def qes_models_kwargs(model_name: str | None = None) -> dict[str, Any]:
    """
    Return constructor kwargs metadata for one model or for the whole registry.
    """
    return _model_kwargs(model_name)


def choose_model(model_name: str, **kwargs):
    """
    Construct a model instance from its canonical name, class name, or alias.
    """
    return _create_model(model_name, **kwargs)


def __getattr__(name: str):
    
    if name in _MODULE_ALIASES:
        module = importlib.import_module(_MODULE_ALIASES[name], package=__name__)
        globals()[name] = module
        return module
    
    if name in _IMPURITY_EXPORTS:
        module  = importlib.import_module("._impurity", package=__name__)
        value   = getattr(module, name)
        globals()[name] = value
        return value
    try:
        return _resolve_model_export(name)
    except ValueError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_get_model_export_names()))

# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------
