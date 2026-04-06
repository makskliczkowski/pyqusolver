''' 
Model registry and factory utilities for QES models. It provides a centralized mechanism for discovering available models, their aliases, and constructor argument specifications. 
This module is used by the main model submodules (Interacting, Noninteracting) to implement lazy loading and the `choose_model` factory function.

-------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
-------------------------------
'''

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass(frozen=True)
class _ModelSpec:
    canonical_name   : str
    class_name       : str
    module_path      : str
    family           : str
    aliases          : tuple[str, ...] = ()
    class_aliases    : tuple[str, ...] = ()
    kwarg_normalizer : Optional[str] = None


_MODEL_SPECS: tuple[_ModelSpec, ...] = (
    _ModelSpec(
        canonical_name   = "heisenberg_kitaev",
        class_name       = "HeisenbergKitaev",
        module_path      = ".Interacting.Spin.heisenberg_kitaev",
        family           = "interacting_spin",
        aliases          = ("kitaev", "heisenberg", "kh", "kitaev_heisenberg", "gamma_kitaev", "kitaev_gamma"),
        kwarg_normalizer = "normalize_kitaev_model_kwargs",
    ),
    _ModelSpec(
        canonical_name   = "transverse_ising",
        class_name       = "TransverseFieldIsing",
        module_path      = ".Interacting.Spin.transverse_ising",
        family           = "interacting_spin",
        aliases          = ("transverse_field_ising", "tfim", "tfi"),
    ),
    _ModelSpec(
        canonical_name   = "j1j2",
        class_name       = "J1J2Model",
        module_path      = ".Interacting.Spin.j1j2",
        family           = "interacting_spin",
        aliases          = ("j1_j2", "j1-j2"),
    ),
    _ModelSpec(
        canonical_name   = "xxz",
        class_name       = "XXZ",
        module_path      = ".Interacting.Spin.xxz",
        family           = "interacting_spin",
    ),
    _ModelSpec(
        canonical_name   = "qsm",
        class_name       = "QSM",
        module_path      = ".Interacting.Spin.qsm",
        family           = "interacting_spin",
    ),
    _ModelSpec(
        canonical_name   = "ultrametric",
        class_name       = "UltrametricModel",
        module_path      = ".Interacting.Spin.ultrametric",
        family           = "interacting_spin",
    ),
    _ModelSpec(
        canonical_name   = "manybody_free_fermions",
        class_name       = "ManyBodyFreeFermions",
        module_path      = ".Interacting.Fermionic.free_fermion_manybody",
        family           = "interacting_fermionic",
        aliases          = ("free_fermions_manybody", "free_fermion_manybody"),
    ),
    _ModelSpec(
        canonical_name   = "hubbard",
        class_name       = "HubbardModel",
        module_path      = ".Interacting.Fermionic.hubbard",
        family           = "interacting_fermionic",
        aliases          = ("spinless_hubbard", "hubbard_spinless"),
    ),
    _ModelSpec(
        canonical_name   = "spinful_hubbard",
        class_name       = "SpinfulHubbardModel",
        module_path      = ".Interacting.Fermionic.spinful_hubbard",
        family           = "interacting_fermionic",
        aliases          = ("hubbard_spinful", "onsite_hubbard", "dqmc_hubbard"),
    ),
    _ModelSpec(
        canonical_name   = "aubry_andre",
        class_name       = "AubryAndre",
        module_path      = ".Noninteracting.conserving.aubry_andre",
        family           = "noninteracting",
    ),
    _ModelSpec(
        canonical_name   = "free_fermions",
        class_name       = "FreeFermions",
        module_path      = ".Noninteracting.conserving.free_fermions",
        family           = "noninteracting",
    ),
    _ModelSpec(
        canonical_name   = "syk2",
        class_name       = "SYK2",
        module_path      = ".Noninteracting.syk",
        family           = "noninteracting",
        aliases          = ("syk",),
    ),
    _ModelSpec(
        canonical_name   = "power_law_random_banded",
        class_name       = "PowerLawRandomBanded",
        module_path      = ".Noninteracting.plrb",
        family           = "noninteracting",
        aliases          = ("plrb",),
        class_aliases    = ("PLRB",),
    ),
    _ModelSpec(
        canonical_name   = "plrb_sp",
        class_name       = "PLRB_SP",
        module_path      = ".Noninteracting.plrb",
        family           = "noninteracting",
    ),
    _ModelSpec(
        canonical_name   = "plrb_mb",
        class_name       = "PLRB_MB",
        module_path      = ".Noninteracting.plrb",
        family           = "noninteracting",
    ),
    _ModelSpec(
        canonical_name   = "rosenzweig_porter",
        class_name       = "RosenzweigPorter",
        module_path      = ".Noninteracting.rpm",
        family           = "noninteracting",
        aliases          = ("rpm",),
        class_aliases    = ("RPM",),
    ),
    _ModelSpec(
        canonical_name   = "rpm_sp",
        class_name       = "RPM_SP",
        module_path      = ".Noninteracting.rpm",
        family           = "noninteracting",
    ),
    _ModelSpec(
        canonical_name   = "rpm_mb",
        class_name       = "RPM_MB",
        module_path      = ".Noninteracting.rpm",
        family           = "noninteracting",
    ),
    _ModelSpec(
        canonical_name   = "kitaev_gamma_majorana",
        class_name       = "KitaevGammaMajorana",
        module_path      = ".Noninteracting.conserving.Majorana.kitaev_gamma_majorana",
        family           = "noninteracting",
        aliases          = ("majorana_kitaev_gamma", "majorana_kitaev"),
    ),
    _ModelSpec(
        canonical_name   = "dummy",
        class_name       = "DummyHamiltonian",
        module_path      = ".dummy",
        family           = "misc",
        aliases          = ("dummy_hamiltonian",),
    ),
)

_MODEL_CLASS_CACHE          : dict[str, type] = {}
_MODEL_NORMALIZER_CACHE     : dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

_CLASS_NAME_TO_SPEC         : dict[str, _ModelSpec] = {}
_CLASS_ALIAS_TO_SPEC        : dict[str, _ModelSpec] = {}
_LOOKUP_TO_SPEC             : dict[str, _ModelSpec] = {}

for _spec in _MODEL_SPECS:
    _CLASS_NAME_TO_SPEC[_spec.class_name] = _spec
    for _class_alias in _spec.class_aliases:
        _CLASS_ALIAS_TO_SPEC[_class_alias] = _spec
    _LOOKUP_TO_SPEC[_spec.canonical_name] = _spec
    for _alias in _spec.aliases:
        _LOOKUP_TO_SPEC[_alias] = _spec

# ----------------------------------------------------------------------

def _normalize_lookup(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def _load_model_class(spec: _ModelSpec) -> type:
    if spec.class_name not in _MODEL_CLASS_CACHE:
        module = importlib.import_module(spec.module_path, package=__package__)
        _MODEL_CLASS_CACHE[spec.class_name] = getattr(module, spec.class_name)
    return _MODEL_CLASS_CACHE[spec.class_name]

def _load_kwarg_normalizer(spec: _ModelSpec) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if spec.kwarg_normalizer is None:
        return None
    if spec.class_name not in _MODEL_NORMALIZER_CACHE:
        module = importlib.import_module(spec.module_path, package=__package__)
        _MODEL_NORMALIZER_CACHE[spec.class_name] = getattr(module, spec.kwarg_normalizer)
    return _MODEL_NORMALIZER_CACHE[spec.class_name]

# ----------------------------------------------------------------------

def get_model_spec(model_name: str, *, family: Optional[str] = None) -> _ModelSpec:
    spec = _CLASS_NAME_TO_SPEC.get(model_name)
    if spec is None:
        spec = _CLASS_ALIAS_TO_SPEC.get(model_name)
    if spec is None:
        spec = _LOOKUP_TO_SPEC.get(_normalize_lookup(model_name))
    if spec is None or (family is not None and spec.family != family):
        families = sorted({entry.canonical_name for entry in _MODEL_SPECS if family is None or entry.family == family})
        if family == "interacting_spin":
            raise ValueError(f"Unknown spin model '{model_name}'. Available: {families}")
        if family == "interacting_fermionic":
            raise ValueError(f"Unknown interacting fermionic model '{model_name}'. Available: {families}")
        if family == "noninteracting":
            raise ValueError(f"Unknown non-interacting model '{model_name}'. Available: {families}")
        raise ValueError(f"Unknown model '{model_name}'. Available: {families}")
    return spec

# ----------------------------------------------------------------------

def normalize_model_kwargs(model_name: str, kwargs: Dict[str, Any], *, family: Optional[str] = None) -> Dict[str, Any]:
    spec        = get_model_spec(model_name, family=family)
    out         = dict(kwargs)
    normalizer  = _load_kwarg_normalizer(spec)
    if normalizer is not None:
        out = normalizer(out)
    return out

def get_model_class(model_name: str, *, family: Optional[str] = None) -> type:
    return _load_model_class(get_model_spec(model_name, family=family))

def create_model(model_name: str, *, family: Optional[str] = None, **kwargs):
    spec = get_model_spec(model_name, family=family)
    cls  = _load_model_class(spec)
    return cls(**normalize_model_kwargs(spec.canonical_name, kwargs, family=family))

# ----------------------------------------------------------------------

def get_model_aliases(*, family: Optional[str] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for spec in _MODEL_SPECS:
        if family is not None and spec.family != family:
            continue
        out[spec.canonical_name] = spec.class_name
        for alias in spec.aliases:
            out[alias] = spec.class_name
    return dict(sorted(out.items()))


def get_model_export_names(*, family: Optional[str] = None) -> tuple[str, ...]:
    names = set()
    for spec in _MODEL_SPECS:
        if family is not None and spec.family != family:
            continue
        names.add(spec.class_name)
        names.add(spec.canonical_name)
        names.update(spec.aliases)
        names.update(spec.class_aliases)
    return tuple(sorted(names))


def resolve_model_export(name: str, *, family: Optional[str] = None):
    spec = get_model_spec(name, family=family)
    if name == spec.class_name or name in spec.class_aliases:
        return _load_model_class(spec)
    return partial(create_model, spec.canonical_name, family=family)


def model_kwargs(model_name: Optional[str] = None, *, family: Optional[str] = None) -> Dict[str, Any]:
    def _describe(spec: _ModelSpec) -> Dict[str, Any]:
        signature = inspect.signature(_load_model_class(spec).__init__)
        kwargs_info = {}
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            default = "<required>" if param.default is inspect._empty else param.default
            annotation = None if param.annotation is inspect._empty else str(param.annotation).replace("typing.", "")
            kwargs_info[name] = {
                "default": default,
                "annotation": annotation,
            }

        return {
            "class_name": spec.class_name,
            "family": spec.family,
            "aliases": [spec.canonical_name, *spec.aliases],
            "kwargs": kwargs_info,
        }

    if model_name is not None:
        spec = get_model_spec(model_name, family=family)
        return _describe(spec)

    out = {}
    for spec in _MODEL_SPECS:
        if family is not None and spec.family != family:
            continue
        out[spec.canonical_name] = _describe(spec)
    return out


def iter_model_specs(*, family: Optional[str] = None) -> Iterable[_ModelSpec]:
    for spec in _MODEL_SPECS:
        if family is None or spec.family == family:
            yield spec

# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------