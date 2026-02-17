"""
Operator implementation package and state-encoding conventions.

This module documents the state conventions used by operator kernels so model
and Hilbert-space code can stay consistent across local spaces.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

# ---------------------------------------------------------------------------
# State conventions
# ---------------------------------------------------------------------------

STATE_CONVENTIONS: Dict[str, Dict[str, Any]] = {
    "spin-1/2": {
        "name"      : "Spin-1/2",
        "summary"   : "Binary computational basis.",
        "integer_encoding"  : (
            "Site i is stored in bit (Ns-1-i). "
            "bit=0 -> |down>, bit=1 -> |up>."
        ),
        "vector_encoding"   : (
            "Length-Ns array with entries in {0,1}. "
            "0 -> |down>, 1 -> |up>."
        ),
        "examples"          : [
            "Ns=4, state |up down up down> -> bits 1010 -> integer 10",
            "Ns=4, integer 12 -> bits 1100 -> |up up down down>",
        ],
    },
    "spin-1": {
        "name"      : "Spin-1",
        "summary"   : "Ternary computational basis with local values m in {+1,0,-1}.",
        "integer_encoding"  : (
            "State is interpreted in base-3 with digit at site i: "
            "0 -> m=+1, 1 -> m=0, 2 -> m=-1."
            "E.g., for Ns=3, state |+1,0,-1> -> digits [0,1,2] -> integer 0*9 + 1*3 + 2 = 5."
        ),
        "vector_encoding"   : (
            "Length-Ns array with entries in {0,1,2}. "
            "0 -> m=+1, 1 -> m=0, 2 -> m=-1."
        ),
        "examples"          : [
            "Ns=3, [0,1,2] means |+1,0,-1> and integer 0*9 + 1*3 + 2 = 5",
            "Ns=2, integer 7 -> ternary '21' -> |-1,0>",
        ],
    },
    "spinless-fermions"     : {
        "name"      : "Spinless Fermions",
        "summary"   : "Binary occupation-number (Fock) basis with Jordan-Wigner signs.",
        "integer_encoding"  : (
            "Site i occupation n_i is bit (Ns-1-i). n_i in {0,1}."
        ),
        "vector_encoding"   : (
            "Length-Ns array with entries in {0,1}; entry is local occupation."
        ),
        "examples"          : [
            "Ns=5, occupation [1,0,1,0,0] -> bits 10100 -> integer 20",
            "Creation/annihilation signs use parity of occupied sites left of i",
        ],
    },
    "abelian-anyons"        : {
        "name"      : "Hard-core Abelian Anyons",
        "summary"   : "Binary occupation basis with exchange phase from statistics angle.",
        "integer_encoding"  : (
            "Binary occupancy like spinless fermions; phase differs by theta."
        ),
        "vector_encoding"   : "Length-Ns array in {0,1} with anyonic exchange phase.",
        "examples"          : [
            "theta=0 -> hard-core boson signs",
            "theta=pi -> fermionic signs",
        ],
    },
    "bosons"                : {
        "name"      : "Bosons (truncated)",
        "summary"   : "Local occupation basis with configurable cutoff n_max.",
        "integer_encoding"  : (
            "Mixed-radix/base-(n_max+1) encoding per site for truncated local occupation."
        ),
        "vector_encoding"   : "Length-Ns array with local occupations n_i in [0, n_max].",
        "examples"          : [
            "cutoff=2, Ns=3, [2,0,1] encoded in base-3 representation",
        ],
    },
}

_ALIASES = {
    "spin_half"         : "spin-1/2",
    "spin-1/2"          : "spin-1/2",
    "spin1/2"           : "spin-1/2",
    "spin_1_2"          : "spin-1/2",
    "spin_1"            : "spin-1",
    "spin1"             : "spin-1",
    "spin-1"            : "spin-1",
    "fermion"           : "spinless-fermions",
    "spinless_fermions" : "spinless-fermions",
    "spinless-fermions" : "spinless-fermions",
    "anyons"            : "abelian-anyons",
    "anyon"             : "abelian-anyons",
    "abelian-anyons"    : "abelian-anyons",
    "boson"             : "bosons",
    "bosons"            : "bosons",
}

def _normalize_local_space_key(local_space: Any) -> str:
    """Normalize LocalSpace/enum/string into canonical convention key."""
    if local_space is None:
        return "spin-1/2"

    if hasattr(local_space, "value"):
        raw = str(local_space.value)
    elif hasattr(local_space, "name") and hasattr(local_space, "typ"):
        # LocalSpace instance: typ is LocalSpaceTypes enum
        raw = str(getattr(local_space.typ, "value", local_space.name))
    elif hasattr(local_space, "typ") and hasattr(local_space.typ, "value"):
        raw = str(local_space.typ.value)
    else:
        raw = str(local_space)

    key = raw.strip().lower().replace("_", "-")
    return _ALIASES.get(key, key)

def get_state_convention(local_space: Any = "spin-1/2") -> Dict[str, Any]:
    """Return state convention metadata for the requested local-space type."""
    key = _normalize_local_space_key(local_space)
    if key not in STATE_CONVENTIONS:
        raise KeyError(f"Unknown local-space convention key: {local_space!r}")
    return deepcopy(STATE_CONVENTIONS[key])

def format_state_convention(local_space: Any = "spin-1/2", include_examples: bool = True) -> str:
    """Return human-readable state convention text."""
    data = get_state_convention(local_space)
    lines = [
        f"{data['name']}: {data['summary']}",
        f"  Integer encoding    : {data['integer_encoding']}",
        f"  Vector encoding     : {data['vector_encoding']}",
    ]
    if include_examples:
        lines.append("  Examples:")
        for ex in data.get("examples", []):
            lines.append(f"    - {ex}")
    return "\n".join(lines)


def all_state_conventions() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy of all state-convention entries."""
    return deepcopy(STATE_CONVENTIONS)

__all__ = [
    "STATE_CONVENTIONS",
    "all_state_conventions",
    "format_state_convention",
    "get_state_convention",
]

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------