"""Utilities for selecting spin operator families for interacting spin models."""

from __future__                         import annotations
from typing                             import Optional, Tuple
from QES.Algebra.Hilbert.hilbert_local  import LocalSpaceTypes

# -----------------------------------------------------------------------------
#! Spin operator family selection and utilities
# -----------------------------------------------------------------------------

def normalize_spin_operator_family(operator_family: str) -> str:
    """
    Normalize user/operator-family identifiers.
    """
    if operator_family is None:
        return "auto"
    key = str(operator_family).strip().lower().replace("_", "-")
    if key in {"auto"}:
        return "auto"
    if key in {"spin-1/2", "spin1/2", "spin-half", "spin-half-1/2", "spinhalf"}:
        return "spin-1/2"
    if key in {"spin-1", "spin1"}:
        return "spin-1"
    raise ValueError(
        f"Unknown operator_family={operator_family!r}. "
        "Expected one of: 'auto', 'spin-1/2', 'spin-1'."
    )

def infer_spin_operator_family_from_hilbert(hilbert_space) -> Optional[str]:
    """
    Infer spin operator family from HilbertSpace local-space metadata.
    """
    if hilbert_space is None:
        return None
    local = getattr(hilbert_space, "local_space", None)
    if local is None and hasattr(hilbert_space, "_local_space"):
        local = getattr(hilbert_space, "_local_space", None)
    if local is None or not hasattr(local, "typ"):
        return None
    if local.typ == LocalSpaceTypes.SPIN_1:
        return "spin-1"
    if local.typ == LocalSpaceTypes.SPIN_1_2:
        return "spin-1/2"
    return None

def select_spin_operator_module(hilbert_space=None, operator_family: str = "auto") -> Tuple[object, str]:
    """
    Select operator module and family string for a spin model.
    """
    requested = normalize_spin_operator_family(operator_family)
    if requested == "auto":
        inferred    = infer_spin_operator_family_from_hilbert(hilbert_space)
        requested   = inferred or "spin-1/2"

    if requested == "spin-1/2":
        import QES.Algebra.Operator.impl.operators_spin as spin_module
        return spin_module, requested

    import QES.Algebra.Operator.impl.operators_spin_1 as spin1_module
    return spin1_module, requested

def spin_axis_operator(spin_module, spin_family: str, axis: str, *, lattice=None, ns=None, type_act=None):
    """
    Build axis operator for selected spin family. Currently supports "spin-1/2" and "spin-1".
    """
    axis = axis.lower()
    if spin_family == "spin-1/2":
        fn = getattr(spin_module, f"sig_{axis}")
        return fn(lattice=lattice, ns=ns, type_act=type_act)
    fn = getattr(spin_module, f"s1_{axis}")
    return fn(lattice=lattice, ns=ns, type_act=type_act)

def spin_mixed_correlation_operator(
    spin_module,
    spin_family: str,
    pair: str,
    *,
    lattice=None,
    ns=None,
):
    """
    Build mixed-axis 2-site correlator S^a_i S^b_j for supported families.
    """
    pair = pair.lower()
    if spin_family != "spin-1/2":
        raise NotImplementedError("Mixed-axis correlators (xy, yz, zx, ...) are currently implemented only for spin-1/2.")
    
    fn = getattr(spin_module, f"sig_{pair}")
    return fn(lattice=lattice, ns=ns)

# ═══════════════════════════════════════════════════════════════════════════
#! Predefined regions for entanglement calculations
# ═══════════════════════════════════════════════════════════════════════════