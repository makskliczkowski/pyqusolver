"""
Symmetry compatibility checking and group construction utilities.

-----------------------------------------------------
File        : QES/Algebra/Symmetries/compatibility.py
Description : Utilities for checking symmetry commutation and building compatible groups.
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
-----------------------------------------------------
"""

from typing import List, Optional, Set, Tuple

from QES.Algebra.Symmetries.base import (
    MomentumSector,
    SymmetryOperator,
)
from QES.Algebra.Symmetries.translation import TranslationSymmetry

# -----------------------------------------------------
#! Checker
# -----------------------------------------------------


def check_compatibility(
    symmetries: List[SymmetryOperator], warn_callback: Optional[callable] = None
) -> Tuple[List[SymmetryOperator], Set[str]]:
    """
    Check mutual compatibility of a list of symmetries and filter incompatible ones.

    Parameters
    ----------
    symmetries : List[SymmetryOperator]
        List of symmetry operators to check.
    warn_callback : Optional[callable]
        Callback function to log warnings, signature (message: str) -> None.

    Returns
    -------
    compatible : List[SymmetryOperator]
        Filtered list containing only mutually compatible symmetries.
    removed : Set[str]
        Names of symmetries that were removed due to incompatibility.
    """
    if not symmetries:
        return [], set()

    # Determine momentum sector from translations if present
    momentum_sector = None
    for sym in symmetries:
        if isinstance(sym, TranslationSymmetry):
            momentum_sector = sym.get_momentum_sector()
            if momentum_sector is not None:
                break

    compatible = []
    removed = set()

    # go through symmetries
    for i, sym_i in enumerate(symmetries):

        is_compatible = True

        # Check compatibility with all previously accepted symmetries
        for sym_j in compatible:
            if not sym_i.commutes_with(sym_j, momentum_sector):
                is_compatible = False
                msg = (
                    f"Symmetry {sym_i.name} ({sym_i.symmetry_class.name}) "
                    f"does not commute with {sym_j.name} ({sym_j.symmetry_class.name})"
                )
                if momentum_sector is not None:
                    msg += f" at momentum sector {momentum_sector.name}"
                msg += "; removing."

                if warn_callback:
                    warn_callback(msg)

                removed.add(sym_i.name)
                break

        if is_compatible:
            compatible.append(sym_i)

    return compatible, removed


# -----------------------------------------------------
#! Builder
# -----------------------------------------------------


def build_symmetry_group(
    generators: List[SymmetryOperator], max_group_size: int = 10000
) -> List[Tuple[SymmetryOperator, ...]]:
    """
    Build the full symmetry group from a list of compatible generators.

    For Abelian symmetries, this returns all distinct combinations (including
    the identity as an empty tuple). For cyclic generators like translation,
    we also include powers T, T^2, ..., T^(L-1).

    Parameters
    ----------
    generators : List[SymmetryOperator]
        List of mutually commuting symmetry generators.
    max_group_size : int
        Maximum allowed group size to prevent combinatorial explosion.

    Returns
    -------
    group : List[Tuple[SymmetryOperator, ...]]
        List of operator tuples representing the full symmetry group.
        Each tuple represents a composition: (op1, op2, ...) means apply op1, then op2, etc.
    """
    if not generators:
        return [()]

    # Start with the identity
    group = [()]

    # For each generator, add it and its combinations with existing elements
    for gen in generators:
        new_elements = []

        # Add the generator itself
        if (gen,) not in group:
            new_elements.append((gen,))

        # Add combinations with existing group elements
        for elem in list(group):
            if elem == ():
                # Already added above
                continue
            # Composition: elem followed by gen
            combined = elem + (gen,)
            if combined not in group and combined not in new_elements:
                new_elements.append(combined)

        group.extend(new_elements)

        # Safety check
        if len(group) > max_group_size:
            raise ValueError(
                f"Symmetry group size exceeded {max_group_size}. "
                "Possible non-Abelian or incorrectly configured symmetries."
            )

    return group


# -----------------------------------------------------
#! Infer
# -----------------------------------------------------


def infer_momentum_sector_from_operators(
    symmetries: List[SymmetryOperator],
) -> Optional[MomentumSector]:
    """
    Infer the momentum sector from translation symmetries in the list.

    Parameters
    ----------
    symmetries : List[SymmetryOperator]
        List of symmetry operators.

    Returns
    -------
    Optional[MomentumSector]
        The common momentum sector if all translations agree, else None.
    """
    sectors = set()
    for sym in symmetries:
        if isinstance(sym, TranslationSymmetry):
            sector = sym.get_momentum_sector()
            if sector is not None:
                sectors.add(sector)

    if len(sectors) == 1:
        return sectors.pop()
    elif len(sectors) > 1:
        # Mixed sectors – generic
        return MomentumSector.GENERIC
    else:
        return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
