"""
file    : Algebra/hamil_energy_helper.py
author  : Maksymilian Kliczkowski
date    : 2025-02-01

This module contains the function to unpack operator terms for Hamiltonian models.
description:
    The function `unpack_operator_terms` unpacks a list of operator terms into separate lists of functions,
    indices, and multiplicative factors, ensuring that there are at least 'ns' operator entries.
    It checks if the provided list of operator terms has fewer than 'ns' entries.
    If so, it appends default operator terms (a tuple with a lambda function returning (x, 0.0),
    and empty lists for indices and multipliers) until its length is equal to 'ns'.
"""

from typing import Any, Callable, List, Tuple

# ListType = TypedList
ListType = list

################################################################################


# Define a default operator function.
def default_operator(x, i):
    """Return the identity-state, zero-value default operator payload."""
    return (x, 0.0)


def default_operator_njit(x, i):
    """Return the Numba-friendly identity-state, zero-value default payload."""
    return (x, 0.0)


def flatten_operator_terms(
    operators, operator_indices=List[List[List[int]]], operator_mult=List[List[Any]]
):
    """
    Flattens the operator terms into a single list of operators, indices, and multiplicative factors.

    Parameters
    ----------
    operators : List[List[Callable]]
        Nested operator-function lists.
    operator_indices : List[List[List[int]]]
        Nested site-index lists matching ``operators``.
    operator_mult : List[List[Any]]
        Nested multiplier lists matching ``operators``.

    Returns
    -------
    Tuple[List[Callable], List[List[int]], List[Any]]
        Flattened operator functions, site indices, and multipliers.
    """

    # Flatten the operator terms
    flat_operators = []
    flat_operator_indices = []
    flat_operator_mult = []

    for i in range(len(operators)):
        for j in range(len(operators[i])):
            flat_operators.append(operators[i][j])
            flat_operator_indices.append(operator_indices[i][j])
            flat_operator_mult.append(operator_mult[i][j])

    return flat_operators, flat_operator_indices, flat_operator_mult


def unpack_operator_terms(
    ns: int, operator_terms: List[Tuple[List[Callable], List[List[int]], List[Any]]]
):
    """
    Unpacks a list of operator terms into separate lists of functions, indices, and multiplicative factors,
    ensuring that there are at least 'ns' operator entries.

    This function checks if the provided list of operator terms has fewer than 'ns' entries.
    If so, it appends default operator terms (a tuple with a lambda function returning (x, 0.0),
    and empty lists for indices and multipliers) until its length is equal to 'ns'. The function then
    unpacks each operator term into three separate numba typed lists:
        - a list of callable functions,
        - a list of index lists,
        - a list of multiplicative factors.

    Parameters
    ----------
    ns : int
        Expected number of operator-entry slots.
    operator_terms : List[Tuple[List[Callable], List[List[int]], List[Any]]]
        Operator-term payload grouped by slot.

    Returns
    -------
    Tuple[list, list, list]
        Function lists, site-index lists, and multiplier lists aligned to the
        expected ``ns`` slots.
    """

    # if len(operator_terms) < ns:
    # for ii in range(ns - len(operator_terms)):
    # operator_terms.append((TypedList(), [[ii]], [0.0]))

    operator_funcs = ListType()
    operator_indices = ListType()
    operator_mult = ListType()

    # handle the unpacking now
    for i in range(ns):
        funcs = ListType()
        sites = ListType()
        mults = ListType()
        for term in operator_terms[i]:
            funcs.append(term[0])
            sites.append(term[1])
            mults.append(term[2])
        if len(funcs) != 0:
            operator_funcs.append(funcs)
            operator_indices.append(sites)
            operator_mult.append(mults)
        else:
            operator_funcs.append(ListType())
            operator_indices.append(ListType())
            operator_mult.append(0.0)

    return operator_funcs, operator_indices, operator_mult


################################################################################
