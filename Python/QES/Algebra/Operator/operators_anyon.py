r"""
Abelian anyon onsite operators.

These operators describe hard-core abelian anyons characterised by an exchange
angle ``statistics_angle``.  For ``statistics_angle = \pi`` they reduce to
fermionic operators, whereas ``statistics_angle = 0`` yields hard-core bosons.
"""

from __future__ import annotations

import numpy as np

from QES.Algebra.Operator.catalog import register_local_operator
from QES.Algebra.Hilbert.hilbert_local import LocalOpKernels, LocalSpaceTypes
from QES.Algebra.Operator.operators_hardcore import (
    hardcore_annihilate_int,
    hardcore_annihilate_np,
    hardcore_create_int,
    hardcore_create_np,
    hardcore_number_int,
    hardcore_number_np,
)
from QES.general_python.algebra.utils import DEFAULT_NP_FLOAT_TYPE


def _register_catalog_entries():
    """
    Register default abelian anyon operators with the global catalog.
    """

    def _creation_factory(statistics_angle: float = np.pi / 2) -> LocalOpKernels:
        def _int_kernel(state, ns, sites):
            return hardcore_create_int(state, ns, sites, statistics_angle)

        def _np_kernel(state, sites):
            return hardcore_create_np(state, sites, statistics_angle)

        return LocalOpKernels(
            fun_int=_int_kernel,
            fun_np=_np_kernel,
            fun_jax=None,
            site_parity=1,
            modifies_state=True,
        )

    def _annihilation_factory(statistics_angle: float = np.pi / 2) -> LocalOpKernels:
        def _int_kernel(state, ns, sites):
            return hardcore_annihilate_int(state, ns, sites, statistics_angle)

        def _np_kernel(state, sites):
            return hardcore_annihilate_np(state, sites, statistics_angle)

        return LocalOpKernels(
            fun_int=_int_kernel,
            fun_np=_np_kernel,
            fun_jax=None,
            site_parity=1,
            modifies_state=True,
        )

    def _number_factory(statistics_angle: float | None = None) -> LocalOpKernels:
        def _int_kernel(state, ns, sites):
            out_state, out_coeff = hardcore_number_int(state, ns, sites)
            return out_state, out_coeff.astype(DEFAULT_NP_FLOAT_TYPE, copy=False)

        def _np_kernel(state, sites):
            out_state, out_coeff = hardcore_number_np(state, sites)
            return out_state, out_coeff.astype(DEFAULT_NP_FLOAT_TYPE, copy=False)

        return LocalOpKernels(
            fun_int=_int_kernel,
            fun_np=_np_kernel,
            fun_jax=None,
            site_parity=1,
            modifies_state=False,
        )

    register_local_operator(
        LocalSpaceTypes.ANYON_ABELIAN,
        key="a_dag",
        factory=_creation_factory,
        description="Creation operator for hard-core abelian anyons.",
        algebra="a_i a_j = e^{iθ} a_j a_i (i<j),   a_i^2 = 0",
        sign_convention="Exchange angle θ: creation multiplies by exp(+i θ N_left).",
        tags=("anyon", "creation"),
    )

    register_local_operator(
        LocalSpaceTypes.ANYON_ABELIAN,
        key="a",
        factory=_annihilation_factory,
        description="Annihilation operator for hard-core abelian anyons.",
        algebra="a_i a_j = e^{iθ} a_j a_i (i<j),   a_i^2 = 0",
        sign_convention="Exchange angle θ: annihilation multiplies by exp(-i θ N_left).",
        tags=("anyon", "annihilation"),
    )

    register_local_operator(
        LocalSpaceTypes.ANYON_ABELIAN,
        key="n",
        factory=_number_factory,
        description=r"Occupation operator for hard-core abelian anyons (diagonal).",
        algebra=r"[n_i, a_j\dag] = δ_{ij} a_j\dag,  [n_i, a_j] = -δ_{ij} a_j",
        sign_convention=r"No phase; diagonal in occupation basis.",
        tags=("anyon", "number"),
    )


_register_catalog_entries()


__all__ = ["_register_catalog_entries"]
