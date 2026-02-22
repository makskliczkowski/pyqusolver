"""
Description:
        This module implements a dummy Hamiltonian class for testing and development purposes.
        It provides a simplified implementation of quantum Hamiltonian mechanics with integer
        and array representations of quantum states. The class includes support for different
        computational backends including Numba and JAX (when available).

        The DummyHamiltonian class inherits from the Hamiltonian base class and implements
        a simple parametric Hamiltonian with diagonal and off-diagonal elements for
        benchmarking and testing matrix operations within the QES framework.
--------------------------------------------------------------------------------
file:       : Algebra/Model/dummy.py
author:     : Maksymilian Kliczkowski
email:      : maksymilian.kliczkowski@pwr.edu.pl
Description : This file defines a dummy Hamiltonian class for testing purposes.
--------------------------------------------------------------------------------
"""

# typing
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from QES.Algebra.hamil import Hamiltonian

# Assume these are available from the QES package:
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Operator.impl import operators_spin as operators_spin_module

##########################################################################################
try:
    from QES.general_python.algebra.utils import (
        DEFAULT_NP_FLOAT_TYPE,
        DEFAULT_NP_INT_TYPE,
        JAX_AVAILABLE,
    )
    from QES.general_python.common import binary as _binary
except ImportError as e:
    raise ImportError(
        "Required modules from QES.general_python.algebra.utils or QES.general_python.common could not be imported."
    ) from e

##########################################################################################

class DummyHamiltonian(Hamiltonian):
    """
    Dummy Hamiltonian class for testing
    """

    def __init__(
        self,
        hilbert_space: Optional[HilbertSpace] = None,
        param: Union[float, complex] = 1.0,
        ns: Optional[int] = None,
        lattice: Optional[Any] = None,
        backend: str = "default",
        dtype=complex,
        build_default_terms: bool = True,
    ):
        """
        Dummy Hamiltonian class for testing
        Parameters
        ----------
        hilbert_space   : HilbertSpace
            Hilbert space object
        param           : float or complex
            Parameter for the Hamiltonian
        """

        # Initialize the Hamiltonian
        if hilbert_space is None:
            if lattice is None and ns is None:
                raise ValueError("Provide either `hilbert_space`, `lattice`, or `ns`.")
            hilbert_space = HilbertSpace(
                ns=ns,
                lattice=lattice,
                backend=backend,
                dtype=dtype,
                local_space="spin-1/2",
            )

        super().__init__(hilbert_space=hilbert_space, lattice=lattice, backend=backend, dtype=dtype)

        self._param = param
        if (
            self.dtype == np.float64 or self.dtype == np.float32 or self.dtype == float
        ) and isinstance(param, complex):
            self._param = np.complex128(param)

        if build_default_terms:
            self._set_local_energy_operators()
            self.finalize_terms()

    # ------------------------------------------------------------------------------------

    def __repr__(self):
        return f"Dummy Hamiltonian with {self.ns} sites and parameter {self._param}"

    def __str__(self):
        return f"dummy,{self.ns},p={self._param}"

    # ------------------------------------------------------------------------------------

    @property
    def param(self):
        """Parameter for the Hamiltonian"""
        return self._param

    @param.setter
    def param(self, value):
        self._param = value
        self.reset_operators()
        self._set_local_energy_operators()
        self.finalize_terms()

    # ------------------------------------------------------------------------------------

    def finalize_terms(self):
        """
        Recompile instruction/local-energy kernels after manual term edits.
        """
        self.setup_instruction_codes()
        self._set_local_energy_functions()

    def _make_pauli_operator(self, axis: str, *, local: bool):
        axis_l = str(axis).lower()
        type_act = (
            operators_spin_module.OperatorTypeActing.Local
            if local
            else operators_spin_module.OperatorTypeActing.Correlation
        )
        if axis_l == "x":
            return operators_spin_module.sig_x(
                lattice=self.lattice, ns=self.ns, type_act=type_act
            )
        if axis_l == "y":
            return operators_spin_module.sig_y(
                lattice=self.lattice, ns=self.ns, type_act=type_act
            )
        if axis_l == "z":
            return operators_spin_module.sig_z(
                lattice=self.lattice, ns=self.ns, type_act=type_act
            )
        raise ValueError(f"Unknown Pauli axis '{axis}'. Use one of x/y/z.")

    def add_local_pauli(
        self,
        axis: str,
        coefficient: Any,
        *,
        sites: Optional[Iterable[int]] = None,
    ) -> int:
        """
        Add local Pauli field term on selected sites.
        """
        op = self._make_pauli_operator(axis, local=True)
        modifies = str(axis).lower() in {"x", "y"}
        return self.add_local_term(op, coefficient, sites=sites, modifies=modifies)

    def add_bond_pauli(
        self,
        axis: str,
        coefficient: Any,
        *,
        bonds: Optional[Iterable[Sequence[int]]] = None,
        order: int = 1,
        unique: bool = True,
    ) -> int:
        """
        Add bond term ``sigma^axis_i sigma^axis_j`` on selected bonds.
        """
        op = self._make_pauli_operator(axis, local=False)
        modifies = str(axis).lower() in {"x", "y"}
        return self.add_bond_term(
            op,
            coefficient,
            bonds=bonds,
            order=order,
            modifies=modifies,
            unique=unique,
        )

    def set_default_chain_terms(self, param: Optional[Union[float, complex]] = None):
        """
        Rebuild default dummy terms (nearest-neighbour XX + staggered Z).
        """
        if param is not None:
            self._param = param
        self.reset_operators()
        self._set_local_energy_operators()
        self.finalize_terms()

    # ------------------------------------------------------------------------------------

    def _set_local_energy_operators(self):
        """Set local energy operators for the Hamiltonian"""

        if self._hilbert_space.local_space.local_dim == 2:
            self._log("Using spin operators", log="info", lvl=1, color="green")
            # sigma^x_i sigma^x_j on NN bonds
            if self.lattice is None:
                bonds = [(i, (i + 1) % self.ns) for i in range(self.ns)]
            else:
                bonds = self.iter_bonds(order=1, unique=True)

            self.add_bond_pauli(
                "x",
                lambda i, j, idx: self._param * i,
                bonds=bonds,
                order=1,
            )
            # staggered sigma^z_i field
            self.add_local_pauli(
                "z",
                lambda i: (self._param / 2.0) * (i % 2),
            )
        else:
            pass

# ----------------------------------------------------------------------------------------
