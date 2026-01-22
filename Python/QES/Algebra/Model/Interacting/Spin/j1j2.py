"""
J1-J2 Heisenberg Model for QES
-----------------------------
Implements the J1-J2 spin model with impurity support, following the structure of HeisenbergKitaev.

File    : QES/Algebra/Model/Interacting/Spin/j1j2.py
Author  : Maksymilian Kliczkowski
Date    : 2025-12-22
"""

from typing import List, Optional, Tuple, Union

import numpy as np

try:
    from QES.Algebra.hamil import Hamiltonian
    from QES.Algebra.hilbert import HilbertSpace
    from QES.general_python.lattices.lattice import Lattice, LatticeType
except ImportError as e:
    raise ImportError(
        "Failed to import QES modules. Ensure that the QES package is correctly installed."
    ) from e

##########################################################################################
# DEFINE CONSTANTS
##########################################################################################

J1J2_X_BOND_NEI = 0
J1J2_Y_BOND_NEI = 1
J1J2_Z_BOND_NEI = 2
SINGLE_TERM_MULT = 2.0
CORR_TERM_MULT = 4.0

##########################################################################################
# HAMILTONIAN CLASS
##########################################################################################


class J1J2Model(Hamiltonian):
    """
    J1-J2 Heisenberg model with optional impurities.
    """

    _ERR_LATTICE_NOT_PROVIDED = "J1J2: Lattice must be provided to define the J1-J2 Hamiltonian."

    def __init__(
        self,
        lattice: Lattice,
        J1: Union[List[float], float] = 1.0,
        J2: Union[List[float], float] = 0.0,
        *,
        hilbert_space: Optional[HilbertSpace] = None,
        impurities: List[Tuple] = [],
        dtype: type = np.float64,
        backend: str = "default",
        use_forward: bool = True,
        **kwargs,
    ):
        """
        Parameters:
            lattice : Lattice
                The lattice on which the Hamiltonian is defined.
            J1 : float or list
                Nearest-neighbor coupling.
            J2 : float or list
                Next-nearest-neighbor coupling.
            hilbert_space : Optional[HilbertSpace]
                Predefined Hilbert space. If None, it will be created based on ns.
            impurities : List[Tuple]
                List of classical impurities. Same format as HeisenbergKitaev.
            dtype : type
                Data type for numerical computations.
            backend : str
                Computational backend to use.
            use_forward : bool
                Whether to use forward nearest neighbors only.
            **kwargs : additional keyword arguments
        """
        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)

        self._lattice = lattice
        self.J1 = J1
        self.J2 = J2
        self.impurities = impurities
        self.dtype = dtype
        self.backend = backend
        self.use_forward = use_forward
        self._setup_impurities(impurities)
        super().__init__(hilbert_space=hilbert_space, **kwargs)

    def _setup_impurities(self, impurities: List[Tuple]):
        self._impurities = []
        for imp in impurities:
            if len(imp) == 2:
                site, ampl = imp
                self._impurities.append((site, 0.0, 0.0, ampl))
            elif len(imp) == 4:
                self._impurities.append(tuple(imp))
            else:
                raise ValueError("Impurity must be (site, ampl) or (site, phi, theta, ampl)")

    def __repr__(self) -> str:
        return f"J1J2Model(lattice={self._lattice}, J1={self.J1}, J2={self.J2}, impurities={self.impurities})"

    def __str__(self):
        return self.__repr__()

    # ------------------------------------------------------------------------------------


##########################################################################################
# EOF
##########################################################################################
