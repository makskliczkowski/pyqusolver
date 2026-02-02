"""
file    : Model/Noninteracting/syk.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl

Description
-----------
Simple SYK model for non-interacting fermions.
"""

from    typing import Any, Optional, Union, TYPE_CHECKING
import  numpy as np


# import the quadratic base
try:
    from QES.Algebra.hamil_quadratic            import QuadraticHamiltonian
    from QES.general_python.algebra.ran_wrapper import RMT, random_matrix
    
    if TYPE_CHECKING:
        from QES.Algebra.hilbert                import HilbertSpace
    
except ImportError:
    raise ImportError("Could not import QES module. Ensure that pyqusolver is installed and accessible.")

# ---------------------------------------------------------------------
#! HAMILTONIAN
# ---------------------------------------------------------------------

class SYK2(QuadraticHamiltonian):
    r"""
    Simple SYK model for non-interacting fermions.
    The Hamiltonian is defined as:

    .. math::
        H = \sum_{i=1}^{N} \epsilon_i c_i^\dagger c_i,
    where :math:`c_i^\dagger` and :math:`c_i` are the creation and
    annihilation operators for fermions at site :math:`i`,
    and :math:`\epsilon_i` are random energies drawn from a Gaussian distribution.

    where :math:`\epsilon_i` are random energies drawn from a Gaussian distribution.

    A complex SYK2 model includes a GUE random matrix as the coupling matrix.
    Conversely, a real SYK2 model includes a GOE random matrix.
    """

    def __init__(
        self,
        ns              : int,
        hilbert_space   : Optional[Union[str, "HilbertSpace"]] = None,
        dtype           : np.dtype = np.dtype(np.float64),
        backend         : str = "default",
        seed            : Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            ns=ns,
            is_sparse=False,
            hilbert_space=hilbert_space,
            dtype=dtype,
            seed=seed,
            backend=backend,
            **kwargs,
        )
        self._name = "SYK2"

    def _hamiltonian_quadratic(self, use_numpy=False):
        """
        Create the Hamiltonian matrix for the SYK model.
        """
        self._log("Building SYK2 Hamiltonian...", lvl=2, color="green", log='debug')
        if self._iscpx:
            mat = random_matrix(
                    shape   =   (self._ns, self._ns),
                    typek   =   RMT.GUE,
                    backend =   "np" if use_numpy else self._backend,
                    dtype   =   self._dtype,
                )
        else:
            mat = random_matrix(
                    shape   =   (self._ns, self._ns),
                    typek   =   RMT.GOE,
                    backend =   "np" if use_numpy else self._backend,
                    dtype   =   self._dtype,
                )
        # Ensure _hamil_sp is always a numpy ndarray
        if not isinstance(mat, np.ndarray):
            mat = np.array(mat, dtype=self._dtype)
        self._hamil_sp = mat

    def add_term(self, *args, **kwargs):
        raise NotImplementedError("Add term not implemented for SYK2 model.")

    @staticmethod
    def repr(ns: int, cpx: bool = True) -> str:
        return f"SYK2(ns={ns},cpx={cpx})"

    def __repr__(self):
        return f"SYK2(ns={self._ns},cpx={self._iscpx})"

    def __str__(self):
        return self.__repr__()

# ---------------------------------------------------------------------
#! END OF HAMILTONIAN
# ---------------------------------------------------------------------
