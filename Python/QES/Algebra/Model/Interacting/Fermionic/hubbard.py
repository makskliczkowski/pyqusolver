"""
Spinless Hubbard-type (t-U) model for interacting fermions.
"""

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np

try:
    import QES.Algebra.Operator.impl.operators_spinless_fermions as ferm_ops
    from QES.Algebra.hamil import Hamiltonian
    from QES.Algebra.hilbert import HilbertSpace
except ImportError as e:
    raise ImportError("Required QES modules are not available.") from e

# -----------------------------------------------------------------------------

class HubbardModel(Hamiltonian):
    r"""
    Spinless t-U model:

        H = -sum_<i,j> ( t_ij c^\dag_i c_j + h.c. )
            + sum_<i,j> U_ij n_i n_j
            - sum_i mu_i n_i

    Note:
        This is a spinless interacting variant.
    """

    _ERR_LATTICE_NOT_PROVIDED = "HubbardModel requires a lattice."

    def __init__(
        self,
        lattice,
        *,
        hilbert_space   : Optional[HilbertSpace] = None,
        t               : Union[float, List[float]] = 1.0,
        U               : Union[float, List[float]] = 1.0,
        mu              : Union[float, List[float], None] = None,
        dtype           : type = np.complex128,
        backend         : str = "default",
        use_forward     : bool = True,
        **kwargs,
    ):
        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)

        if hilbert_space is None:
            kwargs.setdefault("local_space", "spinless-fermions")

        super().__init__(
            is_manybody     =   True,
            hilbert_space   =   hilbert_space,
            lattice         =   lattice,
            is_sparse       =   True,
            dtype           =   dtype,
            backend         =   backend,
            use_forward     =   use_forward,
            **kwargs,
        )

        self._name          = "Spinless Hubbard Model"
        self._t             = self._set_some_coupling(t)
        self._u             = self._set_some_coupling(U)
        self._mu            = self._set_some_coupling(mu) if mu is not None else None
        self._max_local_ch  = 5

        self._set_local_energy_operators()
        self.setup_instruction_codes()
        self._set_local_energy_functions()

    # -------------------------------------------------------------------------

    def set_couplings(
        self,
        *,
        t   : Union[float, List[float], None] = None,
        U   : Union[float, List[float], None] = None,
        mu  : Union[float, List[float], None] = None,
    ) -> None:
        if t is not None:
            self._t = self._set_some_coupling(t)
        if U is not None:
            self._u = self._set_some_coupling(U)
        if mu is not None:
            self._mu = self._set_some_coupling(mu)
        self._set_local_energy_operators()
        self.setup_instruction_codes()
        self._set_local_energy_functions()

    # -------------------------------------------------------------------------

    def _add_hopping_bond(self, op_cdag_c, i: int, j: int, amp):
        self.add(op_cdag_c, multiplier=amp, modifies=True, sites=[i, j])
        self.add(op_cdag_c, multiplier=np.conjugate(amp), modifies=True, sites=[j, i])

    # -------------------------------------------------------------------------

    def _set_local_energy_operators(self):
        ''' Set up local energy operators for the Hubbard model. '''
        super()._set_local_energy_operators()
        lattice         = self._lattice
        op_n_local      = ferm_ops.n(lattice=lattice, type_act=ferm_ops.OperatorTypeActing.Local)
        op_n_local.code = ferm_ops.FermionLookupCodes.n_local
        op_n_n          = ferm_ops.n(lattice=lattice, type_act=ferm_ops.OperatorTypeActing.Correlation)
        op_n_n.code     = ferm_ops.FermionLookupCodes.n_n_corr
        op_cdag_c       = ferm_ops.make_fermionic_mixed("c_dag_c", lattice=lattice, type_act=ferm_ops.OperatorTypeActing.Correlation)
        op_cdag_c.code  = ferm_ops.FermionLookupCodes.c_dag_c_corr

        if self._mu is not None:
            for i in range(self.ns):
                if Hamiltonian._ADD_CONDITION(self._mu, i):
                    self.add(op_n_local, multiplier=-self._mu[i], modifies=False, sites=[i])

        nn_count = (
            [lattice.get_nn_forward_num(i) for i in range(self.ns)]
            if self._use_forward
            else [lattice.get_nn_num(i) for i in range(self.ns)]
        )
        for i in range(self.ns):
            for nidx in range(nn_count[i]):
                j = (
                    lattice.get_nn_forward(i, num=nidx)
                    if self._use_forward
                    else lattice.get_nn(i, num=nidx)
                )
                if lattice.wrong_nei(j):
                    continue
                wx, wy, wz = lattice.bond_winding(i, j)
                phase = lattice.boundary_phase_from_winding(wx, wy, wz)

                hop_amp = -phase * self._t[i]
                if Hamiltonian._ADD_CONDITION(hop_amp):
                    self._add_hopping_bond(op_cdag_c, i, j, hop_amp)

                int_amp = phase * self._u[i]
                if Hamiltonian._ADD_CONDITION(int_amp):
                    self.add(op_n_n, multiplier=int_amp, modifies=False, sites=[i, j])

    def __repr__(self):
        return (
            f"HubbardModel(Ns={self.ns}, "
            f"{Hamiltonian.fmt('t', self._t)}, "
            f"{Hamiltonian.fmt('U', self._u)}, "
            f"{Hamiltonian.fmt('mu', self._mu) if self._mu is not None else 'mu=0'})"
        )

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------