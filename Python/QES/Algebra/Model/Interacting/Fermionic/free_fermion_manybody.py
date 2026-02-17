"""
Many-body free spinless fermion model on a lattice.
This module defines the ManyBodyFreeFermions class, 
which constructs the Hamiltonian for a system of non-interacting spinless fermions 
hopping on a lattice, with optional chemical potential and next-nearest-neighbor hopping. 
The Hamiltonian is built using the QES framework and supports various lattice geometries 
and boundary conditions.
"""

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np

try:
    import  QES.Algebra.Operator.impl.operators_spinless_fermions as ferm_ops
    from    QES.Algebra.hamil import Hamiltonian
    from    QES.Algebra.hilbert import HilbertSpace
except ImportError as e:
    raise ImportError("Required QES modules are not available.") from e

# ----------------------------------------------------------------------------

class ManyBodyFreeFermions(Hamiltonian):
    r"""
    Spinless many-body free-fermion Hamiltonian:

        H = -sum_<i,j> ( t_ij c^\dag_i c_j + h.c. ) - sum_i mu_i n_i

    with optional next-nearest-neighbor hopping `t2`.
    """

    _ERR_LATTICE_NOT_PROVIDED = "ManyBodyFreeFermions requires a lattice."

    def __init__(
        self,
        lattice,
        *,
        hilbert_space   : Optional[HilbertSpace] = None,
        t               : Union[float, List[float]] = 1.0,
        t2              : Union[float, List[float], None] = None,
        mu              : Union[float, List[float], None] = None,
        dtype           : type = np.complex128,
        backend         : str = "default",
        use_forward     : bool = True,
        **kwargs,
    ):
        '''Initialize the many-body free fermion Hamiltonian.'''
        
        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)

        if hilbert_space is None:
            kwargs.setdefault("local_space", "spinless-fermions")

        super().__init__(
            is_manybody=True,
            hilbert_space=hilbert_space,
            lattice=lattice,
            is_sparse=True,
            dtype=dtype,
            backend=backend,
            use_forward=use_forward,
            **kwargs,
        )

        self._name          = "Many-Body Free Fermions"
        self._t             = self._set_some_coupling(t)
        self._t2            = self._set_some_coupling(t2) if t2 is not None else None
        self._mu            = self._set_some_coupling(mu) if mu is not None else None
        self._max_local_ch  = 4 # Two for NN hopping, two for NNN hopping (if present)

        self._set_local_energy_operators()
        self.setup_instruction_codes()
        self._set_local_energy_functions()

    def set_couplings(
        self,
        *,
        t   : Union[float, List[float], None] = None,
        t2  : Union[float, List[float], None] = None,
        mu  : Union[float, List[float], None] = None,
    ) -> None:
        if t is not None:
            self._t = self._set_some_coupling(t)
        if t2 is not None:
            self._t2 = self._set_some_coupling(t2)
        if mu is not None:
            self._mu = self._set_some_coupling(mu)
        self._set_local_energy_operators()
        self.setup_instruction_codes()
        self._set_local_energy_functions()

    def _add_hopping_bond(self, op_cdag_c, i: int, j: int, amp):
        # c^\dag_i c_j
        # h.c. : c^\dag_j c_i
        self.add(op_cdag_c, multiplier=amp, modifies=True, sites=[i, j])
        self.add(op_cdag_c, multiplier=np.conjugate(amp), modifies=True, sites=[j, i])

    def _set_local_energy_operators(self):
        '''Set up local energy operators for the free fermion model.'''
        
        super()._set_local_energy_operators()
        
        lattice         = self._lattice
        op_n_local      = ferm_ops.n(lattice=lattice, type_act=ferm_ops.OperatorTypeActing.Local)
        op_n_local.code = ferm_ops.FermionLookupCodes.n_local
        op_cdag_c       = ferm_ops.make_fermionic_mixed(
            "c_dag_c", lattice=lattice, type_act=ferm_ops.OperatorTypeActing.Correlation
        )
        op_cdag_c.code  = ferm_ops.FermionLookupCodes.c_dag_c_corr

        # Chemical potential
        if self._mu is not None:
            for i in range(self.ns):
                if Hamiltonian._ADD_CONDITION(self._mu, i):
                    self.add(op_n_local, multiplier=-self._mu[i], modifies=False, sites=[i])

        # NN hopping
        nn_count    = (
            [lattice.get_nn_forward_num(i) for i in range(self.ns)]
            if self._use_forward
            else [lattice.get_nn_num(i) for i in range(self.ns)]
        )
        
        # Loop over sites and their neighbors to add hopping terms
        for i in range(self.ns):
            for nidx in range(nn_count[i]):
                j = (
                    lattice.get_nn_forward(i, num=nidx)
                    if self._use_forward
                    else lattice.get_nn(i, num=nidx)
                )
                
                if lattice.wrong_nei(j):
                    continue
                
                wx, wy, wz  = lattice.bond_winding(i, j)
                phase       = lattice.boundary_phase_from_winding(wx, wy, wz)
                amp         = -phase * self._t[i]
                if Hamiltonian._ADD_CONDITION(amp):
                    self._add_hopping_bond(op_cdag_c, i, j, amp)

        # Optional NNN hopping
        if self._t2 is not None and all(
            hasattr(lattice, fn)
            for fn in ("get_nnn_forward_num", "get_nnn_forward", "get_nnn_num", "get_nnn")
        ):
            nnn_count = (
                [lattice.get_nnn_forward_num(i) for i in range(self.ns)]
                if self._use_forward
                else [lattice.get_nnn_num(i) for i in range(self.ns)]
            )
            for i in range(self.ns):
                for nidx in range(nnn_count[i]):
                    j = (
                        lattice.get_nnn_forward(i, num=nidx)
                        if self._use_forward
                        else lattice.get_nnn(i, num=nidx)
                    )
                    if lattice.wrong_nei(j):
                        continue
                    wx, wy, wz  = lattice.bond_winding(i, j)
                    phase       = lattice.boundary_phase_from_winding(wx, wy, wz)
                    amp         = -phase * self._t2[i]
                    if Hamiltonian._ADD_CONDITION(amp):
                        self._add_hopping_bond(op_cdag_c, i, j, amp)

    def __repr__(self):
        return (
            f"ManyBodyFreeFermions(Ns={self.ns}, "
            f"{Hamiltonian.fmt('t', self._t)}, "
            f"{Hamiltonian.fmt('t2', self._t2) if self._t2 is not None else 't2=0'}, "
            f"{Hamiltonian.fmt('mu', self._mu) if self._mu is not None else 'mu=0'})"
        )

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------