"""
J1-J2 Heisenberg spin model.

Supports nearest-neighbor (J1) and next-nearest-neighbor (J2) exchange, local
fields, and optional classical impurity fields.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

try:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Model.Interacting.Spin.hamiltonian_spin import HamiltonianSpin
    from QES.general_python.lattices.lattice import Lattice
except ImportError as e:
    raise ImportError(
        "Failed to import QES modules required by J1J2Model."
    ) from e


class J1J2Model(HamiltonianSpin):
    """
    J1-J2 Heisenberg model with optional local fields and impurities.

    Hamiltonian:
        H = sum_<ij> J1_i (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j)
          + sum_<<ij>> J2_i (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j)
          - sum_i (hx_i Sx_i + hy_i Sy_i + hz_i Sz_i)
          - sum_imp h_imp(site) . S_site
    """

    _ERR_LATTICE_NOT_PROVIDED = "J1J2 requires a lattice."

    def __init__(
        self,
        lattice         : Lattice,
        J1              : Union[List[float], float] = 1.0,
        J2              : Union[List[float], float] = 0.0,
        *,
        hilbert_space   : Optional[HilbertSpace] = None,
        hx              : Union[List[float], float, None] = None,
        hy              : Union[List[float], float, None] = None,
        hz              : Union[List[float], float, None] = None,
        impurities      : Optional[List[Tuple]] = None,
        operator_family : str = "auto",
        dtyp            : type = np.float64,
        backend         : str = "default",
        use_forward     : bool = True,
        **kwargs,
    ):
        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)

        requested_family = self.prepare_spin_family(hilbert_space, operator_family, kwargs)

        complex_required = self.needs_complex_dtype_from_spin_inputs(hy=hy, impurities=impurities)

        super().__init__(
            is_manybody     =   True,
            hilbert_space   =   hilbert_space,
            lattice         =   lattice,
            is_sparse       =   True,
            dtype           =   (np.complex128 if complex_required else kwargs.get("dtype", dtyp)),
            backend         =   backend,
            use_forward     =   use_forward,
            **kwargs,
        )

        self.init_spin_family(requested_family)

        # Couplings and fields are stored as backend arrays or DummyVector.
        self._name  = "J1-J2"
        self._j1    = self._set_some_coupling(J1)
        self._j2    = self._set_some_coupling(J2)
        self._hx    = self._set_some_coupling(hx) if hx is not None else None
        self._hy    = self._set_some_coupling(hy) if hy is not None else None
        self._hz    = self._set_some_coupling(hz) if hz is not None else None

        self._impurities: List[Tuple[int, float, float, float]] = []
        self._setup_impurities(impurities or [])

        self._max_local_ch = 6 + 3 * len(self._impurities)
        self._set_local_energy_operators()
        self.setup_instruction_codes()
        self._set_local_energy_functions()

    def _setup_impurities(self, impurities: List[Tuple]) -> None:
        """
        Normalize impurity tuples:
        - (site, ampl) -> z-polarized
        - (site, phi, theta, ampl) -> spherical direction
        """
        out = []
        for imp in impurities:
            if len(imp) == 2:
                site, ampl = imp
                out.append((int(site), 0.0, 0.0, float(ampl)))
            elif len(imp) == 4:
                site, phi, theta, ampl = imp
                out.append((int(site), float(phi), float(theta), float(ampl)))
            else:
                raise ValueError(
                    "Impurity must be (site, ampl) or (site, phi, theta, ampl)."
                )
        self._impurities = out

    def set_couplings(
        self,
        J1: Union[List[float], float, None] = None,
        J2: Union[List[float], float, None] = None,
        hx: Union[List[float], float, None] = None,
        hy: Union[List[float], float, None] = None,
        hz: Union[List[float], float, None] = None,
    ) -> None:
        """
        Update model couplings/fields.
        """
        if J1 is not None:
            self._j1 = self._set_some_coupling(J1)
        if J2 is not None:
            self._j2 = self._set_some_coupling(J2)
        if hx is not None:
            self._hx = self._set_some_coupling(hx)
        if hy is not None:
            self._hy = self._set_some_coupling(hy)
        if hz is not None:
            self._hz = self._set_some_coupling(hz)

        self._set_local_energy_operators()
        self.setup_instruction_codes()
        self._set_local_energy_functions()

    def _set_local_energy_operators(self):
        """
        Build local and exchange terms.
        """
        super()._set_local_energy_operators()
        lattice = self._lattice
        is_spin_one = self._spin_operator_family == "spin-1"

        op_sx_l = None
        op_sy_l = None
        op_sp_l = None
        op_sm_l = None
        if is_spin_one:
            op_sp_l = self.spin_op(
                "p",
                lattice=lattice, type_act=self._spin_ops_module.OperatorTypeActing.Local
            )
            op_sm_l = self.spin_op(
                "m",
                lattice=lattice, type_act=self._spin_ops_module.OperatorTypeActing.Local
            )
        else:
            op_sx_l = self.spin_op(
                "x",
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Local,
            )
            op_sy_l = self.spin_op(
                "y",
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Local,
            )

        op_sz_l = self.spin_op(
            "z",
            lattice=lattice,
            type_act=self._spin_ops_module.OperatorTypeActing.Local,
        )

        op_sx_sx = None
        op_sy_sy = None
        op_sp_sm = None
        op_sm_sp = None
        if is_spin_one:
            op_sp_sm = self.spin_op(
                "pm",
                lattice=lattice, type_act=self._spin_ops_module.OperatorTypeActing.Correlation
            )
            op_sm_sp = self.spin_op(
                "mp",
                lattice=lattice, type_act=self._spin_ops_module.OperatorTypeActing.Correlation
            )
        else:
            op_sx_sx = self.spin_op(
                "x",
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
            )
            op_sy_sy = self.spin_op(
                "y",
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
            )

        op_sz_sz = self.spin_op(
            "z",
            lattice=lattice,
            type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
        )

        def add_local_axis(axis: str, multiplier, site: int) -> None:
            if not HamiltonianSpin._ADD_CONDITION(multiplier):
                return
            self.add_local_spin_component(
                site,
                axis,
                multiplier,
                op_x=op_sx_l,
                op_y=op_sy_l,
                op_z=op_sz_l,
                op_p=op_sp_l,
                op_m=op_sm_l,
            )

        def add_exchange_isotropic(multiplier, site_i: int, site_j: int) -> None:
            if not HamiltonianSpin._ADD_CONDITION(multiplier):
                return
            self.add_xy_exchange(
                site_i,
                site_j,
                multiplier,
                op_xx=op_sx_sx,
                op_yy=op_sy_sy,
                op_pm=op_sp_sm,
                op_mp=op_sm_sp,
            )
            self.add(op_sz_sz, multiplier=multiplier, modifies=False, sites=[site_i, site_j])

        # Local fields
        for i in range(self.ns):
            if HamiltonianSpin._ADD_CONDITION(self._hx, i):
                add_local_axis("x", -self._hx[i], i)
            if HamiltonianSpin._ADD_CONDITION(self._hy, i):
                add_local_axis("y", -self._hy[i], i)
            if HamiltonianSpin._ADD_CONDITION(self._hz, i):
                add_local_axis("z", -self._hz[i], i)

            for imp_site, phi, theta, ampl in self._impurities:
                if imp_site != i or not HamiltonianSpin._ADD_CONDITION(ampl):
                    continue
                s_t, c_t = np.sin(theta), np.cos(theta)
                s_p, c_p = np.sin(phi), np.cos(phi)
                vx = -ampl * s_t * c_p
                vy = -ampl * s_t * s_p
                vz = -ampl * c_t
                if HamiltonianSpin._ADD_CONDITION(vx):
                    add_local_axis("x", vx, i)
                if HamiltonianSpin._ADD_CONDITION(vy):
                    add_local_axis("y", vy, i)
                if HamiltonianSpin._ADD_CONDITION(vz):
                    add_local_axis("z", vz, i)

        # Nearest-neighbor J1 exchange
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
                jj = phase * self._j1[i]
                add_exchange_isotropic(jj, i, j)

        # Next-nearest-neighbor J2 exchange if lattice supports NNN.
        has_nnn = all(
            hasattr(lattice, fn)
            for fn in ("get_nnn_forward_num", "get_nnn_forward", "get_nnn_num", "get_nnn")
        )
        if has_nnn:
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
                    wx, wy, wz = lattice.bond_winding(i, j)
                    phase = lattice.boundary_phase_from_winding(wx, wy, wz)
                    jj = phase * self._j2[i]
                    add_exchange_isotropic(jj, i, j)

    def __repr__(self) -> str:
        return (
            f"J1J2Model(Ns={self.ns}, J1={Hamiltonian.fmt('J1', self._j1)}, "
            f"J2={Hamiltonian.fmt('J2', self._j2)}, spin_family={self._spin_operator_family})"
        )

    def __str__(self):
        return self.__repr__()
# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------
