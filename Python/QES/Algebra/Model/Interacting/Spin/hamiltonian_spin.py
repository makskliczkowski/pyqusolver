"""Shared spin-model Hamiltonian helpers.

This module provides a lightweight base class for interacting spin models that
need to support multiple local spin families (currently spin-1/2 and spin-1)
with consistent operator convenience naming.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    from QES.Algebra.hamil import Hamiltonian
    from QES.Algebra.Model.Interacting.Spin._spin_ops import (
        normalize_spin_component_name,
        normalize_spin_operator_family,
        select_spin_operator_module,
        spin_component_operator,
    )
except ImportError:
    raise ImportError(
        "This module relies on the Hamiltonian base class and spin operator utilities from the same package. "
        "Ensure that the package is properly structured and all dependencies are available."
    )

# ----------------------------------------------------------------------------
#! HamiltonianSpin: Base class for spin Hamiltonians with multi-family operator dispatch
# ----------------------------------------------------------------------------

class HamiltonianSpin(Hamiltonian):
    """
    Base class for spin Hamiltonians with multi-family operator dispatch.
    """

    @classmethod
    def prepare_spin_family(
        cls,
        hilbert_space,
        operator_family : str,
        kwargs          : dict,
    ) -> str:
        """
        Normalize requested spin family and default local-space when needed.
        """
        requested_family = normalize_spin_operator_family(operator_family)
        if hilbert_space is None and requested_family == "spin-1":
            kwargs.setdefault("local_space", "spin-1")
        return requested_family

    def init_spin_family(self, requested_family: str) -> None:
        """
        Bind concrete operator module for the selected/inferred spin family.
        """
        self._spin_ops_module, self._spin_operator_family = select_spin_operator_module(self.hilbert_space, requested_family)

    @property
    def spin_operator_family(self) -> str:
        """Active spin operator family string."""
        return self._spin_operator_family

    @property
    def is_spin_one(self) -> bool:
        """Return True if selected local operators are spin-1."""
        return self._spin_operator_family == "spin-1"

    def spin_op(self, component: str, *, lattice=None, ns=None, type_act=None):
        """
        Build a spin operator using convenience component names.

        Component aliases follow `operators_spin.py` naming style:
        `x/y/z`, `sig_x/sig_y/sig_z`, `p/m`, `sig_p/sig_m`, `pm/mp`.
        """
        return spin_component_operator(
            self._spin_ops_module,
            self._spin_operator_family,
            component,
            lattice=lattice,
            ns=ns,
            type_act=type_act,
        )

    # ------------------------------------------------------------------------

    def add_local_spin_component(
        self,
        site        : int,
        component   : str,
        coefficient,
        *,
        op_x=None,
        op_y=None,
        op_z=None,
        op_p=None,
        op_m=None,
    ) -> int:
        """
        Add local single-site term for component x/y/z.

        For spin-1:
            Sx = (S+ + S-) / 2
            Sy = (S+ - S-) / (2i)
        """
        axis = normalize_spin_component_name(component)
        if axis not in {"x", "y", "z"}:
            raise ValueError("Local component must be x, y, or z.")
        if not Hamiltonian._ADD_CONDITION(coefficient):
            return 0

        added = 0
        if not self.is_spin_one:
            if axis == "x":
                op = op_x or self.spin_op("x", type_act=self._spin_ops_module.OperatorTypeActing.Local)
                self.add(op, multiplier=coefficient, modifies=True, sites=[site])
            elif axis == "y":
                op = op_y or self.spin_op("y", type_act=self._spin_ops_module.OperatorTypeActing.Local)
                self.add(op, multiplier=coefficient, modifies=True, sites=[site])
            else:
                op = op_z or self.spin_op("z", type_act=self._spin_ops_module.OperatorTypeActing.Local)
                self.add(op, multiplier=coefficient, modifies=False, sites=[site])
            return 1

        # Spin-1 decomposition for transverse components.
        op_plus     = op_p or self.spin_op("p", type_act=self._spin_ops_module.OperatorTypeActing.Local)
        op_minus    = op_m or self.spin_op("m", type_act=self._spin_ops_module.OperatorTypeActing.Local)
        if axis == "x":
            half = 0.5 * coefficient
            self.add(op_plus, multiplier=half, modifies=True, sites=[site])
            self.add(op_minus, multiplier=half, modifies=True, sites=[site])
            added  += 2
        elif axis == "y":
            self.add(op_plus, multiplier=(-0.5j) * coefficient, modifies=True, sites=[site])
            self.add(op_minus, multiplier=(0.5j) * coefficient, modifies=True, sites=[site])
            added  += 2
        else:
            op = op_z or self.spin_op("z", type_act=self._spin_ops_module.OperatorTypeActing.Local)
            self.add(op, multiplier=coefficient, modifies=False, sites=[site])
            added  += 1
        return added

    def add_xy_exchange(
        self,
        site_i  : int,
        site_j  : int,
        coefficient,
        *,
        op_xx   =   None,
        op_yy   =   None,
        op_pm   =   None,
        op_mp   =   None,
    ) -> int:
        """
        Add XY exchange: coefficient * (Sx_i Sx_j + Sy_i Sy_j).
        """
        if not Hamiltonian._ADD_CONDITION(coefficient):
            return 0

        if not self.is_spin_one:
            opx = op_xx or self.spin_op("x", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)
            opy = op_yy or self.spin_op("y", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)
            self.add(opx, multiplier=coefficient, modifies=True, sites=[site_i, site_j])
            self.add(opy, multiplier=coefficient, modifies=True, sites=[site_i, site_j])
            return 2

        # SxSx + SySy = 0.5 (S+S- + S-S+)
        ppm = op_pm or self.spin_op("pm", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)
        pmp = op_mp or self.spin_op("mp", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)
        half = 0.5 * coefficient
        self.add(ppm, multiplier=half, modifies=True, sites=[site_i, site_j])
        self.add(pmp, multiplier=half, modifies=True, sites=[site_i, site_j])
        return 2

    def add_sx_sy_exchange_general(
        self,
        site_i: int,
        site_j: int,
        coeff_sx,
        coeff_sy,
        *,
        op_xx=None,
        op_yy=None,
        op_pp=None,
        op_mm=None,
        op_pm=None,
        op_mp=None,
    ) -> int:
        """
        Add general two-site transverse exchange:
            coeff_sx * Sx_i Sx_j + coeff_sy * Sy_i Sy_j
        """
        added = 0
        if not self.is_spin_one:
            if Hamiltonian._ADD_CONDITION(coeff_sx):
                opx = op_xx or self.spin_op(
                    "x", type_act=self._spin_ops_module.OperatorTypeActing.Correlation
                )
                self.add(opx, multiplier=coeff_sx, modifies=True, sites=[site_i, site_j])
                added += 1
            if Hamiltonian._ADD_CONDITION(coeff_sy):
                opy = op_yy or self.spin_op(
                    "y", type_act=self._spin_ops_module.OperatorTypeActing.Correlation
                )
                self.add(opy, multiplier=coeff_sy, modifies=True, sites=[site_i, site_j])
                added += 1
            return added

        # Spin-1 decomposition:
        # SxSx = 1/4(SpSp + SpSm + SmSp + SmSm)
        # SySy = -1/4(SpSp - SpSm - SmSp + SmSm)
        c_pp = 0.25 * (coeff_sx - coeff_sy)
        c_mm = 0.25 * (coeff_sx - coeff_sy)
        c_pm = 0.25 * (coeff_sx + coeff_sy)
        c_mp = 0.25 * (coeff_sx + coeff_sy)

        opp = op_pp or self.spin_op("p", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)
        omm = op_mm or self.spin_op("m", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)
        opm = op_pm or self.spin_op("pm", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)
        omp = op_mp or self.spin_op("mp", type_act=self._spin_ops_module.OperatorTypeActing.Correlation)

        if Hamiltonian._ADD_CONDITION(c_pp):
            self.add(opp, multiplier=c_pp, modifies=True, sites=[site_i, site_j])
            added += 1
        if Hamiltonian._ADD_CONDITION(c_mm):
            self.add(omm, multiplier=c_mm, modifies=True, sites=[site_i, site_j])
            added += 1
        if Hamiltonian._ADD_CONDITION(c_pm):
            self.add(opm, multiplier=c_pm, modifies=True, sites=[site_i, site_j])
            added += 1
        if Hamiltonian._ADD_CONDITION(c_mp):
            self.add(omp, multiplier=c_mp, modifies=True, sites=[site_i, site_j])
            added += 1
        return added

    @staticmethod
    def needs_complex_dtype_from_spin_inputs(
        hy          =   None,
        impurities  =   None,
    ) -> bool:
        """
        Detect whether Y-like terms force complex dtype.
        """
        if Hamiltonian._ADD_CONDITION(hy):
            return True
        
        if impurities is None:
            return False
        
        for imp in impurities:
            if not isinstance(imp, tuple) or len(imp) < 4:
                continue
            _, phi, theta, ampl = imp[0], imp[1], imp[2], imp[3]
            y_amp               = float(ampl) * np.sin(float(theta)) * np.sin(float(phi))
            if abs(y_amp) > 1e-15:
                return True
        return False

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------