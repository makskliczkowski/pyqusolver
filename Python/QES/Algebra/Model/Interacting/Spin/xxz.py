"""
High-level Hamiltonian class for the XXZ Model.

The XXZ model is a fundamental spin-1/2 chain with anisotropic interactions.

--------------------
File    : Model/Interacting/Spin/xxz.py
Author  : Maksymilian Kliczkowski
Date    : 2025-11-13
Version : 1.0
Changes :
    2025-11-13 (1.0) : Initial implementation based on TFIM pattern. - MK
--------------------
"""

from typing import List, Optional, Union

import numpy as np

# QES package imports
try:
    import QES.Algebra.hamil as hamil_module
    import QES.Algebra.hilbert as hilbert_module
    from QES.Algebra.Model.Interacting.Spin._spin_ops import (
        normalize_spin_operator_family,
        select_spin_operator_module,
        spin_axis_operator,
    )
except ImportError as e:
    raise ImportError("Required QES modules are not available.") from e

# Utilities
try:
    from QES.general_python.lattices.lattice import Lattice, LatticeType
except ImportError as e:
    raise ImportError("Required QES lattice module is not available.") from e

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################


class XXZ(hamil_module.Hamiltonian):
    r"""
    Hamiltonian for the XXZ Model.

    The Hamiltonian is defined as:
        H = -\sum_{\langle i,j \rangle} [J_xy (\sigma^x_i \sigma^x_j + \sigma^y_i \sigma^y_j) + J_z \sigma^z_i \sigma^z_j]
            - h_x \sum_i \sigma^x_i - h_z \sum_i \sigma^z_i

    Alternatively, using anisotropy parameter Δ:
        H = -J \sum_{\langle i,j \rangle} [\sigma^x_i \sigma^x_j + \sigma^y_i \sigma^y_j + \Delta \sigma^z_i \sigma^z_j]
            - h_x \sum_i \sigma^x_i - h_z \sum_i \sigma^z_i

    where:
        - \sigma^α_i are Pauli operators at site i (α = x, y, z).
        - \langle i,j \rangle denotes summation over nearest neighbors.
        - J_xy (or J) is the XY coupling strength.
        - J_z (or J*Δ) is the Ising (Z) coupling strength.
        - Δ = J_z / J_xy is the anisotropy parameter.
        - h_x is the transverse magnetic field.
        - h_z is the longitudinal magnetic field.

    Special cases:
        - Δ = 0: XY model (no Ising interaction)
        - Δ = 1: XXX model (Heisenberg, isotropic)
        - Δ -> ∞: Ising model
        - h_x = 0, h_z = 0: No external fields

    Symmetries:
        - Translation: Always present for uniform couplings
        - U(1): Present when h_x = h_z = 0 (conserves total magnetization S^z_total)
        - Parity_z: Present when h_x = 0 (spin-flip symmetry)
        - Parity_x: Present when h_z = 0 and Δ = 1 (for XXX model)
    """

    _ERR_EITHER_HIL_OR_NS = "XXZ: either the Hilbert space or the number of sites must be provided."

    def __init__(
        self,
        lattice: Lattice,
        hilbert_space: Optional[hilbert_module.HilbertSpace] = None,
        jxy: Union[List[float], float] = 1.0,
        jz: Union[List[float], float] = 1.0,
        delta: Union[List[float], float, None] = None,
        hx: Union[List[float], float] = 0.0,
        hz: Union[List[float], float] = 0.0,
        operator_family: str = "auto",
        dtype: type = np.float64,
        backend: str = "default",
        **kwargs,
    ):
        """
        Constructor for the XXZ Model Hamiltonian.

        ---
        Parameters:
            lattice : Lattice:
                The lattice structure defining sites and neighbors. Required.
            hilbert_space : Optional[hilbert_module.HilbertSpace]:
                The Hilbert space. If None, created based on lattice size.
            jxy : Union[List[float], float]:
                XY coupling strength J_xy. Default is 1.0.
                If delta is provided, this is interpreted as J with J_z = J * delta.
            jz : Union[List[float], float]:
                Ising (Z) coupling strength J_z. Default is 1.0.
                Ignored if delta is provided.
            delta : Union[List[float], float, None]:
                Anisotropy parameter Δ = J_z / J_xy. If provided, overrides jz.
                Default is None (uses jxy and jz directly).
            hx : Union[List[float], float]:
                Transverse magnetic field h_x. Default is 0.0.
            hz : Union[List[float], float]:
                Longitudinal magnetic field h_z. Default is 0.0.
            operator_family : str:
                Operator family to use: "auto", "spin-1/2", or "spin-1".
            dtype : type:
                Data type for the Hamiltonian (default: np.float64).
            backend : str:
                Backend for computations (default: "default").
            **kwargs :
                Additional keyword arguments passed to the base Hamiltonian class.
        """

        if lattice is None:
            raise ValueError("XXZ requires a 'lattice' object to define neighbors.")

        requested_family = normalize_spin_operator_family(operator_family)
        if hilbert_space is None and requested_family == "spin-1":
            kwargs.setdefault("local_space", "spin-1")

        # Initialize the base Hamiltonian class
        super().__init__(
            is_manybody=True,
            hilbert_space=hilbert_space,
            lattice=lattice,
            is_sparse=True,
            dtype=np.complex128,  # enforce complex dtype to support Sy operations safely
            backend=backend,
            **kwargs,
        )
        self._spin_ops_module, self._spin_operator_family = select_spin_operator_module(
            self.hilbert_space, requested_family
        )

        # Set Hamiltonian attributes
        self._name = "XXZ Model"
        self._is_sparse = True
        self._is_manybody = True

        # Initialize parameters
        self._jxy = None
        self._jz = None
        self._delta = None
        self._hx = None
        self._hz = None

        # Maximum local channels: 3 (SxSx + SySy + SzSz) + 2 (hx + hz)
        self._max_local_ch = 5

        # Set couplings
        self.set_couplings(jxy=jxy, jz=jz, delta=delta, hx=hx, hz=hz)

        # Build the Hamiltonian
        self._set_local_energy_operators()
        self.setup_instruction_codes()
        self._set_local_energy_functions()
        self._log(f"XXZ Hamiltonian initialized for {self.ns} sites.", lvl=1, log="debug")

    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Concise, human-readable description of the XXZ instance.

        Examples
        --------
        >>> print(xxz)
        XXZ(Ns=8, Jxy=1.000, Jz=0.500, Δ=0.500, hx=0.000, hz=0.000)
        """
        prec = 3
        sep = ", "

        parts = [f"XXZ(Ns={self.ns}"]

        parts += [
            hamil_module.Hamiltonian.fmt("Jxy", self._jxy, prec=prec),
            hamil_module.Hamiltonian.fmt("Jz", self._jz, prec=prec),
            (
                hamil_module.Hamiltonian.fmt("Δ", self._delta, prec=prec)
                if self._delta is not None
                else ""
            ),
            hamil_module.Hamiltonian.fmt("hx", self._hx, prec=prec),
            hamil_module.Hamiltonian.fmt("hz", self._hz, prec=prec),
        ]

        return sep.join([p for p in parts if p]) + ")"

    def __str__(self):
        return self.__repr__()

    # ----------------------------------------------------------------------------------------------
    #! SETTERS
    # ----------------------------------------------------------------------------------------------

    def set_couplings(
        self,
        jxy: Union[List[float], float, None] = None,
        jz: Union[List[float], float, None] = None,
        delta: Union[List[float], float, None] = None,
        hx: Union[List[float], float, None] = None,
        hz: Union[List[float], float, None] = None,
        **kwargs,
    ):
        """
        Sets or updates the XXZ couplings and fields.

        Parameters:
            jxy : Optional[Union[List[float], float]]:
                XY coupling strength(s).
            jz : Optional[Union[List[float], float]]:
                Z coupling strength(s).
            delta : Optional[Union[List[float], float]]:
                Anisotropy parameter(s). If provided, sets jz = jxy * delta.
            hx : Optional[Union[List[float], float]]:
                Transverse field strength(s).
            hz : Optional[Union[List[float], float]]:
                Longitudinal field strength(s).
        """
        # Set Jxy first
        if jxy is not None:
            self._jxy = self._set_some_coupling(jxy)

        # Handle delta vs jz priority
        if delta is not None:
            self._delta = self._set_some_coupling(delta)
            # Compute Jz from delta
            if self._jxy is not None:
                if isinstance(self._delta, list):
                    self._jz = [self._jxy[i] * self._delta[i] for i in range(len(self._delta))]
                else:
                    if isinstance(self._jxy, list):
                        self._jz = [j * self._delta for j in self._jxy]
                    else:
                        self._jz = self._jxy * self._delta
        elif jz is not None:
            self._jz = self._set_some_coupling(jz)
            # Compute delta from jz and jxy if jxy is non-zero
            if self._jxy is not None and self._jz is not None:
                # Check if jxy is zero to avoid division by zero
                jxy_is_zero = False
                if isinstance(self._jxy, list):
                    jxy_is_zero = all(abs(x) < 1e-15 for x in self._jxy)
                else:
                    # Could be DummyVector, check its value
                    jxy_val = self._jxy[0] if hasattr(self._jxy, "__getitem__") else self._jxy
                    jxy_is_zero = abs(jxy_val) < 1e-15

                if not jxy_is_zero:
                    if isinstance(self._jxy, list) and isinstance(self._jz, list):
                        self._delta = [
                            self._jz[i] / self._jxy[i] if abs(self._jxy[i]) > 1e-15 else 0.0
                            for i in range(len(self._jxy))
                        ]
                    elif isinstance(self._jxy, list):
                        self._delta = [self._jz / j if abs(j) > 1e-15 else 0.0 for j in self._jxy]
                    elif isinstance(self._jz, list):
                        jxy_val = self._jxy[0] if hasattr(self._jxy, "__getitem__") else self._jxy
                        self._delta = [
                            jz_val / jxy_val if abs(jxy_val) > 1e-15 else 0.0 for jz_val in self._jz
                        ]
                    else:
                        jxy_val = self._jxy[0] if hasattr(self._jxy, "__getitem__") else self._jxy
                        jz_val = self._jz[0] if hasattr(self._jz, "__getitem__") else self._jz
                        self._delta = jz_val / jxy_val if abs(jxy_val) > 1e-15 else 0.0
                else:
                    # Jxy is zero, delta is undefined or infinite
                    self._delta = None

        # Set fields
        if hx is not None:
            self._hx = self._set_some_coupling(hx)
        if hz is not None:
            self._hz = self._set_some_coupling(hz)

    def _set_local_energy_operators(self):
        """
        Builds the list of operators defining the XXZ Hamiltonian.

        Adds terms for:
            - Transverse field: -hx[i] * Sx_i for each site i
            - Longitudinal field: -hz[i] * Sz_i for each site i
            - XY coupling: -Jxy[i] * (Sx_i * Sx_j + Sy_i * Sy_j) for each NN pair <i,j>
            - Ising coupling: -Jz[i] * Sz_i * Sz_j for each NN pair <i,j>
        """
        super()._set_local_energy_operators()
        self._log("Building XXZ operator list...", lvl=1, log="info")

        lattice = self._lattice

        # Define Base Operators
        # Local operators (act on one site)
        op_sx_l = None
        op_sp_l = None
        op_sm_l = None
        if self._spin_operator_family == "spin-1":
            op_sp_l = self._spin_ops_module.s1_plus(
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Local,
            )
            op_sm_l = self._spin_ops_module.s1_minus(
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Local,
            )
        else:
            op_sx_l = spin_axis_operator(
                self._spin_ops_module,
                self._spin_operator_family,
                "x",
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Local,
            )
        op_sz_l = spin_axis_operator(
            self._spin_ops_module,
            self._spin_operator_family,
            "z",
            lattice=lattice,
            type_act=self._spin_ops_module.OperatorTypeActing.Local,
        )

        # Correlation operators (act on two sites)
        op_sx_sx_c = None
        op_sy_sy_c = None
        op_sp_sm_c = None
        op_sm_sp_c = None
        if self._spin_operator_family == "spin-1":
            op_sp_sm_c = self._spin_ops_module.s1_pm(
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
            )
            op_sm_sp_c = self._spin_ops_module.s1_mp(
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
            )
        else:
            op_sx_sx_c = spin_axis_operator(
                self._spin_ops_module,
                self._spin_operator_family,
                "x",
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
            )
            op_sy_sy_c = spin_axis_operator(
                self._spin_ops_module,
                self._spin_operator_family,
                "y",
                lattice=lattice,
                type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
            )
        op_sz_sz_c = spin_axis_operator(
            self._spin_ops_module,
            self._spin_operator_family,
            "z",
            lattice=lattice,
            type_act=self._spin_ops_module.OperatorTypeActing.Correlation,
        )

        # Add Transverse Field Terms (hx)
        if self._hx is not None:
            for i in range(self.ns):
                if not np.isclose(self._hx[i], 0.0):
                    if self._spin_operator_family == "spin-1":
                        mult = -0.5 * self._hx[i]
                        self.add(operator=op_sp_l, multiplier=mult, sites=[i], modifies=True)
                        self.add(operator=op_sm_l, multiplier=mult, sites=[i], modifies=True)
                        self._log(
                            f"Adding spin-1 Sx decomposition at site {i} with multiplier {-self._hx[i]:.3f}",
                            lvl=2,
                            log="debug",
                        )
                    else:
                        self.add(operator=op_sx_l, multiplier=-self._hx[i], sites=[i], modifies=True)
                        self._log(
                            f"Adding Sx at site {i} with multiplier {-self._hx[i]:.3f}",
                            lvl=2,
                            log="debug",
                        )

        # Add Longitudinal Field Terms (hz)
        if self._hz is not None:
            for i in range(self.ns):
                if not np.isclose(self._hz[i], 0.0):
                    self.add(operator=op_sz_l, multiplier=-self._hz[i], sites=[i], modifies=False)
                    self._log(
                        f"Adding Sz at site {i} with multiplier {-self._hz[i]:.3f}",
                        lvl=2,
                        log="debug",
                    )

        # Add Interaction Terms (XY and Ising)
        for i in range(self.ns):
            # Use get_nn_forward to avoid double counting bonds
            nn_forward_num = lattice.get_nn_forward_num(i)

            for nn_idx in range(nn_forward_num):
                j_neighbor = lattice.get_nn_forward(i, num=nn_idx)
                if lattice.wrong_nei(j_neighbor):
                    continue

                wx, wy, wz = lattice.bond_winding(i, j_neighbor)
                phase = lattice.boundary_phase_from_winding(wx, wy, wz)

                # XY coupling: Jxy * (Sx_i Sx_j + Sy_i Sy_j)
                if self._jxy is not None:
                    coupling_xy = self._jxy[i] * phase

                    if not np.isclose(coupling_xy, 0.0):
                        if self._spin_operator_family == "spin-1":
                            # SxSx + SySy = 0.5 * (S+S- + S-S+)
                            ladder_mult = -0.5 * coupling_xy
                            self.add(
                                operator=op_sp_sm_c,
                                multiplier=ladder_mult,
                                sites=[i, j_neighbor],
                                modifies=True,
                            )
                            self.add(
                                operator=op_sm_sp_c,
                                multiplier=ladder_mult,
                                sites=[i, j_neighbor],
                                modifies=True,
                            )
                            self._log(
                                f"Adding spin-1 XY ladder terms between ({i}, {j_neighbor}) with multiplier {-coupling_xy:.3f}",
                                lvl=2,
                                log="debug",
                            )
                        else:
                            # Add Sx_i Sx_j term
                            self.add(
                                operator=op_sx_sx_c,
                                multiplier=-coupling_xy,
                                sites=[i, j_neighbor],
                                modifies=True,
                            )
                            self._log(
                                f"Adding SxSx between ({i}, {j_neighbor}) with multiplier {-coupling_xy:.3f}",
                                lvl=2,
                                log="debug",
                            )

                            # Add Sy_i Sy_j term
                            self.add(
                                operator=op_sy_sy_c,
                                multiplier=-coupling_xy,
                                sites=[i, j_neighbor],
                                modifies=True,
                            )
                            self._log(
                                f"Adding SySy between ({i}, {j_neighbor}) with multiplier {-coupling_xy:.3f}",
                                lvl=2,
                                log="debug",
                            )

                # Ising coupling: Jz * Sz_i Sz_j
                if self._jz is not None:
                    coupling_z = self._jz[i] * phase

                    if not np.isclose(coupling_z, 0.0):
                        self.add(
                            operator=op_sz_sz_c,
                            multiplier=-coupling_z,
                            sites=[i, j_neighbor],
                            modifies=False,
                        )
                        self._log(
                            f"Adding SzSz between ({i}, {j_neighbor}) with multiplier {-coupling_z:.3f}",
                            lvl=2,
                            log="debug",
                        )

    # ----------------------------------------------------------------------------------------------
    #! Properties
    # ----------------------------------------------------------------------------------------------

    @property
    def Jxy(self) -> Union[List[float], float]:
        """XY coupling strength(s) J_xy."""
        if (
            self._jxy
            and isinstance(self._jxy, list)
            and all(np.isclose(x, self._jxy[0]) for x in self._jxy)
        ):
            return self._jxy[0]
        return self._jxy

    @property
    def Jz(self) -> Union[List[float], float]:
        """Ising (Z) coupling strength(s) J_z."""
        if (
            self._jz
            and isinstance(self._jz, list)
            and all(np.isclose(x, self._jz[0]) for x in self._jz)
        ):
            return self._jz[0]
        return self._jz

    @property
    def Delta(self) -> Union[List[float], float]:
        """Anisotropy parameter(s) Δ = J_z / J_xy."""
        if (
            self._delta
            and isinstance(self._delta, list)
            and all(np.isclose(x, self._delta[0]) for x in self._delta)
        ):
            return self._delta[0]
        return self._delta

    @property
    def hx(self) -> Union[List[float], float]:
        """Transverse field strength(s) h_x."""
        if (
            self._hx
            and isinstance(self._hx, list)
            and all(np.isclose(x, self._hx[0]) for x in self._hx)
        ):
            return self._hx[0]
        return self._hx

    @property
    def hz(self) -> Union[List[float], float]:
        """Longitudinal field strength(s) h_z."""
        if (
            self._hz
            and isinstance(self._hz, list)
            and all(np.isclose(x, self._hz[0]) for x in self._hz)
        ):
            return self._hz[0]
        return self._hz


# --------------------------------------------------------------------------------------------------
#! END OF FILE
# --------------------------------------------------------------------------------------------------
