r"""
High-level Hamiltonian class for the Quantum Eigen Solver (QES) package.
The Heisenberg-Kitaev model implementation with additional possible
- external magnetic fields h_x, h_z - magnetic field terms in x and z directions,
- Heisenberg coupling J             - isotropic spin interaction,
- Kitaev interactions K_x, K_y, K_z - directional couplings depending on bond orientation,
- anisotropy parameter \Delta       - \Delta (modifies the Heisenberg coupling),
- \Gamma^\gamma interactions        - anisotropic off-diagonal couplings.

------------------------------------------------------------------------------
File        : Algebra/Model/Interacting/Spin/heisenberg_kitaev.py
Author      : Maksymilian Kliczkowski
Date        : 2025-02-17
Version     : 1.0
------------------------------------------------------------------------------
Changelog   :
- 2025-02-17 1.0: Initial version.
- 2025-06-10 1.1: Added impurities support and improved documentation.
- 2025-11-01 1.2: Refactored code for performance improvements.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

# QES package imports
try:
    import QES.Algebra.Operator.impl.operators_spin as operators_spin_module
    from QES.Algebra.hamil import Hamiltonian
    from QES.Algebra.hilbert import HilbertSpace
    from QES.general_python.lattices.honeycomb import (
        X_BOND_NEI,
        Y_BOND_NEI,
        Z_BOND_NEI,
        HoneycombLattice,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import QES modules. Ensure that the QES package is correctly installed."
    ) from e

##########################################################################################
#! IMPORTS
##########################################################################################

try:
    from QES.general_python.lattices.lattice import Lattice, LatticeType
except ImportError as e:
    raise ImportError(
        "Failed to import QES modules. Ensure that the QES package is correctly installed."
    ) from e

# ----------------------------------------------------------------------------------------
#! DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------

HEI_KIT_X_BOND_NEI = X_BOND_NEI
HEI_KIT_Y_BOND_NEI = Y_BOND_NEI
HEI_KIT_Z_BOND_NEI = Z_BOND_NEI

# Pauli matrix scaling factors for HeisenbergKitaev (Pauli σ convention)
# SINGLE_TERM_MULT    = 1.0   # Single-spin terms (σ has eigenvalues ±1)
SINGLE_TERM_MULT = 2.0  # Single-spin terms (σ has eigenvalues ±1)
# CORR_TERM_MULT      = 1.0   # Two-spin correlation terms (σ_i σ_j has eigenvalues ±1)
CORR_TERM_MULT = 4.0  # Two-spin correlation terms (σ_i σ_j has eigenvalues ±1)

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################


class HeisenbergKitaev(Hamiltonian):
    """
    Hamiltonian for an ergodic quantum dot coupled to an external system.
    The external system is modeled as a quantum spin chain.
    """

    #############################

    _ERR_EITHER_HIL_OR_NS = (
        "QSM: either the Hilbert space or the number of particles in the system must be provided."
    )
    _ERR_LATTICE_NOT_PROVIDED = (
        "QSM: Lattice must be provided to define the Kitaev-Heisenberg Hamiltonian."
    )

    def __init__(
        self,
        lattice         : Lattice,
        K               : Union[List[float], None, float] = 1.0,
        *,
        hilbert_space   : Optional[HilbertSpace] = None,
        # Heisenberg couplings
        J               : Union[List[float], None, float] = None,
        dlt             : Union[List[float], None, float] = 1.0,
        # Gamma interactions
        Gamma           : Union[List[float], None, float] = None,
        # Magnetic fields
        hx              : Union[List[float], None, float] = None,
        hy              : Union[List[float], None, float] = None,
        hz              : Union[List[float], None, float] = None,
        # Classical impurities
        # Format: List of tuples, either:
        #   - (site, amplitude)                     -> z-polarized impurity: amplitude * sig^z_i
        #   - (site, phi, theta, amplitude)         -> arbitrary direction using spherical coordinates:
        #       sin(theta)*cos(phi)*ampl * sig^x_i + sin(theta)*sin(phi)*ampl * sig^y_i + cos(theta)*ampl * sig^z_i
        #     where theta is polar angle from z-axis, phi is azimuthal angle in xy-plane
        impurities      : List[Tuple] = [],
        # other parameters
        dtype           : type = np.float64,
        backend         : str = "default",
        use_forward     : bool = True,
        **kwargs,
    ):
        r"""
        Constructor for the Heisenberg-Kitaev-Gamma Hamiltonian with optional magnetic fields
        and classical impurities. Works on general lattices but is primarily designed for honeycomb/hexagonal lattices.

        ---
        Parameters:
            lattice : Lattice
                The lattice on which the Hamiltonian is defined.
            K : Union[List[float], None, float], optional
                Kitaev interaction strength(s). Default is 1.0.
                Can be a single float (uniform) or a list (bond-dependent).
            logger : Optional[Logger], optional
                Logger for debugging and information purposes. Default is None.
            hilbert_space : Optional[HilbertSpace], optional
                Predefined Hilbert space. If None, it will be created based on `ns`. Default is None.
            J : Union[List[float], None, float], optional
                Heisenberg coupling strength(s). Default is 1.0.
                Can be a single float (uniform) or a list (site-dependent).
            Gamma : Union[List[float], None, float], optional
                Off-diagonal anisotropic interaction strength(s). Default is 0.0.
                Can be a single float (uniform) or a list (site-dependent).
            hx : Union[List[float], None, float], optional
                External magnetic field in the x direction. Default is 0.0.
                Can be a single float (uniform) or a list (site-dependent).
            hy : Union[List[float], None, float], optional
                External magnetic field in the y direction. Default is 0.0.
                Can be a single float (uniform) or a list (site-dependent).
            hz : Union[List[float], None, float], optional
                External magnetic field in the z direction. Default is 0.0.
                Can be a single float (uniform) or a list (site-dependent).
            dlt : Union[List[float], None, float], optional
                Anisotropy parameter(s). Default is 1.0.
                Can be a single float (uniform) or a list (site-dependent).
            impurities : List[Tuple], optional
                List of classical impurities. Supports two formats:

                1. Z-polarized (2-tuple): (site_index, amplitude)
                    Adds: amplitude * sig^z_i
                    Example: [(0, 0.5), (3, -0.3)] adds z-impurities at sites 0 and 3.

                2. Arbitrary direction (4-tuple): (site_index, phi, theta, amplitude)
                    Uses spherical coordinates where:
                    - theta: polar angle from z-axis (0 = +z, pi = -z)
                    - phi: azimuthal angle in xy-plane (0 = +x, pi/2 = +y)
                    Adds: sin(theta)*cos(phi)*ampl * sig^x_i
                        + sin(theta)*sin(phi)*ampl * sig^y_i
                        + cos(theta)*ampl * sig^z_i
                    Example:    [(0, 0, np.pi/2, 1.0)]          adds x-polarized impurity at site 0.
                                [(0, np.pi/2, np.pi/2, 1.0)]    adds y-polarized impurity at site 0.
                                [(0, 0, 0, 1.0)]                adds z-polarized impurity at site 0.
            dtype : type, optional
                Data type for numerical computations. Default is np.float64.
            backend : str, optional
                Computational backend to use. Default is "default".
            use_forward : bool, optional
                Whether to use forward nearest neighbors only. Default is True.
            **kwargs : additional keyword arguments
                Additional arguments passed to the Hamiltonian base class.
        """

        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)

        if lattice.typek not in [LatticeType.HONEYCOMB, LatticeType.HEXAGONAL]:
            self._log(
                f"The type of the lattice {lattice} is not standard. Check your intentions...",
                lvl=2,
            )

        # Initialize the Hamiltonian
        self._lattice = lattice
        super().__init__(
            is_manybody=True,
            hilbert_space=hilbert_space,
            lattice=lattice,
            is_sparse=True,
            dtype=(
                dtype if (hy is None and Gamma is None) else np.complex128
            ),  # enforce complex dtype if hy field is used
            backend=backend,
            use_forward=use_forward,
            **kwargs,
        )

        # setup the fields
        self._hx = (
            hx
            if isinstance(hx, (list, np.ndarray, tuple))
            else [hx] * self.ns if hx is not None else None
        )
        self._hy = (
            hy
            if isinstance(hy, (list, np.ndarray, tuple))
            else [hy] * self.ns if hy is not None else None
        )
        self._hz = (
            hz
            if isinstance(hz, (list, np.ndarray, tuple))
            else [hz] * self.ns if hz is not None else None
        )
        # setup the couplings
        self._j = (
            J
            if isinstance(J, (list, np.ndarray, tuple))
            else [J] * self.ns if J is not None else None
        )
        self._gx, self._gy, self._gz = (
            Gamma
            if isinstance(Gamma, (list, np.ndarray, tuple))
            else (Gamma, Gamma, Gamma) if Gamma is not None else (None, None, None)
        )
        self._dlt = (
            dlt
            if isinstance(dlt, (list, np.ndarray, tuple))
            else [dlt] * self.ns if dlt is not None else None
        )
        self._kx, self._ky, self._kz = (
            K if isinstance(K, (list, np.ndarray, tuple)) and len(K) == 3 else (K, K, K)
        )

        # setup the impurities - validate format: either 2-tuple (site, ampl) or (3,4)-tuple (site, phi, theta, ampl)
        self._impurities = []
        self._setup_impurities(impurities)

        self._log(
            f"Initializing Heisenberg-Kitaev Hamiltonian on lattice: {lattice}",
            lvl=1,
            log="info",
            color="green",
            verbose=self._verbose,
        )
        self._log(
            f"Impurities provided: {self._impurities}",
            lvl=2,
            log="info",
            color="blue",
            verbose=self._verbose,
        )

        self._neibz = [[]]
        self._neiby = [[]]
        self._neibx = [[]]
        self._neiadd = [[]]

        # Initialize the Hamiltonian
        self._name = "Kitaev-Heisenberg-Gamma Model"
        self._is_sparse = True

        #! Calculate maximum local coupling channels
        # 3 from Heisenberg + 3 from Kitaev + 6 from Gamma + 3 from fields + 3*impurities (x,y,z components each)
        self._max_local_ch = (
            3
            + (3 if self._j is not None else 0)
            + (2 if self._gx is not None else 0)
            + (2 if self._gy is not None else 0)
            + (2 if self._gz is not None else 0)
            + (1 if self._hz is not None else 0)
            + (1 if self._hy is not None else 0)
            + (1 if self._hx is not None else 0)
            + 3 * len(self._impurities)
        )  # Each impurity can have x, y, z components
        self.set_couplings()

        # functions for local energy calculation in a jitted way (numpy and jax)
        self._set_local_energy_operators()
        self.setup_instruction_codes()  # automatic physics-based setup
        self._set_local_energy_functions()

    # ----------------------------------------------------------------------------------------------

    def _setup_impurities(self, impurities: List[Tuple]):
        """
        Sets up the impurities list in the required format.
        Supports two formats:
            1. Z-polarized (2-tuple): (site_index, amplitude)
            2. Full spherical coordinates (4-tuple): (site_index, phi, theta, amplitude)
        """
        if impurities is None:
            self._impurities = []
        else:
            self._impurities = []
            for imp in impurities:
                if isinstance(imp, tuple):
                    if len(imp) == 2:
                        # Z-polarized: (site, amplitude) -> convert to (site, 0, 0, amplitude) for z-direction
                        site, ampl = imp
                        self._impurities.append((site, 0.0, 0.0, ampl))  # theta=0 means z-polarized
                    elif len(imp) >= 3:
                        # Full spherical: (site, phi, theta, amplitude)
                        if len(imp) == 4:
                            self._impurities.append(imp)
                        else:
                            self._impurities.append(
                                (imp[0], imp[1], imp[2], 1.0)
                            )  # default amplitude=1.0
                    else:
                        raise ValueError(
                            f"Impurity tuple must have 2 or 4 elements, got {len(imp)}: {imp}"
                        )
                else:
                    raise ValueError(f"Each impurity must be a tuple, got {type(imp)}: {imp}")

    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Human-readable, single-line description of the HeiKit Hamiltonian.

        Examples
        --------
        - uniform couplings
        HeiKit(Ns=16, J=1.000, Kx=0.200, Ky=0.200, Kz=0.000, dlt=1.000,
                hz=0.300, hx=0.000, sym=U1 PBC)

        - site-dependent fields
        HeiKit(Ns=16, J=1.000, Kx=0.200, Ky=0.200, Kz=0.000, dlt=1.000,
                hz[min=-0.500, max=0.300], hx=0.000, sym=U1 CBC)
        """
        prec = 3  # decimal places
        sep = ","  # parameter separator

        # string
        parts = [f"Kitaev(Ns={self.ns}"]

        parts += [
            Hamiltonian.fmt("Kx",   self._kx,   prec=prec),
            Hamiltonian.fmt("Ky",   self._ky,   prec=prec),
            Hamiltonian.fmt("Kz",   self._kz,   prec=prec),
            Hamiltonian.fmt("J",    self._j,    prec=prec)  if self._j      is not None else "",
            Hamiltonian.fmt("Gx",   self._gx,   prec=prec)  if self._gx     is not None else "",
            Hamiltonian.fmt("Gy",   self._gy,   prec=prec)  if self._gy     is not None else "",
            Hamiltonian.fmt("Gz",   self._gz,   prec=prec)  if self._gz     is not None else "",
            Hamiltonian.fmt("dlt",  self._dlt,  prec=prec)  if self._dlt    is not None else "",
            Hamiltonian.fmt("hz",   self._hz,   prec=prec)  if self._hz     is not None else "",
            Hamiltonian.fmt("hy",   self._hy,   prec=prec)  if self._hy     is not None else "",
            Hamiltonian.fmt("hx",   self._hx,   prec=prec)  if self._hx     is not None else "",
        ]
        # handle impurities
        if len(self._impurities) > 0:
            imp_strs = []
            for site, phi, theta, ampl in self._impurities:
                phi_i = phi % (2 * np.pi)
                theta_i = theta % np.pi
                # Concise format for directory naming: s{site}_p{phi}_t{theta}_a{ampl}
                # using .2f precision to keep it short
                imp_strs.append(
                    f"s{site}_p{phi_i:.2f}_t{theta_i:.2f}_a{ampl:.2f}"
                )
            parts.append(f"Imps[{'-'.join(imp_strs)}]")

        parts = [p for p in parts if p]
        return sep.join(parts) + ")"

    def __str__(self):
        return self.__repr__()

    @property
    def bonds(self):
        return {
            HEI_KIT_X_BOND_NEI: "x",
            HEI_KIT_Y_BOND_NEI: "y",
            HEI_KIT_Z_BOND_NEI: "z",
            "x": HEI_KIT_X_BOND_NEI,
            "y": HEI_KIT_Y_BOND_NEI,
            "z": HEI_KIT_Z_BOND_NEI,
        }

    # ----------------------------------------------------------------------------------------------
    #! INIT
    # ----------------------------------------------------------------------------------------------

    def set_couplings(
        self,
        hx  : Union[List[float], None, float] = None,
        hy  : Union[List[float], None, float] = None,
        hz  : Union[List[float], None, float] = None,
        kx  : Union[List[float], None, float] = None,
        ky  : Union[List[float], None, float] = None,
        kz  : Union[List[float], None, float] = None,
        j   : Union[List[float], None, float] = None,
        gx  : Union[List[float], None, float] = None,
        gy  : Union[List[float], None, float] = None,
        gz  : Union[List[float], None, float] = None,
        dlt : Union[List[float], None, float] = None,
    ):
        """
        Sets the couplings based on their initial value (list, string, value)
        """
        self._hx = (
            self._set_some_coupling(self._hx if hx is None else hx) if hx is not None else self._hx
        )
        self._hy = (
            self._set_some_coupling(self._hy if hy is None else hy) if hy is not None else self._hy
        )
        self._hz = (
            self._set_some_coupling(self._hz if hz is None else hz) if hz is not None else self._hz
        )
        self._kx = (
            self._set_some_coupling(self._kx if kx is None else kx) if kx is not None else self._kx
        )
        self._ky = (
            self._set_some_coupling(self._ky if ky is None else ky) if ky is not None else self._ky
        )
        self._kz = (
            self._set_some_coupling(self._kz if kz is None else kz) if kz is not None else self._kz
        )
        self._j = self._set_some_coupling(self._j if j is None else j) if j is not None else self._j
        self._dlt = (
            self._set_some_coupling(self._dlt if dlt is None else dlt)
            if dlt is not None
            else self._dlt
        )
        self._gx = (
            self._set_some_coupling(self._gx if gx is None else gx) if gx is not None else self._gx
        )
        self._gy = (
            self._set_some_coupling(self._gy if gy is None else gy) if gy is not None else self._gy
        )
        self._gz = (
            self._set_some_coupling(self._gz if gz is None else gz) if gz is not None else self._gz
        )

    def _set_local_energy_operators(self):
        r"""
        Set up the local and non-local energy operators for the spin lattice.
        This method constructs the operator lists representing local (single-site)
        and correlation (two-site) interactions for the system. It iterates over
        each site and performs the following steps:

            - Initializes lists to store local (operators_local) and non-local
                (operators) operator tuples.
            - For each site:
                - Creates local operators (sig_x and sig_z) acting on the site.
                - Creates correlation operators for sig_x, sig_y, and sig_z.
                - Appends local operators to the local operators list with their associated
                    field strengths (from self._hx and self._hz).
            - For each site, iterates over forward nearest neighbors as provided by the
                lattice object:
                - Retrieves the neighbor indices using lattice.get_nn_forward.
                - Computes the interaction multipliers based on Heisenberg coupling terms
                    (self._j and self._dlt) and Kitaev interaction contributions (self._kx,
                    self._ky, self._kz), with adjustments made according to the bond directions
                    (e.g., HEI_KIT_Z_BOND_NEI, HEI_KIT_Y_BOND_NEI).
                - Appends the corresponding correlation operator tuples to the operator lists.
            - Logs detailed debug messages at various levels throughout the process.

        The resulting operator tuples are stored as:
            - self._local_ops: a list containing tuples of (operator, [site index], coefficient)
                for local energy contributions.
            - self._nonlocal_ops: a list containing tuples of (operator, [site index, neighbor index],
                coefficient) for two-site interactions.

        Model: Heisenberg-Kitaev-Gamma with external fields and impurities.

        \\[
            H_hei = \\sum_{\\langle i,j \rangle} J (S_i^x S_j^x + S_i^y S_j^y + \\Delta S_i^z S_j^z)                        # Heisenberg term with positive sign convention (antiferromagnetic)
            H_kit = -\\sum_{\\langle i,j \rangle_\\gamma} K_\\gamma S_i^\\gamma S_j^\\gamma                                 # Kitaev term with negative sign convention (ferromagnetic)
            H_gam = \\sum_{\\langle i,j \rangle_\\gamma} \\Gamma^\\gamma (S_i^\alpha S_j^\beta + S_i^\beta S_j^\alpha)      # positive sign convention (antiferromagnetic)
            H_mag = -\\sum_{i} (h_x S_i^x + h_y S_i^y + h_z S_i^z)                                                          # Magnetic field terms with negative sign convention
            H_imp = -\\sum_{i} \\vec{h}_{imp,i} \\cdot \\vec{S}_i                                                           # Impurity terms with negative sign convention
            H_all = H_hei + H_kit + H_gam + H_mag + H_imp
        \\]

        Note:
            This method updates internal state and does not return a value.
        """
        super()._set_local_energy_operators()

        # Use the lattice from the Hamiltonian
        lattice = self._lattice
        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)

        #! define the operators beforehand - to avoid multiple creations
        op_sx_l     = operators_spin_module.sig_x(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local
        )
        op_sy_l     = operators_spin_module.sig_y(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local
        )
        op_sz_l     = operators_spin_module.sig_z(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local
        )

        # Kitaev and Heisenberg terms - correlation operators (two-site)
        op_sx_sx_c  = operators_spin_module.sig_x(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Correlation
        )
        op_sy_sy_c  = operators_spin_module.sig_y(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Correlation
        )
        op_sz_sz_c  = operators_spin_module.sig_z(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Correlation
        )

        # Create Gamma operators as products of correlation operators
        op_sx_sy_c  = operators_spin_module.sig_xy(lattice=lattice, spin_value=0.5)
        op_sy_sx_c  = operators_spin_module.sig_yx(lattice=lattice, spin_value=0.5)

        op_sz_sx_c  = operators_spin_module.sig_zx(lattice=lattice, spin_value=0.5)
        op_sx_sz_c  = operators_spin_module.sig_xz(lattice=lattice, spin_value=0.5)

        op_sy_sz_c  = operators_spin_module.sig_yz(lattice=lattice, spin_value=0.5)
        op_sz_sy_c  = operators_spin_module.sig_zy(lattice=lattice, spin_value=0.5)
        nn_nums     = (
            [lattice.get_nn_forward_num(i) for i in range(self.ns)]
            if self._use_forward
            else [lattice.get_nn_num(i) for i in range(self.ns)]
        )

        #! iterate over all the sites
        elems       = 0
        # log             =   'info'
        log         = "debug"
        for i in range(self.ns):
            self._log(f"Starting i: {i}", lvl=1, log=log)

            # ? z-field (single-spin term: applying SINGLE_TERM_MULT scaling for Pauli matrices)
            if Hamiltonian._ADD_CONDITION(self._hz, i):
                z_field = -SINGLE_TERM_MULT * self._hz[i]
                self.add(op_sz_l, multiplier=z_field, modifies=False, sites=[i])
                self._log(f"Adding local Sz at {i} with value {z_field:.2f}", lvl=2, log=log)

            # ? y-field (single-spin term: applying SINGLE_TERM_MULT scaling for Pauli matrices)
            if Hamiltonian._ADD_CONDITION(self._hy, i):
                y_field = -SINGLE_TERM_MULT * self._hy[i]
                self.add(op_sy_l, multiplier=y_field, modifies=True, sites=[i])
                self._log(f"Adding local Sy at {i} with value {y_field:.2f}", lvl=2, log=log)

            # ? x-field (single-spin term: applying SINGLE_TERM_MULT scaling for Pauli matrices)
            if Hamiltonian._ADD_CONDITION(self._hx, i):
                x_field = -SINGLE_TERM_MULT * self._hx[i]
                self.add(op_sx_l, multiplier=x_field, modifies=True, sites=[i])
                self._log(f"Adding local Sx at {i} with value {x_field:.2f}", lvl=2, log=log)

            # ? impurities - now supports arbitrary spin directions via spherical coordinates

            # Format: (site, phi, theta, amplitude) where all are stored in this 4-tuple format
            # Components: sin(theta)*cos(phi)*ampl * sig^x + sin(theta)*sin(phi)*ampl * sig^y + cos(theta)*ampl * sig^z
            for imp_site, phi, theta, ampl in self._impurities:
                if imp_site == i and Hamiltonian._ADD_CONDITION(ampl):
                    # Compute spherical coordinate components
                    sin_theta   = np.sin(theta)
                    cos_theta   = np.cos(theta)
                    sin_phi     = np.sin(phi)
                    cos_phi     = np.cos(phi)

                    # Z-component: cos(theta) * amplitude
                    imp_z       = -SINGLE_TERM_MULT * ampl * cos_theta
                    if Hamiltonian._ADD_CONDITION(imp_z):
                        self.add(op_sz_l, multiplier=imp_z, modifies=False, sites=[i])
                        self._log(
                            f"Adding impurity Sz at {i} with value {imp_z:.4f}", lvl=2, log=log
                        )

                    # X-component: sin(theta) * cos(phi) * amplitude
                    imp_x       = -SINGLE_TERM_MULT * ampl * sin_theta * cos_phi
                    if Hamiltonian._ADD_CONDITION(imp_x):
                        self.add(op_sx_l, multiplier=imp_x, modifies=True, sites=[i])
                        self._log(
                            f"Adding impurity Sx at {i} with value {imp_x:.4f}", lvl=2, log=log
                        )

                    # Y-component: sin(theta) * sin(phi) * amplitude
                    imp_y       = -SINGLE_TERM_MULT * ampl * sin_theta * sin_phi
                    if Hamiltonian._ADD_CONDITION(imp_y):
                        self.add(op_sy_l, multiplier=imp_y, modifies=True, sites=[i])
                        self._log(
                            f"Adding impurity Sy at {i} with value {imp_y:.4f}", lvl=2, log=log
                        )

            # ? now check the correlation operators
            nn_num = nn_nums[i]

            for nn in range(nn_num):
                # get the neighbor index
                nei = (
                    lattice.get_nn_forward(i, num=nn)
                    if self._use_forward
                    else lattice.get_nn(i, num=nn)
                )

                #! check the direction of the bond
                if lattice.wrong_nei(nei):
                    continue

                wx, wy, wz  = lattice.bond_winding(i, nei)
                phase       = lattice.boundary_phase_from_winding(wx, wy, wz)

                #! Heisenberg - value of SzSz, SxSx, SySy (multipliers)
                if True:
                    sz_sz   = phase * CORR_TERM_MULT * self._j[i] * self._dlt[i] if self._j is not None else 0.0    # Heisenberg - value of SzSz (multiplier)
                    sx_sx   = phase * CORR_TERM_MULT * self._j[i] if self._j is not None else 0.0                   # Heisenberg - value of SxSx (multiplier)
                    sy_sy   = phase * CORR_TERM_MULT * self._j[i] if self._j is not None else 0.0                   # Heisenberg - value of SySy (multiplier)

                #! check the directional bond contributions - Kitaev (they are subtracted - ferromagnetic)
                if True:
                    if nn == HEI_KIT_Z_BOND_NEI:
                        sz_sz -= phase * CORR_TERM_MULT * self._kz if self._kz is not None else 0.0
                    elif nn == HEI_KIT_Y_BOND_NEI:
                        sy_sy -= phase * CORR_TERM_MULT * self._ky if self._ky is not None else 0.0
                    elif nn == HEI_KIT_X_BOND_NEI:
                        sx_sx -= phase * CORR_TERM_MULT * self._kx if self._kx is not None else 0.0

                #! Now add the Heisenberg-Kitaev terms
                if True:
                    if Hamiltonian._ADD_CONDITION(sz_sz):
                        self.add(op_sz_sz_c, sites=[i, nei], multiplier=sz_sz, modifies=False)
                        self._log(f"Adding SzSz at {i},{nei} with value {sz_sz:.2f}", lvl=2, log=log)
                        elems += 1
                    if Hamiltonian._ADD_CONDITION(sx_sx):
                        self.add(op_sx_sx_c, sites=[i, nei], multiplier=sx_sx, modifies=True)
                        self._log(f"Adding SxSx at {i},{nei} with value {sx_sx:.2f}", lvl=2, log=log)
                        elems += 1
                    if Hamiltonian._ADD_CONDITION(sy_sy):
                        self.add(op_sy_sy_c, sites=[i, nei], multiplier=sy_sy, modifies=True)
                        self._log(f"Adding SySy at {i},{nei} with value {sy_sy:.2f}", lvl=2, log=log)
                        elems += 1

                #! Gamma terms - these are off-diagonal couplings, they are added - antiferromagnetic
                # NOTE: The definition is H_gam = - sum Gamma * (S_a S_b + S_b S_a)
                # Therefore we apply -Gamma as multiplier
                if True:
                    # ? Gamma_x terms
                    if Hamiltonian._ADD_CONDITION(self._gx) and nn == HEI_KIT_X_BOND_NEI:
                        val     = self._gx * phase * CORR_TERM_MULT
                        elems  += 2
                        self.add(op_sy_sz_c, sites=[i, nei], multiplier=val, modifies=True)
                        self.add(op_sz_sy_c, sites=[i, nei], multiplier=val, modifies=True)
                        self._log(f"Adding Gamma_x(SySz+SzSy) at {i},{nei} with value {val:.2f}", lvl=2, log=log)

                    # #? Gamma_y terms
                    if Hamiltonian._ADD_CONDITION(self._gy) and nn == HEI_KIT_Y_BOND_NEI:
                        val     = self._gy * phase * CORR_TERM_MULT
                        elems  += 2
                        self.add(op_sz_sx_c, sites=[i, nei], multiplier=val, modifies=True)
                        self.add(op_sx_sz_c, sites=[i, nei], multiplier=val, modifies=True)
                        self._log(f"Adding Gamma_y(SzSx + SxSz) at {i},{nei} with value {val:.2f}", lvl=2, log=log)

                    # #? Gamma_z terms
                    if Hamiltonian._ADD_CONDITION(self._gz) and nn == HEI_KIT_Z_BOND_NEI:
                        val     = self._gz * phase * CORR_TERM_MULT
                        elems  += 2
                        self.add(op_sx_sy_c, sites=[i, nei], multiplier=val, modifies=True)
                        self.add(op_sy_sx_c, sites=[i, nei], multiplier=val, modifies=True)
                        self._log(f"Adding Gamma_z(SxSy + SySx) at {i},{nei} with value {val:.2f}", lvl=2, log=log)

                #! Finalize the operator addition for this neighbor
                self._log(f"Finished processing neighbor {nei} of site {i}", lvl=2, log=log)

        self._log(f"Total NN elements added: {elems}", color="red", lvl=3, verbose=self._verbose)
        self._log("Successfully set local energy operators...", lvl=1, log="info", verbose=self._verbose)

##########################################################################################
#! EOF
##########################################################################################
