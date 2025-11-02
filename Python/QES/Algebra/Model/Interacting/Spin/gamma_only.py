r"""
Gamma-Only Interaction Model for Frustrated Quantum Systems.

This module implements the Gamma-only Hamiltonian, which represents a simplified
case of the Kitaev-Heisenberg-Gamma model where only off-diagonal anisotropic
couplings are present. This model is useful for studying quantum spin liquids
and frustrated magnetism on honeycomb and other lattices.

The Hamiltonian is defined as:
    H = Σ_{<ij>} [Γ_x (S_i^x S_j^y + S_i^y S_j^x)
                  + Γ_y (S_i^y S_j^z + S_i^z S_j^y)
                  + Γ_z (S_i^z S_j^x + S_i^x S_j^z)]
        + Σ_i [h_x S_i^x + h_z S_i^z]
        + Σ_{imp} strength_imp * S_z^imp

where:
    - Γ_α are the off-diagonal anisotropic coupling strengths
    - h_x, h_z are magnetic field terms
    - Classical impurities can be added at specified sites

--------------------
File    : Model/Interacting/Spin/gamma_only.py
Author  : Automated Session (Phase 2.3)
Date    : November 2025
Version : 1.0
Changes : 
    2025-11-01 (1.0) : Initial implementation for Gamma-only model.
--------------------

References:
    - Rousochatzakis & Perkins (2017): Physics of the Kitaev model
    - Luo et al. (2021): Gapless QSL in the Gamma-only model
    - Matsuda et al. (2025): Kitaev materials review (RMP)
"""

import numpy as np
from typing import List, Tuple, Union, Optional

# QES package imports
try:
    import QES.Algebra.hilbert as hilbert_module
    import QES.Algebra.hamil as hamil_module
    import QES.Algebra.Operator.operators_spin as operators_spin_module
except ImportError as e:
    raise ImportError(
        "Failed to import QES modules. Ensure that the QES package is correctly installed."
    ) from e

try:
    from QES.general_python.lattices.lattice import Lattice, LatticeType
except ImportError as e:
    raise ImportError(
        "Failed to import QES lattice module. Ensure QES package is installed correctly."
    ) from e

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################


class GammaOnly(hamil_module.Hamiltonian):
    r"""
    Hamiltonian for the Gamma-Only interaction model.
    
    This model implements purely off-diagonal anisotropic interactions on a lattice,
    making it useful for studying quantum spin liquids. It can optionally include
    isotropic magnetic fields and classical impurities.
    
    The model is particularly relevant for:
    - Kitaev materials research (Γ terms of the full model)
    - Quantum spin liquid phases
    - Frustrated magnetic systems
    - Benchmark calculations for Kitaev-Heisenberg systems (set K=J=0)
    
    Example:
        >>> lattice = Lattice(lattice_type=LatticeType.HONEYCOMB, ns=12)
        >>> gamma_only = GammaOnly(
        ...     lattice=lattice,
        ...     Gamma=0.1,  # uniform coupling
        ...     hx=0.05,
        ...     hz=0.05
        ... )
        >>> print(gamma_only)
        GammaOnly(Ns=12, Γx=0.100, Γy=0.100, Γz=0.100, hx=0.050, hz=0.050)
    """

    _ERR_LATTICE_NOT_PROVIDED = (
        "GammaOnly: Lattice must be provided to define the Gamma-Only Hamiltonian."
    )

    def __init__(
        self,
        lattice: Lattice,
        Gamma: Union[List[float], Tuple[float, float, float], float, None] = 0.1,
        logger: Optional["Logger"] = None,
        *,
        hilbert_space: Optional[hilbert_module.HilbertSpace] = None,
        # Magnetic fields
        hx: Union[List[float], None, float] = None,
        hz: Union[List[float], None, float] = None,
        # Classical impurities (site_index, coupling_strength)
        impurities: List[Tuple[int, float]] = [],
        # Other parameters
        dtype: type = np.float64,
        backend: str = "default",
        use_forward: bool = True,
        **kwargs
    ):
        r"""
        Constructor for the Gamma-Only Hamiltonian.

        Parameters
        ----------
        lattice : Lattice
            The lattice on which the Hamiltonian is defined.
            Typically honeycomb or triangular for Kitaev physics.

        Gamma : Union[List[float], Tuple[float, float, float], float, None], optional
            Off-diagonal anisotropic coupling strength(s). Can be:
            - float: uniform coupling, applies to all three components (Γ_x = Γ_y = Γ_z)
            - [γ_x, γ_y, γ_z]: specify each component separately
            - None or 0: no Gamma interactions (trivial model)
            Default: 0.1

        logger : Optional[Logger], optional
            Logger for debugging and information. Default: None

        hilbert_space : Optional[HilbertSpace], optional
            Predefined Hilbert space. If None, created based on lattice size.
            Default: None

        hx : Union[List[float], None, float], optional
            Magnetic field in x direction. Can be:
            - float: uniform field
            - List: site-dependent fields
            - None: no x-field
            Default: None

        hz : Union[List[float], None, float], optional
            Magnetic field in z direction. Can be:
            - float: uniform field
            - List: site-dependent fields
            - None: no z-field
            Default: None

        impurities : List[Tuple[int, float]], optional
            Classical impurities as list of (site_index, coupling_strength) tuples.
            Each impurity contributes s_z^imp * coupling_strength to the Hamiltonian
            at the specified site.
            Example: [(0, 0.5), (3, -0.3)]
            Default: []

        dtype : type, optional
            Data type for numerical computations. Default: np.float64

        backend : str, optional
            Computational backend. Default: "default"

        use_forward : bool, optional
            Whether to use forward neighbors only (avoids double counting).
            Default: True

        **kwargs : additional keyword arguments
            Additional arguments passed to Hamiltonian base class.

        Raises
        ------
        ValueError
            If lattice is not provided.
        """

        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)

        # Validate lattice type for Kitaev physics
        if lattice.typek not in [LatticeType.HONEYCOMB, LatticeType.HEXAGONAL]:
            if logger is not None:
                logger.log(
                    f"Note: Lattice type {lattice.typek} is not standard for Gamma-only models. "
                    f"Honeycomb/hexagonal lattices are typical.",
                    lvl=2
                )

        # Initialize base Hamiltonian
        self._lattice = lattice
        super().__init__(
            is_manybody=True,
            hilbert_space=hilbert_space,
            lattice=lattice,
            is_sparse=True,
            dtype=dtype,
            backend=backend,
            use_forward=use_forward,
            logger=logger,
            **kwargs
        )

        # Create Hilbert space if not provided
        if hilbert_space is None:
            self._hilbert_space = hilbert_module.HilbertSpace(
                ns=self.ns, backend=backend, dtype=dtype, nhl=2
            )

        # Setup magnetic fields
        self._hx = self._set_some_coupling(hx) if hx is not None else None
        self._hz = self._set_some_coupling(hz) if hz is not None else None

        # Setup Gamma interactions - parse input flexibly
        self._setup_gamma_interactions(Gamma)

        # Setup impurities
        if self._logger is not None:
            self._log(f"Setting up impurities: {impurities}", lvl=2, log="info", color="green")
        self._impurities = (
            impurities
            if isinstance(impurities, list)
            and all(isinstance(i, tuple) and len(i) == 2 for i in impurities)
            else []
        )

        # Initialize Hamiltonian metadata
        self._name = "Gamma-Only Model"
        self._is_sparse = True

        # Calculate maximum local coupling channels
        # 2 Gamma components (x, y, z each contributes 2 terms) + fields + impurities
        self._max_local_ch = (
            (2 if self._gx is not None else 0)
            + (2 if self._gy is not None else 0)
            + (2 if self._gz is not None else 0)
            + (1 if self._hz is not None else 0)
            + (1 if self._hx is not None else 0)
            + len(self._impurities)
        )

        # Build the Hamiltonian
        self.set_couplings()
        self._set_local_energy_operators()
        self._set_local_energy_functions()

    # --------------------------------------------------------------------------------------
    #! SETUP AND INITIALIZATION
    # --------------------------------------------------------------------------------------

    def _setup_gamma_interactions(
        self, Gamma: Union[List[float], Tuple[float, float, float], float, None]
    ) -> None:
        """
        Parse and setup Gamma interaction parameters.

        Parameters
        ----------
        Gamma : Union[List[float], Tuple[float, float, float], float, None]
            Gamma coupling specification. Can be:
            - float: uniform coupling applied to all components
            - [γ_x, γ_y, γ_z]: separate components
            - None or 0: no interactions
        """
        if Gamma is None or (isinstance(Gamma, (int, float)) and np.isclose(Gamma, 0.0)):
            # No Gamma interactions
            self._gx = None
            self._gy = None
            self._gz = None
        elif isinstance(Gamma, (list, tuple, np.ndarray)):
            # Parse list/tuple - expect either length 3 or single value repeated
            gamma_array = np.asarray(Gamma)
            if len(gamma_array) == 3:
                # Different coupling for each component
                self._gx = gamma_array[0] if not np.isclose(gamma_array[0], 0.0) else None
                self._gy = gamma_array[1] if not np.isclose(gamma_array[1], 0.0) else None
                self._gz = gamma_array[2] if not np.isclose(gamma_array[2], 0.0) else None
            elif len(gamma_array) == 1:
                # Single value - apply to all components
                g_val = gamma_array[0]
                self._gx = g_val if not np.isclose(g_val, 0.0) else None
                self._gy = g_val if not np.isclose(g_val, 0.0) else None
                self._gz = g_val if not np.isclose(g_val, 0.0) else None
            else:
                raise ValueError(
                    f"Gamma must be a scalar or 3-element sequence, got length {len(gamma_array)}"
                )
        elif isinstance(Gamma, (int, float)):
            # Uniform scalar coupling
            if np.isclose(Gamma, 0.0):
                self._gx = None
                self._gy = None
                self._gz = None
            else:
                self._gx = Gamma
                self._gy = Gamma
                self._gz = Gamma
        else:
            raise TypeError(f"Gamma must be float, tuple, or list, got {type(Gamma)}")

    def __repr__(self) -> str:
        """
        Concise, human-readable description of the GammaOnly instance.

        Examples
        --------
        >>> print(gamma_model)
        GammaOnly(Ns=12, Γx=0.100, Γy=0.100, Γz=0.100, hx=0.050, hz=0.050)

        >>> print(gamma_model)  # site-dependent fields
        GammaOnly(Ns=12, Γx=0.100, Γy=0.100, Γz=0.100,
                  hx=[min=-0.200, max=0.300], hz=0.050)
        """
        prec = 3
        sep = ", "
        parts = [f"GammaOnly(Ns={self.ns}"]

        # Add Gamma components
        if self._gx is not None:
            parts.append(f"Γx={self._gx:.{prec}f}")
        if self._gy is not None:
            parts.append(f"Γy={self._gy:.{prec}f}")
        if self._gz is not None:
            parts.append(f"Γz={self._gz:.{prec}f}")

        # Add fields
        parts += [
            hamil_module.Hamiltonian.fmt("hx", self._hx, prec=prec),
            hamil_module.Hamiltonian.fmt("hz", self._hz, prec=prec),
        ]

        # Add impurities if present
        if self._impurities:
            parts.append(f"impurities={len(self._impurities)}")

        return sep.join(parts) + ")"

    def __str__(self) -> str:
        return self.__repr__()

    # --------------------------------------------------------------------------------------
    #! COUPLING SETUP
    # --------------------------------------------------------------------------------------

    def set_couplings(self) -> None:
        """
        Prepare coupling parameters for Hamiltonian construction.

        This method is called during initialization and can be used to reset couplings
        if the model is rebuilt. Currently a placeholder that ensures consistency.
        """
        # Gamma interactions are already set in _setup_gamma_interactions
        # This method can be extended for dynamic coupling updates if needed
        pass

    def _set_local_energy_operators(self) -> None:
        """
        Build the list of operators defining the Gamma-Only Hamiltonian.

        Adds terms for:
        - Gamma_x: Γ_x (S_i^x S_j^y + S_i^y S_j^x) for each nearest-neighbor pair
        - Gamma_y: Γ_y (S_i^y S_j^z + S_i^z S_j^y) for each nearest-neighbor pair
        - Gamma_z: Γ_z (S_i^z S_j^x + S_i^x S_j^z) for each nearest-neighbor pair
        - Magnetic fields: h_x S_i^x, h_z S_i^z for each site
        - Impurities: coupling_strength * S_i^z at specified sites
        """
        # Clear existing operators
        super()._set_local_energy_operators()

        if self._logger is not None:
            self._log("Building Gamma-Only operator list...", lvl=1, log="info")

        lattice = self._lattice

        # Define base operators
        op_sx_l = operators_spin_module.sig_x(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local
        )
        op_sy_l = operators_spin_module.sig_y(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local
        )
        op_sz_l = operators_spin_module.sig_z(
            lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local
        )

        # Define Gamma operator products (off-diagonal)
        # Γ_x: S_x S_y + S_y S_x
        op_sx_sy_c = op_sx_l * op_sy_l
        op_sy_sx_c = op_sy_l * op_sx_l

        # Γ_y: S_y S_z + S_z S_y
        op_sy_sz_c = op_sy_l * op_sz_l
        op_sz_sy_c = op_sz_l * op_sy_l

        # Γ_z: S_z S_x + S_x S_z
        op_sz_sx_c = op_sz_l * op_sx_l
        op_sx_sz_c = op_sx_l * op_sz_l

        # ====== Add Magnetic Field Terms ======
        for i in range(self.ns):
            # z-field
            if self._hz is not None and not np.isclose(self._hz[i], 0.0, rtol=1e-10):
                self.add(op_sz_l, multiplier=self._hz[i], modifies=False, sites=[i])
                if self._logger is not None:
                    self._log(
                        f"Adding local Sz at {i} with value {self._hz[i]:.2f}",
                        lvl=2,
                        log="debug",
                    )

            # x-field
            if self._hx is not None and not np.isclose(self._hx[i], 0.0, rtol=1e-10):
                self.add(op_sx_l, multiplier=self._hx[i], modifies=True, sites=[i])
                if self._logger is not None:
                    self._log(
                        f"Adding local Sx at {i} with value {self._hx[i]:.2f}",
                        lvl=2,
                        log="debug",
                    )

            # Impurities
            for (imp_site, imp_strength) in self._impurities:
                if imp_site == i:
                    self.add(op_sz_l, multiplier=imp_strength, modifies=False, sites=[i])
                    if self._logger is not None:
                        self._log(
                            f"Adding impurity Sz at {i} with value {imp_strength:.2f}",
                            lvl=2,
                            log="debug",
                        )

        # ====== Add Gamma Interaction Terms ======
        # Iterate over all sites and their nearest neighbors
        nn_nums = (
            [lattice.get_nn_num(i) for i in range(self.ns)]
            if not self._use_forward
            else [lattice.get_nn_forward_num(i) for i in range(self.ns)]
        )

        for i in range(self.ns):
            if self._logger is not None:
                self._log(f"Processing site i={i}", lvl=1, log="debug")

            nn_num = nn_nums[i]

            for nn_idx in range(nn_num):
                # Get neighbor
                j = (
                    lattice.get_nn_forward(i, num=nn_idx)
                    if self._use_forward
                    else lattice.get_nn(i, num=nn_idx)
                )

                if lattice.wrong_nei(j):
                    continue

                # Get bond winding and boundary phase (periodic boundary conditions)
                wx, wy, wz = lattice.bond_winding(i, j)
                phase = lattice.boundary_phase_from_winding(wx, wy, wz)

                # ====== Gamma_x terms: Γ_x (S_i^x S_j^y + S_i^y S_j^x) ======
                if self._gx is not None and not np.isclose(self._gx, 0.0, rtol=1e-10):
                    coupling_gx = self._gx * phase

                    # S_x S_y term
                    self.add(
                        operator=op_sx_sy_c,
                        multiplier=coupling_gx,
                        modifies=True,
                        sites=[i, j],
                    )
                    # S_y S_x term
                    self.add(
                        operator=op_sy_sx_c,
                        multiplier=coupling_gx,
                        modifies=True,
                        sites=[i, j],
                    )
                    if self._logger is not None:
                        self._log(
                            f"Adding Γ_x terms (SxSy + SySx) at ({i},{j}) with value {coupling_gx:.2f}",
                            lvl=2,
                            log="debug",
                        )

                # ====== Gamma_y terms: Γ_y (S_i^y S_j^z + S_i^z S_j^y) ======
                if self._gy is not None and not np.isclose(self._gy, 0.0, rtol=1e-10):
                    coupling_gy = self._gy * phase

                    # S_y S_z term
                    self.add(
                        operator=op_sy_sz_c,
                        multiplier=coupling_gy,
                        modifies=True,
                        sites=[i, j],
                    )
                    # S_z S_y term
                    self.add(
                        operator=op_sz_sy_c,
                        multiplier=coupling_gy,
                        modifies=True,
                        sites=[i, j],
                    )
                    if self._logger is not None:
                        self._log(
                            f"Adding Γ_y terms (SySz + SzSy) at ({i},{j}) with value {coupling_gy:.2f}",
                            lvl=2,
                            log="debug",
                        )

                # ====== Gamma_z terms: Γ_z (S_i^z S_j^x + S_i^x S_j^z) ======
                if self._gz is not None and not np.isclose(self._gz, 0.0, rtol=1e-10):
                    coupling_gz = self._gz * phase

                    # S_z S_x term
                    self.add(
                        operator=op_sz_sx_c,
                        multiplier=coupling_gz,
                        modifies=True,
                        sites=[i, j],
                    )
                    # S_x S_z term
                    self.add(
                        operator=op_sx_sz_c,
                        multiplier=coupling_gz,
                        modifies=True,
                        sites=[i, j],
                    )
                    if self._logger is not None:
                        self._log(
                            f"Adding Γ_z terms (SzSx + SxSz) at ({i},{j}) with value {coupling_gz:.2f}",
                            lvl=2,
                            log="debug",
                        )

    # --------------------------------------------------------------------------------------
    #! PROPERTIES
    # --------------------------------------------------------------------------------------

    @property
    def Gamma_x(self) -> Union[float, None]:
        """Gamma_x coupling strength (off-diagonal S_x S_y coupling)."""
        return self._gx

    @property
    def Gamma_y(self) -> Union[float, None]:
        """Gamma_y coupling strength (off-diagonal S_y S_z coupling)."""
        return self._gy

    @property
    def Gamma_z(self) -> Union[float, None]:
        """Gamma_z coupling strength (off-diagonal S_z S_x coupling)."""
        return self._gz

    @property
    def hx(self) -> Union[List[float], float, None]:
        """Magnetic field in x direction."""
        if self._hx is None:
            return None
        if all(np.isclose(x, self._hx[0]) for x in self._hx):
            return self._hx[0]
        return self._hx

    @property
    def hz(self) -> Union[List[float], float, None]:
        """Magnetic field in z direction."""
        if self._hz is None:
            return None
        if all(np.isclose(x, self._hz[0]) for x in self._hz):
            return self._hz[0]
        return self._hz

    @property
    def impurities(self) -> List[Tuple[int, float]]:
        """List of classical impurities (site_index, coupling_strength)."""
        return self._impurities


# --------------------------------------------------------------------------------------------------
#! END OF FILE
# --------------------------------------------------------------------------------------------------
