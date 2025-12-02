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

import numpy as np
import numba
from typing import List, Tuple, Union, Optional

# QES package imports
try:
    from QES.Algebra.hilbert                    import HilbertSpace
    from QES.Algebra.hamil                      import Hamiltonian
    import QES.Algebra.Operator.operators_spin  as operators_spin_module
    from QES.general_python.lattices.honeycomb  import HoneycombLattice, X_BOND_NEI, Y_BOND_NEI, Z_BOND_NEI
except ImportError as e:
    raise ImportError("Failed to import QES modules. Ensure that the QES package is correctly installed.") from e

##########################################################################################
#! IMPORTS
##########################################################################################

try:
    from QES.general_python.lattices.lattice import Lattice, LatticeType
except ImportError as e:
    raise ImportError("Failed to import QES modules. Ensure that the QES package is correctly installed.") from e

# ----------------------------------------------------------------------------------------
#! DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------

HEI_KIT_X_BOND_NEI  = X_BOND_NEI
HEI_KIT_Y_BOND_NEI  = Y_BOND_NEI
HEI_KIT_Z_BOND_NEI  = Z_BOND_NEI

# Pauli matrix scaling factors for HeisenbergKitaev (Pauli σ convention)
# SINGLE_TERM_MULT    = 1.0   # Single-spin terms (σ has eigenvalues ±1)
SINGLE_TERM_MULT    = 2.0   # Single-spin terms (σ has eigenvalues ±1)
# CORR_TERM_MULT      = 1.0   # Two-spin correlation terms (σ_i σ_j has eigenvalues ±1)
CORR_TERM_MULT      = 4.0   # Two-spin correlation terms (σ_i σ_j has eigenvalues ±1)

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################

class HeisenbergKitaev(Hamiltonian):
    '''
    Hamiltonian for an ergodic quantum dot coupled to an external system.
    The external system is modeled as a quantum spin chain.
    '''

    #############################
    
    _ERR_EITHER_HIL_OR_NS           = "QSM: either the Hilbert space or the number of particles in the system must be provided."
    _ERR_LATTICE_NOT_PROVIDED       = "QSM: Lattice must be provided to define the Kitaev-Heisenberg Hamiltonian."
    
    def __init__(self,
                lattice             : Lattice,
                K                   : Union[List[float], None, float]       = 1.0,
                *,
                hilbert_space       : Optional[HilbertSpace]                = None,
                # Heisenberg couplings
                J                   : Union[List[float], None, float]       = None,
                dlt                 : Union[List[float], None, float]       = 1.0,
                # Gamma interactions
                Gamma               : Union[List[float], None, float]       = None,
                # Magnetic fields
                hx                  : Union[List[float], None, float]       = None,
                hy                  : Union[List[float], None, float]       = None,
                hz                  : Union[List[float], None, float]       = None,
                # Classical impurities (s^z_i * <s^z_imp> * cos(theta), with theta the angle between the spin and the z-axis)
                impurities          : List[Tuple[int, float]]               = [],
                # other parameters
                dtype               : type                                  = np.float64,
                backend             : str                                   = "default",
                use_forward         : bool                                  = True,
                **kwargs):
        '''
        Constructor for the QSM Hamiltonian.
        
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
            impurities : List[Tuple[int, float]], optional
                List of classical impurities defined as tuples of (site index, coupling strength). Default is [].
                Each impurity contributes to the Hamiltonian as s^z_i * <s^z_imp> * cos(theta), with theta the angle between the spin and the z-axis.
                Example: [(0, 0.5), (3, -0.3)] adds impurities at sites 0 and 3 with respective strengths.
            dtype : type, optional
                Data type for numerical computations. Default is np.float64.
            backend : str, optional
                Computational backend to use. Default is "default".
            use_forward : bool, optional
                Whether to use forward nearest neighbors only. Default is True.
            **kwargs : additional keyword arguments
                Additional arguments passed to the Hamiltonian base class.            
        '''
        
        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)
        
        if lattice.typek not in [LatticeType.HONEYCOMB, LatticeType.HEXAGONAL]:
            self._log(f"The type of the lattice {lattice} is not standard. Check your intentions...", lvl = 2)

        # Initialize the Hamiltonian
        self._lattice                   = lattice
        super().__init__(is_manybody    = True, 
                        hilbert_space   = hilbert_space,
                        lattice         = lattice, 
                        is_sparse       = True,
                        dtype           = dtype if (hy is None and Gamma is None) else np.complex128, # enforce complex dtype if hy field is used
                        backend=backend, use_forward=use_forward, **kwargs)

        # Initialize the Hamiltonian
        if hilbert_space is None:
            self._hilbert_space         = HilbertSpace(ns=self.ns, backend=backend, dtype=dtype, nhl=2)
        
        # setup the fields
        self._hx                        = hx    if isinstance(hx, (list, np.ndarray, tuple)) else [hx] * self.ns if hx is not None else None
        self._hy                        = hy    if isinstance(hy, (list, np.ndarray, tuple)) else [hy] * self.ns if hy is not None else None
        self._hz                        = hz    if isinstance(hz, (list, np.ndarray, tuple)) else [hz] * self.ns if hz is not None else None
        # setup the couplings
        self._j                         = J     if isinstance(J, (list, np.ndarray, tuple)) else [J] * self.ns if J is not None else None
        self._gx, self._gy, self._gz    = Gamma if isinstance(Gamma, (list, np.ndarray, tuple)) else (Gamma, Gamma, Gamma) if Gamma is not None else (None, None, None)
        self._dlt                       = dlt   if isinstance(dlt, (list, np.ndarray, tuple)) else [dlt] * self.ns if dlt is not None else None
        self._kx, self._ky, self._kz    = K     if isinstance(K, (list, np.ndarray, tuple)) and len(K) == 3 else (K, K, K)
        # setup the impurities
        self._impurities                = impurities if (isinstance(impurities, list) and all(isinstance(i, tuple) and len(i) == 2 for i in impurities)) else []
        self._log(f"Initializing Heisenberg-Kitaev Hamiltonian on lattice: {lattice}", lvl = 1, log = 'info', color = 'green')
        self._log(f"Impurities provided: {self._impurities}", lvl = 2, log = 'info', color = 'blue')
            
        self._neibz                     = [[]]
        self._neiby                     = [[]]
        self._neibx                     = [[]]
        self._neiadd                    = [[]]
        
        # Initialize the Hamiltonian
        self._name                      = "Kitaev-Heisenberg-Gamma Model"
        self._is_sparse                 = True
        
        #! Calculate maximum local coupling channels
        # 3 from Heisenberg + 3 from Kitaev + 6 from Gamma + 2 from fields + impurities len(self._impurities)
        self._max_local_ch              = 3                                             + \
                                        (3 if self._j is not None else  0)              + \
                                        (2 if self._gx is not None else 0)              + \
                                        (2 if self._gy is not None else 0)              + \
                                        (2 if self._gz is not None else 0)              + \
                                        (1 if self._hz is not None else 0)              + \
                                        (1 if self._hy is not None else 0)              + \
                                        (1 if self._hx is not None else 0)              + \
                                        len(self._impurities)
        self.set_couplings()
        
        # functions for local energy calculation in a jitted way (numpy and jax)
        self._set_local_energy_operators()
        self._lookup_codes              = operators_spin_module.SPIN_LOOKUP_CODES.to_dict()
        self._instr_function            = operators_spin_module.sigma_composition_integer(is_complex = self._iscpx)
        self._instr_max_out             = len(self._instr_codes) + 1
        self._set_local_energy_functions()
    
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
        prec   = 3          # decimal places
        sep    = ","        # parameter separator

        # string
        parts = [f"Kitaev(Ns={self.ns}"]

        parts += [
            Hamiltonian.fmt("Kx",      self._kx,       prec=prec),
            Hamiltonian.fmt("Ky",      self._ky,       prec=prec),
            Hamiltonian.fmt("Kz",      self._kz,       prec=prec),
            Hamiltonian.fmt("J",       self._j,        prec=prec) if self._j   is not None else "",
            Hamiltonian.fmt("Gx",      self._gx,       prec=prec) if self._gx  is not None else "",
            Hamiltonian.fmt("Gy",      self._gy,       prec=prec) if self._gy  is not None else "",
            Hamiltonian.fmt("Gz",      self._gz,       prec=prec) if self._gz  is not None else "",
            Hamiltonian.fmt("dlt",     self._dlt,      prec=prec) if self._dlt is not None else "",
            Hamiltonian.fmt("hz",      self._hz,       prec=prec) if self._hz  is not None else "",
            Hamiltonian.fmt("hy",      self._hy,       prec=prec) if self._hy  is not None else "",
            Hamiltonian.fmt("hx",      self._hx,       prec=prec) if self._hx  is not None else "",
        ]
        
        parts = [p for p in parts if p]
        return sep.join(parts) + ")"
    
    def __str__(self): return self.__repr__()

    @property
    def bonds(self):
        return { HEI_KIT_X_BOND_NEI: 'x', HEI_KIT_Y_BOND_NEI: 'y', HEI_KIT_Z_BOND_NEI: 'z',
            'x': HEI_KIT_X_BOND_NEI, 'y': HEI_KIT_Y_BOND_NEI, 'z': HEI_KIT_Z_BOND_NEI }

    # ----------------------------------------------------------------------------------------------
    #! INIT
    # ----------------------------------------------------------------------------------------------
    
    def set_couplings(self,
                        hx      : Union[List[float], None, float]       = None,
                        hy      : Union[List[float], None, float]       = None,
                        hz      : Union[List[float], None, float]       = None,
                        kx      : Union[List[float], None, float]       = None,
                        ky      : Union[List[float], None, float]       = None,
                        kz      : Union[List[float], None, float]       = None,
                        j       : Union[List[float], None, float]       = None,
                        gx      : Union[List[float], None, float]       = None,
                        gy      : Union[List[float], None, float]       = None,
                        gz      : Union[List[float], None, float]       = None,
                        dlt     : Union[List[float], None, float]       = None):
        '''
        Sets the couplings based on their initial value (list, string, value)
        '''
        self._hx            = self._set_some_coupling(self._hx  if hx  is None else hx ) if hx is not None else self._hx
        self._hy            = self._set_some_coupling(self._hy  if hy  is None else hy ) if hy is not None else self._hy
        self._hz            = self._set_some_coupling(self._hz  if hz  is None else hz ) if hz is not None else self._hz
        self._kx            = self._set_some_coupling(self._kx  if kx  is None else kx ) if kx is not None else self._kx
        self._ky            = self._set_some_coupling(self._ky  if ky  is None else ky ) if ky is not None else self._ky
        self._kz            = self._set_some_coupling(self._kz  if kz  is None else kz ) if kz is not None else self._kz
        self._j             = self._set_some_coupling(self._j   if j   is None else j  ) if j is not None else self._j
        self._dlt           = self._set_some_coupling(self._dlt if dlt is None else dlt) if dlt is not None else self._dlt
        self._gx            = self._set_some_coupling(self._gx  if gx  is None else gx ) if gx is not None else self._gx
        self._gy            = self._set_some_coupling(self._gy  if gy  is None else gy ) if gy is not None else self._gy
        self._gz            = self._set_some_coupling(self._gz  if gz  is None else gz ) if gz is not None else self._gz

    def _set_local_energy_operators(self):
        """
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
                
        Note:
            This method updates internal state and does not return a value.
        """
        super()._set_local_energy_operators()
        
        # Use the lattice from the Hamiltonian
        lattice         =   self._lattice
        if lattice is None:
            raise ValueError(self._ERR_LATTICE_NOT_PROVIDED)
        
        #! define the operators beforehand - to avoid multiple creations
        op_sx_l         =   operators_spin_module.sig_x(lattice = lattice, type_act = operators_spin_module.OperatorTypeActing.Local)
        op_sy_l         =   operators_spin_module.sig_y(lattice = lattice, type_act = operators_spin_module.OperatorTypeActing.Local)
        op_sz_l         =   operators_spin_module.sig_z(lattice = lattice, type_act = operators_spin_module.OperatorTypeActing.Local)

        # Kitaev and Heisenberg terms - correlation operators (two-site)
        op_sx_sx_c      =   operators_spin_module.sig_x(lattice = lattice, type_act = operators_spin_module.OperatorTypeActing.Correlation)
        op_sy_sy_c      =   operators_spin_module.sig_y(lattice = lattice, type_act = operators_spin_module.OperatorTypeActing.Correlation)
        op_sz_sz_c      =   operators_spin_module.sig_z(lattice = lattice, type_act = operators_spin_module.OperatorTypeActing.Correlation)
        
        # Create Gamma operators as products of correlation operators
        op_sx_sy_c      =   operators_spin_module.make_sigma_mixed("xy", lattice=lattice)
        op_sy_sx_c      =   operators_spin_module.make_sigma_mixed("yx", lattice=lattice)
        
        op_sz_sx_c      =   operators_spin_module.make_sigma_mixed("zx", lattice=lattice)
        op_sx_sz_c      =   operators_spin_module.make_sigma_mixed("xz", lattice=lattice)
        
        op_sy_sz_c      =   operators_spin_module.make_sigma_mixed("yz", lattice=lattice)
        op_sz_sy_c      =   operators_spin_module.make_sigma_mixed("zy", lattice=lattice)

        nn_nums         =   [lattice.get_nn_forward_num(i) for i in range(self.ns)] if self._use_forward else \
                            [lattice.get_nn_num(i) for i in range(self.ns)]

        #! iterate over all the sites
        elems           =   0
        for i in range(self.ns):
            self._log(f"Starting i: {i}", lvl = 1, log = 'debug')
            
            #? z-field (single-spin term: applying SINGLE_TERM_MULT scaling for Pauli matrices)
            if self._hz is not None and not np.isclose(self._hz[i], 0.0, rtol=1e-10):
                z_field = SINGLE_TERM_MULT * self._hz[i]
                self.add(op_sz_l, multiplier = z_field, modifies = False, sites = [i])
                self._log(f"Adding local Sz at {i} with value {z_field:.2f}", lvl = 2, log = 'debug')

            #? y-field (single-spin term: applying SINGLE_TERM_MULT scaling for Pauli matrices)
            if self._hy is not None and not np.isclose(self._hy[i], 0.0, rtol=1e-10):
                y_field = SINGLE_TERM_MULT * self._hy[i]
                self.add(op_sy_l, multiplier = y_field, modifies = True, sites = [i])
                self._log(f"Adding local Sy at {i} with value {y_field:.2f}", lvl = 2, log = 'debug')

            #? x-field (single-spin term: applying SINGLE_TERM_MULT scaling for Pauli matrices)
            if self._hx is not None and not np.isclose(self._hx[i], 0.0, rtol=1e-10):
                x_field = SINGLE_TERM_MULT * self._hx[i]
                self.add(op_sx_l, multiplier = x_field, modifies = True, sites = [i])
                self._log(f"Adding local Sx at {i} with value {x_field:.2f}", lvl = 2, log = 'debug')
            
            #? impurities
            for (imp_site, imp_strength) in self._impurities:
                if imp_site == i:
                    imp_field = SINGLE_TERM_MULT * imp_strength
                    self.add(op_sz_l, multiplier = imp_field, modifies = False, sites = [i])
                    self._log(f"Adding impurity Sz at {i} with value {imp_field:.2f}", lvl = 2, log = 'debug')

            #? now check the correlation operators
            nn_num = nn_nums[i]
            
            for nn in range(nn_num):
                # get the neighbor index
                nei         = lattice.get_nn_forward(i, num=nn) if self._use_forward else lattice.get_nn(i, num=nn)
                    
                #! check the direction of the bond
                if lattice.wrong_nei(nei):
                    continue
                
                wx, wy, wz  = lattice.bond_winding(i, nei)
                phase       = lattice.boundary_phase_from_winding(wx, wy, wz)
                
                #! Heisenberg - value of SzSz, SxSx, SySy (multipliers)
                if True:
                    sz_sz   = phase * CORR_TERM_MULT * self._j[i] * self._dlt[i] if self._j is not None else 0.0  # Heisenberg - value of SzSz (multiplier)
                    sx_sx   = phase * CORR_TERM_MULT * self._j[i]                if self._j is not None else 0.0  # Heisenberg - value of SxSx (multiplier)
                    sy_sy   = phase * CORR_TERM_MULT * self._j[i]                if self._j is not None else 0.0  # Heisenberg - value of SySy (multiplier)

                #! check the directional bond contributions - Kitaev
                if True:
                    if nn == HEI_KIT_Z_BOND_NEI:
                        sz_sz  += phase * CORR_TERM_MULT * self._kz if self._kz is not None else 0.0
                    elif nn == HEI_KIT_Y_BOND_NEI:
                        sy_sy  += phase * CORR_TERM_MULT * self._ky if self._ky is not None else 0.0
                    elif nn == HEI_KIT_X_BOND_NEI:
                        sx_sx  += phase * CORR_TERM_MULT * self._kx if self._kx is not None else 0.0
                        
                if True:
                    if not np.isclose(sz_sz, 0.0, rtol=1e-10):
                        self.add(op_sz_sz_c, sites = [i, nei], multiplier = sz_sz, modifies = False)
                        self._log(f"Adding SzSz at {i},{nei} with value {sz_sz:.2f}", lvl = 2, log = 'debug')
                        elems += 1
                    if not np.isclose(sx_sx, 0.0, rtol=1e-10):
                        self.add(op_sx_sx_c, sites = [i, nei], multiplier = sx_sx, modifies = True)
                        self._log(f"Adding SxSx at {i},{nei} with value {sx_sx:.2f}", lvl = 2, log = 'debug')
                        elems += 1
                    if not np.isclose(sy_sy, 0.0, rtol=1e-10):
                        self.add(op_sy_sy_c, sites = [i, nei], multiplier = sy_sy, modifies = True)
                        self._log(f"Adding SySy at {i},{nei} with value {sy_sy:.2f}", lvl = 2, log = 'debug')
                        elems += 1
                
                
                #! Gamma terms
                if True:
                    #? Gamma_x terms
                    if self._gx is not None and not np.isclose(self._gx, 0.0, rtol=1e-10) and nn == HEI_KIT_X_BOND_NEI:
                        val = self._gx * phase * CORR_TERM_MULT
                        self.add(op_sy_sz_c, sites = [i, nei], multiplier = val, modifies = True)
                        self.add(op_sz_sy_c, sites = [i, nei], multiplier = val, modifies = True)
                        self._log(f"Adding Gamma_x(SySz+SzSy) at {i},{nei} with value {val:.2f}", lvl = 2, log = 'debug')
                        elems += 2

                    # #? Gamma_y terms
                    if self._gy is not None and not np.isclose(self._gy, 0.0, rtol=1e-10) and nn == HEI_KIT_Y_BOND_NEI:
                        val = self._gy * phase * CORR_TERM_MULT
                        self.add(op_sz_sx_c, sites = [i, nei], multiplier = val, modifies = True)
                        self.add(op_sx_sz_c, sites = [i, nei], multiplier = val, modifies = True)
                        self._log(f"Adding Gamma_y(SzSx + SxSz) at {i},{nei} with value {val:.2f}", lvl = 2, log = 'debug')
                        elems += 2

                    # #? Gamma_z terms
                    if self._gz is not None and not np.isclose(self._gz, 0.0, rtol=1e-10) and nn == HEI_KIT_Z_BOND_NEI:
                        val = self._gz * phase * CORR_TERM_MULT
                        self.add(op_sx_sy_c, sites = [i, nei], multiplier = val, modifies = True)
                        self.add(op_sy_sx_c, sites = [i, nei], multiplier = val, modifies = True)
                        self._log(f"Adding Gamma_z(SxSy + SySx) at {i},{nei} with value {val:.2f}", lvl = 2, log = 'debug')
                        elems += 2

                #! Finalize the operator addition for this neighbor
                self._log(f"Finished processing neighbor {nei} of site {i}", lvl = 2, log = 'debug')
        
        self._log(f"Total NN elements added: {elems}", color='red', lvl=3)
        self._log("Successfully set local energy operators...", lvl=1, log='info')
        
    # ----------------------------------------------------------------------------------------------

##########################################################################################
#! EOF