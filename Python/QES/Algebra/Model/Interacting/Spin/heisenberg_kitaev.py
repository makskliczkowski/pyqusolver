r"""
High-level Hamiltonian class for the Quantum Eigen Solver (QES) package.
The Heisenberg-Kitaev model implementation with additional possible 
- external magnetic fields h_x, h_z - magnetic field terms in x and z directions,
- Heisenberg coupling J             - isotropic spin interaction,
- Kitaev interactions K_x, K_y, K_z - directional couplings depending on bond orientation,
- anisotropy parameter \Delta       - \Delta (modifies the Heisenberg coupling),
- \Gamma^\gamma interactions        - anisotropic off-diagonal couplings.

Based on the C++ implementation. 

File    : Algebra/Model/Interacting/Spin/heisenberg_kitaev.py
Author  : Maksymilian Kliczkowski
Date    : 2025-02-17
Version : 0.1
"""

import numpy as np
from typing import List, Tuple, Union, Optional

# QES package imports
from QES.Algebra import hilbert as hilbert_module
from QES.Algebra import hamil as hamil_module
from QES.Algebra.Operator import operators_spin as operators_spin_module

##########################################################################################
#! IMPORTS
##########################################################################################

from QES.general_python.lattices.lattice import Lattice, LatticeType

# ----------------------------------------------------------------------------------------
#! DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------

HEI_KIT_X_BOND_NEI = 2
HEI_KIT_Y_BOND_NEI = 1
HEI_KIT_Z_BOND_NEI = 0

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################

class HeisenbergKitaev(hamil_module.Hamiltonian):
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
                logger              : Optional['Logger']                    = None,
                *,
                hilbert_space       : Optional[hilbert_module.HilbertSpace] = None,
                # Heisenberg couplings
                J                   : Union[List[float], None, float]       = None,
                dlt                 : Union[List[float], None, float]       = 1.0,
                # Gamma interactions
                Gamma               : Union[List[float], None, float]       = None,
                # Magnetic fields
                hx                  : Union[List[float], None, float]       = None,
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
                         hilbert_space  = hilbert_space,
                         lattice        = lattice, 
                         is_sparse      = True,
                         dtype          = dtype, backend=backend, logger=logger, use_forward=use_forward, **kwargs)

        # Initialize the Hamiltonian
        if hilbert_space is None:
            self._hilbert_space         = hilbert_module.HilbertSpace(ns=self.ns, backend=backend, dtype=dtype, nhl=2)
        
        # setup the fields
        self._hx                        = hx    if isinstance(hx, (list, np.ndarray, tuple)) else [hx] * self.ns if hx is not None else None
        self._hz                        = hz    if isinstance(hz, (list, np.ndarray, tuple)) else [hz] * self.ns if hz is not None else None
        # setup the couplings
        self._j                         = J     if isinstance(J, (list, np.ndarray, tuple)) else [J] * self.ns if J is not None else None
        self._gx, self._gy, self._gz    = Gamma if isinstance(Gamma, (list, np.ndarray, tuple)) else (Gamma, Gamma, Gamma) if Gamma is not None else (None, None, None)
        self._dlt                       = dlt   if isinstance(dlt, (list, np.ndarray, tuple)) else [dlt] * self.ns if dlt is not None else None
        self._kx, self._ky, self._kz    = K     if isinstance(K, (list, np.ndarray, tuple)) and len(K) == 3 else (K, K, K)
        # setup the impurities
        if self._logger is not None:
            self._log(f"Setting up impurities: {impurities}", lvl = 2, log = 'info', color = 'green')
        self._impurities                = impurities if (isinstance(impurities, list) and all(isinstance(i, tuple) and len(i) == 2 for i in impurities)) else []
            
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
                                        (3 if self._j is not None else 0)               + \
                                        (2 if self._gx is not None else 0)              + \
                                        (2 if self._gy is not None else 0)              + \
                                        (2 if self._gz is not None else 0)              + \
                                        (1 if self._hz is not None else 0)              + \
                                        (1 if self._hx is not None else 0)              + \
                                        len(self._impurities)
        self.set_couplings()
        
        # functions for local energy calculation in a jitted way (numpy and jax)
        self._set_local_energy_operators()
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
        prec   = 1          # decimal places
        sep    = ","        # parameter separator

        # string
        parts = [f"Kitaev(Ns={self.ns}"]

        parts += [
            hamil_module.Hamiltonian.fmt("Kx",      self._kx,       prec=prec),
            hamil_module.Hamiltonian.fmt("Ky",      self._ky,       prec=prec),
            hamil_module.Hamiltonian.fmt("Kz",      self._kz,       prec=prec),
            hamil_module.Hamiltonian.fmt("J",       self._j,        prec=prec) if self._j   is not None else "",
            hamil_module.Hamiltonian.fmt("Gx",      self._gx,       prec=prec) if self._gx  is not None else "",
            hamil_module.Hamiltonian.fmt("Gy",      self._gy,       prec=prec) if self._gy  is not None else "",
            hamil_module.Hamiltonian.fmt("Gz",      self._gz,       prec=prec) if self._gz  is not None else "",
            hamil_module.Hamiltonian.fmt("dlt",     self._dlt,      prec=prec) if self._dlt is not None else "",
            hamil_module.Hamiltonian.fmt("hz",      self._hz,       prec=prec) if self._hz  is not None else "",
            hamil_module.Hamiltonian.fmt("hx",      self._hx,       prec=prec) if self._hx  is not None else "",
        ]

        # symmetry / boundary info from HilbertSpace object
        hilbert_info = self.hilbert_space.get_sym_info().strip()
        if len(hilbert_info) > 0:
            parts.append(hilbert_info)
        parts.append(str(self.lattice.bc))
        
        return sep.join(parts) + ")"
    
    def __str__(self): return self.__repr__()

    # ----------------------------------------------------------------------------------------------
    #! INIT
    # ----------------------------------------------------------------------------------------------
    
    def set_couplings(self,
                        hx      : Union[List[float], None, float]       = None,
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
        lattice         = self._lattice
        
        #! define the operators beforehand - to avoid multiple creations
        op_sx_l         =   operators_spin_module.sig_x(lattice = lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Local)
        op_sy_l         =   operators_spin_module.sig_y(lattice = lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Local)
        op_sz_l         =   operators_spin_module.sig_z(lattice = lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Local)
        
        # Kitaev and Heisenberg terms
        op_sx_sx_c      =   operators_spin_module.sig_x(lattice = lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
        op_sy_sy_c      =   operators_spin_module.sig_y(lattice = lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
        op_sz_sz_c      =   operators_spin_module.sig_z(lattice = lattice,
                                type_act = operators_spin_module.OperatorTypeActing.Correlation)
        # Gamma terms - off-diagonal couplings SxSy + SySx, SxSz + SzSx, SySz + SzSy
        op_sx_sy_c      =   op_sx_l * op_sy_l
        op_sz_sx_c      =   op_sz_l * op_sx_l
        op_sy_sz_c      =   op_sy_l * op_sz_l
        op_sz_sy_c      =   op_sz_l * op_sy_l
        op_sx_sz_c      =   op_sx_l * op_sz_l
        op_sy_sx_c      =   op_sy_l * op_sx_l

        nn_nums         =   [lattice.get_nn_num(i) for i in range(self.ns)] if self._use_forward else \
                            [lattice.get_nn_forward_num(i) for i in range(self.ns)]
        
        #! iterate over all the sites
        for i in range(self.ns):
            self._log(f"Starting i: {i}", lvl = 1, log = 'debug')
            
            #? z-field
            if self._hz is not None and not np.isclose(self._hz[i], 0.0, rtol=1e-10):
                self.add(op_sz_l, multiplier = self._hz[i], modifies = False, sites = [i])
                self._log(f"Adding local Sz at {i} with value {self._hz[i]:.2f}", lvl = 2, log = 'debug')

            #? x-field
            if self._hx is not None and not np.isclose(self._hx[i], 0.0, rtol=1e-10):
                self.add(op_sx_l, multiplier = self._hx[i], modifies = True, sites = [i])
                self._log(f"Adding local Sx at {i} with value {self._hx[i]:.2f}", lvl = 2, log = 'debug')
            
            #? impurities
            for (imp_site, imp_strength) in self._impurities:
                if imp_site == i:
                    self.add(op_sz_l, multiplier = imp_strength, modifies = False, sites = [i])
                    self._log(f"Adding impurity Sz at {i} with value {imp_strength:.2f}", lvl = 2, log = 'debug')
            
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
                    sz_sz   = self._j[i] * self._dlt[i] if self._j is not None else 0.0  # Heisenberg - value of SzSz (multiplier)
                    sx_sx   = self._j[i]                if self._j is not None else 0.0  # Heisenberg - value of SxSx (multiplier)
                    sy_sy   = self._j[i]                if self._j is not None else 0.0  # Heisenberg - value of SySy (multiplier)

                #! check the directional bond contributions - Kitaev
                if True:
                    if nn == HEI_KIT_Z_BOND_NEI:
                        sz_sz  += self._kz
                    elif nn == HEI_KIT_Y_BOND_NEI:
                        sy_sy  += self._ky
                    else:
                        sx_sx  += self._kx
                if True:
                    #? SzSz
                    if not np.isclose(sz_sz, 0.0, rtol=1e-10):
                        self.add(op_sz_sz_c, sites = [i, nei], multiplier = sz_sz * phase, modifies = False)
                        self._log(f"Adding SzSz at {i},{nei} with value {sz_sz:.2f}", lvl = 2, log = 'debug')
                    #? SxSx
                    if not np.isclose(sx_sx, 0.0, rtol=1e-10):
                        self.add(op_sx_sx_c, sites = [i, nei], multiplier = sx_sx * phase, modifies = True)
                        self._log(f"Adding SxSx at {i},{nei} with value {sx_sx:.2f}", lvl = 2, log = 'debug')
                    #? SySy
                    if not np.isclose(sy_sy, 0.0, rtol=1e-10):
                        self.add(op_sy_sy_c, sites = [i, nei], multiplier = sy_sy * phase, modifies = True)
                        self._log(f"Adding SySy at {i},{nei} with value {sy_sy:.2f}", lvl = 2, log = 'debug')
                
                #! Gamma terms
                # if True:
                    #? Gamma_x terms
                    # if self._gx is not None and not np.isclose(self._gx, 0.0, rtol=1e-10):
                    #     self.add(op_sx_sy_c, sites = [i, nei], multiplier = self._gx * phase, modifies = True)
                    #     self.add(op_sy_sx_c, sites = [i, nei], multiplier = self._gx * phase, modifies = True)
                    #     self._log(f"Adding Gamma_x SxSy + SySx at {i},{nei} with value {self._gx:.2f}", lvl = 2, log = 'debug')
                        
                    # #? Gamma_y terms
                    # if self._gy is not None and not np.isclose(self._gy, 0.0, rtol=1e-10):
                    #     self.add(op_sy_sz_c, sites = [i, nei], multiplier = self._gy * phase, modifies = True)
                    #     self.add(op_sz_sy_c, sites = [i, nei], multiplier = self._gy * phase, modifies = True)
                    #     self._log(f"Adding Gamma_y SySz + SzSy at {i},{nei} with value {self._gy:.2f}", lvl = 2, log = 'debug')
                        
                    # #? Gamma_z terms
                    # if self._gz is not None and not np.isclose(self._gz, 0.0, rtol=1e-10):
                    #     self.add(op_sz_sx_c, sites = [i, nei], multiplier = self._gz * phase, modifies = True)
                    #     self.add(op_sx_sz_c, sites = [i, nei], multiplier = self._gz * phase, modifies = True)
                    #     self._log(f"Adding Gamma_z SzSx + SxSz at {i},{nei} with value {self._gz:.2f}", lvl = 2, log = 'debug')

                #! Finalize the operator addition for this neighbor
                self._log(f"Finished processing neighbor {nei} of site {i}", lvl = 2, log = 'debug')
    
        self._log("Successfully set local energy operators...", lvl=1, log='info')

    # ----------------------------------------------------------------------------------------------

####################################################################################################