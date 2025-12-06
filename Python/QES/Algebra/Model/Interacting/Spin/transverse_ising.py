"""
High-level Hamiltonian class for the Transverse Field Ising Model (TFIM).

--------------------
File    : Model/Interacting/Spin/transverse_field_ising.py
Author  : Maksymilian Kliczkowski
Date    : 2025-04-26
Version : 0.2
Changes : 
    2025-10-30 (0.2) : Improved performance and fixed for new matrix representation. - MK
--------------------
"""

import numpy as np
from typing import List, Tuple, Union, Optional

# Assume these are available from the QES package:
try:
    import QES.Algebra.hilbert  as hilbert_module
    import QES.Algebra.hamil    as hamil_module
    from QES.Algebra.Operator   import operators_spin as operators_spin_module
except ImportError as e:
    raise ImportError("Required QES modules are not available.") from e

# Utilities (assuming availability)
try:
    from QES.general_python.lattices.lattice import Lattice, LatticeType
except ImportError as e:
    raise ImportError("Required QES lattice module is not available.") from e

##########################################################################################
#! HAMILTONIAN CLASS
##########################################################################################

class TransverseFieldIsing(hamil_module.Hamiltonian):
    r'''
    Hamiltonian for the Transverse Field Ising Model (TFIM).

    The Hamiltonian is defined as:
        H = -J \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j - h_x \sum_i \sigma^x_i

    where:
        - \sigma^z_i, \sigma^x_i are Pauli operators at site i.
        - \langle i,j \rangle denotes summation over nearest neighbors.
        - J is the Ising coupling strength (ferromagnetic if J > 0).
        - h_x is the strength of the transverse magnetic field.
    '''

    _ERR_EITHER_HIL_OR_NS = "TFIM: either the Hilbert space or the number of sites must be provided."

    def __init__(self,
                lattice             : Lattice, # Lattice is required for TFIM neighbors
                hilbert_space       : Optional[hilbert_module.HilbertSpace] = None,
                j                   : Union[List[float], float]             = 1.0,          # Ising coupling
                hx                  : Union[List[float], float]             = 1.0,          # Transverse field
                hz                  : Union[List[float], float]             = 1.0,          # Perpendicular field
                dtype               : type                                  = np.float64,   # Default to float64
                backend             : str                                   = "default",
                **kwargs):
        '''
        Constructor for the Transverse Field Ising Model Hamiltonian.

        ---
        Parameters:
            lattice : Lattice:
                The lattice structure defining sites and neighbors. Required.
            hilbert_space : Optional[hilbert_module.HilbertSpace]:
                The Hilbert space. If None, created based on lattice size.
            j : Union[List[float], float]:
                Ising coupling strength J. If a list, specifies site-dependent coupling
                (though typically uniform). Default is 1.0.
            hx : Union[List[float], float]:
                Transverse field strength h_x. If a list, specifies site-dependent field.
                Default is 1.0.
            hz : Union[List[float], float]:
                Perpendicular field strength h_z. If a list, specifies site-dependent field.
                Default is 1.0.
            dtype : type:
                Data type for the Hamiltonian (default: np.float64).
            backend : str:
                Backend for computations (default: "default").
            **kwargs :
                Additional keyword arguments passed to the base Hamiltonian class.
        '''

        if lattice is None:
            raise ValueError("TFIM requires a 'lattice' object to define neighbors.")

        # Initialize the base Hamiltonian class
        super().__init__(hilbert_space=hilbert_space, lattice=lattice, is_sparse=True, dtype=dtype, backend=backend, **kwargs)

        # Set Hamiltonian attributes before base class init
        self._name          = "Transverse Field Ising Model"

        # Store model-specific parameters
        self._j             = None                  # Initialize before setting
        self._hx            = None                  # Initialize before setting
        self._hz            = None                  # Initialize before setting
        # Set Hamiltonian attributes
        self._is_sparse     = True
        self._is_manybody   = True
        self._max_local_ch  = self.cardinality + 1  # Each site can be flipped (ns terms) + diagonal
                                                    # (1 - hx) + (n_neighbors - hz + (n_neighbors - J))
                                                    
        #! Build the Hamiltonian Terms
        self.set_couplings(j=j, hx=hx, hz=hz)
        self._set_local_energy_operators()
        self.setup_instruction_codes()          # automatic physics-based setup
        self._set_local_energy_functions()
        self._log(f"TFIM Hamiltonian initialized for {self.ns} sites.", lvl=1, log='debug')

    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Concise, human-readable description of the TFIM instance.

        Examples
        --------
        >>> print(tfim)
        TFIM(Ns=32, J=1.000, hx=0.300)
        >>> print(tfim) # site-dependent h_x
        TFIM(Ns=32, J=1.000, hx=[min=-0.200, max=0.300])
        """
        prec    = 1
        sep     = ","
        tol     = 1e-10                      # tolerance for “all equal”
        
        # fields
        parts   = [f"TFIM(Ns={self.ns}"]

        parts  += [
            hamil_module.Hamiltonian.fmt("J",   self._j,    prec=prec),
            hamil_module.Hamiltonian.fmt("hx",  self._hx,   prec=prec),
            hamil_module.Hamiltonian.fmt("hz",  self._hz,   prec=prec)
        ]

        return sep.join(parts) + ")"

    def __str__(self): return self.__repr__()

    # ----------------------------------------------------------------------------------------------
    #! INIT / SETTERS
    # ----------------------------------------------------------------------------------------------

    def set_couplings(self,
                        j       : Union[List[float], float, None] = None,
                        hx      : Union[List[float], float, None] = None,
                        hz      : Union[List[float], float, None] = None,
                        **kwargs
                        ):
        '''
        Sets or updates the Ising coupling (J) and transverse field (hx).
        Converts scalar inputs to lists matching the number of sites.

        Parameters:
            j : Optional[Union[List[float], float]]:
                Ising coupling strength(s).
            hx : Optional[Union[List[float], float]]:
                Transverse field strength(s).
            hz : Optional[Union[List[float], float]]:
                Perpendicular field strength(s).
        '''
        self._j                 = self._set_some_coupling(j)
        self._hx                = self._set_some_coupling(hx)
        self._hz                = self._set_some_coupling(hz)

    def _set_local_energy_operators(self):
        """
        Builds the list of operators defining the TFIM Hamiltonian.

        Adds terms for:
            - Transverse field: -hx[i] * Sx_i for each site i
            - Ising coupling:   -J[i] * Sz_i * Sz_j for each nearest-neighbor pair <i,j>
        """
        # Clear existing operators (important if rebuilding)
        super()._set_local_energy_operators()
        self._log("Building TFIM operator list...", lvl=1, log='info')

        lattice     = self._lattice
        
        #! Define Base Operators
        # Local operators (act on one site)
        op_sx_l     = operators_spin_module.sig_x(lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local)
        op_sz_l     = operators_spin_module.sig_z(lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Local)

        # Correlation operators (act on two sites)
        op_sz_sz_c  = operators_spin_module.sig_z(lattice=lattice, type_act=operators_spin_module.OperatorTypeActing.Correlation)

        #! Add Transverse Field Terms
        for i in range(self.ns):
            if not np.isclose(self._hx[i], 0.0):
                self.add(operator = op_sx_l, multiplier = -self._hx[i], sites = [i], modifies = True)
                self._log(f"Adding Sx at site {i} with multiplier {-self._hx[i]:.3f}", lvl=2, log='debug')
        
        #! Add Ising Perpendicular Terms
        for i in range(self.ns):
            if not np.isclose(self._hz[i], 0.0):
                self.add(operator = op_sz_l, multiplier = -self._hz[i], sites = [i], modifies = False)
                self._log(f"Adding Sz at site {i} with multiplier {-self._hz[i]:.3f}", lvl=2, log='debug')
        
        #! Add Ising Interaction Terms
        # Sum over unique nearest-neighbor pairs <i,j>
        for i in range(self.ns):
            # Use get_nn_forward to avoid double counting bonds
            nn_forward_num = lattice.get_nn_forward_num(i)
            
            for nn_idx in range(nn_forward_num):
                
                j_neighbor  = lattice.get_nn_forward(i, num=nn_idx)
                if lattice.wrong_nei(j_neighbor):
                    continue
                
                wx, wy, wz  = lattice.bond_winding(i, j_neighbor)
                phase       = lattice.boundary_phase_from_winding(wx, wy, wz)

                # Use the coupling J associated with site i (or average, depending on convention)
                # Simplest: use J[i]. Assume coupling belongs to the bond originating from i.
                coupling_j  = self._j[i] * phase

                if not np.isclose(coupling_j, 0.0):
                    self.add(operator = op_sz_sz_c, multiplier = -coupling_j, sites = [i, j_neighbor], modifies = False)
                    self._log(f"Adding SzSz between sites ({i}, {j_neighbor}) with multiplier {-coupling_j:.3f}", lvl=2, log='debug')

    # ----------------------------------------------------------------------------------------------
    #! Properties
    # ----------------------------------------------------------------------------------------------
    
    @property
    def J(self) -> Union[List[float], float]:
        """Ising coupling strength(s) J."""
        # Return scalar if uniform, otherwise list
        if self._j and all(np.isclose(x, self._j[0]) for x in self._j):
            return self._j[0]
        return self._j

    @property
    def hx(self) -> Union[List[float], float]:
        """Transverse field strength(s) hx."""
        if self._hx and all(np.isclose(x, self._hx[0]) for x in self._hx):
            return self._hx[0]
        return self._hx
    
    # ----------------------------------------------------------------------------------------------
    
# --------------------------------------------------------------------------------------------------
#! END OF FILE
# --------------------------------------------------------------------------------------------------