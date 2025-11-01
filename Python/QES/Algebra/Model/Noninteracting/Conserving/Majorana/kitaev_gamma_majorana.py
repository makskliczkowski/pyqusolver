r"""
This is a model for non-interacting Majorana fermions hopping on lattice in
the field created by magnetic fluxes. For us, this comes from the spin-1/2 
Gamma-model on a honeycomb lattice:
$$
H = \sum_{\{i,j\}^\gamma} \Gamma^{\gamma} (S^\alpha_i S^\beta_j + S^\beta_i S^\alpha_j)
$$
where $\gamma$ is the bond type (x, y, or z) and $\alpha, \beta$ are the two spin components
different from $\gamma$. The spin operators are represented in terms of four Majorana fermions
$$
S_i^\alpha = \frac{i}{2} b_i^\alpha c_i
$$
where $c_i$ is the itinerant Majorana fermion and $b_i^\alpha$ are the three gauge Majorana fermions.
The Hamiltonian can be rewritten as
$$
H = \frac{i}{4} \sum_{\{i,j\}^\gamma} i \frac{\Gamma^{\gamma}}{4} (b_i^\alpha b_j^\beta + b_i^\beta b_j^\alpha) c_i c_j
$$

In general, this model is interacting and is not solvable like the Kitaev model. However,
one can define two types of operators:

a) Directional fermions:
$$
K_{i}^{\alpha,\beta}            = \frac{i}{2} (b_i^\alpha + i b_i^\beta),
$$
$$
K_{i}^{\alpha,\beta \dagger}    = \frac{i}{2} (b_i^\alpha - i b_i^\beta),
$$
K_{i}^{\alpha, \beta}^\dagger   = -i K_{i}^{\beta, \alpha},
$$
$$
\{ K_{i}^{\alpha,\beta}, K_{j}^{\alpha,\beta \dagger} \} = \delta_{i,j},
$$
\{ K_{i}^{\alpha,\beta}, K_{j}^{\alpha,\beta} \} = 0.
$$
These operators create/annihilate a fermion on site $i$ that lives on the direction (rotation around site $i$) 
defined by the ordered pair of spin components $(\alpha, \beta)$. The Hamiltonian can be rewritten
in terms of these operators as
$$
H = -i \sum_{\{i,j\}^\gamma} (K_{i}^{\alpha,\beta} K_{j}^{\alpha,\beta} - K_{i}^{\alpha,\beta}^\dagger K_{i}^{\alpha,\beta}^\dagger) c_i c_j
$$
This Hamiltonian then lives filling the fluctuation sector defined
by the directional fermions being annihilated or created in pairs on each bond directionally.

In this interpretation, one can treat the directional fermions terms with 
mean field approximation (assuming quantum fluctuations are small) and obtain
a quadratic Hamiltonian for the itinerant Majorana fermions $c_i$ only, which
can be solved exactly

We will describe it further later...

b) Alternatively, one can define two complex-fermions on each bond:
$$
F_{ij}^{(\gamma, 1)} = \frac{1}{2} (b_i^\alpha + i b_j^\beta),
$$
$$
F_{ij}^{(\gamma, 2)} = \frac{1}{2} (b_i^\beta + i b_j^\alpha),
$$
and all their hermitian conjugates and fermionic anti-commutation relations.

In such scenario, defining one can plug them in the Hamiltonian:
$$
H = -i \sum_{\{i,j\}^\gamma} \frac{\Gamma^{\gamma}}{2} (n_{ij}^{(\gamma, 1)} + n_{ij}^{(\gamma, 2)} - 1) c_i c_j
$$
where $n_{ij}^{(\gamma, 1)} = F_{ij}^{(\gamma, 1) \dagger} F_{ij}^{(\gamma, 1)}$ and
$n_{ij}^{(\gamma, 2)} = F_{ij}^{(\gamma, 2) \dagger} F_{ij}^{(\gamma, 2)}$ are the
number operators for the bond fermions. In this case, one notices that 
the values of the effective hopping amplitudes for the itinerant Majorana fermions
are determined by the occupation numbers of the bond fermions. These can be
$$
[-1, 0, +1]
$$
depenending on the fermionic occupation. In such case one notices that the value 0 corresponds
to the existence of quantum fluctuations of the gauge Majorana fermions on the bond,
while the values -1 and +1 correspond to the bond being occupied or unoccupied
by the bond fermions, respectively.

Thus, if the quantum fluctuations are behaving randomly, one can treat them as
disorder in the hopping amplitudes of the itinerant Majorana fermions and average the 
results over many disorder realizations.

1. We create mean field in the k-space for the directional fermions
2. We create mean field in the real space by treating the bond fermions occupations as fixed numbers
   that depend on the $\Gamma$ couplings.
3. We create randomly selected bond fermion occupations everywhere and try to average
   over many realizations. To do this, we first check:
   - Do fluctuations average to zero?       - We calculate translationally invariant free fermion model 
                                            and randomly assign hopping amplitudes of -1, 0, +1 with equal probabilities.
                                            Importantly, this needs to be done in a given:
                                                - k-space sector (momentum conservation)
                                                - parity sector #?
                                                - can I do this non-interactingly, how to define translationally invariant disorder?#?
   - What is the variance of fluctuations?  - Calculate from exact diagonalization in ED.
   - Can I compare the results for spin system with the mean field results mapped to true space (see projections)?

------------------------------------------------------
File        : QES/Algebra/Model/Noninteracting/Conserving/Majorana/gamma_majorana.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Date        : 2025-10-30
------------------------------------------------------
"""

import numpy as np
import numba
from typing import Optional, Union, List

# import the quadratic base
try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
except ImportError:
    raise ImportError("Could not import QuadraticHamiltonian base class. "
                      "Please ensure that QES.Algebra.hamil_quadratic module is available.")

try:
    from QES.general_python.lattices.honeycomb import HoneycombLattice
except ImportError:
    raise ImportError("Could not import HoneycombLattice. "
                      "Please ensure that QES.general_python.lattices.honeycomb module is available.")

try:
    from scipy.sparse import coo_matrix
    _HAS_SCIPY = True
except ImportError:
    coo_matrix = None
    _HAS_SCIPY = False

# -----------------------------------------------------
#! DEFINE CONSTANTS
# -----------------------------------------------------

HEI_KIT_X_BOND_NEI = 2
HEI_KIT_Y_BOND_NEI = 1
HEI_KIT_Z_BOND_NEI = 0

# -----------------------------------------------------

class KitaevGammaMajorana(QuadraticHamiltonian):
    '''Kitaev model with Majorana fermions on a honeycomb lattice.'''
    
    def __init__(self, 
                lattice     : HoneycombLattice,
                gamma_z     : Union[float, np.ndarray]              = 1.0,
                gamma_y     : Optional[Union[float, np.ndarray]]    = None,
                gamma_x     : Optional[Union[float, np.ndarray]]    = None,
                *args,
                k_z         : Optional[Union[float, np.ndarray]]    = None,
                k_y         : Optional[Union[float, np.ndarray]]    = None,
                k_x         : Optional[Union[float, np.ndarray]]    = None,
                # -----------
                h_x         : Optional[float]                       = None,
                h_y         : Optional[float]                       = None,
                h_z         : Optional[float]                       = None,
                # -----------
                logger      : Optional[object]                      = None,
                **kwargs):
        
        super().__init__(lattice                =   lattice,
                        particle_conserving     =   True,
                        dtype                   =   np.dtype(np.complex128), # the model is complex by construction
                        backend                 =   "numpy",
                        constant_offset         =   0.0,
                        particles               =   "fermions",
                        is_sparse               =   True,
                        logger                  =   logger,
                        *args,
                        **kwargs)
        
        # setups already for random couplings - can be as well constant
        self.gamma_z    = self._set_some_coupling(gamma_z).astype(self._dtype)
        self.gamma_y    = self._set_some_coupling(gamma_y).astype(self._dtype) if gamma_y is not None else self.gamma_z
        self.gamma_x    = self._set_some_coupling(gamma_x).astype(self._dtype) if gamma_x is not None else self.gamma_z
        self.k_z        = self._set_some_coupling(k_z).astype(self._dtype) if k_z is not None else None
        self.k_y        = self._set_some_coupling(k_y).astype(self._dtype) if k_y is not None else None
        self.k_x        = self._set_some_coupling(k_x).astype(self._dtype) if k_x is not None else None
        
        # will be treated in 3rd order perturbation theory
        self.h_x        = (float(h_x))**3 if h_x is not None else None
        self.h_y        = (float(h_y))**3 if h_y is not None else None
        self.h_z        = (float(h_z))**3 if h_z is not None else None

    # ---------------------------------------------------------------------
    
    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        
        dtype               = self._dtype

        rows: List[int]     = []
        cols: List[int]     = []
        data: List[float]   = []
        Ns                  = self.ns
        
        for site in range(Ns):
            num_nei = self.lattice.get_nn_num(site)
            
            # construct the single-particle Hamiltonian - nearest neighbor hopping
            for nei_idx in range(num_nei):
                nei_site    = self.lattice.get_nn(site, nei_idx)
                bond_type   = self._get_bond_type(nei_idx)
                
                if bond_type == 'x':
                    k_val   = self.k_x[site] if isinstance(self.k_x, np.ndarray) else self.k_x
                    g_val   = self.gamma_x[nei_site] if isinstance(self.gamma_x, np.ndarray) else self.gamma_x
                elif bond_type == 'y':
                    k_val   = self.k_y[site] if isinstance(self.k_y, np.ndarray) else self.k_y
                    g_val   = self.gamma_y[nei_site] if isinstance(self.gamma_y, np.ndarray) else self.gamma_y
                elif bond_type == 'z':
                    k_val   = self.k_z[site] if isinstance(self.k_z, np.ndarray) else self.k_z
                    g_val   = self.gamma_z[nei_site] if isinstance(self.gamma_z, np.ndarray) else self.gamma_z
                
                gamma_mean_field    = -1.0j * (g_val / 2.0)                             # we renormalize the gamma by 2 for mean field  
                k_mean_field        = 1.0j * (k_val) if k_val is not None else 0.0      # we renormalize the k-value by 
                
                rows.append(site)
                cols.append(nei_site)
                data.append(gamma_mean_field + k_mean_field)
            
            # TODO: add next nearest neighbor hopping from magnetic field in 3rd order perturbation theory    
            if self.h_x is not None or self.h_y is not None or self.h_z is not None:
                # get next nearest neighbors
                nn_neis = self.lattice.get_next_nn(site)
                for nn_nei_site in nn_neis:
                    # determine the bond types involved
                    bond_types = self.lattice.get_bond_types_between(site, nn_nei_site)
                    if len(bond_types) != 2:
                        continue  # skip if not exactly two bonds
                    
                    # calculate the effective hopping amplitude
                    h_eff = 0.0
                    for bond_type in bond_types:
                        if bond_type == 'x' and self.h_x is not None:
                            h_eff += self.h_x
                        elif bond_type == 'y' and self.h_y is not None:
                            h_eff += self.h_y
                        elif bond_type == 'z' and self.h_z is not None:
                            h_eff += self.h_z
                    
                    # add the next nearest neighbor hopping term
                    rows.append(site)
                    cols.append(nn_nei_site)
                    data.append(-1.0j * h_eff)  # imaginary hopping for Majorana fermions
            
            
        # Assemble & symmetrize
        if _HAS_SCIPY and not use_numpy and self._is_sparse:
            H               = coo_matrix((np.asarray(data, dtype=dtype), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))), shape=(Ns, Ns)).tocsr()
            H               = 0.5 * (H + H.T.conjugate())
            self._hamil_sp  = H.astype(dtype, copy=False)
        else:
            H               = np.zeros((Ns, Ns), dtype=dtype)
            for r, c, v in zip(rows, cols, data):
                H[r, c] += v
            H               = 0.5 * (H + H.T.conjugate())
            self._hamil_sp  = H.astype(dtype, copy=False)

        self._log("AA Hamiltonian built.", lvl=2, color='green')
            
    # ---------------------------------------------------------------------
    
    def solvable(self, **kwargs) -> np.ndarray:
        pass
    
    # ---------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return (f"KitaevGammaMajorana(lattice={self.lattice}, "
                f"gamma_z={self.gamma_z}, gamma_y={self.gamma_y}, gamma_x={self.gamma_x})")
        
    def __str__(self) -> str:
        return (f"K-G-majorana(ns={self.ns})\n")
    
    # ---------------------------------------------------------------------
    
    def _get_bond_type(self, num_nei: int) -> str:
        if num_nei == HEI_KIT_X_BOND_NEI:
            return 'x'
        elif num_nei == HEI_KIT_Y_BOND_NEI:
            return 'y'
        elif num_nei == HEI_KIT_Z_BOND_NEI:
            return 'z'
        else:
            raise ValueError(f"Invalid nearest neighbor index: {num_nei}")
    
    @property
    def bonds(self):
        return { HEI_KIT_X_BOND_NEI: 'x', HEI_KIT_Y_BOND_NEI: 'y', HEI_KIT_Z_BOND_NEI: 'z',
            'x': HEI_KIT_X_BOND_NEI, 'y': HEI_KIT_Y_BOND_NEI, 'z': HEI_KIT_Z_BOND_NEI }
    
# -------------------------------------------------------------------------
#! End of file
# -------------------------------------------------------------------------