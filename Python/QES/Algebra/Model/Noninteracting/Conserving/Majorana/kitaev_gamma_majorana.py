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
from typing import Optional, Union, List, Dict, Tuple

# import the quadratic base
try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
except ImportError:
    raise ImportError("Could not import QuadraticHamiltonian base class. "
                      "Please ensure that QES.Algebra.hamil_quadratic module is available.")

try:
    from QES.general_python.lattices.honeycomb import HoneycombLattice, X_BOND_NEI, Y_BOND_NEI, Z_BOND_NEI
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

HEI_KIT_X_BOND_NEI = X_BOND_NEI
HEI_KIT_Y_BOND_NEI = Y_BOND_NEI
HEI_KIT_Z_BOND_NEI = Z_BOND_NEI

# -----------------------------------------------------

class KitaevGammaMajorana(QuadraticHamiltonian):
    r"""
    Kitaev-Gamma Majorana model:

      H = (i/4) Σ_<ij>_\gamma [ K_\gamma u_ij  + (Γ_\gamma/2) g_ij ] c_i c_j
        + (i/4) Σ_<<jk>> t^{(2)}_{jk} c_j c_k,

    with fields supplied externally:
    - u_ij              \in {+1,-1}     (Z2 link / flux sector, Kitaev),
    - g_ij              \in {+1,0,-1}   (\Gamma mean-field bond variable),
    - t^{(2)}_{jk}      (NNN; by default \lambda u_jl u_lk from a chosen path).
    """
    
    def __init__(self, 
                lattice     : HoneycombLattice,
                gamma_z     : Union[float, np.ndarray]              = 1.0,
                gamma_y     : Optional[Union[float, np.ndarray]]    = None,
                gamma_x     : Optional[Union[float, np.ndarray]]    = None,
                # next-nearest parameters
                t2          : float                                 = 0.0,
                *args,
                # -----------
                k_z         : Optional[Union[float, np.ndarray]]    = None,
                k_y         : Optional[Union[float, np.ndarray]]    = None,
                k_x         : Optional[Union[float, np.ndarray]]    = None,
                # -----------
                h_x         : Optional[float]                       = None,
                h_y         : Optional[float]                       = None,
                h_z         : Optional[float]                       = None,
                # -----------
                p_flip      : Optional[float]                       = None,
                p_zero      : Optional[float]                       = None,
                p_plus      : Optional[float]                       = None,
                b_flip      : Optional[List[int]]                   = None,
                # -----------
                is_sparse   : bool                                  = True,
                logger      : Optional[object]                      = None,
                dtype       : Optional[np.dtype]                    = complex,
                **kwargs):
        
        super().__init__(lattice                =   lattice,
                        particle_conserving     =   True,
                        dtype                   =   dtype,
                        backend                 =   "numpy",
                        constant_offset         =   0.0,
                        particles               =   "fermions",
                        is_sparse               =   True,
                        logger                  =   logger,
                        *args,
                        **kwargs)
        
        # setups already for random couplings - can be as well constant
        self.gamma_z    = self._set_some_coupling(gamma_z).astype(float) if gamma_z is not None else None
        self.gamma_y    = self._set_some_coupling(gamma_y).astype(float) if gamma_y is not None else self.gamma_z
        self.gamma_x    = self._set_some_coupling(gamma_x).astype(float) if gamma_x is not None else self.gamma_z
        self.k_z        = self._set_some_coupling(k_z).astype(float) if k_z is not None else None
        self.k_y        = self._set_some_coupling(k_y).astype(float) if k_y is not None else None
        self.k_x        = self._set_some_coupling(k_x).astype(float) if k_x is not None else None

        # will be treated in 3rd order perturbation theory
        h_all_none     = (h_x is None) and (h_y is None) and (h_z is None)
        h_x            = h_x if h_x is not None else 1.0
        h_y            = h_y if h_y is not None else 1.0
        h_z            = h_z if h_z is not None else 1.0
        self.lambda_   = (h_x * h_y * h_z) if h_all_none else 0.0
        
        self.u_nn       : Optional[Dict[Tuple[int, int], int]]      = None    # -1/+1
        self.g_nn       : Optional[Dict[Tuple[int, int], int]]      = None    # −1/0/+1
        # NNN field: dictionary keyed by (j,k) on same sublattice
        self.t2_nnn     : Optional[Dict[Tuple[int, int], float]]    = None    # t^{(2)}_{jk}
        
        self._bonds     : List[Tuple[int, int]]                     = None    # list of bonds (i,j) - can be initialized
        self.b_flip     : Optional[List[int]]                       = b_flip  # indices to flip u_ij signs on
        # set fields if probabilities supplied
        self.set_nn_fields(p_flip=p_flip, p_zero=p_zero, p_plus=p_plus)
        

    # ---------------------------------------------------------------------

    def set_nn_fields(self,
                        u_field : Optional[Dict[Tuple[int, int], int]] = None,
                        g_field : Optional[Dict[Tuple[int, int], int]] = None,
                        *,
                        # for u_field
                        p_flip  : Optional[float] = None,
                        # for g_field
                        p_zero  : Optional[float] = None,
                        p_plus  : Optional[float] = None
                    ):
        """Install NN fields (no sampling here). Keys are directed bonds (i,j)."""
        if u_field is None:
            if p_flip is not None:
                u_field = self.build_binary_u_field(self.lattice, p_flip=p_flip)
        self.u_nn = u_field
        
        if g_field is None:
            if (p_zero is not None) and (p_plus is not None):
                g_field = self.build_ternary_g_field(self.lattice, p_plus=p_plus, p_zero=p_zero)
                
        self.g_nn = g_field

    def set_nnn_field(self, t2_field: Optional[Dict[Tuple[int, int], float]]):
        """Install NNN amplitudes t^{(2)}_{jk}. Keys are directed same-sublattice pairs (j,k)."""
        self.t2_nnn = { k : v * self.lambda_ for k, v in t2_field.items() } if t2_field is not None else None

    # ---------------------------------------------------------------------

    def calculate_bonds(self):
        """Calculate list of bonds (i,j) in the lattice."""
        self._bonds = self._lattice.calculate_bonds()
        return self._bonds

    def _handle_flips(self, b_vec: Dict[int, int]):
        """Flip u_ij on selected bonds before building Hamiltonian."""
        if self.b_flip is not None:
            self._log("Flipping u_ij on selected bonds...", color='orange')

            self._bonds = self.calculate_bonds()
        
            for idx_flip in (self.b_flip or []):
                if idx_flip < len(self._bonds):
                    i, j            = self._bonds[idx_flip]
                    b_vec[(i, j)]   = -b_vec[(i, j)]
                    self._log(f"Flipped ij on bond ({i},{j}) to {b_vec[(i, j)]}.", color='orange', lvl=2)
        return b_vec
    
    # ---------------------------------------------------------------------
    
    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        r"""
        Build iA with A real antisymmetric:
          A_ij =  [ K_\gamma u_ij + (Γ_\gamma/2) g_ij ]     for NN
          A_jk =  t^{(2)}_{jk}                              for NNN
          
        Returns
        -------
        H : np.ndarray or scipy.sparse.csr_matrix
            The quadratic Majorana Hamiltonian matrix.
        """
        
        Ns = self.ns
        
        # We'll build A (real antisymmetric), then H = 1j * A.
        if _HAS_SCIPY and self._is_sparse and not use_numpy:
            rows    : List[int]     = []
            cols    : List[int]     = []
            vals    : List[float]   = []
            add                     = lambda r, c, a: (rows.append(r), cols.append(c), vals.append(a))
        else:
            A                       = np.zeros((Ns, Ns), dtype=float)
            def add(r, c, a):
                A[r, c]            += float(a)

        def add_mat(r, c, a):
            """Add antisymmetric entry to A (A[j,i] = -A[i,j])."""
            add(r, c, +a)
            add(c, r, -a)

        u_nn = self._handle_flips(self.u_nn) if self.u_nn is not None else None
        g_nn = self._handle_flips(self.g_nn) if self.g_nn is not None else None

        # -------------------------
        #! NN contributions
        # -------------------------
                
        elems = 0
        for i in range(Ns):
            for idx in range(self.lattice.get_nn_forward_num(i)):

                j = self.lattice.get_nn_forward(i, idx)
                
                if self.lattice.wrong_nei(j):
                    continue
                
                bond = self._get_bond_type(idx)
                
                # pick couplings
                if bond == 'x':
                    Kg  = self.k_x[i] if self.k_x is not None else None
                    Gg  = self.gamma_x[i] if self.gamma_x is not None else None
                elif bond == 'y':
                    Kg  = self.k_y[i] if self.k_y is not None else None
                    Gg  = self.gamma_y[i] if self.gamma_y is not None else None
                else:
                    Kg  = self.k_z[i] if self.k_z is not None else None
                    Gg  = self.gamma_z[i] if self.gamma_z is not None else None
                
                if Kg is None: Kg = 0.0
                if Gg is None: Gg = 0.0

                u_ij    = u_nn[(i, j)] if u_nn is not None else 1           # default u_ij = +1 if not supplied
                g_ij    = g_nn[(i, j)] if g_nn is not None else 0           # default no Γ mean-field if not supplied
                # Majorana fermion hopping (no Pauli matrix scaling):
                #   H = i * sum_ij A_ij c_i c_j  with A_ij antisymmetric
                #   A_ij^NN = K_gamma * u_ij + Gamma_gamma * g_ij  (just the couplings, no factors)
                
                a_ij    = (Kg * u_ij + Gg * g_ij)                              # real, no Pauli scaling
                elems  += 1 

                if self.ns <= 24:
                    self._log(f"Adding NN bond ({i},{j}) type {bond}: K={Kg}, Γ={Gg}, u={u_ij}, g={g_ij} => A_ij={a_ij}", lvl=3, color='blue', log='info')

                if abs(a_ij) > 0:
                    add_mat(i, j, a_ij)

        self._log(f"Total NN elements added: {elems}", color='red')

        # -------------------------
        #! NNN contributions
        # -------------------------
        
        elems = 0
        if self.t2_nnn is not None:
            for (j, k), t2jk in self.t2_nnn.items():
                if j == k or (k, j) in self.t2_nnn or j >= k:
                    continue
                add_mat(j, k, float(t2jk))

        self._log(f"Total NNN elements added: {elems}", color='red')

        # -------------------------
        #! Finalize H = i A
        prefactor   = 1.0j
        # prefactor   = 0.25j
        if _HAS_SCIPY and self._is_sparse and not use_numpy:
            A_sp = coo_matrix((
                            np.asarray(vals, dtype=float),
                                (
                                    np.asarray(rows, dtype=np.int64),
                                    np.asarray(cols, dtype=np.int64)
                                )
                            ),
                            shape=(Ns, Ns)).tocsr()
            # A_sp    = (A_sp - A_sp.T) / 2.0 # enforce antisymmetry once
            H       = (prefactor * A_sp).astype(self._dtype, copy=False)
        else:
            # A       = (A - A.T) / 2.0       # enforce antisymmetry once
            H       = (prefactor * A).astype(self._dtype, copy=False)

        self._hamil_sp = H
        self._log("Majorana quadratic Hamiltonian built (iA with A real antisymmetric).", lvl=2, color='green')
        return self._hamil_sp
            
    # ---------------------------------------------------------------------
    #! STATIC METHODS TO BUILD RANDOM FIELDS
    # ---------------------------------------------------------------------
    
    @staticmethod
    def build_binary_u_field(lattice    : HoneycombLattice,
                            p_flip      : float = 0.0,
                            rng         : Optional[np.random.Generator] = None) -> Dict[Tuple[int, int], int]:
        r"""
        Build u_{ij} \in \{+1,-1\} on NN bonds with flip probability p_flip (flux density proxy).
        Returns oriented dictionary: u[(i,j)] = sign on i->j; only one orientation is stored.
        
        Parameters
        ----------
        lattice : HoneycombLattice
            The honeycomb lattice on which to build the u field.
        p_flip : float, optional
            Probability of flipping the sign of u on each bond, by default 0.0.
        rng : np.random.Generator, optional
            Random number generator to use, by default None (will create a new one).
        """
        
        rng     = rng or np.random.default_rng()
        u       = {}
        Ns      = lattice.ns
        for i in range(Ns):
            for idx in range(lattice.get_nn_forward_num(i)):

                j = lattice.get_nn_forward(i, idx)

                if lattice.wrong_nei(j) or (j, i) in u:
                    continue
                                
                s           = -1 if rng.random() < p_flip else 1
                # Store a single orientation; the matrix builder enforces A[j,i] = -A[i,j].
                u[(i, j)]   = s
        return u

    @staticmethod
    def build_ternary_g_field(lattice   : HoneycombLattice,
                              p_plus    : float = 1.0,
                              p_zero    : float = 0.0,
                              rng       : Optional[np.random.Generator] = None) -> Dict[Tuple[int, int], int]:
        r"""
        Build g_{ij} \in \{+1,0,-1\} on NN bonds with given probabilities.
        Returns antisymmetric directed dictionary g[(i,j)] = -g[(j,i)].
        
        Parameters
        ----------
        lattice : HoneycombLattice
            The honeycomb lattice on which to build the g field.
        p_plus : float, optional
            Probability of g_{ij} = +1, by default 1.0.
        p_zero : float, optional
            Probability of g_{ij} = 0, by default 0.0.
        rng : np.random.Generator, optional
            Random number generator to use, by default None (will create a new one).
        """
        rng     = rng or np.random.default_rng()
        p_minus = 1.0 - p_plus - p_zero
        if p_minus < -1e-12:
            raise ValueError("Probabilities must sum to ≤ 1.")
        g       = {}
        Ns      = lattice.ns
        for i in range(Ns):
            for idx in range(lattice.get_nn_num(i)):
                
                j = lattice.get_nn(i, idx)
                if (j, i) in g:
                    continue
                r = rng.random()
                if r < p_minus:
                    s = -1
                elif r < p_minus + p_zero:
                    s = 0
                else:
                    s = +1
                g[(i, j)] = s
                g[(j, i)] = -s
        return g

    @staticmethod
    def build_t2_from_u(lattice         : HoneycombLattice,
                        u_field         : Dict[Tuple[int, int], int],
                        use_gaugeprod   : bool = True) -> Dict[Tuple[int, int], float]:
        """
        Gauge-covariant NNN builder:  
            t^{(2)}_{jk} = λ * u_{j l} * u_{l k}
        where l is the intermediate site along the chosen inside-hexagon path.
        
        If `use_gauge_product=False`, 
            returns uniform chirality t^{(2)}_{jk} = λ * ν_{jk}.
            
        Parameters
        ----------
        lattice : HoneycombLattice
            The honeycomb lattice on which to build the t2 field.
        u_field : Dict[Tuple[int, int], int]
            The NN u field to use for gauge-covariant construction.
        use_gaugeprod : bool, optional
            Whether to use gauge-covariant construction, by default True.
        """
        t2 = {}
        Ns = lattice.ns
        
        for j in range(Ns):
            for idx in range(lattice.get_nnn_num(j)):
                
                k = lattice.get_nnn(j, idx)
                if (k, j) in t2:
                    continue
                
                if use_gaugeprod:
                    # pick one path j-l-k inside the hexagon with fixed orientation (e.g., anticlockwise)
                    mids: List[int] = lattice.get_nnn_middle_sites(j, k, orientation="anticlockwise")
                    if not mids:
                        continue
                    l = mids[0]
                    val = u_field[(j, l)] * u_field[(l, k)]
                else:
                    nu  = lattice.get_chirality_sign(j, k)  # +/ 1 from orientation
                    val = nu
                t2[(j, k)] = val
                t2[(k, j)] = -val  # antisymmetric convention for directed pair
        return t2
                
    # ---------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return (f"KitaevGammaMajorana(lattice={self.lattice}, "
                f"gamma_z={self.gamma_z}, gamma_y={self.gamma_y}, gamma_x={self.gamma_x})")
        
    def __str__(self) -> str:
        return (f"K-G-majorana(ns={self.ns})")
    
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
    
    # ---------------------------------------------------------------------
    #! Exact solutions
    # ---------------------------------------------------------------------
    
    def kitaev_energy(self, k_vec) -> tuple[float, float]:
        """
        Pure Kitaev-model dispersion:
            E_pm(k) = |f(k)|
            f(k) = Kx * exp(i k\cdot dx) + Ky * exp(i k\cdot dy) + Kz * exp(i k\cdot dz)
        """
        # bond displacements (A -> B)
        
        dx = self.lattice._delta_x
        dy = self.lattice._delta_y
        dz = self.lattice._delta_z

        # Use the SAME couplings that were used to build H_real:
        Kx = float(self.k_x[0])
        Ky = float(self.k_y[0])
        Kz = float(self.k_z[0])

        # Match Bloch convention: kspace_from_realspace uses exp(-i k\cdot r)
        f_k = (
            Kx * np.exp(-1j * np.dot(k_vec, dx)) +
            Ky * np.exp(-1j * np.dot(k_vec, dy)) +
            Kz * np.exp(-1j * np.dot(k_vec, dz))
        )

        Ek = np.abs(f_k)
        return +Ek, -Ek

    
# -------------------------------------------------------------------------
#! End of file
# -------------------------------------------------------------------------