"""
Aubry-André-Harper model for non-interacting fermions in one- and two-dimensional lattices.

The Aubry-André model describes a system of non-interacting fermions in a periodic potential, exhibiting phenomena such as localization and delocalization depending on the strength of the potential and the filling of the fermionic states.

Author: Maksymilian Kliczkowski
"""

from typing import List, Optional, Tuple, Union

import numpy as np

# import the quadratic base
from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
from QES.general_python.lattices import Lattice

try:
    from scipy.sparse import coo_matrix

    _HAS_SCIPY = True
except ImportError:
    coo_matrix = None
    _HAS_SCIPY = False

_DEFAULT_J          = 1.0
_DEFAULT_LAMBDA     = 1.0
_DEFAULT_BETA       = (np.sqrt(5.0) - 1.0) / 2.0  # inverse of golden ratio, common choice for incommensurability
_DEFAULT_PHI        = 0.0

class AubryAndre(QuadraticHamiltonian):
    '''
    Defines and Aubry-André-Harper (AAH) model on a 1D/2D/3D lattice with tunable parameters:
    - J: 
        hopping amplitude (default 1.0)
    - lambda: 
        potential strength (default 1.0)
    - beta: 
        incommensurability parameter (default inverse golden ratio)
    - phi: 
        phase offset (default 0.0)
    - lattice geometry and size (via Lattice or lx, ly, lz, pbc_x/y/z)
    
    Model is:
        H = -J * sum_{<i,j>} (c_i^† c_j + h.c.) + lambda * sum_i cos(2*pi*beta*x_i + phi) n_i  (1D)
        H = -J * sum_{<i,j>} (c_i^† c_j + h.c.) + lambda * sum_i [cos(2*pi*beta*(x_i+y_i) + phi) + cos(2*pi*beta*(x_i-y_i) + phi)] n_i / 4  (2D)
        H = -J * sum_{<i,j>} (c_i^† c_j + h.c.) + lambda * sum_i [cos(2*pi*beta*x_i + phi) + cos(2*pi*beta*y_i + phi) + cos(2*pi*beta*z_i + phi)] n_i / 3  (3D)
    '''
    

    def __init__(
        self,
        lattice     : Optional[Union[str, Lattice]] = None,
        dtype       : type = np.float64,
        backend     : str = "default",
        lmbd        : Optional[float] = _DEFAULT_LAMBDA,
        *,
        J           : Optional[float] = _DEFAULT_J,
        beta        : Optional[float] = _DEFAULT_BETA,
        phi         : float = _DEFAULT_PHI,
        lx          : Optional[int] = 1,
        ly          : Optional[int] = 1,
        lz          : Optional[int] = 1,
        pbc_x       : Optional[bool] = True,
        pbc_y       : Optional[bool] = True,
        pbc_z       : Optional[bool] = True,
        **kwargs,
    ):

        ns_guess = lattice.ns if isinstance(lattice, Lattice) else int(lx) * int(ly) * int(lz)
        super().__init__(
            ns=ns_guess, is_sparse=False, dtype=dtype, lattice=lattice, backend=backend, **kwargs
        )

        if isinstance(lattice, Lattice):
            self._lat   = lattice
            self._ns    = lattice.ns
            self._lx    = lattice.lx
            self._ly    = lattice.ly
            self._lz    = lattice.lz
            self._bc    = lattice.bc
        else:
            self._lat   = None
            self._lx    = int(lx)
            self._ly    = int(ly) if ly is not None else 1
            self._lz    = int(lz) if lz is not None else 1
            self._ns    = self._lx * self._ly * self._lz
            # Force periodic boundary conditions for the native AA lattice path.
            # This keeps the model definition consistent even if callers pass OBC.
            self._bc    = (True, True, True)

        self._name      = "AubryAndre"
        self._J         = float(J)
        self._lmbd      = float(lmbd)
        self._beta      = float(beta)
        self._phi       = float(phi)
        self._twopi     = 2.0 * np.pi

    # -----------------------------------------------------------------
    #! Hamiltonian construction
    # -----------------------------------------------------------------

    def _idx_to_xyz(self, idx: int) -> Tuple[int, int, int]:
        """Row-major: idx = x + lx*(y + ly*z)."""
        x = idx % self._lx
        y = (idx // self._lx) % self._ly
        z = idx // (self._lx * self._ly)
        return x, y, z

    def _xyz_to_idx(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinates (x, y, z) to a linear index in row-major order."""
        return x + self._lx * (y + self._ly * z)

    def _fallback_forward_neighbors(self, i: int) -> List[int]:
        """If no Lattice provided, generate forward neighbors (+x, +y, +z) respecting PBC flags."""

        if self._lat is not None:
            return self._lat.get_nei_forward(i)

        x, y, z             = self._idx_to_xyz(i)
        pbcx, pbcy, pbcz    = self._bc
        neigh: List[int]    = []

        # +x
        if x + 1 < self._lx:
            neigh.append(self._xyz_to_idx(x + 1, y, z))
        elif pbcx and self._lx > 1:
            neigh.append(self._xyz_to_idx(0, y, z))

        # +y
        if self._ly > 1:
            if y + 1 < self._ly:
                neigh.append(self._xyz_to_idx(x, y + 1, z))
            elif pbcy:
                neigh.append(self._xyz_to_idx(x, 0, z))

        # +z
        if self._lz > 1:
            if z + 1 < self._lz:
                neigh.append(self._xyz_to_idx(x, y, z + 1))
            elif pbcz:
                neigh.append(self._xyz_to_idx(x, y, 0))

        return neigh

    def _onsite_potential(self, x: int, y: int, z: int) -> float:
        """AA/Harper potential in 1D/2D/3D."""

        # For better statistics - if phi=0, we can randomize it to avoid special cases. Otherwise, use the provided phi.
        if np.isclose(self._phi, 0.0):
            phi = np.random.uniform(0, self._twopi)
        else:
            phi = self._phi

        # 1D
        if self._ly == 1 and self._lz == 1:
            return -self._J * self._lmbd * np.cos(self._twopi * self._beta * x + phi)
        
        # 2D - diagonal Harper form, matches C++ (with diag()/=4)
        elif self._lz == 1:
            v = np.cos(self._twopi * self._beta * (x + y) + phi) + np.cos(
                self._twopi * self._beta * (x - y) + phi
            )
            return -(self._lmbd * v) / 4.0 * self._J
        
        # 3D - separable sum over axes, normalized by 3 to keep scale comparable
        else:
            v = (
                np.cos(self._twopi * self._beta * x + phi)
                + np.cos(self._twopi * self._beta * y + phi)
                + np.cos(self._twopi * self._beta * z + phi)
            )
            return -(self._lmbd * v) / 3.0 * self._J

    # -----------------------------------------------------------------
    #! Hamiltonian
    # -----------------------------------------------------------------

    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        """
        Build AA(AH) Hamiltonian using lattice neighbors if available.
        Only forward neighbors are added explicitly; the backward hoppings are
        filled by Hermitian completion at the end.
        """

        self._log("Building AA Hamiltonian...", lvl=2, color="green")

        Ns      = self._ns
        J       = self._J
        dtype   = self._dtype

        rows: List[int]     = []
        cols: List[int]     = []
        data: List[float]   = []

        # On-site terms
        for i in range(Ns):
            x, y, z = self._idx_to_xyz(i)
            v_i     = self._onsite_potential(x, y, z)
            rows.append(i)
            cols.append(i)
            data.append(v_i)

        # Hopping (forward neighbors)
        if isinstance(self._lat, Lattice):

            get_fwd = self._lat.get_nei_forward
            for i in range(Ns):
                for j in get_fwd(i):
                    if j >= 0:
                        rows.append(i)
                        cols.append(j)
                        data.append(-J)
        else:
            for i in range(Ns):
                for j in self._fallback_forward_neighbors(i):
                    rows.append(i)
                    cols.append(j)
                    data.append(-J)

        # Assemble & symmetrize
        if _HAS_SCIPY and not use_numpy and self._is_sparse:
            H = coo_matrix(
                (
                    np.asarray(data, dtype=dtype),
                    (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
                ),
                shape=(Ns, Ns),
            ).tocsr()
            
            # Complete the missing backward hoppings without halving amplitudes.
            H               = H + H.T.conjugate()
            self._hamil_sp  = H.astype(dtype, copy=False)
        else:
            
            # If not using sparse, we can directly build the dense matrix and
            # complete the missing backward hoppings at the end.
            H               = np.zeros((Ns, Ns), dtype=dtype)
            for r, c, v in zip(rows, cols, data):
                H[r, c]    += v
            H               = H + H.T.conjugate()
            self._hamil_sp  = H.astype(dtype, copy=False)

        self._log("AA Hamiltonian built.", lvl=2, color="green")

    # -----------------------------------------------------------------

    def add_term(self, *args, **kwargs):
        raise NotImplementedError("Add term not implemented for AA model.")

    # -----------------------------------------------------------------

    @staticmethod
    def repr(
        lx          : int,
        ly          : int = 1,
        lz          : int = 1,
        J           : float = 1.0,
        lmbd        : float = 2.0,
        beta        : float = (np.sqrt(5.0) - 1.0) / 2.0,
        phi         : float = 0.0,
        **kwargs,
    ) -> str:
        '''
        Static method to generate a string representation of the AA model based on its parameters.
        This is useful for logging or saving results with descriptive names.
        '''
        
        if ly > 1:
            if lz > 1:
                return f"AA-3D(lx={lx},ly={ly},lz={lz},J={J},lambda={lmbd:.3f},beta={beta:.3f},phi={phi:.3f})"
            return f"AA-2D(lx={lx},ly={ly},J={J},lambda={lmbd:.3f},beta={beta:.3f},phi={phi:.3f})"
        ns = int(lx) * int(ly) * int(lz)
        return f"AA(ns={ns},J={J},lambda={lmbd:.3f},beta={beta:.3f},phi={phi:.3f})"

    def __repr__(self):
        return self.repr(lx=self._lx, ly=self._ly, lz=self._lz, J=self._J, lmbd=self._lmbd, beta=self._beta, phi=self._phi)

    def __str__(self):
        return self.__repr__()

# ---------------------------------------------------------------------
#! END OF HAMILTONIAN
# ---------------------------------------------------------------------
