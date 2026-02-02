"""
Analytic translational-invariant free-fermion model

-----------------------------------------------------
file    : QES/Algebra/Model/Noninteracting/Conserving/free_fermions.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-05-01
-----------------------------------------------------
"""

from    typing import TYPE_CHECKING, Optional, Union

import  numba
import  numpy as np

# import the quadratic base
try:
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian

    if TYPE_CHECKING:
        from QES.general_python.algebra.utils       import Array
        from QES.general_python.common.flog         import Logger
        from QES.general_python.lattices.lattice    import LatticeBC
        
except ImportError as e:
    raise ImportError(
        "Could not import QuadraticHamiltonian base class. Ensure that QES package is properly installed."
    ) from e
    
try:    
    import          jax
    import          jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    JAX_AVAILABLE   = False
        
# ---------------------------------------------------------------------
#! Spectrum
# ---------------------------------------------------------------------

@numba.njit
def _free_fermions_spectrum(ns: int, t: float, t2: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytic spectrum of the free fermion model.

    Parameters
    ----------
    ns : int
        Number of sites.
    t : float
        Hopping amplitude.

    Returns
    -------
    tuple
        Eigenvalues and eigenvectors.
    """
    k               = np.arange(ns)
    twopi_over_L    = 2.0 * np.pi / ns
    eig_val         = -2.0 * t * np.cos(twopi_over_L * k) - 2.0 * t2 * np.cos(2.0 * twopi_over_L * k)

    #! plane waves
    j               = np.arange(ns)[:, None]  # (ns,1)
    phase           = np.exp(1j * twopi_over_L * j * k) / np.sqrt(ns)
    return eig_val, phase.astype(np.complex128)

if JAX_AVAILABLE:
    
    @jax.jit
    def _free_fermions_spectrum_jax(ns: int, t: float, t2: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Analytic spectrum of the free fermion model.

        Parameters
        ----------
        ns : int
            Number of sites.
        t : float
            Hopping amplitude.

        Returns
        -------
        tuple
            Eigenvalues and eigenvectors.
        """
        k               = jnp.arange(ns)
        twopi_over_L    = 2.0 * jnp.pi / ns
        eig_val         = -2.0 * t * jnp.cos(twopi_over_L * k) - 2.0 * t2 * jnp.cos(2.0 * twopi_over_L * k)

        #! plane waves
        j               = jnp.arange(ns)[:, None]
        phase           = jnp.exp(1j * twopi_over_L * j * k) / jnp.sqrt(ns)
        return eig_val, phase.astype(jnp.complex128)

# ---------------------------------------------------------------------

class FreeFermions(QuadraticHamiltonian):
    r"""
    1D translationally invariant tight-binding chain of free fermions with periodic boundary conditions.

    The Hamiltonian is given by:

    .. math::

        H = -t_1 \sum_{\langle i, j \rangle} \left( c_i^\dagger c_j + c_j^\dagger c_i \right) + \text{constant\_offset} 
            -t_2 \sum_{\langle\langle i, j \rangle\rangle} \left( c_i^\dagger c_j + c_j^\dagger c_i \right) +
            \epsilon_1 n_1 + \epsilon_2 n_L 

    where:
        - :math:`t_1 > 0` is the nearest-neighbor hopping (coupling) amplitude,
        - :math:`t_2 > 0` is the next-nearest-neighbor hopping (coupling) amplitude,
        - :math:`\text{constant\_offset}` is an overall energy shift,
        - :math:`\epsilon_1, \epsilon_2` are on-site energies at sites 1 and L, respectively (boundary sites),
        - :math:`c_i^\dagger` and :math:`c_j` are fermionic creation and annihilation operators,
        - the sum :math:`\langle i, j \rangle` runs over nearest-neighbor pairs with periodic boundary conditions.

    The exact single-particle energy spectrum for periodic boundary conditions is:

    .. math::

        \varepsilon_k = -2t \cos\left( \frac{2\pi k}{N_s} \right), \quad k = 0, 1, \ldots, N_s-1


    .. math::

        U_{j, k} = \frac{1}{\sqrt{N_s}} \exp\left( \frac{2\pi i j k}{N_s} \right)

    where:
        - :math:`N_s` is the number of sites,
        - :math:`j` is the site index,
        - :math:`k` is the momentum index.
    """

    def __init__(
        self,
        ns              : int,
        t               : Union["Array", float] = 1.0,
        t2              : Union["Array", float] = 0.0,
        e1              : float = 0.0,
        e2              : float = 0.0,
        *,
        bc              : Optional[Union["LatticeBC", str]] = None,
        constant_offset : float = 0.0,
        dtype           : Optional[np.dtype] = None,
        backend         : str = "default",
        logger          : Optional["Logger"] = None,
        **kwargs,
    ):
        super().__init__(
            ns=ns,
            particle_conserving=True,
            dtype=dtype,
            backend=backend,
            constant_offset=constant_offset,
            particles="fermions",
            is_sparse=False,
            lattice=None,
            hilbert_space=None,
            logger=logger,
            **kwargs,
        )
        self._t                     = self._set_some_coupling(t).astype(self._dtype)
        self._t2                    = self._set_some_coupling(t2).astype(self._dtype)
        self._e1                    = self._dtype.type(e1)
        self._e2                    = self._dtype.type(e2)
        self._has_analytic_spectrum = True
        self._bc                    = bc
        
        # check if on-site energies break translational invariance
        if self._e1 != 0.0 or self._e2 != 0.0:
            self._log(
                "FreeFermions: On-site energies at boundary sites break translational invariance. "
                "Falling back to numerical diagonalization.",
                lvl = 1,
                log = "warning",
            )
            self._has_analytic_spectrum = False
        
        # check if t and t2 are uniform
        if (np.diff(self._t).max() != 0.0) or (np.diff(self._t2).max() != 0.0):
            self._log(
                "FreeFermions: Non-uniform hopping amplitudes break translational invariance. "
                "Falling back to numerical diagonalization.",
                lvl = 1,
                log = "warning",
            )
            self._has_analytic_spectrum = False
        
        # set the spectrum
        if self._has_analytic_spectrum:
            self._set_free_spectrum()

    # -----------------------------------------------------------------
    #! analytic spectrum
    # -----------------------------------------------------------------

    def _set_free_spectrum(self):
        t   = self._backend.asarray(self._t, dtype=self._dtype)
        t2  = self._backend.asarray(self._t2, dtype=self._dtype)

        if self._is_jax:
            self._eig_val, self._eig_vec = _free_fermions_spectrum_jax(self._ns, jnp.asarray(t), jnp.asarray(t2))
        else:
            self._eig_val, self._eig_vec = _free_fermions_spectrum(self._ns, t, t2)

        # If target dtype is non-complex -> keep only the real part (cast in-place, no extra copy)
        if (
            np.issubdtype(np.dtype(self._dtype), np.complexfloating)
            or self._is_jax
            and jnp.issubdtype(self._dtype, jnp.complexfloating)
        ):
            self._eig_vec = self._eig_vec.astype(self._dtype)
        else:
            # use backend.real for NumPy/JAX symmetry; cast to requested float dtype
            self._eig_vec = self._backend.real(self._eig_vec).astype(self._dtype)

    # -----------------------------------------------------------------
    #! override parent diagonalisation (nothing to diagonalise)
    # -----------------------------------------------------------------

    def diagonalize(self, verbose: bool = False, **kwargs):
        """ Set the solution to the model """
        
        if not self._has_analytic_spectrum:
            # fall back to parent diagonalization
            super().diagonalize(verbose=verbose, **kwargs)
            return
        
        if verbose:
            self._log("FreeFermions: spectrum set analytically.", lvl=2, log="debug")

        # eigenvalues/vectors already cached
        # still apply constant offset if needed
        if self._constant_offset != 0.0 and not getattr(self, "_offset_applied", False) and self._eig_val is not None:
            self._eig_val           = self._eig_val + self._constant_offset
            self._offset_applied    = True

        # parent bookkeeping
        self._calculate_av_en()

    # -----------------------------------------------------------------
    #! the quadratic builder is a no-op
    # -----------------------------------------------------------------

    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        """
        Returns the Hamiltonian in quadratic form.

        Parameters
        ----------
        use_numpy : bool, optional
            If True, use numpy instead of the backend for the Hamiltonian matrix.

        Returns
        -------
        np.ndarray
            The Hamiltonian matrix in quadratic form.
        """
        
        try:
            from QES.general_python.lattices.lattice import LatticeBC, handle_boundary_conditions
            bc_enum = handle_boundary_conditions(self._bc)
        except ImportError as e:
            raise ImportError("Could not import LatticeBC or handle_boundary_conditions. Ensure that QES package is properly installed.") from e
        
        if use_numpy:
            self._hamil_sp = np.zeros((self._ns, self._ns), dtype=self._dtype)
        else:
            self._hamil_sp = self._backend.zeros((self._ns, self._ns), dtype=self._dtype)

        for i in range(self._ns):
            if i < self._ns - 1 or bc_enum == LatticeBC.PBC:
                self._hamil_sp[i, (i + 1) % self._ns] = -self._t[i]
            if i > 0 or bc_enum == LatticeBC.PBC:
                self._hamil_sp[i, (i - 1) % self._ns] = -self._t[i]
            
        for i in range(self._ns):
            if i < self._ns - 2 or bc_enum == LatticeBC.PBC:
                self._hamil_sp[i, (i + 2) % self._ns] = -self._t2[i]
            if i > 1 or bc_enum == LatticeBC.PBC:
                self._hamil_sp[i, (i - 2) % self._ns] = -self._t2[i]
                
        # on-site energies at boundary sites
        if self._e1 != 0.0:
            self._hamil_sp[0, 0] += self._e1
        if self._e2 != 0.0:
            self._hamil_sp[-1, -1] += self._e2
            
        return self._hamil_sp

    # -----------------------------------------------------------------
    #! adding terms not allowed (would spoil analyticity)
    # -----------------------------------------------------------------

    def add_term(self, *_, **__):
        """ Adding invalidates analytic spectrum """
        self._has_analytic_spectrum = False
        super().add_term(*_, **__)

    # -----------------------------------------------------------------

    def __repr__(self):
        return f"FreeFermions(ns={self._ns},t={self._t[0]},c={self._constant_offset})"

    def __str__(self):
        return self.__repr__()

# ---------------------------------------------------------------------
#! End of file
# ---------------------------------------------------------------------
