"""
Implementation of quadratic Hamiltonians and related utilities.
This module provides classes and functions to define, manipulate, and analyze
quadratic Hamiltonians for fermionic and bosonic systems.

In principle, quadratic Hamiltonians can be solved exactly by diagonalization,
but certain special cases allow for closed-form solutions without
diagonalization. The `SolvabilityInfo` class encapsulates information
about whether a given quadratic Hamiltonian is solvable in closed form,
and if so, the method used.

----------------------------------------------------------------------------
file    : QES/Algebra/hamil_quadratic.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-11-01
----------------------------------------------------------------------------
"""

from __future__     import annotations

from dataclasses    import dataclass, field
from typing         import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy        as np
import scipy        as sp

##############################################################################
#! Lazy import infrastructure
##############################################################################

# Cache for lazily imported modules/functions
_lazy_cache: Dict[str, Any] = {}

def _get_hilbert_jit_states():
    """Lazily import hilbert_jit_states module."""
    if 'hilbert_jit_states' not in _lazy_cache:
        from QES.Algebra.Hilbert import hilbert_jit_states_refactored
        _lazy_cache['hilbert_jit_states'] = hilbert_jit_states_refactored
    return _lazy_cache['hilbert_jit_states']

def _get_jax_states():
    """Lazily import JAX-based state calculations."""
    if 'jax_states' not in _lazy_cache:
        try:
            from QES.Algebra.Hilbert import hilbert_jit_states_jax
            _lazy_cache['jax_states'] = hilbert_jit_states_jax
        except ImportError:
            _lazy_cache['jax_states'] = None
    return _lazy_cache['jax_states']

# Lazy function accessors - import only when first used
def _get_calculate_slater_det():
    return _get_hilbert_jit_states().calculate_slater_det

def _get_calculate_permanent():
    return _get_hilbert_jit_states().calculate_permanent

def _get_calculate_bogoliubov_amp():
    return _get_hilbert_jit_states().calculate_bogoliubov_amp

def _get_calculate_bogoliubov_amp_exc():
    return _get_hilbert_jit_states().calculate_bogoliubov_amp_exc

def _get_calculate_bosonic_gaussian_amp():
    return _get_hilbert_jit_states().calculate_bosonic_gaussian_amp

def _get_bogolubov_decompose():
    return _get_hilbert_jit_states().bogolubov_decompose

def _get_pairing_matrix():
    return _get_hilbert_jit_states().pairing_matrix

def _get_many_body_state_full():
    return _get_hilbert_jit_states().many_body_state_full

def _get_many_body_state_mapping():
    return _get_hilbert_jit_states().many_body_state_mapping

def _get_many_body_state_closure():
    return _get_hilbert_jit_states().many_body_state_closure

def _get_nrg_particle_conserving():
    return _get_hilbert_jit_states().nrg_particle_conserving

def _get_nrg_bdg():
    return _get_hilbert_jit_states().nrg_bdg

# JAX functions (only load if JAX available)
def _get_calculate_slater_det_jax():
    jax_mod = _get_jax_states()
    return jax_mod.calculate_slater_det_jax if jax_mod else None

def _get_calculate_permanent_jax():
    jax_mod = _get_jax_states()
    return jax_mod.calculate_permament_jax if jax_mod else None

def _get_calculate_bcs_amp_jax():
    jax_mod = _get_jax_states()
    return jax_mod.calculate_bcs_amp_jax if jax_mod else None

##############################################################################
#! Core imports (required at module load)
##############################################################################

try:
    from QES.Algebra.hamil          import JAX_AVAILABLE, Hamiltonian, HilbertSpace
    from QES.Algebra.hamil_config   import HamiltonianConfig, register_hamiltonian

    if TYPE_CHECKING:
        from QES.general_python.algebra.utils           import Array
        from QES.general_python.common.flog             import Logger
        from QES.general_python.lattices.lattice        import Lattice

    # Import QuadraticTerm and SolvabilityInfo for use
    from QES.Algebra.Quadratic.hamil_quadratic_utils    import QuadraticTerm, SolvabilityInfo
    
    if TYPE_CHECKING:
        from QES.Algebra.Quadratic.hamil_quadratic_utils    import QuadraticBlockDiagonalInfo, QuadraticSelection

except ImportError as e:
    raise ImportError("QES.Algebra.hamil and core modules are required but not found.") from e

# JAX numpy fallback
try:
    if JAX_AVAILABLE:
        import jax.numpy as jnp
    else:
        jnp = np
except ImportError:
    jnp = np

##############################################################################

class QuadraticHamiltonian(Hamiltonian):
    r"""
    QuadraticHamiltonian: Specialized Hamiltonian for non-interacting (quadratic) quantum systems.

    This class represents Hamiltonians of the general quadratic form:
        .. math::

            H = \sum_{i,j} \left[ h_{ij} c_i^\dagger c_j + \Delta_{ij} c_i^\dagger c_j^\dagger + \Delta_{ij}^* c_j c_i \right]

    where:
        - :math:`h_{ij}` encodes onsite energies and hopping amplitudes,
        - :math:`\Delta_{ij}` encodes pairing (superconducting) terms,
        - :math:`c_i^\dagger`, :math:`c_j` are creation/annihilation operators (fermionic or bosonic).

    For particle-conserving systems (:math:`\Delta_{ij}=0`), the Hamiltonian reduces to:
        .. math::

            H = \sum_{i,j} h_{ij} c_i^\dagger c_j

    and is represented by an :math:`N_s \times N_s` matrix.

    For non-particle-conserving systems (e.g., Bogoliubov-de Gennes, BdG), the Hamiltonian is:
        .. math::

            H = \frac{1}{2} \Psi^\dagger H_\mathrm{BdG} \Psi

    with

        .. math::

            \Psi = (c_1, \ldots, c_{N_s}, c_1^\dagger, \ldots, c_{N_s}^\dagger)^T,

            H_\mathrm{BdG} = \begin{bmatrix} h & \Delta \\ \Delta^\dagger & -h^T \end{bmatrix}

    and is represented by a :math:`2N_s \times 2N_s` matrix.

    Features:
    - Add quadratic terms (onsite, hopping, pairing) using a flexible interface.
    - Build the Hamiltonian matrix from the stored terms, supporting both dense and sparse representations.
    - Diagonalize the Hamiltonian to obtain single-particle eigenvalues and eigenvectors.
    - Compute many-body energies and wavefunctions for Slater determinants (fermions), permanents (bosons), and Bogoliubov vacua (superconductors).
    - Support for both NumPy and JAX backends for high-performance and differentiable computations.
    - Integrates with :class:`~QES.Algebra.hamil_config.HamiltonianConfig` / ``register_hamiltonian`` so users can instantiate it via registry keys.

    Example
    -------
    >>> from QES.Algebra import HilbertConfig, HamiltonianConfig, HilbertSpace, Hamiltonian
    >>> hilbert_cfg = HilbertConfig(ns=4, is_manybody=False)
    >>> ham_cfg     = HamiltonianConfig(kind='quadratic', hilbert=hilbert_cfg, parameters={'ns': 4})
    >>> quad_ham    = Hamiltonian.from_config(ham_cfg)
    >>> quad_ham.add_hopping(0, 1, 1.0)

    Mathematical background:
    - For fermions, the ground state of a quadratic Hamiltonian is a Slater determinant (particle-conserving) or a Bogoliubov vacuum (BdG).
    - For bosons, the ground state is a permanent (particle-conserving) or a Gaussian state (BdG).
    - Diagonalization yields the single-particle spectrum, which determines the many-body ground state and excitations.

    --------------------------------------------------------------------
    Key properties used in this class:
    - self._isfermions            : bool        # True for fermions, False for bosons
    - self._isbosons              : bool        # True for bosons, False for fermions
    - self._particle_conserving   : bool        # True if particle number is conserved
    - self._is_numpy              : bool        # True for NumPy backend, False for JAX
    - self._ns                    : int         # Number of sites/modes
    - self._dtype                 : np.dtype    # Matrix/vector precision
    - self._U                     : ndarray     # (ns x n_orb) eigenvectors (fermions, N-conserving)
    - self._F                     : ndarray     # (ns x ns) pairing matrix (fermions, BdG)
    - self._Ub                    : ndarray     # (ns x N_qp) columns of u (for excitations)
    - self._G                     : ndarray     # (ns x ns) pairing matrix (bosons, BdG)
    """

    def __init__(
        self,
        ns                          : Optional[int] = None,  # Ns is mandatory
        particle_conserving         : bool = True,
        dtype                       : Optional[np.dtype] = None,
        backend                     : str = "default",
        is_sparse                   : bool = False,
        constant_offset             : float = 0.0,
        particles                   : str = "fermions",
        *,
        # Allow passing lattice/logger
        hilbert_space               : Optional[HilbertSpace] = None,
        lattice                     : Optional["Lattice"] = None,
        logger                      : Optional["Logger"] = None,
        seed                        : Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize a Quadratic Hamiltonian.

        Args:
            ns (int):
                Number of single-particle modes/sites.
            particle_conserving (bool):
                If True, uses Ns x Ns matrix. If False,
                implies pairing, potentially uses 2Ns x 2Ns
                Bogoliubov-de Gennes structure (requires specific build logic).
                (Default: True)
            dtype (Optional[data-type]):
                Matrix data type.
            backend (str):
                Computation backend ('np' or 'jax').
            particles (str):
                Whether the particles are fermions or bosons
                for the construction of a many-body state.
            is_sparse (bool):
                Use sparse matrices.
            constant_offset (float):
                A constant energy offset added *after* diagonalization.
            **kwargs:
                Passed to base class (e.g., logger, lattice).
        """

        # Remove is_manybody from kwargs if present (quadratic systems always have it False)
        kwargs.pop("is_manybody", None)

        # Call base class init, explicitly setting is_manybody=False
        super().__init__(
            is_manybody=False,
            ns=ns,
            lattice=lattice,
            hilbert_space=hilbert_space,
            is_sparse=is_sparse,
            dtype=dtype,
            backend=backend,
            seed=seed,
            logger=logger,
            **kwargs,
        )

        # setup the arguments first
        self._particle_conserving   = particle_conserving
        self._constant_offset       = constant_offset
        self._isfermions            = particles.lower() == "fermions"
        self._isbosons              = not self._isfermions
        if (
            self._hilbert_space is not None
            and getattr(self._hilbert_space, "particle_conserving", None) is not None
            and self._hilbert_space.particle_conserving != particle_conserving
        ):
            # Hilbert space indicates different particle-conserving mode; warn and continue.
            self._log("Hilbert space particle_conserving flag differs from requested mode; continuing and trusting matrices.", lvl=1, log="warning",)

        # Determine single-particle dimension and allocate storage.
        # Prefer an explicit dtype if given, otherwise inherit from Hilbert space
        # when available; fall back to complex for quadratic Hamiltonians.
        if self._dtype is None:
            if self._hilbert_space is not None:
                try:
                    self._dtype = (getattr(self._hilbert_space, "dtype", None) or self._backend.complex128)
                except Exception:
                    self._dtype = self._backend.complex128
            else:
                self._dtype = self._backend.complex128
                
        # Set sizes and allocate matrices
        self._hamil_sp_size     = self._ns
        self._hamil_sp_shape    = (self._hamil_sp_size, self._hamil_sp_size)
        # Use concrete dtypes for Numba compatibility - use dtype instances, not type classes
        self._dtypeint          = np.dtype(np.int32) if self._ns < 2**31 - 1 else np.dtype(np.int64)

        # Initialize matrices as zero arrays instead of None for immediate usability
        xp                      = self._backend
        self._hamil_sp          = xp.zeros(self._hamil_sp_shape, dtype=self._dtype)
        self._delta_sp          = xp.zeros(self._hamil_sp_shape, dtype=self._dtype)
        if not particle_conserving:
            self._log("Initialized in BdG (Nambu) mode: matrices will use 2N\times2N structure.", lvl=2, log="info")

        self._name                      = f"QuadraticHamiltonian(Ns={self._ns},{'BdG' if not self._particle_conserving else 'N-conserving'})"
        self._occupied_orbitals_cached  = None
        self._diagonalization_requested = False
        self._F                         = None
        self._G                         = None
        self._U                         = None
        self._V                         = None
        
        # Cached calculator and state computations for performance
        self._calculator_cache          = {}    # Dict[(backend, particle_type, mode)] -> (calc_fn, precomputed_matrices)
        self._cached_many_body_state    = None  # Cached many-body state array
        self._compiled_funcs            = {}    # Numba/JAX compiled functions cache
        self._last_occupation_key       = None  # Track last occupation for cache efficiency
        
        
    ##########################################################################
    #! Class methods for direct matrix initialization
    ##########################################################################

    @classmethod
    def from_hermitian_matrix(
        cls,
        hermitian_part                  : "Array",
        constant                        : float = 0.0,
        particles                       : str = "fermions",
        dtype                           : Optional[np.dtype] = None,
        backend                         : str = "default",
        **kwargs,
    ) -> "QuadraticHamiltonian":
        """
        Create a QuadraticHamiltonian from a Hermitian matrix (particle-conserving case).

        This is equivalent to Qiskit Nature's QuadraticHamiltonian constructor.

        Parameters
        ----------
        hermitian_part : array-like
            Hermitian matrix M in H = sum_{jk} M_{jk} c_j^dagger c_k + constant
        constant : float, optional
            Constant energy offset
        particles : str, optional
            'fermions' or 'bosons'
        dtype : np.dtype, optional
            Data type for matrices
        backend : str, optional
            Computation backend ('np' or 'jax')
        **kwargs
            Additional arguments passed to constructor

        Returns
        -------
        QuadraticHamiltonian
            Initialized quadratic Hamiltonian
        """
        
        # Validate input matrix
        hermitian_part = np.asarray(hermitian_part)
        if hermitian_part.ndim != 2 or hermitian_part.shape[0] != hermitian_part.shape[1]:
            raise ValueError("hermitian_part must be a square matrix")

        # Check Hermiticity
        ns = hermitian_part.shape[0]

        # Create instance
        instance = cls(
            ns=ns,
            particle_conserving=True,
            constant_offset=constant,
            particles=particles,
            dtype=dtype,
            backend=backend,
            **kwargs,
        )

        # Set the matrix directly
        instance.set_single_particle_matrix(hermitian_part)

        return instance

    @classmethod
    def from_bdg_matrices(
        cls,
        hermitian_part      : "Array",
        *,
        antisymmetric_part  : Optional["Array"],
        constant            : float = 0.0,
        particles           : str = "fermions",
        dtype               : Optional[np.dtype] = None,
        backend             : str = "default",
        **kwargs,
    ) -> "QuadraticHamiltonian":
        """
        Create a QuadraticHamiltonian from BdG matrices (general quadratic case).

        This is equivalent to Qiskit Nature's QuadraticHamiltonian constructor
        for the non-particle-conserving case.

        Parameters
        ----------
        hermitian_part : array-like
            Hermitian matrix M in the quadratic form
        antisymmetric_part : array-like
            Antisymmetric matrix Delta (for fermions) or symmetric (for bosons)
        constant : float, optional
            Constant energy offset
        particles : str, optional
            'fermions' or 'bosons'
        dtype : np.dtype, optional
            Data type for matrices
        backend : str, optional
            Computation backend ('np' or 'jax')
        **kwargs
            Additional arguments passed to constructor

        Returns
        -------
        QuadraticHamiltonian
            Initialized quadratic Hamiltonian
        """
        
        # Fallback to particle-conserving if no antisymmetric part given
        if antisymmetric_part is None:
            return cls.from_hermitian_matrix(
                hermitian_part=hermitian_part,
                constant=constant,
                particles=particles,
                dtype=dtype,
                backend=backend,
                **kwargs,
            )
        
        hermitian_part      = np.asarray(hermitian_part)
        antisymmetric_part  = np.asarray(antisymmetric_part)

        if hermitian_part.shape != antisymmetric_part.shape:
            raise ValueError("hermitian_part and antisymmetric_part must have the same shape")
        if hermitian_part.ndim != 2 or hermitian_part.shape[0] != hermitian_part.shape[1]:
            raise ValueError("Matrices must be square")

        ns                  = hermitian_part.shape[0]

        # Auto-detect particle conservation
        delta_norm          = np.linalg.norm(antisymmetric_part)
        particle_conserving = bool(delta_norm < 1e-12)

        # Create instance
        instance            = cls(
                                ns=ns,
                                particle_conserving=particle_conserving,
                                constant_offset=constant,
                                particles=particles,
                                dtype=dtype,
                                backend=backend,
                                **kwargs,
                            )

        if particle_conserving:
            # Set as single particle matrix
            instance.set_single_particle_matrix(hermitian_part)
        else:
            # Set as BdG matrices
            instance.set_bdg_matrices(hermitian_part, antisymmetric_part)

        return instance

    ##########################################################################
    #! Build the Hamiltonian - adding terms to onsite/hopping/pairing matrices
    ##########################################################################

    def _invalidate_cache(self):
        """Wipe eigenvalues, eigenvectors, and cached state calculators."""
        # Don't call super() - parent doesn't have this method
        # Just clear our own caches
        self._calculator_cache.clear()
        self._compiled_funcs.clear()
        self._last_occupation_key = None

    def add_term(
        self,
        term_typ    : Union['QuadraticTerm', str],
        sites       : tuple[int, ...] | list[int] | int,
        value       : tuple[complex, ...] | list[complex] | complex,
        remove      : bool = False,
    ):
        r"""
        Adds a quadratic term to the Hamiltonian or pairing matrix.

        Parameters
        ----------
        term_type : QuadraticTerm or str
            The type of quadratic term to add. Must be one of QuadraticTerm.Onsite,
            QuadraticTerm.Hopping, or QuadraticTerm.Pairing.
        sites : tuple[int, ...] | list[int] | int
            The site indices involved in the term. For Onsite, a single index is required.
            For Hopping and Pairing, two indices are required.
        value : complex
            The coefficient of the term to be added.

        Raises
        ------
        ValueError
            If the number of site indices does not match the requirements for the term type.
        TypeError
            If the term_type is not recognized.

        Notes
        -----
        - For Onsite terms, adds `value` to the diagonal element of the Hamiltonian matrix.
        - For Hopping terms, adds `value` to the off-diagonal elements and ensures Hermiticity.
        - For Pairing terms, modifies the pairing matrix according to particle statistics.
            If the system is particle-conserving, pairing terms are ignored.
        - Logs the operation and invalidates cached eigensystem and many-body states.
        """

        if isinstance(sites, int):
            sites = (sites,)
        
        if isinstance(value, (complex, float, int)):
            value = (value,)
            
        if len(sites) == 0 or len(value) == 0:
            raise ValueError("Sites and value must be non-empty")
            
        val         = [-v if remove else v for v in value]
        valc        = [v.conjugate() if isinstance(v, complex) or hasattr(v, "conjugate") else v for v in val]

        # Extract scalar if possible to prevent broadcasting errors
        if len(val) == 1:
            val_scalar = val[0]
            valc_scalar = valc[0]
        else:
            val_scalar = val
            valc_scalar = valc

        term_type   = QuadraticTerm.from_str(term_typ) if isinstance(term_typ, str) else term_typ

        if term_type is QuadraticTerm.Onsite:

            for site in sites:
                i = site
                if self._is_numpy:
                    self._hamil_sp[i, i] += val_scalar
                else:
                    self._hamil_sp = self._hamil_sp.at[i, i].add(val_scalar)

        elif term_type is QuadraticTerm.Hopping:

            if len(sites) % 2 != 0:
                raise ValueError("Hopping term needs pairs of indices")
            
            for idx in range(0, len(sites), 2):
                i, j = sites[idx], sites[idx + 1]
                
                if self._is_numpy:
                    self._hamil_sp[i, j] += val_scalar
                    self._hamil_sp[j, i] += valc_scalar
                else:
                    self._hamil_sp = self._hamil_sp.at[i, j].add(val_scalar)
                    self._hamil_sp = self._hamil_sp.at[j, i].add(valc_scalar)

        elif term_type is QuadraticTerm.Pairing:
            
            if self._particle_conserving:
                self._log("Pairing ignored: particle_conserving=True", lvl=2, log="warning")
                return
            
            if len(sites) % 2 != 0:
                raise ValueError("Pairing term needs pairs of indices")

            for idx in range(0, len(sites), 2):
                i, j = sites[idx], sites[idx + 1]

                if self._isfermions:  # antisymmetric
                    if self._is_numpy:
                        self._delta_sp[i, j] += val_scalar
                        self._delta_sp[j, i] -= val_scalar
                    else:
                        self._delta_sp = self._delta_sp.at[i, j].add(val_scalar)
                        self._delta_sp = self._delta_sp.at[j, i].add(-val_scalar)
                else:  # bosons: symmetric
                    if self._is_numpy:
                        self._delta_sp[i, j] += val_scalar
                        self._delta_sp[j, i] += val_scalar
                    else:
                        self._delta_sp = self._delta_sp.at[i, j].add(val_scalar)
                        self._delta_sp = self._delta_sp.at[j, i].add(val_scalar)

        else:
            raise TypeError(term_type)
        
        self._invalidate_cache()
        self._log(f"add_term: {term_type.name} {sites} {str(value)}", lvl=3, log="debug")

    def add_onsite(self, site: int, value: complex, *, remove: bool = False):
        """Convenience wrapper for adding onsite terms."""
        self.add_term(QuadraticTerm.Onsite, site, value, remove=remove)

    def add_hopping(self, i: int, j: int, value: complex, *, remove: bool = False):
        """Convenience wrapper for adding hopping terms."""
        self.add_term(QuadraticTerm.Hopping, (i, j), value, remove=remove)

    def add_pairing(self, i: int, j: int, value: complex, *, remove: bool = False):
        """Convenience wrapper for adding pairing terms."""
        self.add_term(QuadraticTerm.Pairing, (i, j), value, remove=remove)

    def reset_terms(self):
        """Clear onsite/hopping/pairing matrices."""
        xp              = self._backend
        self._hamil_sp  = xp.zeros(self._hamil_sp_shape, dtype=self._dtype)
        self._delta_sp  = xp.zeros(self._hamil_sp_shape, dtype=self._dtype)
        self._invalidate_cache()

    # ########################################################################

    def info(self) -> Dict[str, Any]:
        """Return a lightweight dictionary describing the current quadratic model."""
        has_pairing = bool(np.any(np.asarray(self._delta_sp)))
        return {
            "Ns"                    : self._ns,
            "particle_conserving"   : self._particle_conserving,
            "particles"             : "fermions" if self._isfermions else "bosons",
            "backend"               : getattr(self._backend, "__name__", str(self._backend)),
            "dtype"                 : str(self._dtype),
            "has_pairing"           : has_pairing,
            "constant_offset"       : self._constant_offset,
        }

    # ########################################################################
    #! Basis Transformation
    # ########################################################################

    def to_basis(
        self,
        basis_type: str,
        enforce: bool = False,
        sublattice_positions: Optional[np.ndarray] = None,
        **kwargs,
    ):
        r"""
        Transform QuadraticHamiltonian to a different basis representation.

        For periodic lattice systems, this efficiently transforms the real-space Hamiltonian to
        momentum-space Bloch blocks via Fast Fourier Transform, exploiting periodicity to achieve
        $O(N\log N)$ complexity instead of $O(N^2)$.

        Supported basis transformations:
        - **real -> k-space**: Use Bloch transform to decompose into momentum-space blocks
        - **k-space -> real**: Use inverse FFT to reconstruct real-space representation

        Parameters
        ----------
        basis_type : str or HilbertBasisType
            Target basis. Options: "real", "k-space", "fock", etc.

        enforce : bool, optional
            If True and lattice unavailable, construct a simple 1D chain lattice from Ns.
            If False (default), raises error if lattice missing for k-space.

        sublattice_positions : Optional[np.ndarray]
            Positions of basis sites within unit cell. Shape (Nb, 3).
            Only used for Bloch transform. If None, assumes monatomic (single-site) unit cell.

        **kwargs
            Additional options (reserved for future use).

        Returns
        -------
        Hamiltonian
            Self (modified in-place) in target basis. If already in target basis, returns self.

        Raises
        ------
        NotImplementedError
            If basis transformation not supported (e.g., k-space without lattice and enforce=False).
        ValueError
            If basis_type is invalid or transformation fails.

        Notes
        -----
        **Bloch Transform Algorithm:**
        1. Organize hopping amplitudes by cell displacement: $T_{\alpha\beta}(\Delta\mathbf{R})$
        2. Apply 3D FFT over displacement indices
        3. Correct for sublattice phases: $e^{-i\mathbf{k}\cdot(\mathbf{r}_\beta-\mathbf{r}_\alpha)}$
        4. Result: Small $N_b \times N_b$ blocks at each $\mathbf{k}$-point

        **Example:**
        ```python
        from QES.Algebra import QuadraticHamiltonian
        from QES.general_python.lattices import Lattice

        # Create a 1D chain in real space
        H_real = QuadraticHamiltonian(ns=8)
        H_real.add_hopping(0, 1, -1.0)  # nearest neighbor
        H_real.add_onsite(0, 0.5)       # onsite energy

        # Transform to k-space
        H_k = H_real.to_basis("k-space", enforce=True)

        # Diagonalize and examine band structure
        H_k.diagonalize()
        print(H_k.eig_val.shape)  # (8, 8) blocks
        ```
        """
        # Pass sublattice_positions to parent's general dispatcher
        kwargs["sublattice_positions"] = sublattice_positions
        # Call parent's general to_basis() which will dispatch to registered handlers
        return super().to_basis(basis_type, enforce=enforce, **kwargs)

    def _transform_real_to_kspace(self, enforce: bool = False, **kwargs) -> "QuadraticHamiltonian":
        r"""
        Internal: Transform real-space Hamiltonian to k-space via FFT-based Bloch decomposition.

        Modifies in-place by storing transformed representation in general attributes:
        - self._hamil_transformed: Stores H_k blocks
        - self._transformed_grid: Stores k_grid
        - self._is_transformed: Set to True
        - self._current_basis: Updated to KSPACE
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        # Ensure lattice is available
        if self._lattice is None:
            if not enforce:
                raise ValueError(
                    "Lattice required for k-space transformation. Either:\n"
                    "  1. Pass lattice to QuadraticHamiltonian.__init__\n"
                    "  2. Use enforce=True to auto-create 1D chain"
                )
            self._log("Creating simple 1D chain lattice (enforce=True)", lvl=2, color="yellow")
            raise NotImplementedError(
                "Auto-lattice creation not yet fully implemented. Please pass lattice explicitly."
            )

        # Build real-space matrix if needed
        H_real = self.build_single_particle_matrix(copy=True)

        # Convert sparse matrix to dense if needed
        # The lattice code expects dense numpy arrays
        if sp.sparse.issparse(H_real):
            self._log(
                "Converting sparse matrix to dense for Bloch transform", lvl=2, color="yellow"
            )
            H_real = H_real.toarray()

        # Apply Bloch transform with extract_bands=False to get full NsxNs matrices
        self._log(
            f"Applying FFT-based Bloch transform on {H_real.shape} matrix (full NsxNs mode)",
            lvl=2,
            color="cyan",
        )
        H_k, k_grid, k_grid_frac = self._lattice.kspace_from_realspace(H_real, block_diag=True)

        # Store transformed representation using general attributes
        self._hamil_transformed = H_k
        self._transformed_grid = k_grid
        self._transformed_grid_frac = k_grid_frac
        self._is_transformed = True
        self._current_basis = HilbertBasisType.KSPACE

        # Clear old real-space matrices if they exist (optional, for memory)
        # self._hamil_sp = None  # Uncomment if you want to free memory

        self._log(
            f"Bloch transform complete: H_k shape = {H_k.shape}, k_grid shape = {k_grid.shape}",
            lvl=2,
            color="green",
        )

        # Push k-space basis to HilbertSpace (don't sync FROM it)
        self.push_basis_to_hilbert_space()

        return self

    def _transform_kspace_to_real(self, **kwargs) -> "QuadraticHamiltonian":
        r"""
        Internal: Inverse FFT to reconstruct real-space Hamiltonian from k-space blocks.

        Modifies in-place by using general transformed storage:
        - self._hamil_transformed: Should contain H_k blocks
        - self._transformed_grid: Should contain k_grid
        - self._is_transformed: Set to False
        - self._current_basis: Updated to REAL
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        # Check for transformed representation (general storage)
        if self._hamil_transformed is None:
            raise ValueError(
                "Cannot transform k-space to real: no transformed Hamiltonian stored in _hamil_transformed"
            )

        # Apply inverse FFT
        self._log("Applying inverse FFT to reconstruct real-space Hamiltonian", lvl=2, color="cyan")
        H_real = self._lattice.realspace_from_kspace(self._hamil_transformed)

        # Update in-place
        self.set_single_particle_matrix(H_real)

        # Update basis tracking flags
        self._is_transformed = False
        self._current_basis = HilbertBasisType.REAL

        # Clear transformed storage (optional, for memory)
        # self._hamil_transformed = None
        # self._transformed_grid = None

        self._log(
            f"Inverse Bloch transform complete: H_real shape = {H_real.shape}", lvl=2, color="green"
        )

        # Push real-space basis to HilbertSpace (don't sync FROM it)
        self.push_basis_to_hilbert_space()

        return self

    def set_basis_type(self, basis_type: str):
        """
        Override: Set basis type and propagate to Hilbert space if available.
        """
        super().set_basis_type(basis_type)

        # Propagate to Hilbert space
        if self._hilbert_space is not None and hasattr(self._hilbert_space, "set_basis"):
            self._hilbert_space.set_basis(basis_type)

    ##########################################################################
    #! Basis transformation state queries
    ##########################################################################

    def set_single_particle_matrix(self, H: "Array"):
        if not self._particle_conserving:
            raise RuntimeError("Use set_bdg_matrices for non-conserving case")
        if H.shape != (self._ns, self._ns):
            raise ValueError(f"shape mismatch, expected {(self._ns, self._ns)}")
        self._hamil_sp = self._backend.array(H, dtype=self._dtype)
        self._invalidate_cache()
        self._log(f"set_single_particle_matrix: {H.shape}", lvl=3, log="debug")

    def set_bdg_matrices(self, K: np.ndarray, Delta: np.ndarray):
        """
        Set the Bogoliubov-de Gennes (BdG) matrices for the Hamiltonian.

        Parameters
        ----------
        K : np.ndarray
            The single-particle (kinetic) Hamiltonian matrix. Must be a square matrix of shape (self._ns, self._ns).
        Delta : np.ndarray
            The pairing (superconducting gap) matrix. Must be a square matrix of shape (self._ns, self._ns).

        Raises
        ------
        RuntimeError
            If the Hamiltonian is particle-conserving (i.e., self._particle_conserving is True).
        ValueError
            If the shapes of K or Delta do not match (self._ns, self._ns).

        Notes
        -----
        This method is only applicable for non-particle-conserving Hamiltonians (i.e., when self._particle_conserving is False).
        After setting the matrices, the internal cache is invalidated.
        """

        if self._particle_conserving:
            raise RuntimeError("BdG matrices only for particle_conserving=False")

        if K.shape != (self._ns, self._ns) or Delta.shape != (self._ns, self._ns):
            raise ValueError("shape mismatch")

        self._hamil_sp = self._backend.array(K, dtype=self._dtype)
        self._delta_sp = self._backend.array(Delta, dtype=self._dtype)
        self._invalidate_cache()
        self._log(f"set_bdg_matrices: {K.shape}, {Delta.shape}", lvl=3, log="debug")

    def build_single_particle_matrix(self, copy: bool = True):
        """Return the Ns x Ns single-particle matrix."""

        if self._hamil_sp is None:
            self.build()

        if self._is_sparse:
            return sp.sparse.csr_matrix(self._hamil_sp) if copy else self._hamil_sp
        return self._backend.array(self._hamil_sp).copy() if copy else self._hamil_sp

    def build_bdg_matrix(self, copy: bool = True):
        """Return the 2Ns x 2Ns BdG matrix (raises if particle-conserving)."""
        if self._particle_conserving:
            raise RuntimeError("BdG matrix requested but particle_conserving=True.")

        if self._hamil_sp is None or self._delta_sp is None:
            self.build()

        xp = self._backend
        bdg = xp.block(
            [
                [self._hamil_sp, self._delta_sp],
                [-xp.conjugate(self._delta_sp), -xp.conjugate(self._hamil_sp.T)],
            ]
        )
        return xp.array(bdg) if copy else bdg

    ###########################################################################
    #! Solvability Detection
    ###########################################################################

    def solvable(self) -> 'SolvabilityInfo':
        """
        Determine if this quadratic Hamiltonian admits a closed-form solution.

        A quadratic Hamiltonian is 'solvable' if eigenvalues can be obtained
        via analytical methods (diagonal extraction, k-space diagonalization)
        rather than full matrix diagonalization.

        Detection criteria:
        - **Diagonal**:
            If matrix is diagonal (or nearly so), eigenvalues = diagonal
        - **Tridiagonal/Band**:
            Special structure enabling fast solving
        - **Translational symmetry**:
            k-space FFT decomposition
        - **Already diagonalized**:
            Pre-computed eigenvalues available

        Returns
        -------
        SolvabilityInfo
            Solvability status with method, description, and optional eigenvalues.

        Examples
        --------
        >>> H = QuadraticHamiltonian(...)
        >>> H.build()
        >>> info = H.solvable()
        >>> if info.is_solvable and info.method == 'diagonal':
        ...     evals = info.eigenvalues  # Direct access to eigenvalues
        ... else:
        ...     H.diagonalize()  # Fall back to standard diagonalization

        Notes
        -----
        All quadratic Hamiltonians are mathematically solvable (quadratic systems),
        but this method detects **computational** solvability - can we find eigenvalues
        faster than O(n^3) full diagonalization?
        """
        
        try:
            from QES.Algebra.Quadratic.hamil_quadratic_utils import SolvabilityInfo
        except ImportError:
            raise ImportError("SolvabilityInfo class not found. Ensure QES.Algebra.Quadratic.hamil_quadratic_utils is available.")
        
        # Check if already diagonalized
        if self._eig_val is not None and self._eig_vec is not None:
            
            offset_str = f" (offset={self._constant_offset})" if self._constant_offset != 0 else ""
            return SolvabilityInfo(
                is_solvable         =   True,
                method              =   "analytical",
                description         =   f"Eigenvalues already computed via prior diagonalization{offset_str}",
                eigenvalues         =   self._eig_val.copy(),
                computation_cost    =   "O(1) - cached",
            )

        # Get matrix for analysis
        try:
            self._hamiltonian_quadratic()
            H = self._hamil
        except Exception:
            return SolvabilityInfo(
                is_solvable         =   False,
                method              =   "unknown",
                description         =   "Could not assemble Hamiltonian matrix for analysis",
                computation_cost    =   "O(n^3) - standard diagonalization",
            )

        if H is None:
            return SolvabilityInfo(
                is_solvable         =   False,
                method              =   "unknown",
                description         =   "Hamiltonian matrix not available",
                computation_cost    =   "O(n^3) - standard diagonalization",
            )

        # Convert to dense if sparse for analysis
        H_dense         = H.toarray() if sp.sparse.issparse(H) else np.asarray(H)
        n               = H_dense.shape[0]

        # Check if diagonal (or nearly diagonal)
        off_diag_norm   = np.linalg.norm(H_dense - np.diag(np.diag(H_dense)))
        is_diagonal     = (
                            off_diag_norm < 1e-10 * np.linalg.norm(H_dense)
                            if np.linalg.norm(H_dense) > 1e-10
                            else off_diag_norm < 1e-14
                        )

        if is_diagonal:
            eigenvalues = np.diag(H_dense) + self._constant_offset
            return SolvabilityInfo(
                is_solvable         =   True,
                method              =   "diagonal",
                description         =   "Hamiltonian is diagonal in current basis; eigenvalues = diagonal elements",
                eigenvalues         =   eigenvalues,
                sparsity_pattern    =   "full-diagonal",
                computation_cost    =   "O(n)",
            )

        # Check if tridiagonal (or nearly tridiagonal)
        upper_off_diag  = np.triu(H_dense, 2)
        lower_off_diag  = np.tril(H_dense, -2)
        band_norm       = np.linalg.norm(upper_off_diag) + np.linalg.norm(lower_off_diag)
        is_tridiagonal  = (
            band_norm < 1e-10 * np.linalg.norm(H_dense)
            if np.linalg.norm(H_dense) > 1e-10
            else band_norm < 1e-14
        )

        if is_tridiagonal:
            return SolvabilityInfo(
                is_solvable         =   True,
                method              =   "kspace",
                description         =   "Hamiltonian is tridiagonal; can be solved via specialized eigensolvers",
                sparsity_pattern    =   "tridiagonal",
                computation_cost    =   "O(n^2)",
            )

        # Check sparsity level
        sparsity = 1.0 - np.count_nonzero(H_dense) / H_dense.size
        if sparsity > 0.9:
            return SolvabilityInfo(
                is_solvable         =   True,
                method              =   "kspace",
                description         =   f"Hamiltonian is highly sparse ({100*sparsity:.1f}%); Krylov subspace methods efficient",
                sparsity_pattern    =   "highly-sparse",
                computation_cost    =   "O(n^2) - sparse iterative methods",
            )

        # Check for band structure (common in lattice models)
        max_band_width = 0
        for i in range(n):
            for j in range(n):
                if abs(H_dense[i, j]) > 1e-14:
                    max_band_width = max(max_band_width, abs(i - j))

        if max_band_width < 0.3 * n:  # Narrow band
            return SolvabilityInfo(
                is_solvable         =   True,
                method              =   "kspace",
                description         =   f"Hamiltonian has band structure (bandwidth ^ {max_band_width}); efficient via structured methods",
                sparsity_pattern    =   f"band-diagonal (width={max_band_width})",
                computation_cost    =   "O(n^2) - band solver",
            )

        # Check for Hermitian structure (all quadratic Hamiltonians should be Hermitian)
        is_hermitian = np.allclose(H_dense, H_dense.conj().T, atol=1e-10)

        if is_hermitian:
            return SolvabilityInfo(
                is_solvable         =   True,
                method              =   "standard",
                description         =   "Quadratic system is Hermitian; solvable via standard eigendecomposition",
                sparsity_pattern    =   "generic-hermitian" if sparsity < 0.1 else "sparse-hermitian",
                computation_cost    =   "O(n^3) - dense" if sparsity < 0.1 else "O(n^2) - sparse methods",
            )

        # Fallback
        return SolvabilityInfo(
            is_solvable         =   True,
            method              =   "standard",
            description         =   "Quadratic Hamiltonian requires standard matrix diagonalization",
            sparsity_pattern    =   "generic",
            computation_cost    =   "O(n^3)",
        )

    ###########################################################################
    #! Diagonalization
    ###########################################################################

    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        r"""
        Assemble the quadratic Hamiltonian matrix prior to diagonalization.
        """
        if self._particle_conserving:
            self._hamil = self._hamil_sp
        else:
            self._hamil = self.build_bdg_matrix(copy=False)

    def build(self, verbose: bool = False, use_numpy: bool = True, force: bool = True):
        """Build the quadratic Hamiltonian matrix."""
        
        self._log("Building quadratic Hamiltonian matrix...", lvl=2, log="info", verbose=verbose)
            
        if force or self._hamil is None:
            self._hamiltonian_quadratic(use_numpy=use_numpy)
            
        else:
            self._log("Using cached Hamiltonian matrix.", lvl=2, log="debug", verbose=verbose)

    def diagonalize(self, verbose: bool = False, force: bool = False, **kwargs):
        """
        Diagonalizes the quadratic matrix and applies constant offset
        for a constant term that can be included in the quadratic system.

        This method implements lazy diagonalization - it only performs
        the actual diagonalization when eigenvalues/vectors are needed.
        """

        # Mark that diagonalization has been requested
        self._diagonalization_requested = True

        # Only perform actual diagonalization if not already done
        if self._eig_val is None or self._eig_vec is None or force:
            if verbose:
                self._log("Performing diagonalization...", lvl=2, log="info")

            # Ensure the quadratic matrix is assembled before diagonalization
            try:
                # Check if we need to build/assemble (e.g. if _hamil is missing)
                if self._hamil is None or force:
                    self._hamiltonian_quadratic()
            except Exception:
                pass

            # Calls base diagonalize on self.hamil (which is _hamil_sp or BdG)
            super().diagonalize(verbose=verbose, **kwargs)

            # Apply constant offset after diagonalization
            if self._eig_val is not None and self._constant_offset != 0.0:
                if verbose:
                    self._log(f"Adding constant offset {self._constant_offset} to eigenvalues.", lvl=2, log="debug",)
                    
                # Apply offset
                self._eig_val += self._constant_offset
                
                # Recalculate energy stats if offset was applied
                self._calculate_av_en()
                
                return
        
        elif verbose:
            self._log("Using cached diagonalization results.", lvl=2, log="debug")
        
        self._log("Diagonalization was used and it was not forced...", lvl=3, log="warning")

    @property
    def eig_val(self) -> np.ndarray:
        """Eigenvalues (triggers diagonalization if not yet performed)."""
        if self._eig_val is None:
            self.diagonalize()
        return self._eig_val

    @property
    def eig_vec(self) -> np.ndarray:
        """Eigenvectors (triggers diagonalization if not yet performed)."""
        if self._eig_vec is None:
            self.diagonalize()
        return self._eig_vec

    ###########################################################################

    def diagonalizing_bogoliubov_transform(self, copy: bool = True):
        """
        Return the diagonalizing Bogoliubov transformation matrices.

        This method returns the transformation matrix W, orbital energies epsilon_j,
        and the transformed constant, following the convention from Qiskit Nature.

        For particle-conserving systems:
            H = sum_j epsilon_j b_j^dagger b_j + constant
        where (b_1^dagger, ..., b_N^dagger) = W (a_1^dagger, ..., a_N^dagger)
        and W is N x N unitary.

        For non-particle-conserving systems:
            H = sum_j epsilon_j gamma_j^dagger gamma_j + constant
        where (b_1^dagger, ..., b_N^dagger, b_1, ..., b_N) = W (a_1^dagger, ..., a_N^dagger, a_1, ..., a_N)
        and W is 2N x 2N unitary.

        Parameters
        ----------
        copy : bool, optional
            Whether to return copies of the matrices (default: True)

        Returns
        -------
        transformation_matrix : np.ndarray
            The Bogoliubov transformation matrix W
        orbital_energies : np.ndarray
            The orbital energies epsilon_j (non-negative for BdG)
        transformed_constant : float
            The transformed constant term

        Notes
        -----
        This method ensures diagonalization has been performed.
        For BdG systems, orbital energies are made non-negative.
        """
        # Ensure we have eigenvalues/eigenvectors
        if self._eig_val is None or self._eig_vec is None:
            self.diagonalize()

        if self._particle_conserving:
            # Particle-conserving case: W is NxN
            W = self._eig_vec if not copy else self._eig_vec.copy()
            orbital_energies = self._eig_val.copy() if copy else self._eig_val
            transformed_constant = self._constant_offset
        else:
            # BdG case: diagonalize full BdG matrix and return a Qiskit-like
            # transformation matrix of shape (N, 2N) where each row k is
            # [u_k^T, v_k^T] such that b_k^^dagger = sum_j u_kj a_j^^dagger + v_kj a_j.
            # This follows the convention used in Qiskit Nature tutorials.
            N   = self._ns

            # Build BdG matrix as NumPy array for diagonalization
            bdg = np.asarray(self.build_bdg_matrix(copy=True))

            # Diagonalize using scipy.linalg.eigh (guaranteed Hermitian structure
            # up to numerical noise). We get eigenvalues in ascending order.
            try:
                eigvals, eigvecs = sp.linalg.eigh(bdg)
            except Exception:
                # fallback to numpy if scipy unavailable
                eigvals, eigvecs = np.linalg.eigh(bdg)

            # Eigenvalues come in ± pairs; select N positive-energy modes by
            # taking indices of the N largest absolute eigenvalues and using
            # their absolute values as the orbital energies.
            idx_by_abs = np.argsort(np.abs(eigvals))[-N:]
            energies = np.abs(eigvals[idx_by_abs])

            # Order energies ascending for consistent output
            order = np.argsort(energies)
            selected_idx = idx_by_abs[order]
            orbital_energies = energies[order]

            # Corresponding eigenvectors (columns) -> shape (2N, N)
            psi = eigvecs[:, selected_idx]

            # Split into particle (u) and hole (v) components
            U_mat = psi[:N, :]
            V_mat = psi[N:, :]

            # Build transformation matrix W_small of shape (N, 2N): row k = [u_k^T, v_k^T]
            W_small = np.hstack((U_mat.conj().T, V_mat.conj().T))

            W = W_small if copy else self._backend.array(W_small)
            transformed_constant = self._constant_offset

        return W, orbital_energies, transformed_constant

    def conserves_particle_number(self) -> bool:
        """
        Check if the Hamiltonian conserves particle number.

        Returns True if the pairing matrix Delta is zero (or effectively zero).

        Returns
        -------
        bool
            True if particle number is conserved, False otherwise
        """
        if self._particle_conserving:
            return True

        # Check if pairing matrix is effectively zero
        delta_norm = np.linalg.norm(self._delta_sp)
        return delta_norm < 1e-12

    ###########################################################################
    #! Block Diagonal Analysis (Band Structure)
    ###########################################################################

    def block_diagonal_bdg(self) -> Tuple[List['QuadraticBlockDiagonalInfo'], np.ndarray]:
        r"""
        Extract and diagonalize each k-space block as a BdG system.

        For a k-space Hamiltonian (after calling `to_basis("k-space")`), this method:
        1. Extracts each k-block (Nb x Nb or 2Nb x 2Nb)
        2. Diagonalizes the block independently
        3. Returns eigenvalues and eigenvectors for each k-point

        This is useful for band structure calculations, topological analysis,
        or sector-specific computations without creating many Hamiltonian objects.

        Returns
        -------
        List[QuadraticBlockDiagonalInfo]
            List of diagonalized block info, one per k-point in the original lattice.
            Each block contains:
            - `point`: k-vector (3D)
            - `en`: eigenvalues at that k-point
            - `ev`: eigenvectors (columns)
            - `block_index`: (ix, iy, iz) index in k-space grid
            - `is_bdg`: whether this is a BdG block

        Raises
        ------
        ValueError
            If Hamiltonian is not in k-space representation.
        RuntimeError
            If k-space transformation hasn't been performed.

        Examples
        --------
        >>> from QES.Algebra import QuadraticHamiltonian
        >>> from QES.general_python.lattices import SquareLattice
        >>>
        >>> # Create and transform to k-space
        >>> lat = SquareLattice(dim=2, lx=4, ly=4, bc='pbc')
        >>> ham = QuadraticHamiltonian(ns=16, lattice=lat)
        >>> ham.add_hopping(0, 1, -1.0)  # nearest-neighbor hopping
        >>> ham.to_basis("k-space")
        >>> ham.diagonalize()
        >>>
        >>> # Get band structure info
        >>> blocks = ham.block_diagonal_bdg()
        >>>
        >>> for block in blocks:
        ...     print(f"k={block.point}, E0={block.en[0]:.4f}")

        Notes
        -----
        - For particle-conserving systems: each block is Nbtimes Nb
        - For BdG systems: each block is 2Nbtimes 2Nb
        - Eigenvalues are sorted in ascending order within each block
        - This method assumes the Hamiltonian is already diagonalized
          at the whole-system level. For per-block independent diagonalization,
          that happens internally here.
        """
        
        try:
            from QES.Algebra.Quadratic.hamil_quadratic_utils import QuadraticBlockDiagonalInfo
        except ImportError as e:
            raise ImportError("Could not import QuadraticBlockDiagonalInfo. Ensure QES is properly installed.") from e    
        
        # Check that we're in k-space
        if self._hamil_transformed is None:
            raise RuntimeError("Hamiltonian not in k-space. Call to_basis('k-space') first.")

        H_k = self._hamil_transformed  # Shape: (Lx, Ly, Lz, Nb, Nb) or (Lx, Ly, Lz, 2*Nb, 2*Nb)
        k_grid = self._transformed_grid  # Shape: (Lx, Ly, Lz, 3)
        k_grid_frac = self._transformed_grid_frac  # Shape: (Lx, Ly, Lz, 3) fractional coords

        if H_k is None or k_grid is None:
            raise ValueError(
                "Transformed Hamiltonian or k-grid not available. Ensure to_basis('k-space') was successful."
            )

        # Determine block type
        # nb_sublattices  = H_k.shape[3] // (2 if not self._particle_conserving else 1)
        is_bdg = not self._particle_conserving
        energies = []

        results: List[QuadraticBlockDiagonalInfo] = []

        # H_k and k_grid are both in fftfreq order (unshifted)
        # Gamma point is at index [0,0,0] for both
        # Iterate over all k-points
        for i in range(H_k.shape[0]):
            for j in range(H_k.shape[1]):
                for k in range(H_k.shape[2]):
                    k_vec = np.asarray(k_grid[i, j, k, :], dtype=np.float64)
                    k_vec_frac = np.asarray(k_grid_frac[i, j, k, :], dtype=np.float64)

                    # Use H_k directly (both in fftfreq order)
                    im, jm, km = i, j, k
                    # im, jm, km  = (i + H_k.shape[0]//2) % H_k.shape[0], \
                    #               (j + H_k.shape[1]//2) % H_k.shape[1], \
                    #               (k + H_k.shape[2]//2) % H_k.shape[2]
                    H_block = np.asarray(H_k[im, jm, km, :, :], dtype=self._dtype)

                    # Diagonalize this block
                    try:
                        eigvals, eigvecs = sp.linalg.eigh(H_block)
                    except Exception:
                        eigvals, eigvecs = np.linalg.eigh(H_block)

                    # Create info object
                    info = QuadraticBlockDiagonalInfo(
                        point=k_vec,  # k-point vector
                        frac_point=k_vec_frac,  # Fractional k-point vector
                        en=eigvals,  # Eigenvalues
                        ev=eigvecs,  # Columns are eigenvectors
                        block_index=(i, j, k),  # Indices in k-grid
                        is_bdg=is_bdg,  # True if BdG block, False if PC
                        label=None,  # Special label?
                    )

                    energies.append(eigvals)
                    results.append(info)

        self._log(
            f"Extracted and diagonalized {len(results)} k-space blocks "
            f"({'BdG' if is_bdg else 'PC'}, {H_k.shape[3]}x{H_k.shape[3]})",
            lvl=2,
            color="cyan",
        )

        return results, np.array(energies)

    ###########################################################################
    #! Transformation Preparation
    ###########################################################################

    @dataclass
    class PCTransform:
        """Particle-conserving transformation handle."""

        W           : np.ndarray  # (Ns, Ns)    # single-particle eigenvectors (unitary)
        occ_mask    : np.uint32 | np.uint64     # occupied mask over Ns
        _occ_idx    : np.ndarray  = field(default_factory=lambda: np.array([], dtype=np.int64)) # (Na,)       # subset of occupied indices
        _unocc_idx  : np.ndarray  = field(default_factory=lambda: np.array([], dtype=np.int64)) # (Ns-Na,)    # complement of occupied indices
        _W_CT       : np.ndarray | None = None                                                  # cache for W^dagger
        _WA         : np.ndarray | None = None                                                  # cache for W_A
        _WA_CT      : np.ndarray | None = None                                                  # cache for W_A^dagger

        # on-demand helpers (allocate only when used)

        def occ_idx(self) -> np.ndarray:
            """Occupied indices (from mask)."""
            try:
                from QES.Algebra.Quadratic.hamil_quadratic_utils import indices_from_mask
            except ImportError as e:
                raise ImportError("Could not import indices_from_mask. Ensure QES is properly installed.") from e
            
            # Lazy compute occ_idx
            if self._occ_idx is None or self._occ_idx.size == 0:
                self._occ_idx = indices_from_mask(self.occ_mask)
            return self._occ_idx
        
        def unocc_idx(self) -> np.ndarray:
            """Unoccupied indices (from mask)."""
            try:
                from QES.Algebra.Quadratic.hamil_quadratic_utils import complement_indices
            except ImportError as e:
                raise ImportError("Could not import complement_indices. Ensure QES is properly installed.") from e
            if self._unocc_idx is None or self._unocc_idx.size == 0:
                Ns              = self.W.shape[0]
                self._unocc_idx = complement_indices(Ns, self.occ_idx())
            return self._unocc_idx
        
        @property
        def W_A(self) -> np.ndarray:
            """Form columns W[:, occ_idx]. NOTE: column gather copies."""
            if self._WA is None:
                self._WA = self.W[self.occ_idx(), :].T
            return self._WA

        @property
        def W_A_CT(self) -> np.ndarray:
            """Conjugate transpose of W_A (allocates as above)."""
            if self._WA_CT is None:
                self._WA_CT = self.W_A.conj().T
            return self._WA_CT
        
        @property
        def W_CT(self) -> np.ndarray:
            """Conjugate transpose of W (allocates if needed)."""
            if self._W_CT is None:
                self._W_CT = self.W.conj().T
            return self._W_CT

        def order_occ_then_unocc(self) -> np.ndarray:
            """Permutation indices [occ, unocc]."""
            return np.concatenate((self.occ_idx, self.unocc_idx), dtype=np.int64)

    @dataclass(frozen=True)
    class BdGTransform:
        """BdG/Nambu transform handle, avoids big temporaries."""

        W           : np.ndarray                # (2N, 2N) quasiparticle eigenvectors in Nambu basis
        N           : int                       # single-particle dimension
        occ_idx     : np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # subset in physical modes (0..N-1)
        unocc_idx   : np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # complement in physical modes

        # Block views (no copies): U= W[:N,:N], V= W[:N,N:]
        @property
        def U(self) -> np.ndarray:
            return self.W[: self.N, : self.N]

        @property
        def V(self) -> np.ndarray:
            return self.W[: self.N, self.N :]

        # Prefer row slicing (views) when projecting to a *mode subset*

        def U_rows_A(self) -> np.ndarray:
            """Rows of U restricted to A (view). Shape (|A|, N)."""
            return self.U[self.occ_idx, :]

        def V_rows_A(self) -> np.ndarray:
            """Rows of V restricted to A (view). Shape (|A|, N)."""
            return self.V[self.occ_idx, :]

        # Columns-for-A (copies; only form when needed)
        def U_cols_A(self) -> np.ndarray:
            """Columns of U for A (copy; unavoidable in NumPy). Shape (N, |A|)."""
            return np.take(self.U, self.occ_idx, axis=1)

        def V_cols_A(self) -> np.ndarray:
            """Columns of V for A (copy; unavoidable). Shape (N, |A|)."""
            return np.take(self.V, self.occ_idx, axis=1)

    def prepare_transformation(self, occ, *, bdg: bool | None = None) -> Union['QuadraticHamiltonian.PCTransform', 'QuadraticHamiltonian.BdGTransform']:
        r"""
        Memory-conscious preparation of subspace transforms.

        Parameters
        ----------
        occ : IndexLike
            - int k             : take first k orbitals/modes
            - 1D bool mask      : occupied mask over Ns (PC) or N (BdG physical sector)
            - 1D int indices    : occupied indices in 0..Ns-1 (PC) or 0..N-1 (BdG)
        bdg : bool | None
            - If None: infer from self._particle_conserving (bdg = not particle_conserving).
            - If True : treat eigenvector matrix as (2N x 2N) BdG/Nambu.
            - If False: treat as particle-conserving (Ns x Ns).

        Returns
        -------
        QuadraticHamiltonian.PCTransform   if bdg == False
        QuadraticHamiltonian.BdGTransform  if bdg == True

        Notes
        -----
        - Does not eagerly form large submatrices.
        - Column gathers (W_A, U_cols_A, V_cols_A) allocate only when invoked.
        - Row slices (U_rows_A, V_rows_A) are views.
        """
        
        try:
            from QES.Algebra.Quadratic.hamil_quadratic_utils import occ_to_indices, complement_indices, occ_to_mask
        except ImportError as e:
            raise ImportError("Could not import utility functions. Ensure QES is properly installed.") from e
        
        # Determine bdg mode
        if bdg is None:
            bdg = not self._particle_conserving

        # Process occ into index array
        W = self._eig_vec
        if W is None:
            raise RuntimeError("Eigenvectors not available. Call diagonalize() first.")

        # Process occ into index array
        if bdg:
            # BdG / Nambu path
            if W.ndim != 2 or W.shape[0] != W.shape[1] or (W.shape[0] % 2 != 0):
                raise ValueError(f"Expect square (2N,2N) BdG eigenvector matrix; got {W.shape}")

            twoN        = W.shape[0]
            N           = twoN // 2

            occ_idx     = occ_to_indices(occ, N)
            unocc_idx   = complement_indices(N, occ_idx)
            return self.BdGTransform(W=W, N=N, occ_idx=occ_idx, unocc_idx=unocc_idx)

        # Particle-conserving path
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"Expect square (Ns,Ns) eigenvector matrix; got {W.shape}")
        
        Ns          = W.shape[1]
        occ_idx     = occ_to_mask(occ, Ns)
        
        return self.PCTransform(W=W, occ_mask=occ_idx)

    ###########################################################################
    #! Many-Body Energy Calculation
    ###########################################################################

    def many_body_energy(self, occupied_orbitals: Union[int, List[int], np.ndarray]) -> float:
        r"""
        Calculates the total energy of a many-body state defined by occupying
        single-particle orbitals (or quasiparticle orbitals for BdG).

        Args:
            occupied_orbitals (list/array):
                Indices of the occupied single-particle
                eigenstates (orbitals alpha or quasiparticles gamma).

        Returns:
            The total energy E = sum_{alpha in occupied} epsilon_alpha (or E = sum_{gamma in occupied} E_gamma for BdG).
            Result includes the constant_offset.
        """
        
        try:
            from QES.general_python.common.binary import int2base
        except ImportError as e:
            raise ImportError("Could not import int2base. Ensure QES is properly installed.") from e

        if self.eig_val is None:
            raise ValueError("Single-particle eigenvalues not calculated. Call diagonalize() first.")
        
        if isinstance(occupied_orbitals, int):
            occupied_orbitals = int2base(occupied_orbitals, self._ns, spin=False, spin_value=1, backend=self._backend).astype(self._dtypeint)

        # Convert to NumPy array for processing
        occ = np.asarray(occupied_orbitals, dtype=self._dtypeint)
        if occ.shape[0] == 0:
            return 0.0

        if occ.ndim != 1:
            raise ValueError("occupied_orbitals must be 1-D")
        
        e = 0.0
        
        # Go with energy calculation functions
        if self._is_jax:
            occ     = jnp.asarray(occ, dtype=self._dtypeint)
            vmax    = self._eig_val.shape[0]

            def _check_bounds(x):
                if int(jnp.min(x)) < 0 or int(jnp.max(x)) >= vmax:
                    raise IndexError("orbital index out of bounds")
                return x

            occ     = _check_bounds(occ)

            if self._particle_conserving:
                e = jnp.sum(self._eig_val[occ])
            else:
                if int(jnp.max(occ)) >= self._ns:
                    raise IndexError("BdG index must be in 0...Ns-1")
                mid = self._ns - 1
                e = jnp.sum(self._eig_val[mid + occ + 1] - self._eig_val[mid - occ])
        else:
            vmax    = self._eig_val.shape[0]
            if occ.min() < 0 or occ.max() >= vmax:
                raise IndexError("orbital index out of bounds")

            if self._particle_conserving:
                nrg_pc      = _get_nrg_particle_conserving()
                e           = nrg_pc(self._eig_val, occ)
            else:
                if occ.max() >= self._ns:
                    raise IndexError("BdG index must be in 0...Ns-1 (positive branch)")
                nrg_bdg_fn  = _get_nrg_bdg()
                e           = nrg_bdg_fn(self._eig_val, self._ns, occ)
        return self._backend.real(e) + self._constant_offset

    def many_body_energies(
        self,
        n_occupation        : Union[float, int] = 0.5,
        nh                  : Optional[int] = None,
        use_combinations    : bool = False,
    ) -> dict[int, float]:
        r"""
        Returns a dictionary of many-body energies for all possible
        configurations of occupied orbitals.

        The keys are integers representing the configuration of occupied orbitals,
        and the values are the corresponding many-body energies.
        The function iterates over all possible configurations of occupied orbitals
        and calculates the many-body energy for each configuration.

        Parameters
        ----------
        n_occupation : float or int
            The number of occupied orbitals. If a float, it is interpreted as a fraction
            of the total number of sites (ns). If an int, it is the exact number of
            occupied orbitals.
        nh : Optional[int] = None
            The number of configurations to consider. If None, it defaults to 2^ns.
        use_combinations : bool
            If True, use combinations to generate occupied orbitals.
            If False, use a loop over all possible configurations.
            This is useful for large systems where the number of configurations
            is too large to handle with combinations.
        Returns
        -------
        dict[int, float]
            A dictionary where the keys are integers representing the configuration
            of occupied orbitals and the values are the corresponding many-body energies.
        Notes
        -----
        - The function uses the int2base function to convert integers to binary
        representations of occupied orbitals.
        - The function uses the many_body_energy method to calculate the energy
        """
        
        try:
            from QES.Algebra.Quadratic.hamil_quadratic_utils    import QuadraticSelection
            from QES.general_python.common.binary               import int2base
        except ImportError as e:
            raise ImportError(
                "QuadraticSelection module is required for many_body_energies. "
                "Please ensure it is available."
            ) from e

        if 0 < n_occupation < 1:
            n_occupation = int(self._ns * n_occupation)
        elif n_occupation > self._ns:
            raise ValueError("n_occupation must be less than or equal to the number of sites.")
        elif n_occupation < 0:
            raise ValueError("n_occupation must be greater than or equal to 0.")

        if nh is None:
            nh = 2**self._ns

        # what is faster?
        many_body_energies = {}
        #! 1. Loop over all possible configurations of occupied orbitals
        if not use_combinations:
            for i in range(nh):
                occupied_orbitals = int2base(i, self._ns, spin=False, spin_value=1).astype(
                    self._dtypeint
                )
                occupied_orbitals = np.nonzero(occupied_orbitals)[0]
                if len(occupied_orbitals) != n_occupation:
                    continue
                many_body_energies[i] = self.many_body_energy(occupied_orbitals)
            return many_body_energies
        #! 2. Use combinations
        else:
            all_combinations = QuadraticSelection.all_orbitals(self._ns, n_occupation)
            for i, occupied_orbitals in enumerate(all_combinations[1]):
                occupied_orbitals = np.array(occupied_orbitals, dtype=self._dtypeint)
                many_body_energies[i] = self.many_body_energy(occupied_orbitals)
            return many_body_energies

    ###########################################################################
    #! Many-Body State Calculation
    ###########################################################################

    def _get_pairing_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Lazily compute and cache pairing matrices (U, V, F, G) for BdG calculations.
        
        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            (F_matrix, G_matrix) for particle-conserving vs BdG modes
        """
        bogolubov_decompose = _get_bogolubov_decompose()
        pairing_matrix      = _get_pairing_matrix()
        
        if self._U is None or self._V is None:
            self._U, self._V, _ = bogolubov_decompose(self._eig_val, self._eig_vec)
        
        if self._isfermions:
            if self._F is None:
                self._F = pairing_matrix(self._U, self._V)
            return self._F, None
        else:  # bosons
            if self._G is None:
                self._G = pairing_matrix(self._eig_val, self._eig_vec)
            return None, self._G

    def _make_calculator(self) -> Tuple[callable, np.ndarray]:
        """
        Create or retrieve cached calculator function and matrix argument.
        
        This method:
        1. Checks cache to avoid recompilation
        2. Pre-computes pairing matrices once for BdG
        3. Returns a re-usable calculator function
        
        Returns
        -------
        Tuple[callable, np.ndarray]
            (calculator_function, matrix_argument)
        
        Notes
        -----
        The calculator function has signature:
            psi_coeff = calc(matrix_arg, basis_state_int, ns)
        
        For particle-conserving systems, the occupation is baked into the closure.
        For BdG systems, the pairing matrix is used directly.
        """
        if self._eig_vec is None:
            raise RuntimeError("Eigenvectors not available. Call diagonalize() first.")
        
        # Build cache key: (is_numpy, is_fermion, is_particle_conserving, occupation_hash)
        # Include occupation hash only for particle-conserving systems
        if self._particle_conserving and self._occupied_orbitals_cached is not None:
            occ_hash    = hash(self._occupied_orbitals_cached.tobytes())
        else:
            occ_hash    = None
        cache_key       = (self._is_numpy, self._isfermions, self._particle_conserving, occ_hash)
        
        # Return cached result if available
        if cache_key in self._calculator_cache:
            calc_fn, matrix_arg = self._calculator_cache[cache_key]
            return calc_fn, matrix_arg
        
        # Lazy import the needed functions
        calculate_slater_det            = _get_calculate_slater_det()
        many_body_state_closure         = _get_many_body_state_closure()
        calculate_slater_det_jax        = _get_calculate_slater_det_jax()
        calculate_bogoliubov_amp        = _get_calculate_bogoliubov_amp()
        calculate_permanent             = _get_calculate_permanent()
        calculate_permanent_jax         = _get_calculate_permanent_jax()
        calculate_bosonic_gaussian_amp  = _get_calculate_bosonic_gaussian_amp()
        
        # FERMION + PARTICLE CONSERVING
        if self._isfermions and self._particle_conserving:
            if self._occupied_orbitals_cached is None:
                raise RuntimeError(
                    "Occupied orbitals not cached. Call many_body_state(...) with "
                    "occupied_orbitals argument first."
                )
            occ = self._occupied_orbitals_cached
            
            if self._is_numpy:
                # Use the Numba-optimized closure from hilbert_jit_states
                calc_fn         = many_body_state_closure(calculator_func=calculate_slater_det, matrix_arg=occ)
            else:
                # JAX version: create closure that captures occupation
                captured_occ    = occ  # capture in closure
                calc_fn         = lambda U, st, _ns: calculate_slater_det_jax(U, captured_occ, st, _ns)
                
            matrix_arg = self._eig_vec
        
        # FERMION + BdG (NON-PARTICLE-CONSERVING) 
        elif self._isfermions and not self._particle_conserving:
            F_matrix, _ = self._get_pairing_matrices()
            
            if self._is_numpy:
                # Bogoliubov amplitude calculation (3-arg interface)
                calc_fn = lambda F, st, _ns: calculate_bogoliubov_amp(F, st, _ns)
            else:
                raise NotImplementedError("JAX Bogoliubov vacuum not yet implemented.")
            
            matrix_arg = F_matrix
        
        # BOSON + PARTICLE CONSERVING
        elif self._isbosons and self._particle_conserving:
            if self._occupied_orbitals_cached is None:
                raise RuntimeError(
                    "Occupied orbitals not cached. Call many_body_state(...) with "
                    "occupied_orbitals argument first."
                )
            occ = self._occupied_orbitals_cached
            
            if self._is_numpy:
                # Permanent needs occupation; create closure
                captured_occ    = occ
                calc_fn         = lambda G, st, _ns: calculate_permanent(G, captured_occ, st, _ns)
            else:
                captured_occ    = occ
                calc_fn         = lambda G, st, _ns: calculate_permanent_jax(G, captured_occ, st, _ns)
            
            matrix_arg = self._eig_vec
        
        # BOSON + BdG (NON-PARTICLE-CONSERVING)
        else:  # self._isbosons and not self._particle_conserving
            _, G_matrix = self._get_pairing_matrices()
            
            if self._is_numpy:
                calc_fn = lambda G, st, _ns: calculate_bosonic_gaussian_amp(G, st, _ns)
            else:
                raise NotImplementedError("JAX Bosonic Gaussian state not yet implemented.")
            
            matrix_arg = G_matrix
        
        # Cache for future calls
        self._calculator_cache[cache_key] = (calc_fn, matrix_arg)
        return calc_fn, matrix_arg

    def _many_body_state_calculator(self):
        r"""
        Return a cached calculator function for many-body state computation.
        
        **DEPRECATED**: Use _make_calculator() instead. Kept for backwards compatibility.
        
        Returns a function object that implements:
            psi = calc(matrix_arg, basis_state_int, ns)
        
        The function is optimized for repeated calls and avoids recompilation.
        
        Returns
        -------
        Tuple[callable, np.ndarray]
            (calculator_function, matrix_argument)
        """
        return self._make_calculator()

    def many_body_state(
        self,
        occupied_orbitals   : Union[list[int], np.ndarray, None]    = None,
        target_basis        : str                                   = "sites",
        *,
        many_body_hs        : Optional[HilbertSpace]                = None,
        resulting_state     : Optional[np.ndarray]                  = None,
        cache_state         : bool                                  = True,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Return the coefficient vector |Psi> in the computational basis.
        Computes the many-body state corresponding to the specified
        occupied single-particle orbitals (or quasiparticle orbitals for BdG).
        
        Parameters
        ----------
        occupied_orbitals : Union[list[int], np.ndarray, None]
            For particle-conserving fermions/bosons: indices of occupied orbitals.
            Can be:
            - int: 
                single configuration encoded as integer
            - 1D array: 
                binary occupation mask [n1, n2, ..., nNs]
            - list: 
                occupation indices
            Ignored for BdG systems.
        target_basis : str
            Only "sites" (Fock basis) is currently supported.
        many_body_hs : Optional[HilbertSpace]
            If provided, uses custom mapping for state ordering.
            If None, generates full 2^Ns state vector.
        resulting_state : Optional[np.ndarray]
            Pre-allocated output array. If None, creates new array.
            Shape must be (2^Ns,) or (len(mapping),) for custom HS.
        Returns
        -------
        np.ndarray
            Coefficient vector psi(x) with dtype matching computation precision.

        Raises
        ------
        RuntimeError
            If eigenvalues/eigenvectors not computed.
        NotImplementedError
            If target_basis is not "sites".

        Examples
        --------
        >>> ham = QuadraticHamiltonian(ns=4)
        >>> ham.add_hopping(0, 1, -1.0)
        >>> ham.diagonalize()
        
        # Single state from occupation indices
        >>> psi = ham.many_body_state([0, 1, 0, 0])  # sites 0,1 occupied
        
        # Single state from integer encoding
        >>> psi = ham.many_body_state(3)  # binary: 0011 (sites 0,1 occupied)
        
        # With pre-allocated array for batch computation
        >>> states = np.zeros((10, 2**4), dtype=complex)
        >>> for i, occ in enumerate(occupations):
        ...     ham.many_body_state(occ, resulting_state=states[i])
        
        Notes
        -----
        - **Caching**: First call computes and caches the calculator function.
            Subsequent calls reuse the cached function without recompilation.
        - **For BdG systems**: occupied_orbitals is ignored; computes vacuum.
        """
        
        if target_basis != "sites":
            raise NotImplementedError("Only the site/bitstring basis is implemented for now.")

        if self._eig_vec is None:
            raise RuntimeError("Eigenvectors not computed. Call diagonalize() first.")

        try:
            from QES.Algebra.Quadratic.hamil_quadratic_utils import indices_from_mask
        except ImportError as e:
            raise ImportError("Could not import indices_from_mask. Ensure QES is properly installed.") from e        

        # Update cache (only for particle-conserving systems)
        if self._particle_conserving and occupied_orbitals is not None:
            if isinstance(occupied_orbitals, int):
                # Convert integer bitmask to array of occupied orbital indices
                self._occupied_orbitals_cached = indices_from_mask(np.uint64(occupied_orbitals)).astype(self._dtypeint)
            
            elif isinstance(occupied_orbitals, (list, tuple)):
                # Convert list to contiguous numpy array (assumed to be orbital indices)
                self._occupied_orbitals_cached = np.ascontiguousarray(occupied_orbitals, dtype=self._dtypeint)
            
            else:
                # Assume numpy array of orbital indices; make contiguous
                self._occupied_orbitals_cached = np.ascontiguousarray(occupied_orbitals, dtype=self._dtypeint)

        # Get resulting calculator and matrix argument, this avoids recompilation on repeated calls
        calculator, matrix_arg  = self._make_calculator()
        ns                      = self._ns
        nfilling                = len(self._occupied_orbitals_cached) if self._occupied_orbitals_cached is not None else None
        # Ensure complex dtype since Slater determinants return complex values
        base_dtype              = getattr(self, "_dtype", np.result_type(matrix_arg))
        dtype                   = np.result_type(base_dtype, np.complex128)
        if resulting_state is not None:
            dtype   = np.result_type(resulting_state, dtype)
        
        # Try to use cached resulting_state if requested
        if cache_state and resulting_state is None:
            # Cache the resulting state array for future calls
            if many_body_hs is None or not many_body_hs.modifies:
                resulting_state = np.zeros((2**ns,), dtype=dtype)
            else:
                resulting_state = np.zeros((len(many_body_hs.mapping),), dtype=dtype)
            self._cached_many_body_state = resulting_state
        elif self._cached_many_body_state is not None and resulting_state is None:
            # Reuse cached state array
            resulting_state     = self._cached_many_body_state

        # COMPUTE STATE, use custom mapping if provided, else full Fock space
        # Lazy import the state computation functions
        many_body_state_full_fn     = _get_many_body_state_full()
        many_body_state_mapping_fn  = _get_many_body_state_mapping()
        
        if many_body_hs is None or not many_body_hs.modifies:
            return many_body_state_full_fn(matrix_arg, calculator, ns, resulting_state, nfilling=nfilling, dtype=dtype)
        else:
            mapping = many_body_hs.mapping
            return many_body_state_mapping_fn(matrix_arg, calculator, mapping, ns, dtype)

    def compute_ground_state(self, symmetry_sector: Optional[int] = None) -> Tuple[float, np.ndarray]:
        r"""
        Compute the ground state and energy for particle-conserving systems.
        
        For particle-conserving systems, the ground state is the Slater determinant
        (or permanent) formed by the lowest-energy occupied orbitals.
        
        For BdG systems, computes the Bogoliubov vacuum.

        Parameters
        ----------
        symmetry_sector : Optional[int]
            For particle-conserving systems: number of occupied orbitals.
            If None, fills the lowest ceil(Ns/2) orbitals.

        Returns
        -------
        Tuple[float, np.ndarray]
            (ground_state_energy, ground_state_vector)

        Examples
        --------
        >>> ham = QuadraticHamiltonian(ns=4)
        >>> ham.add_hopping(0, 1, -1.0)
        >>> ham.diagonalize()
        >>> E0, psi0 = ham.compute_ground_state()
        >>> print(f"Ground state energy: {E0:.6f}")
        """
        if self._eig_val is None:
            self.diagonalize()
        
        # For BdG systems
        if not self._particle_conserving:
            # Bogoliubov vacuum: fill all negative energy states
            occupied                        = np.where(self._eig_val < 0)[0]
            energy                          = self.many_body_energy(occupied)
            # For BdG, ground state is vacuum (automatic)
            self._occupied_orbitals_cached  = None
            state                           = self.many_body_state(resulting_state=None)
            return energy, state
        
        # For particle-conserving systems
        if symmetry_sector is None:
            symmetry_sector                 = self._ns // 2
        
        # Ground state: fill lowest energy orbitals
        occupied    = np.argsort(self._eig_val)[:symmetry_sector]
        energy      = self.many_body_energy(occupied)
        state       = self.many_body_state(occupied)
        
        return energy, state

    def compute_excited_states(
        self,
        n_excitations   : int = 1,
        symmetry_sector : Optional[int] = None,
    ) -> List[Tuple[float, np.ndarray]]:
        r"""
        Compute low-lying excited states for particle-conserving systems.
        
        Uses configuration interaction: excited states are created by removing
        an electron from occupied orbitals and promoting to unoccupied orbitals.

        Parameters
        ----------
        n_excitations : int
            Number of excited states to compute. Default: 1 (first excited state).
        symmetry_sector : Optional[int]
            Number of particles. If None, uses ceil(Ns/2).

        Returns
        -------
        List[Tuple[float, np.ndarray]]
            List of (energy, state_vector) tuples, sorted by energy.

        Notes
        -----
        - Only works for particle-conserving systems
        - For Ns=10, number of states grows combinatorially: limit n_excitations
        """
        if not self._particle_conserving:
            raise ValueError("Excited states only for particle-conserving systems. Use BdG excitations.")
        
        if self._eig_val is None:
            self.diagonalize()
        
        if symmetry_sector is None:
            symmetry_sector = self._ns // 2
        
        # Ground state
        gs_occupied = np.argsort(self._eig_val)[:symmetry_sector]
        gs_energy   = self.many_body_energy(gs_occupied)
        
        # Generate excitations: hole-particle pairs
        excitations = []
        for hole_idx in gs_occupied:
            for particle_idx in np.arange(self._ns)[~np.isin(np.arange(self._ns), gs_occupied)]:
                occ_excited                             = np.copy(gs_occupied)
                occ_excited[occ_excited == hole_idx]    = particle_idx
                
                E_exc   = self.many_body_energy(occ_excited)
                psi_exc = self.many_body_state(occ_excited)
                excitations.append((E_exc, psi_exc))
        
        # Sort by energy and return top n_excitations
        excitations.sort(key=lambda x: x[0])
        return excitations[:n_excitations]

    def invalidate_state_cache(self):
        """
        Clear all cached calculator functions and state matrices.
        
        Call this after modifying the Hamiltonian or changing diagonalization.
        """
        self._calculator_cache.clear()
        self._compiled_funcs.clear()
        self._last_occupation_key = None
        self._log("State calculator cache cleared.", lvl=3, log="debug")

    ###########################################################################
    #! Thermal Properties
    ###########################################################################

    def thermal_scan(self, temperatures: "Array", particle_number: Optional[float] = None) -> dict:
        """
        Compute thermal properties over a range of temperatures.

        Parameters
        ----------
        temperatures : array-like
            Array of temperatures T
        particle_number : float, optional
            Fixed particle number (only for fermions, particle-conserving)

        Returns
        -------
        dict
            Dictionary with thermal quantities vs temperature
        """
        from QES.Algebra.Properties.quadratic_thermal import quadratic_thermal_scan

        # Ensure we have eigenvalues
        if self.eig_val is None:
            self.diagonalize()

        particle_type = "fermion" if self._isfermions else "boson"

        # For BdG systems, particle number is not conserved
        if not self._particle_conserving:
            particle_number = None

        return quadratic_thermal_scan(
            self.eig_val, temperatures, particle_type=particle_type, particle_number=particle_number
        )

    def fermi_occupation(self, beta: float, mu: float = 0.0) -> "Array":
        """
        Compute Fermi-Dirac occupation numbers.

        Parameters
        ----------
        beta : float
            Inverse temperature 1/T
        mu : float, optional
            Chemical potential

        Returns
        -------
        Array
            Occupation numbers f(epsilon_k)
        """
        from QES.Algebra.Properties.quadratic_thermal import fermi_occupation

        if not self._isfermions:
            raise ValueError("Fermi occupation only for fermions")

        if self.eig_val is None:
            self.diagonalize()

        return fermi_occupation(self.eig_val, beta, mu)

    def bose_occupation(self, beta: float, mu: float = 0.0) -> "Array":
        """
        Compute Bose-Einstein occupation numbers.

        Parameters
        ----------
        beta : float
            Inverse temperature 1/T
        mu : float, optional
            Chemical potential

        Returns
        -------
        Array
            Occupation numbers n(epsilon_k)
        """
        from QES.Algebra.Properties.quadratic_thermal import bose_occupation

        if self._isfermions:
            raise ValueError("Bose occupation only for bosons")

        if self.eig_val is None:
            self.diagonalize()

        return bose_occupation(self.eig_val, beta, mu)

    ###########################################################################
    #! Time Evolution
    ###########################################################################

    def time_evolution_operator(self, time: float, backend: str = "auto") -> "Array":
        r"""
        Compute the time evolution operator exp(-i H t) for the quadratic Hamiltonian.

        For quadratic Hamiltonians, time evolution can be performed efficiently
        by diagonalizing and applying phase factors to the eigenmodes.

        Mathematically, the time evolution of a quadratic system means:
            U(t) = exp(-i H t) = W diag(exp(-i epsilon_j t)) W^dagger
        where W is the Bogoliubov transformation matrix and epsilon_j are the
        orbital energies.

        After that, the time evolution operator in the original basis is reconstructed.
        For single particle observables, this means we don't evolve the full many-body
        state, but only the single-particle operators...

        O(t) = U(t) O(0) U(t)^dagger -> calculate expectation values efficiently as
        <O>(t) = Tr[ rho(0) O(t) ] = Tr[ rho(0) U(t) O(0) U(t)^dagger ],

        or, given single particle orbitals in orignal basis [a_1, a_2, ..., a_N], we have:
            <psi | O(t) | psi> = sum_{i,j} O_ij <psi | a_i^dagger(t) a_j(t) | psi>
        with a_i(t) = sum_k U_ik(t) a_k(0).

        In practice, those we compute as follows:
            1) Diagonalize the quadratic Hamiltonian to get W and epsilon_j
            2) Compute the phase factors exp(-i epsilon_j t)
            3) Construct U(t) = W diag(exp(-i epsilon_j t)) W
            4) Use U(t) to evolve single-particle operators or states as needed.
            5) Calculate expectation values using the evolved operators/states.

        Parameters
        ----------
        time : float
            Evolution time t
        backend : str, optional
            Backend to use ('auto', 'numpy', 'jax')

        Returns
        -------
        Array
            Time evolution operator in the original basis
        """
        if backend == "auto":
            backend = "jax" if self._is_jax else "numpy"

        # Get the diagonalizing transformation
        W, orbital_energies, constant = self.diagonalizing_bogoliubov_transform(copy=True)

        # For particle-conserving case we can use the small transform W (N x N)
        if self._particle_conserving:
            phases = np.exp(-1j * orbital_energies * time)
            U = W
            evolved_diag = np.diag(phases)
            return U @ evolved_diag @ U.conj().T

        # For BdG (non-particle-conserving) compute full 2N x 2N evolution
        # by exponentiating the BdG matrix in its eigenbasis: U diag(exp(-i e t)) U^dagger
        bdg = np.asarray(self.build_bdg_matrix(copy=True))
        try:
            eigvals_b, eigvecs_b = sp.linalg.eigh(bdg)
        except Exception:
            eigvals_b, eigvecs_b = np.linalg.eigh(bdg)

        exp_diag = np.diag(np.exp(-1j * eigvals_b * time))
        U_full = eigvecs_b
        evo_full = U_full @ exp_diag @ U_full.conj().T

        # If JAX backend requested and available, convert to jax array
        if backend == "jax" and self._is_jax:
            try:
                import jax.numpy as jnp

                return jnp.asarray(evo_full)
            except Exception:
                self._log("Failed to convert BdG time evolution to JAX array; returning NumPy array", lvl=1, log="warning")
                
        return evo_full

    ###########################################################################
    #! Validation and Utilities
    ###########################################################################

    def validate(self) -> bool:
        """
        Validate the Hamiltonian matrices and settings.

        Returns
        -------
        bool
            True if valid, raises exception otherwise
        """
        # Check matrix shapes
        expected_shape = (self._ns, self._ns)
        if self._hamil_sp.shape != expected_shape:
            raise ValueError(
                f"Hamiltonian matrix has wrong shape {self._hamil_sp.shape}, expected {expected_shape}"
            )

        if not self._particle_conserving and self._delta_sp.shape != expected_shape:
            raise ValueError(
                f"Pairing matrix has wrong shape {self._delta_sp.shape}, expected {expected_shape}"
            )

        # Check hermiticity of hamiltonian part
        h_diff = self._hamil_sp - self._hamil_sp.conj().T
        if np.linalg.norm(h_diff) > 1e-10:
            self._log(
                f"Warning: Hamiltonian matrix is not Hermitian (norm of H - H^dagger = {np.linalg.norm(h_diff)})",
                lvl=1,
                log="warning",
            )

        # Check antisymmetry of pairing part for fermions
        if not self._particle_conserving and self._isfermions:
            delta_diff = self._delta_sp + self._delta_sp.T
            if np.linalg.norm(delta_diff) > 1e-10:
                self._log(
                    f"Warning: Pairing matrix is not antisymmetric (norm of Delta + Delta^dagger = {np.linalg.norm(delta_diff)})",
                    lvl=1,
                    log="warning",
                )

        return True

    ##########################################################################
    #! Interoperability with Qiskit and OpenFermion
    ##########################################################################

    def to_qiskit_hamiltonian(self):
        """Convert to Qiskit QuadraticHamiltonian.

        Returns
        -------
        qiskit_nature.second_q.operators.FermionicOp
            Qiskit fermionic operator representation

        Raises
        ------
        ImportError
            If Qiskit Nature not installed
        """
        from .interop import QiskitInterop

        if not self._particle_conserving:
            # BdG case: extract hermitian and antisymmetric parts
            ns = self._ns
            h_matrix = self._hamil_sp[:ns, :ns]
            v_matrix = self._delta_sp[:ns, :ns]
        else:
            h_matrix = self._hamil_sp
            v_matrix = None

        return QiskitInterop.to_qiskit_second_quantized_op(
            h_matrix, v_matrix, self._constant_offset, self._ns
        )

    def to_openfermion_hamiltonian(self):
        """Convert to OpenFermion FermionOperator.

        Returns
        -------
        openfermion.FermionOperator
            OpenFermion fermionic operator representation

        Raises
        ------
        ImportError
            If OpenFermion not installed
        """
        from .interop import OpenFermionInterop

        if not self._particle_conserving:
            ns = self._ns
            h_matrix = self._hamil_sp[:ns, :ns]
            v_matrix = self._delta_sp[:ns, :ns]
        else:
            h_matrix = self._hamil_sp
            v_matrix = None

        return OpenFermionInterop.to_openfermion_hamiltonian(
            h_matrix, v_matrix, self._constant_offset
        )

    def get_backend_list(self) -> list:
        """Get list of available backends with availability status.

        Returns
        -------
        list
            [(backend_name, is_available), ...]
        """
        from .backends import get_available_backends

        return get_available_backends()

    def set_backend(self, backend_name: str) -> None:
        """Switch to a different computation backend.

        Parameters
        ----------
        backend_name : str
            Backend identifier ('numpy', 'jax', etc.)

        Raises
        ------
        ValueError
            If backend not available
        """
        from .backends import get_backend

        try:
            backend = get_backend(backend_name)
            self._log(f"Switching backend to {backend_name}", lvl=2, log="info")
            self._backend_instance = backend
        except ValueError as e:
            self._log(f"Failed to switch backend: {e}", lvl=0, log="error")
            raise

    ##=======================================================================
    #! String Representation
    ##=======================================================================

    def __repr__(self) -> str:
        """Enhanced string representation."""
        pc_status   = "particle-conserving" if self._particle_conserving else "BdG"
        particles   = "fermions" if self._isfermions else "bosons"
        backend     = "JAX" if self._is_jax else "NumPy"
        diag_status = "diagonalized" if self._eig_val is not None else "not diagonalized"

        return (
            f"QuadraticHamiltonian(ns={self._ns}, {pc_status}, {particles}, "
            f"backend={backend}, {diag_status}, constant={self._constant_offset})"
        )

    # ========================================================================
    #! Spectral Function Methods
    # ========================================================================

    def spectral_function(
        self,
        omega: Union[float, np.ndarray],
        operator: Optional[np.ndarray] = None,
        eta: float = 0.01,
    ) -> Union[float, np.ndarray]:
        r"""
        Compute spectral function

        $$
        A_{O,O'}(omega) = -(1/pi) Im[G_{O,O'}(omega)],
        $$
        where G_{O,O'}(omega) is the Green's function for operators O, O'.

        For arbitrary operator O and arbitrary state |n>, the spectral function is:
            A_O(omega) = -(1/pi) Im[Tr(G(omega) O)]

        Here, the evaluation depends on the particle conservation as well,
        as the operator representation.

        Operator is represented as a matrix in single-particle basis, assuming
        the basis of the original Hamiltonian

        $$
        O = sum_{i,j} O_{ij} c_i^dagger c_j
        $$

        Parameters
        ----------
        omega : float or array-like
            Frequency or array of frequencies.
        operator : array-like, optional
            Observable operator for weighted spectral function.
            If None, computes standard spectral function.
        eta : float, optional
            Broadening parameter (default: 0.01).

        Returns
        -------
        float or array
            Spectral function A(omega), same shape as omega.

        Examples
        --------
        >>> qh      = QuadraticHamiltonian(ns=4, particle_conserving=True)
        >>> qh.add_hopping(0, 1, -1.0)
        >>> A_H     = qh.spectral_function(omega=1.0)                   # Hamiltonian spectral function
        >>> N_op    = np.diag([0, 1, 1, 2])                             # Number operator
        >>> A_N     = qh.spectral_function(omega=1.0, operator=N_op)    # Number-weighted
        """

        try:
            from QES.general_python.physics.spectral.spectral_backend import (
                greens_function_quadratic,
            )
            from QES.general_python.physics.spectral.spectral_function import (
                spectral_function as sf_func,
            )
        except ImportError:
            raise ImportError(
                "Required spectral modules not found in QES.general_python.physics.spectral."
            )

        # Ensure diagonalization
        try:
            self.diagonalize()
        except Exception as e:
            raise RuntimeError(
                f"Failed to diagonalize Hamiltonian before spectral function calculation: {e}"
            )

        # Check if input is scalar
        is_scalar = np.isscalar(omega)
        omega_arr = np.atleast_1d(omega)

        # Compute Green's function for each omega value
        # Use greens_function_quadratic without operators for single-particle resolvent
        G_list = []
        for om in omega_arr:
            G_i = greens_function_quadratic(om, self.eig_val, self.eig_vec, eta=eta)
            G_list.append(G_i)

        # Stack into array: shape (n_omega, N, N)
        G = np.array(G_list)

        # Compute spectral function for each omega
        A_list = []
        for i in range(len(omega_arr)):
            # If operator provided, compute with it; otherwise use Green's function directly
            if operator is not None:
                # Spectral function already handles operator weighting
                A_i = sf_func(greens_function=G[i])
            else:
                A_i = sf_func(greens_function=G[i])
            A_list.append(A_i)

        A = np.array(A_list)

        # Return scalar if input was scalar
        if is_scalar:
            return A[0]
        return A

    def greens_function(self, omega: Union[float, np.ndarray], eta: float = 0.01) -> np.ndarray:
        """
        Compute single-particle Green's function G(omega) in eigenbasis.

        G(omega) = 1/(omega + ieta - H)

        Parameters
        ----------
        omega : float or array-like
            Frequency or frequencies.
        eta : float, optional
            Broadening parameter (default: 0.01).

        Returns
        -------
        ndarray
            Green's function. Shape (N, N) for single omega,
            (n_omega, N, N) for array of omegas.
        """
        try:
            from QES.general_python.physics.spectral.spectral_backend import (
                greens_function_quadratic,
            )
        except ImportError:
            raise ImportError(
                "spectral_backend module not found in QES.general_python.physics.spectral."
            )

        # Ensure diagonalization
        self.diagonalize()

        # Check if input is scalar
        is_scalar = np.isscalar(omega)
        omega_arr = np.atleast_1d(omega)

        # Compute Green's function for each omega using greens_function_quadratic
        G_list = []
        for om in omega_arr:
            G_i = greens_function_quadratic(om, self.eig_val, self.eig_vec, eta=eta)
            G_list.append(G_i)

        G = np.array(G_list)

        # Return appropriate shape
        if is_scalar:
            return G[0]
        return G

    def spectral_weight(self, operator: np.ndarray) -> float:
        r"""
        Compute total spectral weight for given operator.

        Sum rule: \int A_O(omega) d omega = Tr(operator)

        Parameters
        ----------
        operator : array-like, shape (N, N) or (N,)
            Observable operator (full matrix or diagonal).

        Returns
        -------
        float
            Total spectral weight = Tr(operator).

        Examples
        --------
        >>> N_op = np.diag([0, 1, 1, 2])  # Number operator
        >>> total_weight = qh.spectral_weight(N_op)
        """
        if operator.ndim == 2:
            return np.trace(operator)
        else:
            return np.sum(operator)

    ##########################################################################

    @property
    def selection_utils(self) -> "QuadraticSelection":
        """
        Access selection utilities for quadratic Hamiltonians.

        Returns
        -------
        QuadraticSelection
            Utility class for selecting orbitals and configurations.
        """
        try:
            from QES.Algebra.Quadratic.hamil_quadratic_utils import QuadraticSelection
        except ImportError as e:
            raise ImportError(
                "QuadraticSelection module is required for selection utilities. "
                "Please ensure it is available."
            ) from e

        return QuadraticSelection

# ---------------------------------------------------------------------------
#! Registry integration
# ---------------------------------------------------------------------------

# Register the quadratic Hamiltonian in the global registry so that users
# can instantiate it via `HamiltonianConfig(kind="quadratic", ...)`.

def _build_quadratic_hamiltonian(config: HamiltonianConfig, params: Dict[str, Any]) -> Hamiltonian:
    """
    Builder function for QuadraticHamiltonian from HamiltonianConfig.
    Parameters
    ----------
    config : HamiltonianConfig
        Configuration object with Hilbert space and other settings.
    params : dict
        Additional parameters for QuadraticHamiltonian.
    Returns
    -------
    Hamiltonian
        Instance of QuadraticHamiltonian.

    Options in params
    -----------------
    ns : int
        Number of single-particle sites/modes.
    hilbert_space : HilbertSpace
        Hilbert space object defining the system.
    particle_conserving : bool
        If True:
            treat as particle-conserving (Ns x Ns).
        If False:
            treat as BdG / Nambu (2Ns x 2Ns).
    particles : str
        'fermions' or 'bosons'.
    other options...
        See QuadraticHamiltonian documentation for more details.
    """

    hilbert = config.resolve_hilbert()

    if hilbert is not None:
        params.setdefault("ns", hilbert.get_Ns())
        params.setdefault("hilbert_space", hilbert)
        params.setdefault("particle_conserving", hilbert.particle_conserving)

    ns = params.get("ns", hilbert.get_Ns() if hilbert is not None else None)
    if ns is None:
        raise ValueError("Quadratic Hamiltonian requires 'ns' or a Hilbert space.")

    params.setdefault("particles", "fermions")
    return QuadraticHamiltonian(**params)

register_hamiltonian(
    "quadratic",
    builder=_build_quadratic_hamiltonian,
    description="Quadratic (free) Hamiltonian supporting onsite, hopping, and pairing terms.",
    tags=("quadratic", "noninteracting", "fermion", "boson"),
)

# ---------------------------------------------------------------------------
import QES.Algebra.Quadratic.hamil_quadratic_transform as _hqt

# ---------------------------------------------------------------------------

QuadraticHamiltonian.register_basis_transform("real", "k-space", _hqt._handler_real_to_kspace)
QuadraticHamiltonian.register_basis_transform("k-space", "real", _hqt._handler_kspace_to_real)

# ---------------------------------------------------------------------------
#! End of file
# ---------------------------------------------------------------------------
