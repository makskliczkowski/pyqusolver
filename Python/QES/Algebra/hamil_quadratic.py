'''
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
'''

import numpy as np
import scipy as sp

from typing import Any, Dict, List, Tuple, Union, Optional, Sequence
from enum import Enum, unique
from abc import ABC
from functools import partial
from scipy.special import comb
from itertools import combinations
from collections import defaultdict
from scipy.stats import unitary_group
from dataclasses import dataclass, field

##############################################################################

@unique
class QuadraticTerm(Enum):
    '''
    Types of terms to be added to the quadratic Hamiltonian
    '''
    Onsite  =   0
    Hopping =   1
    Pairing =   2
    
    @property
    def mode_num(self):
        return 1 if self == QuadraticTerm.Onsite else 2

##############################################################################

@dataclass(frozen=True)
class SolvabilityInfo:
    """
    Information about closed-form solvability of a quadratic Hamiltonian.
    
    A Hamiltonian is considered 'solvable' if eigenvalues can be obtained
    via a closed-form analytical method rather than requiring full matrix
    diagonalization.
    
    Attributes
    ----------
    is_solvable : bool
        True if closed-form solution is available
    method : str
        Solution method:
        - 'diagonal': 
            Direct eigenvalues from diagonal elements
        - 'kspace': 
            Via k-space Fourier transform (translational symmetry)
        - 'analytical':
            Known analytical form for specific model
        - 'standard': 
            Requires standard matrix diagonalization
        - 'unknown': 
            Method unknown
    description : str
        Human-readable explanation of solvability status
    eigenvalues : Optional[np.ndarray]
        Pre-computed eigenvalues if directly available, else None
    sparsity_pattern : Optional[str]
        Description of matrix sparsity (e.g., 'band-diagonal', 'tridiagonal')
    computation_cost : Optional[str]
        Estimated computational complexity (e.g., 'O(n)', 'O(n log n)', 'O(n^3)')
    
    Examples
    --------
    >>> # Diagonal Hamiltonian
    >>> info = SolvabilityInfo(
    ...     is_solvable         =   True,
    ...     method              =   'diagonal',
    ...     description         =   'Hamiltonian is diagonal in computational basis',
    ...     eigenvalues         =   np.array([1.0, 2.0, 3.0]),
    ...     sparsity_pattern    =   'full-diagonal',
    ...     computation_cost    =   'O(n)'
    ... )
    
    >>> # Translational invariant system
    >>> info = SolvabilityInfo(
    ...     is_solvable         =   True,
    ...     method              =   'kspace',
    ...     description         =   'System has translational symmetry, solvable via FFT'
    ... )
    """
    is_solvable         : bool
    method              : str
    description         : str
    eigenvalues         : Optional[np.ndarray] = None
    sparsity_pattern    : Optional[str] = None
    computation_cost    : Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        status = "SOLVABLE" if self.is_solvable else "NOT_SOLVABLE"
        result = f"{status} ({self.method})\n{self.description}"
        if self.computation_cost:
            result += f"\n  Complexity: {self.computation_cost}"
        if self.sparsity_pattern:
            result += f"\n  Pattern: {self.sparsity_pattern}"
        return result
    
##############################################################################

try:
    from QES.Algebra.hamil import Hamiltonian, HilbertSpace, Lattice, JAX_AVAILABLE, Logger, Array
    from QES.Algebra.hamil_config import HamiltonianConfig, register_hamiltonian
    from QES.Algebra.Hilbert.hilbert_jit_states import (
        calculate_slater_det,
        bogolubov_decompose,
        pairing_matrix,
        calculate_bogoliubov_amp,
        calculate_bogoliubov_amp_exc,
        calculate_bosonic_gaussian_amp,
        calculate_permanent,
        many_body_state_full,
        many_body_state_mapping,
        many_body_state_closure,
        nrg_particle_conserving,
        nrg_bdg,    
    )   
except ImportError as e:
    raise ImportError("QES.Algebra.hamil and QES.Algebra.Hilbert.hilbert_jit_states modules are required but not found.") from e

# JAX interoperability

try:
    if JAX_AVAILABLE:
        import jax 
        import jax.numpy as jnp
        from jax.experimental.sparse import BCOO
        from QES.Algebra.Hilbert.hilbert_jit_states_jax import (
        calculate_slater_det_jax,                               # for calculating fermionic states
        calculate_bcs_amp_jax,                                  # for calculating BCS-like states
        calculate_permament_jax                                 # for calculating permanent states
        )
    else:
        raise ImportError("JAX is not available.")
except ImportError:
    jax                         = None
    jnp                         = np
    BCOO                        = None
    calculate_slater_det_jax    = None
    calculate_bcs_amp_jax       = None
    calculate_permament_jax     = None

# Common utilities

try:
    from QES.general_python.common.binary import int2base, base2int, extract as Extractor
    from QES.general_python.common import indices_from_mask, complement_indices
except ImportError:
    raise ImportError("QES.general_python.common module is required but not found.")

##############################################################################

@dataclass(frozen=True)
class QuadraticBlockDiagonalInfo:
    """
    Information for a single k-space block (or cell) of a quadratic Hamiltonian.
    
    Stores the diagonalized eigenvalues and eigenvectors for a small block
    (typically Nbtimes Nb for particle-conserving or 2Nbtimes 2Nb for BdG), along with
    the k-point coordinate. Useful for band structure analysis and 
    sector-specific computations.
    
    Attributes
    ----------
    point : np.ndarray
        The k-vector or cell index for this block. Shape (3,).
        For k-space: physical k-vector in reciprocal space.
        For real-space blocks: cell coordinates or index.
    frac_point : Optional[np.ndarray]
        The fractional k-vector (e.g., [kx/2pi, ky/2pi, kz/2pi]). Shape (3,).
    en : np.ndarray
        Eigenvalues of this block, sorted in ascending order. Shape (M,) where
        M is the block dimension (Nb for particle-conserving, 2Nb for BdG).
    ev : np.ndarray
        Eigenvectors of this block as columns. Shape (M, M) where ev[:, i] is
        the eigenvector for eigenvalue en[i].
    block_index : Optional[Tuple[int, int, int]]
        Index (ix, iy, iz) in the lattice momentum grid. Useful for indexing
        back into the original Bloch blocks. None if not from k-space.
    is_bdg : bool
        True if this is a BdG block (2Nb x 2Nb), False if particle-conserving (Nb x Nb).
    label : Optional[str]
        Optional label for this block (e.g., "Gamma", "M", "X" for special k-points).
    
    Examples
    --------
    >>> # From band structure calculation
    >>> info = QuadraticBlockDiagonalInfo(
    ...     point=np.array([0.0, 0.0, 0.0]),
    ...     en=np.array([-1.0, 0.0, 1.0]),
    ...     ev=np.eye(3),
    ...     block_index=(0, 0, 0),
    ...     is_bdg=False,
    ...     label="Gamma"
    ... )
    """
    point           : np.ndarray
    frac_point      : Optional[np.ndarray]              # Fractional k-point vector
    en              : np.ndarray
    ev              : np.ndarray
    block_index     : Optional[Tuple[int, int, int]]    = None
    is_bdg          : bool                              = False
    label           : Optional[str]                     = None
    
    def __post_init__(self):
        """Validate dimensions."""
        if self.point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {self.point.shape}")
        
        if len(self.en) != self.ev.shape[0] or self.ev.shape[0] != self.ev.shape[1]:
            raise ValueError(f"ev shape {self.ev.shape} inconsistent with en shape {self.en.shape}")
    
    def __str__(self) -> str:
        """String representation."""
        block_type  = "BdG" if self.is_bdg else "PC"
        idx_str     = f" (idx={self.block_index})" if self.block_index else ""
        label_str   = f" [{self.label}]" if self.label else ""
        frac_str    = f" (frac={self.frac_point})" if self.frac_point is not None else ""
        return (f"QuadraticBlockDiagonalInfo({block_type}){label_str}{idx_str}{frac_str}\n"
                f"  q-point: {self.point}\n"
                f"  eigenvalues: {self.en}\n"
                f"  eigenvector shape: {self.ev.shape}")

##############################################################################

class QuadraticSelection:
    '''Orbital selection utilities.'''
    def all_orbitals_size(n, k):
        """Binomial coefficient C(n, k)."""
        return comb(n, k, exact=True)

    def all_orbitals(n, k):
        """Generate all combinations of k orbitals from n."""
        if isinstance(n, (int, np.integer)):
            arange = np.arange(0, n, dtype = np.int64)
            return arange, combinations(arange, k)
        else:
            return n, combinations(n, k)

    def ran_orbitals(n, k, rng=None):
        """Randomly select k orbitals from n."""
        if isinstance(n, (int, np.integer)):
            arange = np.arange(0, n, dtype=np.int64)
        else:
            arange = np.array(n, dtype=np.int64)
        selected = rng.choice(arange, k, replace=False) if rng else np.random.choice(arange, k, replace=False)
        return arange, selected

    def mask_orbitals(ns        : int, 
                    n_occupation: Union[int, float] = 0.5,
                    *,
                    ordered     : bool = True, 
                    rng         : Optional[np.random.Generator] = None,
                    return_b    : bool = False) -> dict:
        """
        Generate a mask of occupied orbitals based on the number of orbitals and the occupation fraction.
        Parameters
        ----------
        n : int
            Total number of orbitals.
        n_occupation : int or float
            Number of occupied orbitals or fraction of occupied orbitals.
            If `n_occupation` is a float, it should be in the range [0, 1].
        ordered : bool
            If True, the mask will be ordered (0s followed by 1s).
            If False, the mask will be randomly shuffled.
        rng : numpy.random.Generator or None
            Random number generator for shuffling the mask.
            If None, uses the default random generator.
        Returns
            dict
                Dictionary with keys 'mask_a' (occupied orbitals in subsystem a)
                and 'mask_b' (occupied orbitals in subsystem b - if applicable).
        """
        if n_occupation < 0:
            raise ValueError("`n_occupation` must be non-negative.")
        elif n_occupation > ns:
            raise ValueError("`n_occupation` must be less than or equal to `ns`.")
        elif 0 < n_occupation < 1:
            n_occupation = int(n_occupation * ns)
            ordered                 = True

        out_dict = {}
        if ordered:
            mask_a  = np.arange(n_occupation)
        else:    
            mask_a  = np.sort(rng.choice(np.arange(ns), n_occupation, replace=False))
            
        out_dict['mask_a']      = mask_a
        out_dict['mask_a_1h']   = Extractor.to_one_hot(mask_a, ns)
        out_dict['mask_a_int']  = base2int(out_dict['mask_a_1h'], spin=False, spin_value=1)
        if return_b:
            mask_b                  = np.setdiff1d(np.arange(ns), mask_a)
            out_dict['mask_b']      = mask_b
            out_dict['mask_b_1h']   = Extractor.to_one_hot(mask_b, ns)
            out_dict['mask_b_int']  = base2int(out_dict['mask_b_1h'], spin=False, spin_value=1)
            out_dict['order']       = tuple(mask_a) + tuple(mask_b)
        return out_dict
    
    # ------------------------------------------------------------------------
    #! Haar random coefficients
    # ------------------------------------------------------------------------

    def haar_random_coeff(gamma : int,
                        *,
                        rng     : np.random.Generator | None = None,
                        dtype   = np.complex128) -> np.ndarray:
        r"""
        Return a length-gamma complex vector distributed with the **Haar
        measure** on the unit sphere (i.e. what you get from the first
        column of a Haar-random unitary).

        Parameters
        ----------
        gamma : int
            Dimension of states to mix.
        rng : numpy.random.Generator, optional
            Random-number generator to use. `np.random.default_rng()` is the
            default; pass your own for reproducibility.
        dtype : np.dtype, default=np.complex128
            Precision of the returned coefficients.

        Notes
        -----
        * **Mathematical equivalence** - Drawing
        $$\psi_i = x_i + i y_i \\ (x_i,y_i\\sim N(0,1))$$
        and normalising,  
        $$\\psi/\\lVert\\psi\\rVert$$  
        gives exactly the same distribution as the first column of a Haar
        unitary (see, e.g., Mezzadri 2006).
        * If SciPy >= 1.4 is available we use `scipy.stats.unitary_group.rvs`
        (QR-based) instead, but the Gaussian trick is used as a fallback and
        is typically faster.

        Examples
        --------
        >>> gen_random_state_coefficients(4) # doctest: +ELLIPSIS
        array([ 0.44...+0.05...j, -0.40...-0.11...j,  0.63...+0.48...j,
                0.16...+0.29...j])
        """
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")
        if gamma == 1:
            return np.ones(1, dtype=dtype)

        rng = np.random.default_rng() if rng is None else rng

        #! fast path: SciPy's true Haar unitary, if available
        try:
            from scipy.stats import unitary_group
            return unitary_group.rvs(gamma, random_state=rng).astype(dtype)[:, 0]
        except Exception:
            # fall back to Gaussian normalise trick  (still Haar-correct)
            z   = rng.normal(size=gamma) + 1j * rng.normal(size=gamma)
            z   = z.astype(dtype, copy=False)
            z  /= np.linalg.norm(z)
            return z

    def haar_random_unitary(gamma   : int,
                            *,
                            rng     : np.random.Generator | None = None,
                            dtype   = np.complex128) -> np.ndarray:
        r"""
        Generate a Haar-random unitary matrix of shape (gamma, gamma).

        Parameters
        ----------
        gamma : int
            Dimension of the unitary matrix.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        dtype : np.dtype, default=np.complex128
            Desired complex dtype.

        Returns
        -------
        np.ndarray
            Haar-distributed unitary matrix of shape (gamma, gamma).

        Notes
        -----
        If SciPy >= 1.4 is available, uses `scipy.stats.unitary_group.rvs`,
        which samples unitaries via QR decomposition with Haar measure
        (Mezzadri 2006). Otherwise, performs the QR-based method manually.

        Reference
        ---------
        Mezzadri, F. (2006). How to generate random matrices from the classical groups.
        Notices of the AMS, 54(5), 592-604.

        Examples
        --------
        >>> U = haar_random_unitary(4)
        >>> np.allclose(U.conj().T @ U, np.eye(4))
        True
        """
        
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")

        rng = np.random.default_rng() if rng is None else rng

        try:
            return unitary_group.rvs(gamma, random_state=rng).astype(dtype)
        except Exception:
            # Fallback: generate complex Ginibre matrix and QR-decompose
            z      = rng.normal(size=(gamma, gamma)) + 1j * rng.normal(size=(gamma, gamma))
            z      = z.astype(dtype, copy=False)
            q, r   = np.linalg.qr(z)
            # Normalize phases to ensure uniqueness
            d      = np.diag(r)
            ph     = d / np.abs(d)
            q     *= ph[np.newaxis, :]
            return q

    def haar_random_coeff_real(gamma    : int,
                                *,
                                rng     : np.random.Generator | None = None,
                                dtype   = np.float64) -> np.ndarray:
        r"""
        Return a length-gamma real vector distributed with the Haar measure
        on the unit sphere (i.e. what you get from the first column of a Haar
        random orthogonal matrix).

        Parameters
        ----------
        gamma : int
            Dimension of states to mix.
        rng : numpy.random.Generator, optional
            Random-number generator to use. `np.random.default_rng()` is the
            default; pass your own for reproducibility.
        dtype : np.dtype, default=np.float64
            Precision of the returned coefficients.

        Returns
        -------
        np.ndarray
            A real vector of length `gamma` sampled from the Haar measure.

        Notes
        -----
        Uses Gaussian normalisation trick to sample from the Haar measure.
        
        Examples
        --------
        >>> haar_random_coeff_real(4) # doctest: +ELLIPSIS
        array([ 0.44..., -0.40...,  0.63...,  0.16...])
        """
        
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")
        if gamma == 1:
            return np.ones(1, dtype=dtype)

        rng = np.random.default_rng() if rng is None else rng
        # Generate normally distributed real numbers - normalize to unit sphere
        z   = rng.normal(size=gamma)
        z   = z.astype(dtype, copy=False)
        z  /= np.linalg.norm(z)
        return z

    def haar_random_unitary_real(gamma   : int,
                                *,
                                rng     : np.random.Generator | None = None,
                                dtype   = np.float64) -> np.ndarray:
        r"""
        Generate a Haar-random real orthogonal matrix of shape (gamma, gamma).
        Parameters
        ----------
        gamma : int
            Dimension of the orthogonal matrix.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        dtype : np.dtype, default=np.float64
            Desired real dtype.

        Returns
        -------
        np.ndarray
            Haar-distributed orthogonal matrix of shape (gamma, gamma).

        Notes
        -----
        Uses QR decomposition of a Gaussian matrix to sample from the Haar measure.

        Examples
        --------
        >>> U = haar_random_unitary_real(4)
        >>> np.allclose(U.T @ U, np.eye(4))
        True
        """
        
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")

        rng     = np.random.default_rng() if rng is None else rng

        # Generate a random real orthogonal matrix using QR decomposition
        A       = rng.normal(size=(gamma, gamma))
        Q, R    = np.linalg.qr(A)
        d       = np.diag(R)
        Q      *= np.sign(d)
        
        # Return the first column as the Haar-random real vector
        return Q.astype(dtype, copy=False)

    # ------------------------------------------------------------------------
    #! Energy concerned
    # ------------------------------------------------------------------------
    
    def bin_energies(sorted_energies, digits: int = 10):
        """
        Groups indices of sorted energies by their rounded values.
        
        Args:
            sorted_energies (list[tuple[int, float]]):
                A list of (index, energy) pairs, sorted by energy.
            digits (int):
                Number of decimal digits for rounding - used for 
                binning the energies to avoid floating point errors.
        
        Returns:
            dict[float, list[int]]:
                Dictionary mapping rounded energy to list of indices.
        """
        binned = defaultdict(list)
        for idx, energy in sorted_energies:
            key = round(energy, digits)
            binned[key].append(idx)
        return dict(binned)
    
    def man_energies(binned_energies, dtype: np.dtype = np.int32):
        """
        Calculates the number of elements in each energy manifold from a dictionary of binned energies.
        Args:
            binned_energies (dict):
                A dictionary where keys represent energy bins and values are lists of items in each bin.
            dtype (np.dtype, optional):
                The desired data type for the output array of manifold sizes. Defaults to np.int32.
        Returns:
            tuple:
                - energy_manifolds (dict): A dictionary mapping each energy bin to the number of items in that bin.
                - energy_manifold_values (np.ndarray): An array containing the number of items in each energy bin, in the order of iteration.
        """
        
        energy_manifolds        = {}
        energy_manifold_values  = []
        for k, v in binned_energies.items():
            n                   = len(v)
            energy_manifolds[k] = n
            energy_manifold_values.append(n)
        energy_manifold_values = np.array(energy_manifold_values, dtype=dtype)
        return energy_manifolds, energy_manifold_values
    
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
    
    def __init__(self,
                ns                      : Optional[int]         = None,  # Ns is mandatory
                particle_conserving     : bool                  = True,
                dtype                   : Optional[np.dtype]    = None,
                backend                 : str                   = 'default',
                is_sparse               : bool                  = False,
                constant_offset         : float                 = 0.0,
                particles               : str                   = 'fermions',
                *,
                # Allow passing lattice/logger
                hilbert_space           : Optional[HilbertSpace]= None,
                lattice                 : Optional[Lattice]     = None,
                logger                  : Optional['Logger']    = None,
                seed                    : Optional[int]         = None,
                **kwargs):
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
        kwargs.pop('is_manybody', None)
        
        # Call base class init, explicitly setting is_manybody=False
        super().__init__(is_manybody    =   False,
                        ns              =   ns,
                        lattice         =   lattice,
                        hilbert_space   =   hilbert_space,
                        is_sparse       =   is_sparse,
                        dtype           =   dtype,
                        backend         =   backend,
                        seed            =   seed,
                        logger          =   logger,
                        **kwargs)

        # setup the arguments first
        self._particle_conserving   = particle_conserving
        self._constant_offset       = constant_offset
        self._isfermions            = particles.lower() == 'fermions'
        self._isbosons              = not self._isfermions
        if self._hilbert_space is not None and getattr(self._hilbert_space, 'particle_conserving', None) is not None and self._hilbert_space.particle_conserving != particle_conserving:
            # Hilbert space indicates different particle-conserving mode; warn and continue.
            self._log("Hilbert space particle_conserving flag differs from requested mode; continuing and trusting matrices.", lvl=1, log='warning')
        
        # Determine single-particle dimension and allocate storage.
        # Prefer an explicit dtype if given, otherwise inherit from Hilbert space
        # when available; fall back to complex for quadratic Hamiltonians.
        if self._dtype is None:
            if self._hilbert_space is not None:
                try:
                    self._dtype = getattr(self._hilbert_space, 'dtype', None) or self._backend.complex128
                except Exception:
                    self._dtype = self._backend.complex128
            else:
                self._dtype = self._backend.complex128
        self._hamil_sp_size         = self.ns
        self._hamil_sp_shape        = (self._hamil_sp_size, self._hamil_sp_size)
        self._dtypeint              = self._backend.int32 if self.ns < 2**32 - 1 else self._backend.int64

        # Initialize matrices as zero arrays instead of None for immediate usability
        xp                          = self._backend
        self._hamil_sp              = xp.zeros(self._hamil_sp_shape, dtype=self._dtype)
        self._delta_sp              = xp.zeros(self._hamil_sp_shape, dtype=self._dtype)
        if not particle_conserving:
            self._log('Initialized in BdG (Nambu) mode: matrices will use 2N\times2N structure.', lvl=2, log='info')

        self._name                      = f"QuadraticHamiltonian(Ns={self._ns},{'BdG' if not self._particle_conserving else 'N-conserving'})"
        self._occupied_orbitals_cached  = None
        self._diagonalization_requested = False
        self._F                         = None
        self._G                         = None
        self._U                         = None
        self._V                         = None
        self._mb_calculator             = None

    ##########################################################################
    #! Class methods for direct matrix initialization
    ##########################################################################

    @classmethod
    def from_hermitian_matrix(cls,
                            hermitian_part      : Array,
                            constant            : float = 0.0,
                            particles           : str = 'fermions',
                            dtype               : Optional[np.dtype] = None,
                            backend             : str = 'default',
                            **kwargs) -> 'QuadraticHamiltonian':
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
        hermitian_part = np.asarray(hermitian_part)
        if hermitian_part.ndim != 2 or hermitian_part.shape[0] != hermitian_part.shape[1]:
            raise ValueError("hermitian_part must be a square matrix")

        ns = hermitian_part.shape[0]

        # Create instance
        instance = cls(ns           =   ns,
                particle_conserving =   True,
                constant_offset     =   constant,
                particles           =   particles,
                dtype               =   dtype,
                backend             =   backend,
                **kwargs)

        # Set the matrix directly
        instance.set_single_particle_matrix(hermitian_part)

        return instance

    @classmethod
    def from_bdg_matrices(cls,
                         hermitian_part         : Array,
                         antisymmetric_part     : Array,
                         constant               : float = 0.0,
                         particles              : str = 'fermions',
                         dtype                  : Optional[np.dtype] = None,
                         backend                : str = 'default',
                         **kwargs) -> 'QuadraticHamiltonian':
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
        hermitian_part = np.asarray(hermitian_part)
        antisymmetric_part = np.asarray(antisymmetric_part)

        if hermitian_part.shape != antisymmetric_part.shape:
            raise ValueError("hermitian_part and antisymmetric_part must have the same shape")
        if hermitian_part.ndim != 2 or hermitian_part.shape[0] != hermitian_part.shape[1]:
            raise ValueError("Matrices must be square")

        ns = hermitian_part.shape[0]

        # Auto-detect particle conservation
        delta_norm          = np.linalg.norm(antisymmetric_part)
        particle_conserving = delta_norm < 1e-12

        # Create instance
        instance = cls(ns               =   ns,
                    particle_conserving =   particle_conserving,
                    constant_offset     =   constant,
                    particles           =   particles,
                    dtype               =   dtype,
                    backend             =   backend,
                    **kwargs)

        if particle_conserving:
            # Set as single particle matrix
            instance.set_single_particle_matrix(hermitian_part)
        else:
            # Set as BdG matrices
            instance.set_bdg_matrices(hermitian_part, antisymmetric_part)

        return instance

    ##########################################################################
    #! Build the Hamiltonian
    ##########################################################################

    def _invalidate_cache(self):
        """Wipe eigenvalues, eigenvectors, cached many-body calculator."""
        self._eig_val           = None
        self._eig_vec           = None
        self._mb_calculator     = None  # Lazy: will be recomputed when needed

    def add_term(self,
                term_type   : QuadraticTerm,
                sites       : tuple[int, ...] | list[int] | int,
                value       : complex,
                remove      : bool = False):
        """
        Adds a quadratic term to the Hamiltonian or pairing matrix.

        Parameters
        ----------
        term_type : QuadraticTerm
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
            
        val     = -value if remove else value
        valc    = val.conjugate() if isinstance(val, complex) or hasattr(val, "conjugate") else val

        if term_type is QuadraticTerm.Onsite:
            if len(sites) != 1:
                raise ValueError("Onsite term needs one index")
            i = sites[0]
            if self._is_numpy:
                self._hamil_sp[i, i] += val
            else:
                self._hamil_sp = self._hamil_sp.at[i, i].add(val)
                
        elif term_type is QuadraticTerm.Hopping:
            if len(sites) != 2:
                raise ValueError("Hopping term needs two indices")
            i, j = sites
            if self._is_numpy:
                self._hamil_sp[i, j] += val
                self._hamil_sp[j, i] += valc
            else:
                self._hamil_sp  = self._hamil_sp.at[i, j].add(val)
                self._hamil_sp = self._hamil_sp.at[j, i].add(valc)
                
        elif term_type is QuadraticTerm.Pairing:
            if self._particle_conserving:
                self._log("Pairing ignored: particle_conserving=True", lvl=2, log='warning')
                return
            if len(sites) != 2:
                raise ValueError("Pairing term needs two indices")
            i, j = sites
            if self._isfermions:  # antisymmetric
                if self._is_numpy:
                    self._delta_sp[i, j] += value
                    self._delta_sp[j, i] -= value
                else:
                    self._delta_sp = self._delta_sp.at[i, j].add(value)
                    self._delta_sp = self._delta_sp.at[j, i].add(-value)
            else:  # bosons: symmetric
                if self._is_numpy:
                    self._delta_sp[i, j] += value
                    self._delta_sp[j, i] += value
                else:
                    self._delta_sp = self._delta_sp.at[i, j].add(value)
                    self._delta_sp = self._delta_sp.at[j, i].add(value)
                    
        else:
            raise TypeError(term_type)
        self._invalidate_cache()
        self._log(f"add_term: {term_type.name} {sites} {str(value)}", lvl=3, log='debug')

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

    def to_basis(self, basis_type: str, enforce: bool = False, sublattice_positions: Optional[np.ndarray] = None, **kwargs):
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
        kwargs['sublattice_positions'] = sublattice_positions
        # Call parent's general to_basis() which will dispatch to registered handlers
        return super().to_basis(basis_type, enforce=enforce, **kwargs)

    def _transform_real_to_kspace(self, enforce: bool = False, **kwargs) -> 'QuadraticHamiltonian':
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
            raise NotImplementedError("Auto-lattice creation not yet fully implemented. Please pass lattice explicitly.")
        
        # Build real-space matrix if needed
        H_real = self.build_single_particle_matrix(copy=True)
        
        # Convert sparse matrix to dense if needed
        # The lattice code expects dense numpy arrays
        if sp.sparse.issparse(H_real):
            self._log(f"Converting sparse matrix to dense for Bloch transform", lvl=2, color="yellow")
            H_real = H_real.toarray()
        
        # Apply Bloch transform with extract_bands=False to get full NsxNs matrices
        self._log(f"Applying FFT-based Bloch transform on {H_real.shape} matrix (full NsxNs mode)", lvl=2, color="cyan")
        (H_k, k_grid, k_grid_frac)  = self._lattice.kspace_from_realspace(H_real, block_diag=True)  
        
        # Store transformed representation using general attributes
        self._hamil_transformed     = H_k
        self._transformed_grid      = k_grid
        self._transformed_grid_frac = k_grid_frac
        self._is_transformed        = True
        self._current_basis         = HilbertBasisType.KSPACE
        
        # Clear old real-space matrices if they exist (optional, for memory)
        # self._hamil_sp = None  # Uncomment if you want to free memory
        
        self._log(f"Bloch transform complete: H_k shape = {H_k.shape}, k_grid shape = {k_grid.shape}", lvl=2, color="green")
        
        # Push k-space basis to HilbertSpace (don't sync FROM it)
        self.push_basis_to_hilbert_space()
        
        return self

    def _transform_kspace_to_real(self, **kwargs) -> 'QuadraticHamiltonian':
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
            raise ValueError("Cannot transform k-space to real: no transformed Hamiltonian stored in _hamil_transformed")
        
        # Apply inverse FFT
        self._log("Applying inverse FFT to reconstruct real-space Hamiltonian", lvl=2, color="cyan")
        H_real = self._lattice.realspace_from_kspace(self._hamil_transformed)
        
        # Update in-place
        self.set_single_particle_matrix(H_real)
        
        # Update basis tracking flags
        self._is_transformed = False
        self._current_basis  = HilbertBasisType.REAL
        
        # Clear transformed storage (optional, for memory)
        # self._hamil_transformed = None
        # self._transformed_grid = None
        
        self._log(f"Inverse Bloch transform complete: H_real shape = {H_real.shape}", lvl=2, color="green")
        
        # Push real-space basis to HilbertSpace (don't sync FROM it)
        self.push_basis_to_hilbert_space()
        
        return self

    def set_basis_type(self, basis_type: str):
        """
        Override: Set basis type and propagate to Hilbert space if available.
        """
        super().set_basis_type(basis_type)
        
        # Propagate to Hilbert space
        if self._hilbert_space is not None and hasattr(self._hilbert_space, 'set_basis'):
            self._hilbert_space.set_basis(basis_type)

    ##########################################################################
    #! Basis transformation state queries
    ##########################################################################

    def set_single_particle_matrix(self, H: Array):
        if not self._particle_conserving:
            raise RuntimeError("Use set_bdg_matrices for non-conserving case")
        if H.shape != (self._ns, self._ns):
            raise ValueError(f"shape mismatch, expected {(self._ns, self._ns)}")
        self._hamil_sp = self._backend.array(H, dtype=self._dtype)
        self._invalidate_cache()
        self._log(f"set_single_particle_matrix: {H.shape}", lvl=3, log='debug')

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
        
        self._hamil_sp      = self._backend.array(K, dtype=self._dtype)
        self._delta_sp      = self._backend.array(Delta, dtype=self._dtype)
        self._invalidate_cache()
        self._log(f"set_bdg_matrices: {K.shape}, {Delta.shape}", lvl=3, log='debug')
        
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
        
        xp  = self._backend
        bdg = xp.block([
                [self._hamil_sp,                 self._delta_sp],
                [-xp.conjugate(self._delta_sp), -xp.conjugate(self._hamil_sp.T)],
            ])
        return xp.array(bdg) if copy else bdg

    ############################################################################
    
    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        """
        Assemble the quadratic Hamiltonian matrix prior to diagonalization.
        """
        if self._particle_conserving:
            self._hamil = self._hamil_sp
        else:
            self._hamil = self.build_bdg_matrix(copy=False)

    ###########################################################################
    #! Solvability Detection
    ###########################################################################
    
    def solvable(self) -> SolvabilityInfo:
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
        # Check if already diagonalized
        if self._eig_val is not None and self._eig_vec is not None:
            offset_str = f" (offset={self._constant_offset})" if self._constant_offset != 0 else ""
            return SolvabilityInfo(
                is_solvable         =   True,
                method              =   'analytical',
                description         =   f'Eigenvalues already computed via prior diagonalization{offset_str}',
                eigenvalues         =   self._eig_val.copy(),
                computation_cost    =   'O(1) - cached'
            )
        
        # Get matrix for analysis
        try:
            self._hamiltonian_quadratic()
            H = self._hamil
        except Exception:
            return SolvabilityInfo(
                is_solvable         =   False,
                method              =   'unknown',
                description         =   'Could not assemble Hamiltonian matrix for analysis',
                computation_cost    =   'O(n^3) - standard diagonalization'
            )
        
        if H is None:
            return SolvabilityInfo(
                is_solvable         =   False,
                method              =   'unknown',
                description         =   'Hamiltonian matrix not available',
                computation_cost    =   'O(n^3) - standard diagonalization'
            )
        
        # Convert to dense if sparse for analysis
        H_dense = H.toarray() if sp.sparse.issparse(H) else np.asarray(H)
        n = H_dense.shape[0]
        
        # Check if diagonal (or nearly diagonal)
        off_diag_norm = np.linalg.norm(H_dense - np.diag(np.diag(H_dense)))
        is_diagonal = off_diag_norm < 1e-10 * np.linalg.norm(H_dense) if np.linalg.norm(H_dense) > 1e-10 else off_diag_norm < 1e-14
        
        if is_diagonal:
            eigenvalues = np.diag(H_dense) + self._constant_offset
            return SolvabilityInfo(
                is_solvable=True,
                method='diagonal',
                description='Hamiltonian is diagonal in current basis; eigenvalues = diagonal elements',
                eigenvalues=eigenvalues,
                sparsity_pattern='full-diagonal',
                computation_cost='O(n)'
            )
        
        # Check if tridiagonal (or nearly tridiagonal)
        upper_off_diag = np.triu(H_dense, 2)
        lower_off_diag = np.tril(H_dense, -2)
        band_norm = np.linalg.norm(upper_off_diag) + np.linalg.norm(lower_off_diag)
        is_tridiagonal = band_norm < 1e-10 * np.linalg.norm(H_dense) if np.linalg.norm(H_dense) > 1e-10 else band_norm < 1e-14
        
        if is_tridiagonal:
            return SolvabilityInfo(
                is_solvable=True,
                method='kspace',
                description='Hamiltonian is tridiagonal; can be solved via specialized eigensolvers',
                sparsity_pattern='tridiagonal',
                computation_cost='O(n^2)'
            )
        
        # Check sparsity level
        sparsity = 1.0 - np.count_nonzero(H_dense) / H_dense.size
        if sparsity > 0.9:
            return SolvabilityInfo(
                is_solvable=True,
                method='kspace',
                description=f'Hamiltonian is highly sparse ({100*sparsity:.1f}%); Krylov subspace methods efficient',
                sparsity_pattern='highly-sparse',
                computation_cost='O(n^2) - sparse iterative methods'
            )
        
        # Check for band structure (common in lattice models)
        max_band_width = 0
        for i in range(n):
            for j in range(n):
                if abs(H_dense[i, j]) > 1e-14:
                    max_band_width = max(max_band_width, abs(i - j))
        
        if max_band_width < 0.3 * n:  # Narrow band
            return SolvabilityInfo(
                is_solvable=True,
                method='kspace',
                description=f'Hamiltonian has band structure (bandwidth ^ {max_band_width}); efficient via structured methods',
                sparsity_pattern=f'band-diagonal (width={max_band_width})',
                computation_cost='O(n^2) - band solver'
            )
        
        # Check for Hermitian structure (all quadratic Hamiltonians should be Hermitian)
        is_hermitian = np.allclose(H_dense, H_dense.conj().T, atol=1e-10)
        
        if is_hermitian:
            return SolvabilityInfo(
                is_solvable=True,
                method='standard',
                description='Quadratic system is Hermitian; solvable via standard eigendecomposition',
                sparsity_pattern='generic-hermitian' if sparsity < 0.1 else 'sparse-hermitian',
                computation_cost='O(n^3) - dense' if sparsity < 0.1 else 'O(n^2) - sparse methods'
            )
        
        # Fallback
        return SolvabilityInfo(
            is_solvable=True,
            method='standard',
            description='Quadratic Hamiltonian requires standard matrix diagonalization',
            sparsity_pattern='generic',
            computation_cost='O(n^3)'
        )

    ###########################################################################
    #! Diagonalization
    ###########################################################################

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
                self._log("Performing diagonalization...", lvl=2, log='info')
            
            # Ensure the quadratic matrix is assembled before diagonalization
            try:
                if (not self._hamil_sp or (not self._particle_conserving and not self._delta_sp)) and not force:
                    self._hamiltonian_quadratic()
            except Exception:
                # best-effort; if build fails, let base class handle errors
                pass

            # Calls base diagonalize on self.hamil (which is _hamil_sp or BdG)
            super().diagonalize(verbose=verbose, **kwargs)
            
            # Apply constant offset after diagonalization
            if self._eig_val is not None and self._constant_offset != 0.0:
                if verbose:
                    self._log(f"Adding constant offset {self._constant_offset} to eigenvalues.", lvl=2, log='debug')
                self._eig_val += self._constant_offset
                # Recalculate energy stats if offset was applied
                self._calculate_av_en()
                return
        elif verbose:
            self._log("Using cached diagonalization results.", lvl=2, log='debug')
        self._log("Diagonalization was used and it was not forced...", lvl=3, log='warning')

    @property
    def eig_val(self):
        """Eigenvalues (triggers diagonalization if not yet performed)."""
        if self._eig_val is None:
            self.diagonalize()
        return self._eig_val
    
    @property 
    def eig_vec(self):
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
            idx_by_abs          = np.argsort(np.abs(eigvals))[-N:]
            energies            = np.abs(eigvals[idx_by_abs])

            # Order energies ascending for consistent output
            order               = np.argsort(energies)
            selected_idx        = idx_by_abs[order]
            orbital_energies    = energies[order]

            # Corresponding eigenvectors (columns) -> shape (2N, N)
            psi                 = eigvecs[:, selected_idx]

            # Split into particle (u) and hole (v) components
            U_mat               = psi[:N, :]
            V_mat               = psi[N:, :]

            # Build transformation matrix W_small of shape (N, 2N): row k = [u_k^T, v_k^T]
            W_small             = np.hstack((U_mat.conj().T, V_mat.conj().T))

            W                   = W_small if copy else self._backend.array(W_small)
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

    def block_diagonal_bdg(self) -> Tuple[List[QuadraticBlockDiagonalInfo], np.ndarray]:
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
        # Check that we're in k-space
        if self._hamil_transformed is None:
            raise RuntimeError(
                "Hamiltonian not in k-space. Call to_basis('k-space') first."
            )
        
        H_k         = self._hamil_transformed       # Shape: (Lx, Ly, Lz, Nb, Nb) or (Lx, Ly, Lz, 2*Nb, 2*Nb)
        k_grid      = self._transformed_grid        # Shape: (Lx, Ly, Lz, 3)
        k_grid_frac = self._transformed_grid_frac   # Shape: (Lx, Ly, Lz, 3) fractional coords
        
        if H_k is None or k_grid is None:
            raise ValueError("Transformed Hamiltonian or k-grid not available. Ensure to_basis('k-space') was successful.")
        
        # Determine block type
        # nb_sublattices  = H_k.shape[3] // (2 if not self._particle_conserving else 1)
        is_bdg      = not self._particle_conserving
        energies    = []
        
        results: List[QuadraticBlockDiagonalInfo] = []
        
        # H_k and k_grid are both in fftfreq order (unshifted)
        # Gamma point is at index [0,0,0] for both
        # Iterate over all k-points
        for i in range(H_k.shape[0]):
            for j in range(H_k.shape[1]):
                for k in range(H_k.shape[2]):
                    k_vec       = np.asarray(k_grid[i, j, k, :],        dtype=np.float64)
                    k_vec_frac  = np.asarray(k_grid_frac[i, j, k, :],   dtype=np.float64)
                    
                    # Use H_k directly (both in fftfreq order)
                    H_block     = np.asarray(H_k[i, j, k, :, :], dtype=self._dtype)

                    # Diagonalize this block
                    try:
                        eigvals, eigvecs = sp.linalg.eigh(H_block)
                    except Exception:
                        eigvals, eigvecs = np.linalg.eigh(H_block)

                    # Create info object
                    info = QuadraticBlockDiagonalInfo(
                        point       =   k_vec,      # k-point vector
                        frac_point  =   k_vec_frac, # Fractional k-point vector
                        en          =   eigvals,    # Eigenvalues
                        ev          =   eigvecs,    # Columns are eigenvectors
                        block_index =   (i, j, k),  # Indices in k-grid
                        is_bdg      =   is_bdg,     # True if BdG block, False if PC
                        label       =   None,       # Special label?
                    )
                    
                    energies.append(eigvals)
                    results.append(info)
        
        self._log(
            f"Extracted and diagonalized {len(results)} k-space blocks "
            f"({'BdG' if is_bdg else 'PC'}, {H_k.shape[3]}x{H_k.shape[3]})",
            lvl=2, color="cyan"
        )

        return results, np.array(energies)

    ###########################################################################
    #! Transformation Preparation
    ###########################################################################

    @dataclass(frozen=True)
    class PCTransform:
        """Particle-conserving transformation handle."""
        W           : np.ndarray          # (Ns, Ns) single-particle eigenvectors (unitary)
        occ_idx     : np.ndarray          # (Na,)
        unocc_idx   : np.ndarray          # (Ns-Na,)

        # on-demand helpers (allocate only when used)

        def W_A(self) -> np.ndarray:
            """Form columns W[:, occ_idx]. NOTE: column gather copies."""
            return np.take(self.W, self.occ_idx, axis=1)

        def W_A_CT(self) -> np.ndarray:
            """Conjugate transpose of W_A (allocates as above)."""
            WA = np.take(self.W, self.occ_idx, axis=1)
            return WA.conj().T

        def order_occ_then_unocc(self) -> np.ndarray:
            """Permutation indices [occ, unocc]."""
            return np.concatenate((self.occ_idx, self.unocc_idx), dtype=np.int64)

    @dataclass(frozen=True)
    class BdGTransform:
        """BdG/Nambu transform handle, avoids big temporaries."""
        W           : np.ndarray            # (2N, 2N) quasiparticle eigenvectors in Nambu basis
        N           : int                   # single-particle dimension
        occ_idx     : np.ndarray            # subset in physical modes (0..N-1)
        unocc_idx   : np.ndarray            # complement in physical modes

        # Block views (no copies): U= W[:N,:N], V= W[:N,N:]
        @property
        def U(self) -> np.ndarray:
            return self.W[:self.N, :self.N]

        @property
        def V(self) -> np.ndarray:
            return self.W[:self.N, self.N:]

        # ---- Prefer row slicing (views) when projecting to a *mode subset* ----

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

    def prepare_transformation(self, occ, *, bdg: bool | None = None):
        """
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
        if bdg is None:
            bdg = not self._particle_conserving

        W = self._eig_vec
        if W is None:
            raise RuntimeError("Eigenvectors not available. Call diagonalize() first.")

        if bdg:
            # BdG / Nambu path
            if W.ndim != 2 or W.shape[0] != W.shape[1] or (W.shape[0] % 2 != 0):
                raise ValueError(f"Expect square (2N,2N) BdG eigenvector matrix; got {W.shape}")
            
            twoN        = W.shape[0]
            N           = twoN // 2

            occ_idx     = indices_from_mask(occ, N)
            unocc_idx   = complement_indices(N, occ_idx)
            return self.BdGTransform(W=W, N=N, occ_idx=occ_idx, unocc_idx=unocc_idx)

        # Particle-conserving path
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"Expect square (Ns,Ns) eigenvector matrix; got {W.shape}")
        Ns          = W.shape[1]
        occ_idx     = indices_from_mask(occ, Ns)
        unocc_idx   = complement_indices(Ns, occ_idx)
        return self.PCTransform(W=W, occ_idx=occ_idx, unocc_idx=unocc_idx)
    
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
        
        if self.eig_val is None:
            raise ValueError("Single-particle eigenvalues not calculated. Call diagonalize() first.")
        if isinstance(occupied_orbitals, int):
            occupied_orbitals = int2base(occupied_orbitals, self._ns, spin=False, spin_value=1, backend=self._backend).astype(self._dtypeint)
            
        occ = np.asarray(occupied_orbitals, dtype=self._dtypeint)
        if occ.shape[0] == 0:
            return 0.0
        
        if occ.ndim != 1:
            raise ValueError("occupied_orbitals must be 1-D")
        e   = 0.0
        
        if self._is_jax:
            occ     = jnp.asarray(occ, dtype=self._dtypeint)
            vmax    = self._eig_val.shape[0]

            def _check_bounds(x):
                if int(jnp.min(x)) < 0 or int(jnp.max(x)) >= vmax:
                    raise IndexError("orbital index out of bounds")
                return x
            occ = _check_bounds(occ)

            if self._particle_conserving:
                e = jnp.sum(self._eig_val[occ])
            else:
                if int(jnp.max(occ)) >= self._ns:
                    raise IndexError("BdG index must be in 0...Ns-1")
                mid = self._ns - 1
                e   = jnp.sum(self._eig_val[mid + occ + 1] -
                            self._eig_val[mid - occ])
        else:
            vmax = self._eig_val.shape[0]
            if occ.min() < 0 or occ.max() >= vmax:
                raise IndexError("orbital index out of bounds")

            if self._particle_conserving:
                e = nrg_particle_conserving(self._eig_val, occ)
            else:
                if occ.max() >= self._ns:
                    raise IndexError("BdG index must be in 0...Ns-1 (positive branch)")
                e = nrg_bdg(self._eig_val, self._ns, occ)
        return self._backend.real(e) + self._constant_offset

    def many_body_energies(self, n_occupation: Union[float, int] = 0.5, 
                    nh: Optional[int] = None, use_combinations: bool = False) -> dict[int, float]:
        '''
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
        nh : int, optional
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
        '''
        
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
                occupied_orbitals       = int2base(i, self._ns, spin=False, spin_value=1).astype(self._dtypeint)
                occupied_orbitals       = np.nonzero(occupied_orbitals)[0]
                if len(occupied_orbitals) != n_occupation:
                    continue
                many_body_energies[i]   = self.many_body_energy(occupied_orbitals)
            return many_body_energies
        #! 2. Use combinations
        else:
            all_combinations        = QuadraticSelection.all_orbitals(self._ns, n_occupation)
            for i, occupied_orbitals in enumerate(all_combinations[1]):
                occupied_orbitals       = np.array(occupied_orbitals, dtype=self._dtypeint)
                many_body_energies[i]   = self.many_body_energy(occupied_orbitals)        
            return many_body_energies

    ###########################################################################
    #! Many-Body State Calculation
    ###########################################################################
    
    def _many_body_state_calculator(self):
        r"""
        Return a function object that implements:
            psi = calc(matrix_arg, basis_state_int, ns)
        
        together with the constant matrix_arg it needs.

        The closure is JIT-compatible with Numba (nopython=True) when
        self._is_numpy is True; otherwise the returned function calls the
        JAX variant of the same kernel.
        
        The function is used to calculate the many-body state vector
        in the Fock basis (or other specified basis) from the single-particle
        eigenvectors and the occupied orbitals.
        """

        #! Fermions
        if self._isfermions:
            if self._particle_conserving:
                #! Slater determinant needs (U, \alpha_k) - U is the matrix of eigenvectors
                if not hasattr(self, "_occupied_orbitals_cached"):
                    raise RuntimeError( "call many_body_state(...) with "
                                        "`occupied_orbitals` first.")
                occ = self._occupied_orbitals_cached

                if self._is_numpy:
                    calc = many_body_state_closure (
                            calculator_func = calculate_slater_det,
                            matrix_arg      = occ)
                else:
                    calc = lambda U, st, _ns: calculate_slater_det_jax(U, occ, st, _ns)
                return calc, self._eig_vec
            else:
                #! Bogoliubov vacuum / Pfaffian
                if self._is_numpy:
                    if self._F is None:
                        if self._U is None or self._V is None:
                            self._U, self._V, _ = bogolubov_decompose(self._eig_val, self._eig_vec)
                        self._F = pairing_matrix(self._U, self._V)
                    calc = lambda F, st, _ns: calculate_bogoliubov_amp(F, st, _ns)
                else:
                    raise NotImplementedError("JAX Bogoliubov vacuum calculation not implemented.")
                    # calc = lambda F, st, _ns: calculate_bogoliubov_amp_jax(F, st, _ns)
                return calc, self._F

        #! Bosons 
        if self._isbosons:
            if self._particle_conserving:
                #! Permanent / Gaussian state
                if self._is_numpy:
                    calc = lambda G, st, _ns: calculate_permanent(G, st, _ns)
                else:
                    calc = lambda G, st, _ns: calculate_permament_jax(G, st, _ns)
                return calc, self._eig_vec
            else:
                #! Gaussian squeezed vacuum / Hafnian
                if self._is_numpy:
                    if self._G is None:
                        if self._U is None or self._V is None:
                            self._U, self._V, _ = bogolubov_decompose(self._eig_val, self._eig_vec)
                        self._G = pairing_matrix(self._eig_val, self._eig_vec)
                        
                    calc = lambda G, st, _ns: calculate_bosonic_gaussian_amp(G, st, _ns)
                else:
                    raise NotImplementedError("JAX Bosonic Gaussian state calculation not implemented.")
                return calc, self._G
    
    def many_body_state(self,
                        occupied_orbitals : Union[list[int], np.ndarray] | None = None,
                        target_basis      : str                                 = "sites",
                        many_body_hs      : Optional[HilbertSpace]              = None,
                        resulting_state   : Optional[np.ndarray]                = None):
        """
        Return the coefficient vector |Psi> in the computational basis.

        Parameters
        ----------
        occupied_orbitals
            For particle-conserving fermions/bosons: list/array of alpha_k.
            Ignored otherwise.
        target_basis
            Currently only "sites" is supported.
        many_body_hs
            If provided, must expose mapping -> 1-D np.ndarray.
            The output vector is ordered according to that mapping.
            If None, a full vector of length 2**ns is produced.
        batch_size
            If >0, the Fock space is processed in slices of that length
            to keep peak memory low. 0 (default) disables batching.

        Returns
        -------
        np.ndarray
            Coefficient vector psi(x).
        """
        if target_basis != "sites":
            raise NotImplementedError("Only the site/bitstring basis is implemented for now.")

        # If new occupied_orbitals are provided, or the cached state is missing (e.g., after cache invalidation)
        if occupied_orbitals is not None or self._occupied_orbitals_cached is None:
            if isinstance(occupied_orbitals, (list, np.integer)):
                # transform to array
                self._occupied_orbitals_cached = int2base(occupied_orbitals, self._ns, spin=False, backend=self._backend).astype(self._dtypeint)
            else:
                self._occupied_orbitals_cached = np.ascontiguousarray(occupied_orbitals, dtype=self._dtypeint)
        
        #! obtain (calculator, matrix_arg)
        calculator, matrix_arg = self._many_body_state_calculator()

        #! choose mapping / dimensions
        ns           = self._ns
        dtype        = getattr(self, "_dtype", np.result_type(matrix_arg))
        if resulting_state is not None:
            dtype = np.result_type(resulting_state, dtype)
        
        if many_body_hs is None or not many_body_hs.modifies:
            return many_body_state_full(matrix_arg, calculator, ns, resulting_state, dtype=dtype)
        else:            
            mapping = many_body_hs.mapping
            return many_body_state_mapping(matrix_arg,
                                        calculator,
                                        mapping,
                                        ns,
                                        dtype)
        return None # should not be reached

    ###########################################################################
    #! Thermal Properties
    ###########################################################################

    def thermal_scan(self, temperatures: Array, particle_number: Optional[float] = None) -> dict:
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

        particle_type = 'fermion' if self._isfermions else 'boson'

        # For BdG systems, particle number is not conserved
        if not self._particle_conserving:
            particle_number = None

        return quadratic_thermal_scan(
            self.eig_val,
            temperatures,
            particle_type=particle_type,
            particle_number=particle_number
        )

    def fermi_occupation(self, beta: float, mu: float = 0.0) -> Array:
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

    def bose_occupation(self, beta: float, mu: float = 0.0) -> Array:
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

    def time_evolution_operator(self, time: float, backend: str = 'auto') -> Array:
        """
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
        if backend == 'auto':
            backend = 'jax' if self._is_jax else 'numpy'

        # Get the diagonalizing transformation
        W, orbital_energies, constant = self.diagonalizing_bogoliubov_transform(copy=True)

        # For particle-conserving case we can use the small transform W (N x N)
        if self._particle_conserving:
            phases          = np.exp(-1j * orbital_energies * time)
            U               = W
            evolved_diag    = np.diag(phases)
            return U @ evolved_diag @ U.conj().T

        # For BdG (non-particle-conserving) compute full 2N x 2N evolution
        # by exponentiating the BdG matrix in its eigenbasis: U diag(exp(-i e t)) U^dagger
        bdg = np.asarray(self.build_bdg_matrix(copy=True))
        try:
            eigvals_b, eigvecs_b = sp.linalg.eigh(bdg)
        except Exception:
            eigvals_b, eigvecs_b = np.linalg.eigh(bdg)

        exp_diag    = np.diag(np.exp(-1j * eigvals_b * time))
        U_full      = eigvecs_b
        evo_full    = U_full @ exp_diag @ U_full.conj().T

        # If JAX backend requested and available, convert to jax array
        if backend == 'jax' and self._is_jax:
            try:
                import jax.numpy as jnp
                return jnp.asarray(evo_full)
            except Exception:
                self._log('Failed to convert BdG time evolution to JAX array; returning NumPy array', lvl=1, log='warning')

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
            raise ValueError(f"Hamiltonian matrix has wrong shape {self._hamil_sp.shape}, expected {expected_shape}")

        if not self._particle_conserving and self._delta_sp.shape != expected_shape:
            raise ValueError(f"Pairing matrix has wrong shape {self._delta_sp.shape}, expected {expected_shape}")

        # Check hermiticity of hamiltonian part
        h_diff = self._hamil_sp - self._hamil_sp.conj().T
        if np.linalg.norm(h_diff) > 1e-10:
            self._log(f"Warning: Hamiltonian matrix is not Hermitian (norm of H - H^dagger = {np.linalg.norm(h_diff)})", lvl=1, log='warning')

        # Check antisymmetry of pairing part for fermions
        if not self._particle_conserving and self._isfermions:
            delta_diff = self._delta_sp + self._delta_sp.T
            if np.linalg.norm(delta_diff) > 1e-10:
                self._log(f"Warning: Pairing matrix is not antisymmetric (norm of Delta + Delta^dagger = {np.linalg.norm(delta_diff)})", lvl=1, log='warning')

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
            ns          = self._ns
            h_matrix    = self._hamil_sp[:ns, :ns]
            v_matrix    = self._delta_sp[:ns, :ns]
        else:
            h_matrix    = self._hamil_sp
            v_matrix    = None
        
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
            ns          = self._ns
            h_matrix    = self._hamil_sp[:ns, :ns]
            v_matrix    = self._delta_sp[:ns, :ns]
        else:
            h_matrix    = self._hamil_sp
            v_matrix    = None
        
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
            self._log(f"Switching backend to {backend_name}", lvl=2, log='info')
            self._backend_instance = backend
        except ValueError as e:
            self._log(f"Failed to switch backend: {e}", lvl=0, log='error')
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

        return (f"QuadraticHamiltonian(ns={self._ns}, {pc_status}, {particles}, "
                f"backend={backend}, {diag_status}, constant={self._constant_offset})")

    # ========================================================================
    #! Spectral Function Methods
    # ========================================================================
    
    def spectral_function(self, 
                        omega       : Union[float, np.ndarray], 
                        operator    : Optional[np.ndarray] = None,
                        eta         : float = 0.01) -> Union[float, np.ndarray]:
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
            from QES.general_python.physics.spectral.spectral_backend import greens_function_quadratic
            from QES.general_python.physics.spectral.spectral_function import spectral_function as sf_func
        except ImportError:
            raise ImportError("Required spectral modules not found in QES.general_python.physics.spectral.")
        
        # Ensure diagonalization
        try:
            self.diagonalize()
        except Exception as e:
            raise RuntimeError(f"Failed to diagonalize Hamiltonian before spectral function calculation: {e}")
        
        # Check if input is scalar
        is_scalar   = np.isscalar(omega)
        omega_arr   = np.atleast_1d(omega)      
        
        # Compute Green's function for each omega value
        # Use greens_function_quadratic without operators for single-particle resolvent
        G_list      = []
        for om in omega_arr:
            G_i = greens_function_quadratic(om, self.eig_val, self.eig_vec, eta=eta)
            G_list.append(G_i)
        
        # Stack into array: shape (n_omega, N, N)
        G           = np.array(G_list)
        
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
            from QES.general_python.physics.spectral.spectral_backend import greens_function_quadratic
        except ImportError:
            raise ImportError("spectral_backend module not found in QES.general_python.physics.spectral.")
        
        # Ensure diagonalization
        self.diagonalize()
        
        # Check if input is scalar
        is_scalar   = np.isscalar(omega)
        omega_arr   = np.atleast_1d(omega)
        
        # Compute Green's function for each omega using greens_function_quadratic
        G_list      = []
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

# ---------------------------------------------------------------------------
#! Registry integration
# ---------------------------------------------------------------------------

# Register the quadratic Hamiltonian in the global registry so that users
# can instantiate it via `HamiltonianConfig(kind="quadratic", ...)`.

def _build_quadratic_hamiltonian(config: HamiltonianConfig, params: Dict[str, Any]) -> Hamiltonian:
    '''
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
    '''
    
    
    hilbert = config.resolve_hilbert()
    
    if hilbert is not None:
        params.setdefault('ns', hilbert.get_Ns())
        params.setdefault('hilbert_space', hilbert)
        params.setdefault('particle_conserving', hilbert.particle_conserving)
        
    ns = params.get('ns', hilbert.get_Ns() if hilbert is not None else None)
    if ns is None:
        raise ValueError("Quadratic Hamiltonian requires 'ns' or a Hilbert space.")
    
    params.setdefault('particles', 'fermions')
    return QuadraticHamiltonian(**params)

register_hamiltonian(
    'quadratic',
    builder     = _build_quadratic_hamiltonian,
    description = 'Quadratic (free) Hamiltonian supporting onsite, hopping, and pairing terms.',
    tags        = ('quadratic', 'noninteracting', 'fermion', 'boson'),
)

# ---------------------------------------------------------------------------
import QES.Algebra.Quadratic.hamil_quadratic_transform as _hqt
# ---------------------------------------------------------------------------

QuadraticHamiltonian.register_basis_transform("real", "k-space", _hqt._handler_real_to_kspace)
QuadraticHamiltonian.register_basis_transform("k-space", "real", _hqt._handler_kspace_to_real)

# ---------------------------------------------------------------------------
#! End of file
# ---------------------------------------------------------------------------