"""
High-level Hilbert space class for quantum many-body systems.

---------------------------------------------------
File    : QES/Algebra/hilbert.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-02-01
Version : 1.2.0
Changes :
    - 2025.02.01 : 1.0.0 - Initial version of the Hilbert space class                       - MK
    - 2025.10.26 : 1.1.0 - Refactored symmetry group generation and added detailed logging  - MK
    - 2025.10.28 : 1.1.1 - Working on symmetry compatibility and modular symmetries         - MK
    - 2025.12.08 : 1.2.0 - Refactored to inherit from BaseHilbertSpace.
---------------------------------------------------
"""

import math
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    # general thingies
    if TYPE_CHECKING:
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.general_python.common.flog import Logger
        from QES.general_python.lattices.lattice import Lattice, LatticeDirection

    #################################################################################################
    from QES.Algebra.globals import GlobalSymmetry
    from QES.Algebra.Hilbert.hilbert_base import BaseHilbertSpace
    from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType, LocalSpace, StateTypes
    from QES.Algebra.hilbert_config import HilbertConfig
    from QES.Algebra.Symmetries.symmetry_container import (
        create_symmetry_container_from_specs,
        normalize_global_symmetries,
        normalize_symmetry_generators,
    )
except ImportError as e:
    raise ImportError(f"Failed to import required modules in hilbert.py: {e}") from e

#####################################################################################################
#! Hilbert space class
#####################################################################################################


class HilbertSpace(BaseHilbertSpace):
    """
    A class to represent a Hilbert space either in Many-Body Quantum Mechanics or Quantum Information Theory and non-interacting systems.
    """

    # --------------------------------------------------------------------------------------------------
    #! Internal checks and inferences
    # --------------------------------------------------------------------------------------------------

    def _check_init_sym_errors(self, sym_gen, global_syms, gen_mapping):
        """Check for initialization symmetry errors"""
        if sym_gen is not None and not isinstance(sym_gen, (dict, list)):
            raise ValueError("The symmetry generators must be provided as a dictionary or list.")
        if not isinstance(global_syms, list) and global_syms is not None:
            raise ValueError("The global symmetries must be provided as a list.")
        if not isinstance(gen_mapping, bool):
            raise ValueError("The flag for generating the mapping must be a boolean.")

    def _check_ns_infer(self, lattice: "Lattice", ns: int, nh: int):
        """Check and infer the system size Ns from provided parameters"""

        # infer local dimension
        _local_dim = self._local_space.local_dim if self._local_space else 2

        # handle the system physical size dimension
        if ns is not None:
            self._ns = int(ns)
            self._lattice = lattice
            if self._lattice is not None and self._lattice.ns != self._ns:
                self._log(
                    f"Warning: The number of sites in lattice ({self._lattice.ns}) is different than provided ({ns}).",
                    lvl=1,
                    color="yellow",
                    log="info",
                )

        elif lattice is not None:
            self._lattice = lattice
            self._ns = int(lattice.ns)

        elif nh is not None and self._is_many_body:
            # nh = (local_dim)^Ns -> infer Ns from nh and local_space.local_dim
            try:
                if _local_dim <= 1:
                    raise ValueError("The local Hilbert space dimension must be >= 2.")

                # compute Ns via logarithm and validate exact power
                Ns_est = int(round(math.log(float(nh), float(_local_dim))))
                if _local_dim**Ns_est != int(nh):
                    raise ValueError("The provided Nh is inconsistent with Ns and Nhl.")

                self._ns = Ns_est
                self._lattice = None
                self._log(
                    f"Inferred Ns={self._ns} from Nh={nh} and local_dim={_local_dim}",
                    log="debug",
                    lvl=2,
                )
            except Exception:
                raise ValueError("The provided Nh is inconsistent with Ns and Nhl.")

        elif nh is not None and self._is_quadratic:
            # Quadratic mode: treat Nh as an effective basis size; commonly Nh==Ns
            self._ns = int(nh)
            self._lattice = None
            self._log(
                f"Assuming Ns={self._ns} from provided Nh={nh} in quadratic mode.",
                log="info",
                lvl=2,
            )

        else:
            raise ValueError("Either 'ns' or 'lattice' must be provided.")

        try:
            self._nhfull = _local_dim ** (self._ns) if self._ns > 0 else 0

        except OverflowError:
            self._nhfull = float("inf")
            self._log(
                f"Warning: Full Hilbert space size exceeds standard limits (Ns={self._ns}).",
                log="warning",
                lvl=0,
            )

    # --------------------------------------------------------------------------------------------------
    #! Initialization
    # --------------------------------------------------------------------------------------------------

    def __init__(
        self,
        # core definition - elements to define the modes
        ns: Union[int, None] = None,
        lattice: Union["Lattice", None] = None,
        nh: Union[int, None] = None,
        *,
        # mode specificaton
        is_manybody: bool = True,
        part_conserv: Optional[bool] = True,
        # local space properties - for many body
        sym_gen: Union[dict, None] = None,  # symmetry generators (e.g., translation, parity)
        global_syms: Union[
            List[GlobalSymmetry], None
        ] = None,  # global symmetries (e.g., U(1) particle number)
        gen_mapping: bool = True,  # whether to generate the mapping immediately (or on-demand)
        gen_basis: bool = True,  # whether to generate the full basis of representatives (or just generators)
        local_space: Optional[Union[LocalSpace, str]] = None,
        # general parameters
        state_type: StateTypes = StateTypes.INTEGER,
        backend: str = "default",
        dtype: np.dtype = np.float64,
        basis: Optional[str] = None,
        # advanced options
        boundary_flux: Optional[Union[float, Dict["LatticeDirection", float]]] = None,
        state_filter: Optional[Callable[[int], bool]] = None,
        logger: Optional["Logger"] = None,
        verbose: bool = False,
        **kwargs,
    ):
        r"""
        Initializes a HilbertSpace object with specified system and local space properties, symmetries, and backend configuration.

        Initialization Process (Step-by-Step)
        --------------------------------------
        1. **Backend & Logging**:
            Sets up the computational backend (NumPy/JAX) and logging infrastructure.
        2. **System Size Inference**:
            Infers the number of sites ($N_s$) and full Hilbert space dimension ($N_h^{full}$)
            from the provided `ns`, `lattice`, or `nh` arguments.
        3. **Local Space Configuration**:
            Defines the local Hilbert space (e.g., Spin-1/2, Fermion) using `LocalSpace`.
        4. **Basis Inference**:
            Determines the natural basis representation (e.g., Real Space, Fock Space).
        5. **Symmetry Initialization** (via `SymmetryContainer`):
            * **Normalization**:
                Parses `sym_gen` to support both object instances and string identifiers (e.g., 'translation', 'parity') with simplified sector definitions.
            * **Container Setup**:
                Initializes the `SymmetryContainer` which manages group construction and compatibility checking.
            * **Mapping Generation**:
                If `gen_mapping` is True or symmetries are present:
                * Constructs the full symmetry group from generators.
                * Generates representative states (orbits) and normalization factors.
                * Builds the `CompactSymmetryData` structure for efficient O(1) JIT-compatible lookups.
        6. **Finalization**: Sets the effective reduced Hilbert space dimension ($N_h$).

        Parameters
        -----------
        ns (Union[int, None], optional):
            Number of sites in the system. If not provided, inferred from `lattice` or `nh`.
        lattice (Union[Lattice, None], optional):
            Lattice object defining the system structure. If provided, `ns` is set from `lattice.ns`.
        nh (Union[int, None], optional):
            Full Hilbert space dimension. Used if neither `ns` nor `lattice` is provided.
        is_manybody (bool, optional):
            Flag indicating if the system is many-body. Default is True.
        part_conserv (Optional[bool], optional):
            Flag indicating if particle number is conserved. Default is True.

        sym_gen (Union[dict, list, None], optional):
            Specification of symmetry generators. Supports dictionaries or lists.
            **String-based definition is fully supported** to avoid imports.

            **Examples**:

            1. **Translation Symmetry** (String-based):
            ```python
            # Define momentum sector k=(0,0) for a 2D lattice
            sym_gen = {'translation': {'kx': 0, 'ky': 0}}

            # Or simply for 1D
            sym_gen = {'translation': 0}
            ```

            2. **Parity & Reflection** (String-based):
            ```python
            sym_gen = {
                'parity': 1,        # Z-Parity = +1
                'reflection': -1    # Spatial Reflection = -1
            }
            ```

            3. **Explicit Object Instantiation**:
            ```python
            from QES.Algebra.Symmetries.translation import TranslationSymmetry
            sym_gen = [TranslationSymmetry(kx=0)]
            ```

            Supported string keys: 'translation', 'parity', 'reflection', 'inversion',
            'time_reversal', 'fermion_parity', 'particle_hole'.

        global_syms (Union[List[GlobalSymmetry], None], optional)
            List of global symmetry objects for additional quantum number constraints (e.g. U(1)).
            These are applied before local symmetry generators. Default is None.

        gen_mapping (bool, optional)
            Whether to generate state mapping based on symmetries immediately.
            If False, mapping is generated on-demand. Default is False.
            Set to True for immediate symmetry reduction and representative state mapping.

        gen_basis (bool, optional)
            If False, skips generating the full reduced basis and mapping.
            Useful for large systems where only symmetry generators are needed. Default is True.

        local_space (Optional[Union[LocalSpace, str]], optional)
            LocalSpace object or string defining local Hilbert space properties.
            Default is None (uses spin-1/2).
            Supported strings: 'spin-1/2', 'spin-1', 'fermion', 'hardcore-boson', etc.
            Legacy compatibility: you may still pass `nhl` via kwargs
            (`nhl=2` -> spin-1/2, `nhl=3` -> spin-1).

        state_type (str, optional):
            Type of state representation (e.g., "integer"). Default is "integer".
        backend (str, optional):
            Backend to use for vectors and matrices. Default is 'default'.
        dtype (optional):
            Data type for Hilbert space arrays. Default is np.float64.
        basis (Optional[str], optional):
            Initial basis representation ("real", "k-space", "fock", "sublattice", "symmetry").
            If not provided, basis is inferred from system properties. Default is None.
        boundary_flux (Optional[float or dict]):
            Optional Peierls phase specification applied to lattice boundary crossings.
        state_filter (Optional[Callable[[int], bool]]):
            Optional predicate applied to integer-encoded basis labels during symmetry reduction.
        logger (Optional[Logger], optional):
            Logger instance for logging. Default is None.
        **kwargs:
            Additional keyword arguments, such as 'threadnum' (number of threads to use).

        Raises:
            ValueError: If provided arguments do not match expected types or required parameters are missing.
        """

        # Legacy compatibility: nhl is mapped to a local-space family when local_space is omitted.
        nhl_legacy = kwargs.pop("nhl", None)

        #! set locals
        # If we have a LocalSpace.default(), use it; otherwise infer from legacy args or use spin-1/2.
        if isinstance(local_space, LocalSpace):
            _local_space = local_space
        elif isinstance(local_space, str):
            _local_space = LocalSpace.from_str(local_space)
        elif nhl_legacy is not None:
            try:
                nhl_legacy = int(nhl_legacy)
            except Exception as exc:
                raise ValueError(f"Invalid legacy `nhl` value: {nhl_legacy!r}") from exc
            if nhl_legacy == 2:
                _local_space = LocalSpace.default_spin_half()
            elif nhl_legacy == 3:
                _local_space = LocalSpace.default_spin_1()
            elif nhl_legacy > 1:
                _local_space = LocalSpace.default_boson(cutoff=nhl_legacy - 1)
            else:
                raise ValueError("Legacy `nhl` must be >= 2.")
        else:
            _local_space = LocalSpace.default()

        # Call Base initialization
        super().__init__(
            ns,
            lattice,
            _local_space,
            backend,
            state_type,
            dtype,
            logger=logger,
            state_filter=state_filter,
            boundary_flux=boundary_flux,
            **kwargs,
        )

        # Symmetry group and info - Initialize EARLY
        self._global_syms = normalize_global_symmetries(
            self.lattice, global_syms if global_syms is not None else []
        )
        self._is_many_body = is_manybody
        self._is_quadratic = not is_manybody
        self._particle_conserving = part_conserv
        self._verbose = verbose

        #! quick check
        self._check_init_sym_errors(sym_gen, self._global_syms, gen_mapping)

        #! infer the system sizes (sets self._ns, self._lattice, self._nhfull)
        self._check_ns_infer(lattice=lattice, ns=ns, nh=nh)

        #! Basis representation inference
        self._infer_and_set_default_basis(explicit_basis=basis)

        #! Nh: Effective dimension of the *current* representation
        if self._is_quadratic:
            self._nh = self._ns
            if self._verbose:
                self._log(
                    f"Initialized HilbertSpace in quadratic mode: Ns={self._ns}, effective Nh={self._nh}.",
                    log="debug",
                    lvl=1,
                    color="green",
                )
        else:
            self._nh = self._nhfull
            if self._verbose:
                self._log(
                    f"Initialized HilbertSpace in many-body mode: Ns={self._ns}, initial Nh={self._nh} (potentially reducible).",
                    color="green",
                    log="debug",
                    lvl=1,
                )

        #! Initialize the symmetries
        self.representative_list = None
        self.representative_norms = None
        self.full_to_representative_idx = None
        self.full_to_representative_phase = None
        self.full_to_global_map = None

        if self._is_many_body:
            # Normalize symmetry generators (supports dict/list/string formats)
            normalized_sym_gen = normalize_symmetry_generators(sym_gen)
            self._init_representatives(
                normalized_sym_gen, gen_mapping=gen_mapping, gen_basis=gen_basis
            )  # gen_mapping True here enables reprmap

            # Set symmetry group from container
            if self._sym_container is not None:
                self._sym_group = list(self._sym_container.symmetry_group)

        elif self._is_quadratic:
            self._log("Quadratic mode: Skipping symmetry mapping generation.", log="debug", lvl=2)
            self.representative_list = None
            self.representative_norms = None

        # Ensure symmetry group always has at least the identity
        if not self._sym_group or len(self._sym_group) == 0:
            try:
                from QES.Algebra.Operator.operator import operator_identity
            except ImportError as e:
                raise RuntimeError(
                    "Failed to import operator_identity. Ensure QES.Algebra.Operator.operator is available."
                ) from e
            self._sym_group = [operator_identity(self._backend_str)]

    # --------------------------------------------------------------------------------------------------
    #! Configuration from HilbertConfig
    # --------------------------------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: HilbertConfig, **overrides):
        """
        Instantiate a HilbertSpace from a :class:`HilbertConfig`.

        Parameters
        ----------
        config:
            Base configuration blueprint.
        **overrides:
            Keyword arguments applied on top of the blueprint before instantiation.
        """
        cfg = config.with_override(**overrides) if overrides else config
        return cls(**cfg.to_kwargs())

    # --------------------------------------------------------------------------------------------------
    #! Symmetry Sector Iteration (Generator)
    # --------------------------------------------------------------------------------------------------

    @staticmethod
    def iter_symmetry_sectors(
        symmetry_types: List[Tuple["SymmetryGenerators", List]],
        *,
        lattice: Optional["Lattice"] = None,
        ns: Optional[int] = None,
        gen_mapping: bool = True,
        verbose: bool = False,
        **hilbert_kwargs,
    ):
        """
        Generator that yields HilbertSpaces for all symmetry sectors.

        This is a memory-efficient way to iterate over all symmetry sectors
        without creating all HilbertSpaces at once. Useful for:
        - Complete spectrum calculations across all k-sectors
        - Finding ground states in each symmetry sector
        - Symmetry-resolved spectroscopy

        Parameters
        ----------
        symmetry_types : List[Tuple[SymmetryGenerators, List]]
            List of (generator_type, allowed_sectors) pairs. Each pair specifies
            a symmetry and the list of sector values to iterate over.

            Examples:
            - [(SymmetryGenerators.Translation_x, range(L))]
            - [(SymmetryGenerators.Translation_x, range(Lx)),
                (SymmetryGenerators.Translation_y, range(Ly))]
            - [(SymmetryGenerators.Translation_x, range(L)),
                (SymmetryGenerators.Reflection, [1, -1])]

        lattice : Lattice, optional
            Lattice structure. Either lattice or ns must be provided.
        ns : int, optional
            Number of sites. Either lattice or ns must be provided.
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping for O(1) lookups.
        verbose : bool, default=False
            If True, log information about each sector.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            Common options: np (particle number), backend, dtype, logger.

        Yields
        ------
        Tuple[dict, HilbertSpace]
            A tuple of (sector_dict, hilbert_space) where:
            - sector_dict: Dictionary mapping SymmetryGenerators to sector values
                e.g., {SymmetryGenerators.Translation_x: 0}
            - hilbert_space: HilbertSpace instance for that sector

        Examples
        --------
        >>> from QES.Algebra.hilbert import HilbertSpace
        >>> from QES.Algebra.Operator.operator import SymmetryGenerators
        >>> from QES.general_python.lattices.square import SquareLattice
        >>> from QES.general_python.lattices.lattice import LatticeBC
        >>>
        >>> lattice = SquareLattice(dim=1, lx=8, bc=LatticeBC.PBC)
        >>>
        >>> # Iterate over all momentum sectors k=0,1,...,7
        >>> for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        ...     [(SymmetryGenerators.Translation_x, range(8))],
        ...     lattice=lattice
        ... ):
        ...     print(f"k={sector[SymmetryGenerators.Translation_x]}: dim={hilbert.dim}")
        k=0: dim=36
        k=1: dim=32
        ...

        >>> # 2D lattice: iterate over (kx, ky) sectors
        >>> lattice_2d = SquareLattice(dim=2, lx=4, ly=4, bc=LatticeBC.PBC)
        >>> for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        ...     [(SymmetryGenerators.Translation_x, range(4)),
        ...      (SymmetryGenerators.Translation_y, range(4))],
        ...     lattice=lattice_2d
        ... ):
        ...     kx, ky = sector[SymmetryGenerators.Translation_x], sector[SymmetryGenerators.Translation_y]
        ...     print(f"(kx={kx}, ky={ky}): dim={hilbert.dim}")

        >>> # Combined translation + reflection
        >>> for sector, hilbert in HilbertSpace.iter_symmetry_sectors(
        ...     [(SymmetryGenerators.Translation_x, range(8)),
        ...      (SymmetryGenerators.Reflection, [1, -1])],
        ...     lattice=lattice
        ... ):
        ...     k, p = sector[SymmetryGenerators.Translation_x], sector[SymmetryGenerators.Reflection]
        ...     print(f"k={k}, parity={p}: dim={hilbert.dim}")

        Notes
        -----
        - Uses Python generators for memory efficiency - only one HilbertSpace
            is in memory at a time (plus any you explicitly keep references to).
        - For very large systems, consider using verbose=True to track progress.
        - The order of iteration follows itertools.product over the sector lists.
        """
        from itertools import product

        # Validate inputs
        if lattice is None and ns is None:
            raise ValueError("Either 'lattice' or 'ns' must be provided.")

        if not symmetry_types or len(symmetry_types) == 0:
            raise ValueError("At least one symmetry type must be provided.")

        # Extract generator types and sector lists
        gen_types = [sym_type for sym_type, _ in symmetry_types]
        sector_lists = [list(sectors) for _, sectors in symmetry_types]

        # Total number of sectors (for progress reporting)
        total_sectors = 1
        for sectors in sector_lists:
            total_sectors *= len(sectors)

        # Iterate over Cartesian product of all sector values
        for sector_idx, sector_values in enumerate(product(*sector_lists)):
            # Build sym_gen list for this sector
            sym_gen = list(zip(gen_types, sector_values))

            # Build sector dictionary for return
            sector_dict = {gen_type: value for gen_type, value in sym_gen}

            if verbose:
                sector_str = ", ".join(f"{gt.name}={v}" for gt, v in sym_gen)
                print(f"[{sector_idx+1}/{total_sectors}] Building sector: {sector_str}")

            # Create HilbertSpace for this sector
            hilbert = HilbertSpace(
                lattice=lattice, ns=ns, sym_gen=sym_gen, gen_mapping=gen_mapping, **hilbert_kwargs
            )

            yield sector_dict, hilbert

    @staticmethod
    def iter_momentum_sectors(
        lattice: "Lattice",
        *,
        include_parity: bool = False,
        gen_mapping: bool = True,
        verbose: bool = False,
        **hilbert_kwargs,
    ):
        """
        Generator that yields HilbertSpaces for all momentum sectors of a lattice.

        This is a convenience method for the common case of iterating over all
        translation symmetry sectors. Automatically detects lattice dimension
        and generates appropriate momentum quantum numbers.

        Parameters
        ----------
        lattice : Lattice
            Lattice structure with periodic boundary conditions.
        include_parity : bool, default=False
            If True, also iterate over reflection parity sectors (+1, -1).
            This doubles the number of sectors.
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping for O(1) lookups.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.
            Common options: np (particle number), backend, dtype, logger.

        Yields
        ------
        Tuple[Tuple[int, ...], HilbertSpace]
            A tuple of (k_vector, hilbert_space) where:
            - k_vector: Tuple of momentum quantum numbers (kx,) or (kx, ky) or (kx, ky, kz)
            - hilbert_space: HilbertSpace instance for that momentum sector

        Examples
        --------
        >>> # 1D chain with L=8 sites
        >>> lattice = SquareLattice(dim=1, lx=8, bc=LatticeBC.PBC)
        >>> for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice):
        ...     print(f"k={k}: dim={hilbert.dim}")
        k=(0,): dim=36
        k=(1,): dim=30
        ...

        >>> # 2D lattice 4x4
        >>> lattice_2d = SquareLattice(dim=2, lx=4, ly=4, bc=LatticeBC.PBC)
        >>> for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice_2d):
        ...     kx, ky = k
        ...     print(f"(kx={kx}, ky={ky}): dim={hilbert.dim}")

        >>> # With parity
        >>> for k, hilbert in HilbertSpace.iter_momentum_sectors(lattice, include_parity=True):
        ...     kx, parity = k[:-1], k[-1]  # Last element is parity
        ...     print(f"k={kx}, P={parity}: dim={hilbert.dim}")

        Notes
        -----
        - Assumes lattice has lx, ly, lz attributes for lattice dimensions.
        - Only uses Translation_x, Translation_y, Translation_z generators.
        - Momentum quantum numbers run from 0 to L-1 for each direction.
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators

        # Detect lattice dimensions
        lx = getattr(lattice, "lx", None) or getattr(lattice, "Lx", None)
        ly = getattr(lattice, "ly", None) or getattr(lattice, "Ly", None)
        lz = getattr(lattice, "lz", None) or getattr(lattice, "Lz", None)

        if lx is None or lx <= 0:
            raise ValueError("Lattice must have positive lx dimension.")

        # Build symmetry types based on lattice dimensionality
        symmetry_types = [(SymmetryGenerators.Translation_x, range(lx))]

        if ly is not None and ly > 1:
            symmetry_types.append((SymmetryGenerators.Translation_y, range(ly)))

        if lz is not None and lz > 1:
            symmetry_types.append((SymmetryGenerators.Translation_z, range(lz)))

        if include_parity:
            symmetry_types.append((SymmetryGenerators.Reflection, [1, -1]))

        # Use the general generator
        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice=lattice,
            gen_mapping=gen_mapping,
            verbose=verbose,
            **hilbert_kwargs,
        ):
            # Convert sector_dict to tuple for easier use
            k_vector = [sector_dict[SymmetryGenerators.Translation_x]]

            if ly is not None and ly > 1:
                k_vector.append(sector_dict[SymmetryGenerators.Translation_y])

            if lz is not None and lz > 1:
                k_vector.append(sector_dict[SymmetryGenerators.Translation_z])

            if include_parity:
                k_vector.append(sector_dict[SymmetryGenerators.Reflection])

            yield tuple(k_vector), hilbert

    @staticmethod
    def iter_reflection_sectors(
        lattice: "Lattice",
        *,
        momentum_sector: Optional[int] = None,
        gen_mapping: bool = True,
        verbose: bool = False,
        **hilbert_kwargs,
    ):
        """
        Generator that yields HilbertSpaces for all reflection parity sectors.

        Parameters
        ----------
        lattice : Lattice
            Lattice structure.
        momentum_sector : int, optional
            If provided, fix the translation sector to this value.
            Reflection is only compatible with k=0 and k=pi (L/2).
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.

        Yields
        ------
        Tuple[int, HilbertSpace]
            A tuple of (parity, hilbert_space) where parity is +1 or -1.

        Examples
        --------
        >>> for parity, hilbert in HilbertSpace.iter_reflection_sectors(lattice):
        ...     print(f"P={parity:+d}: dim={hilbert.dim}")
        P=+1: dim=...
        P=-1: dim=...
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.reflection import ReflectionSymmetry

        symmetry_types = []

        if momentum_sector is not None:
            symmetry_types.append((SymmetryGenerators.Translation_x, [momentum_sector]))

        symmetry_types.append((SymmetryGenerators.Reflection, ReflectionSymmetry.get_sectors()))

        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice=lattice,
            gen_mapping=gen_mapping,
            verbose=verbose,
            **hilbert_kwargs,
        ):
            parity = sector_dict[SymmetryGenerators.Reflection]
            yield parity, hilbert

    @staticmethod
    def iter_inversion_sectors(
        lattice: "Lattice",
        *,
        momentum_sector: Optional[int] = None,
        gen_mapping: bool = True,
        verbose: bool = False,
        **hilbert_kwargs,
    ):
        """
        Generator that yields HilbertSpaces for all spatial inversion parity sectors.

        Unlike reflection (which uses bit-reversal), inversion uses lattice coordinates
        and works correctly for any lattice type (square, honeycomb, triangular, etc.)
        and any dimension (1D, 2D, 3D).

        Parameters
        ----------
        lattice : Lattice
            Lattice structure.
        momentum_sector : int, optional
            If provided, fix the translation sector to this value.
            Inversion is only compatible with k=0 and k=pi (L/2).
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.

        Yields
        ------
        Tuple[int, HilbertSpace]
            A tuple of (parity, hilbert_space) where parity is +1 or -1.

        Examples
        --------
        >>> # Works for any lattice type
        >>> from QES.general_python.lattices.square import SquareLattice
        >>> lattice = SquareLattice(dim=2, lx=4, ly=4)  # 2D
        >>> for parity, hilbert in HilbertSpace.iter_inversion_sectors(lattice):
        ...     print(f"P={parity:+d}: dim={hilbert.dim}")
        P=+1: dim=...
        P=-1: dim=...

        >>> # Also works for honeycomb, triangular, etc.
        >>> from QES.general_python.lattices.honeycomb import HoneycombLattice
        >>> lattice = HoneycombLattice(lx=3, ly=3)
        >>> for parity, hilbert in HilbertSpace.iter_inversion_sectors(lattice):
        ...     print(f"P={parity:+d}: dim={hilbert.dim}")
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.inversion import InversionSymmetry

        symmetry_types = []

        if momentum_sector is not None:
            symmetry_types.append((SymmetryGenerators.Translation_x, [momentum_sector]))

        symmetry_types.append((SymmetryGenerators.Inversion, InversionSymmetry.get_sectors()))

        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice=lattice,
            gen_mapping=gen_mapping,
            verbose=verbose,
            **hilbert_kwargs,
        ):
            parity = sector_dict[SymmetryGenerators.Inversion]
            yield parity, hilbert

    @staticmethod
    def iter_parity_sectors(
        lattice: "Lattice",
        axis: str = "z",
        *,
        gen_mapping: bool = True,
        verbose: bool = False,
        **hilbert_kwargs,
    ):
        """
        Generator that yields HilbertSpaces for all spin-parity sectors.

        Parameters
        ----------
        lattice : Lattice
            Lattice structure.
        axis : str, default='z'
            Parity axis ('x', 'y', or 'z').
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.

        Yields
        ------
        Tuple[int, HilbertSpace]
            A tuple of (parity, hilbert_space) where parity is +1 or -1.

        Examples
        --------
        >>> for parity, hilbert in HilbertSpace.iter_parity_sectors(lattice, axis='z'):
        ...     print(f"Pz={parity:+d}: dim={hilbert.dim}")
        Pz=+1: dim=...
        Pz=-1: dim=...
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.parity import ParitySymmetry

        axis_map = {
            "x": SymmetryGenerators.ParityX,
            "y": SymmetryGenerators.ParityY,
            "z": SymmetryGenerators.ParityZ,
        }

        if axis.lower() not in axis_map:
            raise ValueError(f"Unknown parity axis '{axis}'. Use 'x', 'y', or 'z'.")

        gen_type = axis_map[axis.lower()]
        symmetry_types = [(gen_type, ParitySymmetry.get_sectors(axis))]

        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice=lattice,
            gen_mapping=gen_mapping,
            verbose=verbose,
            **hilbert_kwargs,
        ):
            parity = sector_dict[gen_type]
            yield parity, hilbert

    @staticmethod
    def iter_all_sectors(
        lattice: "Lattice",
        symmetries: "List[str]" = None,
        *,
        gen_mapping: bool = True,
        verbose: bool = False,
        **hilbert_kwargs,
    ):
        """
        Generator that yields HilbertSpaces for all combinations of specified symmetries.

        This is a high-level convenience method that automatically determines
        valid sectors for each symmetry type based on the lattice.

        Parameters
        ----------
        lattice : Lattice
            Lattice structure with periodic boundary conditions.
        symmetries : List[str], optional
            List of symmetry names to include. Options:
            - 'translation' or 'momentum' : Translation symmetry (all k-sectors)
            - 'reflection' or 'parity_spatial' : Spatial reflection (+1, -1)
            - 'parity_z' or 'spin_parity' : Spin-flip parity Z (+1, -1)
            - 'parity_x' : Spin-flip parity X (+1, -1)

            Default: ['translation'] (all momentum sectors only).
        gen_mapping : bool, default=True
            Whether to generate compact symmetry mapping.
        verbose : bool, default=False
            If True, print progress information.
        **hilbert_kwargs
            Additional keyword arguments passed to HilbertSpace constructor.

        Yields
        ------
        Tuple[dict, HilbertSpace]
            A tuple of (sector_dict, hilbert_space) where sector_dict maps
            symmetry names to their sector values.

        Examples
        --------
        >>> # All momentum sectors
        >>> for sectors, h in HilbertSpace.iter_all_sectors(lattice):
        ...     print(f"k={sectors['kx']}: dim={h.dim}")

        >>> # Momentum + reflection
        >>> for sectors, h in HilbertSpace.iter_all_sectors(
        ...     lattice, symmetries=['translation', 'reflection']
        ... ):
        ...     print(f"k={sectors['kx']}, P={sectors['reflection']}: dim={h.dim}")

        >>> # All symmetries for complete spectrum decomposition
        >>> for sectors, h in HilbertSpace.iter_all_sectors(
        ...     lattice,
        ...     symmetries=['translation', 'reflection', 'parity_z'],
        ...     verbose=True
        ... ):
        ...     # Build Hamiltonian and diagonalize
        ...     H = build_hamiltonian(h)
        ...     energies = np.linalg.eigvalsh(H.toarray())
        ...     sectors['energies'] = energies
        """
        from QES.Algebra.Operator.operator import SymmetryGenerators
        from QES.Algebra.Symmetries.parity import ParitySymmetry
        from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
        from QES.Algebra.Symmetries.translation import TranslationSymmetry

        if symmetries is None:
            symmetries = ["translation"]

        symmetry_types = []
        sector_names = {}  # Maps SymmetryGenerators -> friendly name

        lx = getattr(lattice, "lx", None) or getattr(lattice, "Lx", 1)
        ly = getattr(lattice, "ly", None) or getattr(lattice, "Ly", 1)
        lz = getattr(lattice, "lz", None) or getattr(lattice, "Lz", 1)

        for sym in symmetries:
            sym_lower = sym.lower().replace("_", "").replace("-", "")

            if sym_lower in ["translation", "momentum", "trans"]:
                symmetry_types.append(
                    (
                        SymmetryGenerators.Translation_x,
                        TranslationSymmetry.get_sectors(lattice, "x"),
                    )
                )
                sector_names[SymmetryGenerators.Translation_x] = "kx"

                if ly > 1:
                    symmetry_types.append(
                        (
                            SymmetryGenerators.Translation_y,
                            TranslationSymmetry.get_sectors(lattice, "y"),
                        )
                    )
                    sector_names[SymmetryGenerators.Translation_y] = "ky"

                if lz > 1:
                    symmetry_types.append(
                        (
                            SymmetryGenerators.Translation_z,
                            TranslationSymmetry.get_sectors(lattice, "z"),
                        )
                    )
                    sector_names[SymmetryGenerators.Translation_z] = "kz"

            elif sym_lower in ["reflection", "parityspatial", "spatial", "mirror"]:
                symmetry_types.append(
                    (SymmetryGenerators.Reflection, ReflectionSymmetry.get_sectors())
                )
                sector_names[SymmetryGenerators.Reflection] = "reflection"

            elif sym_lower in ["inversion", "inv", "spatialinversion"]:
                from QES.Algebra.Symmetries.inversion import InversionSymmetry

                symmetry_types.append(
                    (SymmetryGenerators.Inversion, InversionSymmetry.get_sectors())
                )
                sector_names[SymmetryGenerators.Inversion] = "inversion"

            elif sym_lower in ["parityz", "spinparity", "pz", "spinflip"]:
                symmetry_types.append((SymmetryGenerators.ParityZ, ParitySymmetry.get_sectors("z")))
                sector_names[SymmetryGenerators.ParityZ] = "parity_z"

            elif sym_lower in ["parityx", "px"]:
                symmetry_types.append((SymmetryGenerators.ParityX, ParitySymmetry.get_sectors("x")))
                sector_names[SymmetryGenerators.ParityX] = "parity_x"

            elif sym_lower in ["parityy", "py"]:
                symmetry_types.append((SymmetryGenerators.ParityY, ParitySymmetry.get_sectors("y")))
                sector_names[SymmetryGenerators.ParityY] = "parity_y"

            else:
                raise ValueError(
                    f"Unknown symmetry '{sym}'. Options: translation, reflection, inversion, parity_z, parity_x, parity_y"
                )

        for sector_dict, hilbert in HilbertSpace.iter_symmetry_sectors(
            symmetry_types,
            lattice=lattice,
            gen_mapping=gen_mapping,
            verbose=verbose,
            **hilbert_kwargs,
        ):
            # Convert to friendly names
            named_sectors = {sector_names[gen]: val for gen, val in sector_dict.items()}
            yield named_sectors, hilbert

    # --------------------------------------------------------------------------------------------------
    #! Resets
    # --------------------------------------------------------------------------------------------------

    def reset_local_symmetries(self):
        """
        Reset the local symmetries of the Hilbert space.
        """
        from QES.Algebra.Operator.operator import operator_identity

        self._log("Reseting the local symmetries. Can be now recreated.", lvl=2, log="debug")
        self._sym_group = [operator_identity(self._backend_str)]

    ####################################################################################################
    #! Unified symmetry container initialization
    ####################################################################################################

    def _init_sym_container(self, gen: list) -> None:
        """
        Initialize the unified SymmetryContainer.
        Creates and configures the SymmetryContainer based on provided symmetry generators
        and global symmetries.

        The SymmetryContainer handles:
        - Compatibility checking
        - Automatic group construction from generators
        - Representative finding
        - Normalization computation

        Parameters:
        -----------
        gen : list
            List of (SymmetryGenerator, sector_value) tuples
        """

        if (not gen or len(gen) == 0) and not self.check_global_symmetry:
            self._log(
                "No symmetries provided; SymmetryContainer will use identity only.",
                lvl=1,
                log="debug",
                color="green",
            )

            try:
                from QES.Algebra.Symmetries.symmetry_container import SymmetryContainer
            except ImportError as e:
                raise RuntimeError(
                    "Failed to import SymmetryContainer. Ensure QES.Algebra.Symmetries.symmetry_container is available."
                ) from e

            self._sym_basic_info = HilbertSpace.SymBasicInfo()  # basic symmetry info container
            self._sym_container = SymmetryContainer(
                ns=self._ns,
                lattice=self._lattice,
                nhl=self._local_space.local_dim,
                backend=self._backend_str,
                verbose=self._verbose,
            )
            return

        # Prepare generator specs
        generator_specs = gen.copy() if gen is not None else []

        # Create and initialize the container
        # The factory will handle all compatibility checking and filtering
        #! Note: Compact symmetry data is built later in _init_representatives after representatives are generated
        self._sym_container = create_symmetry_container_from_specs(
            ns=self._ns,
            generator_specs=generator_specs,
            global_syms=self._global_syms,
            lattice=self._lattice,
            nhl=self._local_space.local_dim,
            backend=self._backend_str,
            build_group=True,
            verbose=self._verbose,
        )

        # Log symmetry info
        self._sym_basic_info.num_operators = len(self._sym_container.symmetry_group)
        self._sym_basic_info.num_gens = len(self._sym_container.generators)
        if self._verbose:
            self._log(
                f"SymmetryContainer initialized: {self._sym_basic_info.num_gens} generators -> {self._sym_basic_info.num_operators} group elements",
                lvl=2,
                color="green",
            )

        # Compute and cache whether any symmetry eigenvalues/phases are complex.
        try:
            # Prefer the canonical group stored on the SymmetryContainer when present
            has_cpx = False
            group = getattr(self._sym_container, "symmetry_group", None) or self._sym_group
            container = self._sym_container

            if group:
                # Small loop over group elements to check for complex characters
                for elem in group:
                    # elem is a GroupElement (tuple of operators)
                    # We can use the container to get the character for this element
                    if container is not None:
                        char = container.get_character(elem)
                        if isinstance(char, complex) and abs(char.imag) > 1e-14:
                            has_cpx = True
                            break
                        if not np.isreal(char):
                            has_cpx = True
                            break
                    else:
                        # Fallback for legacy behavior if container is somehow missing but group exists
                        # (unlikely given logic above)
                        pass

            self._has_complex_symmetries = bool(has_cpx)
        except Exception:
            # On unexpected errors, be conservative and assume complex
            self._has_complex_symmetries = True

    # --------------------------------------------------------------------------------------------------

    def _init_representatives(self, gen: list, gen_mapping: bool = False, gen_basis: bool = True):
        """
        Initialize the representatives list and norms based on the provided symmetry generators.

        The representative process:
        1. For each state in the full Hilbert space, check if it satisfies global symmetries (e.g., U(1))
        2. Find the representative state (minimum state in symmetry orbit) using local symmetries
        3. If the state is its own representative, calculate normalization and add to mapping
        4. Optionally build full representative map (reprmap) for fast lookup

        Parameters:
        -----------
            gen (list):
                A list of (SymmetryGenerator, sector_value) tuples.
            gen_mapping (bool):
                If True, generate the full representative map for all states.
                This uses more memory but speeds up repeated representative lookups.
        """

        if not self._is_many_body:
            if self._verbose:
                self._log("Skipping mapping initialization in quadratic mode.", log="debug", lvl=1)
            return
        else:
            if self._verbose:
                self._log("Building representatives.", log="info", lvl=1, color="green")

        # Trivial case: no symmetries
        if not gen and not self._global_syms and self._state_filter is None:
            # No symmetries -> trivial mapping: every full-state is its own
            # representative. Set representative_list and representative_norms to None.
            if self._verbose:
                self._log(
                    "No symmetries provided, generating trivial mapping (identity).",
                    log="debug",
                    lvl=2,
                )
            try:
                nh_full = int(self._nhfull)
            except Exception:
                nh_full = self.local_space.local_dim ** int(self._ns)
            self.representative_list = None
            self.representative_norms = None
            self._nh = nh_full
            self._modifies = False
            return

        if gen_mapping and self._verbose:
            self._log(
                "Explicitly requested immediate mapping generation.",
                log="info",
                lvl=3,
                color="blue",
            )

        # Use SymmetryContainer for symmetry group construction
        self._init_sym_container(gen)

        # If basis generation is disabled (e.g., NQS mode), skip representative generation
        if not gen_basis:
            if self._verbose:
                self._log("Skipping basis generation...", lvl=1, color="green", log="debug")
            self._nh = self._nhfull
            self.representative_list = None
            self.representative_norms = None
            return

        # Generate basis using the new memory-efficient method
        try:
            # Check for integer overflow on huge systems
            if self._nhfull > np.iinfo(np.int64).max:
                self._log("System too large for full basis enumeration.", lvl=0, color="red")
                raise OverflowError("Hilbert space dimension exceeds int64 limits.")

            # This single call handles:
            # 1. Finding representatives (Two-pass to save memory)
            # 2. Filtering (global symmetries)
            # 3. Sorting and normalizationv
            # 4. Building compact symmetry map (uint32/uint8)
            self.representative_list, self.representative_norms = (
                self._sym_container.generate_symmetric_basis(
                    nh_full=int(self._nhfull),
                    state_filter=self._state_filter,  #!TODO: use filter if necesssary
                    return_map=gen_mapping,
                )
            )

            #! Set the new Hilbert space size
            self._nh = len(self.representative_list)
            self._sym_container.set_repr_info(self.representative_list, self.representative_norms)

        except Exception as e:
            self._log(f"Failed to generate symmetric basis: {e}", lvl=0, color="red")
            raise e

    ####################################################################################################
    #! BASIS REPRESENTATION MANAGEMENT
    ####################################################################################################

    def _infer_and_set_default_basis(self, explicit_basis: Optional[str] = None):
        """
        Infer and set the default basis for this Hilbert space based on system properties.

        If explicit_basis is provided, use that instead of inferring.

        Logic (for inference):
        - Quadratic systems with lattice        : REAL (position/momentum basis)
        - Quadratic systems without lattice     : FOCK (single-particle occupation basis)
        - Many-body systems with lattice        : REAL (position space on lattice sites)
        - Many-body systems without lattice     : COMPUTATIONAL (integer/Fock basis)

        Parameters
        ----------
        explicit_basis : Optional[str]
            If provided, use this basis type instead of inferring.
            Valid values: "real", "k-space", "fock", "sublattice", "symmetry"

        This is called during initialization to establish the natural basis.
        """
        try:
            from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType
        except ImportError as e:
            raise RuntimeError("Failed to import HilbertBasisType.") from e

        if explicit_basis is not None:
            # Use provided basis
            if isinstance(explicit_basis, str):
                self._basis_type = HilbertBasisType.from_string(explicit_basis)
            else:
                self._basis_type = explicit_basis
            self._log(
                f"HilbertSpace basis explicitly set to: {self._basis_type}", lvl=2, color="cyan"
            )
            return

        # Infer basis from system properties
        if self._is_quadratic:
            # Quadratic system: choose based on lattice availability
            if self._lattice is not None:
                default_basis = HilbertBasisType.REAL
                basis_reason = "quadratic-lattice"
            else:
                default_basis = HilbertBasisType.FOCK
                basis_reason = "quadratic-fock"
        else:
            # Many-body system: choose based on lattice availability
            if self._lattice is not None:
                default_basis = HilbertBasisType.REAL
                basis_reason = "many-body-lattice"
            else:
                default_basis = HilbertBasisType.FOCK
                basis_reason = "many-body-computational"

        # Set the basis type
        self._basis_type = default_basis
        self._log(
            f"HilbertSpace default basis inferred: {default_basis} ({basis_reason})",
            lvl=2,
            color="cyan",
            log="debug",
        )

    # --------------------------------------------------------------------------------------------------
    #! FLUX
    # --------------------------------------------------------------------------------------------------

    @property
    def boundary_flux(self):
        """Return the boundary flux specification applied to the lattice."""
        return self._boundary_flux

    @boundary_flux.setter
    def boundary_flux(self, value: Optional[Union[float, Dict["LatticeDirection", float]]]):
        self._boundary_flux = value
        if self._lattice is not None:
            self._lattice.flux = value

    @property
    def state_filter(self) -> Optional[Callable[[int], bool]]:
        """Predicate applied to integer states during mapping generation."""
        return self._state_filter

    @state_filter.setter
    def state_filter(self, value: Optional[Callable[[int], bool]]):
        self._state_filter = value

    @property
    def modifies(self):
        """
        Return the flag for modifying the Hilbert space.

        Returns:
            bool: The flag for modifying the Hilbert space.
        """
        if self._is_quadratic:
            return False

        return self.representative_list is not None and self._nh != self._nhfull

    @property
    def lattice(self) -> Optional["Lattice"]:
        return self._lattice

    def get_basis(self):
        """
        Get the current basis representation type.

        Returns
        -------
        HilbertBasisType
            Current basis type (default: REAL)
        """
        return getattr(self, "_basis_type", self._get_default_basis())

    def set_basis(self, basis_type: str):
        """
        Set the basis representation type.

        Parameters
        ----------
        basis_type : str or HilbertBasisType
            Target basis ("real", "k-space", "fock", "sublattice", "symmetry")
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        if isinstance(basis_type, str):
            self._basis_type = HilbertBasisType.from_string(basis_type)
        else:
            self._basis_type = basis_type

        self._log(
            f"Hilbert space basis type set to: {self._basis_type}", log="debug", lvl=2, color="blue"
        )

    def _get_default_basis(self):
        """Get default basis type."""
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        return HilbertBasisType.REAL

    @property
    def sites(self):
        return self._ns

    @property
    def n_sites(self):
        return self._ns

    @property
    def Ns(self):
        return self._ns

    @property
    def ns(self):
        return self._ns

    # --------------------------------------------------------------------------------------------------

    @property
    def mapping(self):
        return self.representative_list

    @property
    def repr_list(self):
        return self.representative_list

    @property
    def repr_norms(self):
        return self.representative_norms

    @property
    def compact_symmetry_data(self):
        return self._sym_container.compact_data if self._sym_container else None

    @property
    def dtype(self):
        return self._dtype

    @property
    def sym_group(self):
        """
        Return the symmetry group.

        Returns:
            list: The symmetry group.
        """
        # Expose a callable view for external consumers (tests, utilities)
        # while keeping the internal tuple representation for JIT routines.
        if self._sym_group and isinstance(self._sym_group[0], tuple):

            def wrap_ops_tuple(ops_tuple):
                def op(state):
                    st = state
                    phase = 1.0
                    for g in ops_tuple:
                        # Each g is an Operator; call and accumulate phase
                        st, val = g(st)
                        try:
                            phase = phase * val
                        except Exception:
                            # If phase types differ (e.g., real vs complex), coerce via multiplication
                            phase = phase * (val.real if hasattr(val, "real") else val)
                    return st, phase

                return op

            return [wrap_ops_tuple(t) for t in self._sym_group]
        return self._sym_group

    # --------------------------------------------------------------------------------------------------

    @property
    def local(self):
        return self._local_space.local_dim if self._local_space else 2

    @property
    def local_space(self):
        return self._local_space

    # --------------------------------------------------------------------------------------------------
    #! Local operator builders
    # --------------------------------------------------------------------------------------------------

    def list_local_operators(self):
        """
        Return the identifiers of all onsite operators available in the local space.
        """
        if self._local_space is None:
            return tuple()
        return self._local_space.list_operator_keys()

    def get_operator_elem(self, col_idx: int):
        """
        Get the element of the local operator.
        """
        new_row, sym_eig = self.find_repr(col_idx)
        return new_row, sym_eig

    # --------------------------------------------------------------------------------------------------

    @property
    def dimension(self):
        return self._nh

    @property
    def dim(self):
        return self._nh

    @property
    def fulldim(self):
        return self._nhfull

    @property
    def has_sym(self):
        """
        Returns True if symmetries are defined for this Hilbert space.

        This property checks for the presence of symmetry generators, NOT whether
        the basis has been reduced. Use `has_sym_reduction` to check if the basis
        has actually been reduced.

        Note
        ----
        For NQS applications with large systems, symmetries may be defined
        (`gen_basis=False`) without generating the full representative mapping.
        In this case, `has_sym` returns True while `has_sym_reduction` returns False.
        """
        # Check if symmetry container exists with actual generators
        if self._sym_container is not None:
            if len(self._sym_container.generators) > 0:
                return True
            if len(self._sym_container.symmetry_group) > 1:
                return True

        # Fallback: check if basis was reduced
        return self._nh != self._nhfull

    @property
    def has_sym_generators(self):
        """
        Returns True if symmetry generators are defined (even without basis reduction).

        Use this property when you need to know if symmetry operations are available
        for applying to states (e.g., for NQS symmetry projections).
        """
        return self._sym_container is not None and len(self._sym_container.generators) > 0

    @property
    def has_sym_reduction(self):
        """
        Returns True if the Hilbert space basis has been reduced by symmetries.

        This is the stricter check - it returns True only when representative
        states have been computed and the effective dimension is smaller than
        the full Hilbert space dimension.
        """
        return self._nh != self._nhfull

    @property
    def Nh(self):
        return self._nh

    @property
    def nh(self):
        return self._nh

    @property
    def hilbert_dim(self):
        return self._nh

    @property
    def full(self):
        return self._nhfull

    @property
    def Nhfull(self):
        return self._nhfull

    @property
    def nhfull(self):
        return self._nhfull

    # --------------------------------------------------------------------------------------------------

    @property
    def quadratic(self):
        return self._is_quadratic

    @property
    def is_quadratic(self):
        return self._is_quadratic

    @property
    def many_body(self):
        return self._is_many_body

    @property
    def is_many_body(self):
        return self._is_many_body

    @property
    def particle_conserving(self):
        return self._particle_conserving

    # --------------------------------------------------------------------------------------------------

    @property
    def logger(self):
        return self._logger

    @property
    def operators(self):
        """
        Lazy-loaded operator module for convenient operator access.

        Returns
        -------
        OperatorModule
            Module providing operator factory functions based on the local space type.

        Examples
        --------
        >>> # For spin systems
        >>> hilbert         = HilbertSpace(ns=4, local_space='spin-1/2')
        >>> sig_x           = hilbert.operators.sig_x(sites=[0, 1], ns=ns)
        >>> sig_x_matrix    = sig_x.matrix

        >>> # For fermion systems
        >>> hilbert         = HilbertSpace(ns=4, local_space='fermion')
        >>> cdag            = hilbert.operators.cdag(ns=ns, sites=[0])
        >>> cdag_matrix     = cdag.matrix

        >>> # Get help on available operators
        >>> hilbert.operators.help()
        """
        if not hasattr(self, "_operator_module") or self._operator_module is None:
            from QES.Algebra.Operator.operator_loader import get_operator_module

            local_space_type = self._local_space.typ if self._local_space else None
            self._operator_module = get_operator_module(local_space_type)
        return self._operator_module

    ####################################################################################################
    #! Representation of the Hilbert space
    ####################################################################################################

    def __str__(self):
        """Short string summary of the Hilbert space."""
        mode = "Many-Body" if self._is_many_body else "Quadratic"
        info = f"{mode} Hilbert space: Ns={self._ns}, Nh={self._nh}"
        if self._is_many_body:
            info += f", Local={self._local_space}"
            try:
                conv = self.state_convention.get("name", "unknown")
                info += f", StateConvention={conv}"
            except Exception:
                pass
        if self._global_syms:
            gs = ", ".join(f"{g.get_name_str()}={g.get_val()}" for g in self._global_syms)
            info += f", GlobalSyms=[{gs}]"
        if self._sym_container and getattr(self._sym_container, "generators", None):
            ls = ", ".join(
                f"{gen_type}={sector}" for (_, (gen_type, sector)) in self._sym_container.generators
            )
            info += f", LocalSyms=[{ls}]"
        return info

    def __repr__(self):
        sym_info = self.get_sym_info()
        base    = "SP" if self._is_quadratic else "MB"
        txt     = f"{base} Hilbert space with {self._nh} states and {self._ns} sites; {self._local_space}"
        if sym_info:
            txt += f". Symmetries: {sym_info}"
        return txt

    ####################################################################################################
    #! Find the representative of a state
    ####################################################################################################
    # These methods are now in BaseHilbertSpace, but aliased here for compatibility if needed.
    # BaseHilbertSpace has: find_sym_repr, find_sym_norm, find_repr, find_norm, norm
    # So we don't strictly need to redefine them unless we want to override behavior.
    # Base implementations use _sym_container, which is what we want.

    ####################################################################################################
    #! Full Hilbert space generation
    ####################################################################################

    def get_full_glob_map(self):
        if self.full_to_global_map is None or len(self.full_to_global_map) == 0:
            self.generate_full_glob_map()
        return self.full_to_global_map

    def generate_full_glob_map(self):
        if self.full_to_global_map is not None and len(self.full_to_global_map) > 0:
            return

        full_map = []

        if self._global_syms:
            for j in range(self._nhfull):
                global_checker = True
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                if global_checker:
                    full_map.append(j)
        self.full_to_global_map = np.array(full_map, dtype=np.int64)

    def expand_state(
        self, vec_reduced: np.ndarray, vec_full: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Expand a vector from the reduced symmetry sector back to the full Hilbert space.

        This method handles both global symmetries (via full_map) and local symmetries
        (via symmetry group expansion). If no symmetries are present, returns the input unchanged.

        Parameters:
        -----------
        vec_reduced (np.ndarray):
            The vector in the reduced Hilbert space to expand.
        vec_full (Optional[np.ndarray]):
            Optional preallocated full vector to fill. If None, a new array is created.

        Returns:
        --------
        np.ndarray:
            The expanded vector in the full Hilbert space.
        """
        # Check if we have any symmetries that reduce the space
        if not self.modifies:
            return np.asarray(vec_reduced)

        if vec_full is None:
            if vec_reduced.ndim == 2:
                vec_full = np.zeros(
                    (int(self._nhfull), vec_reduced.shape[1]), dtype=vec_reduced.dtype
                )
            else:
                vec_full = np.zeros(int(self._nhfull), dtype=vec_reduced.dtype)

        # If we have global symmetries with a full map, use that
        if self.check_global_symmetry and self.full_to_global_map is not None:
            nh_full = int(self._nhfull)
            if vec_reduced.ndim == 2:
                vec_full = np.zeros((nh_full, vec_reduced.shape[1]), dtype=vec_reduced.dtype)
                for i, state in enumerate(self.full_to_global_map):
                    if i < len(vec_reduced):
                        vec_full[state, :] = vec_reduced[i, :]
            else:
                vec_full = np.zeros(nh_full, dtype=vec_reduced.dtype)
                for i, state in enumerate(self.full_to_global_map):
                    if i < len(vec_reduced):
                        vec_full[state] = vec_reduced[i]
            return vec_full

        # For local symmetries, expand using the JIT-accelerated symmetry container method
        if self._sym_container is not None and self.representative_list is not None:
            return self._sym_container.expand_state(vec_reduced, vec_full)

        # Fallback: return unchanged
        return np.asarray(vec_reduced)

    ####################################################################################################
    #! Operators for the Hilbert space
    ####################################################################################################

    def __len__(self):
        return self._nh

    def __call__(self, i):
        return self.find_sym_repr(i)

    def __getitem__(self, i):
        """
        Return the i-th basis state of the Hilbert space.

        Args:
            i: The index of the basis state to return or a state to find the representative for.

        Returns:
            np.ndarray: The i-th basis state of the Hilbert space.
        """
        if isinstance(i, (int, np.integer)):
            return (
                self.representative_list[i]
                if (self.representative_list is not None and len(self.representative_list) > 0)
                else i
            )
        return self.find_sym_repr(i)

    def __contains__(self, state):
        """
        Check if a state is in the Hilbert space.

        Args:
            state: The state to check.

        Returns:
            bool: True if the state is in the Hilbert space, False otherwise.
        """
        if isinstance(state, int):
            return (
                state in self.representative_list if self.representative_list is not None else True
            )

        rep, _ = self.find_sym_repr(state)
        return rep in self.representative_list if self.representative_list is not None else True

    def __iter__(self):
        """
        Iterate over the basis states of the Hilbert space.

        Yields:
            int: The next basis state in the Hilbert space.
        """
        if self.representative_list is not None:
            for state in self.representative_list:
                yield state
        else:
            for state in range(self._nh):
                yield state

    def __array__(self, dtype=None):
        """
        Return the basis states of the Hilbert space as a NumPy array.

        Args:
            dtype: The desired data type of the array.


        Returns:
            np.ndarray: The basis states of the Hilbert space as a NumPy array.
        """
        if self.representative_list is not None:
            return np.array(self.representative_list, dtype=dtype)
        return np.arange(self._nh, dtype=dtype)


#####################################################################################################
#! End of file
#####################################################################################################
