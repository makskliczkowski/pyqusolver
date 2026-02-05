"""
file : Algebra/hamil.py

High-level Hamiltonian class for the Quantum Energy Solver (QES) package. This class is used to
define the Hamiltonian of a system. It may be either a Many-Body Quantum Mechanics Hamiltonian or a
non-interacting system Hamiltonian. It may generate a Hamiltonian matrix but in addition it defines
how an operator acts on a state. The Hamiltonian class is an abstract class and is not meant to be
instantiated. It is meant to be inherited by other classes.

Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-02-01
Version : 1.0.0
Changes :
    2025-02-01 (1.0.0) : First implementation of the Hamiltonian class. - MK
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from QES.general_python.common.flog                     import Logger
    from QES.general_python.lattices.lattice                import Lattice

###################################################################################################

try:
    from    QES.Algebra.hilbert                             import HilbertSpace, HilbertConfig
    from    QES.Algebra.Operator.operator                   import OperatorTypeActing, OperatorFunction
    from    QES.Algebra.Operator.basis_aware_operator       import BasisAwareOperator

    from    QES.Algebra.Hilbert.matrix_builder              import build_operator_matrix
    from    QES.Algebra.Hamil.hamil_types                   import *
    from    QES.Algebra.Hamil.hamil_energy                  import local_energy_np_wrap
    from    QES.Algebra.hamil_config                        import HamiltonianConfig, HAMILTONIAN_REGISTRY
    from    QES.Algebra.Hamil.hamil_diag_engine             import DiagonalizationEngine
    import  QES.Algebra.Hamil.hamil_jit_methods             as hjm
    from    QES.Algebra.hamil_cache                         import get_matrix_from_cache, store_matrix_in_cache, generate_cache_key

except ImportError as exc:
    raise ImportError("QES.Algebra.hilbert or QES.Algebra.Operator.operator could not be imported. Ensure QES is properly installed.") from exc

###################################################################################################

###################################################################################################

try:
    import jax.numpy                    as jnp
    from jax.experimental.sparse        import BCOO
    from QES.Algebra.Hamil.hamil_energy import local_energy_jax_wrap

    JAX_AVAILABLE                       = True
except ImportError:
    jax                                 = None
    jnp                                 = None
    lax                                 = None
    BCOO                                = None
    local_energy_jax_wrap               = None
    JAX_AVAILABLE                       = False

####################################################################################################
#! Hamiltonian class - abstract class
####################################################################################################


class Hamiltonian(BasisAwareOperator):
    r"""
    A general Hamiltonian class. This class is used to define the Hamiltonian of a system. It may be
    either a Many-Body Quantum Mechanics Hamiltonian or a non-interacting system Hamiltonian. It may
    generate a Hamiltonian matrix but in addition it defines how an operator acts on a state. The
    Hamiltonian class is an abstract class and is not meant to be instantiated. It is meant to be
    inherited by other classes.

    Inherits from BasisAwareOperator to gain:
    - Basis representation tracking and transformation dispatch
    - Symmetry recording and HilbertSpace synchronization
    - Custom operator registration and instruction code management (via SpecialOperator)
    - Hybrid composition functions for numba-compatible matrix building
    - Support for arbitrary n-site correlators (e.g., n-point correlation functions)
    """

    # Error messages for Hamiltonian class
    _ERR_EIGENVALUES_NOT_AVAILABLE  = (
        "Eigenvalues are not available. Please diagonalize the Hamiltonian first."
    )
    _ERR_HAMILTONIAN_NOT_AVAILABLE  = (
        "Hamiltonian matrix is not available. Please build or initialize the Hamiltonian."
    )
    _ERR_HAMILTONIAN_INITIALIZATION = (
        "Failed to initialize the Hamiltonian matrix. Check Hilbert space, lattice, and parameters."
    )
    _ERR_HAMILTONIAN_BUILD          = (
        "Failed to build the Hamiltonian matrix. Ensure all operators and spaces are properly set."
    )

    # Dictionary of error messages
    _ERRORS = {
        "eigenvalues_not_available"     : _ERR_EIGENVALUES_NOT_AVAILABLE,
        "hamiltonian_not_available"     : _ERR_HAMILTONIAN_NOT_AVAILABLE,
        "hamiltonian_initialization"    : _ERR_HAMILTONIAN_INITIALIZATION,
        "hamiltonian_build"             : _ERR_HAMILTONIAN_BUILD,
    }

    _ADD_TOLERANCE = 1e-10

    def _ADD_CONDITION(x, *args):
        import numbers
        
        if x is None:
            return False
        if args:
            if len(args) == 1:
                y = x[args[0]]
            else:
                y = x[args]
        else:
            y = x
        if y is None:
            return False
        
        # Ensure y is a numeric type or array
        if isinstance(y, (numbers.Number, float, int, complex, np.generic)):
            return not np.isclose(y, 0.0, rtol=Hamiltonian._ADD_TOLERANCE)
        elif isinstance(y, (np.ndarray, list, tuple)):
            y_arr = np.asarray(y)
            return not np.all(np.isclose(y_arr, 0.0, rtol=Hamiltonian._ADD_TOLERANCE))
        else:
            return False

    # ----------------------------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: HamiltonianConfig, **overrides):
        """Construct a Hamiltonian instance from a registered configuration."""
        return HAMILTONIAN_REGISTRY.instantiate(config, **overrides)

    # ----------------------------------------------------------------------------------------------

    def __init__(
        self,
        # concerns the definition of the system type
        is_manybody: bool = True,  # True for many-body Hamiltonian, False for non-interacting
        *,
        hilbert_space: Optional[
            Union[HilbertSpace, HilbertConfig]
        ] = None,  # Required if is_manybody=True
        ns: Optional[
            int
        ] = None,  # Number of sites/modes (if not provided, will be inferred from hilbert_space or lattice)
        lattice: Optional[
            Union[str, List[int], 'Lattice']
        ] = None,  # Alternative way to specify ns and get the Hilbert space
        # concerns the matrix and computation
        is_sparse: bool = True,  # True for sparse matrix, False for dense matrix
        dtype: Optional[
            Union[str, np.dtype]
        ] = None,  # Data type for the Hamiltonian matrix elements (if None, inferred from hilbert_space or backend)
        backend: str = "default",
        # logger and other kwargs
        use_forward: bool = False,
        logger: Optional["Logger"] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the Hamiltonian class.

        Parameters
        ----------
        is_manybody : bool, optional
            If True, the Hamiltonian is treated as a many-body Hamiltonian.
            If False, it is treated as a non-interacting (single-particle) Hamiltonian. Default is True.
        hilbert_space : HilbertSpace, HilbertConfig, or None, optional
            The Hilbert space object describing the system or a blueprint to build it. Required if is_manybody=True.
            Can be configured with symmetries defined via strings (e.g., `sym_gen={'translation': 0}`) for convenience.
        lattice : str or list of int or None, optional
            Lattice information or list of site indices. Used to infer the number of sites (ns) and optionally construct the Hilbert space.
        is_sparse : bool, optional
            If True, the Hamiltonian matrix is stored in a sparse format. Default is True.
        dtype : data-type, optional
            Data type for the Hamiltonian matrix elements. If None, inferred from Hilbert space or backend.
        backend : str, optional
            Computational backend to use ('default', 'np', 'jax', etc.). Default is 'default'.
        logger : Logger, optional
            Logger class, may be inherited from the Hilbert space
        **kwargs
            Additional keyword arguments, such as 'ns' (number of sites/modes), or 'lattice' for further customization.

        Raises
        ------
        ValueError
            If required information (such as Hilbert space or lattice) is missing or inconsistent.
        """
        if isinstance(hilbert_space, HilbertConfig):
            hilbert_space           = HilbertSpace.from_config(hilbert_space)
        self._use_forward           = use_forward

        name = kwargs.pop('name', 'Hamiltonian')

        # Initialize BasisAwareOperator base class (which inherits from SpecialOperator -> Operator -> GeneralMatrix)
        # This sets up backend, sparse, dtype, matrix storage, diagonalization infrastructure,
        # the instruction code system, custom operator registry, AND basis tracking/transformation dispatch
        BasisAwareOperator.__init__(self,
                        ns              =   ns,
                        lattice         =   lattice,
                        name            =   name,
                        backend         =   backend,
                        is_sparse       =   is_sparse,
                        is_manybody     =   is_manybody,
                        hilbert_space   =   hilbert_space,
                        dtype           =   dtype,
                        logger          =   logger,
                        seed            =   seed,
                        verbose         =   verbose,
                        **kwargs)

        # =====================================================================
        #! HAMILTONIAN-SPECIFIC TRANSFORMED REPRESENTATION
        # =====================================================================
        # Note: BasisAwareOperator provides _operator_transformed for general use
        # This alias is Hamiltonian-specific for backward compatibility
        self._hamil_transformed = (
            None  # General storage for transformed Hamiltonian (e.g., H_k, H_fock, etc.)
        )

        # Sync symmetry info from HilbertSpace if available
        self._sync_symmetry_from_hilbert()

        # for the matrix representation of the Hamiltonian
        self._hamil: np.ndarray = (
            None  # will store the Hamiltonian matrix with Nh x Nh full Hilbert space
        )

        #! single particle Hamiltonian info
        self._hamil_sp: np.ndarray = (
            None  # will store Ns x Ns (2Ns x 2Ns for BdG) matrix for quadratic Hamiltonian
        )
        self._delta_sp: np.ndarray = None
        self._constant_offset = 0.0
        self._isfermions = True
        self._isbosons = False

        self._loc_energy_int_fun: Optional[Callable] = None
        self._loc_energy_np_fun: Optional[Callable] = None
        self._loc_energy_jax_fun: Optional[Callable] = None

    # ----------------------------------------------------------------------------------------------
    #! Representation - helpers
    ################################################################################################

    @staticmethod
    def repr(**kwargs):
        return "Hamiltonian"

    # ----------------------------------------------------------------------------------------------

    def __str__(self):
        """
        Returns the string representation of the Hamiltonian class.

        Returns:
            str :
                The string representation of the Hamiltonian class.
        """
        return f"{'Quadratic' if self._is_quadratic else ''} Hamiltonian with {self._nh} elements and {self._ns} modes."

    def __repr__(self):
        """
        Returns a detailed string representation of the Hamiltonian instance.

        Includes:
            - Hamiltonian type (Many-Body or Quadratic)
            - Number of Hilbert space elements (Nh)
            - Number of modes/sites (Ns)
            - Hilbert space and lattice info
            - Computational backend and dtype
            - Sparsity and memory usage (if available)
        """
        htype = "Many-Body" if self._is_manybody else "Quadratic"
        hilbert_info = f"HilbertSpace(Nh={self._nh})" if self._hilbert_space is not None else ""
        lattice_info = f"Lattice({self._lattice})" if self._lattice is not None else ""
        backend_info = f"backend='{self._backendstr}', dtype={self._dtype}"
        sparse_info = "sparse" if self._is_sparse else "dense"

        return (
            f"<{htype} Hamiltonian | Nh={self._nh}, Ns={self._ns}, "
            f"{hilbert_info}, {lattice_info}, {backend_info}, {sparse_info}>"
        )

    # ----------------------------------------------------------------------------------------------

    def matrix(self, *args, **kwargs):
        """
        Generates the matrix representation.
        Overrides Operator.matrix to provide default Hilbert space.
        """
        # Inject internal Hilbert space if not provided
        # Operator.matrix(dim=None, matrix_type='sparse', dtype=None, hilbert_1=None, hilbert_2=None, use_numpy=True, **kwargs)

        # Check if hilbert_1 is provided in kwargs
        if "hilbert_1" not in kwargs:
            # Check if it might be in args? Operator.matrix args are consumed before keywords.
            # But checking args is hard without inspecting signature.
            # Given standard usage, people pass hilbert_1 as kwarg or positional.
            # If args is empty, safe to inject.
            if not args and self._hilbert_space is not None:
                kwargs["hilbert_1"] = self._hilbert_space

        return super().matrix(*args, **kwargs)

    @property
    def signature(self):
        """Unique signature for the Hamiltonian configuration."""
        # Use instructions if available
        if hasattr(self, '_instr_codes') and self._instr_codes:
            terms = []
            for code, coeff, sites in zip(self._instr_codes, self._instr_coeffs, self._instr_sites):
                # Filter -1 from sites
                real_sites = tuple(s for s in sites if s != -1)
                terms.append((code, complex(coeff), real_sites))

            # Sort to ensure order independence
            terms.sort(key=lambda x: (x[0], x[2], x[1].real, x[1].imag))
            return str(tuple(terms))
        return str(id(self))

    def _check_build(self):
        """Ensure the Hamiltonian is built before use."""
        if not self._is_built:
            self.build()

    def randomize(self, **kwargs):
        """Randomize the Hamiltonian matrix."""
        raise NotImplementedError("Randomization is not implemented for this Hamiltonian class.")

    def clear(self):
        """
        Clears the Hamiltonian matrix and related properties.
        If you want to re-build the Hamiltonian, you need to call the build method again.
        """
        # Call parent's clear to handle matrix, eigenvalues, krylov, diag_engine
        super().clear()

        # Hamiltonian-specific cleanup
        self._hamil = None
        self._hamil_sp = None
        self._delta_sp = None
        self._constant_sp = None
        self._log("Hamiltonian cleared...", lvl=2, color="blue")

    # ----------------------------------------------------------------------------------------------
    #! Basis Transformation Methods
    # ----------------------------------------------------------------------------------------------

    # Class-level registry for basis transformations (subclasses can register handlers)
    _basis_transform_handlers = {}

    @classmethod
    def register_basis_transform(cls, from_basis: str, to_basis: str, handler):
        """
        Register a basis transformation handler for this Hamiltonian class.

        Parameters
        ----------
        from_basis : str
            Source basis (e.g., "real", "k-space")
        to_basis : str
            Target basis
        handler : callable
            Function with signature: handler(self, **kwargs) -> self
            Should modify self in-place and return self

        Example
        -------
        >>> def real_to_kspace(self, **kwargs):
        ...     # Implementation
        ...     return self
        >>> QuadraticHamiltonian.register_basis_transform("real", "k-space", real_to_kspace)
        """
        key = (cls.__name__, from_basis, to_basis)
        cls._basis_transform_handlers[key] = handler

    def _get_basis_transform_handler(self, from_basis: str, to_basis: str):
        """
        Get the registered transformation handler for this transformation.

        Checks the class hierarchy for registered handlers, starting from the most
        specific class (self) and moving up to parent classes. This allows subclasses
        to use handlers registered by their parents.

        Parameters
        ----------
        from_basis : str
            Source basis
        to_basis : str
            Target basis

        Returns
        -------
        callable or None
            The handler function if registered, None otherwise
        """
        # Check class hierarchy for registered handler
        # Start with current class, then check parent classes
        for cls in self.__class__.__mro__:
            key = (cls.__name__, from_basis, to_basis)
            if key in self._basis_transform_handlers:
                return self._basis_transform_handlers[key]

        # No handler found in hierarchy
        return None

    def to_basis(self, basis_type: str, enforce: bool = False, **kwargs):
        r"""
        Transform the Hamiltonian representation to a different basis.

        This is a general dispatcher method that:
        1. Validates the target basis
        2. Checks if already in target basis
        3. Dispatches to registered transformation handlers
        4. Updates basis tracking state
        5. Synchronizes with HilbertSpace (if available)

        Parameters
        ----------
        basis_type : str or HilbertBasisType
            Target basis representation. Examples: "real", "k-space", "fock", "sublattice", "symmetry".

        enforce : bool, optional
            If True and lattice is unavailable, attempt to construct a simple lattice from Ns.
            Default: False.

        **kwargs
            Additional arguments for specific basis transformations (e.g., sublattice_positions for Bloch).

        Returns
        -------
        Hamiltonian
            Self (modified in-place) in the target basis.

        Raises
        ------
        NotImplementedError
            If basis transformation is not supported by this Hamiltonian class.
        ValueError
            If basis_type is invalid.

        Notes
        -----
        Subclasses should register transformation handlers using `register_basis_transform()`.
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        # Normalize basis_type
        if isinstance(basis_type, str):
            try:
                target_basis = HilbertBasisType.from_string(basis_type)
            except ValueError:
                raise ValueError(f"Unknown basis type: {basis_type}")
        else:
            target_basis = basis_type

        # Get current basis
        current_basis = self._current_basis

        # Check if already in target basis
        if target_basis == current_basis:
            self._log(f"Already in {target_basis} basis. Returning self.", lvl=1, color="blue")
            return self

        # Validate transformation is feasible
        is_valid, reason = self.validate_basis_transformation(str(target_basis))
        if not is_valid and not enforce:
            raise ValueError(f"Cannot transform to {target_basis}: {reason}")

        # Look up registered handler
        from_str = str(current_basis).lower()
        to_str = str(target_basis).lower()
        handler = self._get_basis_transform_handler(from_str, to_str)

        if handler is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support transformation from "
                f"{current_basis} to {target_basis}. "
                f"Register handler using `register_basis_transform()`."
            )

        # Execute transformation (handler modifies self in-place)
        self._log(f"Transforming: {current_basis} -> {target_basis}", lvl=1, color="cyan")
        handler(self, enforce=enforce, **kwargs)

        # Synchronize with HilbertSpace if available
        self.sync_basis_with_hilbert_space()

        return self

    def get_basis_type(self) -> str:
        """
        Get the current basis representation of the Hamiltonian.

        Returns
        -------
        str
            The current basis type (default: "real").
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        return getattr(self, "_basis_type", HilbertBasisType.REAL)

    def set_basis_type(self, basis_type: str):
        """
        Set the basis type metadata for this Hamiltonian.

        This primarily updates metadata; actual transformation should use to_basis().

        Parameters
        ----------
        basis_type : str
            Basis type identifier.
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        if isinstance(basis_type, str):
            basis_type = HilbertBasisType.from_string(basis_type)

        self._basis_type = basis_type
        self._log(f"Basis type set to {basis_type}", lvl=2, color="cyan")

    # -------------------------------------------------------------------------
    #! Basis Transformation Infrastructure (General)
    # -------------------------------------------------------------------------

    def _infer_and_set_default_basis(self):
        """
        Infer and set the default basis for this Hamiltonian based on system properties.

        Priority:
        1. If HilbertSpace is provided, inherit its basis type
        2. Otherwise, infer from system properties:
            - Quadratic with lattice:        REAL (position space)
            - Quadratic without lattice:     FOCK (single-particle occupation basis)
            - Many-body with lattice:        REAL (position space / lattice sites)
            - Many-body without lattice:     COMPUTATIONAL (integer basis)

        This is called during initialization to establish the original basis.
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        # Priority 1: Inherit from HilbertSpace if available
        if self._hilbert_space is not None and hasattr(self._hilbert_space, "_basis_type"):

            default_basis = self._hilbert_space._basis_type
            self._basis_metadata["inherited_from_hilbert"] = True
            if self._verbose:
                self._log(
                    f"Hamiltonian basis inherited from HilbertSpace: {default_basis}",
                    lvl=1,
                    color="cyan",
                )
        else:
            # Priority 2: Infer from Hamiltonian properties
            if self._is_quadratic:
                # Quadratic system: choose based on lattice availability
                if self._lattice is not None:
                    default_basis = HilbertBasisType.REAL
                    self._basis_metadata["system_type"] = "quadratic-real"
                else:
                    default_basis = HilbertBasisType.FOCK
                    self._basis_metadata["system_type"] = "quadratic-fock"
            else:
                # Many-body system: choose based on lattice availability
                if self._lattice is not None:
                    default_basis = HilbertBasisType.REAL
                    self._basis_metadata["system_type"] = "manybody-real"
                else:
                    default_basis = HilbertBasisType.FOCK
                    self._basis_metadata["system_type"] = "manybody-fock"

        # Set original and current basis
        self._original_basis = default_basis
        self._current_basis = default_basis
        self._is_transformed = False
        if self._verbose:
            self._log(
                f"Default basis inferred: {default_basis} ({self._basis_metadata.get('system_type', 'unknown')})",
                lvl=2,
                color="cyan",
            )

    def get_transformation_state(self) -> Dict[str, Any]:
        """
        Query the current transformation state of the Hamiltonian.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'original_basis'      : The basis Hamiltonian was created in
            - 'current_basis'       : The basis it's currently represented in
            - 'is_transformed'      : Boolean flag if transformed representation is stored
            - 'has_real_space'      : Whether real-space matrix is available
            - 'has_transformed'     : Whether transformed representation is available
            - 'transformed_shape'   : Shape of _hamil_transformed if available
            - 'grid_shape'          : Shape of _transformed_grid if available
            - 'symmetry_info'       : Information about applied symmetries
            - 'metadata'            : Additional basis metadata
        """
        state = {
            "original_basis": str(self._original_basis) if self._original_basis else None,
            "current_basis": str(self._current_basis) if self._current_basis else None,
            "is_transformed": self._is_transformed,
            "has_real_space": self._hamil is not None or self._hamil_sp is not None,
            "has_transformed": self._hamil_transformed is not None,
            "transformed_shape": (
                self._hamil_transformed.shape if self._hamil_transformed is not None else None
            ),
            "grid_shape": (
                self._transformed_grid.shape if self._transformed_grid is not None else None
            ),
            "symmetry_info": self._symmetry_info,
            "metadata": self._basis_metadata.copy(),
        }
        return state

    def print_transformation_state(self):
        """Print a human-readable summary of the transformation state."""
        state = self.get_transformation_state()
        self._log("Transformation State:", lvl=1, color="bold")
        self._log(f"  Original basis: {state['original_basis']}", lvl=1)
        self._log(f"  Current basis: {state['current_basis']}", lvl=1)
        self._log(f"  Is transformed: {state['is_transformed']}", lvl=1)
        self._log(f"  Real-space available: {state['has_real_space']}", lvl=1)
        self._log(f"  Transformed repr available: {state['has_transformed']}", lvl=1)
        if state["transformed_shape"]:
            self._log(f"  Transformed shape: {state['transformed_shape']}", lvl=1)
        if state["grid_shape"]:
            self._log(f"  Grid shape: {state['grid_shape']}", lvl=1)
        if state["symmetry_info"]:
            self._log(f"  Symmetry: {state['symmetry_info']}", lvl=1)

    def record_symmetry_application(
        self, symmetry_name: Optional[str] = None, sector: Optional[str] = None
    ):
        """
        Record information about applied symmetries to track basis reductions.

        Parameters
        ----------
        symmetry_name : str
            Name of the symmetry applied (e.g., "Z2", "U1", "SU2")
        sector : str, optional
            Which sector of the symmetry (e.g., "even", "odd", "up", "down")
        """
        # Call the parent class's method to handle fetching from HilbertSpace if symmetry_name is None
        super().record_symmetry_application(symmetry_name, sector)

        # Use the (potentially updated by super()) symmetry_name and sector for Hamiltonian-specific recording
        if self._symmetry_info:  # Check if super() successfully set _symmetry_info
            if self._verbose:
                self._log(
                    f"Hamiltonian recorded symmetry application: {self._symmetry_info}",
                    lvl=3,
                    color="yellow",
                )

    def validate_basis_transformation(self, target_basis: str) -> Tuple[bool, str]:
        """
        Validate whether a basis transformation is feasible.

        Parameters
        ----------
        target_basis : str
            The target basis for transformation

        Returns
        -------
        Tuple[bool, str]
            (is_valid, reason_or_warning)
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        try:
            target = HilbertBasisType.from_string(target_basis)
        except ValueError:
            return False, f"Unknown basis type: {target_basis}"

        # Specific validation rules
        if target == HilbertBasisType.KSPACE:
            if self._lattice is None:
                return False, "Cannot transform to k-space: no lattice information available"
            if not hasattr(self._lattice, "is_periodic"):
                return False, "Cannot transform to k-space: lattice periodicity unknown"

        if self._current_basis == target:
            return False, f"Already in {target_basis} basis"

        return True, "Transformation is valid"

    def sync_basis_with_hilbert_space(self):
        """
        Synchronize this Hamiltonian's basis with its HilbertSpace.

        If HilbertSpace has a different basis, update this Hamiltonian to match.
        This ensures consistency between quantum system representation and Hamiltonian representation.
        """
        if self._hilbert_space is None:
            return

        if not hasattr(self._hilbert_space, "_basis_type"):
            return

        hilbert_basis = self._hilbert_space._basis_type

        if self._current_basis != hilbert_basis:
            self._log(
                f"Syncing basis: Hamiltonian {self._current_basis} -> HilbertSpace {hilbert_basis}",
                lvl=2,
                color="yellow",
            )
            self._current_basis = hilbert_basis

    def push_basis_to_hilbert_space(self):
        """
        Push this Hamiltonian's basis information to its HilbertSpace.

        Use this when the Hamiltonian's basis representation is the source of truth.
        """
        if self._hilbert_space is None:
            return

        if hasattr(self._hilbert_space, "set_basis"):
            self._hilbert_space.set_basis(str(self._current_basis))
            self._log(f"Basis pushed to HilbertSpace: {self._current_basis}", lvl=2, color="cyan")

    # -------------------------------------------------------------------------
    #! Getter methods
    # -------------------------------------------------------------------------

    # quadratic Hamiltonian properties

    @property
    def particle_conserving(self):
        return self._particle_conserving if not self._is_manybody else None

    @property
    def is_particle_conserving(self):
        return self.particle_conserving

    @property
    def is_bdg(self):
        return not self.particle_conserving

    # lattice and sites properties

    @property
    def hamil(self) -> Union[np.ndarray, sp.sparse.spmatrix]:
        return self._hamil

    @hamil.setter
    def hamil(self, hamil):
        self._hamil = hamil
        self._matrix = hamil

    @property
    def hamil_transformed(self):
        return self._hamil_transformed

    @hamil_transformed.setter
    def hamil_transformed(self, hamil_transformed):
        self._hamil_transformed = hamil_transformed

    @property
    def grid_transformed(self):
        return self._transformed_grid

    @grid_transformed.setter
    def grid_transformed(self, grid_transformed):
        self._transformed_grid = grid_transformed

    # single-particle Hamiltonian properties
    @property
    def hamil_sp(self):
        return self._hamil_sp

    @hamil_sp.setter
    def hamil_sp(self, hamil_sp):
        self._hamil_sp = hamil_sp

    @property
    def delta_sp(self):
        return self._delta_sp

    @delta_sp.setter
    def delta_sp(self, delta_sp):
        self._delta_sp = delta_sp

    @property
    def constant_sp(self):
        return self._constant_sp

    @constant_sp.setter
    def constant_sp(self, constant_sp):
        self._constant_sp = constant_sp

    # ----------------------------------------------------------------------------------------------
    #! Matrix reference override for many-body vs quadratic
    # ----------------------------------------------------------------------------------------------

    def _get_matrix_reference(self):
        """
        Returns the appropriate matrix based on whether the Hamiltonian is many-body or quadratic.

        For many-body Hamiltonians (_is_manybody=True):
            Returns _hamil (the full many-body Hamiltonian matrix)
        For quadratic/single-particle Hamiltonians (_is_manybody=False):
            Returns _hamil_sp (the single-particle hopping/pairing matrix)

        This ensures that all GeneralMatrix methods (diag, trace, norms, etc.)
        operate on the correct matrix representation.
        """
        if self._is_manybody:
            return self._hamil
        else:
            return self._hamil_sp

    def _set_matrix_reference(self, matrix) -> None:
        """
        Sets the appropriate matrix based on whether the Hamiltonian is many-body or quadratic.
        """
        if self._is_manybody:
            self._hamil = matrix
            self._matrix = matrix
        else:
            self._hamil_sp = matrix

    # ----------------------------------------------------------------------------------------------

    def indices_around_energy(self, energy: float, fraction: Union[int, float] = 500):
        """
        Returns the indices around the average energy of the Hamiltonian.
        This is used to track the average energy during calculations.

        Returns:
            tuple : (index, value) of the average energy.
        """
        hilbert_size = len(self.eig_val)
        if fraction <= 1.0:
            fraction_in = int(fraction * hilbert_size)
        else:
            fraction_in = int(fraction)

        #! get the energy index
        energy_in = self.av_en if energy is None else energy
        energy_index = (
            int(np.argmin(np.abs(self.eig_val - energy_in))) if energy is None else self.av_en_idx
        )

        #! left
        left_index = min(max(0, energy_index - fraction_in // 2), energy_index - 5)
        right_index = max(min(hilbert_size, energy_index + fraction_in // 2), energy_index + 5)

        return left_index, right_index, energy_index

    # ----------------------------------------------------------------------------------------------
    #! Local energy getters
    # ----------------------------------------------------------------------------------------------

    @property
    def fun_int(self):
        return self._loc_energy_int_fun

    def get_loc_energy_int_fun(self):
        return self._loc_energy_int_fun

    @property
    def fun_npy(self):
        return self._loc_energy_np_fun

    def get_loc_energy_np_fun(self):
        return self._loc_energy_np_fun

    @property
    def fun_jax(self):
        return self._loc_energy_jax_fun

    def get_loc_energy_jax_fun(self):
        return self._loc_energy_jax_fun

    def get_loc_energy_arr_fun(self, backend: str = "default", typek: str = "int"):
        """
        Returns the local energy of the Hamiltonian
        Returns:
            A function that takes an integer k and returns the local energy for an array representation in
            a given backend - either NumPy or JAX.
        """
        if typek == "int":
            return self.get_loc_energy_int_fun()
        if typek == "jax":
            return self.get_loc_energy_jax_fun()
        if typek == "npy":
            return self.get_loc_energy_np_fun()

        if (backend == "default" or backend == "jax" or backend == "jnp") and JAX_AVAILABLE:
            return self.fun_jax
        return self.fun_npy

    # ----------------------------------------------------------------------------------------------
    #! Memory properties
    # ----------------------------------------------------------------------------------------------

    @property
    def h_memory(self):
        return super().mat_memory

    @property
    def h_memory_gb(self):
        return self.h_memory / (1024.0**3)

    # ----------------------------------------------------------------------------------------------
    #! Standard getters
    # ----------------------------------------------------------------------------------------------

    def get_mean_lvl_spacing(self, use_npy=True):
        """
        Returns the mean level spacing of the Hamiltonian. The mean level spacing is defined as the
        average difference between consecutive eigenvalues.

        Returns:
            float : The mean level spacing of the Hamiltonian.
        """
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        if not JAX_AVAILABLE or self._backend == np or use_npy:
            return self._backend.mean(self._backend.diff(self._eig_val))
        return hjm.mean_level_spacing(self._eig_val)

    def get_bandwidth(self):
        """
        Returns the bandwidth of the Hamiltonian. The bandwidth is defined as the difference between
        the highest and the lowest eigenvalues - values are sorted in ascending order.
        """
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        return self._eig_val[-1] - self._eig_val[0]

    def get_energywidth(self, use_npy=True):
        """
        Returns the energy width of the Hamiltonian. The energy width is defined as trace of the
        Hamiltonian matrix squared.
        """
        if self._hamil.size == 0:
            raise ValueError(Hamiltonian._ERR_HAMILTONIAN_NOT_AVAILABLE)
        if not JAX_AVAILABLE or self._backend == np or use_npy:
            return self._backend.trace(self._backend.dot(self._hamil, self._hamil))
        return hjm.energy_width(self._hamil)

    # ----------------------------------------------------------------------------------------------

    def init(self, use_numpy: bool = False):
        """
        Initializes the Hamiltonian matrix. Uses Batched-coordinate (BCOO) sparse matrices if JAX is
        used, otherwise uses NumPy arrays. The Hamiltonian matrix is initialized to be a matrix of
        zeros if the Hamiltonian is not sparse, otherwise it is initialized to be an empty sparse
        matrix.

        Parameters:
            use_numpy (bool):
                A flag indicating whether to use NumPy or JAX.
        """
        self._log("Initializing the Hamiltonian matrix...", lvl=2, log="debug")

        jax_maybe_avail = self._is_jax
        if jax_maybe_avail and use_numpy:
            self._log("JAX is available but NumPy is forced...", lvl=1, log="warning")
            jax_maybe_avail = False

        if self._is_quadratic:
            # Initialize Quadratic Matrix (_hamil_sp)
            # Shape determined by subclass, assume (Ns, Ns) for now
            ham_shape = getattr(self, "_hamil_sp_shape", (self._ns, self._ns))
            self._log(
                f"Initializing Quadratic Hamiltonian structure {ham_shape} (Sparse={self.sparse})...",
                lvl=3,
                log="debug",
            )

            if self.sparse:
                if self._is_numpy:
                    self._hamil_sp = sp.sparse.csr_matrix(ham_shape, dtype=self._dtype)
                else:
                    indices = self._backend.zeros((0, 2), dtype=np.int64)
                    data = self._backend.zeros((0,), dtype=self._dtype)
                    self._hamil_sp = BCOO((data, indices), shape=ham_shape)
                    self._delta_sp = BCOO((data, indices), shape=ham_shape)
            else:
                self._hamil_sp = self._backend.zeros(ham_shape, dtype=self._dtype)
                self._delta_sp = self._backend.zeros(ham_shape, dtype=self._dtype)
            self._hamil = None
        else:
            if self.sparse:
                self._log(
                    "Initializing the Hamiltonian matrix as a sparse matrix...", lvl=3, log="debug"
                )

                # --------------------------------------------------------------------------------------

                if not jax_maybe_avail or use_numpy:
                    self._log(
                        "Initializing the Hamiltonian matrix as a CSR sparse matrix...",
                        lvl=3,
                        log="debug",
                    )
                    self._hamil = sp.sparse.csr_matrix((self._nh, self._nh), dtype=self._dtype)
                else:
                    self._log(
                        "Initializing the Hamiltonian matrix as a sparse matrix...",
                        lvl=3,
                        log="debug",
                    )
                    # Create an empty sparse Hamiltonian matrix using JAX's BCOO format
                    indices = self._backend.zeros((0, 2), dtype=ACTIVE_INT_TYPE)
                    data = self._backend.zeros((0,), dtype=self._dtype)
                    self._hamil = BCOO((data, indices), shape=(self._nh, self._nh))

                # --------------------------------------------------------------------------------------

            else:
                self._log(
                    "Initializing the Hamiltonian matrix as a dense matrix...", lvl=3, log="debug"
                )
                if not JAX_AVAILABLE or self._backend == np:
                    self._hamil = self._backend.zeros((self._nh, self._nh), dtype=self._dtype)
                else:
                    # do not initialize the Hamiltonian matrix
                    self._hamil = None
        self._log(
            f"Hamiltonian matrix initialized and it's {'many-body' if self._is_manybody else 'quadratic'}",
            lvl=3,
            color="green",
            log="debug",
        )

    # ----------------------------------------------------------------------------------------------
    #! Many body Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian_validate(self):
        """Check if the Hamiltonian matrix is valid."""

        matrix_to_check = self._hamil if (self._is_manybody) else self._hamil_sp

        if matrix_to_check is None:
            self._log("Hamiltonian matrix is not initialized.", lvl=3, color="red", log="debug")
        else:
            valid = False
            # For dense matrices (NumPy/JAX ndarray) which have the 'size' attribute.
            if hasattr(matrix_to_check, "size"):
                if matrix_to_check.size > 0:
                    valid = True
            # For SciPy sparse matrices: check the number of nonzero elements.
            elif hasattr(matrix_to_check, "nnz"):
                if matrix_to_check.nnz > 0:
                    valid = True
            # For JAX sparse matrices (e.g., BCOO): verify if the data array has entries.
            elif hasattr(matrix_to_check, "data") and hasattr(matrix_to_check, "indices"):
                if matrix_to_check.data.shape[0] > 0:
                    valid = True

            if valid:
                self._log(
                    "Hamiltonian matrix calculated and valid.", lvl=3, color="green", log="debug"
                )
            else:
                self._log(
                    "Hamiltonian matrix calculated but empty or invalid.",
                    lvl=3,
                    color="red",
                    log="debug",
                )
                matrix_to_check = None

    # ----------------------------------------------------------------------------------------------

    def build(self, verbose: bool = False, use_numpy: bool = True, force: bool = False):
        """
        Builds the Hamiltonian matrix. It checks the internal masks
        wheter it's many-body or quadratic...

        Args:
            verbose (bool) :
                A flag to indicate whether to print the progress of the build.
            use_numpy (bool) :
                Force numpy usage.

        """
        if self.hamil is not None:
            if not force:
                self._log("Hamiltonian matrix already built. Use force=True to rebuild.", lvl=1)
                return
            else:
                self._log("Forcing rebuild of the Hamiltonian matrix...", lvl=1)
                self.hamil = None  # Clear existing Hamiltonian to force rebuild

        if verbose:
            self._log(
                f"Building Hamiltonian (Type: {'Many-Body' if self._is_manybody else 'Quadratic'})...",
                lvl=1,
                color="orange",
            )

        if self._is_manybody:
            # Ensure operators/local energy functions are defined
            if self._loc_energy_int_fun is None and self._loc_energy_np_fun is None and self._loc_energy_jax_fun is None:
                self._log("Local energy functions not set, attempting to set them...", lvl=2, log="debug")
                try:
                    # Only reset/re-add if we don't have instructions yet
                    if not self._instr_codes:
                        self._log("No instructions found, calling _set_local_energy_operators...", lvl=2, log="debug")
                        self._set_local_energy_operators()

                    # Ensure instruction codes are setup (lookup tables etc)
                    if hasattr(self, 'setup_instruction_codes'):
                        self.setup_instruction_codes()

                    # Build the functions
                    self._set_local_energy_functions()
                except Exception as e:
                    raise RuntimeError(f"Failed to set up operators/local energy functions: {e}")

            # Setup instruction codes if not already done (critical for signature)
            if hasattr(self, 'setup_instruction_codes') and not getattr(self, '_lookup_codes', None):
                 self.setup_instruction_codes()

        # Check Cache
        if not force and self._is_manybody:
            cache_key     = generate_cache_key(self)
            cached_matrix = get_matrix_from_cache(cache_key)

            if cached_matrix is not None:
                self._log("Hamiltonian matrix found in cache. Using cached version.", lvl=1, color="green")
                self.hamil      = cached_matrix
                self._is_built  = True
                return

        ################################
        #! Initialize the Hamiltonian
        ################################
        init_start = time.perf_counter()
        try:
            self.init(use_numpy)
        except Exception as e:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_INITIALIZATION} : {str(e)}") from e

        if self._is_quadratic:
            if hasattr(self._hamil_sp, "block_until_ready"):
                self._hamil_sp = self._hamil_sp.block_until_ready()

            if hasattr(self._delta_sp, "block_until_ready"):
                self._delta_sp = self._delta_sp.block_until_ready()

        if hasattr(self._hamil, "block_until_ready"):
            self._hamil = self._hamil.block_until_ready()

        # initialize duration
        init_duration = time.perf_counter() - init_start
        if verbose:
            self._log(f"Initialization completed in {init_duration:.6f} seconds", lvl=2)

        ################################
        #! Build the Hamiltonian matrix
        ################################§
        ham_start = time.perf_counter()
        try:
            if self._is_manybody:
                self._hamiltonian(use_numpy)
                # Store in cache
                if not force: # Only cache if not forced? Or always?
                    # Recalculate key just in case signature changed? No, it shouldn't.
                    cache_key = generate_cache_key(self)
                    store_matrix_in_cache(cache_key, self._hamil)
            else:
                self._hamiltonian_quadratic(use_numpy)
        except Exception as e:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_BUILD} : {str(e)}") from e
        ham_duration = time.perf_counter() - ham_start

        if (self._hamil is not None and self._hamil.size > 0) or (
            self._hamil_sp is not None and self._hamil_sp.size > 0
        ):
            if verbose:
                self._log(f"Hamiltonian matrix built in {ham_duration:.6f} seconds.", lvl=1)
        else:
            raise ValueError(
                f"{Hamiltonian._ERR_HAMILTONIAN_BUILD} : The Hamiltonian matrix is empty or invalid."
            )
        self._is_built = True

    # ----------------------------------------------------------------------------------------------
    #! Local energy methods - Abstract methods
    # ----------------------------------------------------------------------------------------------

    def loc_energy_int(self, k_map: int, i: int):
        """
        Calculates the local energy (off-diagonal terms) for a state.

        This low-level function determines how the Hamiltonian acts on a single basis
        state (represented by an integer index `k_map`). It returns the indices
        of the connected states and the corresponding matrix elements.

        Parameters
        ----------
        k_map : int
            The integer index representing the basis state (column index).
        i : int
            The site index (argument used for compatibility, typically unused for
            global Hamiltonian action but relevant for local updates).

        Returns
        -------
        row_indices : np.ndarray
            Array of integers (int64) representing the indices of the connected basis states.
        values : np.ndarray
            Array of matrix elements (complex128 or float64) corresponding to the transitions.
            H_{row, k_map} = values

        Notes
        -----
        This function must return NumPy arrays regardless of the active backend, as it is
        primarily used for constructing sparse matrices on the CPU.
        """
        if self._loc_energy_int_fun is None:
            self._set_local_energy_functions()
        return self._loc_energy_int_fun(k_map, i)

    def loc_energy_arr_jax(self, k: Union[int, np.ndarray]):
        """
        Calculates the local energy based on the Hamiltonian. This method should be implemented by subclasses.
        Uses an array as a state input.
        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        """
        return self._loc_energy_jax_fun(k)

    def loc_energy_arr_np(self, k: Union[np.ndarray]):
        """
        Calculates the local energy based on the Hamiltonian. This method should be implemented by subclasses.
        Uses an array as a state input.
        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        """
        return self._loc_energy_np_fun(k)

    def loc_energy_arr(self, k: Union[int, np.ndarray]) -> Tuple[List[int], List[int]]:
        """
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by
        subclasses to provide a specific implementation.
        This is meant to check how does the Hamiltonian act on a state.
        Parameters:
            k (Union[int, np.ndarray]) : The k'th element of the Hilbert space - may use mapping if necessary.
        Returns:
            Tuple[List[int], List[int]] :  (row_indices, values)
                - row_indices               : The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        """
        if self._is_jax or isinstance(k, jnp.ndarray):
            if self._loc_energy_jax_fun is None:
                self._set_local_energy_functions()
            return self.loc_energy_arr_jax(k)

        # go!
        if self._loc_energy_np_fun is None:
            self._set_local_energy_functions()
        return self.loc_energy_arr_np(k)

    def loc_energy(self, k: Union[int, np.ndarray], i: int = 0):
        """
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by
        subclasses to provide a specific implementation.

        This is meant to check how does the Hamiltonian act on a state at a given site.

        Parameters:
            k (Union[int, Backend.ndarray])         : The k'th element of the Hilbert space - may use mapping if necessary.
            i (int)                                 : The i'th site.
        """
        if isinstance(k, (int, np.integer, int, jnp.integer)):
            return self.loc_energy_int(self._hilbert_space[k], i)
        elif isinstance(k, List):
            # concatenate the results
            rows, cols, data = [], [], []
            for k_i in k:
                # run the local energy calculation for a single element
                new_rows, new_cols, new_data = self.loc_energy(k_i, i)
                rows.extend(new_rows)
                cols.extend(new_cols)
                data.extend(new_data)
            return rows, cols, data
        # otherwise, it is an array (no matter which backend)
        return self.loc_energy_arr(k)

    # ----------------------------------------------------------------------------------------------
    # ! Hamiltonian matrix calculation
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian(self, use_numpy: bool = False, sparse: Optional[bool] = None):
        """
        Generates the Hamiltonian matrix. The diagonal elements are straightforward to calculate,
        while the off-diagonal elements are more complex and depend on the specific Hamiltonian.
        It iterates over the Hilbert space to calculate the Hamiltonian matrix.

        Note: This method may be overridden by subclasses to provide a more efficient implementation
        """

        if not self._is_manybody:
            raise ValueError(Hamiltonian._ERR_MODE_MISMATCH)

        if self._hilbert_space is None or self._nh == 0:
            raise ValueError(Hamiltonian._ERR_HILBERT_SPACE_NOT_PROVIDED)

        if self._loc_energy_int_fun is None and (use_numpy or self._is_numpy):
            raise RuntimeError("MB build requires local energy functions (_loc_energy_int_fun).")

        if sparse is not None:
            self._is_sparse = sparse

        # -----------------------------------------------------------------------------------------
        matrix_type = "sparse" if self.sparse else "dense"
        self._log(
            f"Calculating the {matrix_type} Hamiltonian matrix...", lvl=1, color="blue", log="debug"
        )
        # -----------------------------------------------------------------------------------------

        # Check if JAX is available and the backend is not NumPy
        jax_maybe_av = self._is_jax

        # Choose implementation based on backend availability.sym_eig_py
        if not jax_maybe_av or use_numpy:
            self._log("Calculating the Hamiltonian matrix using NumPy...", lvl=2, log="info")

            # Calculate the Hamiltonian matrix using the optimized matrix builder
            self._hamil = build_operator_matrix(
                hilbert_space=self._hilbert_space,
                operator_func=self._loc_energy_int_fun,
                sparse=self._is_sparse,
                max_local_changes=self._max_local_ch_o,
                dtype=self._dtype,
                ns=self._ns,
                nh=self._hilbert_space.nh,
            )
        else:
            raise ValueError("JAX not yet implemented for the build...")

        # Check if the Hamiltonian matrix is calculated and valid using various backend checks
        self._hamiltonian_validate()

    # ----------------------------------------------------------------------------------------------
    #! Single particle Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        """
        Generates the Hamiltonian matrix whenever the Hamiltonian is single-particle.
        This method needs to be implemented by the subclasses.
        """
        pass

    # ----------------------------------------------------------------------------------------------
    #! Calculators
    # ----------------------------------------------------------------------------------------------

    def _calculate_av_en(self):
        """
        Calculates the properties of the Hamiltonian matrix that are related to the energy.
        """

        if self._eig_val is None or self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        self._av_en = self._backend.mean(self._eig_val)
        self._min_en = self._backend.min(self._eig_val)
        self._max_en = self._backend.max(self._eig_val)
        self._std_en = self._backend.std(self._eig_val)
        self._nh = self._eig_val.size

        # average energy index
        self._av_en_idx = self._backend.argmin(self._backend.abs(self._eig_val - self._av_en))

    def calculate_en_idx(self, en: float):
        """
        Calculates the index of the energy level closest to the given energy.

        Args:
            en (float) : The energy level.

        Returns:
            int : The index of the energy level closest to the given energy.
        """
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        return self._backend.argmin(self._backend.abs(self._eig_val - en))

    # ----------------------------------------------------------------------------------------------
    #! Diagonalization methods
    # ----------------------------------------------------------------------------------------------

    def _diagonalize_quadratic_prepare(self, backend, **kwargs):
        """
        Diagonalizes the Hamiltonian matrix whenever the Hamiltonian is single-particle.
        This method needs to be implemented by the subclasses.
        """
        if self._is_quadratic:
            if True:
                if self.particle_conserving:
                    self._log(
                        "Diagonalizing the quadratic Hamiltonian matrix without BdG...",
                        lvl=2,
                        log="debug",
                    )
                    self._hamil = self._hamil_sp
                else:
                    self._log(
                        "Diagonalizing the quadratic Hamiltonian matrix with BdG...",
                        lvl=2,
                        log="debug",
                    )
                    if self._isfermions:
                        self._hamil = backend.block(
                            [
                                [self._hamil_sp, self._delta_sp],
                                [-self._delta_sp.conj(), -self._hamil_sp.conj().T],
                            ]
                        )
            else:  # bosons - use \Sigma H to make it Hermitian
                sigma = backend.block(
                    [
                        [backend.eye(self.ns), backend.zeros_like(self._hamil)],
                        [backend.zeros_like(self._hamil), -backend.eye(self.ns)],
                    ]
                )
                self._hamil = sigma @ backend.block(
                    [
                        [self._hamil_sp, self._delta_sp],
                        [self._delta_sp.conj().T, self._hamil_sp.conj().T],
                    ]
                )

    def diagonalize(
        self, method: str = "exact", build: bool = False, verbose: bool = False, **kwargs
    ):
        """
        Diagonalizes the Hamiltonian matrix using a modular, flexible approach.

        This method provides a unified interface to multiple diagonalization strategies:

        method:
            - 'auto'            : Automatically select method based on matrix size/properties
            - 'exact'           : Full diagonalization (all eigenvalues)
            - 'lanczos'         : Lanczos iteration for sparse symmetric matrices
                - 'nh'              : Hamiltonian size threshold for auto Lanczos
                - 'k'               : Number of eigenvalues to compute
                - 'which'           : Which eigenvalues to find
                - 'tol'             : Convergence tolerance
                - 'maxiter'         : Maximum iterations
                - 'sigma'           : Shift for shift-invert mode

        -----------
        Additional iterative methods:
        - 'block_lanczos'   : Block Lanczos for multiple eigenpairs
        - 'arnoldi'         : Arnoldi iteration for general matrices
        - 'shift-invert'    : Shift-invert mode for interior eigenvalues
        - 'OPInv'           : Operator Inversion method - for SciPy backend only

        Other features:
        -----------
        - Backend selection: NumPy, SciPy, JAX
        - Krylov basis storage for transformations
        - Convergence control and diagnostics

        The method stores Krylov basis information when using iterative methods,
        enabling transformations between Krylov and original basis spaces.

        Parameters:
        -----------

        verbose : bool, optional
            Enable verbose output. Default is False.

        Method Selection:
        --------
        method : str
            Diagonalization method ('auto', 'exact', 'lanczos',
            'block_lanczos', 'arnoldi', 'shift-invert'). Default: 'auto'.
        backend : str
            Computational backend ('numpy', 'scipy', 'jax').
            Default: inferred from Hamiltonian.
        use_scipy : bool
            Prefer SciPy implementations. Default: True.

        Eigenvalue Selection:
        --------
        k : int
            Number of eigenvalues to compute. Default: 6 for iterative,
            all for exact.
        which : str
            Which eigenvalues to find:
            - Lanczos/Block Lanczos: 'smallest', 'largest', 'both'
            - Arnoldi: 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
            - Shift-invert: eigenvalues nearest to sigma
            Default: 'smallest'.
        sigma : float
            Shift value for shift-invert mode. Finds eigenvalues
            nearest to this value. Default: 0.0.

        Convergence Control:
        --------
        tol : float
            Convergence tolerance. Default: 1e-10.
        maxiter : int
            Maximum number of iterations. Default: auto.

        Block Lanczos Specific:
        --------
        block_size : int
            Size of block for Block Lanczos. Default: min(k, 3).
        reorthogonalize : bool
            Enable reorthogonalization. Default: True.

        Basis Storage:
        --------
        store_basis : bool
            Store Krylov basis for transformations. Default: True.

        Updates:
        --------
        self._eig_val : ndarray
            Computed eigenvalues (sorted).
        self._eig_vec : ndarray
            Computed eigenvectors (columns).
        self._krylov : ndarray
            Krylov basis (for iterative methods, if store_basis=True).
        self._diag_engine : DiagonalizationEngine
            Engine instance with full result information.

        Examples:
        ---------
            >>> # Auto-select method and compute all eigenvalues (small matrix)
            >>> hamil.diagonalize(verbose=True)

            >>> # Lanczos for 10 smallest eigenvalues
            >>> hamil.diagonalize(method='lanczos', k=10, which='smallest')

            >>> # Block Lanczos for 20 eigenvalues with custom block size
            >>> hamil.diagonalize(method='block_lanczos', k=20, block_size=5,
            ...                   tol=1e-8, verbose=True)

            >>> # Use JAX backend for GPU acceleration
            >>> hamil.diagonalize(method='exact', backend='jax')

        Raises:
        -------
            ValueError
                If Hamiltonian matrix not available or invalid parameters.
            RuntimeError
                If diagonalization fails.

        See Also:
        ---------
            to_original_basis : Transform vector from Krylov to original basis
            to_krylov_basis : Transform vector from original to Krylov basis
            has_krylov_basis : Check if Krylov basis is available
        """

        if self.nh == 0:
            self._log("The Hilbert space is empty, skipping!", log="warning", lvl=3, color="yellow")
            return

        # Start timing
        diag_start      = time.perf_counter()

        # Determine build strategy
        use_numpy       = kwargs.get("use_numpy", True)
        force_build     = kwargs.get('force',     False)

        if build:
            self.build(verbose=verbose, use_numpy=use_numpy, force=force_build)
        else:
            if method == 'exact':
                # Exact diagonalization requires the matrix.
                # If matrix not present, we MUST build it (checking cache first).
                self._check_build()
            else:
                # Iterative methods (lanczos, etc.) can operate matrix-free via matvec.
                # We do NOT force full matrix construction here.
                # But we MUST ensure operators/functions are setup.
                if self._is_manybody:
                    if self._loc_energy_int_fun is None:
                        # Setup instruction codes and functions if missing
                        if hasattr(self, 'setup_instruction_codes') and not self._instr_codes:
                            self.setup_instruction_codes()

                        # Only set local energy functions if we have instructions or it's a legacy build
                        if self._instr_codes or not hasattr(self, '_instr_codes'):
                            self._set_local_energy_functions()

        # Extract parameters
        if True:
            method = method or self._diag_method

            backend_str = kwargs.get("backend", None)
            if backend_str in kwargs:
                kwargs.pop("backend")

            use_scipy = kwargs.get("use_scipy", True)
            if use_scipy in kwargs:
                kwargs.pop("use_scipy")

            store_basis = kwargs.get("store_basis", False)
            if store_basis in kwargs:
                kwargs.pop("store_basis")

            k = kwargs.get("k", None)
            if k in kwargs:
                kwargs.pop("k")

            which = kwargs.get("which", "smallest")
            if which in kwargs:
                kwargs.pop("which")

            hermitian = kwargs.get("hermitian", True)
            if hermitian in kwargs:
                kwargs.pop("hermitian")

        # Determine backend
        if backend_str is None:
            # Infer from current Hamiltonian backend
            if self._is_jax:
                backend_str = "jax"
            else:
                backend_str = "scipy" if use_scipy else "numpy"

        # Special handling for quadratic Hamiltonians
        if self._is_quadratic:
            self._diagonalize_quadratic_prepare(self._backend, **kwargs)

        # Log start
        if verbose:
            self._log(
                f"Diagonalization started using method='{method}'...",
                lvl=1,
                verbose=verbose,
                color="blue",
            )
            if k is not None and method != "exact":
                self._log(f"Computing {k} eigenvalues", lvl=2, verbose=verbose)
            self._log(f"Backend: {backend_str}", lvl=2, verbose=verbose)

        # Initialize or reuse diagonalization engine
        if self._diag_engine is None or self._diag_engine.method != method:
            self._diag_engine = DiagonalizationEngine(
                method=method,
                backend=backend_str,
                use_scipy=use_scipy,
                verbose=verbose,
                logger=self._logger,
            )

        # Prepare solver kwargs (remove our custom parameters)
        solver_kwargs = {
            key: val
            for key, val in kwargs.items()
            if key
            not in [
                "method",
                "backend",
                "use_scipy",
                "hilbert",
                "store_basis",
                "hermitian",
                "k",
                "which",
            ]
        }

        matrix_to_diag = self._hamil
        if method == "exact":
            if JAX_AVAILABLE and BCOO is not None and isinstance(matrix_to_diag, BCOO):
                if verbose:
                    self._log(
                        "Converting JAX sparse Hamiltonian to dense array for exact diagonalization.",
                        lvl=2,
                        color="yellow",
                    )
                matrix_to_diag = np.asarray(matrix_to_diag.todense())
            elif sp.sparse.issparse(matrix_to_diag):
                if verbose:
                    self._log(
                        "Converting SciPy sparse Hamiltonian to dense array for exact diagonalization.",
                        lvl=2,
                        color="yellow",
                    )
                matrix_to_diag = matrix_to_diag.toarray()

        # Perform diagonalization
        try:
            # Create matvec closure with hilbert space for thread buffer pre-allocation
            try:
                _matvec_with_hilbert = self.matvec_fun
            except (AttributeError, ValueError):
                # Fallback if matvec_fun cannot be created (e.g. QuadraticHamiltonian without operator function)
                _matvec_with_hilbert = None

            result = self._diag_engine.diagonalize(
                A=matrix_to_diag,
                matvec=_matvec_with_hilbert,
                n=self._nh,
                k=k,
                hermitian=hermitian,
                which=which,
                store_basis=store_basis,
                dtype=self._dtype,
                **solver_kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Diagonalization failed with method '{method}': {e}") from e

        # Store results in Hamiltonian
        # Ensure eigenvalues are numpy array (safeguard against backend returning tuples/lists)
        self._eig_val = np.asarray(result.eigenvalues) if result.eigenvalues is not None else None
        self._eig_vec = result.eigenvectors

        # Store Krylov basis if available
        if store_basis and self._diag_engine.has_krylov_basis():
            self._krylov = self._diag_engine.get_krylov_basis()
        else:
            self._krylov = None

        # JAX: ensure results are materialized
        if JAX_AVAILABLE:
            if hasattr(self._eig_val, "block_until_ready"):
                self._eig_val = self._eig_val.block_until_ready()

        # Calculate average energy
        self._calculate_av_en()

        # Log completion
        diag_duration = time.perf_counter() - diag_start

        if verbose:
            method_used = self._diag_engine.get_method_used()
            self._log(
                f"Diagonalization ({method_used}) completed in {diag_duration:.6f} seconds, {diag_duration/3600:.6f} hours.",
                lvl=2,
                color="green",
            )
            self._log(f"  Ground state energy: {self._eig_val[0]:.10f}", lvl=2, log="debug")

    # ----------------------------------------------------------------------------------------------
    #! Energy related methods -> building the Hamiltonian
    # ----------------------------------------------------------------------------------------------

    def reset_operators(self):
        """
        Resets the Hamiltonian operators and instruction codes.
        Call this to clear all added operators before rebuilding the Hamiltonian.
        """

        # Reset all operator terms and instructions via parent class method
        self.reset_operator_terms()

        # Reset Hamiltonian-specific local energy function aliases
        self._loc_energy_int_fun = None
        self._loc_energy_np_fun = None
        self._loc_energy_jax_fun = None

    def _set_local_energy_operators(self):
        """
        This function is meant to be overridden by subclasses to set the local energy operators.
        The local energy operators are used to calculate the local energy of the Hamiltonian.
        Note:
            It is the internal function that knows about the structure of the Hamiltonian.
        """
        self.reset_operators()

    def _set_local_energy_functions(self):
        """
        Configure and set local energy functions for different numerical backends.

        This method initializes three versions of the local energy function:
            - Integer operations: JIT-compiled function using instruction codes
            - NumPy operations: Function using NumPy array operations
            - JAX operations: Function using JAX for GPU acceleration (if available)

        Notes
        -----
        Uses the parent class `_build_composition_functions()` for the integer version,
        then adds Hamiltonian-specific NumPy and JAX wrappers.
        """

        if not self._is_manybody:
            self._log(
                "Skipping local energy function setup for Quadratic Hamiltonian.", log="debug"
            )
            return

        # Build the integer composition function using parent class method
        # Set Hamiltonian's local energy function alias to the composition function
        self._build_composition_functions()
        self._loc_energy_int_fun = self._composition_int_fun

        if self._loc_energy_int_fun is None:
            self._log(
                "Integer local energy function not available.", lvl=3, color="red", log="error"
            )
            # return

        # Set the NumPy functions
        try:
            operators_mod_np = [
                [(op.npy, sites, vals) for (op, sites, vals) in self._ops_mod_sites[i]]
                for i in range(self.ns)
            ]
            operators_mod_np_nsites = [
                [(op.npy, [], vals) for (op, _, vals) in self._ops_mod_nosites[i]]
                for i in range(self.ns)
            ]
            operators_nmod_np = [
                [(op.npy, sites, vals) for (op, sites, vals) in self._ops_nmod_sites[i]]
                for i in range(self.ns)
            ]
            operators_nmod_np_nsites = [
                [(op.npy, [], vals) for (op, _, vals) in self._ops_nmod_nosites[i]]
                for i in range(self.ns)
            ]
            self._loc_energy_np_fun = local_energy_np_wrap(
                self.ns,
                operator_terms_list=operators_mod_np,
                operator_terms_list_ns=operators_mod_np_nsites,
                operator_terms_list_nmod=operators_nmod_np,
                operator_terms_list_nmod_ns=operators_nmod_np_nsites,
                n_max=self._max_local_ch,
                dtype=self._dtype,
            )
        except Exception as e:
            self._log(
                f"Failed to set NumPy local energy functions: {e}", lvl=3, color="red", log="error"
            )
            self._loc_energy_np_fun = None

        # Set the JAX functions
        if JAX_AVAILABLE:
            try:
                operators_jax = [
                    [(op.jax, sites, vals) for (op, sites, vals) in self._ops_mod_sites[i]]
                    for i in range(self.ns)
                ]
                operators_jax_nosites = [
                    [(op.jax, None, vals) for (op, _, vals) in self._ops_mod_nosites[i]]
                    for i in range(self.ns)
                ]
                operators_local_jax = [
                    [(op.jax, sites, vals) for (op, sites, vals) in self._ops_nmod_sites[i]]
                    for i in range(self.ns)
                ]
                operators_local_jax_nosites = [
                    [(op.jax, None, vals) for (op, _, vals) in self._ops_nmod_nosites[i]]
                    for i in range(self.ns)
                ]
                self._loc_energy_jax_fun = local_energy_jax_wrap(
                    self.ns,
                    operator_terms_list=operators_jax,
                    operator_terms_list_ns=operators_jax_nosites,
                    operator_terms_list_nmod=operators_local_jax,
                    operator_terms_list_nmod_ns=operators_local_jax_nosites,
                    n_max=self._max_local_ch,
                    dtype=self._dtype,
                )
            except Exception as e:
                self._log(
                    f"Failed to set JAX local energy functions: {e}",
                    lvl=3,
                    color="red",
                    log="error",
                )
        else:
            self._log(
                "JAX is not available, skipping JAX local energy function setup.",
                lvl=3,
                color="yellow",
                log="debug",
            )
            self._loc_energy_jax_fun = None

        # Calculate max local changes from operator lists
        self._max_local_ch_o = max(
            self._max_local_ch_o,
            max(len(op) for op in self._ops_mod_sites)
            + max(len(op) for op in self._ops_mod_nosites)
            + max(len(op) for op in self._ops_nmod_sites)
            + max(len(op) for op in self._ops_nmod_nosites),
        )

        # Set the OperatorFunction for Operator inheritance
        self._fun = OperatorFunction(
            fun_int=self._loc_energy_int_fun,
            fun_np=self._loc_energy_np_fun,
            fun_jax=self._loc_energy_jax_fun,
            modifies_state=True,
            necessary_args=0,
        )
        self._type_acting = OperatorTypeActing.Global

        self._log(
            f"Max local changes set to {self._max_local_ch_o}", lvl=2, color="green", log="debug"
        )
        self._log("Successfully set local energy functions...", lvl=2, log="debug")

    # ----------------------------------------------------------------------------------------------
    # Help and Documentation
    # ----------------------------------------------------------------------------------------------

    @classmethod
    def help(cls, topic: Optional[str] = None) -> str:
        """
        Display help information about Hamiltonian capabilities.

        Parameters
        ----------
        topic : str, optional
            Specific topic to get help on. Options:
            - None or 'all':
                Full overview
            - 'construction':
                Building Hamiltonians
            - 'operators':
                Adding operators to Hamiltonian
            - 'local_energy':
                Local energy functions
            - 'diagonalization':
                Diagonalization (inherited)
            - 'quadratic':
                Quadratic Hamiltonians
            - 'kspace':
                K-space methods
            - 'inherited':
                Methods from Operator/GeneralMatrix

        Returns
        -------
        str
            Help text for the requested topic.

        Examples
        --------
        >>> Hamiltonian.help()              # Full overview
        >>> Hamiltonian.help('operators')   # Adding operators
        """

        try:
            from QES.Algebra.Hamil.hamil_help import hamil_overview, hamil_topics
        except ImportError as e:
            return f"Help documentation is not available: {e}"

        if topic is None or topic == "all":
            result = hamil_overview
            for t in hamil_topics.values():
                result += t
            print(result)
            return result

        if topic in hamil_topics:
            print(hamil_topics[topic])
            return hamil_topics[topic]

        return f"Unknown topic '{topic}'. Available: {list(hamil_topics.keys())}"


# --------------------------------------------------------------------------------------------------
#! End of File
# --------------------------------------------------------------------------------------------------
