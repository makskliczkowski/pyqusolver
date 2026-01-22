"""
BasisAwareOperator - Intermediate class for operators with basis transformation support.

This module provides the BasisAwareOperator class, which extends SpecialOperator
to handle basis representation tracking and transformation dispatching.

This abstraction separates basis-related logic from physics-specific implementations,
allowing multiple operator types (Hamiltonian, Lindbladian, Observables) to share
basis transformation infrastructure.

Hierarchy:
    Operator                -> Base matrix operations
    SpecialOperator         -> Instruction codes, custom operators
    BasisAwareOperator      -> Basis tracking, transformation dispatch (THIS CLASS)
    Hamiltonian             -> Physics: many-body/quadratic, eigenvalues, etc.

------------------------------------------------------------------------
File                        : Algebra/Operator/basis_aware_operator.py
Author                      : Maksymilian Kliczkowski
Date                        : December 2025
Description                 : BasisAwareOperator class for basis transformations. An intermediate class
                            between SpecialOperator and physics-specific operators like Hamiltonian.
License                     : MIT
------------------------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from QES.Algebra.Operator.special_operator import SpecialOperator
except ImportError as e:
    raise ImportError("SpecialOperator is required for BasisAwareOperator.") from e

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType
    from QES.general_python.common.flog import Logger
    from QES.general_python.lattices.lattice import Lattice

# ---------------------------------------------------------------------------
#! BasisAwareOperator Class
# ---------------------------------------------------------------------------


class BasisAwareOperator(SpecialOperator, ABC):
    """
    Intermediate class for operators that need basis representation tracking.

    This class extends SpecialOperator to provide:

    1. **Basis Tracking**:                  Maintains original and current basis representation
    2. **Transformation Registry**:         Class-level registry for basis transform handlers
    3. **Transformation Dispatch**:         Validates and dispatches to registered handlers
    4. **HilbertSpace Synchronization**:    Keeps basis consistent with HilbertSpace
    5. **Metadata Management**:             Tracks transformation history and symmetry info

    Subclasses (Hamiltonian, Lindbladian, etc.) inherit this infrastructure
    and only need to register specific transformation handlers.

    Attributes
    ----------
    _original_basis : HilbertBasisType
        The basis the operator was originally constructed in.
    _current_basis : HilbertBasisType
        The current basis representation.
    _is_transformed : bool
        Whether a non-trivial transformation has been applied.
    _basis_metadata : Dict[str, Any]
        Metadata about basis transformations and symmetries.
    _symmetry_info : str
        Human-readable symmetry information string.
    """

    # Class-level registry for basis transformations
    # Format: {(from_basis, to_basis): handler_function}
    # Each subclass maintains its own registry via class hierarchy
    _basis_transform_handlers: Dict[Tuple[str, str], Callable] = {}

    # -------------------------------------------------------------------------
    #! Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        *,
        ns: Optional[int] = None,
        name: str = "BasisAwareOperator",
        lattice: Optional["Lattice"] = None,
        hilbert_space: Optional["HilbertSpace"] = None,
        backend: str = "default",
        is_sparse: bool = True,
        is_manybody: bool = True,
        dtype: Optional[np.dtype] = None,
        logger: Optional["Logger"] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the BasisAwareOperator.

        Parameters
        ----------
        ns : Optional[int]
            Number of sites/modes.
        name : str
            Operator name.
        lattice : Lattice, optional
            Lattice structure.
        hilbert_space : HilbertSpace, optional
            Hilbert space (highest priority for configuration).
        backend : str
            Computational backend.
        is_sparse : bool
            Use sparse matrix representation.
        is_manybody : bool
            Many-body (True) or quadratic/single-particle (False).
        dtype : np.dtype, optional
            Data type for matrix elements.
        logger : Logger, optional
            Logger instance.
        seed : int, optional
            Random seed.
        **kwargs
            Additional arguments passed to SpecialOperator.
        """

        # Initialize basis tracking BEFORE parent init (so _handle_system can use it)
        self._original_basis: Optional["HilbertBasisType"] = None
        self._current_basis: Optional["HilbertBasisType"] = None
        self._is_transformed: bool = False
        self._basis_metadata: Dict[str, Any] = {}
        self._symmetry_info: Optional[str] = None
        self._symname: Optional[str] = None

        # Storage for transformed representations
        self._operator_transformed: Optional[Any] = None  # Transformed operator matrix
        self._transformed_grid: Optional[np.ndarray] = None  # Grid for k-space, etc.

        # Call parent init
        super().__init__(
            ns=ns,
            name=name,
            lattice=lattice,
            hilbert_space=hilbert_space,
            backend=backend,
            is_sparse=is_sparse,
            is_manybody=is_manybody,
            dtype=dtype,
            logger=logger,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

        # Infer and set default basis after system is configured
        self._infer_and_set_default_basis()

    # -------------------------------------------------------------------------
    #! Basis Transform Registry (Class Methods)
    # -------------------------------------------------------------------------

    @classmethod
    def register_basis_transform(cls, from_basis: str, to_basis: str, handler: Callable):
        """
        Register a basis transformation handler for this class.

        The handler will be called as: handler(self, enforce=False, **kwargs)
        and should modify self in-place to transform to the target basis.

        Parameters
        ----------
        from_basis : str
            Source basis identifier (e.g., "real", "k-space").
        to_basis : str
            Target basis identifier.
        handler : Callable
            Function that performs the transformation.
            Signature: handler(operator: BasisAwareOperator, enforce: bool = False, **kwargs) -> None

        Examples
        --------
        >>> @classmethod
        >>> def _transform_real_to_kspace(cls, enforce=False, **kwargs):
        ...     # Implementation
        ...     pass
        >>> Hamiltonian.register_basis_transform("real", "k-space", _transform_real_to_kspace)
        """
        # Ensure each class has its own registry (not shared with parent)
        if "_basis_transform_handlers" not in cls.__dict__:
            cls._basis_transform_handlers = {}

        key = (from_basis.lower(), to_basis.lower())
        cls._basis_transform_handlers[key] = handler

    def _get_basis_transform_handler(self, from_basis: str, to_basis: str) -> Optional[Callable]:
        """
        Get the registered transformation handler for this transformation.

        Checks the class hierarchy for registered handlers, starting from the most
        specific class and moving up to parent classes.

        Parameters
        ----------
        from_basis : str
            Source basis.
        to_basis : str
            Target basis.

        Returns
        -------
        Callable or None
            The handler function if registered, None otherwise.
        """
        key = (from_basis.lower(), to_basis.lower())

        # Check class hierarchy for registered handler
        for cls in self.__class__.__mro__:
            if hasattr(cls, "_basis_transform_handlers"):
                handlers = cls._basis_transform_handlers
                if key in handlers:
                    return handlers[key]

        return None

    @classmethod
    def list_registered_transforms(cls) -> List[Tuple[str, str]]:
        """
        List all registered basis transformations for this class and its parents.

        Returns
        -------
        List[Tuple[str, str]]
            List of (from_basis, to_basis) tuples.
        """
        transforms = set()
        for c in cls.__mro__:
            if hasattr(c, "_basis_transform_handlers"):
                transforms.update(c._basis_transform_handlers.keys())
        return list(transforms)

    # -------------------------------------------------------------------------
    #! Basis Inference
    # -------------------------------------------------------------------------

    def _infer_and_set_default_basis(self):
        """
        Infer and set the default basis based on system properties.

        Priority:
        1. Inherit from HilbertSpace if available
        2. Infer from system properties:
            - Quadratic with lattice:        REAL (position space)
            - Quadratic without lattice:     FOCK (single-particle occupation)
            - Many-body with lattice:        REAL (lattice sites)
            - Many-body without lattice:     COMPUTATIONAL (integer basis)
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        # Priority 1: Inherit from HilbertSpace if available
        if self._hilbert_space is not None and hasattr(self._hilbert_space, "_basis_type"):
            default_basis = self._hilbert_space._basis_type
            self._basis_metadata["inherited_from_hilbert"] = True
            self._log(f"Basis inherited from HilbertSpace: {default_basis}", lvl=2, color="cyan")
        else:
            # Priority 2: Infer from operator properties
            if self._is_quadratic:
                if self._lattice is not None:
                    default_basis = HilbertBasisType.REAL
                    self._basis_metadata["system_type"] = "quadratic-real"
                else:
                    default_basis = HilbertBasisType.FOCK
                    self._basis_metadata["system_type"] = "quadratic-fock"
            else:  # Many-body
                if self._lattice is not None:
                    default_basis = HilbertBasisType.REAL
                    self._basis_metadata["system_type"] = "manybody-real"
                else:
                    default_basis = HilbertBasisType.FOCK
                    self._basis_metadata["system_type"] = "manybody-fock"

            self._log(
                f"Basis inferred: {default_basis} ({self._basis_metadata.get('system_type', 'unknown')})",
                lvl=2,
                color="cyan",
            )

        # Set original and current basis
        self._original_basis = default_basis
        self._current_basis = default_basis
        self._is_transformed = False

    # -------------------------------------------------------------------------
    #! Basis Transformation
    # -------------------------------------------------------------------------

    def to_basis(self, basis_type: str, enforce: bool = False, **kwargs) -> "BasisAwareOperator":
        """
        Transform the operator to a different basis representation.

        This method:
        1. Validates the target basis
        2. Checks if already in target basis
        3. Dispatches to registered transformation handlers
        4. Updates basis tracking state
        5. Synchronizes with HilbertSpace (if available)

        Parameters
        ----------
        basis_type : str or HilbertBasisType
            Target basis. Examples: "real", "k-space", "fock", "sublattice", "symmetry".
        enforce : bool, optional
            If True, attempt transformation even if validation warns against it.
        **kwargs
            Additional arguments for specific transformations.

        Returns
        -------
        BasisAwareOperator
            Self (modified in-place).

        Raises
        ------
        NotImplementedError
            If no handler is registered for this transformation.
        ValueError
            If basis_type is invalid.
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

        current_basis = self._current_basis

        # Already in target basis?
        if target_basis == current_basis:
            self._log(f"Already in {target_basis} basis, skipping transformation.", lvl=2)
            return self

        # Validate transformation
        is_valid, reason = self.validate_basis_transformation(str(target_basis))
        if not is_valid and not enforce:
            raise ValueError(f"Basis transformation invalid: {reason}")

        # Find handler
        from_str = str(current_basis).lower()
        to_str = str(target_basis).lower()
        handler = self._get_basis_transform_handler(from_str, to_str)

        if handler is None:
            available = self.list_registered_transforms()
            raise NotImplementedError(
                f"No transformation registered: {from_str} -> {to_str}. " f"Available: {available}"
            )

        # Execute transformation
        self._log(f"Transforming: {current_basis} -> {target_basis}", lvl=1, color="cyan")
        handler(self, enforce=enforce, **kwargs)

        # Update state
        self._current_basis = target_basis
        self._is_transformed = target_basis != self._original_basis

        # Sync with HilbertSpace
        self.sync_basis_with_hilbert_space()

        return self

    def validate_basis_transformation(self, target_basis: str) -> Tuple[bool, str]:
        """
        Validate whether a basis transformation is feasible.

        Parameters
        ----------
        target_basis : str
            The target basis.

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
                return False, "Cannot transform to k-space: no lattice information"
            if hasattr(self._lattice, "bc") and not self._lattice.is_periodic:
                return False, "Cannot transform to k-space: lattice is not periodic"

        if self._current_basis == target:
            return False, f"Already in {target_basis} basis"

        return True, "Transformation is valid"

    # -------------------------------------------------------------------------
    #! Basis Properties and Getters
    # -------------------------------------------------------------------------

    @property
    def basis(self) -> Optional["HilbertBasisType"]:
        """Current basis representation."""
        return self._current_basis

    @property
    def original_basis(self) -> Optional["HilbertBasisType"]:
        """Original basis the operator was constructed in."""
        return self._original_basis

    @property
    def is_transformed(self) -> bool:
        """Whether the operator is in a transformed representation."""
        return self._is_transformed

    def get_basis_type(self) -> str:
        """Get current basis as string."""
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        return str(getattr(self, "_current_basis", HilbertBasisType.REAL))

    def set_basis_type(self, basis_type: str):
        """
        Set basis type metadata (does not perform transformation).

        Use to_basis() for actual transformation.
        """
        from QES.Algebra.Hilbert.hilbert_local import HilbertBasisType

        if isinstance(basis_type, str):
            self._current_basis = HilbertBasisType.from_string(basis_type)
        else:
            self._current_basis = basis_type

        self._log(f"Basis type set to {self._current_basis}", lvl=2, color="cyan")

    # -------------------------------------------------------------------------
    #! Transformation State
    # -------------------------------------------------------------------------

    def get_transformation_state(self) -> Dict[str, Any]:
        """
        Query the current transformation state.

        Returns
        -------
        Dict[str, Any]
            Dictionary with transformation state information.
        """
        return {
            "original_basis": str(self._original_basis) if self._original_basis else None,
            "current_basis": str(self._current_basis) if self._current_basis else None,
            "is_transformed": self._is_transformed,
            "has_transformed_repr": self._operator_transformed is not None,
            "transformed_shape": (
                self._operator_transformed.shape if self._operator_transformed is not None else None
            ),
            "grid_shape": (
                self._transformed_grid.shape if self._transformed_grid is not None else None
            ),
            "symmetry_info": self._symmetry_info,
            "metadata": self._basis_metadata.copy(),
        }

    def print_transformation_state(self):
        """Print human-readable transformation state summary."""
        state = self.get_transformation_state()
        self._log("Transformation State:", lvl=1, color="red")
        self._log(f"Original basis: {state['original_basis']}", lvl=2)
        self._log(f"Current basis: {state['current_basis']}", lvl=2)
        self._log(f"Is transformed: {state['is_transformed']}", lvl=2)
        self._log(f"Has transformed repr: {state['has_transformed_repr']}", lvl=2)
        if state["transformed_shape"]:
            self._log(f"Transformed shape: {state['transformed_shape']}", lvl=2)
        if state["grid_shape"]:
            self._log(f"Grid shape: {state['grid_shape']}", lvl=2)
        if state["symmetry_info"]:
            self._log(f"Symmetry: {state['symmetry_info']}", lvl=2)

    # -------------------------------------------------------------------------
    #! Symmetry Recording
    # -------------------------------------------------------------------------

    def record_symmetry_application(
        self, symmetry_name: Optional[str] = None, sector: Optional[str] = None
    ):
        """
        Record information about applied symmetries.

        If symmetry_name is not provided, attempts to extract symmetry info from HilbertSpace.

        Parameters
        ----------
        symmetry_name : str, optional
            Name of symmetry (e.g., "Z2", "U1", "SU2"). If None, fetched from HilbertSpace.
        sector : str, optional
            Which sector (e.g., "even", "odd").
        """
        # Try to get symmetry info from HilbertSpace if not provided
        if symmetry_name is None:
            symmetry_name, sector = self._get_symmetry_from_hilbert()

        if symmetry_name is None:
            self._log("No symmetry info available to record.", lvl=2, color="yellow")
            return

        sector_str = f" [{sector}]" if sector else ""
        self._symmetry_info = f"{symmetry_name}{sector_str}"
        self._basis_metadata["symmetry_applied"] = True
        self._basis_metadata["symmetries"] = self._basis_metadata.get("symmetries", []) + [
            symmetry_name
        ]

    def _get_symmetry_from_hilbert(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract symmetry information from HilbertSpace.

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            (symmetry_name, sector) or (None, None) if not available.
        """
        if self._hilbert_space is None:
            return None, None

        # Try symmetry_directory_name method (returns formatted string like "k0_p1")
        if hasattr(self._hilbert_space, "symmetry_directory_name"):
            try:
                sym_dir = self._hilbert_space.symmetry_directory_name
                if sym_dir:
                    return sym_dir, None
            except Exception:
                pass

        # Try _sym_container for detailed info
        if (
            hasattr(self._hilbert_space, "_sym_container")
            and self._hilbert_space._sym_container is not None
        ):
            sym_container = self._hilbert_space._sym_container

            # Get generator names if available
            if hasattr(sym_container, "generators"):
                gens = sym_container.generators
                if gens:
                    sym_names = [str(g) for g in gens]
                    return ", ".join(sym_names), None

        # Try _sym_basic_info
        if hasattr(self._hilbert_space, "_sym_basic_info"):
            info = self._hilbert_space._sym_basic_info
            if hasattr(info, "num_gens") and info.num_gens > 0:
                return f"Symmetry({info.num_gens} generators)", None

        return None, None

    def _sync_symmetry_from_hilbert(self):
        """
        Synchronize symmetry information from HilbertSpace.

        Called during initialization or when HilbertSpace changes.
        """
        if self._hilbert_space is None:
            return

        # Check if HilbertSpace has symmetries applied
        has_symmetries = False

        if (
            hasattr(self._hilbert_space, "_sym_container")
            and self._hilbert_space._sym_container is not None
        ):
            has_symmetries = True
        elif hasattr(self._hilbert_space, "_sym_basic_info"):
            info = self._hilbert_space._sym_basic_info
            if hasattr(info, "num_gens") and info.num_gens > 0:
                has_symmetries = True

        if has_symmetries:
            self.record_symmetry_application()  # Will fetch from HilbertSpace

    @property
    def sym(self):
        return self._symmetry_info

    # -------------------------------------------------------------------------
    #! HilbertSpace Synchronization
    # -------------------------------------------------------------------------

    def sync_basis_with_hilbert_space(self):
        """
        Synchronize this operator's basis with its HilbertSpace.

        If HilbertSpace has a different basis, update this operator to match.
        """
        if self._hilbert_space is None:
            return

        if not hasattr(self._hilbert_space, "_basis_type"):
            return

        hilbert_basis = self._hilbert_space._basis_type

        if self._current_basis != hilbert_basis:
            self._log(
                f"Syncing basis: {self._current_basis} -> HilbertSpace {hilbert_basis}",
                lvl=2,
                color="yellow",
            )
            self._current_basis = hilbert_basis

    def push_basis_to_hilbert_space(self):
        """
        Push this operator's basis to its HilbertSpace.

        Use when the operator's basis is the source of truth.
        """
        if self._hilbert_space is None:
            return

        if hasattr(self._hilbert_space, "set_basis"):
            self._hilbert_space.set_basis(str(self._current_basis))
            self._log(f"Basis pushed to HilbertSpace: {self._current_basis}", lvl=2, color="cyan")

    # -------------------------------------------------------------------------
    #! Transformed Representation Properties
    # -------------------------------------------------------------------------

    @property
    def operator_transformed(self):
        """Get the transformed operator representation."""
        return self._operator_transformed

    @operator_transformed.setter
    def operator_transformed(self, value):
        """Set the transformed operator representation."""
        self._operator_transformed = value

    @property
    def grid_transformed(self):
        """Get the transformation grid (e.g., k-points for k-space)."""
        return self._transformed_grid

    @grid_transformed.setter
    def grid_transformed(self, value):
        """Set the transformation grid."""
        self._transformed_grid = value


# ---------------------------------------------------------------------------
#! End of file
# ---------------------------------------------------------------------------
