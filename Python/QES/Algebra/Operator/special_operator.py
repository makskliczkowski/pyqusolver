"""
SpecialOperator - Base class for composite operators with custom composition support.

This module provides the SpecialOperator class, which extends the base Operator
class to support:
- Custom operator registration (for n-site correlators, non-standard operators)
- Instruction code management for JIT-compiled composition functions
- Hybrid composition functions that handle both predefined and custom operators

Classes like Hamiltonian inherit from SpecialOperator to gain these capabilities.

------------------------------------------------------------------------
File        : Algebra/Operator/special_operator.py
Author      : Maksymilian Kliczkowski
Date        : December 2025
------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numba
import numpy as np

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# ---------------------------------------------------------------------------
# QES imports
# ---------------------------------------------------------------------------

try:
    from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
    from QES.Algebra.Operator.operator import (
        Operator,
        OperatorFunction,
        OperatorTypeActing,
        SymmetryGenerators,
    )

except ImportError as e:
    raise ImportError(
        "QES modules are required for SpecialOperator. Ensure QES is installed."
    ) from e

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Operator.operator_loader import OperatorModule
    from QES.general_python.common.flog import Logger
    from QES.general_python.lattices.lattice import Lattice

# ---------------------------------------------------------------------------
#! Constants for custom operator codes
# ---------------------------------------------------------------------------

CUSTOM_OP_BASE  = 1000  # Custom operators start from this code
CUSTOM_OP_MAX   = 9999  # Maximum custom operator code

def is_custom_code(code: int) -> bool:
    """Check if an instruction code represents a custom operator."""
    return CUSTOM_OP_BASE <= code <= CUSTOM_OP_MAX

# ---------------------------------------------------------------------------
#! SpecialOperator Base Class
# ---------------------------------------------------------------------------

class SpecialOperator(Operator, ABC):
    """
    Base class for composite operators with instruction-based composition.

    SpecialOperator extends Operator to support:

    1. **Instruction Code System**: Maps operators to integer codes for
        efficient JIT-compiled matrix construction.

    2. **Custom Operator Registry**: Allows registration of arbitrary
        operators (e.g., 3-site correlators) that aren't predefined.

    3. **Hybrid Composition Functions**: Creates JIT-compiled functions
        that handle both predefined and custom operators efficiently.

    This class is meant to be inherited by:
    - Hamiltonian (sum of operators with coefficients)
    - Lindbladian (for open quantum systems)
    - Custom composite operators

    Attributes
    ----------
    _lookup_codes : Dict[str, int]
        Maps operator names to instruction codes.
    _instr_codes : List[int]
        List of instruction codes for each term.
    _instr_coeffs : List[complex]
        List of coefficients for each term.
    _instr_sites : List[List[int]]
        List of site indices for each term.
    _instr_max_arity : int
        Maximum number of sites any single term acts on.
    _instr_function : Callable
        JIT-compiled composition function.
    _custom_op_registry : Dict[int, Callable]
        Maps custom codes to operator functions.
    _has_custom_ops : bool
        Whether custom operators are present.
    """

    # Error messages
    _ERR_CUSTOM_OP_NO_INT           = "Custom operator '{}' must have an 'int' function (numba-compatible) to be used in instruction-based composition."
    _ERR_COUP_VEC_SIZE              = "Invalid coupling vector size. Coupling must be a scalar, a string, or a list/array of length ns."
    _ERR_HILBERT_SPACE_NOT_PROVIDED = (
        "Hilbert space is not provided or is invalid. Please supply a valid HilbertSpace object."
    )
    _ERR_NS_NOT_PROVIDED            = (
        "'ns' (number of sites/modes) must be provided, e.g., via 'ns' kwarg or a Lattice object."
    )
    _ERR_NEED_LATTICE               = "Lattice information is required but not provided. Please specify a lattice or number of sites."
    _ERR_MODE_MISMATCH              = "Operation not supported for the current Hamiltonian mode (Many-Body/Quadratic). Check 'is_manybody' flag."

    _ERRORS                         = {
                                        "custom_op_no_int"              : _ERR_CUSTOM_OP_NO_INT,
                                        "coupling_vector_size"          : _ERR_COUP_VEC_SIZE,
                                        "hilbert_space_not_provided"    : _ERR_HILBERT_SPACE_NOT_PROVIDED,
                                        "ns_not_provided"               : _ERR_NS_NOT_PROVIDED,
                                        "need_lattice"                  : _ERR_NEED_LATTICE,
                                        "mode_mismatch"                 : _ERR_MODE_MISMATCH,
                                    }

    ################################################################################################
    #! Initialization
    ################################################################################################

    def _handle_system(
        self,
        ns: Optional[int],
        hilbert_space: Optional[HilbertSpace],
        lattice: Optional["Lattice"],
        logger: Optional["Logger"],
        **kwargs,
    ):
        """
        Handle the system configuration with clear priority:
        1. HilbertSpace (highest priority) - contains everything
        2. Lattice - contains ns and structure
        3. ns (lowest priority) - just the number of sites

        If HilbertSpace is not provided, create one from lattice/ns.
        The local_space type is inferred from _physics_type if set.
        """

        try:
            from QES.Algebra.hilbert import HilbertSpace
        except ImportError as e:
            raise ImportError(
                "QES HilbertSpace module is required for SpecialOperator. Ensure QES is installed."
            ) from e

        # Priority 1: HilbertSpace takes precedence over everything
        if hilbert_space is not None:
            self._hilbert_space = hilbert_space
            self._ns = hilbert_space.ns
            self._lattice = hilbert_space.lattice if hilbert_space.lattice is not None else lattice

            # Infer dtype from HilbertSpace if not explicitly set
            if self._dtype is None:
                self._dtype = hilbert_space.dtype

            # Infer local space type from HilbertSpace for physics_type
            if (
                self._physics_type is None
                and hasattr(hilbert_space, "_local_space")
                and hilbert_space._local_space is not None
            ):
                self._physics_type = hilbert_space._local_space.typ

            # Check mode consistency
            if self._is_manybody and not hilbert_space._is_many_body:
                raise ValueError(
                    SpecialOperator._ERR_MODE_MISMATCH
                    + " Hamiltonian is many-body but HilbertSpace is quadratic."
                )
            if self._is_quadratic and not hilbert_space._is_quadratic:
                raise ValueError(
                    SpecialOperator._ERR_MODE_MISMATCH
                    + " Hamiltonian is quadratic but HilbertSpace is many-body."
                )

            if self._logger:
                if self._verbose:
                    self._log(
                        f"Using provided HilbertSpace: ns={self._ns}, lattice={'yes' if self._lattice else 'no'}",
                        lvl=1,
                        color="green",
                    )
            return

        # Priority 2: Lattice - infer ns from lattice
        if lattice is not None:
            self._lattice = lattice
            self._ns = lattice.ns
            if self._logger:
                if self._verbose:
                    self._log(
                        f"Inferred ns={self._ns} from provided lattice.", lvl=3, color="green"
                    )

        # Priority 3: ns directly provided
        elif ns is not None:
            self._ns = ns
            self._lattice = None
            if self._logger:
                if self._verbose:
                    self._log(f"Using provided ns={self._ns} (no lattice).", lvl=3, color="green")
        else:
            # Nothing provided - error
            raise ValueError(self._ERRORS["ns_not_provided"])

        # HilbertSpace not provided - create one
        # Determine local_space from physics_type
        local_space_arg = None
        if self._physics_type is not None:
            try:
                from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
            except ImportError as e:
                raise ImportError(
                    "QES HilbertLocal module is required for SpecialOperator. Ensure QES is installed."
                ) from e

            # Map physics_type to local_space string
            physics_to_local = {
                LocalSpaceTypes.SPIN_HALF: "spin-1/2",
                LocalSpaceTypes.SPIN_ONE: "spin-1",
                LocalSpaceTypes.FERMION: "fermion",
                LocalSpaceTypes.HARDCORE_BOSON: "hardcore-boson",
                LocalSpaceTypes.BOSON: "boson",
                "spin-1/2": "spin-1/2",
                "spin-1": "spin-1",
                "fermion": "fermion",
                "hardcore-boson": "hardcore-boson",
                "boson": "boson",
            }
            local_space_arg = physics_to_local.get(self._physics_type, None)

        self._hilbert_space = HilbertSpace(
            ns=self._ns,
            lattice=self._lattice,
            is_manybody=self._is_manybody,
            local_space=local_space_arg,
            dtype=self._dtype,
            backend=self._backendstr,
            logger=logger,
            **kwargs,
        )

        if self._logger:
            self._log(
                f"Created HilbertSpace: ns={self._ns}, local_space={local_space_arg or 'default'}",
                lvl=3,
                color="green",
                log="debug",
            )

    def _handle_dtype(self, dtype: Optional[Union[str, np.dtype]]):
        """
        Handle the dtype of the Hamiltonian. Overrides the method
        from 'Matrix' class to infer dtype from Hilbert space when possible.

        Parameters:
        -----------
            dtype (str or np.dtype, optional):
                The dtype to use for the Hamiltonian.
        """

        self._iscpx = False

        if dtype is not None:
            super()._handle_dtype(dtype)
        else:
            if self._hilbert_space is not None:
                try:
                    hs_dtype = getattr(self._hilbert_space, "dtype", None)
                    if getattr(self._hilbert_space, "has_complex_symmetries", False):
                        self._dtype = np.complex128
                        self._iscpx = True
                    else:
                        # fall back to the Hilbert's dtype when present
                        self._dtype = hs_dtype if hs_dtype is not None else np.float64
                        try:
                            if np.issubdtype(np.dtype(self._dtype), np.complexfloating):
                                self._dtype = np.complex128
                                self._iscpx = True
                        except Exception:
                            pass
                except Exception:
                    self._dtype = np.float64
            else:
                self._dtype = np.float64

        if getattr(self, '_iscpx', False):
            if self._verbose:
                self._log("I am complex!", lvl = 2, color='red')
        else:
            if self._verbose:
                self._log("I am real!", lvl=2, color="green")

    # -------------------------------------------------------------------------

    def __init__(
        self,
        *,
        ns: Optional[int] = None,
        name: str = "SpecialOperator",
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
        Initialize the SpecialOperator.

        Parameters
        ----------
        ns : int
            Number of sites/modes in the system.
        name : str
            Name of the operator.
        lattice : Lattice, optional
            Lattice object for the system.
        backend : str
            Computational backend ('default', 'np', 'jax').
        is_sparse : bool
            Whether to use sparse matrix representation.
        dtype : np.dtype, optional
            Data type for matrix elements.
        logger : Logger, optional
            Logger for debug/info messages.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs
            Additional arguments passed to Operator.
        """

        # Pre-compute backend for _handle_system (before calling Operator.__init__)
        self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k) = (
            Operator._set_backend(backend, seed)
        )
        self._is_jax = JAX_AVAILABLE and self._backend != np
        self._is_numpy = not self._is_jax

        self._dtype = dtype
        self._hilbert_space = None  # Will be set by _handle_system
        self._logger = (
            hilbert_space.logger if (logger is None and hilbert_space is not None) else logger
        )
        self._verbose = verbose
        self._iterator = 0  # For tracking iterations in matvec

        #! general Hamiltonian info
        self._name = "Hamiltonian" if name is None else name
        self._is_manybody = is_manybody
        self._is_quadratic = not is_manybody
        self._particle_conserving = False

        # Physics type for local space inference (can be set by subclasses before __init__)
        # Default to spin-1/2 for many-body systems
        if not hasattr(self, "_physics_type") or self._physics_type is None:
            self._physics_type = kwargs.get("physics_type", None)

        # Handle system (ns, hilbert_space, lattice) before Operator init
        self._handle_system(ns, hilbert_space, lattice, logger, **kwargs)
        self._handle_dtype(dtype)

        # Update ns/lattice for Operator init (they might have been set in _handle_system)
        ns      = self._ns
        lattice = self._lattice

        # Initialize base Operator
        super().__init__(
            ns=ns,
            name=name,
            lattice=lattice,
            backend=backend,
            is_sparse=is_sparse,
            dtype=dtype,
            logger=logger,
            seed=seed,
            **kwargs,
        )

        # Override lattice from Hilbert space (takes precedence)
        self._nh = self._hilbert_space.nh
        if self._lattice is not None:
            self._local_nei = self._lattice.cardinality
        else:
            self._local_nei = ns  # assume fully connected if no lattice provided
        self._startns = 0  # for starting hamil calculation (potential loop over sites)

        # Instruction code system
        self._lookup_codes: Dict[str, int] = (
            {}
        )  # Name to code mapping -> maps operator names to instruction codes
        self._instr_codes: List[int] = (
            []
        )  # Instruction codes for each term. Defined by each module (see OperatorModule)
        self._instr_coeffs: List[complex] = []  # Coefficients for each term
        self._instr_sites: List[List[int]] = []  # Site indices for each term
        self._instr_max_arity: int = 2  # Default: 2-site correlations

        # Composition function (set by subclass or setup method)
        self._instr_function: Optional[Callable] = None
        self._instr_max_out: int = self._ns + 1  # Max output size for composition function

        # Custom operator registry
        self._custom_op_registry: Dict[int, Callable] = {}  # Custom operator functions by code
        self._custom_op_counter: int = CUSTOM_OP_BASE
        self._custom_op_arity: Dict[int, int] = {}  # Arity of each custom operator
        self._has_custom_ops: bool = False  # Whether custom operators are present, for optimization

        # Operator term lists (organized by site and type)
        # These are used by subclasses like Hamiltonian to build composition functions
        self._ops_nmod_nosites: List[List[Tuple]] = [
            [] for _ in range(self._ns)
        ]  # Non-modifying, global operators
        self._ops_nmod_sites: List[List[Tuple]] = [
            [] for _ in range(self._ns)
        ]  # Non-modifying, local operators
        self._ops_mod_nosites: List[List[Tuple]] = [
            [] for _ in range(self._ns)
        ]  # Modifying, global operators
        self._ops_mod_sites: List[List[Tuple]] = [
            [] for _ in range(self._ns)
        ]  # Modifying, local operators

        # Compiled composition functions for different backends
        self._composition_int_fun: Optional[Callable] = (
            None  # JIT-compiled integer state composition
        )
        self._composition_np_fun: Optional[Callable] = None  # NumPy composition function
        self._composition_jax_fun: Optional[Callable] = None  # JAX composition function

        # Buffers for instruction function to avoid reallocation
        self._instr_buffers: Any = None

        # Cache for matvec function
        self._cached_matvec_fun: Optional[Callable] = None

    # -------------------------------------------------------------------------

    def _log(
        self, msg: str, log: str = "info", lvl: int = 0, color: str = "white", verbose: bool = True
    ) -> None:
        """
        Log the message.

        Args:
            msg (str) : The message to log.
            log (str) : The logging level. Default is 'info'.
            lvl (int) : The level of the message.
        """
        if not verbose:
            return

        msg = f"[{self.name}] {msg}"
        self._hilbert_space._log(msg, log=log, lvl=lvl, color=color, append_msg=False)

    # -------------------------------------------------------------------------
    #! Properties
    # -------------------------------------------------------------------------

    @property
    def is_complex(self) -> bool:
        """Whether the operator is complex-valued."""
        return getattr(self, "_iscpx", False)
    
    @property
    def iscpx(self) -> bool:
        """Whether the operator is complex-valued."""
        return getattr(self, "_iscpx", False)

    @property
    def has_custom_operators(self) -> bool:
        """Whether custom operators have been registered."""
        return self._has_custom_ops

    @property
    def n_instructions(self) -> int:
        """Number of instruction terms."""
        return len(self._instr_codes)

    @property
    def max_arity(self) -> int:
        """Maximum number of sites any term acts on."""
        return self._instr_max_arity

    @property
    def lookup_codes(self) -> Dict[str, int]:
        """Dictionary mapping operator names to instruction codes."""
        return self._lookup_codes.copy()

    @property
    def quadratic(self):
        return self._is_quadratic

    def is_quadratic(self):
        return self._is_quadratic

    @property
    def manybody(self):
        return self._is_manybody

    def is_manybody(self):
        return self._is_manybody

    @property
    def max_local_changes(self):
        return self._max_local_ch if self._is_manybody else 2

    @property
    def max_local(self):
        return self.max_local_changes

    @property
    def max_operator_changes(self):
        return self._max_local_ch_o if self._is_manybody else 2

    @property
    def max_operator(self):
        return self.max_operator_changes

    @property
    def cardinality(self):
        return self._local_nei

    # modes and hilbert space properties

    @property
    def modes(self):
        return (
            self._hilbert_space.local_space.local_dim if self._hilbert_space is not None else None
        )

    @property
    def hilbert_space(self):
        return self._hilbert_space

    @property
    def hilbert(self):
        return self._hilbert_space

    @property
    def hilbert_size(self):
        return self._hilbert_space.nh

    @property
    def dim(self):
        return self.hilbert_size

    @property
    def nh(self):
        return self.hilbert_size

    @property
    def matvec_fun(self) -> Optional[Callable]:
        """
        Returns an optimized matrix-vector multiplication function for iterative solvers.
        Pre-calculates all necessary contexts and uses sequential kernels to avoid
        overhead from multithreading and buffer allocations.
        """
        if self._cached_matvec_fun is not None:
            return self._cached_matvec_fun


        hilbert_in = self._hilbert_space
        if hilbert_in is None:
            return None

        nh = hilbert_in.nh
        nhfull = hilbert_in.nhfull
        has_sym = nh != nhfull

        op_func = self._fun._fun_int
        if op_func is None:
            return None

        from QES.Algebra.Hilbert.matrix_builder import canonicalize_args

        op_args = canonicalize_args(())

        # Determine Path
        compact_data = getattr(hilbert_in, "compact_symmetry_data", None)
        use_fast = has_sym and (compact_data is not None)
        self._iterator = 0

        matvec_func = None

        if not has_sym:
            # Path 1: No symmetry (Full Hilbert space)
            from QES.Algebra.Hilbert.matrix_builder import _apply_op_batch_seq_jit

            def _matvec_seq(x):
                is_1d = x.ndim == 1
                v_in = x[:, np.newaxis] if is_1d else x
                # We allocate v_out here as it must be returned to the solver
                v_out = np.zeros(v_in.shape, dtype=x.dtype)
                self._iterator += 1
                _apply_op_batch_seq_jit(v_in, v_out, op_func, op_args)
                # log_memory_status("After _apply_op_batch_seq_jit", logger=self._logger)
                # print(f"Matvec iteration {self._iterator} completed.")
                return v_out.ravel() if is_1d else v_out

            matvec_func = _matvec_seq

        elif use_fast:
            # Path 2: Fast path (CompactSymmetryData lookups)
            from QES.Algebra.Hilbert.matrix_builder import _apply_op_batch_compact_seq_jit

            cd = compact_data
            basis_args = (
                cd.representative_list,
                cd.normalization,
                cd.repr_map,
                cd.phase_idx,
                cd.phase_table,
            )

            def _matvec_compact_seq(x):
                is_1d = x.ndim == 1
                v_in = x[:, np.newaxis] if is_1d else x
                v_out = np.zeros(v_in.shape, dtype=x.dtype)
                _apply_op_batch_compact_seq_jit(v_in, v_out, op_func, op_args, basis_args)
                self._iterator += 1
                # log_memory_status("After _apply_op_batch_compact_seq_jit", logger=self._logger)
                return v_out.ravel() if is_1d else v_out

            matvec_func = _matvec_compact_seq

        else:
            # Path 3: Fallback (Projected compact basis)
            from QES.Algebra.Symmetries.jit.matrix_builder_jit import (
                _apply_op_batch_projected_compact_seq_jit,
            )

            sc = hilbert_in.sym_container
            cg = sc.compiled_group
            tb = sc.tables
            ns_val = np.int64(hilbert_in.ns)

            basis_in_args = (compact_data.representative_list, compact_data.normalization)
            basis_out_args = (
                compact_data.repr_map,
                compact_data.normalization,
                compact_data.representative_list,
            )
            cg_args = (cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, cg.chi)
            tb_args = tb.args

            @numba.njit
            def _matvec_projected_seq(x):
                is_1d = x.ndim == 1
                v_in = x[:, np.newaxis] if is_1d else x
                v_out = np.zeros(v_in.shape, dtype=x.dtype)
                _apply_op_batch_projected_compact_seq_jit(
                    v_in,
                    v_out,
                    op_func,
                    op_args,
                    basis_in_args,
                    basis_out_args,
                    cg_args,
                    tb_args,
                    ns_val,
                )
                # log_memory_status("After _apply_op_batch_projected_compact_seq_jit", logger=self._logger)
                return v_out.ravel() if is_1d else v_out

            matvec_func = _matvec_projected_seq

        self._cached_matvec_fun = matvec_func
        return matvec_func

    # ----------------------------------------------------------------------------------------------
    #! ACCESS TO OTHER MODULES RELATED TO HAMILTONIAN
    # ----------------------------------------------------------------------------------------------

    @property
    def operators(self) -> "OperatorModule":
        """
        Lazy-loaded operator module for convenient operator access.

        Returns
        -------
        OperatorModule
            Module providing operator factory functions based on the Hilbert space's local space type.

        Examples
        --------
        >>> # Via Hamiltonian (inherits from HilbertSpace)
        >>> model           = KitaevModel(lattice, ...)
        >>> ops             = model.operators
        >>> sig_x           = ops.sig_x(ns=model.ns, sites=[0, 1])
        >>> sig_x_matrix    = sig_x.matrix

        >>> # For fermion Hamiltonians
        >>> hamil           = FermionHamiltonian(ns=4, ...)
        >>> cdag            = hamil.operators.cdag(ns=4, sites=[0])
        >>> n_op            = hamil.operators.n(ns=4, sites=[0])

        >>> # Get help
        >>> hamil.operators.help()
        """
        if not hasattr(self, "_operator_module") or self._operator_module is None:

            try:
                from QES.Algebra.Operator.operator_loader import get_operator_module
            except ImportError as e:
                raise ImportError(
                    "Operator module could not be loaded. Ensure QES is properly installed."
                ) from e

            # Get local space type from Hilbert space if available
            if self._hilbert_space is not None and hasattr(self._hilbert_space, "_local_space"):
                local_space_type = self._hilbert_space._local_space.typ
            else:
                # Default to spin-1/2 for many-body, None for quadratic
                from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes

                local_space_type = LocalSpaceTypes.SPIN_HALF if self._is_manybody else None
            self._operator_module = get_operator_module(local_space_type)
        return self._operator_module

    def correlators(
        self,
        *,
        correlators: List[str] = None,
        compute: bool = True,
        # operator path
        indices_pairs: List[Tuple[int, int]] = None,
        type_acting="global",
        # compute path
        nstates_to_store=None,
        n_susceptibility_states: int = 10,
        safety_factor=0.6,
        **kwargs,
    ) -> dict:
        """
        Computes correlators using the operator module.

        Parameters
        ----------
        indices_pairs : List[Tuple[int, int]]
            List of index pairs for which to compute the correlators.

        correlators : optional
            List of correlator types to compute, e.g.,:
            - SPIN      : ['xx', 'yy', 'zz']),
            - FERMION   : ['cdagc', 'ccdag']),
        **kwargs
            Additional keyword arguments to pass to the operator module's correlators method.

        Returns
        -------
        dict
            Dictionary containing the computed correlators.

        Raises
        ------
        AttributeError
            If the operator module does not have a 'correlators' attribute.

        Example
        ------
        >>> hamil       = Spin1/2Model(...)
        >>> hamil.correlators(indices_pairs=[(0,2), (1,3)], correlators=['xx', 'yy'])
        { 'xx': ..., 'yy': { (i,j) : OP for i,j in ... } }
        """

        op_module = self.operators  # Ensure operator module is loaded

        if not compute:
            if hasattr(op_module, "correlators"):
                return op_module.correlators(
                    indices_pairs=indices_pairs,
                    correlators=correlators,
                    type_acting=type_acting,
                    **kwargs,
                )
            else:
                raise AttributeError("Operator module does not have a 'correlators' attribute.")
        else:
            if hasattr(op_module, "compute_correlations"):
                return op_module.compute_correlations(
                    eigenvalues=self.eigenvalues,
                    eigenvectors=self.eigenvectors,
                    correlators=correlators,
                    hilbert=self.hilbert_space,
                    lattice=self._lattice,
                    logger=self._logger,
                    nstates_to_store=nstates_to_store,
                    n_susceptibility_states=n_susceptibility_states,
                    safety_factor=safety_factor,
                )
            else:
                raise AttributeError(
                    "Operator module does not have a 'compute_correlations' attribute."
                )

    @property
    def entanglement(self):
        r"""
        Lazy-loaded entanglement module for comprehensive entanglement entropy calculations.

        Returns
        -------
        EntanglementModule
            Module providing entanglement entropy calculations for arbitrary bipartitions.
            Supports both correlation matrix (fast, for quadratic Hamiltonians) and
            many-body (exact, for any state) methods.

        Features
        --------
        **Single-Particle Methods (Quadratic Hamiltonians)**:
            - Fast O(L³) entropy from correlation matrix C_ij = <c_i† c_j>
            - Supports both NumPy and JAX backends (GPU acceleration)
            - Batch calculation for multiple bipartitions
            - Works for ANY bipartition (contiguous or non-contiguous)

        **Many-Body Methods (Any State)**:
            - Exact Schmidt decomposition for arbitrary states
            - Works for contiguous AND non-contiguous bipartitions
            - Required for interacting systems

        **Topological Entanglement Entropy**:
            - Kitaev-Preskill construction: \gamma = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
            - Levin-Wen construction for disk geometry
            - Extracts universal topological term

        **Wick's Theorem Verification**:
            - Check if a state is a valid Slater determinant
            - Verify Wick contractions: <c†c†cc> = <c†c><c†c> - <c†c><c†c>

        **Mask Generation Utilities** (via MaskGenerator class):
            - Contiguous, alternating, random subsystems
            - Sublattice masks for bipartite lattices
            - Kitaev-Preskill regions for TEE
            - Bitmask conversions

        Examples
        --------
        Basic entropy calculation:
            >>> hamil = FreeFermions(ns=12, t=1.0)
            >>> hamil.diagonalize()
            >>> ent = hamil.entanglement
            >>>
            >>> # Contiguous bipartition (correlation method is exact)
            >>> bipart = ent.bipartition([0, 1, 2, 3, 4])
            >>> S = ent.entropy_correlation(bipart, orbitals=[0,1,2,3,4,5])

        Non-contiguous bipartition (works correctly!):
            >>> bipart_nc = ent.bipartition([0, 2, 4, 6, 8])  # Even sites
            >>> state = hamil.many_body_state([0,1,2,3,4,5])
            >>> S_corr = ent.entropy_correlation(bipart_nc, orbitals)  # Fast
            >>> S_mb = ent.entropy_many_body(bipart_nc, state)  # Also works
            >>> # Both methods give same result for free fermions!

        Compare methods:
            >>> result = ent.compare_methods(bipart, orbitals)
            >>> print(f"Correlation: {result['correlation']:.4f}")
            >>> print(f"Many-body:   {result['many_body']:.4f}")

        Topological entanglement entropy:
            >>> result = ent.topological_entropy(orbitals, construction='kitaev_preskill')
            >>> gamma = result['gamma']  # TEE
            >>> print(f"γ = {gamma:.4f} (γ=log(2) for toric code)")

        Wick's theorem verification:
            >>> result = ent.verify_wicks_theorem(orbitals)
            >>> print(f"Is Slater determinant: {result['is_valid']}")
            >>> print(f"Max Wick error: {result['max_error']:.2e}")

        Mask generation:
            >>> from QES.general_python.physics.entanglement_module import MaskGenerator
            >>> mask = MaskGenerator.contiguous(ns=12, size_a=4)
            >>> even, odd = MaskGenerator.alternating(ns=12)
            >>> regions = MaskGenerator.kitaev_preskill(ns=12)

        Entropy scaling:
            >>> results = ent.entropy_scan([0,1,2,3,4])
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(results['sizes'], results['entropies'])
            >>> plt.xlabel('Subsystem size')
            >>> plt.ylabel('Entanglement entropy')

        Get detailed help:
            >>> hamil.entanglement.help()

        See Also
        --------
        - entropy_correlation : Fast method for quadratic Hamiltonians
        - entropy_many_body : Exact method for any state
        - topological_entropy : Kitaev-Preskill/Levin-Wen TEE
        - verify_wicks_theorem : Check Slater determinant property
        - MaskGenerator : Utility class for generating subsystem masks
        """
        if not hasattr(self, "_entanglement_module") or self._entanglement_module is None:
            from QES.general_python.physics.entanglement_module import get_entanglement_module

            self._entanglement_module = get_entanglement_module(self)
        return self._entanglement_module

    @property
    def statistical(self):
        r"""
        Lazy-loaded statistical properties module.

        Provides functions for calculating statistical properties of eigenstates
        and eigenvalues: LDOS, DOS, matrix element distributions, IPR, etc.

        Returns
        -------
        StatisticalModule
            Module with statistical analysis methods.

        Examples
        --------
        Density of States:
            >>> hamil.diagonalize()
            >>> dos = hamil.statistical.dos(nbins=100)

        Local DOS (strength function):
            >>> psi0 = hamil.eig_vec[:, 0]  # Ground state
            >>> ldos = hamil.statistical.ldos(overlaps=psi0)

        Inverse Participation Ratio:
            >>> ipr = hamil.statistical.ipr(hamil.eig_vec[:, 0])

        See Also
        --------
        QES.Algebra.Properties.statistical : Full module documentation
        """
        if not hasattr(self, "_statistical_module") or self._statistical_module is None:
            from QES.Algebra.Properties.statistical import get_statistical_module

            self._statistical_module = get_statistical_module(self)
        return self._statistical_module

    @property
    def time_evo(self):
        r"""
        Lazy-loaded time evolution module.

        Provides efficient methods for quantum quench dynamics and time evolution
        of states using the eigenbasis decomposition.

        Returns
        -------
        TimeEvolutionModule
            Module with time evolution methods.

        Features
        --------
        - Efficient time evolution via eigenbasis decomposition
        - Batch evolution for multiple time points (one BLAS call)
        - Expectation value calculation for observables
        - Quench state construction (AF, FM, domain wall, random, etc.)
        - Diagonal ensemble calculations
        - JAX support for GPU acceleration

        Examples
        --------
        Basic time evolution:
            >>> hamil.diagonalize()
            >>> psi0 = np.zeros(hamil.nh); psi0[0] = 1.0  # Initial state
            >>> psi_t = hamil.time_evo.evolve(psi0, t=1.0)

        Multiple times efficiently:
            >>> times = np.linspace(0, 10, 100)
            >>> psi_all = hamil.time_evo.evolve_batch(psi0, times)  # (dim, 100)

        Expectation value dynamics:
            >>> O = ...  # Observable operator
            >>> O_t = hamil.time_evo.expectation(psi0, O, times)

        Create quench initial states:
            >>> from QES.Algebra.Properties.time_evo import QuenchTypes
            >>> psi0 = hamil.time_evo.quench_state(QuenchTypes.NEEL)

        See Also
        --------
        QES.Algebra.Properties.time_evo : Full module documentation
        """
        if not hasattr(self, "_time_evo_module") or self._time_evo_module is None:
            from QES.Algebra.Properties.time_evo import get_time_evolution_module

            self._time_evo_module = get_time_evolution_module(self)
        return self._time_evo_module

    @property
    def spectral(self):
        r"""
        Lazy-loaded spectral function module.

        Provides Green's functions, spectral functions, and dynamical
        correlation functions using full ED or Lanczos methods.

        Returns
        -------
        SpectralModule
            Module with spectral function methods.

        Features
        --------
        - Green's function G(ω) from eigenbasis or Lanczos
        - Spectral function A(ω) = -Im[G(ω)]/π
        - Dynamic structure factor S(q,ω)
        - Dynamical susceptibility χ(ω)
        - Finite temperature support

        Examples
        --------
        Spectral function:
            >>> hamil.diagonalize()
            >>> omega = np.linspace(-5, 5, 200)
            >>> A = hamil.spectral.spectral_function(omega, operator, eta=0.1)

        Dynamic structure factor (Lanczos - no full diag needed):
            >>> S_omega = hamil.spectral.dynamic_structure_factor(
            ...     omega, S_q_operator, use_lanczos=True, max_krylov=100
            ... )

        Susceptibility:
            >>> chi = hamil.spectral.susceptibility(omega, Sx_op, Sx_op)

        See Also
        --------
        QES.general_python.physics.spectral.spectral_backend : Full module
        """
        if not hasattr(self, "_spectral_module") or self._spectral_module is None:
            from QES.general_python.physics.spectral.spectral_backend import get_spectral_module

            self._spectral_module = get_spectral_module(self)
        return self._spectral_module

    # -------------------------------------------------------------------------
    #! Coupling Vector Setup
    # -------------------------------------------------------------------------

    def _set_some_coupling(
        self, coupling: Union[list, np.ndarray, float, complex, int, str]
    ) -> "Array":
        """
        Distinghuishes between different initial values for the coupling and returns it.
        One distinguishes between:
            - a full vector of a correct size
            - single value
            - random string
        ---
        Parameters:
            - coupling : some coupling to be set
        ---
        Returns:
            array to be used latter with corresponding couplings
        """
        if isinstance(coupling, list) and len(coupling) == self.ns:
            return self._backend.array(coupling, dtype=self._dtype)
        elif isinstance(coupling, (float, int, complex)):
            from QES.Algebra.Operator.matrix import DummyVector

            return DummyVector(coupling, backend=self._backend).astype(self._dtype)
        elif isinstance(coupling, str):
            from QES.general_python.algebra.ran_wrapper import random_vector

            return random_vector(self.ns, coupling, backend=self._backend, dtype=self._dtype)
        else:
            raise ValueError(self._ERR_COUP_VEC_SIZE)

    # ----------------------------------------------------------------------------------------------
    #! Adding operators to the internal collections
    # ----------------------------------------------------------------------------------------------

    def add(
        self,
        operator: Operator,
        multiplier: Union[float, complex, int],
        modifies: bool = False,
        sites=None,
    ):
        """
        Add an operator to the internal operator collections based on its locality.

        ---
        Parameters:
            operator:
                The operator to be added. This can be any object representing an operation,
                typically in the context of a quantum system.
            sites (list[int]):
                A list of site indices where the operator should act. If empty,
                the operator will be associated with site 0.
            multiplier (numeric): A scaling factor to be applied to the operator.
            is_local (bool, optional):
                Determines the type of operator. If True, the operator is
                considered local (i.e., it does not modify the state) and is
                appended to the local operator list. If False, it is added to
                the non-local operator list. Defaults to False.
        ---
        Behavior:

            - Determines the primary site for the operator based on the first element in the 'sites' list, or defaults to index 0 if 'sites' is empty.
            - Depending on the value of 'is_local', the operator is appended to either the local operator collection (_local_ops) or the non-local operator collection (_nonlocal_ops).
            - Logs a debug message indicating the addition of the operator along with its details.

        ---
        Returns:
            None

        ---
        Example:
        >> operator    = sig_x
        >> sites       = [0, 1]
        >> hamiltonian.add(operator, sites, multiplier=1.0, is_local=True)
        >> This would add the operator 'sig_x' to the local operator list at site 0 with a multiplier of 1.0.
        """

        if abs(multiplier) < 1e-15:
            self._log(
                f"Skipping addition of operator {operator} with negligible multiplier {multiplier}",
                lvl=3,
                log="debug",
            )
            return

        try:
            from QES.Algebra.Operator.operator import create_add_operator
        except ImportError as e:
            raise ImportError(
                "QES Operator module could not be loaded. Ensure QES is properly installed."
            ) from e

        # Ensure the Hamiltonian is many-body
        if not self._is_manybody:
            raise TypeError(
                "Method 'add' is intended for Many-Body Hamiltonians to define local energy terms."
            )

        # check if the sites are provided, if one sets the operator, we would put it at a given site
        i = 0 if (sites is None or len(sites) == 0) else sites[0]
        op_tuple = create_add_operator(operator, multiplier, sites)
        modifies = modifies or operator.modifies
        # if the operator is meant to be local, it does not modify the state
        if not modifies:
            if operator.type_acting == OperatorTypeActing.Global:
                self._ops_nmod_nosites[i].append((op_tuple))
                self._log(
                    f"Adding non-modifying operator {operator} at site {i} (global) with multiplier {op_tuple[2]}",
                    lvl=2,
                    log="debug",
                )
            else:
                self._ops_nmod_sites[i].append((op_tuple))
                self._log(
                    f"Adding non-modifying operator {operator} at site {i} (sites: {str(op_tuple[1])}) with multiplier {op_tuple[2]}",
                    lvl=2,
                    log="debug",
                )
        else:
            if operator.type_acting == OperatorTypeActing.Global:
                self._ops_mod_nosites[i].append((op_tuple))
                self._log(
                    f"Adding modifying operator {operator} at site {i} (global) with multiplier {op_tuple[2]}",
                    lvl=2,
                    log="debug",
                )
            else:
                self._ops_mod_sites[i].append((op_tuple))
                self._log(
                    f"Adding modifying operator {operator} at site {i} (sites: {str(op_tuple[1])}) with multiplier {op_tuple[2]}",
                    lvl=2,
                    log="debug",
                )

        #! handle the codes for the local energy functions - this is critical for building the Hamiltonian
        op_code = self._resolve_operator_code(operator, sites)

        # Store Data, flattened
        self._instr_codes.append(op_code)
        self._instr_coeffs.append(multiplier)

        # Pad sites to self._instr_max_arity (extend if custom operator has more sites)
        s_list = list(sites) if sites else [0]
        arity = len(s_list)

        # Update max arity if this operator has more sites than current max
        if arity > self._instr_max_arity:
            self._instr_max_arity = arity
            self._log(
                f"Extended max arity to {arity} for operator {operator.name}", lvl=2, log="debug"
            )

        while len(s_list) < self._instr_max_arity:
            s_list.append(-1)
        self._instr_sites.append(s_list)

    # -------------------------------------------------------------------------
    #! Instruction Code Setup
    # -------------------------------------------------------------------------

    def setup_instruction_codes(self, physics_type: Optional[str] = None) -> None:
        r"""
        Set up instruction codes and composition function based on physics type.

        This method configures the lookup codes, instruction function, and
        maximum output size for efficient JIT-compiled composition.

        IMPORTANT: Call this method AFTER adding all operators via add() method.

        Parameters
        ----------
        physics_type : str, optional
            The physics type. Options:
            - 'spin' or 'spin-1/2':
                Spin-1/2 systems
            - 'spin-1':
                Spin-1 systems
            - 'fermion' or 'spinless-fermions':
                Spinless fermions
            - 'boson' or 'bosons':
                Bosonic systems
            - None:
                Auto-detect from local space type

        Variables Set
        -------------
        _physics_type : str
            Detected or specified physics type (e.g., 'spin-1/2', 'spinless-fermions').

        _lookup_codes : Dict[str, int]
            **Catalog** of all available operator types for this physics type.
            Maps operator names to their integer codes.
            Example for spin-1/2:
                {'Sx': 1, 'Sy': 2, 'Sz': 3, 'Sp': 4, 'Sm': 5, ...}
            This is the full catalog, not just operators used in this Hamiltonian.
            Used in _resolve_operator_code() to translate operator names to codes.

        _instr_function : Callable
            JIT-compiled function that applies a sequence of operators to a basis state.
            Signature:
                (state, nops, codes, sites, coeffs, ns, out_states, out_vals)
            Used by _build_composition_functions() to create the final wrapper.

        _instr_max_out : int
            Maximum number of output states the composition function can produce.
            Set to:
                len(_instr_codes) + 1 if operators exist, else ns + 1.

            **Why len(_instr_codes) + 1?**
            The composition function loops through all operators (n_ops = len(_instr_codes)).
            Each operator produces AT MOST one off-diagonal output state.
            All diagonal contributions are accumulated into a single sum.
            Maximum outputs = n_ops (off-diagonal) + 1 (diagonal sum).

            **Why ns + 1 when no operators?**
            Fallback for safety:
                at most ns spin flips + 1 diagonal term.

        Related Variables (set by add() method, used here)
        ---------------------------------------------------
        _instr_codes : List[int]
            List of operator codes for each term added to this Hamiltonian.
            Each add() call appends a code. Order matches _instr_coeffs and _instr_sites.
            Example: [3, 3, 1, 1, 1, ...] for SzSz + Sx + Sx + ...

        _instr_coeffs : List[float|complex]
            Coefficients for each operator term (matches _instr_codes order).

        _instr_sites : List[List[int]]
            Sites each operator acts on. Shape (n_ops, max_arity), padded with -1.

        Example
        -------
        >>> hamil._set_local_energy_operators()  # Adds operators via add()
        >>> hamil.setup_instruction_codes()      # Sets up codes AFTER operators added
        >>> hamil._set_local_energy_functions()  # Builds JIT functions using codes

        See Also
        --------
        - add :
            Add operator terms (populates _instr_codes, _instr_coeffs, _instr_sites)
        - _build_composition_functions :
            Creates JIT wrapper using these codes
        - reset_instructions :
            Clears all instruction codes
        - SpecialOperator.help('instruction_codes') :
            Full documentation
        """
        # Auto-detect physics type if not provided
        if physics_type is None:
            physics_type = self._detect_physics_type()

        # Load operator module for this physics type
        self._physics_type = physics_type
        module = self.operators

        # Get the catalog of all operator codes and the composition function
        # _lookup_codes     : Dict[str, int]    - maps operator names to integer codes
        # _instr_function   : Callable          - JIT-compiled composition function
        
        is_complex = getattr(self, "_iscpx", False)
        if not is_complex and self._dtype is not None:
            try:
                is_complex = np.issubdtype(np.dtype(self._dtype), np.complexfloating)
            except Exception:
                pass
        self._lookup_codes, self._instr_function = module.get_codes(is_complex=is_complex)

        # Set maximum output buffer size:
        # - Each operator in _instr_codes produces at most 1 off-diagonal state
        # - Plus 1 for the accumulated diagonal contribution
        # - This is exact: max_out = n_ops + 1
        self._instr_max_out = len(self._instr_codes) + 1 if self._instr_codes else self.ns + 1

    def _get_instruction_function(self) -> Optional[Callable]:
        """
        Get the appropriate instruction function for composition.

        If custom operators are present, creates a hybrid composition
        function. Otherwise returns the standard composition function.

        Returns
        -------
        Callable or None
            The composition function, or None if not available.
        """
        if not self._has_custom_ops:
            return self._instr_function

        # Create hybrid composition function
        return self._create_hybrid_composition()

    def _create_hybrid_composition(self) -> Optional[Callable]:
        """
        Create a hybrid composition function that handles custom operators.

        Returns
        -------
        Callable or None
            The hybrid composition function.
        """
        physics_type    = self._physics_type or self._detect_physics_type()
        is_complex      = getattr(self, "_iscpx", False) or self._dtype == np.complex128
        try:
            return self.operators.get_composition(
                is_cpx=is_complex,
                custom_op_funcs=self._custom_op_registry,
                custom_op_arity=self._custom_op_arity,
            )
        except Exception as e:
            self._log(
                f"Failed to create hybrid composition function: {e}",
                lvl=3,
                color="red",
                log="error",
            )

        # Fallback to standard function
        return self._instr_function

    def _finalize_instruction_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert instruction lists to numpy arrays for JIT compilation.

        Returns
        -------
        sites_arr : np.ndarray
            Shape (n_ops, max_arity), filled with -1 for unused slots.
        coeffs_arr : np.ndarray
            Coefficients for each term.
        codes_arr : np.ndarray
            Instruction codes for each term.
        """

        # Create rectangular sites array with -1 padding
        n_ops = len(self._instr_codes)
        max_arity = self._instr_max_arity
        sites_arr = np.full((n_ops, max_arity), -1, dtype=np.int32)

        for k, s_list in enumerate(self._instr_sites):
            sites_arr[k, : len(s_list)] = s_list

        coeffs_arr = np.array(self._instr_coeffs, dtype=self._dtype)
        codes_arr = np.array(self._instr_codes, dtype=np.int32)

        return sites_arr, coeffs_arr, codes_arr

    def _detect_physics_type(self) -> str:
        """Auto-detect physics type from instance attributes."""

        # Check for explicit local space type
        if hasattr(self, "_local_space") and self._local_space is not None:
            return self._local_space.typ.value

        # Check for Hilbert space with local space
        if hasattr(self, "_hilbert_space") and self._hilbert_space is not None:
            local_space = getattr(self._hilbert_space, "_local_space", None)
            if local_space is not None:
                return local_space.typ.value

        # Check common flags
        if getattr(self, "_isfermions", False):
            return "spinless-fermions"
        if getattr(self, "_isbosons", False):
            return "bosons"
        if getattr(self, "_isspin1", False):
            return "spin-1"

        return "spin-1/2"

    # -------------------------------------------------------------------------
    #! Custom Operator Registration
    # -------------------------------------------------------------------------

    def _resolve_operator_code(self, operator: Operator, sites: Optional[List[int]]) -> int:
        """
        Resolve the instruction code for an operator.

        For predefined operators with known codes, returns the code directly.
        For custom operators, registers them and returns a unique custom code.

        Parameters
        ----------
        operator : Operator
            The operator to resolve.
        sites : List[int], optional
            Sites the operator acts on.

        Returns
        -------
        int
            The instruction code.
        """
        # Try operator's own code attribute
        if hasattr(operator, "code") and operator.code is not None:
            if operator.code < CUSTOM_OP_BASE:
                return operator.code

        # Try lookup by name
        if operator.name in self._lookup_codes:
            return self._lookup_codes[operator.name]

        # Register as custom operator
        return self._register_custom_operator(operator, sites)

    def _register_custom_operator(self, operator: Operator, sites: Optional[List[int]]) -> int:
        """
        Register a custom operator.

        Parameters
        ----------
        operator : Operator
            The custom operator to register.
        sites : List[int], optional
            Sites the operator acts on.

        Returns
        -------
        int
            The assigned custom operator code.
        """
        # Validate operator has integer function
        if not hasattr(operator, "int") or operator.int is None:
            raise ValueError(self._ERR_CUSTOM_OP_NO_INT.format(operator.name))

        # Assign unique code
        custom_code = self._custom_op_counter
        self._custom_op_counter += 1

        # Determine arity
        arity = len(sites) if sites else 1

        # Register
        self._custom_op_registry[custom_code] = operator.int
        self._custom_op_arity[custom_code] = arity
        self._has_custom_ops = True

        # Add to lookup codes
        self._lookup_codes[operator.name] = custom_code

        if hasattr(self, "_log"):
            self._log(
                f"Registered custom operator '{operator.name}' with code {custom_code}, arity={arity}",
                lvl=2,
                log="debug",
            )

        return custom_code

    # -------------------------------------------------------------------------
    #! Reset/Clear Methods
    # -------------------------------------------------------------------------

    def reset_instructions(self) -> None:
        """Reset all instruction codes and custom operator registrations."""
        self._instr_codes = []
        self._instr_coeffs = []
        self._instr_sites = []
        self._instr_max_arity = 2

        self._custom_op_registry = {}
        self._custom_op_counter = CUSTOM_OP_BASE
        self._custom_op_arity = {}
        self._has_custom_ops = False

        self._cached_matvec_fun = None

    def reset_operator_terms(self) -> None:
        """
        Reset all operator term lists and composition functions.

        This resets:
        - All operator lists (modifying/non-modifying, global/local)
        - All composition functions (int, numpy, JAX)
        - Calls reset_instructions() for instruction codes and custom operators

        Subclasses should call this at the start of their build process.
        """
        # Reset operator lists by site
        for i in range(self.ns):
            self._ops_nmod_nosites[i] = []
            self._ops_nmod_sites[i] = []
            self._ops_mod_nosites[i] = []
            self._ops_mod_sites[i] = []

        # Reset composition functions
        self._composition_int_fun = None
        self._composition_np_fun = None
        self._composition_jax_fun = None

        # Reset instruction codes and custom operator registry
        self.reset_instructions()

    # -------------------------------------------------------------------------
    #! Composition Function Creation
    # -------------------------------------------------------------------------

    def _build_composition_functions(self) -> None:
        """
        Build JIT-compiled composition functions from the instruction codes.

        This creates the integer-state composition function wrapper that can
        be called efficiently during matrix construction or local calculations.

        Subclasses can override this to add additional backend support
        (e.g., NumPy, JAX versions).
        """
        if not self._instr_codes:
            self._log("No instruction codes to build composition function.", lvl=2, log="debug")
            return

        try:
            compile_start = time.perf_counter()

            # Finalize arrays
            sites_arr, coeffs_arr, codes_arr = self._finalize_instruction_arrays()
            n_ops = len(self._instr_codes)
            ns_val = self.ns

            # Get the appropriate instruction function
            instr_function = self._get_instruction_function()
            if instr_function is None:
                self._log(
                    "No instruction function available.", lvl=2, log="warning", color="yellow"
                )
                return

            # Create the wrapper
            max_out = self._instr_max_out
            is_complex = getattr(self, "_iscpx", False) or self._dtype == np.complex128

            if is_complex:

                @numba.njit(nogil=True, fastmath=True, boundscheck=False, cache=False)
                def wrapper(state: int):
                    states_buf = np.empty(max_out, dtype=np.int64)
                    vals_buf = np.empty(max_out, dtype=np.complex128)
                    return instr_function(
                        state, n_ops, codes_arr, sites_arr, coeffs_arr, ns_val, states_buf, vals_buf
                    )

            else:

                @numba.njit(nogil=True, fastmath=True, boundscheck=False, cache=False)
                def wrapper(state: int):
                    states_buf = np.empty(max_out, dtype=np.int64)
                    vals_buf = np.empty(max_out, dtype=np.float64)
                    return instr_function(
                        state, n_ops, codes_arr, sites_arr, coeffs_arr, ns_val, states_buf, vals_buf
                    )

            # Compile by calling once
            _ = wrapper(0)
            compile_end = time.perf_counter()
            self._composition_int_fun = wrapper
            self._log(
                f"Composition function compiled in {compile_end - compile_start:.6f} seconds.",
                log="info",
                lvl=3,
                color="green",
                verbose=self._verbose,
            )

        except Exception as e:
            self._log(f"Failed to build composition function: {e}", lvl=3, color="red", log="error")
            self._composition_int_fun = None

    # -------------------------------------------------------------------------
    #! Add Operator Term
    # -------------------------------------------------------------------------

    def add_instruction(
        self,
        operator: Operator,
        coefficient: Union[float, complex],
        sites: Optional[List[int]] = None,
    ) -> None:
        """
        Add an operator term to the instruction list.

        Parameters
        ----------
        operator : Operator
            The operator to add.
        coefficient : float or complex
            Coefficient for the term.
        sites : List[int], optional
            Sites the operator acts on.
        """
        if abs(coefficient) < 1e-15:
            return  # Skip negligible terms

        # Resolve instruction code
        op_code     = self._resolve_operator_code(operator, sites)

        # Normalize sites for hashing
        s_list      = list(sites) if sites else [0]
        term_key    = (op_code, tuple(s_list))

        # Check if term already exists -> Fuse it!
        if term_key in self._term_index_map:
            idx                      = self._term_index_map[term_key]
            self._instr_coeffs[idx] += coefficient
            # If fusion makes coefficient zero, we could remove it, but that's expensive (O(N)).
            # We keep it; negligible terms might be filtered later if needed.
            return

        # Store instruction
        idx         = len(self._instr_codes)
        self._instr_codes.append(op_code)
        self._instr_coeffs.append(coefficient)
        self._term_index_map[term_key] = idx

        # Handle sites
        arity       = len(s_list)

        # Extend max arity if needed
        if arity > self._instr_max_arity:
            self._instr_max_arity = arity

        # Pad to max arity (for JIT array) - Note: stored list keeps padding
        # But for fusion key we used unpadded sites
        padded_s_list = list(s_list)
        while len(padded_s_list) < self._instr_max_arity:
            padded_s_list.append(-1)

        self._instr_sites.append(padded_s_list)

    # -------------------------------------------------------------------------
    #! Help and Documentation
    # -------------------------------------------------------------------------

    @classmethod
    def help(cls, topic: Optional[str] = None) -> str:
        """
        Display help information about SpecialOperator capabilities.

        Parameters
        ----------
        topic : str, optional
            Specific topic to get help on. Options:
            - None or 'all':
                Full overview
            - 'instruction_codes':
                Instruction code system explained
            - 'operators':
                Adding operators and terms
            - 'custom_operators':
                Registering custom operators
            - 'composition':
                Composition functions for local energy
            - 'modules':
                Lazy-loaded property modules
            - 'variables':
                Internal variable reference

        Returns
        -------
        str
            Help text for the requested topic.

        Examples
        --------
        >>> SpecialOperator.help()                    # Full overview
        >>> SpecialOperator.help('instruction_codes') # Code system details
        >>> hamil.help('variables')                   # Variable reference
        """
        topics = {
            "instruction_codes": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  SpecialOperator: Instruction Code System                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  The instruction code system enables efficient JIT-compiled Hamiltonian      ║
║  application. It maps operator terms to integer codes for fast evaluation.   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  KEY VARIABLES (Catalog vs Instance):                                        ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  _lookup_codes : Dict[str, int]  [CATALOG]                                   ║
║      Dictionary of ALL available operator types for the physics type.        ║
║      Example for spin-1/2:                                                   ║
║        {'Sx': 1, 'Sy': 2, 'Sz': 3, 'Sp': 4, 'Sm': 5,                         ║
║         'Sx/L': 11, 'Sy/L': 12, 'Sz/L': 13, ...                              ║
║         'SxSy/C': 21, 'SzSz/C': 23, ...}                                     ║
║      Set by: setup_instruction_codes() from operator module                  ║
║      Used by: _resolve_operator_code() to translate names -> codes            ║
║                                                                              ║
║  _instr_codes : List[int]  [INSTANCE]                                        ║
║      List of operator codes actually added to THIS Hamiltonian.              ║
║      Each add() call appends one code.                                       ║
║      Example for H = -J Σ SzSz - h Σ Sx:                                     ║
║        [23, 23, 23, ..., 11, 11, 11, ...]  (SzSz bonds + Sx sites)           ║
║      Length = number of operator terms in Hamiltonian                        ║
║                                                                              ║
║  _instr_max_out : int  [BUFFER SIZE]                                         ║
║      Maximum output states the composition function can produce.             ║
║      Formula: len(_instr_codes) + 1                                          ║
║      Reason: Each operator -> at most 1 off-diagonal + 1 diagonal sum         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WORKFLOW:                                                                   ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  1. _set_local_energy_operators()   # Calls add() to populate _instr_codes   ║
║  2. setup_instruction_codes()       # Gets _lookup_codes, sets _instr_max_out║
║  3. _build_composition_functions()  # Creates JIT wrapper using codes        ║
║                                                                              ║
║  IMPORTANT: Call setup_instruction_codes() AFTER adding all operators!       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "operators": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SpecialOperator: Adding Operators                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Adding Terms:                                                               ║
║    .add(operator, multiplier, modifies=False, sites=None)                    ║
║        Add operator term: multiplier × operator at sites                     ║
║        - modifies: True if operator changes the state (e.g., Sx flips spin)  ║
║        - sites: List of site indices for local operators                     ║
║                                                                              ║
║    .add_instruction(operator, coefficient, sites=None)                       ║
║        Lower-level: add instruction code directly                            ║
║                                                                              ║
║  Operator Collections (organized by type):                                   ║
║    _ops_nmod_nosites[site] - Non-modifying, global operators                 ║
║    _ops_nmod_sites[site]   - Non-modifying, local operators (e.g., Sz)       ║
║    _ops_mod_nosites[site]  - Modifying, global operators                     ║
║    _ops_mod_sites[site]    - Modifying, local operators (e.g., Sx, Sp)       ║
║                                                                              ║
║  Reset Methods:                                                              ║
║    .reset_operator_terms() - Clear all operator collections                  ║
║    .reset_instructions()   - Clear instruction codes only                    ║
║                                                                              ║
║  Access Operator Module:                                                     ║
║    .operators              - Get physics-specific operator factory           ║
║    .correlators(...)       - Compute correlation operators                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "custom_operators": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   SpecialOperator: Custom Operators                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Custom operators extend beyond predefined types (Sx, Sy, Sz, etc.).         ║
║  They're automatically registered when add() encounters unknown operators.   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Registration Variables:                                                     ║
║    _custom_op_registry : Dict[int, Callable]                                 ║
║        Maps custom codes (≥1000) to numba-compiled operator functions        ║
║        Function signature: (state: int, ns: int, sites: tuple) -> (s', c)     ║
║                                                                              ║
║    _custom_op_arity : Dict[int, int]                                         ║
║        Maps custom codes to number of sites they act on                      ║
║                                                                              ║
║    _custom_op_counter : int                                                  ║
║        Next available code (starts at CUSTOM_OP_BASE = 1000)                 ║
║                                                                              ║
║    _has_custom_ops : bool                                                    ║
║        True if any custom operators registered                               ║
║                                                                              ║
║  Creating Custom Operators:                                                  ║
║    1. Create Operator with .int attribute (numba-compiled function)          ║
║    2. Add to Hamiltonian via add() - auto-registered if not in _lookup_codes ║
║    3. System creates hybrid composition for standard + custom operators      ║
║                                                                              ║
║  Example:                                                                    ║
║    @numba.njit                                                               ║
║    def my_op_int(state, ns, sites):                                          ║
║        # Apply custom operator                                               ║
║        return new_state, coefficient                                         ║
║                                                                              ║
║    my_op = Operator(name='MyOp', int=my_op_int)                              ║
║    hamil.add(my_op, multiplier=1.0, sites=[0, 1])                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "composition": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  SpecialOperator: Composition Functions                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Composition functions apply the Hamiltonian to a basis state efficiently.   ║
║  They return (output_states, output_values) for sparse matrix building.      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Function Types:                                                             ║
║    _composition_int_fun  - JIT-compiled, integer state input                 ║
║    _composition_np_fun   - NumPy-based, array state input                    ║
║    _composition_jax_fun  - JAX-based, GPU-accelerated                        ║
║                                                                              ║
║  Building Process:                                                           ║
║    1. setup_instruction_codes() gets _instr_function from operator module    ║
║    2. _build_composition_functions() creates wrapper with captured arrays:   ║
║       - codes_arr: np.array of instruction codes                             ║
║       - sites_arr: np.array of site indices (n_ops, max_arity)               ║
║       - coeffs_arr: np.array of coefficients                                 ║
║                                                                              ║
║  Wrapper Signature (created by _build_composition_functions):                ║
║    def wrapper(state: int) -> (np.array, np.array):                          ║
║        # Allocates output buffers of size _instr_max_out                     ║
║        # Calls _instr_function(state, n_ops, codes, sites, coeffs, ns, ...)  ║
║        # Returns (output_states, output_values)                              ║
║                                                                              ║
║  Output Buffer Sizing (_instr_max_out):                                      ║
║    - Each operator term produces at most 1 off-diagonal state                ║
║    - All diagonal contributions accumulate into 1 term                       ║
║    - Max outputs = len(_instr_codes) + 1                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "modules": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SpecialOperator: Lazy-Loaded Modules                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Modules are loaded on first access (lazy evaluation pattern).               ║
║  Each provides specialized functionality without import overhead.            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  .operators                                                                  ║
║      Physics-specific operator factory (spin, fermion, boson, etc.)          ║
║      Provides: sig_x, sig_z, c_dag, n_op, correlator factories               ║
║      Methods: .get_codes(), .get_composition(), .correlators()               ║
║                                                                              ║
║  .entanglement                                                               ║
║      Entanglement entropy calculations                                       ║
║      Methods: .entropy_correlation(), .entropy_many_body(),                  ║
║               .topological_entropy(), .verify_wicks_theorem()                ║
║                                                                              ║
║  .statistical                                                                ║
║      Statistical properties of eigenstates                                   ║
║      Methods: .dos(), .ldos(), .ipr(), .level_spacing(),                     ║
║               .fidelity_susceptibility(), .survival_probability()            ║
║                                                                              ║
║  .time_evo                                                                   ║
║      Time evolution and quench dynamics                                      ║
║      Methods: .evolve(), .evolve_batch(), .expectation(),                    ║
║               .quench_state(), .diagonal_ensemble()                          ║
║                                                                              ║
║  .spectral                                                                   ║
║      Spectral functions and Green's functions                                ║
║      Methods: .greens_function(), .spectral_function(),                      ║
║               .dynamic_structure_factor(), .susceptibility()                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            "variables": r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   SpecialOperator: Variable Reference                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTRUCTION CODE SYSTEM                                                     ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  _lookup_codes      : Dict[str,int] - Catalog of operator name -> code        ║
║  _instr_codes       : List[int]     - Codes for operators in this instance   ║
║  _instr_coeffs      : List[float]   - Coefficients for each operator         ║
║  _instr_sites       : List[List]    - Sites for each operator (padded)       ║
║  _instr_max_arity   : int           - Max sites per operator (for padding)   ║
║  _instr_max_out     : int           - Max output states = n_ops + 1          ║
║  _instr_function    : Callable      - JIT-compiled composition kernel        ║
║  _physics_type      : str           - Physics type (spin-1/2, fermion, etc.) ║
║                                                                              ║
║  CUSTOM OPERATORS                                                            ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  _custom_op_registry : Dict[int,Callable] - Custom code -> function           ║
║  _custom_op_arity    : Dict[int,int]      - Custom code -> num sites          ║
║  _custom_op_counter  : int                - Next custom code (≥1000)         ║
║  _has_custom_ops     : bool               - Any custom ops registered?       ║
║                                                                              ║
║  OPERATOR COLLECTIONS (by site)                                              ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  _ops_nmod_nosites[i] - Non-modifying, global operators at site i            ║
║  _ops_nmod_sites[i]   - Non-modifying, local operators at site i             ║
║  _ops_mod_nosites[i]  - Modifying, global operators at site i                ║
║  _ops_mod_sites[i]    - Modifying, local operators at site i                 ║
║                                                                              ║
║  COMPOSITION FUNCTIONS                                                       ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  _composition_int_fun : Callable    - Integer state composition (JIT)        ║
║  _composition_np_fun  : Callable    - NumPy composition function             ║
║  _composition_jax_fun : Callable    - JAX composition function               ║
║                                                                              ║
║  SYSTEM PROPERTIES                                                           ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  _is_manybody       : bool          - Many-body (True) or quadratic (False)  ║
║  _is_quadratic      : bool          - Quadratic Hamiltonian flag             ║
║  _max_local_ch      : int           - Max local changes (cardinality+1)      ║
║  _hilbert_space     : HilbertSpace  - Associated Hilbert space               ║
║  _lattice           : Lattice       - Lattice geometry                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
        }

        overview = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SpecialOperator                                 ║
║     Abstract base for operators with instruction code composition system.    ║
║      Enables efficient JIT-compiled Hamiltonian matrix construction.         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Inheritance: SpecialOperator -> Operator -> GeneralMatrix -> LinearOperator   ║
║  Extended by: BasisAwareOperator -> Hamiltonian -> Model Subclasses            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Core Concept: Instruction Code System                                       ║
║    • Each operator type has an integer code (e.g., Sx=1, Sz=3, SzSz=23)      ║
║    • Operators added via add() are stored as codes + sites + coefficients    ║
║    • JIT-compiled composition function evaluates H|state⟩ efficiently        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Key Variables:                                                              ║
║    _lookup_codes    - Dict of ALL operator types (catalog)                   ║
║    _instr_codes     - List of codes for THIS Hamiltonian's operators         ║
║    _instr_max_out   - Output buffer size = len(_instr_codes) + 1             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Topics (use .help('topic') for details):                                    ║
║    'instruction_codes' - The code system explained in detail                 ║
║    'operators'         - Adding operator terms                               ║
║    'custom_operators'  - Registering custom operators                        ║
║    'composition'       - Composition functions for local energy              ║
║    'modules'           - Lazy-loaded property modules                        ║
║    'variables'         - Complete variable reference                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

        if topic is None or topic == "all":
            result = overview
            for t in topics.values():
                result += t
            print(result)
            return result

        if topic in topics:
            print(topics[topic])
            return topics[topic]

        print(f"Unknown topic '{topic}'. Available: {list(topics.keys())}")
        return f"Unknown topic '{topic}'. Available: {list(topics.keys())}"

# ---------------------------------------------------------------------------
#! END OF FILE
# ---------------------------------------------------------------------------
