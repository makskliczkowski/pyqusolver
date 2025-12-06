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
import numba
import numpy as np
import scipy as sp
from numba.typed import List as NList
from typing import List, Tuple, Union, Optional, Callable, Dict, Any, TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from QES.Algebra.Operator.operator_loader   import OperatorModule
    from QES.general_python.common.flog         import Logger
    from QES.general_python.lattices.lattice    import Lattice
    from QES.general_python.algebra.utils       import Array
    
###################################################################################################

try:
    from    QES.Algebra.hilbert                 import HilbertSpace, HilbertConfig
    from    QES.Algebra.Operator.operator       import Operator, OperatorTypeActing, create_add_operator, OperatorFunction

    from    QES.Algebra.Hilbert.matrix_builder  import build_operator_matrix
    from    QES.Algebra.Hamil.hamil_types       import *
    from    QES.Algebra.Hamil.hamil_energy      import local_energy_np_wrap
    import  QES.Algebra.Hamil.hamil_jit_methods as hjm
    
    from    QES.Algebra.hamil_config            import (
                                                    HamiltonianConfig,
                                                    HAMILTONIAN_REGISTRY,
                                                    register_hamiltonian,
                                                )    
except ImportError as exc:
    raise ImportError("QES.Algebra.hilbert or QES.Algebra.Operator.operator could not be imported. Ensure QES is properly installed.") from exc

###################################################################################################
from QES.Algebra.Hamil.hamil_diag_engine import DiagonalizationEngine

###################################################################################################

try:
    import                  jax
    from                    jax import jit
    import                  jax.lax as lax
    import                  jax.numpy as jnp
    from                    jax.experimental.sparse import BCOO, CSR
    from                    QES.Algebra.Hamil.hamil_energy import local_energy_jax_wrap
    JAX_AVAILABLE           = True
except ImportError:
    jax                     = None
    jnp                     = None
    lax                     = None
    BCOO                    = None
    CSR                     = None
    local_energy_jax_wrap   = None
    JAX_AVAILABLE           = False

####################################################################################################
#! Hamiltonian class - abstract class
####################################################################################################

class Hamiltonian(Operator):
    '''
    A general Hamiltonian class. This class is used to define the Hamiltonian of a system. It may be 
    either a Many-Body Quantum Mechanics Hamiltonian or a non-interacting system Hamiltonian. It may 
    generate a Hamiltonian matrix but in addition it defines how an operator acts on a state. The
    Hamiltonian class is an abstract class and is not meant to be instantiated. It is meant to be
    inherited by other classes.
    '''
    
    # Error messages for Hamiltonian class
    _ERR_EIGENVALUES_NOT_AVAILABLE      = "Eigenvalues are not available. Please diagonalize the Hamiltonian first."
    _ERR_HAMILTONIAN_NOT_AVAILABLE      = "Hamiltonian matrix is not available. Please build or initialize the Hamiltonian."
    _ERR_HAMILTONIAN_INITIALIZATION     = "Failed to initialize the Hamiltonian matrix. Check Hilbert space, lattice, and parameters."
    _ERR_HAMILTONIAN_BUILD              = "Failed to build the Hamiltonian matrix. Ensure all operators and spaces are properly set."
    _ERR_HILBERT_SPACE_NOT_PROVIDED     = "Hilbert space is not provided or is invalid. Please supply a valid HilbertSpace object."
    _ERR_NS_NOT_PROVIDED                = "'ns' (number of sites/modes) must be provided, e.g., via 'ns' kwarg or a Lattice object."
    _ERR_NEED_LATTICE                   = "Lattice information is required but not provided. Please specify a lattice or number of sites."
    _ERR_COUP_VEC_SIZE                  = "Invalid coupling vector size. Coupling must be a scalar, a string, or a list/array of length ns."
    _ERR_MODE_MISMATCH                  = "Operation not supported for the current Hamiltonian mode (Many-Body/Quadratic). Check 'is_manybody' flag."
    
    # Dictionary of error messages
    _ERRORS = {
        "eigenvalues_not_available"     : _ERR_EIGENVALUES_NOT_AVAILABLE,
        "hamiltonian_not_available"     : _ERR_HAMILTONIAN_NOT_AVAILABLE,
        "hamiltonian_initialization"    : _ERR_HAMILTONIAN_INITIALIZATION,
        "hamiltonian_build"             : _ERR_HAMILTONIAN_BUILD,
        "hilbert_space_not_provided"    : _ERR_HILBERT_SPACE_NOT_PROVIDED,
        "need_lattice"                  : _ERR_NEED_LATTICE,
        "coupling_vector_size"          : _ERR_COUP_VEC_SIZE,
        "mode_mismatch"                 : _ERR_MODE_MISMATCH,
        "ns_not_provided"               : _ERR_NS_NOT_PROVIDED
    }
    
    _ADD_TOLERANCE                      = 1e-10
    def _ADD_CONDITION(x, *args):
        if x is None:                   return False
        if args:
            if len(args) == 1:
                y = x[args[0]]
            else:
                y = x[args]
        else:
            y = x
        if y is None:                   return False
        return not np.isclose(y, 0.0, rtol=Hamiltonian._ADD_TOLERANCE)

    # ----------------------------------------------------------------------------------------------
    
    @classmethod
    def from_config(cls, config: HamiltonianConfig, **overrides):
        """Construct a Hamiltonian instance from a registered configuration."""
        return HAMILTONIAN_REGISTRY.instantiate(config, **overrides)

    # ----------------------------------------------------------------------------------------------
    #! Initialization
    ################################################################################################

    def _handle_system(self, ns : Optional[int], hilbert_space : Optional[HilbertSpace], lattice : Optional['Lattice'], logger : Optional['Logger'], **kwargs):
        ''' Handle the system configuration. '''
        
        # ----------------------------------------------------------------------------------------------
                
        if ns is not None:  # if the number of sites is provided, set it
            self._ns        = ns
            self._lattice   = lattice
            if self._logger and self._hilbert_space is not None:
                self._log(f"Inferred number of sites ns={self._ns} from provided ns argument.", lvl = 3, color = 'green', log='debug')
                
        elif hilbert_space is not None:
            # if the Hilbert space is provided, get the number of sites
            self._ns        = hilbert_space.ns
            self._lattice   = hilbert_space.lattice
            if self._dtype is None:
                self._dtype = hilbert_space.dtype
            if self._logger and self._hilbert_space is not None:
                self._log(f"Inferred number of sites ns={self._ns} from provided Hilbert space.", lvl = 3, color = 'green', log='debug')
                
        elif lattice is not None:
            self._ns        = lattice.ns
            self._lattice   = lattice
            
            if self._logger and self._hilbert_space is not None:
                self._log(f"Inferred number of sites ns={self._ns} from provided lattice.", lvl = 3, color = 'green', log='debug')
        else:
            # if the number of sites is not provided, raise an error
            raise ValueError(Hamiltonian._ERR_NS_NOT_PROVIDED)
    
        if self._hilbert_space is None:
            # try to infer from lattice or number of sites
            if self._lattice is None:
                # if the lattice is not provided, create Hilbert space from number of sites
                if self._ns is None:
                    raise ValueError(Hamiltonian._ERR_NS_NOT_PROVIDED)
                
            self._hilbert_space = HilbertSpace(ns       = self._ns,
                                            lattice     = self._lattice,
                                            is_manybody = self._is_manybody,
                                            dtype       = self._dtype,
                                            backend     = self._backendstr,
                                            logger      = logger,
                                            **kwargs)    
        else:
            # otherwise proceed 
            if self._is_manybody:
                if not self._hilbert_space._is_many_body:
                    raise ValueError(Hamiltonian._ERR_MODE_MISMATCH)
                self._hamil_sp = None
            else:
                if not self._hilbert_space._is_quadratic:
                    raise ValueError(Hamiltonian._ERR_MODE_MISMATCH)
                self._hamil = None    
    
        if self._hilbert_space.ns != self._ns:
            raise ValueError(f"Ns mismatch: {self._hilbert_space.ns} != {self._ns}")
    
    def _handle_dtype(self, dtype: Optional[Union[str, np.dtype]]):
        '''
        Handle the dtype of the Hamiltonian. Overrides the method
        from 'Matrix' class to infer dtype from Hilbert space when possible.
        
        Parameters:
        -----------
            dtype (str or np.dtype, optional):
                The dtype to use for the Hamiltonian.
        '''
        if dtype is not None:
            super()._handle_dtype(dtype)
        else:
            if self._hilbert_space is not None:
                try:
                    hs_dtype = getattr(self._hilbert_space, 'dtype', None)
                    if getattr(self._hilbert_space, 'has_complex_symmetries', False):
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
    
    # ----------------------------------------------------------------------------------------------
    
    def __init__(self,
                # concerns the definition of the system type
                is_manybody     : bool                                          = True,     # True for many-body Hamiltonian, False for non-interacting
                *,
                hilbert_space   : Optional[Union[HilbertSpace, HilbertConfig]]  = None,     # Required if is_manybody=True
                ns              : Optional[int]                                 = None,     # Number of sites/modes (if not provided, will be inferred from hilbert_space or lattice)
                lattice         : Optional[Union[str, List[int]]]               = None,     # Alternative way to specify ns and get the Hilbert space
                # concerns the matrix and computation
                is_sparse       : bool                                          = True,     # True for sparse matrix, False for dense matrix
                dtype           : Optional[Union[str, np.dtype]]                = None,     # Data type for the Hamiltonian matrix elements (if None, inferred from hilbert_space or backend)
                backend         : str                                           = 'default',
                # logger and other kwargs
                use_forward     : bool                                          = False,
                logger          : Optional['Logger']                            = None,
                seed            : Optional[int]                                 = None,
                **kwargs):
        """
        Initialize the Hamiltonian class.

        Parameters
        ----------
        is_manybody : bool, optional
            If True, the Hamiltonian is treated as a many-body Hamiltonian.
            If False, it is treated as a non-interacting (single-particle) Hamiltonian. Default is True.
        hilbert_space : HilbertSpace, HilbertConfig, or None, optional
            The Hilbert space object describing the system or a blueprint to build it. Required if is_manybody=True.
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

        # Pre-compute backend for _handle_system (before calling Operator.__init__)
        (self._backendstr, self._backend, self._backend_sp, (self._rng, self._rng_k)) = Hamiltonian._set_backend(backend, seed)
        
        self._is_jax                = JAX_AVAILABLE and self._backend != np
        self._is_numpy              = not self._is_jax
        self._is_manybody           = is_manybody
        self._is_quadratic          = not is_manybody
        self._particle_conserving   = False
        self._use_forward           = use_forward
        
        self._dtype                 = dtype
        self._hilbert_space         = hilbert_space
        self._logger                = self._hilbert_space.logger if (logger is None and self._hilbert_space is not None) else logger
        
        #! general Hamiltonian info
        self._name                  = "Hamiltonian"
        
        # Handle system (ns, hilbert_space, lattice) before Operator init
        self._handle_system(ns, hilbert_space, lattice, logger, **kwargs)
        self._handle_dtype(dtype)
        
        # Initialize Operator base class (which now inherits from GeneralMatrix)
        # This sets up backend, sparse, dtype, matrix storage, diagonalization infrastructure
        Operator.__init__(self, 
                        ns        =   self.ns, 
                        name      =   self._name,
                        backend   =   backend,
                        is_sparse =   is_sparse,
                        dtype     =   dtype,
                        logger    =   self._logger,
                        seed      =   seed,
                        modifies  =   True,
                        quadratic =   not is_manybody)        
        
        # Override lattice from Hilbert space (takes precedence)
        self._lattice               = self._hilbert_space.lattice
        self._nh                    = self._hilbert_space.nh                
        if self._lattice is not None:
            self._local_nei         = self.lattice.cardinality
        else:
            self._local_nei         = ns # assume fully connected if no lattice provided
        
        #! other properties
        self._startns               = 0 # for starting hamil calculation (potential loop over sites)
        
        # =====================================================================
        #! GENERAL BASIS TRANSFORMATION TRACKING (supports any basis type)
        # =====================================================================
        self._hamil_transformed     = None                  # General storage for transformed Hamiltonian (e.g., H_k, H_fock, etc.)
        
        # Infer and set the default basis for this Hamiltonian type
        self._infer_and_set_default_basis()
        
        # for the matrix representation of the Hamiltonian
        self._hamil                 : np.ndarray            = None          # will store the Hamiltonian matrix with Nh x Nh full Hilbert space
        
        #! single particle Hamiltonian info
        self._hamil_sp              : np.ndarray            = None          # will store Ns x Ns (2Ns x 2Ns for BdG) matrix for quadratic Hamiltonian
        self._delta_sp              : np.ndarray            = None
        self._constant_offset       = 0.0
        self._isfermions            = True
        self._isbosons              = False
        
        #! set the local energy functions and the corresponding methods
        self._ops_nmod_nosites      = [[] for _ in range(self.ns)]          # operators that do not modify the state and do not act on any site (through the function call)
        self._ops_nmod_sites        = [[] for _ in range(self.ns)]          # operators that do not modify the state and act on a given site(s)
        self._ops_mod_nosites       = [[] for _ in range(self.ns)]          # operators that modify the state and do not act on any site (through the function call)
        self._ops_mod_sites         = [[] for _ in range(self.ns)]          # operators that modify the state and act on a given site(s)
        self._loc_energy_int_fun    : Optional[Callable]    = None
        self._loc_energy_np_fun     : Optional[Callable]    = None
        self._loc_energy_jax_fun    : Optional[Callable]    = None
            
        #! FOR MATRIX BUILDING    
        self._lookup_codes          : Dict[str, int]        = {}            # lookup codes for operators -> we will make numba friendly piece of junk
        self._instr_codes           : List[int]             = []            # instruction codes for building the Hamiltonian matrix
        self._instr_coeffs          : List[complex]         = []            # instruction coefficients for building the Hamiltonian matrix
        self._instr_max_arity       : int                   = 2             # maximum arity of the instructions -> 1 (local), 2 (correlation), etc.
        self._instr_sites           : List[List[int]]       = []            # instruction sites for building the Hamiltonian matrix
        
        # set by the subclass
        self._instr_function        : Callable              = None          # fun(state, noperators, codes, sites, coeffs, ns) -> new_state, coeff
        self._instr_max_out         : int                   = self.ns + 1   # maximum number of output states from the instruction function
        self._instr_buffers         : Any                   = None          # buffers for instruction function to avoid reallocation
    
    # ----------------------------------------------------------------------------------------------
    #! Representation - helpers
    ################################################################################################

    @staticmethod
    def repr(**kwargs):
        return "Hamiltonian"

    # ----------------------------------------------------------------------------------------------
    
    def _log(self, msg : str, log : str = 'info', lvl : int = 0, color : str = "white"):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (str) : The logging level. Default is 'info'.
            lvl (int) : The level of the message.
        """
        msg = f"[{self.name}] {msg}"
        self._hilbert_space._log(msg, log = log, lvl = lvl, color = color, append_msg=False)
    
    def __str__(self):
        '''
        Returns the string representation of the Hamiltonian class.
        
        Returns:
            str :
                The string representation of the Hamiltonian class.
        '''
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
        htype           = "Many-Body" if self._is_manybody else "Quadratic"
        hilbert_info    = (
                    f"HilbertSpace(Nh={self._nh})"
                    if self._hilbert_space is not None
                    else ""
                    )
        lattice_info    = (
                    f"Lattice({self._lattice})"
                    if self._lattice is not None
                    else ""
                    )
        backend_info    = f"backend='{self._backendstr}', dtype={self._dtype}"
        sparse_info     = "sparse" if self._is_sparse else "dense"

        return (
            f"<{htype} Hamiltonian | Nh={self._nh}, Ns={self._ns}, "
            f"{hilbert_info}, {lattice_info}, {backend_info}, {sparse_info}>"
        )
    
    # ----------------------------------------------------------------------------------------------
    
    def randomize(self, **kwargs):
        ''' Randomize the Hamiltonian matrix.'''
        raise NotImplementedError("Randomization is not implemented for this Hamiltonian class.")
    
    def clear(self):
        '''
        Clears the Hamiltonian matrix and related properties.
        If you want to re-build the Hamiltonian, you need to call the build method again.
        '''
        # Call parent's clear to handle matrix, eigenvalues, krylov, diag_engine
        super().clear()
        
        # Hamiltonian-specific cleanup
        self._hamil         = None
        self._hamil_sp      = None
        self._delta_sp      = None
        self._constant_sp   = None
        self._log("Hamiltonian cleared...", lvl = 2, color = 'blue')
    
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
        from_str    = str(current_basis).lower()
        to_str      = str(target_basis).lower()
        handler     = self._get_basis_transform_handler(from_str, to_str)
        
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
        return getattr(self, '_basis_type', HilbertBasisType.REAL)
    
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
        if self._hilbert_space is not None and hasattr(self._hilbert_space, '_basis_type'):
            
            default_basis                                   = self._hilbert_space._basis_type
            self._basis_metadata['inherited_from_hilbert']  = True
            self._log(f"Hamiltonian basis inherited from HilbertSpace: {default_basis}", lvl=1, color="cyan")
        else:
            # Priority 2: Infer from Hamiltonian properties
            if self._is_quadratic:
                # Quadratic system: choose based on lattice availability
                if self._lattice is not None:
                    default_basis = HilbertBasisType.REAL
                    self._basis_metadata['system_type'] = 'quadratic-real'
                else:
                    default_basis = HilbertBasisType.FOCK
                    self._basis_metadata['system_type'] = 'quadratic-fock'
            else:
                # Many-body system: choose based on lattice availability
                if self._lattice is not None:
                    default_basis = HilbertBasisType.REAL
                    self._basis_metadata['system_type'] = 'manybody-real'
                else:
                    default_basis = HilbertBasisType.FOCK
                    self._basis_metadata['system_type'] = 'manybody-fock'

            self._log(f"Hamiltonian default basis inferred: {default_basis} ({self._basis_metadata.get('system_type', 'unknown')})", lvl=2, color="cyan")
        
        # Set original and current basis
        self._original_basis    = default_basis
        self._current_basis     = default_basis
        self._is_transformed    = False
        self._log(f"Default basis inferred: {default_basis} ({self._basis_metadata.get('system_type', 'unknown')})", lvl=2, color="cyan")
    
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
            'original_basis'        : str(self._original_basis) if self._original_basis else None,
            'current_basis'         : str(self._current_basis) if self._current_basis else None,
            'is_transformed'        : self._is_transformed,
            'has_real_space'        : self._hamil is not None or self._hamil_sp is not None,
            'has_transformed'       : self._hamil_transformed is not None,
            'transformed_shape'     : self._hamil_transformed.shape if self._hamil_transformed is not None else None,
            'grid_shape'            : self._transformed_grid.shape if self._transformed_grid is not None else None,
            'symmetry_info'         : self._symmetry_info,
            'metadata'              : self._basis_metadata.copy(),
        }
        return state
    
    def print_transformation_state(self):
        """Print a human-readable summary of the transformation state."""
        state = self.get_transformation_state()
        self._log(f"Transformation State:", lvl=1, color="bold")
        self._log(f"  Original basis: {state['original_basis']}", lvl=1)
        self._log(f"  Current basis: {state['current_basis']}", lvl=1)
        self._log(f"  Is transformed: {state['is_transformed']}", lvl=1)
        self._log(f"  Real-space available: {state['has_real_space']}", lvl=1)
        self._log(f"  Transformed repr available: {state['has_transformed']}", lvl=1)
        if state['transformed_shape']:
            self._log(f"  Transformed shape: {state['transformed_shape']}", lvl=1)
        if state['grid_shape']:
            self._log(f"  Grid shape: {state['grid_shape']}", lvl=1)
        if state['symmetry_info']:
            self._log(f"  Symmetry: {state['symmetry_info']}", lvl=1)
    
    def record_symmetry_application(self, symmetry_name: str, sector: Optional[str] = None):
        """
        Record information about applied symmetries to track basis reductions.
        
        Parameters
        ----------
        symmetry_name : str
            Name of the symmetry applied (e.g., "Z2", "U1", "SU2")
        sector : str, optional
            Which sector of the symmetry (e.g., "even", "odd", "up", "down")
        """
        sector_str                                  = f" [{sector}]" if sector else ""
        self._symmetry_info                         = f"{symmetry_name}{sector_str}"
        self._basis_metadata['symmetry_applied']    = True
        self._basis_metadata['symmetries']          = self._basis_metadata.get('symmetries', []) + [symmetry_name]
        self._log(f"Recorded symmetry application: {self._symmetry_info}", lvl=2, color="yellow")
    
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
            if not hasattr(self._lattice, 'is_periodic'):
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
        
        if not hasattr(self._hilbert_space, '_basis_type'):
            return
        
        hilbert_basis = self._hilbert_space._basis_type
        
        if self._current_basis != hilbert_basis:
            self._log(f"Syncing basis: Hamiltonian {self._current_basis} -> HilbertSpace {hilbert_basis}", lvl=2, color="yellow")
            self._current_basis = hilbert_basis
    
    def push_basis_to_hilbert_space(self):
        """
        Push this Hamiltonian's basis information to its HilbertSpace.
        
        Use this when the Hamiltonian's basis representation is the source of truth.
        """
        if self._hilbert_space is None:
            return
        
        if hasattr(self._hilbert_space, 'set_basis'):
            self._hilbert_space.set_basis(str(self._current_basis))
            self._log(f"Basis pushed to HilbertSpace: {self._current_basis}", lvl=2, color="cyan")
    
    # -------------------------------------------------------------------------
    #! Getter methods
    # -------------------------------------------------------------------------

    @property
    def quadratic(self):                return self._is_quadratic
    def is_quadratic(self):             return self._is_quadratic
    
    @property
    def manybody(self):                 return self._is_manybody
    def is_manybody(self):              return self._is_manybody

    @property
    def max_local_changes(self):        return self._max_local_ch if self._is_manybody else 2
    @property
    def max_local(self):                return self.max_local_changes
    @property
    def max_operator_changes(self):     return self._max_local_ch_o if self._is_manybody else 2
    @property
    def max_operator(self):             return self.max_operator_changes

    # quadratic Hamiltonian properties
    
    @property
    def particle_conserving(self):      return self._particle_conserving if not self._is_manybody else None
    @property
    def is_particle_conserving(self):   return self.particle_conserving
    @property
    def is_bdg(self):                   return not self.particle_conserving
    
    # lattice and sites properties
    
    @property
    def cardinality(self):              return self._local_nei
    
    # modes and hilbert space properties
    
    @property
    def modes(self):                    return self._hilbert_space.local_space.local_dim if self._hilbert_space is not None else None
    @property
    def hilbert_space(self):            return self._hilbert_space
    @property
    def hilbert_size(self):             return self._hilbert_space.nh
    @property
    def dim(self):                      return self.hilbert_size
    @property
    def nh(self):                       return self.hilbert_size

    @property
    def hamil(self)                     -> Union[np.ndarray, sp.sparse.spmatrix]: return self._hamil
    @hamil.setter
    def hamil(self, hamil):             self._hamil = hamil; self._matrix = hamil

    @property
    def hamil_transformed(self):        return self._hamil_transformed
    @hamil_transformed.setter
    def hamil_transformed(self, hamil_transformed): self._hamil_transformed = hamil_transformed
        
    @property
    def grid_transformed(self):         return self._transformed_grid
    @grid_transformed.setter
    def grid_transformed(self, grid_transformed): self._transformed_grid = grid_transformed

    @property
    def matrix(self):
        '''
        Returns the Hamiltonian matrix, building it if necessary.
        
        This property automatically calls build() if the matrix has not been
        constructed yet, providing a convenient way to access the matrix.
        '''
        if self._hamil is None:
            raise ValueError("Hamiltonian matrix not built yet. Please call the build() method first.")
        return self._hamil
    @property
    def matvec_fun(self):
        '''
        Returns the matrix-vector multiplication function for the Hamiltonian.
        This is useful for iterative diagonalization methods.
        '''        
        def _matvec(x, *args):
            return self.matvec(x, *args, hilbert=self._hilbert_space)
        return _matvec
    
    # single-particle Hamiltonian properties
    @property
    def hamil_sp(self):                 return self._hamil_sp
    @hamil_sp.setter
    def hamil_sp(self, hamil_sp):       self._hamil_sp = hamil_sp
    
    @property
    def delta_sp(self):                 return self._delta_sp
    @delta_sp.setter
    def delta_sp(self, delta_sp):       self._delta_sp = delta_sp
    
    @property
    def constant_sp(self):              return self._constant_sp
    
    @constant_sp.setter
    def constant_sp(self, constant_sp): self._constant_sp = constant_sp
    
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
            self._hamil     = matrix
            self._matrix    = matrix
            self._is_built  = matrix is not None
        else:
            self._hamil_sp  = matrix
            self._is_built  = matrix is not None
    
    # ----------------------------------------------------------------------------------------------
    #! ACCESS TO OTHER MODULES RELATED TO HAMILTONIAN
    # ----------------------------------------------------------------------------------------------
    
    @property
    def operators(self) -> 'OperatorModule':
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
        >>> c_dag           = hamil.operators.c_dag(ns=4, sites=[0])
        >>> n_op            = hamil.operators.n(ns=4, sites=[0])
        
        >>> # Get help
        >>> hamil.operators.help()
        """
        if not hasattr(self, '_operator_module') or self._operator_module is None:
            
            try:
                from QES.Algebra.Operator.operator_loader import get_operator_module
            except ImportError as e:
                raise ImportError("Operator module could not be loaded. Ensure QES is properly installed.") from e
            
            # Get local space type from Hilbert space if available
            if self._hilbert_space is not None and hasattr(self._hilbert_space, '_local_space'):
                local_space_type    = self._hilbert_space._local_space.typ
            else:
                # Default to spin-1/2 for many-body, None for quadratic
                from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
                local_space_type    = LocalSpaceTypes.SPIN_HALF if self._is_manybody else None
            self._operator_module   = get_operator_module(local_space_type)
        return self._operator_module
    
    def correlators(self, indices_pairs:List[Tuple[int, int]], correlators=None, type_acting='global', **kwargs) -> dict:
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

        op_module = self.operators # Ensure operator module is loaded
        if hasattr(op_module, 'correlators'):
            return op_module.correlators(indices_pairs=indices_pairs, correlators=correlators, type_acting=type_acting, **kwargs)
        else:
            raise AttributeError("Operator module does not have a 'correlators' attribute.")
    
    @property
    def entanglement(self):
        """
        Lazy-loaded entanglement module for convenient entanglement entropy calculations.
        
        Returns
        -------
        EntanglementModule
            Module providing entanglement entropy calculations for arbitrary bipartitions.
            Supports both correlation matrix (fast, for quadratic Hamiltonians) and 
            many-body (exact, for any state) methods.
            
        Examples
        --------
        >>> # For quadratic Hamiltonians (fast correlation matrix method)
        >>> hamil       = FreeFermions(ns=12, t=1.0)
        >>> hamil.diagonalize()
        >>> ent         = hamil.entanglement
        >>> bipart      = ent.bipartition([0, 1, 2, 3, 4])  # First 5 sites
        >>> S           = ent.entropy_correlation(bipart, orbitals=[0,1,2,3,4])
        
        >>> # For any Hamiltonian (exact many-body method)
        >>> state       = hamil.many_body_state([0,1,2,3,4])
        >>> S           = ent.entropy_many_body(bipart, state)
        
        >>> # Non-contiguous bipartitions
        >>> bipart      = ent.bipartition([0, 2, 4, 6, 8])  # Even sites
        >>> S           = ent.entropy_correlation(bipart, [0,1,2])
        
        >>> # Entropy scaling
        >>> results     = ent.entropy_scan([0,1,2,3,4])
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(results['sizes'], results['entropies'])
        
        >>> # Get help
        >>> hamil.entanglement.help()
        """
        if not hasattr(self, '_entanglement_module') or self._entanglement_module is None:
            from QES.general_python.physics.entanglement_module import get_entanglement_module
            self._entanglement_module = get_entanglement_module(self)
        return self._entanglement_module
    
    # ----------------------------------------------------------------------------------------------
    
    def indices_around_energy(self, energy: float, fraction: Union[int, float] = 500):
        '''
        Returns the indices around the average energy of the Hamiltonian.
        This is used to track the average energy during calculations.
        
        Returns:
            tuple : (index, value) of the average energy.
        '''
        hilbert_size    = len(self.eig_val)
        if fraction <= 1.0:
            fraction_in = int(fraction * hilbert_size)
        else:
            fraction_in = int(fraction)
        
        #! get the energy index
        energy_in       = self.av_en if energy is None else energy
        energy_index    = int(np.argmin(np.abs(self.eig_val - energy_in))) if energy is None else self.av_en_idx
        
        #! left
        left_index      = min(max(0, energy_index - fraction_in // 2), energy_index - 5)
        right_index     = max(min(hilbert_size, energy_index + fraction_in // 2), energy_index + 5)

        return left_index, right_index, energy_index
    
    # ----------------------------------------------------------------------------------------------
    #! Local energy getters
    # ----------------------------------------------------------------------------------------------
    
    @property
    def fun_int(self):                  return self._loc_energy_int_fun
    def get_loc_energy_int_fun(self):   return self._loc_energy_int_fun
    
    @property
    def fun_npy(self):                  return self._loc_energy_np_fun
    def get_loc_energy_np_fun(self):    return self._loc_energy_np_fun
    
    @property
    def fun_jax(self):                  return self._loc_energy_jax_fun
    def get_loc_energy_jax_fun(self):   return self._loc_energy_jax_fun
    
    def get_loc_energy_arr_fun(self, backend: str = 'default', typek: str = 'int'):
        '''
        Returns the local energy of the Hamiltonian
        Returns:
            A function that takes an integer k and returns the local energy for an array representation in
            a given backend - either NumPy or JAX.
        '''
        if typek == 'int':
            return self.get_loc_energy_int_fun()
        if typek == 'jax':
            return self.get_loc_energy_jax_fun()
        if typek == 'npy':
            return self.get_loc_energy_np_fun()
        
        if (backend == 'default' or backend == 'jax' or backend == 'jnp') and JAX_AVAILABLE:
            return self.fun_jax
        return self.fun_npy
    
    # ----------------------------------------------------------------------------------------------
    #! Memory properties
    # ----------------------------------------------------------------------------------------------
    
    @property
    def h_memory(self):                 return super().mat_memory
    @property
    def h_memory_gb(self):              return self.h_memory / (1024.0 ** 3)

    # ----------------------------------------------------------------------------------------------
    #! Standard getters
    # ----------------------------------------------------------------------------------------------
    
    def get_mean_lvl_spacing(self, use_npy = True):
        '''
        Returns the mean level spacing of the Hamiltonian. The mean level spacing is defined as the
        average difference between consecutive eigenvalues.
        
        Returns:
            float : The mean level spacing of the Hamiltonian.
        '''
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        if (not JAX_AVAILABLE or self._backend == np or use_npy):
            return self._backend.mean(self._backend.diff(self._eig_val))
        return hjm.mean_level_spacing(self._eig_val)
    
    def get_bandwidth(self):
        '''
        Returns the bandwidth of the Hamiltonian. The bandwidth is defined as the difference between
        the highest and the lowest eigenvalues - values are sorted in ascending order.
        '''
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        return self._eig_val[-1] - self._eig_val[0]
    
    def get_energywidth(self, use_npy = True):
        '''
        Returns the energy width of the Hamiltonian. The energy width is defined as trace of the
        Hamiltonian matrix squared.
        '''
        if self._hamil.size == 0:
            raise ValueError(Hamiltonian._ERR_HAMILTONIAN_NOT_AVAILABLE)
        if (not JAX_AVAILABLE or self._backend == np or use_npy):
            return self._backend.trace(self._backend.dot(self._hamil, self._hamil))
        return hjm.energy_width(self._hamil)
    
    # ----------------------------------------------------------------------------------------------
    
    def _set_some_coupling(self, coupling: Union[list, np.ndarray, float, complex, int, str]) -> 'Array':
        '''
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
        '''
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
    
    def init(self, use_numpy : bool = False):
        '''
        Initializes the Hamiltonian matrix. Uses Batched-coordinate (BCOO) sparse matrices if JAX is
        used, otherwise uses NumPy arrays. The Hamiltonian matrix is initialized to be a matrix of
        zeros if the Hamiltonian is not sparse, otherwise it is initialized to be an empty sparse
        matrix.
        
        Parameters:
            use_numpy (bool):
                A flag indicating whether to use NumPy or JAX.
        '''
        self._log("Initializing the Hamiltonian matrix...", lvl = 2, log = "debug")
        
        jax_maybe_avail = self._is_jax
        if jax_maybe_avail and use_numpy:
            self._log("JAX is available but NumPy is forced...", lvl = 1, log = 'warning')
            jax_maybe_avail = False
            
        if self._is_quadratic:
            # Initialize Quadratic Matrix (_hamil_sp)
            # Shape determined by subclass, assume (Ns, Ns) for now
            ham_shape = getattr(self, '_hamil_sp_shape', (self._ns, self._ns))
            self._log(f"Initializing Quadratic Hamiltonian structure {ham_shape} (Sparse={self.sparse})...", lvl=3, log="debug")
            
            if self.sparse:
                if self._is_numpy:
                    self._hamil_sp = sp.sparse.csr_matrix(ham_shape, dtype=self._dtype)
                else:
                    indices         = self._backend.zeros((0, 2), dtype=ACTIVE_INT_TYPE)
                    data            = self._backend.zeros((0,), dtype=self._dtype)
                    self._hamil_sp  = BCOO((data, indices), shape=ham_shape)
                    self._delta_sp  = BCOO((data, indices), shape=ham_shape)
            else:
                self._hamil_sp      = self._backend.zeros(ham_shape, dtype=self._dtype)
                self._delta_sp      = self._backend.zeros(ham_shape, dtype=self._dtype)
            self._hamil = None
        else:
            if self.sparse:
                self._log("Initializing the Hamiltonian matrix as a sparse matrix...", lvl = 3, log = "debug")
                
                # --------------------------------------------------------------------------------------
                
                if not jax_maybe_avail or use_numpy:
                    self._log("Initializing the Hamiltonian matrix as a CSR sparse matrix...", lvl = 3, log = "debug")
                    self._hamil = sp.sparse.csr_matrix((self._nh, self._nh), dtype = self._dtype)
                else:
                    self._log("Initializing the Hamiltonian matrix as a sparse matrix...", lvl = 3, log = "debug")
                    # Create an empty sparse Hamiltonian matrix using JAX's BCOO format
                    indices     = self._backend.zeros((0, 2), dtype=ACTIVE_INT_TYPE)
                    data        = self._backend.zeros((0,), dtype=self._dtype)
                    self._hamil = BCOO((data, indices), shape=(self._nh, self._nh))
                    
                # --------------------------------------------------------------------------------------
                
            else:
                self._log("Initializing the Hamiltonian matrix as a dense matrix...", lvl = 3, log = "debug")
                if not JAX_AVAILABLE or self._backend == np:
                    self._hamil     = self._backend.zeros((self._nh, self._nh), dtype=self._dtype)
                else:
                    # do not initialize the Hamiltonian matrix
                    self._hamil     = None
        self._log(f"Hamiltonian matrix initialized and it's {'many-body' if self._is_manybody else 'quadratic'}", lvl = 3, color = "green", log = "debug")
    
    # ----------------------------------------------------------------------------------------------
    #! Many body Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------

    def _hamiltonian_validate(self):
        ''' Check if the Hamiltonian matrix is valid. '''
        
        matrix_to_check = self._hamil if (self._is_manybody) else self._hamil_sp
        
        if matrix_to_check is None:
            self._log("Hamiltonian matrix is not initialized.", lvl=3, color="red", log = "debug")
        else:
            valid   = False
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
                self._log("Hamiltonian matrix calculated and valid.", lvl=3, color="green", log = "debug")
            else:
                self._log("Hamiltonian matrix calculated but empty or invalid.", lvl=3, color="red", log = "debug")
                matrix_to_check = None

    def _transform_to_backend(self):
        '''
        Transforms the Hamiltonian matrix to the backend.
        
        Note: With the modular backend architecture, matrices are typically already
        in the correct backend upon creation. This method is retained for compatibility.
        '''
        # Since backend-agnostic code paths handle backend selection during construction,
        # no explicit transformation is needed here.
        self._log(f"Hamiltonian matrix verified for backend {self._backendstr}", lvl=2, color="green")
    
    # ----------------------------------------------------------------------------------------------

    def build(self, verbose: bool = False, use_numpy: bool = True, force: bool = False):
        '''
        Builds the Hamiltonian matrix. It checks the internal masks 
        wheter it's many-body or quadratic...
        
        Args:
            verbose (bool) :
                A flag to indicate whether to print the progress of the build.
            use_numpy (bool) :
                Force numpy usage.
            
        '''
        if self.hamil is not None:
            if not force:
                self._log("Hamiltonian matrix already built. Use force=True to rebuild.", lvl=1)
                return
            else:
                self._log("Forcing rebuild of the Hamiltonian matrix...", lvl=1)
                self.hamil = None # Clear existing Hamiltonian to force rebuild
        
        if verbose:
            self._log(f"Building Hamiltonian (Type: {'Many-Body' if self._is_manybody else 'Quadratic'})...", lvl=1, color = 'orange')
        
        if self._is_manybody:
            # Ensure operators/local energy functions are defined
            if self._loc_energy_int_fun is None and self._loc_energy_np_fun is None and self._loc_energy_jax_fun is None:
                self._log("Local energy functions not set, attempting to set them via _set_local_energy_operators...", lvl=2, log="debug")
                try:
                    self._set_local_energy_operators()  # Should be implemented by MB subclass
                    self._set_local_energy_functions()  
                except Exception as e:
                    raise RuntimeError(f"Failed to set up operators/local energy functions: {e}")
        
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
            self._log(f"Initialization completed in {init_duration:.6f} seconds", lvl = 2)
        
        ################################
        #! Build the Hamiltonian matrix
        ################################§
        ham_start = time.perf_counter()
        try:
            if self._is_manybody:
                self._hamiltonian(use_numpy)
            else:
                self._hamiltonian_quadratic(use_numpy)
        except Exception as e:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_BUILD} : {str(e)}") from e
        ham_duration = time.perf_counter() - ham_start
        
        if (self._hamil is not None and self._hamil.size > 0) or (self._hamil_sp is not None and self._hamil_sp.size > 0):
            if verbose:
                self._log(f"Hamiltonian matrix built in {ham_duration:.6f} seconds.", lvl = 1)
        else:
            raise ValueError(f"{Hamiltonian._ERR_HAMILTONIAN_BUILD} : The Hamiltonian matrix is empty or invalid.")

    # ----------------------------------------------------------------------------------------------
    #! Local energy methods - Abstract methods
    # ----------------------------------------------------------------------------------------------
    
    def loc_energy_int(self, k_map : int, i : int):
        '''
        Calculates the local energy.  MUST return NumPy arrays.

        Parameters:
            k_map (int) : The mapping of the k'th element.
            i (int)     : The i'th site.

        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        if self._loc_energy_int_fun is None:
            self._set_local_energy_functions()
        return self._loc_energy_int_fun(k_map, i)
    
    def loc_energy_arr_jax(self, k : Union[int, np.ndarray]):
        '''
        Calculates the local energy based on the Hamiltonian. This method should be implemented by subclasses.
        Uses an array as a state input.
        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        return self._loc_energy_jax_fun(k)
    
    def loc_energy_arr_np(self, k : Union[np.ndarray]):
        '''
        Calculates the local energy based on the Hamiltonian. This method should be implemented by subclasses.
        Uses an array as a state input.
        Returns:
            Tuple[np.ndarray, np.ndarray]   :  (row_indices, values)
                - row_indices               :  The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        return self._loc_energy_np_fun(k)
    
    def loc_energy_arr(self, k : Union[int, np.ndarray]) -> Tuple[List[int], List[int]]:
        '''
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by
        subclasses to provide a specific implementation.
        This is meant to check how does the Hamiltonian act on a state.
        Parameters:
            k (Union[int, np.ndarray]) : The k'th element of the Hilbert space - may use mapping if necessary.
        Returns:
            Tuple[List[int], List[int]] :  (row_indices, values)
                - row_indices               : The row indices after the operator acts.
                - values                    : The corresponding matrix element values.
        '''
        if self._is_jax or isinstance(k, jnp.ndarray):
            if self._loc_energy_jax_fun is None:
                self._set_local_energy_functions()
            return self.loc_energy_arr_jax(k)
        
        # go!
        if self._loc_energy_np_fun is None:
            self._set_local_energy_functions()
        return self.loc_energy_arr_np(k)
    
    def loc_energy(self, k : Union[int, np.ndarray], i : int = 0):
        '''
        Calculates the local energy of the Hamiltonian. This method is meant to be overridden by 
        subclasses to provide a specific implementation.
        
        This is meant to check how does the Hamiltonian act on a state at a given site.
        
        Parameters:
            k (Union[int, Backend.ndarray])         : The k'th element of the Hilbert space - may use mapping if necessary.
            i (int)                                 : The i'th site.
        '''
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

    def _hamiltonian(self, use_numpy : bool = False, sparse : Optional[bool] = None):
        '''
        Generates the Hamiltonian matrix. The diagonal elements are straightforward to calculate,
        while the off-diagonal elements are more complex and depend on the specific Hamiltonian.
        It iterates over the Hilbert space to calculate the Hamiltonian matrix. 
        
        Note: This method may be overridden by subclasses to provide a more efficient implementation
        '''
        
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
        self._log(f"Calculating the {matrix_type} Hamiltonian matrix...", lvl=1, color="blue", log = 'debug')
        # -----------------------------------------------------------------------------------------
        
        # Check if JAX is available and the backend is not NumPy
        jax_maybe_av = self._is_jax
        
        # Choose implementation based on backend availability.sym_eig_py
        if not jax_maybe_av or use_numpy:
            self._log("Calculating the Hamiltonian matrix using NumPy...", lvl=2, log = 'info')
            
            # Calculate the Hamiltonian matrix using the optimized matrix builder
            self._hamil = build_operator_matrix(
                hilbert_space       =   self._hilbert_space,
                operator_func       =   self._loc_energy_int_fun,
                sparse              =   self._is_sparse,
                max_local_changes   =   self._max_local_ch_o,
                dtype               =   self._dtype,
                ns                  =   self._ns,
                nh                  =   self._hilbert_space.nh,
            )
        else:
            raise ValueError("JAX not yet implemented for the build...")

        # Check if the Hamiltonian matrix is calculated and valid using various backend checks
        self._hamiltonian_validate()
    
    # ----------------------------------------------------------------------------------------------
    #! Single particle Hamiltonian matrix
    # ----------------------------------------------------------------------------------------------
    
    def _hamiltonian_quadratic(self, use_numpy : bool = False):
        '''
        Generates the Hamiltonian matrix whenever the Hamiltonian is single-particle. 
        This method needs to be implemented by the subclasses.
        '''
        pass
    
    # ----------------------------------------------------------------------------------------------
    #! Calculators
    # ----------------------------------------------------------------------------------------------
    
    def _calculate_av_en(self):
        '''
        Calculates the properties of the Hamiltonian matrix that are related to the energy.
        '''
        
        if self._eig_val is None or self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        self._av_en     = self._backend.mean(self._eig_val)
        self._min_en    = self._backend.min(self._eig_val)
        self._max_en    = self._backend.max(self._eig_val)
        self._std_en    = self._backend.std(self._eig_val)
        self._nh        = self._eig_val.size
        
        # average energy index
        self._av_en_idx = self._backend.argmin(self._backend.abs(self._eig_val - self._av_en))
    
    def calculate_en_idx(self, en : float):
        '''
        Calculates the index of the energy level closest to the given energy.
        
        Args:
            en (float) : The energy level.
        
        Returns:
            int : The index of the energy level closest to the given energy.
        '''
        if self._eig_val.size == 0:
            raise ValueError(Hamiltonian._ERR_EIGENVALUES_NOT_AVAILABLE)
        return self._backend.argmin(self._backend.abs(self._eig_val - en))
    
    # ----------------------------------------------------------------------------------------------
    #! Diagonalization methods
    # ----------------------------------------------------------------------------------------------

    def _diagonalize_quadratic_prepare(self, backend, **kwargs):
        '''
        Diagonalizes the Hamiltonian matrix whenever the Hamiltonian is single-particle. 
        This method needs to be implemented by the subclasses.
        '''
        if self._is_quadratic:
            if True:
                if self.particle_conserving:
                    self._log("Diagonalizing the quadratic Hamiltonian matrix without BdG...", lvl = 2, log = 'debug')
                    self._hamil = self._hamil_sp
                else:
                    self._log("Diagonalizing the quadratic Hamiltonian matrix with BdG...", lvl = 2, log = 'debug')
                    if self._isfermions:
                        self._hamil = backend.block([   [ self._hamil_sp, self._delta_sp ],
                                                        [-self._delta_sp.conj(), -self._hamil_sp.conj().T ]])
            else: # bosons - use \Sigma H to make it Hermitian
                sigma = backend.block([ [backend.eye(self.ns), backend.zeros_like(self._hamil)  ],
                                        [backend.zeros_like(self._hamil), -backend.eye(self.ns) ]])
                self._hamil = sigma @ backend.block([[ self._hamil_sp,  self._delta_sp          ],
                                        [self._delta_sp.conj().T, self._hamil_sp.conj().T       ]])
    
    def diagonalize(self, verbose: bool = False, **kwargs):
        """
        Diagonalizes the Hamiltonian matrix using a modular, flexible approach.
        
        This method provides a unified interface to multiple diagonalization strategies:
        - 'auto'            : Automatically select method based on matrix size/properties
        - 'exact'           : Full diagonalization (all eigenvalues)
        - 'lanczos'         : Lanczos iteration for sparse symmetric matrices
            - 'nh'              : Hamiltonian size threshold for auto Lanczos
            - 'k'               : Number of eigenvalues to compute
            - 'which'           : Which eigenvalues to find
            - 'tol'             : Convergence tolerance
            - 'max_iter'        : Maximum iterations
            - 'sigma'           : Shift for shift-invert mode
        -----------
        Additional iterative methods:
        - 'block_lanczos'   : Block Lanczos for multiple eigenpairs
        - 'arnoldi'         : Arnoldi iteration for general matrices
        - 'shift-invert'    : Shift-invert mode for interior eigenvalues
        
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
        max_iter : int
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
        
        # Start timing
        diag_start      = time.perf_counter()
        
        # Extract parameters
        if True:
            method          = kwargs.get("method", self._diag_method)
            if method in kwargs: kwargs.pop("method")
            
            backend_str     = kwargs.get("backend", None)
            if backend_str in kwargs: kwargs.pop("backend")
            
            use_scipy       = kwargs.get("use_scipy", True)
            if use_scipy in kwargs: kwargs.pop("use_scipy")

            store_basis     = kwargs.get("store_basis", True)
            if store_basis in kwargs: kwargs.pop("store_basis")

            k               = kwargs.get("k", None)
            if k in kwargs: kwargs.pop("k")
            
            which           = kwargs.get("which", "smallest")
            if which in kwargs: kwargs.pop("which")
            
            hermitian       = kwargs.get("hermitian", True)
            if hermitian in kwargs: kwargs.pop("hermitian")

        # Determine backend
        if backend_str is None:
            # Infer from current Hamiltonian backend
            if self._is_jax:
                backend_str = 'jax'
            else:
                backend_str = 'scipy' if use_scipy else 'numpy'
        
        # Special handling for quadratic Hamiltonians
        if self._is_quadratic:
            self._diagonalize_quadratic_prepare(self._backend, **kwargs)
        
        # Log start
        if verbose:
            self._log(f"Diagonalization started using method='{method}'...", lvl=1)
            if k is not None and kwargs.get('method', 'exact') != 'exact':
                self._log(f"Computing {k} eigenvalues", lvl=2)
            self._log(f"Backend: {backend_str}", lvl=2)
        
        # Initialize or reuse diagonalization engine
        if self._diag_engine is None or self._diag_engine.method != method:
            self._diag_engine = DiagonalizationEngine(
                                        method      = method,
                                        backend     = backend_str,
                                        use_scipy   = use_scipy,
                                        verbose     = verbose,
                                        logger      = self._logger
                                    )
        
        # Prepare solver kwargs (remove our custom parameters)
        solver_kwargs   = {key: val for key, val in kwargs.items() 
                        if key not in ['method', 'backend', 'use_scipy', 'store_basis', 'hermitian', 'k', 'which']}
        
        matrix_to_diag  = self._hamil
        if method == 'exact':
            if JAX_AVAILABLE and BCOO is not None and isinstance(matrix_to_diag, BCOO):
                if verbose:
                    self._log("Converting JAX sparse Hamiltonian to dense array for exact diagonalization.", lvl=2, color="yellow")
                matrix_to_diag = np.asarray(matrix_to_diag.todense())
            elif sp.sparse.issparse(matrix_to_diag):
                if verbose:
                    self._log("Converting SciPy sparse Hamiltonian to dense array for exact diagonalization.", lvl=2, color="yellow")
                matrix_to_diag = matrix_to_diag.toarray()

        # Perform diagonalization
        try:
            result = self._diag_engine.diagonalize(
                A               = matrix_to_diag,
                matvec          = self.matvec_fun,
                n               = self._nh, 
                k               = k,
                hermitian       = hermitian,
                which           = which,
                store_basis     = store_basis,
                dtype           = self._dtype,
                **solver_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Diagonalization failed with method '{method}': {e}") from e
        
        # Store results in Hamiltonian
        self._eig_val       = result.eigenvalues
        self._eig_vec       = result.eigenvectors
        
        # Store Krylov basis if available
        if store_basis and self._diag_engine.has_krylov_basis():
            self._krylov    = self._diag_engine.get_krylov_basis()
        else:
            self._krylov    = None
        
        # JAX: ensure results are materialized
        if JAX_AVAILABLE:
            if hasattr(self._eig_val, "block_until_ready"):
                self._eig_val = self._eig_val.block_until_ready()
            if hasattr(self._eig_vec, "block_until_ready"):
                self._eig_vec = self._eig_vec.block_until_ready()
        
        # Calculate average energy
        self._calculate_av_en()
        
        # Log completion
        diag_duration = time.perf_counter() - diag_start
        
        if verbose:
            method_used = self._diag_engine.get_method_used()
            self._log(f"Diagonalization ({method_used}) completed in {diag_duration:.6f} seconds, {diag_duration/3600:.6f} hours.", lvl=2, color="green")
            
            if hasattr(result, 'converged'):
                if result.converged:
                    if result.iterations is not None:
                        self._log(f"  Converged in {result.iterations} iterations", lvl=2)
                else:
                    self._log(f"  Warning: Did not converge after {result.iterations} iterations", lvl=2, color="yellow")
            self._log(f"  Ground state energy: {self._eig_val[0]:.10f}", lvl=2, log='debug')

    # ----------------------------------------------------------------------------------------------
    #! Energy related methods -> building the Hamiltonian
    # ----------------------------------------------------------------------------------------------
    
    def reset_operators(self):
        '''
        Resets the Hamiltonian operators...
        '''
        self._ops_nmod_nosites      = [[] for _ in range(self.ns)]      # operators that do not modify the state and do not act on any site (through the function call)
        self._ops_nmod_sites        = [[] for _ in range(self.ns)]      # operators that do not modify the state and act on a given site(s)
        self._ops_mod_nosites       = [[] for _ in range(self.ns)]      # operators that modify the state and do not act on any site (through the function call)
        self._ops_mod_sites         = [[] for _ in range(self.ns)]      # operators that modify the state and act on a given site(s)
        self._loc_energy_int_fun    = None
        self._loc_energy_np_fun     = None
        self._loc_energy_jax_fun    = None

    def setup_instruction_codes(self, physics_type: Optional[str] = None) -> None:
        """
        Automatically set up instruction codes and composition function based on the physics type.
        
        This method configures the lookup codes, instruction function, and maximum output size
        for efficient JIT-compiled local energy calculations. It automatically detects the
        physics type from the Hilbert space if not specified.
        
        Parameters
        ----------
        physics_type : str, optional
            The physics type to use. Options:
            - 'spin' or 'spin-1/2'  : Spin-1/2 systems (Pauli matrices)
            - 'spin-1'              : Spin-1 systems
            - 'fermion' or 'spinless-fermions' : Spinless fermion systems
            - 'boson' or 'bosons'   : Bosonic systems (not yet implemented)
            - None                  : Auto-detect from Hilbert space
            
        Raises
        ------
        ValueError
            If physics type cannot be determined or is not supported.
            
        Notes
        -----
        After calling this method, the following attributes are configured:
        - `_lookup_codes`    : Dict mapping operator names to integer codes
        - `_instr_function`  : JIT-compiled function for operator composition
        - `_instr_max_out`   : Maximum number of output states
        
        Examples
        --------
        >>> # Auto-detect from Hilbert space
        >>> hamil.setup_instruction_codes()
        
        >>> # Explicitly set for spin systems
        >>> hamil.setup_instruction_codes('spin')
        
        >>> # For fermion systems
        >>> hamil.setup_instruction_codes('fermion')
        """
        from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
        
        # Auto-detect physics type if not provided
        if physics_type is None:
            if hasattr(self, '_hilbert_space') and self._hilbert_space is not None:
                local_space = getattr(self._hilbert_space, '_local_space', None)
                if local_space is not None:
                    physics_type = local_space.typ.value
                elif hasattr(self._hilbert_space, 'local_space_type'):
                    physics_type = self._hilbert_space.local_space_type
            
            # Fall back to checking common flags
            if physics_type is None:
                if hasattr(self, '_isfermions') and self._isfermions:
                    physics_type = 'spinless-fermions'
                elif hasattr(self, '_isbosons') and self._isbosons:
                    physics_type = 'bosons'
                else:
                    physics_type = 'spin-1/2'  # Default to spin
        
        # Normalize physics type string
        physics_type = physics_type.lower().strip()
        
        # Set up based on physics type
        if physics_type     in ['spin', 'spin-1/2', LocalSpaceTypes.SPIN_1_2.value]:
            self._setup_spin_instruction_codes()
        elif physics_type   in ['spin-1', LocalSpaceTypes.SPIN_1.value]:
            self._setup_spin1_instruction_codes()
        elif physics_type   in ['fermion', 'fermions', 'spinless-fermions', LocalSpaceTypes.SPINLESS_FERMIONS.value]:
            self._setup_fermion_instruction_codes()
        elif physics_type   in ['boson', 'bosons', LocalSpaceTypes.BOSONS.value]:
            self._setup_boson_instruction_codes()
        else:
            raise ValueError(
                f"Unsupported physics type: '{physics_type}'. "
                f"Supported types: 'spin', 'spin-1/2', 'spin-1', 'fermion', 'spinless-fermions', 'boson', 'bosons'"
            )
        
        self._log(f"Instruction codes configured for physics type: {physics_type}", lvl=2, log='debug')

    def _setup_spin_instruction_codes(self) -> None:
        """Set up instruction codes for spin-1/2 systems."""
        try:
            import QES.Algebra.Operator.operators_spin as operators_spin_module
        except ImportError as e:
            raise ImportError("Spin operator module not available. Ensure QES is properly installed.") from e
        
        self._lookup_codes   = operators_spin_module.SPIN_LOOKUP_CODES.to_dict()
        self._instr_function = operators_spin_module.sigma_composition_integer(is_complex=self._iscpx)
        self._instr_max_out  = len(self._instr_codes) + 1 if self._instr_codes else self.ns + 1

    def _setup_spin1_instruction_codes(self) -> None:
        """Set up instruction codes for spin-1 systems."""
        # For now, fall back to spin-1/2 implementation
        # TODO: Implement full spin-1 support
        self._log("Spin-1 instruction codes using spin-1/2 fallback.", lvl=2, log='warning', color='yellow')
        self._setup_spin_instruction_codes()

    def _setup_fermion_instruction_codes(self) -> None:
        """Set up instruction codes for spinless fermion systems."""
        try:
            import QES.Algebra.Operator.operators_spinless_fermions as operators_fermion_module
        except ImportError as e:
            raise ImportError("Fermion operator module not available. Ensure QES is properly installed.") from e
        
        # Check if fermion module has lookup codes
        if hasattr(operators_fermion_module, 'FERMION_LOOKUP_CODES'):
            self._lookup_codes = operators_fermion_module.FERMION_LOOKUP_CODES.to_dict()
        else:
            # Build a basic lookup table for fermions
            self._lookup_codes = {
                'c_dag'     : 1,
                'c_dag/L'   : 2,
                'c_dag/C'   : 3,
                'c_ann'     : 4,
                'c_ann/L'   : 5,
                'c_ann/C'   : 6,
                'n'         : 7,
                'n/L'       : 8,
                'n/C'       : 9,
            }
            self._log("Using default fermion lookup codes.", lvl=2, log='debug')
        
        # Check if fermion module has composition function
        if hasattr(operators_fermion_module, 'fermion_composition_integer'):
            self._instr_function = operators_fermion_module.fermion_composition_integer(is_complex=self._iscpx)
        else:
            # Fallback: no specialized composition function
            self._instr_function = None
            self._log("No fermion composition function available. Using fallback.", lvl=2, log='warning', color='yellow')
        
        self._instr_max_out = len(self._instr_codes) + 1 if self._instr_codes else self.ns + 1

    def _setup_boson_instruction_codes(self) -> None:
        """Set up instruction codes for bosonic systems."""
        # Bosons not yet fully implemented
        self._log("Boson instruction codes not yet implemented. Using empty codes.", lvl=2, log='warning', color='yellow')
        self._lookup_codes      = {}
        self._instr_function    = None
        self._instr_max_out     = self.ns + 1

    # ----------------------------------------------------------------------------------------------
    #! Adding operators to the Hamiltonian
    # ----------------------------------------------------------------------------------------------
    
    def add(self, operator: Operator, multiplier: Union[float, complex, int], modifies: bool = False, sites = None):
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
            self._log(f"Skipping addition of operator {operator} with negligible multiplier {multiplier}", lvl = 3, log = 'debug')
            return
        
        # Ensure the Hamiltonian is many-body
        if not self._is_manybody:
            raise TypeError("Method 'add' is intended for Many-Body Hamiltonians to define local energy terms.")
        
        # check if the sites are provided, if one sets the operator, we would put it at a given site
        i           = 0 if (sites is None or len(sites) == 0) else sites[0]
        op_tuple    = create_add_operator(operator, multiplier, sites)
        modifies    = modifies or operator.modifies
        # if the operator is meant to be local, it does not modify the state
        if not modifies:
            if operator.type_acting == OperatorTypeActing.Global:
                self._ops_nmod_nosites[i].append((op_tuple))
                self._log(f"Adding non-modifying operator {operator} at site {i} (global) with multiplier {op_tuple[2]}", lvl = 2, log = 'debug')
            else:
                self._ops_nmod_sites[i].append((op_tuple))
                self._log(f"Adding non-modifying operator {operator} at site {i} (sites: {str(op_tuple[1])}) with multiplier {op_tuple[2]}", lvl = 2, log = 'debug')
        else:
            if operator.type_acting == OperatorTypeActing.Global:
                self._ops_mod_nosites[i].append((op_tuple))
                self._log(f"Adding modifying operator {operator} at site {i} (global) with multiplier {op_tuple[2]}", lvl = 2, log = 'debug')
            else:
                self._ops_mod_sites[i].append((op_tuple))
                self._log(f"Adding modifying operator {operator} at site {i} (sites: {str(op_tuple[1])}) with multiplier {op_tuple[2]}", lvl = 2, log = 'debug')
                
        #! handle the codes for the local energy functions - this is critical for building the Hamiltonian
        if not hasattr(operator, 'code'):
            # Fallback mapping based on name
            op_code = self._lookup_codes[operator.name]
        else:
            op_code = operator.code
        
        # Store Data, flattened
        self._instr_codes.append(op_code)
        self._instr_coeffs.append(multiplier)
        
        # Pad sites to to self._instr_max_arity
        s_list                                      = list(sites) if sites else [0] 
        while len(s_list) < self._instr_max_arity:  s_list.append(-1)
        self._instr_sites.append(s_list)

    def _set_local_energy_operators(self):
        '''
        This function is meant to be overridden by subclasses to set the local energy operators.
        The local energy operators are used to calculate the local energy of the Hamiltonian.
        Note:
            It is the internal function that knows about the structure of the Hamiltonian.
        '''
        self.reset_operators()
    
    def _set_local_energy_finalize_arrays(self):
        """Call this before creating the JIT wrapper"""
        n_ops       = len(self._instr_codes)
        max_arity   = self._instr_max_arity
        
        # Create rectangular array filled with -1
        sites_arr   = np.full((n_ops, max_arity), -1, dtype=np.int32)
        
        # Fill it (this Python loop is fast enough as it runs once per build)
        for k, s_list in enumerate(self._instr_sites):
            sites_arr[k, :len(s_list)] = s_list
            
        return sites_arr, np.array(self._instr_coeffs), np.array(self._instr_codes)
    
    def _set_local_energy_functions(self):
        """
        Private method that configures and sets local energy functions for different numerical backends.
        
        This method initializes three versions of the local energy function based on the available operator representations:
            - Integer operations: Constructs a version using the integer attributes (op.int) of both nonlocal and local operators.
            - NumPy operations: Constructs a version using the NumPy functions (op.npy) for numerical evaluations.
            - JAX operations: If JAX is available, constructs a version using the JAX functions (op.jax).
            
        The method uses the following wrapper functions to create the local energy functions:
        For each backend, the method:
        
            1. Iterates over the nonlocal and local operators for each site.
            2. Extracts the appropriate operator function (int, npy, or jax), along with corresponding sites and values.
            3. Wraps the extracted operators using the respective wrapper function (local_energy_int_wrap, local_energy_np_wrap, or local_energy_jax_wrap)
                to create the corresponding local energy function.
            4. Stores the resulting function into instance attributes (_loc_energy_int_fun, _loc_energy_np_fun, _loc_energy_jax_fun).
        
        --- 
        Note:
            - This function assumes that the instance has the attributes:
                - self.ns: number of sites.
                - self._nonlocal_ops: a list of nonlocal operator tuples for each site.
                - self._local_ops: a list of local operator tuples for each site.
            - JAX version is set only if the flag JAX_AVAILABLE is True.
        """
        
        if not self._is_manybody:
            self._log("Skipping local energy function setup for Quadratic Hamiltonian.", log='debug')
            return
        
        try:
            compile_start                       = time.perf_counter()
            nops_val                            = len(self._instr_codes)
            sites_arr, coeffs_arr, codes_arr    = self._set_local_energy_finalize_arrays()
            ns_val                              = self.ns
            instr_function                      = self._instr_function
            
            # Create unique buffer ID for this instance
            max_out                             = self._instr_max_out
            use_complex                         = self._iscpx

            if use_complex:
                @numba.njit(nogil=True, inline='always', fastmath=True)
                def wrapper(k: int):
                    # Allocate inside - unavoidable with current Numba limitations
                    # Using inline='always' reduces function call overhead
                    states_buf  = np.empty(max_out, dtype=np.int64)
                    vals_buf    = np.empty(max_out, dtype=np.complex128)
                    return instr_function(k, nops_val, codes_arr, sites_arr, coeffs_arr, ns_val, states_buf, vals_buf)
            else:
                @numba.njit(nogil=True, inline='always', fastmath=True)
                def wrapper(k: int):
                    states_buf  = np.empty(max_out, dtype=np.int64)
                    vals_buf    = np.empty(max_out, dtype=np.float64)
                    return instr_function(k, nops_val, codes_arr, sites_arr, coeffs_arr, ns_val, states_buf, vals_buf)
            
            # Compile the function by calling it once with a dummy argument
            _           = wrapper(0)
            compile_end = time.perf_counter()
            self._log(f"Local energy function compiled in {compile_end - compile_start:.6f} seconds.", log='info', lvl=3, color="green")
            self._loc_energy_int_fun = wrapper

        except Exception as e:
            self._log(f"Failed to set integer local energy function: {e}", lvl=3, color="red", log='error')
            self._loc_energy_int_fun            = None
            return
            
        #! LOCAL ENERGY FUNCTION FOR INTEGER IS SET BY THE CHILDREN CLASS
        
        # set the NumPy functions
        try:
            # set the numpy functions
            operators_mod_np                = [[(op.npy, sites, vals)   for (op, sites, vals) in self._ops_mod_sites[i]]    for i in range(self.ns)]
            operators_mod_np_nsites         = [[(op.npy, [], vals)      for (op, _, vals) in self._ops_mod_nosites[i]]      for i in range(self.ns)]
            operators_nmod_np               = [[(op.npy, sites, vals)   for (op, sites, vals) in self._ops_nmod_sites[i]]   for i in range(self.ns)]
            operators_nmod_np_nsites        = [[(op.npy, [], vals)      for (op, _, vals) in self._ops_nmod_nosites[i]]     for i in range(self.ns)]
            self._loc_energy_np_fun         = local_energy_np_wrap(self.ns,
                                                    operator_terms_list             = operators_mod_np,
                                                    operator_terms_list_ns          = operators_mod_np_nsites,
                                                    operator_terms_list_nmod        = operators_nmod_np,
                                                    operator_terms_list_nmod_ns     = operators_nmod_np_nsites,
                                                    n_max                           = self._max_local_ch,
                                                    dtype                           = self._dtype)
        except Exception as e:
            self._log(f"Failed to set NumPy local energy functions: {e}", lvl=3, color="red", log='error')
            self._loc_energy_np_fun = None
        
        # set the jax functions
        if JAX_AVAILABLE:
            try:
                operators_jax               = [[(op.jax, sites, vals)   for (op, sites, vals)   in self._ops_mod_sites[i]]      for i in range(self.ns)]
                operators_jax_nosites       = [[(op.jax, None, vals)    for (op, _, vals)       in self._ops_mod_nosites[i]]    for i in range(self.ns)]
                operators_local_jax         = [[(op.jax, sites, vals)   for (op, sites, vals)   in self._ops_nmod_sites[i]]     for i in range(self.ns)]
                operators_local_jax_nosites = [[(op.jax, None, vals)    for (op, _, vals)       in self._ops_nmod_nosites[i]]   for i in range(self.ns)]
                self._loc_energy_jax_fun    = local_energy_jax_wrap(self.ns,
                                                    operator_terms_list             = operators_jax,
                                                    operator_terms_list_ns          = operators_jax_nosites,
                                                    operator_terms_list_nmod        = operators_local_jax,
                                                    operator_terms_list_nmod_ns     = operators_local_jax_nosites,
                                                    n_max                           = self._max_local_ch,
                                                    dtype                           = self._dtype)
            except Exception as e:
                self._log(f"Failed to set JAX local energy functions: {e}", lvl=3, color="red", log='error')
        else:
            self._log("JAX is not available, skipping JAX local energy function setup.", lvl=3, color="yellow", log='debug')
            self._loc_energy_jax_fun = None

        # log success
        
        self._max_local_ch_o = max(self._max_local_ch_o,max(len(op) for op in self._ops_mod_sites)      + \
                                                        max(len(op) for op in self._ops_mod_nosites)    + \
                                                        max(len(op) for op in self._ops_nmod_sites)     + \
                                                        max(len(op) for op in self._ops_nmod_nosites))
        
        # Set the OperatorFunction for Operator inheritance
        self._fun = OperatorFunction(
            fun_int         =   self._loc_energy_int_fun,
            fun_np          =   self._loc_energy_np_fun,
            fun_jax         =   self._loc_energy_jax_fun,
            modifies_state  =   True,
            necessary_args  =   0
        )
        self._type_acting   = OperatorTypeActing.Global
        
        self._log(f"Max local changes set to {self._max_local_ch_o}", lvl=2, color="green", log='debug')
        self._log("Successfully set local energy functions...", lvl=2, log ='debug')

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
        topics = {
            'construction': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Hamiltonian: Construction                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Key Parameters:                                                             ║
║    hilbert_space : HilbertSpace - Hilbert space with symmetries              ║
║    lattice       : Lattice      - Lattice geometry (alternative to ns)       ║
║    ns            : int          - Number of sites (if no lattice)            ║
║    is_manybody   : bool         - True for many-body, False for quadratic    ║
║    dtype         : np.dtype     - Data type (float64 or complex128)          ║
║    backend       : str          - 'numpy', 'jax', or 'default'               ║
║                                                                              ║
║  Model-Specific Subclasses (use these instead of base Hamiltonian):          ║
║    HeisenbergKitaev(lattice, K=1.0, J=None, hz=None, ...)                    ║
║    TransverseIsing(lattice, J=1.0, h=1.0, ...)                               ║
║    FreeFermions(ns=N, t=1.0, mu=0.0, ...)                                    ║
║    BCSHamiltonian(ns=N, t=1.0, delta=0.5, ...)                               ║
║                                                                              ║
║  From Configuration:                                                         ║
║    Hamiltonian.from_config(HamiltonianConfig(...))                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            'operators': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Hamiltonian: Adding Operators                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Adding Terms to Hamiltonian:                                                ║
║    .add(operator, multiplier, modifies=False, sites=None)                    ║
║        Add an operator term: multiplier × operator                           ║
║                                                                              ║
║  Operator Collections:                                                       ║
║    .reset_operators()        - Clear all operator terms                      ║
║    ._ops_nmod_nosites        - Operators that don't modify state (global)    ║
║    ._ops_nmod_sites          - Operators that don't modify state (local)     ║
║    ._ops_mod_nosites         - Operators that modify state (global)          ║
║    ._ops_mod_sites           - Operators that modify state (local)           ║
║                                                                              ║
║  Accessing Operators:                                                        ║
║    .operators                - Get the operator module for this Hamiltonian  ║
║    .correlators(pairs, types)- Compute correlation operators                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            'local_energy': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Hamiltonian: Local Energy Functions                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Local Energy (for NQS/VMC):                                                 ║
║    .local_energy(state)      - Compute E_loc = ⟨state|H|state⟩/⟨state|state⟩ ║
║    .fun_int                  - Integer-based local energy function           ║
║    .fun_npy                  - NumPy-based local energy function             ║
║    .fun_jax                  - JAX-based local energy function               ║
║                                                                              ║
║  Instruction Codes (for JIT compilation):                                    ║
║    ._lookup_codes            - Dict mapping operator names to codes          ║
║    ._instr_codes             - Instruction codes for operators               ║
║    ._instr_function          - JIT-compiled instruction function             ║
║    ._instr_max_out           - Maximum output dimension                      ║
║                                                                              ║
║  Setting Up Local Energy:                                                    ║
║    ._set_local_energy_operators()    - Configure operator instructions       ║
║    ._set_local_energy_functions()    - Build JIT functions                   ║
║    .setup_instruction_codes()        - Auto-setup based on physics type      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            'diagonalization': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Hamiltonian: Diagonalization                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Main Method (overrides Operator.diagonalize with extra features):           ║
║    .diagonalize(method='auto', k=None, which='smallest', ...)                ║
║                                                                              ║
║  Hamiltonian-Specific Features:                                              ║
║    - Automatic handling of quadratic Hamiltonians                            ║
║    - Average energy calculation                                              ║
║    - Integration with Hilbert space symmetries                               ║
║                                                                              ║
║  After Diagonalization:                                                      ║
║    .eigenvalues, .eig_val    - Energy eigenvalues                            ║
║    .eigenvectors, .eig_vec   - Energy eigenstates                            ║
║    .ground_state             - Ground state |ψ₀⟩                             ║
║    .ground_energy            - Ground state energy E₀                        ║
║    .spectral_gap             - E₁ - E₀                                       ║
║    .av_en                    - Average energy                                ║
║                                                                              ║
║  Matrix Building:                                                            ║
║    .build_hamiltonian()      - Build the Hamiltonian matrix                  ║
║    .hamil                    - Access the built Hamiltonian matrix           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            'quadratic': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Hamiltonian: Quadratic Hamiltonians                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Non-Interacting Systems (is_manybody=False):                                ║
║    H = Σᵢⱼ hᵢⱼ c†ᵢ cⱼ + Σᵢⱼ Δᵢⱼ c†ᵢ c†ⱼ + h.c.                              ║
║                                                                              ║
║  Properties:                                                                 ║
║    .is_quadratic             - True if non-interacting                       ║
║    .is_manybody              - True if many-body                             ║
║    ._hamil_sp                - Single-particle Hamiltonian matrix            ║
║    ._delta_sp                - Pairing matrix (for BCS)                      ║
║                                                                              ║
║  Quadratic Diagonalization:                                                  ║
║    - Automatically handled in .diagonalize()                                 ║
║    - Uses single-particle basis transformations                              ║
║    - Much more efficient than many-body (O(N³) vs O(2^N))                    ║
║                                                                              ║
║  Many-Body State Construction:                                               ║
║    .many_body_state(orbitals)- Build many-body state from filled orbitals    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            'kspace': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Hamiltonian: K-Space Methods                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Transformation to K-Space:                                                  ║
║    .to_kspace()                      - Transform Hamiltonian to k-space      ║
║        Returns: (H_k, kgrid, kgrid_frac)                                     ║
║                                                                              ║
║    .from_kspace(H_k, kgrid)          - Transform back to real space          ║
║        Returns: H_real                                                       ║
║                                                                              ║
║    .transform_operator_to_kspace(O)  - Transform any operator to k-space     ║
║        Returns: (O_k, kgrid, kgrid_frac)                                     ║
║                                                                              ║
║  Requirements:                                                               ║
║    - Lattice must be provided                                                ║
║    - Lattice must support k-space transformations                            ║
║                                                                              ║
║  Use Cases:                                                                  ║
║    - Band structure calculations                                             ║
║    - Momentum-resolved spectral functions                                    ║
║    - Bloch state analysis                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            'inherited': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                Hamiltonian: Inherited from Operator/GeneralMatrix            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  From Operator:                                                              ║
║    .apply(state, *args)      - Apply Hamiltonian to state                    ║
║    .matrix(dim, hilbert)     - Generate matrix representation                ║
║    .matvec(v, hilbert)       - Matrix-vector product                         ║
║    .type_acting              - Operator type (usually Global)                ║
║                                                                              ║
║  From GeneralMatrix:                                                         ║
║    Spectral Analysis:                                                        ║
║      .spectral_gap, .spectral_width, .level_spacing()                        ║
║      .participation_ratio(n), .degeneracy(tol)                               ║
║      .level_spacing_ratio()   - For chaos analysis                           ║
║                                                                              ║
║    Matrix Operations:                                                        ║
║      .diag                    - Diagonal elements (auto-selects matrix)      ║
║      .expectation_value(ψ, φ), .overlap(v1, v2)                              ║
║      .trace_matrix(), .frobenius_norm(), .spectral_norm()                    ║
║      .commutator(O), .anticommutator(O)                                      ║
║                                                                              ║
║    Memory & Control:                                                         ║
║      .memory_mb, .memory_gb, .clear()                                        ║
║      .to_sparse(), .to_dense()                                               ║
║                                                                              ║
║  IMPORTANT: Matrix Reference Override                                        ║
║    Hamiltonian overrides _get_matrix_reference() to auto-select:             ║
║      • Many-body (is_manybody=True)  → uses _hamil (full many-body matrix)   ║
║      • Quadratic (is_manybody=False) → uses _hamil_sp (single-particle)      ║
║    This ensures diag, trace, norms, etc. use the correct matrix!             ║
║                                                                              ║
║  Use Operator.help() or GeneralMatrix.help() for full inherited methods.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        }
        
        overview = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                               Hamiltonian                                    ║
║         Quantum Hamiltonian class for many-body and quadratic systems.       ║
║            Supports diagonalization, local energy, and k-space.              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Inheritance: Hamiltonian → Operator → GeneralMatrix → LinearOperator       ║
║  Model Subclasses: HeisenbergKitaev, TransverseIsing, FreeFermions, ...      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quick Start:                                                                ║
║    1. Create model:    H = TransverseIsing(ns=10, J=1.0, h=0.5)              ║
║    2. Build matrix:    H.build_hamiltonian()                                 ║
║    3. Diagonalize:     H.diagonalize()                                       ║
║    4. Get results:     E0 = H.ground_energy; psi0 = H.ground_state           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Topics (use .help('topic') for details):                                    ║
║    'construction'    - Building Hamiltonians                                 ║
║    'operators'       - Adding operator terms                                 ║
║    'local_energy'    - Local energy for NQS/VMC                              ║
║    'diagonalization' - Diagonalization methods                               ║
║    'quadratic'       - Non-interacting (quadratic) Hamiltonians              ║
║    'kspace'          - K-space transformations                               ║
║    'inherited'       - Methods from Operator/GeneralMatrix                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        if topic is None or topic == 'all':
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

# --------------------------------------------------------------------------------------------------
#! End of File
# --------------------------------------------------------------------------------------------------