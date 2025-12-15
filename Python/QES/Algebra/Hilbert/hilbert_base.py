"""
Base Hilbert space class for quantum many-body systems.

This module provides the abstract base class for Hilbert spaces, handling:
- Basic system properties (Ns, lattice, local dimensions)
- Backend configuration
- Symmetry container management
- Logging and debugging utilities

---------------------------------------------------
File    : QES/Algebra/hilbert_base.py
Author  : Maksymilian Kliczkowski
Date    : 2025-12-08
Version : 1.0.0
---------------------------------------------------
"""
from    __future__ import annotations

import  numpy       as np
from    abc         import ABC
from    dataclasses import dataclass
from    typing      import Union, Optional, List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from QES.general_python.common.flog             import Logger
    from QES.general_python.lattices.lattice        import Lattice, LatticeDirection
    from QES.Algebra.Hilbert.hilbert_local          import LocalSpace, StateTypes
    from QES.Algebra.globals                        import GlobalSymmetry
    from QES.Algebra.Symmetries.symmetry_container  import SymmetryContainer, CompactSymmetryData

# ------------------------------------------------------------------------------------------------------
#! Base Hilbert Space Class
# ------------------------------------------------------------------------------------------------------

class BaseHilbertSpace(ABC):
    """
    Abstract base class for Hilbert spaces.
    
    Handles common functionality like backend setup, logging, and 
    symmetry container management.
    """
    
    @dataclass
    class SymBasicInfo:
        num_operators   : int   = 0
        num_gens        : int   = 0
        num_sectors     : int   = 0
        num_states      : int   = 0

    def __init__(self,
                    ns              : Optional[int],
                    lattice         : Optional['Lattice'],
                    local_space     : Optional['LocalSpace'],
                    backend         : str,
                    state_type      : Union[str, type],
                    dtype           : np.dtype,
                    *,
                    state_filter    : Optional[callable]    = None,
                    boundary_flux   : Optional[float]       = None,
                    logger          : Optional['Logger']    = None,
                    **kwargs
                ):
        """
        Initialize the base Hilbert space.
        
        Parameters
        ----------
        ns : int
            Number of sites
        lattice : Lattice
            Lattice structure
        local_space : LocalSpace
            Local Hilbert space configuration
        backend : str
            Computation backend ('numpy', 'jax', etc.)
        state_type : str or type
            State representation type
        dtype : np.dtype
            Data type for arrays
        logger : Logger
            Logger instance
        state_filter : callable
            Predicate to filter basis states
        boundary_flux : float
            Boundary flux for lattice
        kwargs : dict
            Additional arguments
        """
        
        self._logger                                            = self._check_logger(logger)
        self._backend, self._backend_str, self._state_type      = self.reset_backend(backend, state_type)
        self._dtype                                             = dtype if dtype is not None else self._backend.float64
        
        self._ns                                                = ns
        self._lattice                                           = lattice
        self._boundary_flux     = boundary_flux
        if self._lattice is not None and boundary_flux is not None:
            self._lattice.flux  = boundary_flux
        
        self._local_space                                       = local_space
        self._threadnum                                         = kwargs.get('threadnum', 1) # number of threads to use
        
        # Symmetry container (initialized by subclasses)
        self._sym_group                                         = []                            
        self._sym_container     : Optional['SymmetryContainer'] = None
        self._sym_basic_info                                    = self.SymBasicInfo()
        self._global_syms       : List['GlobalSymmetry']        = []
        self._has_complex_symmetries                            = False
        
        # Mapping properties (managed by subclasses but defined here for type safety)
        self.representative_list                                = None
        self.representative_norms                               = None
        self._nh                : int                           = 0
        self._nhfull            : Union[int, float]             = 0
        
        # State filtering - predicate to filter basis states
        self._state_filter                                      = state_filter
        
    # --------------------------------------------------------------------------------------------------
    #! Backend and Logging
    # --------------------------------------------------------------------------------------------------

    def _check_logger(self, logger: Optional['Logger']) -> 'Logger':
        ''' Check and return the logger instance '''
        if logger is None:
            from QES.general_python.common.flog import get_global_logger
            return get_global_logger()
        return logger

    def _log(self, msg: str, log: Union[int, str] = 'info', lvl: int = 0, color: str = "white", append_msg=True):
        """Log a message."""
        if self._logger is None:
            return
        
        try:
            from QES.general_python.common.flog import Logger
        except ImportError:
            return
        
        if isinstance(log, str):
            log = Logger.LEVELS_R.get(log, 20) # Default to INFO level if not found
            
        if append_msg:
            msg = f"[{self.__class__.__name__}] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)

    @staticmethod
    def reset_backend(backend: str, state_type: Union[str, type]):
        """Reset the backend for the Hilbert space."""
        if isinstance(backend, str):
            from QES.general_python.algebra.utils import get_backend
            _backend_str = backend
            _backend = get_backend(backend)
        else:
            _backend_str = 'np' if backend == np else 'jax'
            _backend = backend
        
        statetype = BaseHilbertSpace.reset_statetype(state_type, _backend)
        return _backend, _backend_str, statetype

    @staticmethod
    def reset_statetype(state_type: Union[str, type, object], backend):
        """Reset the state type for the Hilbert space."""
        if state_type is int:
            return int
            
        # Handle Enum or string-like objects
        if hasattr(state_type, "value"):
             s_val = str(state_type.value)
        else:
             s_val = str(state_type)
             
        if s_val.lower() in ("integer", "int"):
            return int
            
        return backend.array

    # --------------------------------------------------------------------------------------------------
    #! Properties
    # --------------------------------------------------------------------------------------------------

    @property
    def ns(self)                -> int:                     return self._ns
    @property
    def lattice(self)           -> Optional['Lattice']:     return self._lattice
    @property
    def local_space(self)       -> Optional['LocalSpace']:  return self._local_space
    @property
    def backend(self):          return self._backend
    @property
    def backend_str(self)       -> str:                     return self._backend_str
    @property
    def logger(self)            -> 'Logger':                return self._logger
    @property
    def dtype(self)             -> np.dtype:                return self._dtype

    @property
    def dim(self)               -> int:                     return self._nh
    @property
    def nh(self)                -> int:                     return self._nh
    @property
    def nhfull(self)            -> Union[int, float]:       return self._nhfull

    # --------------------------------------------------------------------------------------------------
    #! Symmetry Properties
    # --------------------------------------------------------------------------------------------------

    @property
    def sym_container(self) -> Optional['SymmetryContainer']:
        return self._sym_container

    @property
    def sym_group(self):
        """Return the symmetry group from the container."""
        if self._sym_container:
            return self._sym_container.symmetry_group
        return []

    @property
    def check_global_symmetry(self) -> bool:
        """Check if there are any global symmetries."""
        return len(self._global_syms) > 0

    @property
    def has_complex_symmetries(self) -> bool:
        """Check if symmetry phases are complex."""
        return self._has_complex_symmetries

    @property
    def has_compact_symmetry_data(self) -> bool:
        """Check if compact symmetry data is available."""
        return self._sym_container is not None and self._sym_container.has_compact_data

    @property
    def compact_symmetry_data(self) -> Optional['CompactSymmetryData']:
        """Get the CompactSymmetryData structure."""
        if self._sym_container is None:
            return None
        return self._sym_container.compact_data

    @property
    def repr_idx(self) -> Optional[np.ndarray]:
        """Representative index array: state -> representative index (uint32)."""
        if self._sym_container is not None and self._sym_container.has_compact_data:
            return self._sym_container.compact_data.repr_map
        return None
    
    @property
    def repr_phase(self) -> Optional[np.ndarray]:
        """Symmetry phases for all states."""
        if self._sym_container is not None and self._sym_container.has_compact_data:
            cd = self._sym_container.compact_data
            return cd.phase_table[cd.phase_idx]
        return None
    
    @property
    def normalization(self) -> Optional[np.ndarray]:
        """Normalization factors for representatives."""
        if self._sym_container is not None and self._sym_container.has_compact_data:
            return self._sym_container.compact_data.normalization
        return self.representative_norms

    # --------------------------------------------------------------------------------------------------
    #! Symmetry Methods (Delegated)
    # --------------------------------------------------------------------------------------------------

    def find_sym_repr(self, state, nb: float = 1.0) -> Tuple[int, Union[float, complex]]:
        """Find representative state using symmetry container."""
        if self._sym_container is None:
            return state, 1.0
        return self._sym_container.find_representative(state, nb)

    def find_sym_norm(self, state) -> Union[float, complex]:
        """Compute normalization factor using symmetry container."""
        if self._sym_container is None:
            return 1.0
        return self._sym_container.compute_normalization(state)

    # Aliases
    def find_repr(self, state, nb: float = 1.0):    return self.find_sym_repr(state, nb)
    def find_norm(self, state):                     return self.find_sym_norm(state)
    
    def norm(self, state):
        """Get normalization for a representative state index."""
        if self.representative_norms is not None and state < len(self.representative_norms):
            return self.representative_norms[state]
        return self.find_norm(state)

    def get_sym_info(self) -> str:
        """Create information string about symmetries."""
        tmp = ""
        if self._sym_container is not None and self._sym_container.generators:
            for op, (gen_type, sector) in self._sym_container.generators:
                tmp += f"{gen_type}={sector},"
                
        if self.check_global_symmetry:
            for g in self._global_syms:
                tmp += f"{g.get_name_str() if hasattr(g, 'get_name_str') else g.name}={g.get_val() if hasattr(g, 'get_val') else ''},"
        
        return tmp[:-1] if tmp else ""

    # --------------------------------------------------------------------------------------------------
    #! Directory Naming
    # --------------------------------------------------------------------------------------------------

    @property
    def symmetry_directory_name(self) -> str:
        """Return filesystem-safe string of symmetry sectors."""
        if self._sym_container is None or not self._sym_container.generators:
            return ""
        
        names = []
        for item in self._sym_container.generators:
            # item is (operator, spec)
            if isinstance(item, tuple) and len(item) >= 1:
                op = item[0]
            else:
                op = item
            
            if hasattr(op, 'directory_name'):
                names.append(op.directory_name)
            elif hasattr(op, 'sector'):
                from QES.Algebra.Symmetries.base import SymmetryOperator
                sector_str = SymmetryOperator._sector_to_str(op.sector)
                names.append(f"{op.__class__.__name__.lower()}_{sector_str}")
        
        return (",".join(names)).replace('_', "=") if names else "nosym"

    @property
    def lattice_directory_name(self) -> str:
        """Return filesystem-safe lattice string."""
        if self._lattice is not None:
            return str(self._lattice)
        elif self._ns is not None and self._ns > 0:
            return f"n={self._ns}"
        return "unknown"

    @property
    def full_directory_name(self) -> str:
        """Return complete directory name."""
        return f"{self.lattice_directory_name}/{self.symmetry_directory_name}"

    # --------------------------------------------------------------------------------------------------
    #! Magic Methods
    # --------------------------------------------------------------------------------------------------

    def __str__(self):
        info = f"HilbertSpace: Ns={self._ns}, Nh={self._nh}"
        if self._local_space:
            info += f", Local={self._local_space}"
        sym_info = self.get_sym_info()
        if sym_info:
            info += f", Symmetries=[{sym_info}]"
        return info

    def __repr__(self):
        return self.__str__()

    def transform_to_reduced_space(self, vec_full: np.ndarray) -> np.ndarray:
        """
        Transform vector(s) from full Hilbert space to reduced symmetry sector.
        
        Parameters
        ----------
        vec_full : np.ndarray
            Vector(s) in full Hilbert space. Shape (N_full,) or (N_full, N_batch).
            
        Returns
        -------
        np.ndarray
            Vector(s) in reduced Hilbert space. Shape (N_reduced,) or (N_reduced, N_batch).
        """
        if not self.has_compact_symmetry_data:
            # If no symmetries (or just global constraints), check if dimensions match
            if self.nh == self.nhfull:
                return vec_full
            else:
                raise NotImplementedError("Transform without compact symmetry data not implemented.")
        
        cd          = self.compact_symmetry_data
        repr_map    = cd.repr_map
        phase_idx   = cd.phase_idx
        phase_table = cd.phase_table
        norms       = cd.normalization
        
        # Check dimensions
        if vec_full.shape[0] != len(repr_map):
            raise ValueError(f"Vector dimension {vec_full.shape[0]} does not match full Hilbert space {len(repr_map)}")
            
        # Initialize reduced vector
        input_shape = vec_full.shape
        reduced_shape = (self.dim,) + input_shape[1:]
        
        vec_red = np.zeros(reduced_shape, dtype=self.dtype)
        
        # Ensure complex if needed (phases are complex)
        if np.iscomplexobj(vec_full) or self.has_complex_symmetries:
            vec_red = vec_red.astype(np.complex128)
            
        # 1. Find valid indices (states that belong to this sector)
        # _INVALID_REPR_IDX = 0xFFFFFFFF (uint32 max)
        valid_mask = (repr_map != 0xFFFFFFFF)
        
        if not np.any(valid_mask):
            return vec_red
            
        valid_indices = np.where(valid_mask)[0]
        target_k      = repr_map[valid_indices]
        phases        = phase_table[phase_idx[valid_indices]]
        
        # Handle broadcasting for batch dimension
        # We sum c_s * conj(chi(g_s))
        if vec_full.ndim > 1:
            # (N_valid, 1) * (N_valid, Batch) -> (N_valid, Batch)
            contributions = np.conj(phases)[:, np.newaxis] * vec_full[valid_indices]
        else:
            # (N_valid,) * (N_valid,) -> (N_valid,)
            contributions = np.conj(phases) * vec_full[valid_indices]
            
        # Accumulate: vec_red[k] += contribution
        np.add.at(vec_red, target_k, contributions)
        
        # Normalize: vec_red[k] /= N_k
        if vec_full.ndim > 1:
            vec_red /= norms[:, np.newaxis]
        else:
            vec_red /= norms
        
        return vec_red
    
# ------------------------------------------------------------------------------------------------------
#! End of File
# ------------------------------------------------------------------------------------------------------
