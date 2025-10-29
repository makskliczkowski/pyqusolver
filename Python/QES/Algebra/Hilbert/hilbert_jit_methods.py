"""
file        : QES/Algebra/hilbert_jit_methods.py
description : JIT methods for Hilbert space operations.
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl

version     : 1.1.0
changes     : 
    - 2025-10-29: Removed deprecated methods (find_repr_int, jitted wrappers)
                  Moved to deprecated.py for reference
                  Kept only actively used methods: get_mapping, find_representative_int, get_matrix_element

DEPRECATED METHODS:
    The following methods have been moved to QES.Algebra.Hilbert.deprecated:
    - find_repr_int() -> Use SymmetryContainer.find_representative()
    - jitted_find_repr_int() -> Never used, removed
    - jitted_get_mapping() -> Never used, removed  
    - jitted_get_matrix_element() -> Never used, removed
    
    See deprecated.py for the old implementations and migration guide.
"""

import numpy as np
import numba

#! private
_INT_BINARY_REPR       = 64
_INT_HUGE_REPR         = np.iinfo(np.int64).max
_SYM_NORM_THRESHOLD    = 1e-12
from QES.general_python.common.binary import bin_search
from QES.general_python.algebra.utils import get_backend, JAX_AVAILABLE, ACTIVE_INT_TYPE, Array, maybe_jit

@property
def has_complex_symmetries(self) -> bool:
    """
    Return True if any configured symmetry encodes complex eigenvalues/phases.

    This inspects the primary and secondary symmetry groups (when present)
    and looks for non-real `eigval` attributes. It is conservative: if an
    unexpected error occurs during inspection we return True to avoid
    accidentally dropping complex information.
    """
    try:
        # No symmetry group -> no complex symmetry
        if not getattr(self, '_sym_group', None):
            return False

        for op in self._sym_group:
            eig = getattr(op, 'eigval', None)
            if eig is None:
                continue
            try:
                if not np.isreal(eig):
                    return True
                
            except Exception:
                if isinstance(eig, complex) and eig.imag != 0:
                    return True

        if getattr(self, '_sym_group_sec', None):
            for op in self._sym_group_sec:
                eig = getattr(op, 'eigval', None)
                if eig is None:
                    continue
                try:
                    if not np.isreal(eig):
                        return True
                except Exception:
                    if isinstance(eig, complex) and eig.imag != 0:
                        return True
    except Exception:
        # On error, be conservative and signal complex symmetries.
        return True

    return False


####################################################################################################
#! ACTIVE JIT METHODS (used by hilbert.py)
####################################################################################################

if True:
    
    # a) mapping - ACTIVE: Used by get_mapping() wrapper in hilbert.py
    
    @numba.njit
    def get_mapping(mapping, state):
        """
        Get the mapping of the state.
        
        Args:
            mapping (list):
                The mapping of the states.
            state (int):
                The state to get the mapping for.
        
        Returns:
            int:
                The mapping of the state.
        """
        return mapping[state] if len(mapping) > state else state

    # b) find representative - ACTIVE: Used by find_representative() wrapper in hilbert.py
    
    @numba.jit(forceobj=True)
    def find_representative_int(
                            _state                  : int,
                            _mapping                : np.ndarray,
                            _normalization          : np.ndarray,
                            _normalization_beta     : float,
                            _sym_group,
                            _reprmap                : np.ndarray = None
            ):
        """
        Find the representative of a given state in a symmetry sector.
        
        This is actively used by the find_representative() wrapper in hilbert.py.
        It handles the case where we need to find representatives and apply
        proper normalization factors for matrix elements.
        
        Args:
            _state: The state to find representative for
            _mapping: Mapping array of representatives
            _normalization: Normalization factors for each representative
            _normalization_beta: Target sector normalization
            _sym_group: Symmetry group (legacy, uses SymmetryContainer now)
            _reprmap: Precomputed representative map (optional)
            
        Returns:
            tuple: (representative_index, normalization_factor)
        """
        if _mapping is None or len(_mapping) == 0:
            return (_state, 1.0)
        
        # if the map exists, use it!
        if _reprmap is not None and len(_reprmap) > 0:
            idx, sym_eig    = _reprmap[_state, 0], _reprmap[_state, 1]
            sym_eigc        = sym_eig.conjugate() if hasattr(sym_eig, "conjugate") else sym_eig
            return (idx, _normalization[idx] / _normalization_beta * sym_eigc)
        
        mapping_size = len(_mapping)
        
        # find the representative already in the mapping (can be that the matrix element already 
        # represents the representative state)
        idx = bin_search.binary_search_numpy(_mapping, 0, mapping_size - 1, _state)
        
        if idx != bin_search._BAD_BINARY_SEARCH_STATE:
            return (idx, _normalization[idx] / _normalization_beta)
        
        # Note: In new implementation, SymmetryContainer handles representative finding
        # This path should rarely be hit
        # otherwise, we need to find the representative by acting on the state with the symmetry operators
        # and finding the one that gives the smallest value - standard procedure
        # For now, return no contribution if not in mapping
        return (_state, 0.0)

    # c) get matrix element - ACTIVE: Used by get_matrix_element() wrapper in hilbert.py

    @numba.jit(forceobj=True)
    def get_matrix_element(
            k               : int,
            new_k           : int,
            kmap            = None,
            h_conj          = False,
            _mapping        : np.ndarray = None,
            _norm           : np.ndarray = None,
            _sym_group                   = None,
            _reprmap        : np.ndarray = None
        ):
        """
        Get the matrix element of a given state using information provided from the symmetry group and 
        a given Hilbert space.
        Args:
            k (int):
                The state to get the matrix element for.
            new_k (int):
                The new state to get the matrix element for.
            kmap (int):
                The mapping of the states.
            h_conj (bool):
                A flag to indicate if the Hamiltonian is conjugated.
            _mapping (list):
                The mapping of the states.
            _norm (list):
                The normalization of the states.
            _sym_group:
                The symmetry group.
            _reprmap:
                The mapping of the representatives.
        """
        
        # check the mapping, if it is None, we need to get the mapping
        if kmap is None:
            kmap = get_mapping(_mapping, k)
        
        # try to process the elements
        if kmap == new_k:
            # the element k is already the same as new_k and obviously we 
            # and we add this at k (not kmap as it only checks the representative)
            return (new_k, k), 1
        
        # otherwise we need to check the representative of the new k
        # get the norm of the k'th element of the Hilbert space - how to return to the representative
        # of the new k
        norm = _norm[k] if _norm is not None else 1.0
        # find the representative of the new k
        idx, symeig = find_representative_int(new_k, _mapping, _norm, norm, _sym_group, _reprmap)
        return ((idx, k), symeig) if not h_conj else ((k, idx), symeig)

####################################################################################################
# END OF ACTIVE JIT METHODS
####################################################################################################
# 
# DEPRECATED METHODS REMOVED (see deprecated.py):
# - find_repr_int()                     - Moved to deprecated.py, use SymmetryContainer instead
# - jitted_find_repr_int()              - Never used, removed
# - jitted_find_representative_int()    - Never used, removed
# - jitted_get_mapping()                - Never used, removed
# - jitted_get_matrix_element()         - Never used, removed
#
####################################################################################################
