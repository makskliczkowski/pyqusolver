"""
file        : QES/Algebra/Hilbert/deprecated.py
description : Deprecated methods from Hilbert space operations.
              These methods are preserved for reference but should not be used.
              All functionality has been replaced by SymmetryContainer-based implementations.
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl

version     : 1.0.0
created     : 2025-10-29

DEPRECATION NOTICE:
    All methods in this file are deprecated as of October 2025.
    They have been replaced by character-based normalization and 
    SymmetryContainer API methods.
    
    DO NOT USE THESE METHODS IN NEW CODE.
    
    Migration guide:
    - _find_sym_norm_int()          -> SymmetryContainer.compute_normalization()
    - _find_repr_int()              -> SymmetryContainer.find_representative()
    - _register_modular_symmetry()  -> Use SymmetryContainer directly
    - find_repr_int()               -> Use SymmetryContainer.find_representative()

    For JIT methods, use:
    - find_representative_int() (still active)
    - get_matrix_element() (still active)
    - get_mapping() (still active)
"""

import warnings
import numpy as np
from typing import Tuple, Optional, List

####################################################################################################
# DEPRECATED HILBERT SPACE METHODS (removed from hilbert.py October 2025)
####################################################################################################

def _deprecated_warning(old_method: str, new_method: str):
    """Helper to emit deprecation warnings."""
    warnings.warn(
        f"{old_method} is deprecated and will be removed in future versions. "
        f"Use {new_method} instead.",
        DeprecationWarning,
        stacklevel=3
    )

####################################################################################################
# LEGACY SYMMETRY REGISTRATION
####################################################################################################

def _register_modular_symmetry(self, symmetry_container):
    """
    DEPRECATED: Register symmetry container in the old modular format.
    
    This method is preserved for reference only. It registered symmetries using
    the old _sym_group_modular structure which has been replaced by direct
    SymmetryContainer usage.
    
    USE INSTEAD: Work directly with SymmetryContainer object.
    
    Original functionality:
    - Built symmetry group from generators
    - Stored in _sym_group_modular
    - Used phase-based normalization (incorrect for momentum sectors)
    """
    _deprecated_warning("_register_modular_symmetry", "SymmetryContainer API")
    
    # Store symmetry container reference
    if symmetry_container is None:
        return
        
    self._sym_container = symmetry_container
    
    # Build symmetry group from generators
    from itertools import product
    
    generators = symmetry_container.generators
    if not generators:
        self._sym_group_modular = [()]  # Identity only
        return
    
    # Get the order of each generator
    orders = []
    for gen in generators:
        order = getattr(gen, 'order', None)
        if order is None:
            # For discrete symmetries, assume order 2
            order = 2
        orders.append(order)
    
    # Generate all combinations
    group_elements = []
    for powers in product(*[range(o) for o in orders]):
        element = []
        for gen, power in zip(generators, powers):
            if power > 0:
                element.extend([gen] * power)
        group_elements.append(tuple(element))
    
    self._sym_group_modular = group_elements

####################################################################################################
# LEGACY NORMALIZATION METHODS (phase-based, incorrect for momentum)
####################################################################################################

def _find_sym_norm_int(self, state: int) -> float:
    """
    DEPRECATED: Old phase-based normalization (incorrect for momentum sectors).
    
    This method computed normalization by summing PHASES of symmetric partners,
    which is mathematically incorrect for momentum sectors. The correct approach
    is to use CHARACTERS in the projection formula.
    
    USE INSTEAD: SymmetryContainer.compute_normalization(state)
    
    Why it was wrong:
    - Summed exp(iφ) phases instead of characters χ_k(g)
    - Did not respect momentum quantum number k
    - All k sectors got same normalization values
    - Led to states appearing in wrong sectors
    
    Correct formula:
        N = sqrt(sum_{g in G} χ_k(g)^* <state|g|state>)
    
    where χ_k(T^n) = exp(2πi*k*n/L) for translations.
    """
    _deprecated_warning("_find_sym_norm_int", "SymmetryContainer.compute_normalization")
    
    if not hasattr(self, '_sym_group_modular') or self._sym_group_modular is None:
        return 1.0
    
    # OLD (WRONG) METHOD: Sum phases
    _sum = 0.0
    for op_tuple in self._sym_group_modular:
        _st = state
        _retval = 1.0
        
        if len(op_tuple) == 0:
            _st = state
            _retval = 1.0
        else:
            for op in op_tuple:
                _st, phase = op(_st)
                _retval *= phase
        
        # This is the ERROR: summing phases without character projection
        _sum += abs(_retval) ** 2
    
    return np.sqrt(_sum) if _sum > 1e-12 else 0.0

def _find_sym_norm_base(self, state) -> float:
    """
    DEPRECATED: Stub for base representation.
    USE INSTEAD: SymmetryContainer.compute_normalization(state)
    """
    _deprecated_warning("_find_sym_norm_base", "SymmetryContainer.compute_normalization")
    return self._find_sym_norm_int(int(state))

####################################################################################################
# LEGACY REPRESENTATIVE FINDING (without proper character handling)
####################################################################################################

def _find_repr_int(self, state: int) -> Tuple[int, float]:
    """
    DEPRECATED: Old representative finding without character-based normalization.
    
    This method found representatives but didn't properly handle momentum sectors
    because it used _find_sym_norm_int which had incorrect normalization.
    
    USE INSTEAD: SymmetryContainer.find_representative(state)
    
    The new method:
    - Finds minimum state in orbit (representative) ✓ (this was correct)
    - Uses character-based normalization ✓ (this was missing)
    - Properly filters by momentum quantum number ✓ (this was missing)
    """
    _deprecated_warning("_find_repr_int", "SymmetryContainer.find_representative")
    
    if not hasattr(self, '_sym_group_modular') or self._sym_group_modular is None:
        return state, 1.0
    
    _sec = np.iinfo(np.int64).max
    _val = 1.0
    
    for op_tuple in self._sym_group_modular:
        _st = state
        _retval = 1.0
        
        if len(op_tuple) == 0:
            _st = state
            _retval = 1.0
        else:
            for op in op_tuple:
                _st, phase = op(_st)
                _retval *= phase
        
        if _st < _sec:
            _sec = _st
            _val = _retval
    
    # Return representative and phase (not normalization!)
    return _sec, _val

def _find_repr_base(self, state) -> Tuple[int, float]:
    """
    DEPRECATED: Stub for base representation.
    USE INSTEAD: SymmetryContainer.find_representative(state)
    """
    _deprecated_warning("_find_repr_base", "SymmetryContainer.find_representative")
    return self._find_repr_int(int(state))

####################################################################################################
# DEPRECATED JIT METHODS (removed from hilbert_jit_methods.py October 2025)
####################################################################################################

def find_repr_int_DEPRECATED(state, _sym_group, _reprmap: Optional[np.ndarray] = None):
    """
    DEPRECATED: JIT method for finding representatives (unused after migration).
    
    This was used by the old _find_repr_int method but is no longer needed
    since SymmetryContainer handles representative finding internally.
    
    USE INSTEAD: SymmetryContainer.find_representative(state)
    
    Note: This was removed from hilbert_jit_methods.py because it was never
    called after the SymmetryContainer migration.
    """
    _deprecated_warning("find_repr_int", "SymmetryContainer.find_representative")
    
    if _reprmap is not None and isinstance(_reprmap, np.ndarray) and len(_reprmap) > 0:
        idx = _reprmap[state, 0]
        sym_eig = _reprmap[state, 1]
        return idx, sym_eig
    
    if _sym_group is None or len(_sym_group) == 0:
        return state, 1.0

    _sec = np.iinfo(np.int64).max
    _val = 1.0
    
    for op_tuple in _sym_group:
        _st = state
        _retval = 1.0
        
        if len(op_tuple) == 0:
            _st = state
            _retval = 1.0
        else:
            for op in op_tuple:
                _st, phase = op(_st)
                _retval *= phase
        
        if _st < _sec:
            _sec = _st
            _val = _retval
    
    return _sec, _val

####################################################################################################
# DEPRECATED JITTED WRAPPERS (never used)
####################################################################################################

def jitted_find_repr_int_DEPRECATED(state, _sym_group, _reprmap=None):
    """
    DEPRECATED: Numba-jitted wrapper for find_repr_int (never used).
    Removed from imports October 2025.
    """
    _deprecated_warning("jitted_find_repr_int", "SymmetryContainer.find_representative")
    return find_repr_int_DEPRECATED(state, _sym_group, _reprmap)

def jitted_get_mapping_DEPRECATED(mapping, state):
    """
    DEPRECATED: Numba-jitted wrapper for get_mapping (never used).
    Removed from imports October 2025.
    
    Note: get_mapping() itself is still active and used.
    Only the jitted wrapper was unused.
    """
    _deprecated_warning("jitted_get_mapping", "get_mapping (non-jitted version)")
    return mapping[state] if len(mapping) > state else state

def jitted_get_matrix_element_DEPRECATED(k, new_k, kmap=None, h_conj=False, 
                                         _mapping=None, _norm=None, 
                                         _sym_group=None, _reprmap=None):
    """
    DEPRECATED: Numba-jitted wrapper for get_matrix_element (never used).
    Removed from imports October 2025.
    
    Note: get_matrix_element() itself is still active and used.
    Only the jitted wrapper was unused.
    """
    _deprecated_warning("jitted_get_matrix_element", "get_matrix_element (non-jitted version)")
    # Would call the actual get_matrix_element, but it's still active in hilbert_jit_methods.py
    raise NotImplementedError("This jitted wrapper is deprecated. Use get_matrix_element directly.")

####################################################################################################
# MIGRATION GUIDE
####################################################################################################

MIGRATION_GUIDE = """
MIGRATION GUIDE: Old Methods -> New SymmetryContainer API
==========================================================

1. NORMALIZATION:
   OLD: hilbert._find_sym_norm_int(state)
   NEW: hilbert._sym_container.compute_normalization(state)
   
   Key difference: New method uses character-based projection formula
   N = sqrt(sum_{g in G} χ_k(g)^* <state|g|state>)

2. REPRESENTATIVE FINDING:
   OLD: hilbert._find_repr_int(state)
   NEW: hilbert._sym_container.find_representative(state)
   
   Both find minimum state in orbit, but new method properly integrates
   with character-based normalization.

3. SYMMETRY REGISTRATION:
   OLD: hilbert._register_modular_symmetry(sym_container)
   NEW: Pass SymmetryContainer to Hilbert constructor
   
   The container is now used directly, no intermediate registration needed.

4. MOMENTUM SUPERPOSITION:
   OLD: Used _sym_group_modular for group elements
   NEW: Uses _sym_container.generators directly
   
   Example:
   OLD: for element in hilbert._sym_group_modular: ...
   NEW: for gen in hilbert._sym_container.generators: ...

5. JIT METHODS:
   REMOVED: find_repr_int, jitted_* wrappers
   KEPT: get_mapping, find_representative_int, get_matrix_element
   
   The kept methods are still actively used by the high-level API.

WHY THE CHANGE?
===============
The old phase-based normalization was mathematically incorrect for momentum
sectors. It summed phases instead of using character projection, causing:
- States to appear in wrong momentum sectors
- Incorrect normalization values
- Sectors not summing to correct Hilbert space dimension

The new character-based approach:
- Uses proper group representation theory
- Characters χ_k(T^n) = exp(2πi*k*n/L) for momentum k
- Ensures states only appear in correct k sectors
- Normalization values match theoretical predictions
- Sectors sum to correct dimension (e.g., 16 for L=4 spin-1/2)

VERIFICATION:
=============
After migration, verify:
1. Sum of sector dimensions equals full Hilbert space
2. States appear in expected sectors based on orbit period
3. Normalization values match sqrt(|G|/|Stabilizer|)
4. Representative finding gives consistent results

See test/test_momentum_integration.py for examples.
"""

def print_migration_guide():
    """Print the migration guide for reference."""
    print(MIGRATION_GUIDE)

####################################################################################################
