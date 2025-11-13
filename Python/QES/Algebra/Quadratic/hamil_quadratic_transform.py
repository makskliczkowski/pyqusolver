'''
Transformations that convert Hamiltonian quadratic matrices to other bases.

----------------
File            : Algebra/Quadratic/hamil_quadratic_transform.py
Author          : Maksymilian Kliczkowski
----------------
'''

# ---------------------------------------------------------------------------
#! Basis Transformation Handler Registration
# ---------------------------------------------------------------------------

# Register real -> k-space transformation handler
def _handler_real_to_kspace(self, enforce=False, **kwargs):
    """Handler for REAL -> KSPACE transformation."""
    return self._transform_real_to_kspace(enforce=enforce, **kwargs)

# Register k-space -> real transformation handler
def _handler_kspace_to_real(self, enforce=False, **kwargs):
    """Handler for KSPACE -> REAL transformation."""
    return self._transform_kspace_to_real(**kwargs)

# ----------------------------------------------------------------------------
#! EOF 
# ----------------------------------------------------------------------------