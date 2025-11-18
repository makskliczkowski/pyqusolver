"""
Lazy operator loader module.

Provides convenient access to operator modules without requiring explicit imports.
Operators are loaded on-demand when first accessed.

Usage:
    # Via Hamiltonian
    ops         = hamil.operators
    sig_x       = ops.sig_x(ns=4, sites=[0, 1])

    # Via HilbertSpace
    ops         = hilbert.operators
    c_dag       = ops.c_dag(ns=4, sites=[0])
    
    # Get matrix representation
    sig_x_mat   = sig_x.matrix

------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : November 2025
Email           : maxgrom97@gmail.com
------------------------------------------------------------------------
"""

from typing import Callable, Optional, TYPE_CHECKING
from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes

if TYPE_CHECKING:
    from QES.Algebra.Operator import Operator
    from QES.Algebra.Operator import operators_spin
    from QES.Algebra.Operator import operators_spinless_fermions
    from QES.Algebra.Operator import operators_hardcore

class OperatorModule:
    """
    Lazy loader for operator modules based on LocalSpace type.
    
    Provides attribute access to operator factory functions without requiring
    upfront imports. Modules are imported only when first accessed.
    """
    
    def __init__(self, local_space_type: Optional[LocalSpaceTypes] = None):
        """
        Initialize the operator module loader.
        
        Parameters
        ----------
        local_space_type : LocalSpaceTypes, optional
            The type of local space to determine which operators to load.
            If None, defaults to SPIN_HALF.
        """
        self._local_space_type  = local_space_type or LocalSpaceTypes.SPIN_1_2
        self._spin_module       = None
        self._fermion_module    = None
        self._hardcore_module   = None
        self._anyon_module      = None

    def _load_spin_operators(self):
        """Lazy load spin operator module."""
        if self._spin_module is None:
            from QES.Algebra.Operator import operators_spin
            self._spin_module = operators_spin
        return self._spin_module
    
    def _load_fermion_operators(self):
        """Lazy load spinless fermion operator module."""
        if self._fermion_module is None:
            from QES.Algebra.Operator import operators_spinless_fermions
            self._fermion_module = operators_spinless_fermions
        return self._fermion_module
    
    def _load_hardcore_operators(self):
        """Lazy load hardcore boson operator module."""
        if self._hardcore_module is None:
            from QES.Algebra.Operator import operators_hardcore
            self._hardcore_module = operators_hardcore
        return self._hardcore_module
    
    def _load_anyon_operators(self):
        """Lazy load anyon operator module."""
        if self._anyon_module is None:
            from QES.Algebra.Operator import operators_anyon
            self._anyon_module = operators_anyon
        return self._anyon_module
    
    def __getattr__(self, name: str) -> Callable[..., 'Operator']:
        """
        Dynamically load and return operator factory functions.
        
        Parameters
        ----------
        name : str
            Name of the operator factory function
            
        Returns
        -------
        Callable
            The operator factory function
            
        Raises
        ------
        AttributeError
            If the operator is not found in any loaded module
        """
        # Determine which module to load based on local space type and operator name
        if self._local_space_type in (LocalSpaceTypes.SPIN_1_2, LocalSpaceTypes.SPIN_1):
            # Spin operators: sig_x, sig_y, sig_z, sig_plus, sig_minus, etc.
            if name.startswith('sig_') or name.startswith('sigma_'):
                module = self._load_spin_operators()
                if hasattr(module, name):
                    return getattr(module, name)
        
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            # Fermion operators: c_dag, c_ann, n_op, etc.
            if name in ('c_dag', 'c', 'c_ann', 'n', 'n_op', 'fermion_number'):
                module = self._load_fermion_operators()
                if hasattr(module, name):
                    return getattr(module, name)
        
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            # Boson operators (includes hardcore bosons)
            if name in ('b_dag', 'b', 'b_ann', 'n', 'n_op'):
                module = self._load_hardcore_operators()
                if hasattr(module, name):
                    return getattr(module, name)
        
        # Fallback: try all modules in order
        for loader in [self._load_spin_operators, 
                       self._load_fermion_operators,
                       self._load_hardcore_operators]:
            try:
                module = loader()
                if hasattr(module, name):
                    return getattr(module, name)
            except ImportError:
                continue
        
        # If not found, raise AttributeError
        raise AttributeError(
            f"Operator '{name}' not found for LocalSpace type {self._local_space_type}. "
            f"Available operators depend on the local space type:\n"
            f"  - SPIN_1_2/SPIN_1   : sig_x, sig_y, sig_z, sig_plus, sig_minus, sig_z_total\n"
            f"  - SPINLESS_FERMIONS : c_dag, c (or c_ann), n (or n_op)\n"
            f"  - BOSONS            : b_dag, b (or b_ann), n (or n_op)"
        )
    
    def __dir__(self):
        """
        Return list of available operator names.
        
        Returns
        -------
        list
            List of available operator factory function names
        """
        operators = []
        
        if self._local_space_type in (LocalSpaceTypes.SPIN_1_2, LocalSpaceTypes.SPIN_1):
            operators.extend(['sig_x', 'sig_y', 'sig_z', 'sig_plus', 'sig_minus', 
                'sig_z_total', 'sigma_x', 'sigma_y', 'sigma_z'])
        
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            operators.extend(['c_dag', 'c', 'c_ann', 'n', 'n_op', 'fermion_number'])
        
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            operators.extend(['b_dag', 'b', 'b_ann', 'n', 'n_op'])
        
        return operators
    
    # ----------------------------
    # Utility method to print help
    # ----------------------------

    @staticmethod
    def overlap(a, O, b, backend: str = 'default'):
        """
        Compute the overlap <a|O|b> using the specified backend.
        """
        from QES.Algebra.backends import overlap
        return overlap(a, O, b, backend=backend)
    
    @staticmethod
    def kron(A, B, backend: str = 'default'):
        """
        Compute the Kronecker product A ⊗ B using the specified backend.
        """
        from QES.Algebra.backends import kron
        return kron(A, B, backend=backend)
    
    @staticmethod
    def outer(a, b, backend: str = 'default'):
        """
        Compute the outer product |a><b| using the specified backend.
        """
        from QES.Algebra.backends import outer
        return outer(a, b, backend=backend)
    
    def help(self):
        """
        Print help message showing available operators for current local space type.
        """
        print(f"Operator Module for LocalSpace: {self._local_space_type}")
        print("=" * 70)
        print("\nAvailable operators:")
        
        if self._local_space_type in (LocalSpaceTypes.SPIN_1_2, LocalSpaceTypes.SPIN_1):
            print("\n  Spin Operators:")
            print("    sig_x        - Pauli X (sigma_x) operator")
            print("    sig_y        - Pauli Y (sigma_y) operator") 
            print("    sig_z        - Pauli Z (sigma_z) operator")
            print("    sig_plus     - Raising operator (sigma_+)")
            print("    sig_minus    - Lowering operator (sigma_-)")
            print("    sig_z_total  - Total magnetization (sigma_i sigma_z^i)")
            
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            print("\n  Fermionic Operators:")
            print("    c_dag        - Creation operator (c^+)")
            print("    c, c_ann     - Annihilation operator (c)")
            print("    n, n_op      - Number operator (n = c^+c)")
            
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            print("\n  Boson Operators:")
            print("    b_dag        - Creation operator (b^+)")
            print("    b, b_ann     - Annihilation operator (b)")
            print("    n, n_op      - Number operator (n = b^+b)")
        
        print("\nUsage:")
        print("  op = operators.sig_x(ns=4, sites=[0, 1])")
        print("  matrix = op.matrix")
        print("=" * 70)


def get_operator_module(local_space_type: Optional[LocalSpaceTypes] = None) -> OperatorModule:
    """
    Get an OperatorModule for the specified local space type.
    
    Parameters
    ----------
    local_space_type : LocalSpaceTypes, optional
        The type of local space. If None, defaults to SPIN_1_2.
        
    Returns
    -------
    OperatorModule
        Lazy loader for operator factory functions
        
    Examples
    --------
    >>> ops     = get_operator_module(LocalSpaceTypes.SPIN_1_2)
    >>> sig_x   = ops.sig_x(ns=4, sites=[0])
    """
    return OperatorModule(local_space_type)

# ============================================================================
#! END OF FILE
# ============================================================================
