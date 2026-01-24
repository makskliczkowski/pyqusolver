"""
Global cache for Hamiltonian and Operator matrices.

This module provides a caching mechanism for heavy matrix constructions,
allowing reuse of operators across different instances if the configuration matches.
"""

from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hamil import Hamiltonian

# Global cache storage
# Key: (system_size, symmetry_sector_info, dtype, backend_name, is_sparse, operator_signature)
# Value: The constructed matrix
_HAMILTONIAN_CACHE: Dict[Tuple, Any] = {}

# Use weakrefs? Maybe not for matrices if we want them to persist across re-creation of Hamiltonian objects.
# But we should be careful about memory. The user asked for caching, implying persistence.


def get_matrix_from_cache(key: Tuple) -> Optional[Any]:
    """Retrieve a matrix from the global cache."""
    return _HAMILTONIAN_CACHE.get(key)


def store_matrix_in_cache(key: Tuple, matrix: Any):
    """Store a matrix in the global cache."""
    _HAMILTONIAN_CACHE[key] = matrix


def clear_cache():
    """Clear the global Hamiltonian cache."""
    _HAMILTONIAN_CACHE.clear()


def generate_cache_key(hamiltonian: "Hamiltonian") -> Tuple:
    """
    Generate a unique key for the Hamiltonian configuration.

    Includes:
    - ns (system size)
    - Hilbert space signature (symmetry sector, etc.)
    - Backend
    - Dtype
    - Sparse flag
    - Operator signature (based on added terms)
    """
    # 1. System size
    ns = hamiltonian.ns

    # 2. Hilbert space signature
    # Assuming hilbert_space has a representation or we use its str/repr if unique enough.
    # Better: use properties.
    if hamiltonian.hilbert_space:
        # TODO: Need a robust way to key Hilbert space configuration
        # For now, using str(hilbert_space) + basis info
        hs_key = (str(hamiltonian.hilbert_space),)
        # Ideally HilbertSpace should have a .signature() method
    else:
        hs_key = ("NoHilbert",)

    # 3. Backend & Dtype
    backend_str = (
        hamiltonian.backend if isinstance(hamiltonian.backend, str) else str(hamiltonian.backend)
    )
    dtype_str = str(hamiltonian.dtype)
    # is_sparse is a method in GeneralMatrix, or use .sparse property
    is_sparse = hamiltonian.sparse

    # 4. Operator terms signature
    # This is tricky. We need to capture the exact operator definition.
    # Hamiltonian stores terms in _ops_mod_sites, etc.
    # We can serialize them or use a hash.
    # For now, let's assume if the user rebuilt the Hamiltonian object with same config,
    # the operator structure might be same if they used same add_term calls.
    # But if they changed terms, it's different.
    # A simple way: use a hash of the string representation of all terms.
    # Or rely on the user to manually clear cache if they change definitions but keep same params (unlikely).

    # Let's try to capture terms.
    # Hamiltonian inherits from BasisAwareOperator -> SpecialOperator -> Operator
    # Terms are in self._ops_mod_sites etc.
    # It might be expensive to hash all terms every time.

    # For the scope of this task "Lazy operator/Hamiltonian construction",
    # maybe we just cache based on structural parameters if the terms are standard?
    # But "Hamiltonian" usually implies specific terms.

    # The requirement says "keyed by (system size, symmetry sector, dtype, backend)".
    # It doesn't explicitly mention the actual operator terms.
    # BUT caching a Hamiltonian matrix that has different terms but same size/symmetry would be WRONG.
    # So we MUST include the terms in the key.

    # Let's construct a tuple of sorted string representations of terms.
    # This might be slow for many terms.
    # We will defer this implementation detail to the Hamiltonian class to provide a signature.

    # Returning a placeholder for now, actual implementation will rely on hamiltonian.signature property
    if hasattr(hamiltonian, "signature"):
        op_sig = hamiltonian.signature
    else:
        # Fallback: object id (no caching across instances) or some basic info
        op_sig = id(hamiltonian)

    return (ns, hs_key, backend_str, dtype_str, is_sparse, op_sig)
