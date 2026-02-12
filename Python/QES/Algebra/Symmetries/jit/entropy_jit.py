"""
Entropy for symmetry-reduced Hilbert spaces
===============================================
"""

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

try:
    from QES.general_python.physics.entropy import entropy
except ImportError as e:
    raise ImportError("Required physics modules not found in general_python") from e

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace

# ---------------------------------------------


def mutual_information(
    state: np.ndarray, i: int, j: int, hilbert: "HilbertSpace", q: float = 1.0,
    rho_i: np.ndarray = None, rho_j: np.ndarray = None,
) -> float:
    """
    Compute mutual information I(i:j) = S_i + S_j - S_ij for a symmetry-reduced state.

    Parameters
    ----------
    state : np.ndarray
        Symmetry-reduced state vector.
    i, j : int
        Site indices.
    hilbert : HilbertSpace
        Hilbert space with symmetries.
    q : float
        Renyi entropy order (default 1.0 for von Neumann).
    rho_i : np.ndarray, optional
        Pre-computed reduced density matrix for site i. If None, computed internally.
    rho_j : np.ndarray, optional
        Pre-computed reduced density matrix for site j. If None, computed internally.

    Returns
    -------
    float
        Mutual information I(i:j).
    """
    try:
        from QES.general_python.physics.density_matrix import rho_spectrum
        from QES.general_python.physics.entropy import entropy

        from .density_jit import rho_symmetries
    except ImportError as e:
        raise ImportError("Required physics modules not found in general_python") from e

    # S_i: Entropy of single site i (use pre-computed rho if provided)
    if rho_i is None:
        rho_i   = rho_symmetries(state, va=[i], hilbert=hilbert)
    S_i     = entropy(rho_spectrum(rho_i), q=q)

    # S_j: Entropy of single site j (use pre-computed rho if provided)
    if rho_j is None:
        rho_j   = rho_symmetries(state, va=[j], hilbert=hilbert)
    S_j     = entropy(rho_spectrum(rho_j), q=q)

    # S_ij: Entropy of pair (i,j) — always computed fresh
    rho_ij  = rho_symmetries(state, va=[i, j], hilbert=hilbert)
    S_ij    = entropy(rho_spectrum(rho_ij), q=q)

    return S_i + S_j - S_ij


# ---------------------------------------------


def topological_entropy(
    state: np.ndarray, regions: Dict[str, np.ndarray], hilbert: "HilbertSpace", q: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Topological Entanglement Entropy (TEE) for a symmetry-reduced state.

    Parameters
    ----------
    state : np.ndarray
        Symmetry-reduced state vector.
    regions : Dict[str, np.ndarray]
        Dictionary of regions (name -> list of site indices).
        E.g., {'A': [0,1], 'B': [2,3], ...}
    hilbert : HilbertSpace
        Hilbert space with symmetries.
    q : float
        Renyi entropy order.

    Returns
    -------
    gamma : float
        Topological entanglement entropy.
    entropies : Dict[str, float]
        Individual region entropies.
    regions : Dict[str, np.ndarray]
        Input regions.
    """
    try:
        from QES.general_python.physics.density_matrix import rho_spectrum
        from QES.general_python.physics.entropy import entropy

        from .density_jit import rho_symmetries
    except ImportError as e:
        raise ImportError("Required physics modules not found in general_python") from e

    entropies = {}

    # Calculate entropy for each region
    for name, mask in regions.items():
        if len(mask) == 0:
            entropies[name] = 0.0
            continue

        # Use rho_symmetries with mask
        rho = rho_symmetries(state, va=mask, hilbert=hilbert)
        entropies[name] = entropy(rho_spectrum(rho), q=q)

    # Calculate Gamma (detect construction based on keys)
    gamma = 0.0

    # Kitaev-Preskill: γ = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    kp_keys = ["A", "B", "C", "AB", "BC", "AC", "ABC"]
    if all(k in entropies for k in kp_keys):
        gamma = (
            entropies["A"]
            + entropies["B"]
            + entropies["C"]
            - entropies["AB"]
            - entropies["BC"]
            - entropies["AC"]
            + entropies["ABC"]
        )

    # Levin-Wen: γ = S_inner + S_outer - S_inner_outer
    elif all(k in entropies for k in ["inner", "outer", "inner_outer"]):
        gamma = entropies["inner"] + entropies["outer"] - entropies["inner_outer"]

    return {"gamma": gamma, "entropies": entropies, "regions": regions}


# ---------------------------------------------
# End of file
# ---------------------------------------------
