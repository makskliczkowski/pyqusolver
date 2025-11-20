"""
Sparse Lanczos driver used for ED benchmarking with observable computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np

try:  # optional SciPy dependency
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover
    sp = None  # type: ignore
    spla = None  # type: ignore


@dataclass
class LanczosConfig:
    lattice_label       : str   = "3x4"
    n_states            : int   = 5
    max_iter            : int   = 200
    tol                 : float = 1e-8
    dtype               : str   = "float64"
    compute_observables : bool  = True


@dataclass
class LanczosResult:
    energies            : np.ndarray = field(default_factory=lambda: np.empty(0))
    eigenvectors        : Optional[np.ndarray] = None
    observables         : Dict[str, Dict] = field(default_factory=dict)
    metadata            : dict = field(default_factory=dict)


class LanczosSolver:
    """
    Sparse Lanczos ED solver with observable computation from eigenvectors.
    
    Computes ground state and low-lying excited states, then evaluates:
    - Single-spin expectations <S_i^α>
    - Spin-spin correlators <S_i^α S_j^α>
    - Plaquette operators W_p
    """

    def __init__(self, hamiltonian, lattice, hilbert_space, config: Optional[LanczosConfig] = None):
        self.hamiltonian    = hamiltonian
        self.lattice        = lattice
        self.hilbert_space  = hilbert_space
        self.config         = config or LanczosConfig()

    def _to_sparse_matrix(self):
        """
        Convert the QES Hamiltonian to a SciPy sparse matrix using .matrix property.
        """
        mat = self.hamiltonian.matrix
        if mat is None:
            return None
        
        # If it's already sparse, return as is
        if sp is not None and sp.issparse(mat):
            return mat
        
        # If dense numpy array, convert to sparse
        if isinstance(mat, np.ndarray):
            if sp is not None:
                return sp.csr_matrix(mat)
            return mat
        
        return mat

    def _compute_observables_for_state(self, state_idx: int, state_vector: np.ndarray, energy: float) -> Dict:
        """
        Compute all observables for a single eigenstate.
        """
        # Use importlib to import observables module directly, bypassing __init__.py
        import importlib.util
        import sys
        from pathlib import Path
        
        # Get the path to observables.py
        obs_path = Path(__file__).parent.parent / "nqs" / "observables.py"
        spec = importlib.util.spec_from_file_location("observables_module", obs_path)
        obs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obs_module)
        
        return obs_module.compute_observables_from_eigenvector(
            state_vector=state_vector,
            energy=energy,
            hamiltonian=self.hamiltonian,
            lattice=self.lattice,
            hilbert_space=self.hilbert_space
        )

    def run(self) -> LanczosResult:
        """
        Execute the solver and compute observables for all requested states.
        
        Returns:
            LanczosResult with energies, eigenvectors, and observables for each state
        """
        mat = self._to_sparse_matrix()
        if mat is None or spla is None:
            return LanczosResult(
                energies=np.full(self.config.n_states, np.nan),
                eigenvectors=None,
                metadata={"status": "sparse_matrix_unavailable"},
            )

        n_states = min(self.config.n_states, mat.shape[0] - 1)
        vals, vecs = spla.eigsh(
            mat,
            k=n_states,
            which="SA",
            tol=self.config.tol,
            maxiter=self.config.max_iter,
        )
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        
        # Compute observables for each state
        observables = {}
        if self.config.compute_observables:
            for i in range(n_states):
                state_key = f"state_{i}"
                observables[state_key] = self._compute_observables_for_state(
                    state_idx=i,
                    state_vector=vecs[:, i],
                    energy=vals[i]
                )
        
        return LanczosResult(
            energies=np.asarray(vals),
            eigenvectors=vecs,
            observables=observables,
            metadata={
                "solver": "scipy.eigsh",
                "lattice": self.config.lattice_label,
                "n_states": n_states,
                "hilbert_dim": mat.shape[0],
            },
        )
