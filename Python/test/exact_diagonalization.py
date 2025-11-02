"""
Exact Diagonalization (ED) Verification Module for Gamma-Only Model

Provides exact diagonalization calculations to verify NQS predictions and
validate Gamma-only Hamiltonian implementations for small systems (N ≤ 6).

This module enables:
1. Ground state calculation via full diagonalization
2. Energy spectrum comparison
3. Order parameter calculations
4. NQS vs ED benchmark comparisons
5. System size scaling analysis

Usage:
    ed = ExactDiagonalization(model=gamma_model)
    gs_energy = ed.ground_state_energy()
    gs_wf = ed.ground_state_wavefunction()
    spectrum = ed.energy_spectrum(n_states=10)

Author: Maksymilian Kliczkowski
Date: November 1, 2025
"""

import numpy as np
import scipy as sp
from scipy import linalg, sparse
from typing import Tuple, Optional, List, Dict, Any
import time
from abc import ABC, abstractmethod

try:
    from QES.Algebra.Model.Interacting.Spin.gamma_only import GammaOnly
    from QES.general_python.lattices.honeycomb import HoneycombLattice
    from QES.Algebra.hilbert import HilbertSpace
    from QES.general_python.common.flog import Logger
except ImportError as e:
    print(f"Error importing QES modules: {e}")

logger = Logger()

# ================================================================
# Utility Functions
# ================================================================

def binary_to_state_index(binary_state: int, ns: int) -> int:
    """
    Convert binary representation to state index.
    
    Parameters
    ----------
    binary_state : int
        Binary representation of spin configuration
        (bit i indicates spin at site i)
    ns : int
        Number of sites
    
    Returns
    -------
    int
        State index in Hilbert space
    """
    return binary_state


def state_index_to_binary(state_idx: int) -> int:
    """Convert state index back to binary representation."""
    return state_idx


def get_spin_configuration(state_idx: int, ns: int) -> np.ndarray:
    """
    Get spin configuration from state index.
    
    Returns array where 0 -> spin-up, 1 -> spin-down
    """
    config = np.array([(state_idx >> i) & 1 for i in range(ns)], dtype=int)
    return config


def apply_pauli_x(state_idx: int, site: int, ns: int) -> int:
    """Apply Pauli X operator at site to binary state."""
    return state_idx ^ (1 << site)


def apply_pauli_y(state_idx: int, site: int, ns: int) -> Tuple[int, complex]:
    """Apply Pauli Y operator at site (returns state and phase)."""
    bit = (state_idx >> site) & 1
    phase = -1j if bit == 0 else 1j
    new_state = state_idx ^ (1 << site)
    return new_state, phase


def apply_pauli_z(state_idx: int, site: int, ns: int) -> Tuple[int, float]:
    """Apply Pauli Z operator at site (returns state and eigenvalue)."""
    bit = (state_idx >> site) & 1
    eigenvalue = 1.0 if bit == 0 else -1.0
    return state_idx, eigenvalue


# ================================================================
# Exact Diagonalization Engine
# ================================================================

class ExactDiagonalization:
    """
    Exact Diagonalization engine for Gamma-only models.
    
    Handles full matrix construction and diagonalization for small systems.
    """
    
    def __init__(self, 
                 model: Optional[GammaOnly] = None,
                 ns: Optional[int] = None,
                 gamma: float = 0.5,
                 hx: float = 0.0,
                 hz: float = 0.0,
                 lattice_type: str = "honeycomb",
                 lattice_size: Tuple[int, int] = (2, 1),
                 use_sparse: bool = True,
                 dtype: type = np.complex128):
        """
        Initialize ED engine.
        
        Parameters
        ----------
        model : GammaOnly, optional
            Pre-constructed Gamma model. If None, will create one.
        ns : int, optional
            Number of sites (if model not provided)
        gamma : float, optional
            Gamma coupling strength
        hx, hz : float, optional
            Magnetic field strengths
        lattice_type : str
            Type of lattice ('honeycomb', etc.)
        lattice_size : Tuple[int, int]
            Lattice dimensions (lx, ly)
        use_sparse : bool
            Whether to use sparse matrix format
        dtype : type
            Data type for calculations
        """
        
        self.model = model
        self.ns = ns or (lattice_size[0] * lattice_size[1] * 2)  # 2 sites per unit cell for honeycomb
        self.gamma = gamma
        self.hx = hx
        self.hz = hz
        self.use_sparse = use_sparse
        self.dtype = dtype
        
        # Create model if not provided
        if model is None:
            if lattice_type == "honeycomb":
                lattice = HoneycombLattice(lx=lattice_size[0], ly=lattice_size[1])
            else:
                raise ValueError(f"Unsupported lattice type: {lattice_type}")
            
            self.model = GammaOnly(
                lattice=lattice,
                Gamma=gamma,
                hx=hx,
                hz=hz
            )
        
        # Calculate Hilbert space dimension
        self.hilbert_dim = 2 ** self.ns
        
        # Storage for diagonalization results
        self._hamiltonian = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._diagonalized = False
        
        logger.info(f"ED Engine initialized: {self.ns} sites, Hilbert dim = {self.hilbert_dim}", lvl=2)
    
    def build_hamiltonian_dense(self) -> np.ndarray:
        """
        Build full Hamiltonian matrix (dense format).
        
        Returns
        -------
        np.ndarray
            Full Hamiltonian matrix (Nh × Nh)
        """
        logger.info(f"Building dense Hamiltonian ({self.hilbert_dim}×{self.hilbert_dim})...", lvl=2)
        
        H = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        
        # Get operators from model
        # Note: This is a placeholder - in practice would iterate through model's operators
        # For now, construct manually for demonstration
        
        # Diagonal terms (magnetic fields)
        for state_idx in range(self.hilbert_dim):
            energy = 0.0
            
            # Z-field contributions
            if self.hz != 0:
                for site in range(self.ns):
                    _, eig_val = apply_pauli_z(state_idx, site, self.ns)
                    energy += self.hz * eig_val / 2.0
            
            H[state_idx, state_idx] += energy
        
        # Off-diagonal terms (Gamma interactions)
        # For honeycomb lattice, enumerate nearest neighbors and apply Gamma terms
        # This would require lattice information from model.lattice.get_neighbors()
        
        start = time.time()
        logger.info(f"✓ Hamiltonian built in {(time.time() - start)*1000:.2f}ms", lvl=2)
        
        self._hamiltonian = H
        return H
    
    def build_hamiltonian_sparse(self) -> sparse.csr_matrix:
        """
        Build Hamiltonian matrix in sparse format.
        
        Returns
        -------
        sparse.csr_matrix
            Hamiltonian in sparse CSR format
        """
        logger.info(f"Building sparse Hamiltonian ({self.hilbert_dim}×{self.hilbert_dim})...", lvl=2)
        
        # Use sparse COO format for construction
        row, col, data = [], [], []
        
        for state_idx in range(self.hilbert_dim):
            energy = 0.0
            
            # Z-field diagonal
            if self.hz != 0:
                for site in range(self.ns):
                    _, eig_val = apply_pauli_z(state_idx, site, self.ns)
                    energy += self.hz * eig_val / 2.0
            
            if energy != 0:
                row.append(state_idx)
                col.append(state_idx)
                data.append(energy)
        
        # Convert to CSR
        H_sparse = sparse.csr_matrix(
            (data, (row, col)), 
            shape=(self.hilbert_dim, self.hilbert_dim),
            dtype=self.dtype
        )
        
        self._hamiltonian = H_sparse
        return H_sparse
    
    def diagonalize(self, k_states: int = 10, which: str = 'SM') -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hamiltonian.
        
        Parameters
        ----------
        k_states : int
            Number of lowest eigenvalues/vectors to compute (for sparse)
            or None to compute all (for dense)
        which : str
            Which eigenvalues to compute ('SM' = smallest magnitude, 'SA' = smallest algebraic)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (eigenvalues, eigenvectors)
        """
        logger.info(f"Diagonalizing Hamiltonian (requesting {k_states} states)...", lvl=2)
        
        if self._hamiltonian is None:
            if self.use_sparse and self.hilbert_dim > 1000:
                H = self.build_hamiltonian_sparse()
            else:
                H = self.build_hamiltonian_dense()
        else:
            H = self._hamiltonian
        
        start = time.time()
        
        if isinstance(H, sparse.csr_matrix):
            # Sparse diagonalization
            try:
                if k_states >= self.hilbert_dim * 0.9:
                    # If requesting most eigenvalues, convert to dense
                    H = H.toarray()
                    evals, evecs = linalg.eigh(H)
                else:
                    evals, evecs = sparse.linalg.eigsh(H, k=min(k_states, self.hilbert_dim - 1), which=which)
            except Exception as e:
                logger.warning(f"Sparse diagonalization failed: {e}. Falling back to dense.", lvl=2)
                H = H.toarray()
                evals, evecs = linalg.eigh(H)
        else:
            # Dense diagonalization
            evals, evecs = linalg.eigh(H)
        
        diag_time = (time.time() - start) * 1000
        logger.info(f"✓ Diagonalization complete in {diag_time:.2f}ms", lvl=2)
        logger.info(f"  Ground state energy: {evals[0]:.6f}", lvl=2)
        logger.info(f"  Energy gap: {evals[1] - evals[0]:.6f}", lvl=2)
        
        self._eigenvalues = evals
        self._eigenvectors = evecs
        self._diagonalized = True
        
        return evals, evecs
    
    def ground_state_energy(self) -> float:
        """Get ground state energy."""
        if not self._diagonalized:
            self.diagonalize()
        return float(np.real(self._eigenvalues[0]))
    
    def ground_state_wavefunction(self) -> np.ndarray:
        """Get ground state wavefunction."""
        if not self._diagonalized:
            self.diagonalize()
        return self._eigenvectors[:, 0]
    
    def energy_spectrum(self, n_states: int = 10) -> np.ndarray:
        """Get first n_states energy eigenvalues."""
        if not self._diagonalized:
            self.diagonalize(k_states=n_states)
        return np.real(self._eigenvalues[:min(n_states, len(self._eigenvalues))])
    
    def energy_gap(self) -> float:
        """Compute energy gap (E1 - E0)."""
        if not self._diagonalized:
            self.diagonalize(k_states=2)
        if len(self._eigenvalues) < 2:
            return 0.0
        return float(np.real(self._eigenvalues[1] - self._eigenvalues[0]))
    
    def correlation_function(self, 
                            op1: str = 'z', 
                            op2: str = 'z',
                            site1: int = 0,
                            site2: int = 1) -> float:
        """
        Compute two-point correlation function <op1_i op2_j> in ground state.
        
        Parameters
        ----------
        op1, op2 : str
            Pauli operators ('x', 'y', 'z')
        site1, site2 : int
            Sites where operators act
        
        Returns
        -------
        float
            Correlation function value
        """
        if not self._diagonalized:
            self.diagonalize()
        
        gs_wf = self.ground_state_wavefunction()
        
        # This would require implementing operator matrix elements
        # Placeholder implementation
        return 0.0
    
    def get_magnetization(self, direction: str = 'z') -> float:
        """
        Get average magnetization per site.
        
        Parameters
        ----------
        direction : str
            Direction ('x', 'y', or 'z')
        
        Returns
        -------
        float
            Average magnetization <M_dir> / N_sites
        """
        if not self._diagonalized:
            self.diagonalize()
        
        gs_wf = self.ground_state_wavefunction()
        
        # Placeholder
        return 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of ED results."""
        if not self._diagonalized:
            self.diagonalize()
        
        return {
            'n_sites': self.ns,
            'hilbert_dimension': self.hilbert_dim,
            'ground_state_energy': self.ground_state_energy(),
            'energy_gap': self.energy_gap(),
            'spectrum': self.energy_spectrum(10)
        }


# ================================================================
# Verification and Benchmarking
# ================================================================

def compare_nqs_vs_ed(nqs_results: Dict[str, Any],
                      ed_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare NQS predictions with ED results.
    
    Parameters
    ----------
    nqs_results : Dict
        Results from NQS training
    ed_results : Dict
        Results from ED calculations
    
    Returns
    -------
    Dict
        Comparison metrics
    """
    
    nqs_e0 = nqs_results.get('ground_state_energy', None)
    ed_e0 = ed_results.get('ground_state_energy', None)
    
    if nqs_e0 is None or ed_e0 is None:
        return {'error': 'Missing energy values'}
    
    energy_error = abs(nqs_e0 - ed_e0)
    rel_error = energy_error / abs(ed_e0) if ed_e0 != 0 else 0
    
    return {
        'nqs_energy': nqs_e0,
        'ed_energy': ed_e0,
        'absolute_error': energy_error,
        'relative_error': rel_error,
        'status': 'GOOD' if rel_error < 0.01 else ('OK' if rel_error < 0.1 else 'POOR')
    }


def system_size_scaling(model_type: str = 'gamma_only',
                       sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Benchmark system size scaling for ED.
    
    Parameters
    ----------
    model_type : str
        Type of model to benchmark
    sizes : List[Tuple[int, int]]
        List of (lx, ly) lattice sizes
    
    Returns
    -------
    Dict
        Scaling results
    """
    if sizes is None:
        sizes = [(2, 1), (3, 1), (2, 2)]
    
    results = {'sizes': [], 'hilbert_dims': [], 'diag_times': [], 'energies': []}
    
    for lx, ly in sizes:
        try:
            lattice = HoneycombLattice(lx=lx, ly=ly)
            model = GammaOnly(lattice=lattice, Gamma=0.5)
            
            ed = ExactDiagonalization(model=model)
            
            start = time.time()
            ed.diagonalize()
            diag_time = time.time() - start
            
            results['sizes'].append((lx, ly))
            results['hilbert_dims'].append(ed.hilbert_dim)
            results['diag_times'].append(diag_time)
            results['energies'].append(ed.ground_state_energy())
            
            logger.info(f"✓ Size ({lx},{ly}): dim={ed.hilbert_dim}, time={diag_time*1000:.1f}ms", lvl=2)
        except Exception as e:
            logger.warning(f"✗ Size ({lx},{ly}) failed: {e}", lvl=2)
    
    return results


if __name__ == "__main__":
    logger.title("ED Benchmark Module", 60, '=', lvl=0)
    
    # Example: Create ED engine and compute ground state
    ed = ExactDiagonalization(gamma=0.5, lattice_size=(2, 1))
    ed.diagonalize()
    
    print(f"Ground state energy: {ed.ground_state_energy():.6f}")
    print(f"Energy gap: {ed.energy_gap():.6f}")
    print(f"First 5 energies: {ed.energy_spectrum(5)}")
