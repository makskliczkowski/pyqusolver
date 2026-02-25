"""
DQMC Model Module.
Defines the mapping between a Hamiltonian and the DQMC framework.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import jax.numpy as jnp
from QES.Algebra.hamil import Hamiltonian

def _extract_scalar(value: Any, default: Any) -> Any:
    """Return a scalar from Python/JAX/NumPy containers, fallback to default."""
    if value is None:
        return default
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        if arr.size == 0:
            return default
        return arr.reshape(-1)[0].item()
    except Exception:
        return value

class DQMCModel:
    """
    Base class for models that can be solved via DQMC.
    Handles extraction of the kinetic matrix K and interaction terms V.
    """
    def __init__(self, hamiltonian: Hamiltonian, beta: float, M: int):
        self.hamiltonian = hamiltonian
        self._beta = beta
        self.M = M
        self.dtau = beta / M
        self.n_sites = hamiltonian.ns
        self._kinetic_matrix = None
        
        # Metadata for the sampler
        self.n_channels = 2  # Default spin-up/dn
        self.field_type = "discrete" # "discrete" or "continuous"

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self.dtau = value / self.M

    @property
    def kinetic_matrix(self):
        """Returns the single-particle kinetic matrix K."""
        if self._kinetic_matrix is None:
            self._kinetic_matrix = self._extract_kinetic_matrix()
        return self._kinetic_matrix

    def _extract_kinetic_matrix(self):
        """Extract K from the Hamiltonian."""
        if hasattr(self.hamiltonian, 'hamil_sp') and self.hamiltonian.hamil_sp is not None:
            return np.array(self.hamiltonian.hamil_sp)
        return np.zeros((self.n_sites, self.n_sites))

    def get_hs_parameters(self) -> Dict[str, Any]:
        """Returns parameters needed for the HS transformation."""
        raise NotImplementedError()

    def get_propagators(self, config_tau, exp_K, exp_invK):
        """
        Build B and iB matrices for a given time slice configuration.
        Returns: (n_channels, N, N), (n_channels, N, N)
        """
        raise NotImplementedError()

    def calculate_update_deltas(self, s_old, s_new, site_idx):
        """
        Calculate the diagonal update factors (delta) for each channel.
        Returns: tuple of length n_channels
        """
        raise NotImplementedError()

class HubbardDQMCModel(DQMCModel):
    """
    Specific implementation for the Hubbard model (spinful).
    Standard SU(2) invariant HS transformation in the magnetic channel.
    """
    def __init__(self, hamiltonian: Hamiltonian, beta: float, M: int, U: float):
        super().__init__(hamiltonian, beta, M)
        self.U = U
        self.n_channels = 2
        self.lmbd = np.arccosh(np.exp(np.abs(self.U) * self.dtau / 2.0))

    def get_hs_parameters(self):
        return {"lambda": self.lmbd}

    def get_propagators(self, config_tau, exp_K, exp_invK):
        # v_up = exp(lambda * s), v_dn = exp(-lambda * s)
        v_up = jnp.exp(self.lmbd * config_tau)
        v_dn = jnp.exp(-self.lmbd * config_tau)
        
        B_up = exp_K * v_up[None, :]
        B_dn = exp_K * v_dn[None, :]
        
        iB_up = (1.0 / v_up)[:, None] * exp_invK
        iB_dn = (1.0 / v_dn)[:, None] * exp_invK
        
        return jnp.stack([B_up, B_dn]), jnp.stack([iB_up, iB_dn])

    def calculate_update_deltas(self, s_old, s_new, site_idx):
        ds = s_new - s_old
        d_up = jnp.exp(self.lmbd * ds) - 1.0
        d_dn = jnp.exp(-self.lmbd * ds) - 1.0
        return (d_up, d_dn)

    def get_checkerboard_decomposition(self) -> List[List[Tuple[int, int]]]:
        """
        Groups bonds into disjoint sets for checkerboard decomposition.
        Currently implemented for 2D square lattices.
        """
        ns = self.n_sites
        lat = self.hamiltonian.lattice
        if not lat or lat._type.name != "SQUARE":
            # Fallback: single group if not square (not optimized)
            return []
            
        groups = [[] for _ in range(4)]
        for i in range(ns):
            coords = lat.get_coordinates(i)
            x, y = int(coords[0]), int(coords[1])
            
            for nidx in range(lat.get_nn_num(i)):
                j = lat.get_nn(i, num=nidx)
                if lat.wrong_nei(j):
                    continue
                j = int(j)
                if j <= i:
                    continue
                    
                coords_j = lat.get_coordinates(j)
                xj, yj = int(coords_j[0]), int(coords_j[1])
                
                # Check horizontal bond
                if yj == y:
                    if x % 2 == 0:
                        groups[0].append((i, j))
                    else:
                        groups[1].append((i, j))
                # Check vertical bond
                elif xj == x:
                    if y % 2 == 0:
                        groups[2].append((i, j))
                    else:
                        groups[3].append((i, j))
        
        return [g for g in groups if len(g) > 0]

    def _extract_kinetic_matrix(self):
        # Specific extraction for Hubbard-like models in QES
        # Often these models have a '_t' attribute for hopping
        ns = self.n_sites
        K = np.zeros((ns, ns))
        lat = self.hamiltonian.lattice
        
        t_raw = _extract_scalar(getattr(self.hamiltonian, "_t", 1.0), 1.0)
        try:
            t_val = complex(t_raw).real
        except (TypeError, ValueError):
            t_val = 1.0

        if lat:
            for i in range(ns):
                for nidx in range(lat.get_nn_num(i)):
                    j = lat.get_nn(i, num=nidx)
                    if not lat.wrong_nei(j):
                        K[int(i), int(j)] = -t_val
        return K

def choose_dqmc_model(hamiltonian: Hamiltonian, beta: float, M: int, **kwargs) -> DQMCModel:
    """
    Factory function to select the appropriate DQMC wrapper for a given Hamiltonian.
    """
    model_name = str(getattr(hamiltonian, "_name", type(hamiltonian).__name__))
    name = model_name.lower()
    
    if "hubbard" in name:
        u_raw = _extract_scalar(kwargs.get("U", getattr(hamiltonian, "_u", 0.0)), 0.0)
        try:
            u_val = float(complex(u_raw).real)
        except (TypeError, ValueError):
            u_val = float(u_raw)

        return HubbardDQMCModel(hamiltonian, beta, M, u_val)
    
    # Add more models here (Heisenberg, Multiorbital, etc.)
    raise ValueError(f"No DQMC wrapper implemented for Hamiltonian type: {model_name}")
