"""
Symmetry handling for Neural Quantum States (NQS).

This module provides the `NQSSymmetricAnsatz` class, which manages the application
of symmetry projections to the NQS ansatz. It wraps the base ansatz function
to sum over symmetry orbits, enforcing the desired symmetry sector.

----------------------------------------------------------
File        : QES/NQS/src/nqs_symmetry.py
Description : NQS Symmetry Ansatz Handler
Author      : Maksymilian Kliczkowski
Date        : 2025-12-10
----------------------------------------------------------
"""

from dataclasses    import dataclass
from typing         import Optional, Callable, TYPE_CHECKING

try:
    import  jax
    import  jax.numpy as jnp
    from    jax.scipy.special import logsumexp
    JAX_AVAILABLE   = True
except ImportError:
    JAX_AVAILABLE   = False
    jax             = None
    jnp             = None
    logsumexp       = None

if TYPE_CHECKING:
    from QES.NQS.nqs import NQS

@dataclass
class NQSSymmetricAnsatz:
    """
    Configuration manager for the symmetric ansatz in NQS.

    This class provides methods to set, unset, and check the status of
    symmetry projections for the NQS ansatz. It ensures the symmetrized
    ansatz is properly JIT-compiled and applied.

    Methods:
        set(use_symmetries: bool = True) -> None:
            Enables or disables symmetrization based on the NQS's Hilbert space.
        unset() -> None:
            Disables symmetrization.
        wrap(base_func: Callable) -> Callable:
            Wraps a given base ansatz function with the active symmetry projection.

    Properties:
        active (bool):
            Returns True if symmetrization is currently active, False otherwise.
    """
    _nqs        : 'NQS'
    _projector  : Optional[Callable]    = None
    _is_active  : bool                  = False

    def set(self, use_symmetries: bool = True) -> None:
        """
        Enable or disable symmetrization for the NQS ansatz.

        If `use_symmetries` is True, attempts to set up a symmetry projector
        based on the NQS's Hilbert space. If successful, the NQS's ansatz
        pipeline is rebuilt to include this projection. Symmetrization is
        only supported for the JAX backend.

        Parameters
        ----------
        use_symmetries : bool, default=True
            If True, enables symmetrization. If False, calls `unset()`.
        """
        if not use_symmetries:
            self.unset()
            return

        # Check backend
        if not self._nqs._isjax:
            self._nqs.log("Symmetry projection skipped: only supported for JAX backend.", lvl=1, color='yellow')
            self.unset() # Ensure inactive if backend is not JAX
            return
        
        if not JAX_AVAILABLE:
            self._nqs.log("JAX is not available, cannot enable symmetry projection.", lvl=0, color='red')
            self.unset()
            return

        # Check Hilbert space
        hilbert = self._nqs._hilbert
        if hilbert is None:
            self._nqs.log("Symmetry projection skipped: No Hilbert space provided to NQS.", lvl=1, color='yellow')
            self.unset()
            return

        sym_container = getattr(hilbert, 'sym_container', None) or getattr(hilbert, '_sym_container', None)
        if sym_container is None or len(sym_container.generators) == 0:
            self._nqs.log("Symmetry projection skipped: No symmetry generators found in Hilbert space.", lvl=1, color='yellow')
            self.unset()
            return

        try:
            self._projector = sym_container.get_jittable_projector()
            self._is_active = True
            self._nqs._rebuild_ansatz_function()
            self._nqs.log(f"Symmetry projection enabled for {len(sym_container.generators)} generators: {self._nqs._model.sym}", lvl=1, color='green')
        except Exception as e:
            self._nqs.log(f"Failed to enable symmetry projection: {e}", lvl=0, color='red')
            self._nqs.log("Continuing without symmetry projection.", lvl=0, color='yellow')
            self.unset()

    def unset(self) -> None:
        """
        Disables symmetrization and rebuilds the NQS ansatz pipeline.

        This effectively removes the symmetry projection layer from the ansatz.
        """
        if self._is_active:
            self._is_active = False
            self._projector = None
            self._nqs._rebuild_ansatz_function()
            self._nqs.log("Symmetry projection disabled.", lvl=1, color='blue')

    @property
    def active(self) -> bool:
        """
        Returns True if symmetrization is currently active, False otherwise.
        """
        return self._is_active

    def wrap(self, base_func: Callable) -> Callable:
        """
        Wraps a given base ansatz function with the active symmetry projection logic.

        If symmetrization is not active, the original `base_func` is returned.
        Otherwise, a JIT-compiled function is returned that applies the symmetry
        projection by summing over the symmetry orbit with appropriate weights.

        Parameters
        ----------
        base_func : Callable
            The base ansatz function to wrap (signature: `(params, inputs) -> log_psi`).

        Returns
        -------
        Callable
            The wrapped (symmetrized) ansatz function, or the original `base_func`
            if symmetrization is not active.
        """
        if not self._is_active or self._projector is None:
            return base_func
        
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for the symmetry wrapper, but it is not available.")

        projector = self._projector

        def symmetrized_ansatz(params, inputs):
            # Project states to orbit: (batch, group_size, ns) or (group_size, ns) if 1D
            orbit_states, orbit_weights = projector(inputs)
            
            # Handle dimensions
            if orbit_states.ndim == 2: # 1D input case: (group_size, ns)
                group_size  = orbit_states.shape[0]
                batch_dim   = 1
            else: # 2D input case: (batch, group_size, ns)
                batch_dim   = orbit_states.shape[0]
                group_size  = orbit_states.shape[1]
            
            # Flatten for network evaluation: (batch_dim * group_size, ns)
            flat_states                 = orbit_states.reshape(-1, orbit_states.shape[-1])
            
            # Evaluate network on all orbit states: (batch_dim * group_size,)
            log_psi_flat                = base_func(params, flat_states) 
            
            # Reshape back to (batch_dim, group_size)
            log_psi                     = log_psi_flat.reshape(batch_dim, group_size)
            
            # Compute symmetrized log psi: log( sum( w * exp(log_psi) ) )
            # We use logsumexp with 'b' argument for weights: log(sum(b * exp(a)))
            # JAX's logsumexp supports complex inputs/weights.
            
            # Ensure weights are broadcastable if they are 1D (from 1D input)
            if orbit_weights.ndim == 1 and batch_dim == 1:
                 # orbit_weights is (group_size,), log_psi is (1, group_size)
                 # Expand weights to (1, group_size)
                 orbit_weights = jnp.expand_dims(orbit_weights, 0)
            
            log_psi_sym                 = logsumexp(log_psi, axis=1, b=orbit_weights)
            
            # If input was 1D, return scalar (squeeze batch dim)
            if inputs.ndim == 1:
                log_psi_sym = jnp.squeeze(log_psi_sym, axis=0)
            
            return jnp.where(jnp.isinf(log_psi_sym), -jnp.inf, log_psi_sym) # Handle -inf for zero sum

        return jax.jit(symmetrized_ansatz)

# ----------------------------------
# End of file
# ---------------------------------