"""
DQMC model wrappers.

This module keeps the historical `DQMCModel` API but delegates the actual
model-to-DQMC translation to adapter objects.  The goal is to keep the solver
generic while preserving the current import surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from QES.Algebra.hamil import Hamiltonian

from .dqmc_adapter import (
    DQMCModelAdapter,
    HubbardAdapter,
    SpinlessDensityDensityAdapter,
    choose_dqmc_adapter,
)


class DQMCModel:
    """
    Thin wrapper around a DQMC adapter.

    The model stores simulation-wide metadata (`beta`, `M`, `dtau`) and exposes
    the adapter contract through the historical `DQMCModel` interface used by
    the sampler and solver.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        beta: float,
        M: int,
        adapter: Optional[DQMCModelAdapter] = None,
    ):
        """
        Wrap a QES Hamiltonian together with a DQMC adapter.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            Physical model supplying lattice and coupling metadata.
        beta : float
            Inverse temperature ``beta``.
        M : int
            Number of Trotter slices, so ``dtau = beta / M``.
        adapter : DQMCModelAdapter, optional
            Explicit adapter implementing the DQMC representation.  When
            omitted, a suitable adapter is chosen automatically.
        """
        self.hamiltonian = hamiltonian
        self.M = M
        self._beta = beta
        self.adapter = adapter if adapter is not None else choose_dqmc_adapter(hamiltonian, beta, M)
        self._sync_from_adapter()

    def _sync_from_adapter(self):
        """Copy adapter dimensions and HS metadata onto the historical model surface."""
        self.dtau = self._beta / self.M
        self.n_sites = self.adapter.n_sites
        self.n_channels = self.adapter.n_channels
        self.field_type = self.adapter.field_type
        self.n_hs_fields = self.adapter.n_hs_fields
        self.max_update_size = self.adapter.max_update_size
        self.term_sites = self.adapter.term_sites

    @property
    def beta(self):
        """Inverse temperature ``beta`` used to define ``dtau = beta / M``."""
        return self._beta

    @beta.setter
    def beta(self, value):
        """
        Update ``beta`` and rebuild the HS representation if needed.

        This is not a metadata-only change: the HS coupling constants and the
        slice propagators both depend explicitly on ``dtau = beta / M``.
        """
        self._beta = value
        # Rebuild the adapter because ``dtau = beta / M`` enters the HS
        # coupling itself, not only the kinetic exponentials.
        if isinstance(self.adapter, HubbardAdapter):
            hs_params = self.adapter.get_hs_parameters()
            hs_kwargs = {}
            if "p" in hs_params:
                hs_kwargs["p"] = hs_params["p"]
            if "proposal_sigma" in hs_params:
                hs_kwargs["proposal_sigma"] = hs_params["proposal_sigma"]
            self.adapter = HubbardAdapter(
                hamiltonian=self.hamiltonian,
                beta=value,
                M=self.M,
                U=self.adapter.U,
                hs_kind=hs_params.get("name", "magnetic"),
                **hs_kwargs,
            )
        elif isinstance(self.adapter, SpinlessDensityDensityAdapter):
            self.adapter = SpinlessDensityDensityAdapter(
                hamiltonian=self.hamiltonian,
                beta=value,
                M=self.M,
            )
        else:
            self.adapter = choose_dqmc_adapter(self.hamiltonian, value, self.M, **self.get_hs_parameters())
        self._sync_from_adapter()

    @property
    def kinetic_matrix(self):
        """Return the one-body matrix ``K`` entering ``exp(-dtau K)``."""
        return self.adapter.kinetic_matrix

    def get_hs_parameters(self) -> Dict[str, Any]:
        """Return public HS metadata for diagnostics and reproducibility."""
        return self.adapter.get_hs_parameters()

    def get_sign_metadata(self) -> Dict[str, Any]:
        """Return explicit fermion sign-handling metadata for this model."""
        if hasattr(self.adapter, "get_sign_metadata"):
            return self.adapter.get_sign_metadata()
        return {
            "sign_tracking": "measured_on_abs_weight_ensemble",
            "reweighting": True,
            "weight_sampling": "absolute_determinant_ratio",
            "supports_complex_phase": False,
        }

    def validate_sign_policy(self, policy: str = "strict"):
        """Validate the requested public sign policy against the adapter metadata."""
        if hasattr(self.adapter, "validate_sign_policy"):
            self.adapter.validate_sign_policy(policy=policy)

    def get_sampling_metadata(self) -> Dict[str, Any]:
        """Return metadata describing the current local-update sampling path."""
        if hasattr(self.adapter, "get_sampling_metadata"):
            return self.adapter.get_sampling_metadata()
        return {
            "field_type": str(self.field_type),
            "proposal_kind": "unspecified",
            "provides_measure_gradient": False,
            "max_update_size": int(self.max_update_size),
            "update_mode": "immediate_local",
            "supports_delayed_updates": False,
        }

    def copy(self) -> "DQMCModel":
        """
        Return an independent DQMC model with the same Hamiltonian and HS setup.

        The Hamiltonian object itself is reused, but the DQMC adapter/model
        state is rebuilt so later metadata updates such as `beta` changes do not
        leak between solver clones.
        """
        hs_params = self.get_hs_parameters()
        kwargs: Dict[str, Any] = {}
        if "name" in hs_params:
            kwargs["hs"] = hs_params["name"]
        if "p" in hs_params:
            kwargs["p"] = hs_params["p"]
        if "proposal_sigma" in hs_params:
            kwargs["proposal_sigma"] = hs_params["proposal_sigma"]
        return choose_dqmc_model(self.hamiltonian, self.beta, self.M, **kwargs)

    def get_propagators(self, config_tau, exp_K, exp_invK):
        """Return one-slice propagators ``B_tau`` and inverses for a field slice."""
        return self.adapter.get_propagators(config_tau, exp_K, exp_invK)

    def calculate_update_deltas(self, s_old, s_new, site_idx):
        """Return localized diagonal update factors ``exp(V' - V) - 1``."""
        return self.adapter.calculate_update_deltas(s_old, s_new, site_idx)

    def initial_fields(self, key, shape):
        """Draw an initial auxiliary-field configuration."""
        return self.adapter.initial_fields(key, shape)

    def propose_field_value(self, s_old, key, term_idx):
        """Propose a local auxiliary-field move."""
        return self.adapter.propose_field_value(s_old, key, term_idx)

    def field_weight_ratio(self, s_old, s_new, term_idx):
        """Return the HS-measure factor ``b(s_new) / b(s_old)``."""
        return self.adapter.field_weight_ratio(s_old, s_new, term_idx)

    def proposal_ratio(self, s_old, s_new, term_idx):
        """Return the Hastings correction for the local proposal rule."""
        return self.adapter.proposal_ratio(s_old, s_new, term_idx)

    def log_field_weight(self, s, term_idx):
        """Return the local HS log-measure up to a normalization constant."""
        if hasattr(self.adapter, "log_field_weight"):
            return self.adapter.log_field_weight(s, term_idx)
        return 0.0

    def grad_log_field_weight(self, s, term_idx):
        """Return the local HS log-measure gradient when available."""
        if hasattr(self.adapter, "grad_log_field_weight"):
            return self.adapter.grad_log_field_weight(s, term_idx)
        return 0.0

    def measure_equal_time(self, greens, kinetic_matrix):
        """Delegate equal-time measurements to the adapter."""
        return self.adapter.measure_equal_time(greens, kinetic_matrix)

    def measure_equal_time_by_chain(self, greens, kinetic_matrix):
        """Delegate per-chain equal-time measurements to the adapter."""
        if hasattr(self.adapter, "measure_equal_time_by_chain"):
            return self.adapter.measure_equal_time_by_chain(greens, kinetic_matrix)
        return self.adapter.measure_equal_time(greens, kinetic_matrix)

    @property
    def supports_checkerboard(self) -> bool:
        """Whether the adapter exposes a checkerboard bond grouping."""
        return self.adapter.supports_checkerboard


class HubbardDQMCModel(DQMCModel):
    """
    Historical Hubbard-specific entrypoint preserved for compatibility.
    """

    def __init__(self, hamiltonian: Hamiltonian, beta: float, M: int, U: float):
        """Construct the compatibility Hubbard wrapper around `HubbardAdapter`."""
        super().__init__(
            hamiltonian=hamiltonian,
            beta=beta,
            M=M,
            adapter=HubbardAdapter(hamiltonian=hamiltonian, beta=beta, M=M, U=U),
        )

    def get_checkerboard_decomposition(self):
        """Return checkerboard bond groups when the underlying adapter supports them."""
        if hasattr(self.adapter, "get_checkerboard_decomposition"):
            return self.adapter.get_checkerboard_decomposition()
        return []


def choose_dqmc_model(hamiltonian: Hamiltonian, beta: float, M: int, **kwargs) -> DQMCModel:
    """
    Build the historical `DQMCModel` wrapper from a QES Hamiltonian.

    The returned object stores the simulation-wide Trotter metadata and
    delegates the actual DQMC representation to an adapter chosen from the
    Hamiltonian type and the requested HS options.
    """
    adapter = choose_dqmc_adapter(hamiltonian, beta, M, **kwargs)
    if isinstance(adapter, HubbardAdapter) and adapter.get_hs_parameters().get("name") == "magnetic":
        return HubbardDQMCModel(hamiltonian=hamiltonian, beta=beta, M=M, U=adapter.U)
    return DQMCModel(hamiltonian=hamiltonian, beta=beta, M=M, adapter=adapter)
