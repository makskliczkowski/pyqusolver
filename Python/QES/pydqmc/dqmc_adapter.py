"""
Adapters that translate QES Hamiltonians into DQMC-ready building blocks.

The adapter layer keeps the solver generic:

- QES model/Hamiltonian objects define the physical model,
- adapters define how that model is represented inside auxiliary-field DQMC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from QES.Algebra.hamil import Hamiltonian

from .hs import HSTransformation, choose_hs_transformation
from .measurements import ensure_chain_axis


def extract_scalar(value: Any, default: Any) -> Any:
    """Return a scalar from Python/NumPy/JAX containers, else `default`."""
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


def _extract_site_coupling(values: Any, index: int, default: float = 0.0) -> complex:
    """Read one site-dependent coupling from scalar/list-like Hamiltonian metadata."""
    if values is None:
        return default
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 0:
        return default
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr[index].item()


def _real_scalar_or_raise(value: Any, *, name: str) -> float:
    """Convert a coupling to a real scalar and reject genuinely complex values."""
    scalar = np.asarray(value).item() if np.asarray(value).ndim == 0 else value
    imag = np.imag(scalar)
    if np.abs(imag) > 1e-12:
        raise ValueError(f"{name} must be real for the current DQMC adapter, got {value!r}.")
    return float(np.real(scalar))


def build_lattice_kinetic_matrix(hamiltonian: Hamiltonian) -> np.ndarray:
    """
    Build the single-particle matrix entering ``exp(-dtau K)`` from lattice metadata.

    Math:
        DQMC evolves the quadratic part of the Hamiltonian through the one-body
        matrix ``K``.  For tight-binding models this includes hopping and any
        diagonal chemical-potential shift.
    """
    if hasattr(hamiltonian, "hamil_sp") and hamiltonian.hamil_sp is not None:
        return np.array(hamiltonian.hamil_sp)

    ns = int(hamiltonian.ns)
    K = np.zeros((ns, ns), dtype=np.complex128)
    lat = getattr(hamiltonian, "lattice", None)
    if lat is None:
        return K

    use_forward = bool(getattr(hamiltonian, "_use_forward", False))
    mu_raw = getattr(hamiltonian, "_mu", None)
    for i in range(ns):
        mu_i = _extract_site_coupling(mu_raw, i, 0.0)
        K[i, i] -= mu_i

    t_raw = getattr(hamiltonian, "_t", None)
    for i in range(ns):
        nn_count = lat.get_nn_forward_num(i) if use_forward else lat.get_nn_num(i)
        for nidx in range(nn_count):
            j = lat.get_nn_forward(i, num=nidx) if use_forward else lat.get_nn(i, num=nidx)
            if lat.wrong_nei(j):
                continue
            j = int(j)
            wx, wy, wz = lat.bond_winding(i, j)
            phase = lat.boundary_phase_from_winding(wx, wy, wz)
            amp = -phase * _extract_site_coupling(t_raw, i, 1.0)
            K[i, j] += amp
            if use_forward:
                K[j, i] += np.conjugate(amp)
    return K


def validate_supported_kinetic_matrix(kinetic_matrix: np.ndarray):
    """
    Reject kinetic matrices with genuinely complex phases.

    The current `pydqmc` implementation does not track a complex fermion phase
    or perform sign/phase reweighting.  We therefore fail fast for one-body
    matrices carrying a nontrivial imaginary part instead of silently sampling a
    magnitude-only weight.
    """
    kinetic_matrix = np.asarray(kinetic_matrix)
    if np.max(np.abs(np.imag(kinetic_matrix))) > 1e-12:
        raise ValueError(
            "Complex one-body phases are not supported by QES.pydqmc because "
            "explicit fermion sign/phase tracking is not implemented."
        )


def _is_real_matrix(matrix: np.ndarray, tol: float = 1e-12) -> bool:
    """Return whether a matrix is real up to numerical tolerance."""
    return float(np.max(np.abs(np.imag(np.asarray(matrix))))) <= tol


def _chemical_potential_is_zero(mu_values: Any, tol: float = 1e-12) -> bool:
    """Return whether the Hamiltonian chemical-potential metadata is identically zero."""
    if mu_values is None:
        return True
    mu_array = np.asarray(mu_values, dtype=np.complex128)
    return float(np.max(np.abs(mu_array))) <= tol


def _is_bipartite_from_matrix(kinetic_matrix: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Determine whether the off-diagonal hopping graph encoded by `K` is bipartite.

    The check uses the support of the real-space one-body matrix rather than
    lattice-type labels so it remains valid for custom bipartite geometries.
    """
    matrix = np.asarray(kinetic_matrix)
    n_sites = int(matrix.shape[0])
    adjacency = [[] for _ in range(n_sites)]
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if abs(matrix[i, j]) > tol or abs(matrix[j, i]) > tol:
                adjacency[i].append(j)
                adjacency[j].append(i)

    colors = np.full((n_sites,), -1, dtype=np.int8)
    for start in range(n_sites):
        if colors[start] != -1:
            continue
        colors[start] = 0
        queue = [start]
        while queue:
            site = queue.pop(0)
            for neighbor in adjacency[site]:
                if colors[neighbor] == -1:
                    colors[neighbor] = 1 - colors[site]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[site]:
                    return False
    return True


def extract_density_bonds(hamiltonian: Hamiltonian) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the interaction bonds ``(i, j, V_ij)`` from a QES spinless Hubbard model.

    The current faithful spinless DQMC path assumes repulsive real bond
    couplings so that the discrete density-difference HS transformation remains
    real-valued.
    """
    lat = getattr(hamiltonian, "lattice", None)
    if lat is None:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)

    bonds: List[Tuple[int, int]] = []
    couplings: List[float] = []
    use_forward = bool(getattr(hamiltonian, "_use_forward", True))
    u_raw = getattr(hamiltonian, "_u", None)

    for i in range(int(hamiltonian.ns)):
        nn_count = lat.get_nn_forward_num(i) if use_forward else lat.get_nn_num(i)
        for nidx in range(nn_count):
            j = lat.get_nn_forward(i, num=nidx) if use_forward else lat.get_nn(i, num=nidx)
            if lat.wrong_nei(j):
                continue
            j = int(j)
            if (not use_forward) and j <= i:
                continue

            wx, wy, wz = lat.bond_winding(i, j)
            phase = lat.boundary_phase_from_winding(wx, wy, wz)
            v_ij = phase * _extract_site_coupling(u_raw, i, 0.0)
            v_ij = _real_scalar_or_raise(v_ij, name="Spinless density-density coupling")
            if v_ij < 0.0:
                raise ValueError(
                    "The current spinless density-density DQMC adapter supports only repulsive bonds V >= 0."
                )
            bonds.append((i, j))
            couplings.append(v_ij)

    return np.asarray(bonds, dtype=np.int32), np.asarray(couplings, dtype=np.float64)


@dataclass
class DQMCModelAdapter:
    """
    Minimal contract between a QES Hamiltonian and the DQMC engine.

    The adapter must provide:

    - the single-particle kinetic matrix ``K``,
    - the HS field specification,
    - the local propagator update rule induced by an HS flip.
    """

    hamiltonian: Hamiltonian
    beta: float
    M: int
    n_channels: int = 1
    field_type: str = "discrete"

    def __post_init__(self):
        """Populate cached dimensions derived from the Hamiltonian and Trotter grid."""
        self.dtau = self.beta / self.M
        self.n_sites = self.hamiltonian.ns
        self.n_hs_fields = self.n_sites
        self.max_update_size = 1
        self._term_sites = np.arange(self.n_sites, dtype=np.int32)[:, None]
        self._kinetic_matrix = None

    @property
    def kinetic_matrix(self):
        """Return the single-particle kinetic matrix ``K`` entering ``exp(-dtau K)``."""
        if self._kinetic_matrix is None:
            self._kinetic_matrix = self.build_kinetic_matrix()
            validate_supported_kinetic_matrix(self._kinetic_matrix)
        return self._kinetic_matrix

    @property
    def term_sites(self) -> np.ndarray:
        """Return the site support of each local HS term."""
        return self._term_sites

    def build_kinetic_matrix(self):
        """Build the one-body kinetic matrix."""
        return build_lattice_kinetic_matrix(self.hamiltonian)

    def get_hs_parameters(self) -> Dict[str, Any]:
        """Return HS parameters needed by the sampler/diagnostics."""
        raise NotImplementedError()

    def initial_fields(self, key, shape):
        """Draw an initial auxiliary-field configuration."""
        return jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=shape)

    def get_propagators(self, config_tau, exp_K, exp_invK):
        """
        Return slice propagators ``B_tau`` and inverses ``B_tau^{-1}``.

        The returned arrays have shape ``(n_channels, N, N)``.
        """
        raise NotImplementedError()

    def propose_field_value(self, s_old, key, term_idx):
        """Return a proposed new local auxiliary-field value."""
        del key, term_idx
        return -s_old

    def field_weight_ratio(self, s_old, s_new, term_idx):
        """Return the HS-measure ratio ``b(s_new) / b(s_old)``."""
        del s_old, s_new, term_idx
        return 1.0

    def proposal_ratio(self, s_old, s_new, term_idx):
        """Return the Hastings correction for the local proposal."""
        del s_old, s_new, term_idx
        return 1.0

    def calculate_update_deltas(self, s_old, s_new, term_idx):
        """
        Return the localized diagonal update factors for each fermion channel.

        Math:
            after a local field flip the interaction matrix changes only on the
            support of the corresponding HS term, so each channel is updated by
            a small diagonal correction

                Delta = diag(exp(V' - V) - 1).
        """
        raise NotImplementedError()

    def measure_equal_time(self, greens, kinetic_matrix):
        """
        Return model-specific equal-time observables from channel Green's functions.

        Adapters can override this to expose interaction-specific estimators.
        """
        chain_values = self.measure_equal_time_by_chain(greens, kinetic_matrix)
        return {name: float(jnp.mean(jnp.real(jnp.asarray(values)))) for name, values in chain_values.items()}

    def measure_equal_time_by_chain(self, greens, kinetic_matrix):
        """
        Return per-chain equal-time observables from channel Green's functions.

        The default implementation exposes only the per-chain density.
        """
        density = 0.0
        for green in greens:
            green = ensure_chain_axis(green)
            density += jnp.mean(1.0 - jnp.diagonal(green, axis1=-2, axis2=-1), axis=-1)
        return {"density": density}

    def get_sign_metadata(self) -> Dict[str, Any]:
        """
        Return explicit metadata about fermion sign handling in the current run.

        The present implementation uses local determinant-ratio magnitudes in
        the acceptance rule and does not yet perform explicit sign/phase
        tracking or reweighting.  We expose that contract directly so users do
        not need to infer it from the implementation.
        """
        return {
            "sign_tracking": "measured_on_abs_weight_ensemble",
            "reweighting": True,
            "weight_sampling": "absolute_determinant_ratio",
            "supports_complex_phase": False,
            "sign_envelope": "unsupported",
            "expected_average_sign": None,
            "sign_reason": "No sign-safe envelope has been declared for this adapter.",
        }

    def validate_sign_policy(self, policy: str = "strict"):
        """
        Enforce the supported sign envelope for public solver entrypoints.

        Policies:

        - `strict`: reject regimes not known to be sign-benign in the current baseline
        - `allow_unsupported`: permit the run but keep explicit sign metadata
        """
        policy_normalized = str(policy).strip().lower()
        if policy_normalized not in {"strict", "allow_unsupported"}:
            raise ValueError(
                f"Unknown sign_policy={policy!r}. Expected 'strict' or 'allow_unsupported'."
            )
        sign_metadata = self.get_sign_metadata()
        if policy_normalized == "strict" and sign_metadata.get("sign_envelope") != "known_sign_free":
            raise ValueError(
                "QES.pydqmc does not support this regime under the default strict sign policy. "
                f"Reason: {sign_metadata.get('sign_reason', 'unsupported sign structure')}. "
                "Use sign_policy='allow_unsupported' only if you explicitly accept "
                "that this regime is outside the current validated sign-safe envelope."
            )

    def get_sampling_metadata(self) -> Dict[str, Any]:
        """Return metadata describing the current local-update sampling path."""
        return {
            "field_type": str(self.field_type),
            "max_update_size": int(self.max_update_size),
            "update_mode": "immediate_local",
            "supports_delayed_updates": False,
            "proposal_kind": "unspecified",
            "provides_measure_gradient": False,
        }

    @property
    def supports_checkerboard(self) -> bool:
        """Whether the adapter can group kinetic bonds into non-overlapping checkerboard sets."""
        return False


class _HSBackedAdapter(DQMCModelAdapter):
    """
    Adapter base for models whose DQMC representation is fully described by an HS object.
    """

    hs: HSTransformation

    def _sync_from_hs(self):
        """Mirror HS metadata onto the adapter fields expected by the sampler."""
        self.n_channels = self.hs.n_channels
        self.field_type = self.hs.field_type
        self.n_hs_fields = self.hs.n_fields
        self.max_update_size = self.hs.max_update_size
        self._term_sites = self.hs.term_sites

    def get_hs_parameters(self) -> Dict[str, Any]:
        """Expose the underlying HS transformation parameters unchanged."""
        return self.hs.parameters()

    def get_propagators(self, config_tau, exp_K, exp_invK):
        """Delegate slice-propagator construction to the HS transformation."""
        return self.hs.propagators(config_tau, exp_K, exp_invK)

    def calculate_update_deltas(self, s_old, s_new, term_idx):
        """Delegate localized diagonal update factors to the HS transformation."""
        return self.hs.update_deltas(s_old, s_new, term_idx)

    def initial_fields(self, key, shape):
        """Delegate initial-field generation to the HS transformation."""
        return self.hs.initial_fields(key, shape)

    def propose_field_value(self, s_old, key, term_idx):
        """Delegate local field proposals to the HS transformation."""
        return self.hs.propose_local_value(s_old, key, term_idx)

    def field_weight_ratio(self, s_old, s_new, term_idx):
        """Delegate local HS-measure ratios to the HS transformation."""
        return self.hs.field_weight_ratio(s_old, s_new, term_idx)

    def proposal_ratio(self, s_old, s_new, term_idx):
        """Delegate Hastings corrections to the HS transformation."""
        return self.hs.proposal_ratio(s_old, s_new, term_idx)

    def log_field_weight(self, s, term_idx):
        """Delegate local log-measure evaluation to the HS transformation."""
        return self.hs.log_field_weight(s, term_idx)

    def grad_log_field_weight(self, s, term_idx):
        """Delegate local log-measure gradients to the HS transformation."""
        return self.hs.grad_log_field_weight(s, term_idx)

    def get_sampling_metadata(self) -> Dict[str, Any]:
        """Expose the HS distribution metadata together with update-path limits."""
        hs_parameters = self.hs.parameters()
        return {
            "field_type": str(self.field_type),
            "proposal_kind": hs_parameters.get("proposal_kind", "unspecified"),
            "provides_measure_gradient": bool(hs_parameters.get("provides_measure_gradient", False)),
            "max_update_size": int(self.max_update_size),
            "update_mode": "immediate_local",
            "supports_delayed_updates": False,
        }


class HubbardAdapter(_HSBackedAdapter):
    """
    Magnetic-channel Hirsch decoupling for an onsite Hubbard-like model.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        beta: float,
        M: int,
        U: float,
        hs_kind: str = "magnetic",
        **hs_kwargs,
    ):
        """Build an onsite-Hubbard adapter with the requested HS decoupling channel."""
        super().__init__(hamiltonian=hamiltonian, beta=beta, M=M, n_channels=2, field_type="discrete")
        self.U = float(np.real(complex(U)))
        self.hs: HSTransformation = choose_hs_transformation(
            hs_kind,
            U=self.U,
            dtau=self.dtau,
            n_sites=self.n_sites,
            **hs_kwargs,
        )
        self._sync_from_hs()

    def measure_equal_time(self, greens, kinetic_matrix):
        """Return chain-averaged onsite Hubbard equal-time observables."""
        chain_values = self.measure_equal_time_by_chain(greens, kinetic_matrix)
        return {name: float(jnp.mean(jnp.real(jnp.asarray(values)))) for name, values in chain_values.items()}

    def measure_equal_time_by_chain(self, greens, kinetic_matrix):
        """
        Standard per-chain equal-time observables for the onsite Hubbard model.

        Math:
            ``n_sigma(i) = 1 - G_sigma(ii)`` and
            ``<n_up(i) n_dn(i)>`` follows from Wick's theorem in the DQMC
            equal-time ensemble for decoupled fermions at fixed HS field.
        """
        greens = tuple(ensure_chain_axis(green) for green in greens)
        n_up = 1.0 - jnp.diagonal(greens[0], axis1=-2, axis2=-1)
        n_dn = 1.0 - jnp.diagonal(greens[1], axis1=-2, axis2=-1)

        e_kin = jnp.zeros_like(jnp.sum(n_up, axis=-1))
        kinetic_trace = jnp.trace(kinetic_matrix)
        for green in greens:
            e_kin += kinetic_trace - jnp.trace(kinetic_matrix @ green, axis1=-2, axis2=-1)

        double_occ = n_up * n_dn
        e_int = self.U * jnp.sum(double_occ, axis=-1)
        total_e = e_kin + e_int
        mz2 = jnp.mean(n_up + n_dn - 2.0 * double_occ, axis=-1)

        return {
            "energy": total_e,
            "density": jnp.mean(n_up + n_dn, axis=-1),
            "double_occupancy": jnp.mean(double_occ, axis=-1),
            "mz2": mz2,
        }

    def get_sign_metadata(self) -> Dict[str, Any]:
        """
        Return a conservative sign-envelope classification for the onsite spinful path.

        Known sign-benign cases in the current baseline:

        - attractive onsite Hubbard with the charge HS channel and real kinetic matrix
        - repulsive onsite Hubbard at zero chemical potential on a bipartite real hopping graph
        """
        kinetic_matrix = np.asarray(self.kinetic_matrix)
        hs_name = str(self.get_hs_parameters().get("name", "")).strip().lower()
        mu_zero = _chemical_potential_is_zero(getattr(self.hamiltonian, "_mu", None))
        real_kinetic = _is_real_matrix(kinetic_matrix)
        bipartite_graph = _is_bipartite_from_matrix(kinetic_matrix)

        metadata = super().get_sign_metadata()
        metadata.update({
            "hs_channel": hs_name,
            "real_kinetic": bool(real_kinetic),
            "zero_chemical_potential": bool(mu_zero),
            "bipartite_hopping_graph": bool(bipartite_graph),
        })

        attractive_sign_safe_channels = {"charge", "gaussian"}
        repulsive_half_filled_sign_safe_channels = {"magnetic", "gaussian", "compact_interpolating"}

        if real_kinetic and self.U < 0.0 and hs_name in attractive_sign_safe_channels:
            metadata.update({
                "sign_envelope": "known_sign_free",
                "expected_average_sign": 1.0,
                "sign_reason": (
                    "Attractive onsite Hubbard with real kinetic matrix and matched spin-channel HS "
                    "has identical spin determinants, so the Monte Carlo weight is nonnegative."
                ),
            })
            return metadata

        if (
            real_kinetic
            and self.U >= 0.0
            and hs_name in repulsive_half_filled_sign_safe_channels
            and mu_zero
            and bipartite_graph
        ):
            metadata.update({
                "sign_envelope": "known_sign_free",
                "expected_average_sign": 1.0,
                "sign_reason": (
                    "Repulsive onsite Hubbard at zero chemical potential on a bipartite real hopping graph "
                    "with a real onsite HS decoupling is sign-free by particle-hole symmetry."
                ),
            })
            return metadata

        if self.U >= 0.0 and hs_name == "magnetic" and not mu_zero:
            metadata.update({
                "sign_envelope": "unsupported",
                "expected_average_sign": None,
                "sign_reason": (
                    "Repulsive onsite Hubbard away from half filling is not supported in the current "
                    "baseline because sign reweighting is not implemented."
                ),
            })
            return metadata

        metadata.update({
            "sign_envelope": "unsupported",
            "expected_average_sign": None,
            "sign_reason": (
                "This onsite Hubbard regime is not currently guaranteed sign-free in QES.pydqmc."
            ),
        })
        return metadata

    @property
    def supports_checkerboard(self) -> bool:
        """Whether the lattice admits the square-lattice bond partition used by checkerboard updates."""
        lat = getattr(self.hamiltonian, "lattice", None)
        return bool(lat) and getattr(getattr(lat, "_type", None), "name", None) == "SQUARE"

    def get_checkerboard_decomposition(self) -> List[List[Tuple[int, int]]]:
        """
        Group non-overlapping bonds for a square-lattice checkerboard update.
        """
        ns = self.n_sites
        lat = self.hamiltonian.lattice
        if not self.supports_checkerboard:
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
                if yj == y:
                    groups[0 if x % 2 == 0 else 1].append((i, j))
                elif xj == x:
                    groups[2 if y % 2 == 0 else 3].append((i, j))
        return [group for group in groups if group]


class SpinlessDensityDensityAdapter(_HSBackedAdapter):
    """
    DQMC adapter for spinless density-density models with bond HS fields.

    The current faithful path targets the existing QES spinless Hubbard model,
    whose interaction terms live on bonds rather than onsite spin channels.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        beta: float,
        M: int,
        hs_kind: str = "bond_density",
    ):
        """Build a bond-field adapter for repulsive spinless density-density interactions."""
        super().__init__(hamiltonian=hamiltonian, beta=beta, M=M, n_channels=1, field_type="discrete")
        self.bonds, self.bond_couplings = extract_density_bonds(hamiltonian)
        self.hs: HSTransformation = choose_hs_transformation(
            hs_kind,
            V=self.bond_couplings,
            bonds=self.bonds,
            dtau=self.dtau,
            n_sites=self.n_sites,
        )
        self._sync_from_hs()

    def get_hs_parameters(self) -> Dict[str, Any]:
        """Return HS metadata together with the interaction-channel label used by this adapter."""
        return {
            **self.hs.parameters(),
            "interaction_type": "bond_density_density",
        }

    def measure_equal_time(self, greens, kinetic_matrix):
        """Return chain-averaged equal-time observables for the spinless density-density model."""
        chain_values = self.measure_equal_time_by_chain(greens, kinetic_matrix)
        return {name: float(jnp.mean(jnp.real(jnp.asarray(values)))) for name, values in chain_values.items()}

    def measure_equal_time_by_chain(self, greens, kinetic_matrix):
        """
        Per-chain equal-time observables for a spinless density-density model.

        Math:
            for one spinless channel,

                n_i = 1 - G_{ii},

            and Wick's theorem gives the bond correlator

                <n_i n_j> = <n_i><n_j> - G_{ij} G_{ji}

            for ``i != j``.
        """
        green = ensure_chain_axis(greens[0])
        density = 1.0 - jnp.diagonal(green, axis1=-2, axis2=-1)
        e_kin = jnp.trace(kinetic_matrix) - jnp.trace(kinetic_matrix @ green, axis1=-2, axis2=-1)

        if self.bonds.shape[0] == 0:
            e_int = jnp.zeros_like(e_kin)
            bond_corr = jnp.zeros_like(density[..., 0])
        else:
            bond_i = jnp.asarray(self.bonds[:, 0], dtype=jnp.int32)
            bond_j = jnp.asarray(self.bonds[:, 1], dtype=jnp.int32)
            n_i = density[:, bond_i]
            n_j = density[:, bond_j]
            gij = green[:, bond_i, bond_j]
            gji = green[:, bond_j, bond_i]
            bond_density = n_i * n_j - gij * gji
            e_int = jnp.sum(jnp.asarray(self.bond_couplings)[None, :] * bond_density, axis=-1)
            bond_corr = jnp.mean(bond_density, axis=-1)

        total_e = e_kin + e_int
        return {
            "energy": total_e,
            "density": jnp.mean(density, axis=-1),
            "bond_density_density": bond_corr,
        }

    def get_sign_metadata(self) -> Dict[str, Any]:
        """Return explicit metadata for the current spinless bond-density baseline."""
        metadata = super().get_sign_metadata()
        metadata.update({
            "sign_envelope": "unsupported",
            "expected_average_sign": None,
            "sign_reason": (
                "The spinless bond-density DQMC path does not currently provide a proven sign-free "
                "envelope or sign reweighting workflow."
            ),
        })
        return metadata


def choose_dqmc_adapter(
    hamiltonian: Hamiltonian,
    beta: float,
    M: int,
    **kwargs,
) -> DQMCModelAdapter:
    """
    Pick the DQMC adapter matching a QES Hamiltonian.

    Default onsite-Hubbard convention:

    - repulsive ``U > 0``  -> magnetic HS channel,
    - attractive ``U < 0`` -> charge HS channel.

    Math:
        in the exact HS identity the coupled operator is naturally

            n_up - n_dn        for repulsive U,
            n_up + n_dn - 1    for attractive U,

        which is the standard sign-dependent choice used in DQMC.
    """
    model_name = str(getattr(hamiltonian, "_name", type(hamiltonian).__name__)).lower()
    if "hubbard" not in model_name:
        raise ValueError(
            f"No DQMC adapter implemented for Hamiltonian type: "
            f"{getattr(hamiltonian, '_name', type(hamiltonian).__name__)}"
        )

    hilbert = getattr(hamiltonian, "hilbert_space", None)
    local_space = getattr(hilbert, "_local_space", None)
    local_space_str = str(getattr(local_space, "value", local_space)).lower() if local_space is not None else ""

    if "spinful" in model_name or type(hamiltonian).__name__ == "SpinfulHubbardModel":
        u_raw = extract_scalar(kwargs.get("U", getattr(hamiltonian, "_u", 0.0)), 0.0)
        try:
            u_val = float(complex(u_raw).real)
        except (TypeError, ValueError):
            u_val = float(u_raw)
        hs_kind = kwargs.get("hs", "magnetic" if u_val >= 0.0 else "charge")
        hs_kwargs = {}
        if "p" in kwargs:
            hs_kwargs["p"] = kwargs["p"]
        if "proposal_sigma" in kwargs:
            hs_kwargs["proposal_sigma"] = kwargs["proposal_sigma"]
        return HubbardAdapter(
            hamiltonian=hamiltonian,
            beta=beta,
            M=M,
            U=u_val,
            hs_kind=hs_kind,
            **hs_kwargs,
        )

    if "spinless" in local_space_str or "spinless" in model_name:
        hs_kind = kwargs.get("hs", "bond_density")
        return SpinlessDensityDensityAdapter(hamiltonian=hamiltonian, beta=beta, M=M, hs_kind=hs_kind)

    u_raw = extract_scalar(kwargs.get("U", getattr(hamiltonian, "_u", 0.0)), 0.0)
    try:
        u_val = float(complex(u_raw).real)
    except (TypeError, ValueError):
        u_val = float(u_raw)
    hs_kind = kwargs.get("hs", "magnetic" if u_val >= 0.0 else "charge")
    hs_kwargs = {}
    if "p" in kwargs:
        hs_kwargs["p"] = kwargs["p"]
    if "proposal_sigma" in kwargs:
        hs_kwargs["proposal_sigma"] = kwargs["proposal_sigma"]
    return HubbardAdapter(hamiltonian=hamiltonian, beta=beta, M=M, U=u_val, hs_kind=hs_kind, **hs_kwargs)
