"""
Hubbard-Stratonovich transformations used by DQMC.

This module contains only the auxiliary-field decoupling logic:

- what field lives on each interaction term,
- how that field changes the fermionic one-body propagator,
- how local field proposals and measure weights are defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Sequence

import numpy as np
import jax
import jax.numpy as jnp


class LocalHSDistribution:
    r"""
    Local auxiliary-field manifold together with its sampling rule.

    This factor captures the field domain and measure ``b(s)`` in the generic
    HS identity

        exp[-dtau H_int] = \int ds  b(s) exp[a(s) O].
    """

    name: str = "distribution"
    field_type: str = "generic"
    proposal_kind: str = "unspecified"
    provides_measure_gradient: bool = False

    def initial_fields(self, key, shape):
        """Draw an initial auxiliary-field array on the local HS manifold."""
        raise NotImplementedError()

    def propose_local_value(self, s_old, key, term_idx):
        """Propose one local auxiliary-field update."""
        raise NotImplementedError()

    def field_weight_ratio(self, s_old, s_new, term_idx):
        """Return the local measure ratio ``b(s_new) / b(s_old)``."""
        del s_old, s_new, term_idx
        return 1.0

    def log_field_weight(self, s, term_idx):
        """Return the local log measure ``log b(s)`` up to a constant."""
        del s, term_idx
        return 0.0

    def grad_log_field_weight(self, s, term_idx):
        """Return the derivative of the local log measure when available."""
        del s, term_idx
        return 0.0

    def proposal_ratio(self, s_old, s_new, term_idx):
        """Return the Hastings correction for the proposal distribution."""
        del s_old, s_new, term_idx
        return 1.0

    def parameters(self) -> Dict[str, Any]:
        """Return public distribution metadata for logging and reproducibility."""
        return {
            "distribution": self.name,
            "field_type": self.field_type,
            "proposal_kind": self.proposal_kind,
            "provides_measure_gradient": self.provides_measure_gradient,
        }


@dataclass
class DiscreteIsingDistribution(LocalHSDistribution):
    """Uniform Ising field ``s = +-1`` with deterministic flip proposals."""

    name: str = "ising"
    field_type: str = "discrete"
    proposal_kind: str = "deterministic_flip"
    provides_measure_gradient: bool = False

    def initial_fields(self, key, shape):
        """Sample independent Ising variables ``s = +-1`` for all HS terms."""
        return jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=shape)

    def propose_local_value(self, s_old, key, term_idx):
        """Propose the deterministic local flip ``s -> -s``."""
        del key, term_idx
        return -s_old


@dataclass
class CompactUniformDistribution(LocalHSDistribution):
    """
    Uniform compact field ``s in [-pi, pi)`` with wrapped Gaussian proposals.
    """

    proposal_sigma: float = 0.5
    name: str = "compact_uniform"
    field_type: str = "compact"
    proposal_kind: str = "local_random_walk"
    provides_measure_gradient: bool = True

    def initial_fields(self, key, shape):
        """Sample a uniform compact field on ``[-pi, pi)``."""
        return jax.random.uniform(key, shape=shape, minval=-jnp.pi, maxval=jnp.pi)

    def propose_local_value(self, s_old, key, term_idx):
        """Make a wrapped Gaussian random-walk proposal on the compact interval."""
        del term_idx
        trial = s_old + self.proposal_sigma * jax.random.normal(key, shape=())
        return jnp.mod(trial + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    def parameters(self) -> Dict[str, Any]:
        """Return public proposal metadata for reproducibility."""
        return {**super().parameters(), "proposal_sigma": self.proposal_sigma}


@dataclass
class GaussianDistribution(LocalHSDistribution):
    """
    Noncompact Gaussian field with symmetric random-walk proposals.

    Math:
        the field measure contributes ``exp(-s^2 / 2)`` up to an irrelevant
        normalization, so local Metropolis ratios include

            exp[-(s_new^2 - s_old^2) / 2].
    """

    proposal_sigma: float = 0.5
    name: str = "gaussian"
    field_type: str = "continuous"
    proposal_kind: str = "local_random_walk"
    provides_measure_gradient: bool = True

    def initial_fields(self, key, shape):
        """Sample independent standard-normal HS fields."""
        return jax.random.normal(key, shape=shape)

    def propose_local_value(self, s_old, key, term_idx):
        """Make a symmetric Gaussian random-walk proposal ``s' = s + eta``."""
        del term_idx
        return s_old + self.proposal_sigma * jax.random.normal(key, shape=())

    def field_weight_ratio(self, s_old, s_new, term_idx):
        """Return the Gaussian-measure ratio ``exp[-(s_new^2 - s_old^2)/2]``."""
        del term_idx
        return jnp.exp(-0.5 * (s_new**2 - s_old**2))

    def log_field_weight(self, s, term_idx):
        """Return the Gaussian log measure ``-s^2 / 2`` up to a constant."""
        del term_idx
        return -0.5 * s**2

    def grad_log_field_weight(self, s, term_idx):
        """Return the exact derivative of the Gaussian log measure."""
        del term_idx
        return -s

    def parameters(self) -> Dict[str, Any]:
        """Return public proposal metadata for reproducibility."""
        return {**super().parameters(), "proposal_sigma": self.proposal_sigma}


class DiagonalCouplingRule:
    """
    Electron-coupling part ``a(s) O`` of an HS transformation.

    The coupling rule is independent from the field measure.  It converts local
    field values into diagonal single-particle potentials.
    """

    n_channels: int = 1
    n_fields: int = 0
    max_update_size: int = 1
    n_sites: int = 0

    @property
    def term_sites(self) -> np.ndarray:
        """Return the interaction support of each HS term as site indices."""
        raise NotImplementedError()

    def parameters(self) -> Dict[str, Any]:
        """Return coupling-rule metadata for diagnostics."""
        return {}

    def site_potentials(self, config_tau):
        """Map one slice of HS fields to channel-resolved diagonal potentials."""
        raise NotImplementedError()

    def update_deltas(self, s_old, s_new, term_idx):
        """Return local diagonal factors ``exp(V' - V) - 1`` for one term update."""
        raise NotImplementedError()


class LinearSiteCouplingRule(DiagonalCouplingRule):
    r"""
    Linear diagonal coupling ``V_c(i) = sum_t s_t eta_{c,t,i}``.
    """

    def __init__(self, _term_sites: np.ndarray, _site_couplings: np.ndarray, _term_couplings: np.ndarray):
        """Store a linear HS coupling tensor in both term-local and site-expanded form."""
        self._term_sites = _term_sites
        self._site_couplings = _site_couplings
        self._term_couplings = _term_couplings
        self.n_channels = int(self._site_couplings.shape[0])
        self.n_fields = int(self._site_couplings.shape[1])
        self.n_sites = int(self._site_couplings.shape[2])
        self.max_update_size = int(self._term_sites.shape[1])

    @property
    def term_sites(self) -> np.ndarray:
        """Return the support sites of each HS term."""
        return self._term_sites

    def site_potentials(self, config_tau):
        """Assemble channel/site potentials ``V_c(i) = sum_t s_t eta_{c,t,i}`` for one slice."""
        couplings = jnp.asarray(self._site_couplings)
        return jnp.einsum("t,cti->ci", jnp.asarray(config_tau), couplings)

    def update_deltas(self, s_old, s_new, term_idx):
        """Return local diagonal factors induced by changing one linear HS field value."""
        ds = s_new - s_old
        term_couplings = jnp.asarray(self._term_couplings)
        return jnp.exp(ds * term_couplings[:, term_idx, :]) - 1.0


class OnsiteScalarCouplingRule(DiagonalCouplingRule):
    r"""
    Onsite coupling generated by channel-dependent scalar functions ``a_c(s)``.

    Math:
        each term acts on one site only, so the slice potential is

            V_c(i) = a_c(s_i).
    """

    def __init__(self, n_sites: int, channel_functions: Sequence[Callable[[Any], Any]], metadata: Dict[str, Any]):
        """Build an onsite rule from per-channel scalar couplings ``a_c(s)``."""
        self.n_sites = int(n_sites)
        self.channel_functions = tuple(channel_functions)
        self.metadata = dict(metadata)
        self.n_channels = len(self.channel_functions)
        self.n_fields = int(self.n_sites)
        self.max_update_size = 1
        self._term_sites = np.arange(self.n_sites, dtype=np.int32)[:, None]

    @property
    def term_sites(self) -> np.ndarray:
        """Return the one-site support of each onsite HS term."""
        return self._term_sites

    def parameters(self) -> Dict[str, Any]:
        """Return descriptive metadata for the onsite coupling rule."""
        return dict(self.metadata)

    def site_potentials(self, config_tau):
        """Evaluate the onsite channel potentials ``V_c(i) = a_c(s_i)`` for one slice."""
        config_tau = jnp.asarray(config_tau)
        return jnp.stack([fn(config_tau) for fn in self.channel_functions], axis=0)

    def update_deltas(self, s_old, s_new, term_idx):
        """Return onsite diagonal factors ``exp[a_c(s_new)-a_c(s_old)] - 1``."""
        del term_idx
        old_vals = jnp.stack([fn(s_old) for fn in self.channel_functions])
        new_vals = jnp.stack([fn(s_new) for fn in self.channel_functions])
        return (jnp.exp(new_vals - old_vals) - 1.0)[:, None]


class BondDensityDifferenceRule(DiagonalCouplingRule):
    """
    Compact bond-centered coupling for spinless density-difference HS fields.

    Each bond field acts only on its two endpoints, so the slice potential can
    be assembled by scattering the signed bond amplitudes directly onto those
    endpoints instead of storing a dense site-expanded coupling tensor.
    """

    def __init__(self, n_sites: int, bonds: np.ndarray, alphas: np.ndarray):
        """Store bond endpoints and bond couplings in compact form."""
        self.n_sites = int(n_sites)
        self._bonds = np.asarray(bonds, dtype=np.int32)
        self._alphas = np.asarray(alphas, dtype=np.float64)
        self.n_channels = 1
        self.n_fields = int(self._bonds.shape[0])
        self.max_update_size = 2

    @property
    def term_sites(self) -> np.ndarray:
        """Return the two endpoint sites touched by each bond field."""
        return self._bonds

    def site_potentials(self, config_tau):
        """Assemble spinless bond-field potentials by scattering onto bond endpoints."""
        config_tau = jnp.asarray(config_tau)
        signed = config_tau * jnp.asarray(self._alphas)
        bond_i = jnp.asarray(self._bonds[:, 0], dtype=jnp.int32)
        bond_j = jnp.asarray(self._bonds[:, 1], dtype=jnp.int32)
        potentials = jnp.zeros((1, self.n_sites), dtype=signed.dtype)
        potentials = potentials.at[0, bond_i].add(signed)
        potentials = potentials.at[0, bond_j].add(-signed)
        return potentials

    def update_deltas(self, s_old, s_new, term_idx):
        """Return the two-site diagonal factors induced by one bond-field change."""
        ds = s_new - s_old
        alpha = jnp.asarray(self._alphas)[term_idx]
        return jnp.asarray(
            [[jnp.exp(ds * alpha) - 1.0, jnp.exp(-ds * alpha) - 1.0]],
            dtype=jnp.asarray(alpha).dtype,
        )


@dataclass
class HSTransformation:
    """
    Base class for DQMC auxiliary-field decouplings.

    A concrete transformation is built from:

    - a local field distribution, which defines the manifold and measure,
    - a coupling rule, which maps local field values to diagonal fermion
      potentials on the interaction support.
    """

    dtau: float
    field_type: str = "discrete"
    name: str = "hs"
    n_channels: int = 1
    n_fields: int = 0
    max_update_size: int = 1
    n_sites: int = 0

    def parameters(self) -> Dict[str, Any]:
        """Return public HS parameters for diagnostics and reproducibility."""
        return {
            "name": self.name,
            "field_type": self.field_type,
            "n_channels": self.n_channels,
            "n_fields": self.n_fields,
        }

    def initial_fields(self, key, shape):
        """Delegate initial-field generation to the local distribution."""
        return self.distribution.initial_fields(key, shape)

    def propose_local_value(self, s_old, key, term_idx):
        """Delegate local proposal generation to the field distribution."""
        return self.distribution.propose_local_value(s_old, key, term_idx)

    def field_weight_ratio(self, s_old, s_new, term_idx):
        """Delegate the local measure ratio to the field distribution."""
        return self.distribution.field_weight_ratio(s_old, s_new, term_idx)

    def proposal_ratio(self, s_old, s_new, term_idx):
        """Delegate the Hastings correction to the field distribution."""
        return self.distribution.proposal_ratio(s_old, s_new, term_idx)

    def log_field_weight(self, s, term_idx):
        """Delegate the local log measure to the field distribution."""
        return self.distribution.log_field_weight(s, term_idx)

    def grad_log_field_weight(self, s, term_idx):
        """Delegate the local log-measure gradient to the field distribution."""
        return self.distribution.grad_log_field_weight(s, term_idx)

    @property
    def term_sites(self) -> np.ndarray:
        """Return the site support of each HS interaction term."""
        return self.rule.term_sites

    @property
    def term_couplings(self) -> np.ndarray:
        """Expose linear term-local couplings when the rule is linear in the HS fields."""
        if hasattr(self.rule, "_term_couplings"):
            return getattr(self.rule, "_term_couplings")
        raise NotImplementedError("This HS transformation uses a nonlinear coupling rule.")

    @property
    def site_couplings(self) -> np.ndarray:
        """Expose linear site-expanded couplings when the rule is linear in the HS fields."""
        if hasattr(self.rule, "_site_couplings"):
            return getattr(self.rule, "_site_couplings")
        raise NotImplementedError("This HS transformation uses a nonlinear coupling rule.")

    def propagators(self, config_tau, exp_K, exp_invK):
        r"""
        Build one-slice propagators from the HS field configuration.

        Math:
            the HS field enters only through a diagonal one-body potential

                V_c(i) = \sum_t \eta_{c,t,i}(s_t),

            so each slice matrix is ``B_c = e^{-dtau K} e^{V_c}``.
        """
        site_potentials = self.rule.site_potentials(config_tau)
        diag_factors = jnp.exp(site_potentials)
        Bs = exp_K[None, :, :] * diag_factors[:, None, :]
        iBs = (1.0 / diag_factors)[:, :, None] * exp_invK[None, :, :]
        return Bs, iBs

    def update_deltas(self, s_old, s_new, term_idx):
        """Delegate the localized diagonal correction to the coupling rule."""
        return self.rule.update_deltas(s_old, s_new, term_idx)


@dataclass
class ComposableHSTransformation(HSTransformation):
    """
    Concrete HS transformation assembled from a distribution and coupling rule.
    """

    distribution: LocalHSDistribution | None = None
    rule: DiagonalCouplingRule | None = None

    def __post_init__(self):
        """Validate and expose dimensions inherited from the chosen distribution and rule."""
        if self.distribution is None or self.rule is None:
            raise ValueError("ComposableHSTransformation requires both `distribution` and `rule`.")
        self.field_type = self.distribution.field_type
        self.n_channels = self.rule.n_channels
        self.n_fields = self.rule.n_fields
        self.max_update_size = self.rule.max_update_size
        self.n_sites = self.rule.n_sites

    def parameters(self) -> Dict[str, Any]:
        """Merge transformation, distribution, and coupling metadata into one public dictionary."""
        return {
            **super().parameters(),
            **self.distribution.parameters(),
            **self.rule.parameters(),
        }


@dataclass
class MagneticHubbardHS(ComposableHSTransformation):
    r"""
    Discrete Hirsch decoupling in the magnetic channel for onsite repulsive Hubbard interactions.

    Math:
        for

            U n_{i\uparrow} n_{i\downarrow},

        we rewrite the interaction in particle-hole symmetric form and decouple
        it with a discrete Ising field ``s_i(\tau) = \pm 1``:

            e^{-dtau U (n_\uparrow - 1/2)(n_\downarrow - 1/2)}
              = C \sum_{s=\pm1} e^{\lambda s (n_\uparrow - n_\downarrow)},

        where

            cosh(\lambda) = exp(|U| dtau / 2).
    """

    U: float = 0.0
    n_sites: int = 1

    def __post_init__(self):
        """Instantiate the discrete magnetic-channel Hirsch transformation."""
        self.name = "magnetic"
        self.U = float(np.real(complex(self.U)))
        self.n_sites = int(self.n_sites)
        lmbd = np.arccosh(np.exp(np.abs(self.U) * self.dtau / 2.0))
        def up_channel(s):
            """Return the magnetic-channel onsite coupling on the up-spin block."""
            return lmbd * s

        def down_channel(s):
            """Return the magnetic-channel onsite coupling on the down-spin block."""
            return -lmbd * s

        self.distribution = DiscreteIsingDistribution()
        self.rule = OnsiteScalarCouplingRule(
            n_sites=self.n_sites,
            channel_functions=[up_channel, down_channel],
            metadata={"U": self.U, "lambda": lmbd},
        )
        self.lmbd = lmbd
        super().__post_init__()

    def parameters(self) -> Dict[str, Any]:
        """Return the onsite interaction and the Hirsch coupling ``lambda``."""
        return {**super().parameters(), "U": self.U, "lambda": self.lmbd}


@dataclass
class ChargeHubbardHS(ComposableHSTransformation):
    r"""
    Discrete Hirsch decoupling in the charge channel for attractive onsite Hubbard interactions.

    Math:
        for attractive interactions the paper's generic form couples the HS
        field to charge,

            O = n_\uparrow + n_\downarrow - 1,

        so both spin channels see the same sign.
    """

    U: float = 0.0
    n_sites: int = 1

    def __post_init__(self):
        """Instantiate the discrete charge-channel Hirsch transformation."""
        self.name = "charge"
        self.U = float(np.real(complex(self.U)))
        self.n_sites = int(self.n_sites)
        lmbd = np.arccosh(np.exp(np.abs(self.U) * self.dtau / 2.0))
        def charge_channel(s):
            """Return the charge-channel onsite coupling seen by both spin blocks."""
            return lmbd * s

        self.distribution = DiscreteIsingDistribution()
        self.rule = OnsiteScalarCouplingRule(
            n_sites=self.n_sites,
            channel_functions=[charge_channel, charge_channel],
            metadata={"U": self.U, "lambda": lmbd},
        )
        self.lmbd = lmbd
        super().__post_init__()

    def parameters(self) -> Dict[str, Any]:
        """Return the onsite interaction and the Hirsch coupling ``lambda``."""
        return {**super().parameters(), "U": self.U, "lambda": self.lmbd}


@dataclass
class GaussianHubbardHS(ComposableHSTransformation):
    r"""
    Gaussian onsite HS transformation for repulsive or attractive Hubbard interactions.

    Source:
        Karakuzu et al., arXiv:2211.05074, Eqs. (11)-(16).

    Math:
        the noncompact Gaussian field satisfies

            a(s) = sqrt(dtau |U|) s,

        while the coupled operator is

            n_\uparrow - n_\downarrow    for U > 0,
            n_\uparrow + n_\downarrow - 1 for U < 0.
    """

    U: float = 0.0
    n_sites: int = 1
    proposal_sigma: float = 0.5

    def __post_init__(self):
        """Instantiate the noncompact Gaussian Hubbard HS transformation."""
        self.name = "gaussian"
        self.U = float(np.real(complex(self.U)))
        self.n_sites = int(self.n_sites)
        self.amplitude = float(np.sqrt(self.dtau * np.abs(self.U)))
        same_sign = self.U < 0.0

        def a(s):
            """Return the scalar onsite coupling ``a(s) = sqrt(dtau |U|) s``."""
            return self.amplitude * s

        if same_sign:
            channel_functions = [a, a]
        else:
            channel_functions = [a, lambda s: -a(s)]

        self.distribution = GaussianDistribution(proposal_sigma=float(self.proposal_sigma))
        self.rule = OnsiteScalarCouplingRule(
            n_sites=self.n_sites,
            channel_functions=channel_functions,
            metadata={"U": self.U, "amplitude": self.amplitude},
        )
        super().__post_init__()

    def parameters(self) -> Dict[str, Any]:
        """Return onsite interaction strength and Gaussian coupling amplitude."""
        return {
            **super().parameters(),
            "U": self.U,
            "amplitude": self.amplitude,
            "proposal_sigma": float(self.proposal_sigma),
        }


@dataclass
class CompactInterpolatingHubbardHS(ComposableHSTransformation):
    r"""
    Compact continuous onsite HS transformation interpolating between Lee and Hirsch.

    Source:
        Karakuzu et al., "A flexible class of exact Hubbard-Stratonovich
        transformations", arXiv:2211.05074, Eqs. (30), (31), and (35).

    Math:
        for repulsive onsite Hubbard interactions we use

            a_p(s) = sqrt(c_p) * atan(p sin s) / atan(p),    s in [-pi, pi],

        with a uniform compact measure on ``[-pi, pi]``.
    """

    U: float = 0.0
    p: float = 0.0
    n_sites: int = 1
    proposal_sigma: float = 0.5

    def __post_init__(self):
        """Instantiate the compact interpolating onsite HS transformation."""
        self.name = "compact_interpolating"
        self.U = float(np.real(complex(self.U)))
        self.p = float(self.p)
        if self.p < 0.0:
            raise ValueError("CompactInterpolatingHubbardHS requires p >= 0.")
        self.n_sites = int(self.n_sites)
        self.proposal_sigma = float(self.proposal_sigma)
        self.cp = self._solve_cp()
        self.sqrt_cp = float(np.sqrt(self.cp))

        def a(s):
            """Evaluate the compact onsite coupling map ``a_p(s)`` for one field value."""
            s = jnp.asarray(s)
            if self.p == 0.0:
                scaled = jnp.sin(s)
            else:
                scaled = jnp.arctan(self.p * jnp.sin(s)) / jnp.arctan(self.p)
            return self.sqrt_cp * scaled

        self.distribution = CompactUniformDistribution(proposal_sigma=self.proposal_sigma)
        self.rule = OnsiteScalarCouplingRule(
            n_sites=self.n_sites,
            channel_functions=[a, lambda s: -a(s)],
            metadata={"U": self.U, "p": self.p, "cp": self.cp},
        )
        super().__post_init__()

    def _atan_scaled(self, s):
        """Return the bounded odd profile entering the compact coupling map."""
        if self.p == 0.0:
            return np.sin(s)
        return np.arctan(self.p * np.sin(s)) / np.arctan(self.p)

    def _solve_cp(self) -> float:
        """Solve for the normalization constant ``c_p`` that enforces the exact HS identity."""
        target = np.exp(self.dtau * np.abs(self.U) / 2.0)
        if np.isclose(target, 1.0):
            return 0.0

        def avg_cosh(cp_value: float) -> float:
            """Compute the compact-field average entering the defining equation for ``c_p``."""
            grid = np.linspace(-np.pi, np.pi, 2001, dtype=np.float64)
            vals = self._atan_scaled(grid)
            return float(np.mean(np.cosh(np.sqrt(cp_value) * vals)))

        lo, hi = 0.0, max(self.dtau * np.abs(self.U), 1.0)
        while avg_cosh(hi) < target:
            hi *= 2.0

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if avg_cosh(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def parameters(self) -> Dict[str, Any]:
        """Return the interaction strength and compact-transformation parameters."""
        return {
            **super().parameters(),
            "U": self.U,
            "p": self.p,
            "cp": self.cp,
            "proposal_sigma": self.proposal_sigma,
        }


@dataclass
class BondDensityDifferenceHS(ComposableHSTransformation):
    r"""
    Discrete HS decoupling for repulsive spinless density-density bond terms.

    Math:
        for one interaction bond ``(i, j)`` with coupling ``V_{ij} > 0`` we use
        the particle-hole symmetric form

            V_{ij} (n_i - 1/2)(n_j - 1/2),

        and decouple it with an Ising field on that bond:

            e^{-dtau V_{ij} (n_i - 1/2)(n_j - 1/2)}
              = C \sum_{s=\pm1} e^{alpha_{ij} s (n_i - n_j)}.
    """

    V: Sequence[float] | float = 0.0
    bonds: Iterable[Sequence[int]] = ()
    n_sites: int = 0

    def __post_init__(self):
        """Instantiate the bond density-difference HS transformation for spinless models."""
        self.name = "bond_density"
        self.n_sites = int(self.n_sites)

        bonds = np.asarray(list(self.bonds), dtype=np.int32)
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            raise ValueError("BondDensityDifferenceHS requires bonds with shape (n_bonds, 2).")

        couplings = np.asarray(self.V, dtype=np.float64)
        if couplings.ndim == 0:
            couplings = np.full((bonds.shape[0],), float(couplings), dtype=np.float64)
        if couplings.shape[0] != bonds.shape[0]:
            raise ValueError("Number of bond couplings must match number of bonds.")
        if np.any(couplings < 0.0):
            raise ValueError("BondDensityDifferenceHS currently supports only repulsive real couplings V >= 0.")

        alphas = np.arccosh(np.exp(couplings * self.dtau / 2.0))
        self.distribution = DiscreteIsingDistribution()
        self.rule = BondDensityDifferenceRule(
            n_sites=self.n_sites,
            bonds=bonds,
            alphas=alphas,
        )
        self.alphas = alphas
        super().__post_init__()

    def parameters(self) -> Dict[str, Any]:
        """Return bond-count and bond-coupling range information for diagnostics."""
        return {
            **super().parameters(),
            "n_bonds": self.n_fields,
            "alpha_min": float(np.min(self.alphas)) if self.n_fields else 0.0,
            "alpha_max": float(np.max(self.alphas)) if self.n_fields else 0.0,
        }


def choose_hs_transformation(kind: str, **kwargs) -> HSTransformation:
    """
    Factory for DQMC Hubbard-Stratonovich transformations.

    Supported names
    ---------------
    - ``"magnetic"``: discrete Hirsch decoupling for repulsive onsite Hubbard
    - ``"charge"``: discrete charge-channel decoupling for attractive onsite Hubbard
    - ``"gaussian"``: noncompact Gaussian onsite transformation
    - ``"compact"``: compact interpolating onsite transformation
    - ``"bond_density"``: discrete bond density-difference decoupling for spinless models
    """
    name = str(kind).strip().lower()
    if name in {"magnetic", "hirsch", "onsite_magnetic"}:
        return MagneticHubbardHS(
            dtau=kwargs["dtau"],
            U=kwargs["U"],
            n_sites=int(kwargs.get("n_sites", 1)),
        )
    if name in {"charge", "onsite_charge"}:
        return ChargeHubbardHS(
            dtau=kwargs["dtau"],
            U=kwargs["U"],
            n_sites=int(kwargs.get("n_sites", 1)),
        )
    if name in {"gaussian", "continuous_gaussian"}:
        return GaussianHubbardHS(
            dtau=kwargs["dtau"],
            U=kwargs["U"],
            n_sites=int(kwargs.get("n_sites", 1)),
            proposal_sigma=float(kwargs.get("proposal_sigma", 0.5)),
        )
    if name in {"compact", "compact_interpolating", "lee", "interpolating"}:
        return CompactInterpolatingHubbardHS(
            dtau=kwargs["dtau"],
            U=kwargs["U"],
            p=float(kwargs.get("p", 0.0)),
            n_sites=int(kwargs.get("n_sites", 1)),
            proposal_sigma=float(kwargs.get("proposal_sigma", 0.5)),
        )
    if name in {"bond_density", "density_difference", "spinless_bond"}:
        return BondDensityDifferenceHS(
            dtau=kwargs["dtau"],
            V=kwargs["V"],
            bonds=kwargs["bonds"],
            n_sites=int(kwargs["n_sites"]),
        )
    raise ValueError(
        f"Unknown or unsupported HS transformation {kind!r}. "
        f"Currently supported: 'magnetic', 'charge', 'gaussian', 'compact', 'bond_density'."
    )
