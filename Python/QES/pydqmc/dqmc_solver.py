"""
Determinant Quantum Monte Carlo solver layer for QES.

This module binds the DQMC sampler, measurement helpers, and the generic QES
Monte Carlo solver interface into one high-level execution path.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Union, Any, Dict, Iterable, Mapping
import numpy as np
import jax.numpy as jnp
from QES.Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain, McsReturn
from QES.Solver.MonteCarlo.diagnostics import compute_autocorr_time
from QES.pydqmc.dqmc_sampler import DQMCSampler
from QES.pydqmc.dqmc_model import DQMCModel, choose_dqmc_model
from QES.pydqmc.measurements import measure_equal_time, measure_equal_time_by_chain, measure_time_displaced, mean_observables, reweight_observables
from QES.pydqmc.postprocessing import derive_observables, summarize_result, summarize_series


def _jsonify(value):
    """Convert NumPy/JAX-heavy results into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(val) for val in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@dataclass
class DQMCConfig:
    """
    Structured runtime configuration for `run_dqmc(...)`.

    The high-level entrypoint still supports the existing keyword-only calling
    style, but this dataclass provides a typed container that can be logged,
    passed around, and serialized more cleanly.
    """

    model: Any
    beta: float | None = None
    M: int | None = None
    warmup: int = 0
    sweeps: int = 10
    measure_every: int = 1
    n_stable: int = 10
    num_chains: int = 1
    seed: int | None = None
    collect_unequal: bool = False
    residual_recompute_threshold: float | None = 1e-6
    refresh_strategy: str = "jax_udt"
    residual_check_interval: int | None = None
    sign_policy: str = "strict"
    observable_hooks: tuple[Any, ...] = field(default_factory=tuple)
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def runtime_kwargs(self) -> Dict[str, Any]:
        """Return the `run_dqmc(...)` keyword arguments encoded in this config."""
        return {
            "beta": self.beta,
            "M": self.M,
            "warmup": self.warmup,
            "sweeps": self.sweeps,
            "measure_every": self.measure_every,
            "n_stable": self.n_stable,
            "num_chains": self.num_chains,
            "seed": self.seed,
            "collect_unequal": self.collect_unequal,
            "residual_recompute_threshold": self.residual_recompute_threshold,
            "refresh_strategy": self.refresh_strategy,
            "residual_check_interval": self.residual_check_interval,
            "sign_policy": self.sign_policy,
            "observable_hooks": self.observable_hooks,
            **self.extra_options,
        }

    def summary_dict(self) -> Dict[str, Any]:
        """Return a serializable config summary without the model object itself."""
        return _jsonify({
            "beta": self.beta,
            "M": self.M,
            "warmup": self.warmup,
            "sweeps": self.sweeps,
            "measure_every": self.measure_every,
            "n_stable": self.n_stable,
            "num_chains": self.num_chains,
            "seed": self.seed,
            "collect_unequal": self.collect_unequal,
            "residual_recompute_threshold": self.residual_recompute_threshold,
            "refresh_strategy": self.refresh_strategy,
            "residual_check_interval": self.residual_check_interval,
            "sign_policy": self.sign_policy,
            "observable_hooks": len(self.observable_hooks),
            "extra_options": dict(self.extra_options),
        })


@dataclass
class DQMCResult:
    """
    Structured return type for `run_dqmc(...)`.

    The class keeps dict-like indexing for backwards compatibility while also
    supporting typed fields, summary export, and JSON persistence for the
    analysis-ready parts of a run.
    """

    solver: Any
    train_result: Any
    observables: Dict[str, Any]
    diagnostics: Dict[str, Any]
    setup: Dict[str, Any]
    energy_history: list[float]
    energy_autocorr_time: float | None
    config: Dict[str, Any]
    unequal_time: Any = None

    def to_dict(self, include_solver: bool = True) -> Dict[str, Any]:
        """Return the result as a plain dictionary."""
        payload = {
            "train_result": self.train_result,
            "observables": self.observables,
            "diagnostics": self.diagnostics,
            "setup": self.setup,
            "energy_history": self.energy_history,
            "energy_autocorr_time": self.energy_autocorr_time,
            "config": self.config,
        }
        if include_solver:
            payload["solver"] = self.solver
        if self.unequal_time is not None:
            payload["unequal_time"] = self.unequal_time
        return payload

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable result summary."""
        payload = self.to_dict(include_solver=False)
        if self.train_result is not None:
            payload["train_result"] = _jsonify({
                "losses": getattr(self.train_result, "losses", None),
                "losses_mean": getattr(self.train_result, "losses_mean", None),
                "losses_std": getattr(self.train_result, "losses_std", None),
                "finished": getattr(self.train_result, "finished", None),
            })
        return _jsonify(payload)

    def save(self, path: str | Path):
        """Save the analysis-ready result bundle as JSON."""
        path = Path(path)
        path.write_text(json.dumps(self.to_serializable_dict(), indent=2, sort_keys=True))

    def summarize_energy(self, warmup: int = 0, bin_size: int = 1) -> Dict[str, Any]:
        """Return standard postprocessed statistics for the recorded energy trace."""
        return summarize_series(self.energy_history, warmup=warmup, bin_size=bin_size)

    def summarize(self, warmup: int = 0, bin_size: int = 1) -> Dict[str, Any]:
        """Return a compact postprocessed summary of the run."""
        return summarize_result(self, warmup=warmup, bin_size=bin_size)

    def derive(self, formulas: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate user-defined derived observables from the stored observable dictionary."""
        return derive_observables(self.observables, formulas)

    def __getitem__(self, key: str):
        """Provide backwards-compatible dictionary-style access."""
        return self.to_dict(include_solver=True)[key]

    def __contains__(self, key: str) -> bool:
        """Support `key in result` checks."""
        return key in self.to_dict(include_solver=True)

    def keys(self) -> Iterable[str]:
        """Return the public result keys."""
        return self.to_dict(include_solver=True).keys()


def load_dqmc_result(path: str | Path) -> DQMCResult:
    """Load a JSON result summary saved by `DQMCResult.save(...)`."""
    payload = json.loads(Path(path).read_text())
    return DQMCResult(
        solver=None,
        train_result=payload.get("train_result"),
        observables=payload.get("observables", {}),
        diagnostics=payload.get("diagnostics", {}),
        setup=payload.get("setup", {}),
        energy_history=payload.get("energy_history", []),
        energy_autocorr_time=payload.get("energy_autocorr_time"),
        config=payload.get("config", {}),
        unequal_time=payload.get("unequal_time"),
    )

class DQMCSolver(MonteCarloSolver):
    """
    High-level solver for determinant quantum Monte Carlo simulations.

    The solver owns the long-lived Monte Carlo workflow: thermalization,
    repeated sampler sweeps, equal-time accumulation, optional unequal-time
    measurements, and basic diagnostics.
    """
    def __init__(
        self,
        model: Union[DQMCModel, 'Hamiltonian'],
        beta: Optional[float] = None,
        M: Optional[int] = None,
        n_stable: int = 10,
        num_chains: int = 1,
        seed: Optional[int] = None,
        directory: Optional[str] = "dqmc_results",
        **kwargs
    ):
        """
        Construct a high-level DQMC solver from either a DQMC model or a Hamiltonian.

        When a bare Hamiltonian is provided, the solver first builds the
        corresponding DQMC adapter/model using `beta` and `M`.
        """
        solver_kwargs = dict(kwargs)
        residual_recompute_threshold = solver_kwargs.pop("residual_recompute_threshold", 1e-6)
        refresh_strategy = solver_kwargs.pop("refresh_strategy", "jax_udt")
        residual_check_interval = solver_kwargs.pop("residual_check_interval", None)
        sign_policy = str(solver_kwargs.pop("sign_policy", "strict"))
        observable_hooks = tuple(solver_kwargs.pop("observable_hooks", ()))
        if not isinstance(model, DQMCModel):
            if beta is None or M is None:
                raise ValueError("beta and M must be provided if model is a Hamiltonian.")
            model = choose_dqmc_model(model, beta, M, **solver_kwargs)
        model.validate_sign_policy(policy=sign_policy)
            
        super().__init__(
            seed=seed,
            shape=(model.M, model.n_hs_fields),
            hilbert=model.hamiltonian.hilbert_space,
            directory=directory,
            **solver_kwargs
        )
        self.model = model
        self.sampler = DQMCSampler(
            model,
            n_stable=n_stable,
            num_chains=num_chains,
            seed=seed if seed else 42,
            residual_recompute_threshold=residual_recompute_threshold,
            refresh_strategy=refresh_strategy,
            residual_check_interval=residual_check_interval,
        )
        self._info = f"DQMC Solver for {model.hamiltonian._name}"
        self._last_acc_rate = 0.0
        self._energies = []
        self.observable_hooks = observable_hooks
        self.sign_policy = sign_policy
        
        # Accumulators for measurement
        self._gs_accum      = None
        self._n_measurements = 0
        self._observable_num = {}
        self._sign_sum = 0.0
        self._sign_measurements = 0
        self._avg_sign_history = []
        self._last_observables = {}
        self._last_raw_observables = {}
        self._unequal_num = None
        self._unequal_sign_sum = 0.0

    @property
    def lastloss(self):
        """Return the latest energy estimator, used e.g. in tempering swaps."""
        if len(self._energies) > 0:
            return self._energies[-1]
        return 0.0

    def set_beta(self, beta: float):
        """Update ``beta`` and refresh the sampler's cached Trotter factors."""
        self.model.beta = beta
        self.sampler.dtau = self.model.dtau
        self.sampler.recompute_everything()

    def train_step(self, i: int, par: McsTrain, verbose: bool = False, **kwargs) -> McsReturn:
        """
        Perform one Monte Carlo sweep and record the current observables.

        One sweep means one pass through all Trotter slices and local HS terms
        for every chain managed by the sampler.
        """
        acc_rate = self.sampler.sweep()
        self._last_acc_rate = float(acc_rate)
        should_measure = (i % max(1, int(par.mc_corr))) == 0
        obs = {"energy": self.lastloss}
        if should_measure:
            if self._gs_accum is None:
                self._gs_accum = [jnp.zeros_like(g) for g in self.sampler.Gs_avg]
            for c in range(self.model.n_channels):
                self._gs_accum[c] += self.sampler.Gs_avg[c]
            self._n_measurements += 1
            obs = self._measure_current_observables()
            self._energies.append(obs["energy"])
        
        # Optional: collection of unequal time Gs
        collect_unequal = kwargs.get("collect_unequal", False)
        if collect_unequal and should_measure:
            unequal_gs = self.sampler.compute_unequal_time_Gs()
            current_signs = jnp.asarray(getattr(self.sampler, "current_signs", jnp.ones((self.sampler.num_chains,))))
            sign_sum_step = jnp.sum(current_signs)
            weighted_unequal = jnp.tensordot(current_signs, unequal_gs, axes=(0, 0))
            if self._unequal_num is None:
                self._unequal_num = weighted_unequal
            else:
                self._unequal_num = self._unequal_num + weighted_unequal
            self._unequal_sign_sum += float(jnp.real(sign_sum_step))
            self._unequal_gs = unequal_gs
        
        losses = [obs["energy"]] if should_measure else []
        return McsReturn(losses=losses, finished=False)

    def train(self, par: McsTrain, verbose: bool = False, **kwargs):
        """Thermalize, sample, and return the recorded energy history."""
        # Reset accumulators
        self._gs_accum = None
        self._n_measurements = 0
        self._observable_num = {}
        self._sign_sum = 0.0
        self._sign_measurements = 0
        self._avg_sign_history = []
        self._last_observables = {}
        self._last_raw_observables = {}
        self._unequal_num = None
        self._unequal_sign_sum = 0.0
        
        # Warmup
        for _ in range(par.mcth):
            self.sampler.sweep()
            
        # Sampling
        self._energies = []
        for step in range(par.mcsam):
            self.train_step(step, par, verbose=verbose, **kwargs)
            
        return McsReturn(
            losses=self._energies, 
            losses_mean=[float(np.mean(self._energies))] if self._energies else [],
            losses_std=[float(np.std(self._energies))] if self._energies else [],
            finished=True
        )

    def get_gs_avg(self):
        """Return the measurement-average equal-time Green's functions."""
        if self._gs_accum is None or self._n_measurements == 0:
            return self.sampler.Gs_avg
        return tuple(g / self._n_measurements for g in self._gs_accum)

    def measure_energy(self):
        """
        Return the equal-time energy estimator.

        The actual estimator lives in `pydqmc.measurements` so that solver
        control flow stays separate from model-specific observable formulas.
        """
        observables = self.measure_observables()
        return float(observables.get("energy", 0.0))

    def measure_observables(self) -> Dict[str, Any]:
        """
        Measure equal-time observables from the accumulated Green's functions.
        """
        if self._observable_num and abs(self._sign_sum) > 1e-14:
            observables = {
                name: float(np.real(value / self._sign_sum))
                for name, value in self._observable_num.items()
            }
            observables["average_sign"] = float(self._sign_sum / max(1, self._sign_measurements))
            self._apply_observable_hooks(observables, current_signs=None)
            return observables

        observables = measure_equal_time(
            self.model,
            self.get_gs_avg(),
            kinetic_matrix=self.sampler.K_jax,
        )
        current_signs = getattr(self.sampler, "current_signs", None)
        if current_signs is not None:
            observables["average_sign"] = float(jnp.mean(jnp.real(jnp.asarray(current_signs))))
        self._apply_observable_hooks(observables, current_signs=None)
        return observables

    def _measure_current_observables(self) -> Dict[str, Any]:
        """
        Measure and accumulate one sign-reweighted equal-time sample.

        Built-in observables are reweighted explicitly over the chain axis using
        the current per-chain sign/phase factors computed on the `|W|` ensemble.
        """
        current_signs = jnp.asarray(getattr(self.sampler, "current_signs", jnp.ones((self.sampler.num_chains,))))
        by_chain = measure_equal_time_by_chain(
            self.model,
            self.sampler.Gs_avg,
            kinetic_matrix=self.sampler.K_jax,
        )
        raw_means = mean_observables(by_chain)
        observables = reweight_observables(by_chain, current_signs)
        sign_sum_step = jnp.sum(current_signs)
        avg_sign_step = float(jnp.mean(jnp.real(current_signs)))
        self._avg_sign_history.append(avg_sign_step)
        self._sign_sum += float(jnp.real(sign_sum_step))
        self._sign_measurements += int(current_signs.shape[0])

        for name, values in by_chain.items():
            numer = jnp.sum(current_signs * jnp.asarray(values))
            self._observable_num[name] = self._observable_num.get(name, 0.0) + float(jnp.real(numer))

        observables["average_sign"] = avg_sign_step
        self._last_observables = dict(observables)
        self._last_raw_observables = dict(raw_means)
        self._apply_observable_hooks(observables, current_signs=current_signs)
        return observables

    def _apply_observable_hooks(self, observables: Dict[str, Any], current_signs=None):
        """
        Apply custom observable hooks with best-effort sign-aware reduction.

        Hooks may return either scalars or per-chain arrays of shape
        `(num_chains,)`. Per-chain outputs are sign-reweighted and accumulated.
        """
        for hook in self.observable_hooks:
            try:
                extra = hook(
                    self.model,
                    self.sampler.Gs_avg,
                    self.sampler.K_jax,
                    dict(observables),
                    current_signs=current_signs,
                    greens_by_chain=self.sampler.Gs_avg,
                )
            except TypeError:
                extra = hook(self.model, self.sampler.Gs_avg, self.sampler.K_jax, dict(observables))
            if not extra:
                continue

            for name, value in dict(extra).items():
                arr = jnp.asarray(value)
                if current_signs is not None and arr.ndim == 1 and arr.shape[0] == self.sampler.num_chains:
                    numer = jnp.sum(current_signs * arr)
                    self._observable_num[name] = self._observable_num.get(name, 0.0) + float(jnp.real(numer))
                    denom = jnp.sum(current_signs)
                    observables[name] = float(jnp.real(numer / denom)) if float(jnp.abs(denom)) > 1e-14 else float("nan")
                elif arr.ndim == 0:
                    observables[name] = float(jnp.real(arr))
                else:
                    observables[name] = _jsonify(arr)

    def measure_unequal_time(self):
        """
        Return the chain-averaged unequal-time Green's function estimator.
        """
        if self._unequal_num is not None and abs(self._unequal_sign_sum) > 1e-14:
            return self._unequal_num / self._unequal_sign_sum
        return measure_time_displaced(getattr(self, "_unequal_gs", None))

    def measure_diagnostics(self) -> Dict[str, Any]:
        """
        Return basic numerical diagnostics for the current DQMC state.

        The most important number is the equal-time inverse residual: if it
        grows too much, the fast-update path is drifting away from the defining
        Green's-function relation and a full refresh/stabilization step is
        required more often.
        """
        residuals = self.sampler.last_equal_time_residuals
        if residuals is None:
            residuals = self.sampler.compute_equal_time_residuals()
        if self._sign_measurements > 0:
            average_sign = float(self._sign_sum / self._sign_measurements)
            last_average_sign = float(self._avg_sign_history[-1]) if self._avg_sign_history else None
        else:
            current_signs = jnp.asarray(getattr(self.sampler, "current_signs", jnp.ones((self.sampler.num_chains,))))
            average_sign = float(jnp.mean(jnp.real(current_signs)))
            last_average_sign = average_sign
        return {
            "acceptance_rate": float(self._last_acc_rate),
            "green_residual_mean": float(jnp.mean(residuals)),
            "green_residual_max": float(jnp.max(residuals)),
            "forced_refreshes": int(self.sampler.num_forced_refreshes),
            "refresh_drift": float(self.sampler.last_refresh_drift),
            "refresh_strategy": self.sampler.refresh_strategy,
            "residual_check_interval": int(self.sampler.residual_check_interval),
            "sign_policy": self.sign_policy,
            "average_sign": average_sign,
            "last_average_sign": last_average_sign,
            **self.model.get_sign_metadata(),
        }

    def clone(self) -> DQMCSolver:
        """Create a new solver with the same model and sampler settings."""
        return DQMCSolver(
            model       =   self.model.copy(),
            n_stable    =   self.sampler.n_stable,
            num_chains  =   self.sampler.num_chains,
            seed        =   None,
            residual_recompute_threshold=self.sampler.residual_recompute_threshold,
            refresh_strategy=self.sampler.refresh_strategy,
            residual_check_interval=self.sampler.residual_check_interval,
            sign_policy=self.sign_policy,
            observable_hooks=self.observable_hooks,
        )

    def swap(self, other: DQMCSolver):
        """Swap the current auxiliary-field configurations with another solver."""
        self.sampler.configs, other.sampler.configs = other.sampler.configs, self.sampler.configs
        self.sampler.recompute_everything()
        other.sampler.recompute_everything()

    def save_weights(self, directory=None, name="configs"):
        """Save the current auxiliary-field configuration array to disk."""
        if directory:
            path = f"{directory}/{name}.npy"
            np.save(path, np.array(self.sampler.configs))

    def load_weights(self, directory=None, name="configs"):
        """Load auxiliary-field configurations from disk and rebuild cached Green's functions."""
        if directory:
            path = f"{directory}/{name}.npy"
            self.sampler.configs = jnp.array(np.load(path, allow_pickle=False))
            self.sampler.recompute_everything()

    def save_checkpoint(self, path: str | Path):
        """
        Save a restart-oriented checkpoint containing HS fields plus runtime metadata.

        This differs from `DQMCResult.save(...)`: checkpoints are meant for
        continuing a compatible solver, while result bundles are meant for
        analysis and sharing.
        """
        path = Path(path)
        payload = {
            "beta": float(self.model.beta),
            "M": int(self.model.M),
            "n_sites": int(self.model.n_sites),
            "n_hs_fields": int(self.model.n_hs_fields),
            "field_type": str(self.model.field_type),
            "hs": self.model.get_hs_parameters(),
            "sign": self.model.get_sign_metadata(),
            "sampling": self.sampler.get_capabilities(),
        }
        np.savez_compressed(
            path,
            configs=np.asarray(self.sampler.configs),
            metadata=json.dumps(_jsonify(payload), sort_keys=True),
        )

    def load_checkpoint(self, path: str | Path, strict: bool = True):
        """
        Load a checkpoint created by `save_checkpoint(...)` into this solver.

        In strict mode the saved lattice/HS dimensions must match the current
        solver before the auxiliary fields are restored.
        """
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            configs = checkpoint["configs"]
            metadata = json.loads(str(checkpoint["metadata"]))
        if strict:
            expected_shape = (self.sampler.num_chains, self.model.M, self.model.n_hs_fields)
            if tuple(configs.shape) != expected_shape:
                raise ValueError(
                    "Checkpoint shape does not match solver shape: "
                    f"saved={tuple(configs.shape)}, current={expected_shape}."
                )
            if int(metadata.get("n_sites", self.model.n_sites)) != int(self.model.n_sites):
                raise ValueError("Checkpoint lattice size does not match the current solver.")
            if str(metadata.get("field_type", self.model.field_type)) != str(self.model.field_type):
                raise ValueError("Checkpoint field type does not match the current solver.")
        self.sampler.configs = jnp.asarray(configs)
        self.sampler.recompute_everything()


def run_dqmc(
    model,
    *,
    config: DQMCConfig | None = None,
    beta: float | None = None,
    M: int | None = None,
    warmup: int = 0,
    sweeps: int = 10,
    measure_every: int = 1,
    n_stable: int = 10,
    num_chains: int = 1,
    seed: int | None = None,
    collect_unequal: bool = False,
    residual_recompute_threshold: float | None = 1e-6,
    refresh_strategy: str = "jax_udt",
    residual_check_interval: int | None = None,
    sign_policy: str = "strict",
    **kwargs,
) -> DQMCResult:
    """
    Run a simple DQMC simulation and return a compact summary.

    This is the intended high-level entrypoint for the working baseline:
    construct a solver, thermalize, sample, and collect the most useful
    observables/diagnostics in one place.

    Simplest path:
        call ``run_dqmc(model, beta=..., M=...)`` with no explicit HS keyword.
        The adapter picks the standard onsite channel automatically:

        - repulsive onsite Hubbard: magnetic HS,
        - attractive onsite Hubbard: charge HS.

    Math:
        after Trotter decomposition, each slice is

            B_tau = exp(-dtau K) exp(V_tau[s]),

        and the equal-time Green's function is reconstructed from

            G = (I + B_M ... B_1)^(-1).

        The HS choice only changes the diagonal potential ``V_tau[s]`` and the
        sampling rule for ``s``; the stabilized Green-function machinery is the
        same.

    Returns
    -------
    DQMCResult
        Compact result bundle containing the solver object, training summary,
        measured observables, diagnostics, and optional unequal-time data.
    """
    if isinstance(model, DQMCConfig):
        config = model
        model = config.model
    elif config is None:
        config = DQMCConfig(
            model=model,
            beta=beta,
            M=M,
            warmup=warmup,
            sweeps=sweeps,
            measure_every=measure_every,
            n_stable=n_stable,
            num_chains=num_chains,
            seed=seed,
            collect_unequal=collect_unequal,
            residual_recompute_threshold=residual_recompute_threshold,
            refresh_strategy=refresh_strategy,
            residual_check_interval=residual_check_interval,
            sign_policy=sign_policy,
            observable_hooks=tuple(kwargs.get("observable_hooks", ())),
            extra_options={k: v for k, v in kwargs.items() if k != "observable_hooks"},
        )
    else:
        model = config.model if model is None else model
        merged_options = dict(config.extra_options)
        merged_options.update({k: v for k, v in kwargs.items() if k != "observable_hooks"})
        config = DQMCConfig(
            model=model,
            beta=config.beta if beta is None else beta,
            M=config.M if M is None else M,
            warmup=config.warmup if warmup == 0 else warmup,
            sweeps=config.sweeps if sweeps == 10 else sweeps,
            measure_every=config.measure_every if measure_every == 1 else measure_every,
            n_stable=config.n_stable if n_stable == 10 else n_stable,
            num_chains=config.num_chains if num_chains == 1 else num_chains,
            seed=config.seed if seed is None else seed,
            collect_unequal=config.collect_unequal if not collect_unequal else collect_unequal,
            residual_recompute_threshold=(
                config.residual_recompute_threshold
                if residual_recompute_threshold == 1e-6
                else residual_recompute_threshold
            ),
            refresh_strategy=config.refresh_strategy if refresh_strategy == "jax_udt" else refresh_strategy,
            residual_check_interval=(
                config.residual_check_interval
                if residual_check_interval is None
                else residual_check_interval
            ),
            sign_policy=config.sign_policy if sign_policy == "strict" else sign_policy,
            observable_hooks=tuple(kwargs.get("observable_hooks", config.observable_hooks)),
            extra_options=merged_options,
        )

    solver = DQMCSolver(
        model=model,
        beta=config.beta,
        M=config.M,
        n_stable=config.n_stable,
        num_chains=config.num_chains,
        seed=config.seed,
        residual_recompute_threshold=config.residual_recompute_threshold,
        refresh_strategy=config.refresh_strategy,
        residual_check_interval=config.residual_check_interval,
        sign_policy=config.sign_policy,
        observable_hooks=config.observable_hooks,
        **config.extra_options,
    )
    train_params = McsTrain(
        mcth=config.warmup,
        mcsam=config.sweeps,
        mc_corr=max(1, int(config.measure_every)),
        mcchain=config.num_chains,
    )
    train_result = solver.train(train_params, collect_unequal=config.collect_unequal)
    observables = solver.measure_observables()
    diagnostics = solver.measure_diagnostics()

    autocorr_time = None
    if len(solver._energies) >= 2:
        try:
            autocorr_time = float(compute_autocorr_time(np.asarray(solver._energies, dtype=float)))
        except Exception:
            autocorr_time = None

    setup = {
        "hamiltonian": getattr(solver.model.hamiltonian, "_name", type(solver.model.hamiltonian).__name__),
        "hs": solver.model.get_hs_parameters(),
        "sign": {**solver.model.get_sign_metadata(), "sign_policy": solver.sign_policy},
        "sampling": solver.sampler.get_capabilities(),
        "n_channels": int(solver.model.n_channels),
        "n_hs_fields": int(solver.model.n_hs_fields),
        "field_type": str(solver.model.field_type),
    }
    unequal_time = solver.measure_unequal_time() if config.collect_unequal else None
    return DQMCResult(
        solver=solver,
        train_result=train_result,
        observables=observables,
        diagnostics=diagnostics,
        setup=setup,
        energy_history=list(solver._energies),
        energy_autocorr_time=autocorr_time,
        config=config.summary_dict(),
        unequal_time=unequal_time,
    )
