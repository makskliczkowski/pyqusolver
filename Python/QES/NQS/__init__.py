"""
Neural Quantum States for QES.

This package provides the maintained NQS solver stack used for variational
ground states, TDVP time evolution, spectral-response workflows, and entropy
estimators.

State conventions are resolved from the model and Hilbert metadata by default.
The network layer stays general-purpose, while NQS selects a small specialized
adapter path when a faster sampler or evaluation route is available.

Public entry points
-------------------
- ``NQS``: main solver that owns the ansatz, sampler, and resolved state convention.
- ``NQSTrainer``: training and TDVP orchestration layer.

The intended flow is:

    Hamiltonian or model -> NQS -> SR or TDVP -> observables and spectra
    
---------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Version     : 2.0
License     : MIT
---------------------------------
"""

import  importlib
from    dataclasses import MISSING, asdict, dataclass, field, fields
from    typing      import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

# ----------------------------------------------------------------------------
#! Configuration dataclasses
# ----------------------------------------------------------------------------

def _type_to_str(tp: Any) -> str:
    """Return a compact type annotation representation."""
    return str(tp).replace("typing.", "")

def _default_to_repr(dc_field) -> Any:
    """Represent dataclass defaults without invoking factories."""
    if dc_field.default is not MISSING:
        return dc_field.default
    if dc_field.default_factory is not MISSING:
        fac_name = getattr(dc_field.default_factory, "__name__", "factory")
        return f"<{fac_name}>"
    return "<required>"

def _freeze_for_signature(value: Any) -> Any:
    """Convert nested config content into a deterministic hashable form."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_for_signature(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_signature(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_signature(v) for v in value))
    return value

# ----------------------------------------------------------------------------

class _ConfigSchemaMixin:
    """
    Shared config introspection helpers for dataclass-based configs.
    """

    FIELD_GROUPS: ClassVar[Dict[str, Tuple[str, ...]]] = {}

    @classmethod
    def schema(cls, grouped: bool = True) -> Dict[str, Any]:
        """
        Return config fields with type/default metadata.

        Parameters
        ----------
        grouped : bool
            If True, return fields grouped by ``FIELD_GROUPS``.
        """
        base = {}
        for dc_field in fields(cls):
            if not dc_field.init:
                continue
            base[dc_field.name] = {
                "type": _type_to_str(dc_field.type),
                "default": _default_to_repr(dc_field),
            }

        if not grouped or not getattr(cls, "FIELD_GROUPS", None):
            return base

        grouped_schema: Dict[str, Dict[str, Dict[str, Any]]] = {}
        seen = set()
        for group_name, names in cls.FIELD_GROUPS.items():
            group_entries = {k: base[k] for k in names if k in base}
            if group_entries:
                grouped_schema[group_name] = group_entries
                seen.update(group_entries.keys())

        other = {k: v for k, v in base.items() if k not in seen}
        if other:
            grouped_schema["other"] = other

        return grouped_schema

    @classmethod
    def defaults(cls) -> Dict[str, Any]:
        """Return default values for all init fields."""
        out = {}
        for dc_field in fields(cls):
            if dc_field.init:
                out[dc_field.name] = _default_to_repr(dc_field)
        return out

    @classmethod
    def describe(cls, grouped: bool = True) -> str:
        """
        Return a human-readable summary of available kwargs and defaults.
        """
        info = cls.schema(grouped=grouped)
        lines = [f"{cls.__name__} kwargs:"]

        if grouped and cls.FIELD_GROUPS:
            for group_name, entries in info.items():
                lines.append(f"[{group_name}]")
                for key, meta in entries.items():
                    lines.append(f"  - {key}: {meta['type']} = {meta['default']!r}")
        else:
            for key, meta in info.items():
                lines.append(f"  - {key}: {meta['type']} = {meta['default']!r}")

        return "\n".join(lines)

    def help(self, grouped: bool = True) -> None:
        """Print available kwargs, types, and defaults."""
        print(self.__class__.describe(grouped=grouped))

@dataclass
class NQSPhysicsConfig(_ConfigSchemaMixin):
    """
    Configuration for the physical model in NQS.
    Behaves like a dictionary and allows dynamic argument addition.
    """
    model_type      : str                           = "kitaev"      # e.g. "kitaev", "heisenberg", etc.
    lattice_type    : str                           = "honeycomb"   # e.g. "honeycomb", "square", etc.
    lx              : int                           = 3
    ly              : int                           = 3
    lz              : int                           = 1
    bc              : str                           = "pbc"
    num_sites       : Optional[int]                 = None          # Total number of sites (optional if lattice dims are given)
    flux            : Optional[Union[str, Dict]]    = None # Boundary fluxes, see Lattice class for details
    # magnetic field
    hx              : float                         = 0.0
    hy              : float                         = 0.0
    hz              : float                         = 0.0
    # Impurities: list of (site, amplitude) or (site, phi, theta, amplitude)
    impurities      : List[Tuple] = field(default_factory=list)
    # General couplings and other dynamic arguments
    args            : Dict[str, Any] = field(default_factory=dict)

    # ---------------
    # Lazy properties for derived quantities can be added here if needed
    # ---------------
    
    FIELD_GROUPS    : ClassVar[Dict[str, Tuple[str, ...]]] = {
        "model"     : (
            "model_type",
            "hx",
            "hy",
            "hz",
            "impurities",
            "args",
        ),
        "lattice"   : (
            "lattice_type",
            "lx",
            "ly",
            "lz",
            "bc",
            "flux",
        ),
    }

    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        return self.args.get(key)

    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            self.args[key] = value

    def get(self, key, default=None):
        try:
            val = self[key]
            return val if val is not None else default
        except (AttributeError, KeyError):
            return default

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def to_dict(self):
        d       = asdict(self)
        extra   = d.pop("args")
        d.update(extra)
        return d

    def normalized_bc(self) -> Any:
        """Return a boundary-condition value accepted by the lattice layer."""
        if not isinstance(self.bc, str):
            return self.bc

        key = self.bc.strip().lower()
        aliases = {
            "periodic"  : "pbc",
            "open"      : "obc",
            "mixed"     : "mbc",
            "special"   : "sbc",
        }
        return aliases.get(key, key)

    def normalized_lattice_type(self) -> str:
        """
        Return a lattice identifier accepted by the lattice layer.

        ``QES.NQS`` examples and estimator helpers occasionally use 1D aliases
        such as ``"chain"``. The underlying lattice factory exposes these as a
        1D square lattice, so the config layer normalizes them before lattice
        construction or architecture estimation.
        """
        key     = str(self.lattice_type).strip().lower()
        aliases = {"chain": "square", "line": "square", "1d": "square"}
        return aliases.get(key, key)

    def resolved_lattice_dims(self) -> Tuple[int, int, int]:
        """
        Return the effective lattice extents used for construction.

        One-dimensional aliases are represented by the square-lattice backend
        with ``ly = lz = 1``.
        """
        key = str(self.lattice_type).strip().lower()
        if key in {"chain", "line", "1d"}:
            return int(self.lx), 1, 1
        return int(self.lx), int(self.ly), int(self.lz)

    def make_lattice(self):
        """
        Create the lattice declared by this configuration and cache ``num_sites``.
        """
        from QES.general_python.lattices import choose_lattice

        lx, ly, lz      = self.resolved_lattice_dims()
        lattice_kwargs  = self.get("lattice_kwargs", {})
        lattice         = choose_lattice(
                            typek=self.normalized_lattice_type(),
                            lx=lx,
                            ly=ly,
                            lz=lz,
                            bc=self.normalized_bc(),
                            flux=self.flux,
                            **lattice_kwargs,
                        )
        self.num_sites  = int(lattice.ns)
        return lattice

    def signature(self) -> Tuple[Any, ...]:
        """Return a deterministic signature for cache invalidation."""
        lx, ly, lz = self.resolved_lattice_dims()
        return (
            self.model_type,
            self.normalized_lattice_type(),
            lx,
            ly,
            lz,
            self.normalized_bc(),
            _freeze_for_signature(self.flux),
            float(self.hx),
            float(self.hy),
            float(self.hz),
            _freeze_for_signature(self.impurities),
            _freeze_for_signature(self.args),
        )

    def resolve_num_sites(self) -> int:
        """
        Resolve and cache the total number of lattice sites.

        This keeps lightweight network estimation paths working even before a
        full Hamiltonian/model object has been constructed.
        """
        if self.num_sites is not None:
            return int(self.num_sites)
        return int(self.make_lattice().ns)

    # -----------------

    def make_hamiltonian(self, **kwargs):
        """
        Creates a Hamiltonian instance based on this configuration.
        """
        from QES.Algebra.Model              import choose_model
        from QES.Algebra.hilbert            import HilbertSpace

        lattice         = self.make_lattice()

        # Allow overriding hilbert kwargs from args, but also support direct kwargs
        hilbert_kwargs  = self.get("hilbert_kwargs", {})
        if "dtype" in kwargs and "dtype" not in hilbert_kwargs:
            hilbert_kwargs["dtype"] = kwargs["dtype"]
            
        hilbert         = HilbertSpace(lattice=lattice, **hilbert_kwargs)
        model_kwargs    = self.args.copy()

        structural_fields = {
            "model_type",
            "lattice_type",
            "lx",
            "ly",
            "lz",
            "bc",
            "num_sites",
            "flux",
            "hx",
            "hy",
            "hz",
            "impurities",
            "args",
        }
        for dc_field in fields(self):
            if not dc_field.init or dc_field.name in structural_fields:
                continue
            if dc_field.name not in model_kwargs:
                value = getattr(self, dc_field.name)
                if value is not None:
                    model_kwargs[dc_field.name] = value
        
        # Ensure we don't pass lattice_kwargs/hilbert_kwargs to choose_model
        model_kwargs.pop("lattice_kwargs", None)
        model_kwargs.pop("hilbert_kwargs", None)

        model_kwargs.update({
            "lattice"       : lattice,
            "hilbert_space" : hilbert,
            "hx"            : self.hx if self.hx != 0 else None,
            "hy"            : self.hy if self.hy != 0 else None,
            "hz"            : self.hz if self.hz != 0 else None,
            "impurities"    : self.impurities,
        })
                
        # Update with any additional model-related args
        model_kwargs.update(kwargs)

        return choose_model(self.model_type, **model_kwargs), hilbert, lattice

@dataclass
class NQSSolverConfig(_ConfigSchemaMixin):
    """
    Configuration for NQS construction and training defaults.

    This dataclass stores solver-level defaults and can be converted to a
    :class:`NQSTrainConfig` using :meth:`make_train_config`.
    """
    ansatz              : str       = "rbm"
    sampler             : str       = "MCSampler"
    # CPU-friendly defaults. GPU-heavy profiles are selected by NQS.get_auto_config().
    n_chains            : int       = 32
    n_samples           : int       = 256
    n_sweep             : int       = 48
    n_therm             : int       = 24
    lr                  : float     = 1e-2
    epochs              : int       = 300
    dtype               : str       = "complex128"
    backend             : str       = "jax"

    # Training / scheduling defaults
    phases              : Union[str, tuple, None] = None
    lr_scheduler        : Optional[Union[str, Callable]] = None
    reg                 : Optional[float] = None
    reg_scheduler       : Optional[Union[str, Callable]] = None
    diag_shift          : float = 5e-3
    diag_scheduler      : Optional[Union[str, Callable]] = None

    # Sampler controls used in NQS.train(...)
    n_batch             : int = 128
    n_update            : Optional[int] = 1
    upd_fun             : Optional[Union[str, Any]] = "local"
    update_kwargs       : Dict[str, Any] = field(default_factory=dict)
    pt_betas            : Optional[List[float]] = None
    global_p            : Optional[float] = None
    global_update       : Optional[Union[str, Callable]] = None
    global_fraction     : Optional[float] = None

    # TDVP/SR + linear solver defaults
    lin_solver          : Union[str, Callable] = "minres_qlp"
    pre_solver          : Union[str, Callable] = "jacobi"
    ode_solver          : Union[str, Any] = "Euler"
    lin_sigma           : Optional[float] = None
    lin_is_gram         : bool = True
    lin_force_mat       : bool = False
    use_sr              : bool = True
    use_minsr           : bool = False
    rhs_prefactor       : float = -1.0
    grad_clip           : Optional[float] = 1.0
    timing_mode         : str = "detailed"

    # Optimizer and stopping
    optimizer           : Optional[str] = None
    early_stopping      : bool = False
    patience            : int = 50

    # SOTA configuration estimated from physics
    sota_config         : Any = field(default=None, init=False)
    _sota_signature     : Any = field(default=None, init=False, repr=False)

    FIELD_GROUPS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "core": (
            "ansatz",
            "sampler",
            "dtype",
            "backend",
            "epochs",
        ),
        "sampling": (
            "n_batch",
            "n_chains",
            "n_samples",
            "n_sweep",
            "n_therm",
            "n_update",
            "upd_fun",
            "update_kwargs",
            "pt_betas",
            "global_p",
            "global_update",
            "global_fraction",
        ),
        "schedules": (
            "phases",
            "lr",
            "lr_scheduler",
            "reg",
            "reg_scheduler",
            "diag_shift",
            "diag_scheduler",
        ),
        "tdvp_solver": (
            "lin_solver",
            "pre_solver",
            "ode_solver",
            "lin_sigma",
            "lin_is_gram",
            "lin_force_mat",
            "use_sr",
            "use_minsr",
            "rhs_prefactor",
            "grad_clip",
            "timing_mode",
        ),
        "training_utils": (
            "optimizer",
            "early_stopping",
            "patience",
        ),
    }

    def __post_init__(self):
        """
        Lazy estimation can be triggered here if enough information is present.
        """
        pass

    def estimate(self, physics_config: NQSPhysicsConfig, **kwargs):
        """
        Automatically estimates network parameters based on physics configuration.
        """
        from QES.NQS.src.nqs_network_integration import estimate_network_params
        num_sites       = physics_config.resolve_num_sites()
        lattice_dims    = physics_config.resolved_lattice_dims()
        signature       = (
            physics_config.signature(),
            self.ansatz,
            self.dtype,
            _freeze_for_signature(kwargs),
        )

        self.sota_config = estimate_network_params(net_type=self.ansatz,
            num_sites=num_sites,
            lattice_dims=lattice_dims,
            lattice_type=physics_config.normalized_lattice_type(),
            model_type=physics_config.model_type,
            bc=physics_config.normalized_bc(),
            dtype=self.dtype,
            **kwargs
        )
        self._sota_signature = signature
        return self.sota_config

    def make_net(self, physics_config: NQSPhysicsConfig, **kwargs):
        """
        Creates a network instance based on this configuration and a physics configuration.
        """
        from QES.NQS.src.nqs_network_integration import NetworkFactory

        signature = (physics_config.signature(), self.ansatz, self.dtype, _freeze_for_signature(kwargs))
        if self.sota_config is None or self._sota_signature != signature:
            self.estimate(physics_config, **kwargs)

        factory_kwargs = self.sota_config.to_factory_kwargs()
        factory_kwargs.update(kwargs)

        if self.ansatz.strip().lower().replace("-", "_") == "eqgcnn":
            factory_kwargs.setdefault("lattice", physics_config.make_lattice())

        return NetworkFactory.create(network_type=self.ansatz, backend=self.backend, **factory_kwargs)

    def make_train_config(self, **kwargs) -> "NQSTrainConfig":
        """
        Create a training config seeded from this solver config.
        """
        return NQSTrainConfig.from_solver(self, **kwargs)

@dataclass
class NQSTrainConfig(_ConfigSchemaMixin):
    """
    Training configuration helper for :meth:`NQS.train`.

    This class centralizes training defaults and converts them into keyword
    arguments expected by ``NQS.train(...)``.
    """

    # Core loop
    n_epochs            : int = 300
    checkpoint_every    : int = 50
    save_best_checkpoint: bool = True
    best_checkpoint_every: Optional[int] = None
    best_checkpoint_min_delta: float = 0.0
    save_best_stats     : bool = False
    phases              : Union[str, tuple, None] = None
    load_checkpoint     : bool = False
    checkpoint_step     : Optional[Union[int, str]] = None
    override            : bool = True
    reset_weights       : bool = False
    reset_stats         : bool = True
    save_path           : Optional[str] = None

    # Sampling / update controls
    n_batch             : int = 128
    n_update            : Optional[int] = None
    num_samples         : Optional[int] = None
    num_chains          : Optional[int] = None
    num_thermal         : Optional[int] = None
    num_sweep           : Optional[int] = None
    pt_betas            : Optional[List[float]] = None
    upd_fun             : Optional[Union[str, Any]] = None
    update_kwargs       : Optional[Dict[str, Any]] = None
    global_p            : Optional[float] = None
    global_update       : Optional[Union[str, Callable]] = None
    global_fraction     : Optional[float] = None

    # Optimisation schedules and values
    lr                  : Optional[float] = 1e-2
    lr_scheduler        : Optional[Union[str, Callable]] = None
    reg                 : Optional[float] = None
    reg_scheduler       : Optional[Union[str, Callable]] = None
    diag_shift          : float = 5e-3
    diag_scheduler      : Optional[Union[str, Callable]] = None

    # Scheduler aliases forwarded as prefixed kwargs (consumed in NQSTrainer)
    lr_init             : Optional[float] = None
    lr_final            : Optional[float] = None
    lr_decay_rate       : Optional[float] = None
    lr_patience         : Optional[int] = None
    lr_min_delta        : Optional[float] = None
    lr_cooldown         : Optional[int] = None
    lr_step_size        : Optional[int] = None
    lr_max_epochs       : Optional[int] = None
    lr_es               : Optional[int] = None
    lr_es_min           : Optional[float] = None

    reg_init            : Optional[float] = None
    reg_final           : Optional[float] = None
    reg_decay_rate      : Optional[float] = None
    reg_patience        : Optional[int] = None
    reg_min_delta       : Optional[float] = None
    reg_step_size       : Optional[int] = None
    reg_max_epochs      : Optional[int] = None

    diag_init           : Optional[float] = None
    diag_final          : Optional[float] = None
    diag_decay_rate     : Optional[float] = None
    diag_patience       : Optional[int] = None
    diag_min_delta      : Optional[float] = None
    diag_step_size      : Optional[int] = None
    diag_max_epochs     : Optional[int] = None

    # TDVP / SR and linear solver
    lin_solver          : Union[str, Callable] = "minres_qlp"
    pre_solver          : Union[str, Callable] = "jacobi"
    ode_solver          : Union[str, Any] = "Euler"
    tdvp                : Optional[Any] = None
    lin_sigma           : Optional[float] = None
    lin_is_gram         : bool = True
    lin_force_mat       : bool = False
    use_sr              : bool = True
    use_minsr           : bool = False
    rhs_prefactor       : float = -1.0
    grad_clip           : Optional[float] = 10.0

    # Utilities
    optimizer           : Optional[Any] = None
    early_stopper       : Optional[Any] = None
    lower_states        : Optional[List[Any]] = None
    timing_mode         : str = "detailed"
    symmetrize          : Optional[bool] = None
    background          : bool = False
    use_pbar            : bool = True
    patience            : Optional[int] = None

    # Optional exact benchmark
    exact_predictions   : Optional[Any] = None
    exact_method        : Optional[str] = None

    # Extendable bag of kwargs forwarded to NQS.train
    extra_kwargs        : Dict[str, Any] = field(default_factory=dict)

    FIELD_GROUPS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "core_loop": (
            "n_epochs",
            "checkpoint_every",
            "save_best_checkpoint",
            "best_checkpoint_every",
            "best_checkpoint_min_delta",
            "save_best_stats",
            "load_checkpoint",
            "checkpoint_step",
            "override",
            "reset_weights",
            "reset_stats",
            "save_path",
            "phases",
        ),
        "sampling_updates": (
            "n_batch",
            "n_update",
            "num_samples",
            "num_chains",
            "num_thermal",
            "num_sweep",
            "pt_betas",
            "upd_fun",
            "update_kwargs",
            "global_p",
            "global_update",
            "global_fraction",
        ),
        "schedules": (
            "lr",
            "lr_scheduler",
            "reg",
            "reg_scheduler",
            "diag_shift",
            "diag_scheduler",
            "lr_init",
            "lr_final",
            "lr_decay_rate",
            "lr_patience",
            "lr_min_delta",
            "lr_cooldown",
            "lr_step_size",
            "lr_max_epochs",
            "lr_es",
            "lr_es_min",
            "reg_init",
            "reg_final",
            "reg_decay_rate",
            "reg_patience",
            "reg_min_delta",
            "reg_step_size",
            "reg_max_epochs",
            "diag_init",
            "diag_final",
            "diag_decay_rate",
            "diag_patience",
            "diag_min_delta",
            "diag_step_size",
            "diag_max_epochs",
        ),
        "tdvp_solver": (
            "lin_solver",
            "pre_solver",
            "ode_solver",
            "tdvp",
            "lin_sigma",
            "lin_is_gram",
            "lin_force_mat",
            "use_sr",
            "use_minsr",
            "rhs_prefactor",
            "grad_clip",
        ),
        "utilities": (
            "optimizer",
            "early_stopper",
            "lower_states",
            "timing_mode",
            "symmetrize",
            "background",
            "use_pbar",
            "patience",
            "exact_predictions",
            "exact_method",
            "extra_kwargs",
        ),
    }

    KWARG_ALIASES: ClassVar[Dict[str, str]] = {
        "p_global"      : "global_p",
        "lr_initial"    : "lr_init",
        "reg_initial"   : "reg_init",
        "diag_initial"  : "diag_init",
    }

    @classmethod
    def kwargs_aliases(cls) -> Dict[str, str]:
        """Return accepted compatibility aliases for train kwargs."""
        return dict(cls.KWARG_ALIASES)

    @classmethod
    def describe_kwargs(cls, grouped: bool = True, include_aliases: bool = True) -> str:
        """Describe available kwargs and include alias mapping when requested."""
        text = cls.describe(grouped=grouped)
        if include_aliases:
            text += "\nAliases:"
            for alias, canonical in cls.KWARG_ALIASES.items():
                text += f"\n  - {alias} -> {canonical}"
        return text

    @classmethod
    def from_solver(cls, solver_config: "NQSSolverConfig", **kwargs) -> "NQSTrainConfig":
        """
        Build a training config from a solver config.
        """
        return cls(
            n_epochs=solver_config.epochs,
            phases=solver_config.phases,
            n_batch=solver_config.n_batch,
            n_update=solver_config.n_update,
            num_samples=solver_config.n_samples,
            num_chains=solver_config.n_chains,
            num_thermal=solver_config.n_therm,
            num_sweep=solver_config.n_sweep,
            pt_betas=solver_config.pt_betas,
            upd_fun=solver_config.upd_fun,
            update_kwargs=(solver_config.update_kwargs.copy() if solver_config.update_kwargs else None),
            global_p=solver_config.global_p,
            global_update=solver_config.global_update,
            global_fraction=solver_config.global_fraction,
            lr=solver_config.lr,
            lr_scheduler=solver_config.lr_scheduler,
            reg=solver_config.reg,
            reg_scheduler=solver_config.reg_scheduler,
            diag_shift=solver_config.diag_shift,
            diag_scheduler=solver_config.diag_scheduler,
            lin_solver=solver_config.lin_solver,
            pre_solver=solver_config.pre_solver,
            ode_solver=solver_config.ode_solver,
            lin_sigma=solver_config.lin_sigma,
            lin_is_gram=solver_config.lin_is_gram,
            lin_force_mat=solver_config.lin_force_mat,
            use_sr=solver_config.use_sr,
            use_minsr=solver_config.use_minsr,
            rhs_prefactor=solver_config.rhs_prefactor,
            grad_clip=solver_config.grad_clip,
            timing_mode=solver_config.timing_mode,
            optimizer=solver_config.optimizer,
            patience=(solver_config.patience if solver_config.early_stopping else None),
            **kwargs,
        )

    def to_train_kwargs(
        self,
        *,
        solver_config       : Optional["NQSSolverConfig"] = None,
        exact_predictions   : Any = None,
        exact_method        : Optional[str] = None,
        **overrides,
    ) -> Dict[str, Any]:
        """
        Convert config to keyword arguments for :meth:`NQS.train`.
        """
        
        optimizer = self.optimizer
        if optimizer is None and solver_config is not None:
            optimizer   = solver_config.optimizer

        patience = self.patience
        if patience is None and solver_config is not None and solver_config.early_stopping:
            patience    = solver_config.patience

        ex_pred         = self.exact_predictions if exact_predictions is None else exact_predictions
        ex_meth         = self.exact_method if exact_method is None else exact_method

        train_kwargs = {
            "n_epochs"                  : self.n_epochs,
            "checkpoint_every"          : self.checkpoint_every,
            "save_best_checkpoint"      : self.save_best_checkpoint,
            "best_checkpoint_every"     : self.best_checkpoint_every,
            "best_checkpoint_min_delta" : self.best_checkpoint_min_delta,
            "save_best_stats"           : self.save_best_stats,
            "load_checkpoint"           : self.load_checkpoint,
            "checkpoint_step"           : self.checkpoint_step,
            "override"                  : self.override,
            "reset_weights"             : self.reset_weights,
            "reset_stats"               : self.reset_stats,
            "save_path"                 : self.save_path,
            "phases"                    : self.phases,
            "n_batch"                   : self.n_batch,
            "n_update"                  : self.n_update,
            "num_samples"               : self.num_samples,
            "num_chains"                : self.num_chains,
            "num_thermal"               : self.num_thermal,
            "num_sweep"                 : self.num_sweep,
            "pt_betas"                  : self.pt_betas,
            "upd_fun"                   : self.upd_fun,
            "update_kwargs"             : self.update_kwargs,
            "global_p"                  : self.global_p,
            "global_update"             : self.global_update,
            "global_fraction"           : self.global_fraction,
            "lr"                        : self.lr,
            "lr_scheduler"              : self.lr_scheduler,
            "reg"                       : self.reg,
            "reg_scheduler"             : self.reg_scheduler,
            "diag_shift"                : self.diag_shift,
            "diag_scheduler"            : self.diag_scheduler,
            "lin_solver"                : self.lin_solver,
            "pre_solver"                : self.pre_solver,
            "ode_solver"                : self.ode_solver,
            "tdvp"                      : self.tdvp,
            "lin_sigma"                 : self.lin_sigma,
            "lin_is_gram"               : self.lin_is_gram,
            "lin_force_mat"             : self.lin_force_mat,
            "use_sr"                    : self.use_sr,
            "use_minsr"                 : self.use_minsr,
            "rhs_prefactor"             : self.rhs_prefactor,
            "grad_clip"                 : self.grad_clip,
            "timing_mode"               : self.timing_mode,
            "early_stopper"             : self.early_stopper,
            "optimizer"                 : optimizer,
            "lower_states"              : self.lower_states,
            "symmetrize"                : self.symmetrize,
            "background"                : self.background,
            "use_pbar"                  : self.use_pbar,
            "patience"                  : patience,
            "exact_predictions"         : ex_pred,
            "exact_method"              : ex_meth,
        }

        schedule_kwargs = {
            "lr_init"                   : self.lr_init,
            "lr_initial"                : self.lr_init,
            "lr_final"                  : self.lr_final,
            "lr_decay_rate"             : self.lr_decay_rate,
            "lr_patience"               : self.lr_patience,
            "lr_min_delta"              : self.lr_min_delta,
            "lr_cooldown"               : self.lr_cooldown,
            "lr_step_size"              : self.lr_step_size,
            "lr_max_epochs"             : self.lr_max_epochs,
            "lr_es"                     : self.lr_es,
            "lr_es_min"                 : self.lr_es_min,
            "reg_init"                  : self.reg_init,
            "reg_initial"               : self.reg_init,
            "reg_final"                 : self.reg_final,
            "reg_decay_rate"            : self.reg_decay_rate,
            "reg_patience"              : self.reg_patience,
            "reg_min_delta"             : self.reg_min_delta,
            "reg_step_size"             : self.reg_step_size,
            "reg_max_epochs"            : self.reg_max_epochs,
            "diag_init"                 : self.diag_init,
            "diag_initial"              : self.diag_init,
            "diag_final"                : self.diag_final,
            "diag_decay_rate"           : self.diag_decay_rate,
            "diag_patience"             : self.diag_patience,
            "diag_min_delta"            : self.diag_min_delta,
            "diag_step_size"            : self.diag_step_size,
            "diag_max_epochs"           : self.diag_max_epochs,
        }
        train_kwargs.update({k: v for k, v in schedule_kwargs.items() if v is not None})
        train_kwargs.update(self.extra_kwargs)
        train_kwargs.update(overrides)

        return {k: v for k, v in train_kwargs.items() if v is not None}

# ----------------------------------------------------------------------------
# Convenience loading helpers
# ----------------------------------------------------------------------------

@dataclass
class NQSLoadBundle:
    """
    Convenience container returned by :func:`load_nqs`.

    Attributes
    ----------
    nqs : NQS
        Fully constructed NQS instance.
    model, hilbert, lattice
        Physical objects created from :class:`NQSPhysicsConfig`.
    net : Any
        Neural ansatz instance used by NQS.
    physics_config, solver_config
        Effective configs used to construct the bundle.
    checkpoint_metadata : Dict[str, Any]
        Metadata loaded from checkpoint (if available).
    """

    nqs                 : "NQS"
    model               : Any
    hilbert             : Any
    lattice             : Any
    net                 : Any
    physics_config      : NQSPhysicsConfig
    solver_config       : NQSSolverConfig
    checkpoint_metadata : Dict[str, Any] = field(default_factory=dict)


def _as_physics_config(config: Union[NQSPhysicsConfig, Dict[str, Any]]) -> NQSPhysicsConfig:
    if isinstance(config, NQSPhysicsConfig):
        return config
    if isinstance(config, dict):
        return NQSPhysicsConfig(**config)
    raise TypeError("physics_config must be NQSPhysicsConfig or dict.")

def _as_solver_config(config: Union[NQSSolverConfig, Dict[str, Any]]) -> NQSSolverConfig:
    if isinstance(config, NQSSolverConfig):
        return config
    if isinstance(config, dict):
        return NQSSolverConfig(**config)
    raise TypeError("solver_config must be NQSSolverConfig or dict.")

def load_nqs(
    physics_config      : Union[NQSPhysicsConfig, Dict[str, Any]],
    solver_config       : Union[NQSSolverConfig, Dict[str, Any]],
    *,
    checkpoint_step     : Optional[Union[int, str]] = "latest",
    checkpoint_file     : Optional[str] = None,
    load_weights        : bool = True,
    hamiltonian_kwargs  : Optional[Dict[str, Any]] = None,
    net_kwargs          : Optional[Dict[str, Any]] = None,
    nqs_kwargs          : Optional[Dict[str, Any]] = None,
) -> NQSLoadBundle:
    """
    Build ``model + lattice + net + NQS`` in one call and optionally restore checkpoint weights.

    Parameters
    ----------
    physics_config, solver_config
        Config objects (or plain dicts) describing the physical system and NQS defaults.
    checkpoint_step, checkpoint_file
        Passed to ``NQS.load_weights`` when ``load_weights=True``.
    load_weights
        If True, restore NQS parameters from checkpoint after construction.
    hamiltonian_kwargs, net_kwargs, nqs_kwargs
        Extra keyword arguments forwarded to
        ``physics_config.make_hamiltonian(...)``, ``solver_config.make_net(...)``,
        and ``NQS(...)`` respectively.
    """

    p_cfg       = _as_physics_config(physics_config)
    s_cfg       = _as_solver_config(solver_config)

    h_kwargs    = dict(hamiltonian_kwargs or {})
    net_kws     = dict(net_kwargs or {})
    nqs_kws     = dict(nqs_kwargs or {})

    dtype       = nqs_kws.get("dtype", None)
    if dtype is None:
        dtype = getattr(s_cfg, "dtype", None)
    if dtype is not None and "dtype" not in h_kwargs:
        h_kwargs["dtype"] = dtype

    model, hilbert, lattice = p_cfg.make_hamiltonian(**h_kwargs)

    # Avoid backend duplication with NQSSolverConfig.make_net(...),
    # which already forwards ``backend=self.backend``.
    net_backend = net_kws.pop("backend", s_cfg.backend)

    from QES.NQS.src.nqs_network_integration import resolve_model_representation

    representation_info = resolve_model_representation(model, hilbert=hilbert)
    net_kws.setdefault("representation_info", representation_info)
    original_backend = s_cfg.backend
    if net_backend != original_backend:
        s_cfg.backend = net_backend
    try:
        net = s_cfg.make_net(p_cfg, **net_kws)
    finally:
        s_cfg.backend = original_backend

    # Avoid redundant specification of sampler and backend, but allow nqs_kwargs to override solver defaults if desired.
    nqs_kws.setdefault("sampler", s_cfg.sampler)
    nqs_kws.setdefault("backend", net_backend)
    if dtype is not None:
        nqs_kws.setdefault("dtype", dtype)
    if hilbert is not None and hasattr(hilbert, "ns"):
        nqs_kws.setdefault("shape", (int(hilbert.ns),))
    elif hasattr(model, "lattice") and getattr(model, "lattice") is not None:
        nqs_kws.setdefault("shape", (int(model.lattice.ns),))
        
    # Sampler parameters with solver defaults, but allow train_config overrides in nqs_kwargs
    nqs_kws.setdefault("s_numchains", s_cfg.n_chains)
    nqs_kws.setdefault("s_numsamples", s_cfg.n_samples)
    nqs_kws.setdefault("s_sweep_steps", s_cfg.n_sweep)
    nqs_kws.setdefault("s_therm_steps", s_cfg.n_therm)

    nqs_cls = __getattr__("NQS")
    psi     = nqs_cls(logansatz=net, model=model, hilbert=hilbert, **nqs_kws)

    metadata: Dict[str, Any] = {}
    if load_weights:
        psi.load_weights(step=checkpoint_step, filename=checkpoint_file)
        try:
            metadata = psi.ckpt_manager.load_metadata(step=checkpoint_step, filename=checkpoint_file)
        except Exception:
            metadata = {}

    return NQSLoadBundle(
        nqs=psi,
        model=model,
        hilbert=hilbert,
        lattice=lattice,
        net=net,
        physics_config=p_cfg,
        solver_config=s_cfg,
        checkpoint_metadata=metadata,
    )

# ----------------------------------------------------------------------------
# Lazy Import Configuration
# ----------------------------------------------------------------------------

_LAZY_IMPORTS = {
    "ansatze"                           : (".ansatze",                      "__name__"),
    "NQS"                               : (".nqs",                          "NQS"),
    "VMCSampler"                        : ("QES.Solver.MonteCarlo.vmc",     "VMCSampler"),
    "NQSTrainer"                        : (".src.nqs_train",                "NQSTrainer"),
    "NQSTrainStats"                     : (".src.nqs_train",                "NQSTrainStats"),
    "NQSObservable"                     : (".src.nqs_engine",               "NQSObservable"),
    "NQSLoss"                           : (".src.nqs_engine",               "NQSLoss"),
    "NQSEvalEngine"                     : (".src.nqs_engine",               "NQSEvalEngine"),
    "EvaluationResult"                  : (".src.nqs_engine",               "EvaluationResult"),
    "NetworkFactory"                    : (".src.nqs_network_integration",  "NetworkFactory"),
    "TDVP"                              : (".src.tdvp",                     "TDVP"),
    "TDVPStepInfo"                      : (".src.tdvp",                     "TDVPStepInfo"),
    "NQSCorrelatorResult"               : (".src.nqs_spectral",             "NQSCorrelatorResult"),
    "NQSSpectralMapResult"              : (".src.nqs_spectral",             "NQSSpectralMapResult"),
    "NQSTDVPRecord"                     : (".src.nqs_spectral",             "NQSTDVPRecord"),
    "NQSSpectralResult"                 : (".src.nqs_spectral",             "NQSSpectralResult"),
    "NQSPrecisionPolicy"                : (".src.nqs_precision",            "NQSPrecisionPolicy"),
    "resolve_precision_policy"          : (".src.nqs_precision",            "resolve_precision_policy"),
    "cast_for_precision"                : (".src.nqs_precision",            "cast_for_precision"),
    "load_exact_impl"                   : (".src.nqs_exact",                "load_exact_impl"),
    "compute_ed_entanglement_entropy"   : (".src.nqs_entropy",              "compute_ed_entanglement_entropy"),
    "compute_renyi_entropy"             : (".src.nqs_entropy",              "compute_renyi_entropy"),
    "compute_entropy_sweep"             : (".src.nqs_entropy",              "compute_entropy_sweep"),
    "bipartition_cuts"                  : (".src.nqs_entropy",              "bipartition_cuts"),
    "NQSDataset"                        : (".src.nqs_dataset",              "NQSDataset"),
    "EDDataset"                         : (".src.nqs_dataset",              "EDDataset"),
    "CommonDataset"                     : (".src.nqs_dataset",              "CommonDataset"),
}

_LAZY_CACHE = {}

if TYPE_CHECKING:
    from QES.Solver.MonteCarlo.vmc import VMCSampler

    from .nqs                           import NQS
    from .src.nqs_engine                import EvaluationResult, NQSEvalEngine, NQSLoss, NQSObservable
    from .src.nqs_network_integration   import NetworkFactory
    from .src.nqs_train                 import NQSTrainer, NQSTrainStats
    from .src.tdvp                      import TDVP, TDVPStepInfo
    from .src.nqs_spectral              import (
        NQSCorrelatorResult,
        NQSSpectralMapResult,
        NQSSpectralResult,
        NQSTDVPRecord,
    )
    from .src.nqs_precision             import NQSPrecisionPolicy, cast_for_precision, resolve_precision_policy
    from .src.nqs_exact                 import load_exact_impl
    from .src.nqs_entropy               import bipartition_cuts, compute_ed_entanglement_entropy, compute_entropy_sweep, compute_renyi_entropy
    from .src.nqs_dataset               import NQSDataset, EDDataset, CommonDataset


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy imports (PEP 562)."""
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            if module_path.startswith("."):
                module = importlib.import_module(module_path, package=__name__)
            else:
                module = importlib.import_module(module_path)

            result = module if attr_name == "__name__" else getattr(module, attr_name)
            _LAZY_CACHE[name] = result
            return result
        except (ImportError, AttributeError) as e:
            # Special handling for NQS as it's critical
            if name == "NQS":
                raise ImportError(
                    f"Could not import NQS module. Ensure QES is installed correctly.\nOriginal error: {e}"
                ) from e
            # VMCSampler might be optional or handle gracefully
            if name == "VMCSampler":
                return None
            raise ImportError(
                f"Failed to import lazy attribute '{name}' from '{module_path}': {e}"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """Support for dir() and tab completion."""
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

# Metadata
MODULE_DESCRIPTION  = "Neural Quantum States (NQS) with TDVP and Adaptive Scheduling."
__version__         = "2.0.0"

# Helper Functions

def info():
    """Prints the library capability summary."""
    print(__doc__)

def quick_start(mode: str = "ground"):
    """
    Prints a runnable boilerplate script to the console.
    Usage: QES.NQS.quick_start(mode='ground'|'excited')

    Parameters:
    -----------
    mode : str
        'ground' for ground state optimization (default)
    """
    boilerplate = """
# ==========================================
# QES NQS Quick Start Script
# ==========================================
from QES.NQS import NQS, NQSPhysicsConfig, NQSSolverConfig, NQSTrainConfig
"""

    ground_body = """
# 1. Declare the physical system and solver defaults
p_cfg = NQSPhysicsConfig(model_type="kitaev", lattice_type="honeycomb", lx=4, ly=3)
s_cfg = NQSSolverConfig(ansatz="rbm", backend="jax", epochs=300, lr=1e-2)

# 2. Build model + Hilbert + ansatz
model, hilbert, lattice = p_cfg.make_hamiltonian()
net = s_cfg.make_net(p_cfg, alpha=1.0)

# 3. Initialize NQS
psi = NQS(ansatz=net, model=model, hilbert=hilbert, backend=s_cfg.backend)

# 4. Train
train_cfg = NQSTrainConfig.from_solver(s_cfg, phases="kitaev")
stats = psi.train(**train_cfg.to_train_kwargs())
print(f"Final Energy: {stats.history[-1]:.5f}")

# 5. Measure
energy = psi.compute_energy(num_samples=s_cfg.n_samples)
print(f"Measured energy: {energy.mean:.6f} +/- {energy.error_of_mean:.6f}")
"""

    excited_body = """
# 1. Build the shared physical problem
p_cfg = NQSPhysicsConfig(model_type="kitaev", lattice_type="honeycomb", lx=4, ly=3)
model, hilbert, lattice = p_cfg.make_hamiltonian()

# 2. Ground state used as a lower-state penalty target
ground_cfg = NQSSolverConfig(ansatz="rbm", backend="jax", epochs=200)
net_g = ground_cfg.make_net(p_cfg, alpha=1.0)
psi_ground = NQS(ansatz=net_g, model=model, hilbert=hilbert, backend=ground_cfg.backend)
psi_ground.beta_penalty = 10.0

# 3. Excited-state ansatz
excited_cfg = NQSSolverConfig(ansatz="cnn", backend="jax", epochs=200)
net_e = excited_cfg.make_net(p_cfg)
psi_excited = NQS(ansatz=net_e, model=model, hilbert=hilbert, backend=excited_cfg.backend)

# 4. Train against the lower state
train_cfg = NQSTrainConfig.from_solver(excited_cfg, lower_states=[psi_ground], phases="kitaev")
stats = psi_excited.train(**train_cfg.to_train_kwargs())
print(f"Final Energy (Excited): {stats.history[-1]:.5f}")
"""

    if mode.lower() == "excited":
        print(boilerplate + excited_body)
    else:
        print(boilerplate + ground_body)

# --------------------------------------------------------------

# Export
__all__ = [
    # Base classes and core utilities
    "NQS",
    "NQSTrainer",
    "NQSTrainStats",
    "NQSObservable",
    "NQSLoss",
    "NQSEvalEngine",
    "EvaluationResult",
    "TDVP",
    "TDVPStepInfo",
    # Spectral and precision utilities
    "NQSCorrelatorResult",
    "NQSSpectralMapResult",
    "NQSTDVPRecord",
    "NQSSpectralResult",
    "NQSPrecisionPolicy",
    "resolve_precision_policy",
    "cast_for_precision",
    "load_exact_impl",
    # Entropy and dataset utilities
    "compute_ed_entanglement_entropy",
    "compute_renyi_entropy",
    "compute_entropy_sweep",
    "bipartition_cuts",
    # Datasets
    "NetworkFactory",
    "VMCSampler",
    # Configs and loading utilities
    "NQSPhysicsConfig",
    "NQSSolverConfig",
    "NQSTrainConfig",
    "NQSLoadBundle",
    "load_nqs",
    "ansatze",
    # info and quick start
    "quick_start",
    "info",
]

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------
