r"""
Rosenzweig - Porter model (RPM) Hamiltonian and related utilities.

The model is defined by a Hamiltonian matrix with elements that decay as a power-law
with distance from the diagonal, controlled by a parameter gamma. The model can be used to study
localization and delocalization phenomena in disordered systems, as well as the transition between them.

Mathematical definition:
    H_{ij} = \epsilon_i \delta_{ij} + \lambda V_{ij}
where \epsilon_i are random energies drawn from a Gaussian distribution, V_{ij} are random couplings drawn from a GOE distribution, and \lambda is a scaling factor that depends on
the system size and the parameter gamma.

File    : Model/Noninteracting/rpm.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

try:
    from QES.Algebra.hamil                      import Hamiltonian
    from QES.Algebra.hamil_quadratic            import QuadraticHamiltonian
    from QES.general_python.algebra.ran_wrapper import RMT, random_matrix, set_global_seed

    if TYPE_CHECKING:
        from QES.Algebra.hilbert import HilbertSpace

except ImportError as exc:
    raise ImportError("Could not import QES module. Ensure that pyqusolver is installed and accessible.") from exc

# ---------------------------------------------------------------

def _sample_rpm_matrix(nh: int, gamma: float, dtype: np.dtype, seed: Optional[int]) -> np.ndarray:
    if nh <= 0:
        raise ValueError("RPM: matrix dimension must be positive.")

    if seed is not None:
        np.random.seed(int(seed))

    lam     = nh ** (-0.5 * float(gamma))
    diag    = np.random.normal(loc=0.0, scale=1.0, size=nh)
    goe     = random_matrix((nh, nh), typek=RMT.GOE, backend="np", dtype=np.float64)
    hmat    = np.diag(diag) + lam * np.asarray(goe, dtype=np.float64)
    return hmat.astype(dtype, copy=False)

# ---------------------------------------------------------------

class RPM_SP(QuadraticHamiltonian):
    """Single-particle Rosenzweig-Porter model represented by an Ns x Ns matrix."""

    def __init__(self, ns: int, gamma: float,
        *,
        hilbert_space   : Optional["HilbertSpace"] = None,
        dtype           : np.dtype = np.dtype(np.float64),
        backend         : str = "default",
        seed            : Optional[int] = None,
        **kwargs,
    ):
        super().__init__(ns=ns, is_sparse=False, hilbert_space=hilbert_space,
            dtype=dtype, backend=backend, seed=seed, **kwargs,
        )
        self._gamma = float(gamma)
        self._seed = seed
        self._name = "RPM(SP)"
        set_global_seed(self._seed, backend=self._backend)

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def many_body(self) -> bool:
        return False

    def randomize(self, **kwargs):
        if kwargs.get("seed", None) is not None:
            self._seed = int(kwargs["seed"])
        set_global_seed(self._seed, backend=self._backend)
        self._hamiltonian_quadratic(use_numpy=kwargs.get("use_numpy", True))

    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        self._log(f"SP: Building RPM Hamiltonian with gamma={self._gamma}, seed={self._seed}", lvl=2, color="green", log="debug")
        hmat            = _sample_rpm_matrix(self._ns, self._gamma, self._dtype, self._seed)
        xp              = np if use_numpy else self._backend
        self._hamil_sp  = hmat if xp is np else xp.asarray(hmat, dtype=self._dtype)

    @staticmethod
    def repr(**kwargs) -> str:
        ns  = kwargs.get("ns", "?")
        g   = kwargs.get("gamma", 1.0)
        return f"RPM(ns={ns},g={g:.3f},mb=0)"

    def __repr__(self):
        return self.repr(ns=self._ns, gamma=self._gamma)

# ---------------------------------------------------------------------

class RPM_MB(Hamiltonian):
    """Many-body Rosenzweig-Porter model represented directly in the full Nh x Nh space."""

    def __init__( self, ns: int, gamma: float, *,
        hilbert_space   : Optional["HilbertSpace"] = None,
        dtype           : np.dtype = np.dtype(np.float64),
        backend         : str = "default",
        seed            : Optional[int] = None,
        **kwargs,
    ):
        if hilbert_space is None:
            from QES.Algebra.hilbert import HilbertSpace
            hilbert_space = HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)

        super().__init__(is_manybody=True, hilbert_space=hilbert_space,
            is_sparse=False, seed=seed, dtype=dtype, backend=backend, **kwargs)
        self._gamma     = float(gamma)
        self._seed      = seed
        self._name      = "RPM(MB)"
        set_global_seed(self._seed, backend=self._backend)

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def many_body(self) -> bool:
        return True

    # --------------------------------------------------------------

    def randomize(self, **kwargs):
        if kwargs.get("seed", None) is not None:
            self._seed = int(kwargs["seed"])
        set_global_seed(self._seed, backend=self._backend)
        self._hamiltonian(use_numpy=kwargs.get("use_numpy", True))

    def _hamiltonian(self, use_numpy: bool = False):
        self._log(f"MB: Building RPM Hamiltonian with gamma={self._gamma}, seed={self._seed}", lvl=2, color="green", log="debug")
        hmat        = _sample_rpm_matrix(self._nh, self._gamma, self._dtype, self._seed)
        xp          = np if use_numpy else self._backend
        self._hamil = hmat if xp is np else xp.asarray(hmat, dtype=self._dtype)
        self._hamiltonian_validate()

    # --------------------------------------------------------------

    def _set_local_energy_operators(self):
        pass

    @staticmethod
    def repr(**kwargs) -> str:
        ns  = kwargs.get("ns", "?")
        g   = kwargs.get("gamma", 1.0)
        return f"RPM(ns={ns},g={g:.3f}, mb=1)"

    def __repr__(self):
        return self.repr(ns=self._ns, gamma=self._gamma)
    
# ---------------------------------------------------------------------

class RosenzweigPorter:
    """Compatibility wrapper delegating to RPM_SP or RPM_MB."""

    def __new__(cls, *args, many_body: bool = True, **kwargs):
        impl = RPM_MB if many_body else RPM_SP
        return impl(*args, **kwargs)

RPM = RosenzweigPorter

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------