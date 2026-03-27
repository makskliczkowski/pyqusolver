r"""
Implementation of the Power-Law Random Banded (PLRB) model in both single-particle and many-body forms.
The PLRB model is defined by a Hamiltonian matrix with elements that decay as a power-law
with distance from the diagonal, controlled by parameters a and b. The model can be used to study
localization and delocalization phenomena in disordered systems.

The model has the decay of the off-diagonal elements given by:
    H_{ij} \sim \frac{1}{\sqrt{1 + (|i-j|/b)^{2a}}}
where a controls the power-law decay and b sets the scale of the band width. The diagonal elements are random and typically drawn from a Gaussian distribution.

----------------------------------------------------
file    : Model/Noninteracting/plrb.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Version : 2.0
Modified: March 2026
------------------------------------------------
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

try:
    from QES.Algebra.hamil                      import Hamiltonian
    from QES.Algebra.hamil_quadratic            import QuadraticHamiltonian
    from QES.general_python.algebra.ran_wrapper import set_global_seed

    if TYPE_CHECKING:
        from QES.Algebra.hilbert import HilbertSpace

except ImportError as exc:
    raise ImportError("Could not import QES module. Ensure that pyqusolver is installed and accessible.") from exc

try:
    import numba
except ImportError:
    numba = None

# --------------------------------------------------------------

if numba is not None:
    @numba.njit(cache=True)
    def _fill_plrb_numba(hmat: np.ndarray, a: float, b: float, seed: int) -> None:
        if seed >= 0:
            np.random.seed(seed)

        n       = hmat.shape[0]
        inv_b   = 1.0 / b
        power   = 2.0 * a

        for i in range(n):
            for j in range(i, n):
                rnd = 2.0 * np.random.random() - 1.0
                if i == j:
                    val     = rnd
                else:
                    dist    = abs(i - j) * inv_b
                    val     = rnd / np.sqrt(1.0 + dist**power)
                hmat[i, j]  = val
                hmat[j, i]  = val

def _sample_plrb_matrix(nh: int, a: float, b: float, dtype: np.dtype, seed: Optional[int], use_numba: bool) -> np.ndarray:
    if nh <= 0:
        raise ValueError("PLRB: matrix dimension must be positive.")
    if b <= 0.0:
        raise ValueError("PLRB: parameter b must be > 0.")

    hmat        = np.zeros((nh, nh), dtype=np.float64)
    seed_int    = -1 if seed is None else int(seed)

    if use_numba and numba is not None:
        _fill_plrb_numba(hmat, float(a), float(b), seed_int)
    else:
        rng     = np.random.default_rng(seed)
        power   = 2.0 * float(a)
        inv_b   = 1.0 / float(b)
        for i in range(nh):
            for j in range(i, nh):
                rnd = rng.uniform(-1.0, 1.0)
                if i == j:
                    val     = rnd
                else:
                    dist    = abs(i - j) * inv_b
                    val     = rnd / np.sqrt(1.0 + dist**power)
                hmat[i, j]  = val
                hmat[j, i]  = val

    return hmat.astype(dtype, copy=False)

# --------------------------------------------------------------

class PLRB_SP(QuadraticHamiltonian):
    """Single-particle PLRB model represented by an Ns x Ns matrix."""

    def __init__(
        self,
        ns              : int,
        a               : float = 1.0,
        b               : float = 1.0,
        *,
        hilbert_space   : Optional["HilbertSpace"] = None,
        dtype           : np.dtype = np.dtype(np.float64),
        backend         : str = "default",
        seed            : Optional[int] = None,
        use_numba       : bool = True,
        **kwargs,
    ):
        kwargs.pop("many_body", None)
        super().__init__(
            ns              = ns,
            is_sparse       = False,
            hilbert_space   = hilbert_space,
            dtype           = dtype,
            backend         = backend,
            seed            = seed,
            **kwargs,
        )
        self._a         = float(a)
        self._b         = float(b)
        self._seed      = seed
        self._use_numba = bool(use_numba)
        self._name      = "PLRB_SP"
        set_global_seed(self._seed, backend=self._backend)

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def many_body(self) -> bool:
        return False

    # --------------------------------------------------------------

    def randomize(self, **kwargs):
        if kwargs.get("seed", None) is not None:
            self._seed = int(kwargs["seed"])
        set_global_seed(self._seed, backend=self._backend)
        self._hamiltonian_quadratic(use_numpy=kwargs.get("use_numpy", True))

    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        self._log(f"SP: Setting PLRB Hamiltonian with a={self._a}, b={self._b}, seed={self._seed}", lvl=2, color="green", log="debug",)
        hmat            = _sample_plrb_matrix(self._ns, self._a, self._b, self._dtype, self._seed, self._use_numba)
        xp              = np if use_numpy else self._backend
        self._hamil_sp  = hmat if xp is np else xp.asarray(hmat, dtype=self._dtype)

    @staticmethod
    def repr(**kwargs) -> str:
        ns  = kwargs.get("ns", "?")
        a   = kwargs.get("a", 1.0)
        b   = kwargs.get("b", 1.0)
        return f"PLRB(ns={ns},a={a:.3f},b={b:.3f},mb=0)"

    def __repr__(self):
        return self.repr(ns=self._ns, a=self._a, b=self._b)

    def __str__(self):
        return self.__repr__()

# --------------------------------------------------------------

class PLRB_MB(Hamiltonian):
    """Many-body PLRB model represented directly in the full Nh x Nh space."""

    def __init__(
        self,
        ns: int,
        a: float = 1.0,
        b: float = 1.0,
        *,
        hilbert_space: Optional["HilbertSpace"] = None,
        dtype: np.dtype = np.dtype(np.float64),
        backend: str = "default",
        seed: Optional[int] = None,
        use_numba: bool = True,
        **kwargs,
    ):
        kwargs.pop("many_body", None)
        if hilbert_space is None:
            from QES.Algebra.hilbert import HilbertSpace
            hilbert_space = HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)

        super().__init__(
            is_manybody=True, hilbert_space=hilbert_space,
            is_sparse=False, seed=seed, dtype=dtype, backend=backend, **kwargs,
        )
        self._a         = float(a)
        self._b         = float(b)
        self._seed      = seed
        self._use_numba = bool(use_numba)
        self._name      = "PLRB(MB)"
        set_global_seed(self._seed, backend=self._backend)

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def many_body(self) -> bool:
        return True

    def randomize(self, **kwargs):
        if kwargs.get("seed", None) is not None:
            self._seed = int(kwargs["seed"])
        set_global_seed(self._seed, backend=self._backend)
        self._hamiltonian(use_numpy=kwargs.get("use_numpy", True))

    # --------------------------------------------------------------

    def _hamiltonian(self, use_numpy: bool = False):
        self._log(f"MB: Setting PLRB Hamiltonian with a={self._a}, b={self._b}, seed={self._seed}", lvl=2, color="green", log="debug",)
        hmat        = _sample_plrb_matrix(self._nh, self._a, self._b, self._dtype, self._seed, self._use_numba)
        xp          = np if use_numpy else self._backend
        self._hamil = hmat if xp is np else xp.asarray(hmat, dtype=self._dtype)
        self._hamiltonian_validate()

    def _set_local_energy_operators(self):
        pass

    @staticmethod
    def repr(**kwargs) -> str:
        ns  = kwargs.get("ns", "?")
        a   = kwargs.get("a", 1.0)
        b   = kwargs.get("b", 1.0)
        return f"PLRB(ns={ns},a={a:.3f},b={b:.3f},mb=1)"

    def __repr__(self):
        return self.repr(ns=self._ns, a=self._a, b=self._b)

    def __str__(self):
        return self.__repr__()

# --------------------------------------------------------------

class PowerLawRandomBanded:
    """Compatibility wrapper delegating to PLRB_SP or PLRB_MB."""

    def __new__(cls, *args, many_body: bool = True, **kwargs):
        ''' 
        Parameters:
        -----------
        many_body (bool): 
            If True, returns a many-body PLRB model (PLRB_MB).
            If False, returns a single-particle PLRB model (PLRB_SP).
        args, kwargs:
            Parameters for the PLRB model constructor (e.g. ns, a, b, seed, use_numba).
        
        '''
        impl = PLRB_MB if many_body else PLRB_SP
        return impl(*args, **kwargs)

PLRB = PowerLawRandomBanded

# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------