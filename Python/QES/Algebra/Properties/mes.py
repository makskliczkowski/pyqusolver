r'''
Here, a minimum entangled state (MES) finder is implemented using JAX autodiff
optimization in coefficient space. The main function `find_mes`
performs multiple random restarts to find distinct MES, enforcing orthogonality
directly in coefficient space during the search. 
This approach allows us to identify states in the ground state manifold that minimize 
the entanglement entropy for a given partition, which is crucial 
for extracting topological entanglement entropy and understanding the underlying topological order.

Perform extraction of anyonic statistics from ED ground states using Minimum Entangled States (MES).

Overview
--------
On a torus, a topologically ordered phase with N anyon types has an N-fold
degenerate ground-state manifold H_GS. Exact diagonalization yields an
orthonormal basis of ground states

    { |phi_alpha> } for alpha = 1, ..., m,

with m = dim H_GS. These states are collected into a matrix

    V in C^{Ns x m},

whose columns are the basis vectors:

    V[:, alpha] = |phi_alpha>.

For any complex coefficient vector c in C^m with ||c||_2 = 1, a normalized
ground state is given by

    |psi(c)> = V @ c.

To probe topological order, one considers non-contractible real-space
bipartitions that wrap around the torus. For a given cut direction mu
(for example, a cylinder cut along x or y), define the bipartition
A_mu union B_mu and the reduced density matrix

    rho_A^(mu)(c) = Tr_{B_mu} |psi(c)><psi(c)|.

The entanglement entropy for this cut is

    S^(mu)(c) = S_A^(mu)(|psi(c)>) = -Tr rho_A^(mu)(c) log rho_A^(mu)(c).

In a topological phase, for sufficiently large systems the entanglement
entropy of the ground states obeys

    S_a^(mu) = alpha * L_mu - gamma_a,

where L_mu is the linear size of the cut in direction mu, alpha is a
non-universal area-law coefficient, and gamma_a is the topological
contribution associated with anyon type a. One can write

    gamma_a = log(D / d_a),

where d_a is the quantum dimension of anyon a, and D is the total
quantum dimension,

    D = sqrt(sum_a d_a^2).

For a fixed cut mu, the area-law piece alpha * L_mu is the same for all
ground states in H_GS, while gamma_a depends on the anyon sector. Thus,
minimizing S^(mu)(c) over normalized c picks out a state with minimal
topological contribution and, in the ideal case, definite anyon type
crossing that cut. Such a state is called a Minimum Entangled State (MES)
for the cut mu.

Pipeline
--------
1. ED ground-space basis:
   Use exact diagonalization on a torus cluster to obtain the degenerate
   ground-state manifold H_GS and construct the matrix V whose columns
   form an orthonormal basis of this manifold.

2. Entanglement entropy functional:
   For a chosen non-contractible bipartition A_mu | B_mu, implement
   a function S_func(psi) that computes the entanglement entropy
   S_A^(mu)(psi) of a given many-body state |psi> with respect to this cut.
   In the MES framework, this is used as

       S^(mu)(c) = S_func(V @ c).

3. MES search for a given cut:
   For the chosen cut mu, search over normalized coefficient vectors
   c in C^m to minimize S^(mu)(c). The global minimum defines a MES

       |Xi_0^(mu)> = |psi(c_0^(mu))>.

   To obtain a complete MES basis, impose orthogonality constraints in
   coefficient space and repeat the minimization in the orthogonal
   complement to the previously found MES. This yields a set

       { |Xi_a^(mu)> } for a = 0, ..., m - 1,

   that spans H_GS and is adapted to the chosen non-contractible cut.

4. Two perpendicular cuts and modular S-matrix:
   Repeat the MES construction for two perpendicular non-contractible
   cuts, e.g. mu = x and nu = y, to obtain two MES bases

       { |Xi_a^(x)> } and { |Xi_b^(y)> }.

   The overlap matrix

       O_ab = <Xi_a^(x) | Xi_b^(y)>,

   computed between these two MES bases, encodes the modular S-matrix
   up to phases and permutations. In the ideal topological limit one has

       S_ab = (1 / D) * O_ab,

   where D is the total quantum dimension. The first row (or column)
   of S corresponds to the overlaps with the vacuum anyon sector and
   satisfies

       S_0a = d_a / D.

5. Extraction of quantum dimensions and statistics:
   After identifying which MES corresponds to the vacuum anyon (typically
   the MES with the smallest entanglement entropy), reorder and rephase
   the MES bases so that the vacuum is at index 0 and S_00 is real and
   positive. Then, from the modular S-matrix one obtains

       D   = 1 / S_00,
       d_a = S_0a / S_00.

   These quantum dimensions characterize the anyon types. Additional
   modular data (such as the T-matrix) can be accessed by combining the
   MES basis with twisted boundary conditions or Dehn twists. Together,
   the modular matrices S and T encode the mutual and self statistics of
   anyons and provide a fingerprint of the underlying topological order.

In this module, the MESFinder class and the find_mes helper are used to
minimize S^(mu)(c) numerically within the ED ground-state manifold, while
compute_modular_s_matrix constructs the overlap matrix O_ab between two
MES bases corresponding to perpendicular cuts and uses it to extract a
normalized modular S-matrix and quantum dimensions.

JAX backend:
- `method='jax-grad'`: direct JAX autodiff optimization (requires JAX-traceable S_func)

--------------------------------
Author      : Maksymilian Kliczkowski
Created     : 2026-02-12
Version     : 2.0
Updates     : - Added JAX-based optimization backend for potentially faster convergence.
--------------------------------
'''

from    __future__ import annotations

import  os
import  numpy as np
from    typing import Callable, List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from    dataclasses import dataclass
from    pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax             = None
    jnp             = None
    JAX_AVAILABLE   = False

if TYPE_CHECKING:
    from QES.general_python.lattices.lattice    import Lattice
    from QES.general_python.common.flog         import Logger

# ---------------------------------------
#! Topological Results Dataclass
# ---------------------------------------

@dataclass
class TopologicalResults:
    r"""
    Data class to store results of topological analysis from MES.
    
    Attributes
    ----------
    S_matrix : np.ndarray
        The modular S-matrix (normalized overlap matrix). S_ij = <MES_i|MES_j> / D, where D is the total quantum dimension.
    overlap_matrix : np.ndarray
        The raw overlap matrix between two MES bases. This is the unnormalized version of S_matrix
        and can be used to extract quantum dimensions.
    quantum_dimensions : np.ndarray
        Individual quantum dimensions d_i for each anyon type.
    total_quantum_dimension : float
        The total quantum dimension D = sqrt(sum d_i^2). This is a key quantity characterizing the topological order.
    is_abelian : bool
        True if all d_i are approximately 1.
    is_non_abelian : bool
        True if any d_i > 1 (within tolerance).
    """
    S_matrix                : np.ndarray
    overlap_matrix          : np.ndarray
    quantum_dimensions      : np.ndarray
    total_quantum_dimension : float
    is_abelian              : bool
    is_non_abelian          : bool

    def __repr__(self) -> str:
        return (f"TopologicalResults(D={self.total_quantum_dimension:.4f}, "
                f"Abelian={self.is_abelian}, Non-Abelian={self.is_non_abelian}, "
                f"d_i={self.quantum_dimensions})")

# ---------------------------------------
#! Minimum Entangled States (MES) Finder
# ---------------------------------------

def _gauge_fix_first_component(c: np.ndarray) -> np.ndarray:
    '''
    Gauge-fix global phase so c[0] is real and non-negative.
    This makes the optimization landscape smoother and helps with convergence.

    Parameters
    ----------
    c : np.ndarray
        Coefficient vector to be gauge-fixed.    
    '''
    c = np.asarray(c, dtype=np.complex128).copy()
    if c.size == 0:
        return c

    c0_abs = np.abs(c[0])
    if c0_abs > 1e-15:
        c *= np.exp(-1j * np.angle(c[0]))

    c[0] = c[0].real + 0.0j
    if c[0].real < 0.0:
        c *= -1.0
        c[0] = c[0].real + 0.0j
    return c

def project_onto_complement(c_raw: np.ndarray, c_list: List[np.ndarray]) -> np.ndarray:
    """
    Project c_raw onto the orthogonal complement of c_list and renormalize.
    Falls back to a random vector if projection collapses numerically.
    The function orthogonalizes c_raw against each vector in c_list and normalizes the result. 
    If the resulting vector has a very small norm (indicating near-linear dependence), it generates a random vector and repeats the process. 
    This ensures that the returned vector is numerically stable and orthogonal to all vectors in c_list.
    
    Parameters
    ----------
    c_raw : np.ndarray
        The raw coefficient vector to be projected.
    c_list : List[np.ndarray]
        List of coefficient vectors to be projected against (orthogonalized against).
    """
    c = c_raw.astype(np.complex128, copy=True)
    for c_prev in c_list:
        c -= np.vdot(c_prev, c) * c_prev

    # Normalize the state
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        c = np.random.randn(*c_raw.shape) + 1j * np.random.randn(*c_raw.shape)
        for c_prev in c_list:
            c -= np.vdot(c_prev, c) * c_prev
        nrm = np.linalg.norm(c)
        if nrm < 1e-14:
            c       = np.zeros_like(c_raw, dtype=np.complex128)
            c[0]    = 1.0
            for c_prev in c_list:
                c -= np.vdot(c_prev, c) * c_prev
            nrm = np.linalg.norm(c)
            if nrm < 1e-14:
                c       = np.zeros_like(c_raw, dtype=np.complex128)
                c[0]    = 1.0
                nrm     = 1.0

    return c / nrm

class MESFinder:
    ''' Class to find Minimum Entangled States (MES) by minimizing the entanglement entropy. '''
    
    def __init__(self, V: np.ndarray, S_func: Callable[[np.ndarray], float],
                 *,
                 S_func_c: Optional[Callable[[np.ndarray], float]] = None,
                 logger: Optional['Logger'] = None):
        """
        Initialize the MESFinder.
        
        Parameters
        ----------
        V : np.ndarray
            A matrix of shape (Ns, m) where m is the dimension of the degenerate manifold.
            Each column is a basis state of the manifold.
        S_func : Callable[[np.ndarray], float]
            A function that takes a state vector of size Ns and returns its entanglement entropy.
        S_func_c : Optional[Callable[[np.ndarray], float]]
            **Fast path**: a JAX-traceable function that takes a *coefficient* vector c of size m
            and returns the entropy directly (without requiring V @ c first).  When provided, the
            optimization hot-loop bypasses the large matrix-vector product and the Ns-length gather,
            dramatically reducing per-step cost for large Hilbert spaces.
            Build this via :func:`find_mes_save` or manually with precomputed Schmidt matrices.
        logger : Optional['Logger']
            An optional logger object to log messages.
        """
        self.V          = V
        self.S_func     = S_func
        self._S_func_c  = S_func_c      # coefficient-space fast path (takes c, not psi)
        self.m          = V.shape[1]
        self.nh         = V.shape[0]
        # Pre-allocate buffer for state vector to avoid multiple large allocations.
        # Ensure it is complex as coefficients c are complex.
        self._psi_buf   = np.zeros(self.nh, dtype=np.result_type(V.dtype, np.complex128))
        self.logger     = logger if logger is not None else None
        
    def _log(self, message: str, *, verbose: bool = False, lvl: int = 0, color: Optional[str] = None):
        ''' Helper method to log messages using the provided logger. '''
        
        if not verbose:
            return
        
        if self.logger:
            self.logger.info(message, lvl=lvl, color=color)
        else:
            print(message)
        
    # --------------------------------
    #! Utility methods
    # --------------------------------
        
    @staticmethod
    def normalize(c):
        ''' Normalize a complex vector. '''
        nrm = np.linalg.norm(c)
        if nrm < 1e-15: 
            return c
        return c / nrm

    @staticmethod
    def random_c(m, rng=None):
        ''' Generate random complex vector and normalize it. '''
        if rng is None:
            c = np.random.randn(m) + 1j*np.random.randn(m)
        else:
            c = rng.standard_normal(m) + 1j*rng.standard_normal(m)
        return MESFinder.normalize(c)

    # --------------------------------
    #! Find
    # --------------------------------
    
    def __call__(self, c):
        ''' Compute the entropy S(V @ c) for a given vector c. '''
        
        # Use fast coefficient-space path when available (avoids Ns-length work).
        if self._S_func_c is not None:
            return float(self._S_func_c(jnp.asarray(c)))
        # Fallback: optimized matrix-vector multiplication into pre-allocated buffer.
        np.dot(self.V, c, out=self._psi_buf)
        return self.S_func(self._psi_buf)

    # --------------------------------
    #! Optimization methods
    # --------------------------------

    def _minimize_jax_grad(
        self,
        c0                  : np.ndarray,
        max_iter            : int = 200,
        tol                 : float = 1e-10,
        lr                  : float = 0.05,
        beta1               : float = 0.9,
        beta2               : float = 0.999,
        adam_eps            : float = 1e-8,
        patience            : int = 30,
        c_constraints       : Optional[List[np.ndarray]] = None,
        verbose             : bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Direct autodiff optimization in coefficient space with JAX value-and-grad.
        Requires `S_func` to be JAX-traceable.
        
        Parameters
        ----------
        c0 : np.ndarray
            Initial coefficient vector (should be normalized).
        max_iter : int
            Maximum number of optimization steps.
        tol : float
            Tolerance for convergence (based on change in objective value).
        lr : float
            Learning rate for Adam optimizer.
        beta1 : float
            Beta1 parameter for Adam optimizer.
        beta2 : float
            Beta2 parameter for Adam optimizer.
        adam_eps : float
            Epsilon parameter for Adam optimizer to prevent division by zero.
        patience : int
            Number of consecutive steps with small improvement before stopping.
        c_constraints : Optional[List[np.ndarray]]
            List of coefficient vectors to be orthogonalized against (for finding multiple MES).
        verbose : bool
            If True, print optimization progress.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available.")
        c_constraints = [] if c_constraints is None else c_constraints

        m       = self.m
        c0      = self.normalize(c0)
        if c_constraints:
            c0  = project_onto_complement(c0, c_constraints)
            
        # Gauge-fix initial vector to improve optimization landscape smoothness.
        c0                  = _gauge_fix_first_component(c0)
        params              = jnp.concatenate([jnp.asarray(c0.real), jnp.asarray(c0.imag)])
        c_constraints_jax   = [jnp.asarray(c_prev) for c_prev in c_constraints] # Convert constraints to JAX arrays for use in the optimization loop.

        # When a coefficient-space entropy function is available, skip the Ns-length
        # matrix-vector product and the random gather inside S_func entirely —
        # the precomputed Schmidt matrices live in a (m, dA, dB) tensor in L2/L3 cache.
        use_fast_path = self._S_func_c is not None
        if not use_fast_path:
            Vj = jnp.asarray(self.V)

        # Helper function to convert real parameters back to complex coefficients, applying constraints and gauge-fixing.
        def _params_to_c(theta: "jnp.ndarray") -> "jnp.ndarray":
            c   = theta[:m] + 1j * theta[m:]
            nrm = jnp.linalg.norm(c)
            c   = c / jnp.maximum(nrm, 1e-15)
            if c_constraints_jax:
                for c_prev in c_constraints_jax:
                    c       = c - jnp.vdot(c_prev, c) * c_prev # Project onto orthogonal complement of constraints
                nrm2    = jnp.linalg.norm(c)
                c       = c / jnp.maximum(nrm2, 1e-15)

            abs0    = jnp.abs(c[0])
            phase   = jnp.where(abs0 > 1e-15, jnp.exp(-1j * jnp.angle(c[0])), jnp.ones((), dtype=c.dtype))
            c       = c * phase
            c       = jnp.where(jnp.real(c[0]) < 0.0, -c, c)
            # Explicitly zero the imaginary part of c[0] using the array's own dtype.
            c       = c.at[0].set(jnp.real(c[0]).astype(c.dtype))
            return c

        # Loss function: fast path takes c directly (no Ns-length work in hot loop);
        # fallback does V@c then calls the psi-level S_func.
        if use_fast_path:
            def _loss(theta: "jnp.ndarray") -> "jnp.ndarray":
                c   = _params_to_c(theta)
                val = self._S_func_c(c)
                return jnp.real(jnp.asarray(val))
        else:
            def _loss(theta: "jnp.ndarray") -> "jnp.ndarray":
                c   = _params_to_c(theta)
                psi = Vj @ c
                val = self.S_func(psi)
                return jnp.real(jnp.asarray(val))

        # Validate that S_func is JAX-traceable by computing value and gradient at the initial point. This will raise an error if S_func cannot be traced.
        # JIT-compile value_and_grad once — eliminates per-step Python re-tracing overhead.
        try:
            value_and_grad  = jax.jit(jax.value_and_grad(_loss))
            val0, grad0     = value_and_grad(params)
            _               = float(val0)
            _               = np.asarray(grad0)
        except Exception as exc:
            raise RuntimeError("method='jax-grad' requires S_func to be JAX-traceable; build S_func with JAX-based density/entropy kernels.") from exc

        # Initialize Adam optimizer state
        mom1        = jnp.zeros_like(params)
        mom2        = jnp.zeros_like(params)
        best_params = params
        best_val    = float(val0)
        prev_val    = best_val
        stall       = 0
        nfev        = 1
        # Precompute log of beta values for numerically stable bias correction.
        log_b1      = np.log(beta1)
        log_b2      = np.log(beta2)
        # Use a relative + absolute stall criterion so early stopping actually fires.
        _stall_rtol = max(tol * 1e6, 1e-6)
        _stall_atol = tol
        every       = int(kwargs.get('every', 10))
        # Amortise float() device-syncs: only materialise val every check_every steps.
        # This lets JAX pipeline multiple XLA dispatches between synchronisations.
        check_every = int(kwargs.get('check_every', 5))

        # Optimization loop with Adam updates.
        # float() is called only for logging/convergence checks, not to drive the update —
        # all JAX operations remain on-device between iterations.
        for step in range(1, max_iter + 1):
            val, grad   = value_and_grad(params)
            # Adam update — precomputed bias corrections avoid repeated Python ** ops.
            nfev       += 1
            mom1        = beta1 * mom1 + (1.0 - beta1) * grad
            mom2        = beta2 * mom2 + (1.0 - beta2) * (grad * grad)
            bc1         = 1.0 - np.exp(step * log_b1)
            bc2         = 1.0 - np.exp(step * log_b2)
            mom1_hat    = mom1 / bc1
            mom2_hat    = mom2 / bc2
            params      = params - lr * mom1_hat / (jnp.sqrt(mom2_hat) + adam_eps)

            # Only sync to host every check_every steps to allow JAX to pipeline dispatches.
            if step % check_every == 0 or step == max_iter:
                valf        = float(val)
                if valf < best_val:
                    best_val    = valf
                    best_params = params

                # Relative + absolute improvement criterion — much more likely to fire.
                improvement = abs(valf - prev_val)
                if improvement < _stall_atol + _stall_rtol * abs(prev_val):
                    stall += check_every
                else:
                    stall = 0
                prev_val = valf

            # Print optimization progress if verbose
            if verbose and step % every == 0:
                self._log(f"JAX-GRAD step {step}/{max_iter}: val={valf:.8f}, best={best_val:.8f}", verbose=verbose, lvl=2)

            if stall >= patience:
                break

        best_c = np.asarray(_params_to_c(best_params))
        return self.normalize(best_c), float(best_val), nfev

    # --------------------------------
    
    def minimize_entropy(
        self,
        max_iter            : int = 200,
        tol                 : float = 1e-10,
        n_restarts          : int = 5,
        # constraints are passed to the optimization method to enforce orthogonality in coefficient space for finding multiple MES.
        c_constraints       : Optional[List[np.ndarray]] = None,
        rng                 : Optional[np.random.Generator] = None,
        seed                : Optional[int] = None,
        method              : str = 'jax-grad',
        # options can include method-specific parameters such as learning rate, Adam parameters, etc.
        options             : Optional[Dict[str, Any]] = None,
        verbose             : bool = False,
        **kwargs
    ):
        r'''
        Minimize S(V @ c) over normalized complex coefficients using JAX autodiff.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of optimization steps for each restart.
        tol : float
            Tolerance for convergence (based on change in objective value).
        n_restarts : int
            Number of random restarts to perform (to find global minimum).
        c_constraints : Optional[List[np.ndarray]]
            List of coefficient vectors to be orthogonalized against (for finding multiple MES).
        rng : Optional[np.random.Generator]
            Optional random number generator for reproducibility. If None, uses default RNG.
        seed : Optional[int]
            Optional seed for random number generator (used if rng is None).
        method : str
            Optimization method to use. Currently only 'jax-grad' is supported.
        options : Optional[Dict[str, Any]]
            Additional options for the optimization method (e.g. learning rate, Adam parameters).
            Expected keys for method='jax-grad': 'lr', 'beta1', 'beta2', 'adam_eps', 'patience'.
        verbose : bool
            If True, print optimization progress and results.
        kwargs : dict
            Additional keyword arguments for future extensions or method-specific parameters.
            Includes:
            - every : int     Frequency of progress updates when verbose=True (default: every 10 steps).
        '''
        options             = {} if options is None else dict(options)
        c_constraints       = [] if c_constraints is None else c_constraints
        rng                 = np.random.default_rng(seed) if rng is None else rng
        method_lc           = method.lower().replace("_", "-") if isinstance(method, str) else ""
        valid_methods       = {"jax-grad", "jaxgrad", "jax-ad", "jax-autodiff"}
        if method_lc not in valid_methods:
            raise ValueError("Only method='jax-grad' is supported in MESFinder.")
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for MES optimization (method='jax-grad').")

        best_c      = None
        best_val    = np.inf
        nfev_total  = 0
        for i in range(n_restarts):
            c0          = self.random_c(self.m, rng=rng)
            if c_constraints:
                c0      = project_onto_complement(c0, c_constraints)
            c0          = _gauge_fix_first_component(c0)

            c_opt, _, nfev = self._minimize_jax_grad(
                                c0              =   c0,
                                max_iter        =   max_iter,
                                tol             =   tol,
                                lr              =   options.get("lr", 0.05),
                                beta1           =   options.get("beta1", 0.9),
                                beta2           =   options.get("beta2", 0.999),
                                adam_eps        =   options.get("adam_eps", 1e-8),
                                patience        =   options.get("patience", 30),
                                c_constraints   =   c_constraints,
                                verbose         =   verbose,
                                **kwargs
                            )

            c_opt = self.normalize(c_opt)
            # Enforce constraints and gauge-fixing on the final optimized vector
            # to ensure it is in the correct subspace and has a consistent phase convention.
            if c_constraints:
                c_opt = project_onto_complement(c_opt, c_constraints)

            # Gauge-fix the first component to be real and non-negative.
            c_opt       = _gauge_fix_first_component(c_opt)
            val         = float(self(c_opt))
            nfev_total += nfev
            self._log(f"Restart {i+1}/{n_restarts}: val={val:.8f}, nfev={nfev}", verbose=verbose)

            if val < best_val:
                best_val = val
                best_c   = c_opt

        info = {
            "method"        : "jax_grad_adam",
            "nfev_total"    : nfev_total,
            "best_val"      : best_val,
            "best_c_norm"   : np.linalg.norm(best_c) if best_c is not None else None,
            "n_restarts"    : n_restarts
        }
        if verbose:
            self._log(f"Optimization completed: best_val={best_val:.8f}, nfev_total={nfev_total}, method={info['method']}", verbose=verbose, lvl=1, color="green")
        
        # Final normalization, projection onto constraints, 
        # and gauge-fixing to ensure the returned coefficient vector is in the correct subspace and has a consistent phase convention. 
        best_c = self.normalize(best_c)
        if c_constraints:
            best_c = project_onto_complement(best_c, c_constraints)
        best_c = _gauge_fix_first_component(best_c)
        return best_c, float(self(best_c)), info

    def find(
        self,
        n_trials            : int = 20, 
        overlap_tol         : float = 1e-4, 
        state_max           : int = 10,
        # optimization options
        rng                 : Optional[np.random.Generator] = None,
        seed                : Optional[int] = None,
        max_iter            : int = 200,
        tol                 : float = 1e-10,
        n_restarts          : int = 3,
        method              : str = 'jax-grad',
        options             : Optional[Dict[str, Any]] = None,
        verbose             : bool = False,
        **kwargs
    ) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[Dict]]:
        r''' 
        Find multiple MES by performing multiple runs of entropy minimization with random initializations.
        
        Parameters
        ----------
        n_trials : int
            Number of random initializations to perform for finding MES.
        overlap_tol : float
            Tolerance for considering two MES as the same (based on coefficient overlap).
        state_max : int
            Maximum number of MES states to find. The algorithm will stop once this many unique MES are found.
        Optimization options (passed to minimize_entropy):
        rng : Optional[np.random.Generator]
            Random number generator for reproducibility. If None, a new generator will be created.
        seed : Optional[int]
            Seed for random number generator if rng is None.
        max_iter : int
            Maximum iterations for the optimization algorithm.
        tol : float
            Tolerance for convergence in the optimization algorithm.
        n_restarts : int
            Number of restarts for the optimization algorithm to find a good minimum.
        method : str
            Optimization method to use. Only `method='jax-grad'` is supported.
        options : Optional[Dict[str, Any]]
            Additional options for JAX-Adam (`lr`, `beta1`, `beta2`, `adam_eps`, `patience`).
        verbose : bool
            If True, print detailed information about the optimization process. 
        kwargs : dict
            Additional keyword arguments for future extensions or method-specific options.
            Includes:
            - every: int (for verbose logging frequency)
        '''    
        
        rng             = np.random.default_rng(seed) if rng is None else rng
        minima          = []
        values          = []
        coeffs          = []
        c_constraints   = []
        diagnostics     = []

        # To find multiple MES, we perform multiple optimization runs with random initializations.
        for trial in range(n_trials):
            self._log(f"Trial {trial+1}/{n_trials} - Starting MES optimization...", verbose=verbose, lvl=1)

            c_opt, val, info = self.minimize_entropy(
                max_iter=max_iter, tol=tol, n_restarts=n_restarts, c_constraints=c_constraints, rng=rng, seed=None,
                method=method, options=options, verbose=verbose, **kwargs
                )
                
            # Check for uniqueness against found coefficients in the subspace
            # This is more efficient than checking full state vectors
            is_unique = True
            
            for c_prev in coeffs:
                
                # In the subspace, we check overlap |<c_prev|c_opt>|
                if abs(np.vdot(c_prev, c_opt)) > 1.0 - overlap_tol:
                    if verbose:
                        self._log("  -> Found a state that is not unique (overlap with previous state exceeds tolerance). Skipping.", verbose=verbose, lvl=2, color="yellow")
                    is_unique = False
                    break
            
            if is_unique:
                psi = self.V @ c_opt
                
                # Store the found MES, its coefficients, the value of the entropy, and diagnostic information about the optimization.
                minima.append(psi)
                coeffs.append(c_opt)
                c_constraints.append(self.normalize(c_opt))
                values.append(float(val))
                diagnostics.append(info)
                
                self._log(f"Trial {trial+1}: Found unique MES with S={val:.6f}", verbose=verbose, lvl=2, color="green")
            
            if len(minima) >= state_max:
                break

        return minima, values, coeffs, diagnostics

# ---------------------------------------
#! Convenience function to find MES
# ---------------------------------------

def find_mes(V: np.ndarray, S_func: Callable[[np.ndarray], float], *, S_func_c: Optional[Callable] = None, logger: Optional['Logger'] = None, **kwargs):
    """
    Convenience function to find MES using MESFinder.
    
    Parameters
    ----------
    V : np.ndarray
        A matrix of shape (Ns, m) where m is the dimension of the degenerate manifold. Each column is a basis state of the manifold.
    S_func : Callable[[np.ndarray], float]
        A function that takes a state vector of size Ns and returns its entanglement entropy.
    S_func_c : Optional[Callable[[np.ndarray], float]]
        Fast-path entropy function that takes a coefficient vector c of size m directly,
        bypassing the expensive V @ c matmul and Ns-length gather in the hot loop.
        Build with :func:`build_s_func_c` or from :func:`find_mes_save`.
    logger : Optional['Logger']
        An optional logger object to log messages during the optimization process.
    **kwargs
        Additional keyword arguments to pass to the MESFinder's find method (e.g., n_trials 
        - n_trials
            Number of random initializations to perform for finding MES.
        - overlap_tol    
            Tolerance for considering two MES as the same (based on coefficient overlap).
        - state_max
            Maximum number of MES states to find.
        - Optimization options (e.g., max_iter, tol, n_restarts, method, options, verbose).
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("find_mes requires JAX for MES optimization (method='jax-grad').")
    
    finder = MESFinder(V, S_func, S_func_c=S_func_c, logger=logger)
    return finder.find(**kwargs)

def find_mes_save(lattice       : 'Lattice', 
                  V             : np.ndarray, 
                  save_path     : str,
                  *,
                  states        : Optional[int] = None,
                  mes_cuts      : Optional[List[str]] = None,
                  logger        : Optional['Logger'] = None,
                  use_schmidt   : bool = True,
                  **kwargs):
    r'''
    High-level function to find MES for specified cuts and save the results.
    It takes the lattice, the ground state manifold V, and a save path for results.
    
    Parameters
    ----------    
    lattice : 'Lattice'
        The lattice object containing information about the system and methods to define cuts.
    V : np.ndarray
        A matrix of shape (Nh, m) where m is the dimension of the degenerate manifold. Each column is a basis state of the manifold.
    save_path : str
        The directory path where MES results will be saved. The function will attempt to 
        save a file named "mes_results_all_cuts.npz" containing the MES states for all cuts.
    states : Optional[int]
        If provided, the function will truncate the input V to only consider the first `states` 
        columns (ground states) for the MES analysis. If None, all states in V
        will be used.
    mes_cuts : Optional[List[str]]
        A list of cut types to analyze for MES. Each cut type should correspond to a 
        method in the lattice object that defines the region for entanglement entropy calculation. If None, defaults to ["half_x", "half_y"] which are common cuts for 2D systems. The function

    Returns
    -------
    mes_results : Dict[str, Dict]
        A dictionary where each key is a cut type and the value is another dictionary containing:
        - 'states': List of MES states found for that cut.
        - 'values': List of entanglement entropy values for the MES states.
        - 'coeffs': List of coefficient vectors corresponding to the MES states in the ground state manifold.
        - 'diagnostics': List of diagnostic information from the optimization process for each MES found.
        - 'region': The list of sites defining the region for the cut used in the entropy calculation.
    best_mes : Tuple[str, np.ndarray]
        A tuple containing the cut type and the MES state with the lowest entanglement entropy across all cuts analyzed. If no MES are found, returns (None, None).
    '''

    if not JAX_AVAILABLE:
        raise RuntimeError("find_mes_save requires JAX for MES optimization (method='jax-grad').")

    try:
        from QES.general_python.lattices.lattice    import Lattice
        from QES.general_python.physics             import density_matrix_jax, entropy_jax
    except ImportError as exc:
        raise ImportError("find_mes_save requires QES.general_python.lattices.lattice and QES.general_python.physics modules.") from exc

    # Transform the states
    try:
        if states is not None:
            V           = V[:, :states]
            if logger:  logger.info(f"Truncated to {states} states for MES analysis.", lvl=2, color='blue')
                
        V_gs_jax        = jnp.asarray(V)
        n_gs            = V_gs_jax.shape[1]
        if logger:      logger.info(f"Ground state manifold: {n_gs} states loaded for MES analysis.", lvl=1, color='blue')
    except ImportError as e:
        if logger:      logger.error(f"JAX is not available: {e}")

    # -------------------------------

    def mes_entropy_function(region_sites, ns, q=1.0):
        '''Returns a JAX-traceable entropy callback for MES search (psi-level, fallback).'''
        if len(region_sites) == 0 or len(region_sites) == ns:
            return lambda psi: jnp.asarray(0.0)
        
        @jax.jit
        def _s_func(psi):
            probs = density_matrix_jax.schmidt(psi, va=region_sites, ns=ns, eig=False, contiguous=False, square=True, return_vecs=False,)
            return entropy_jax.renyi_entropy_jax(probs, q=q)
        return _s_func

    def build_s_func_c(region_sites, ns, V_basis, q=1.0):
        """
        Fast-path: precompute Schmidt matrices for each basis vector once, then
        evaluate entropy purely in the m-dimensional coefficient space.

        Timeline per optimization step (before vs after):
          Before: V@c  [Nsxm matvec]  +  gather [Ns random reads] + SVD [dAxdB]
          After:  einsum [m additions of dAxdB matrices] +  SVD [dAxdB]

        For Ns=2^18, m=3, dA=dB=512 this is ~100x fewer memory ops per step.
        """
        if len(region_sites) == 0 or len(region_sites) == ns:
            return lambda c: jnp.asarray(0.0)

        try:
            from QES.general_python.physics.density_matrix      import mask_subsystem
            from QES.general_python.physics.density_matrix_jax  import psi_jax
        except ImportError:
            return None     # fall back to psi-level S_func

        # Precompute (m, dA, dB) — done once, lives in device memory.
        (size_a, _), order  = mask_subsystem(list(region_sites), ns, 2, False)
        m_basis             = V_basis.shape[1]
        V_mats              = jnp.stack([psi_jax(V_basis[:, i], size_a, ns, 2, order) for i in range(m_basis)])

        @jax.jit
        def _s_func_c(c: "jnp.ndarray") -> "jnp.ndarray":
            # Linear combination of precomputed (dA, dB) matrices — sequential memory access.
            Psi_mat = jnp.einsum('i,iab->ab', c, V_mats)
            _, s, _ = jnp.linalg.svd(Psi_mat, full_matrices=False)
            return entropy_jax.renyi_entropy_jax(s ** 2, q=q)
        return _s_func_c
    
    # -------------------------------
    mes_cuts        = mes_cuts if mes_cuts is not None else ["half_x", "half_y"]
    mes_results     = {}

    # For each specified cut, find the MES and their entropies, and store the results in a dictionary.
    for cut_kind in mes_cuts:
        if logger:  logger.title(f"Finding MES for cut: {cut_kind}", lvl=0, color='green')
        region_cut  = lattice.get_region(kind=cut_kind).A
        S_func      = mes_entropy_function(region_cut, lattice.ns, q=1.0)
        # Build fast-path coefficient-space entropy (precomputed Schmidt matrices).
        S_func_c    = build_s_func_c(region_cut, lattice.ns, V_gs_jax, q=1.0) if use_schmidt else None
        if S_func_c is not None:
            if logger: logger.info(f"  Using precomputed Schmidt basis (fast path) for cut '{cut_kind}'.", lvl=2, color='cyan')
        mes_states, mes_values, mes_coeffs, mes_diag = find_mes(
                        V_gs_jax,
                        S_func,
                        S_func_c        =   S_func_c,
                        n_trials        =   kwargs.get('n_trials', 20),
                        n_restarts      =   kwargs.get('n_restarts', 3),
                        max_iter        =   kwargs.get('max_iter', 100),
                        state_max       =   n_gs,
                        overlap_tol     =   kwargs.get('overlap_tol', 1e-5),
                        options         =   kwargs.get('options', {'lr': 0.05, 'patience': 25}),
                        verbose         =   kwargs.get('verbose', True),
                        logger          =   logger,
                        every           =   kwargs.get('every', 10),
                    )
        mes_results[cut_kind] = {
            'states'        : mes_states,
            'values'        : mes_values,
            'coeffs'        : mes_coeffs,
            'diagnostics'   : mes_diag,
            'region'        : region_cut,
        }
        if mes_values:
            if logger:      logger.info(f"MES ({cut_kind}) entropies: {mes_values}", lvl=1, color='blue')
        else:
            if logger:      logger.warning(f"No MES found for cut={cut_kind}", color='yellow')
            
    # Identify the best MES state across all cuts based on the lowest entropy value, and log the result.
    best_mes_state  = None
    best_mes_cut    = None
    available       = [k for k in mes_results if len(mes_results[k]['values']) > 0]
    if available:
        best_mes_cut    = min(available, key=lambda k: mes_results[k]['values'][0])
        best_mes_state  = mes_results[best_mes_cut]['states'][0]
        if logger:      logger.info(f"Best MES cut: {best_mes_cut}, S={mes_results[best_mes_cut]['values'][0]:.6f}", lvl=1, color='magenta')
    # Save the all MES states and values for potential future analysis
    try:
        mes_save_path   = Path(save_path) / "mes_results_all_cuts.npz"
        np.savez(mes_save_path, **{k: np.array(v['states']) for k, v in mes_results.items()})
    except Exception as e:
        if logger:      logger.warning(f"Failed to save MES results: {e}", color='yellow')

    return mes_results, (best_mes_cut, best_mes_state)

def load_mes_save(save_path     : str, *, 
                  mes_cuts      : Optional[List[str]] = None,
                  filename      : Optional[str] = None, 
                  logger        : Optional[Logger] = None,
                  make_modular  : bool = False
                  ) -> Dict[str, Dict]:
    '''Load MES results from a saved .npz file.'''
    try:
        if filename is None:
            filename = "mes_results_all_cuts.npz"
            
        # Load the MES results...
        save_path   = Path(os.path.abspath(save_path)) / filename
        if logger:  logger.info(f"Attempting to load MES results from: {save_path}", lvl=1)
        
        if not save_path.exists():
            raise IOError(f"File does not exist: {save_path}")
        
        mes_cuts    = mes_cuts if mes_cuts is not None else ["half_x", "half_y"]
        mes_results = {}
        data        = np.load(save_path, allow_pickle=True)
        for k in mes_cuts:
            if k not in data.files:
                if logger:  logger.warning(f"Cut '{k}' not found in saved MES results.", color='yellow', lvl=2)
            else:
                if logger:  logger.info(f"Loaded MES states for cut '{k}' from saved results.", color='green', lvl=2)
                mes_results[k] = {'states': data[k].tolist()}
                
        if make_modular:
            if logger:  
                logger.info("Computing modular S-matrix and topological statistics from loaded MES results...", lvl=2)
            
            if "half_x" in mes_results and "half_y" in mes_results:
                mes_results["modular"] = compute_modular_s_matrix(mes_results["half_x"]["states"], mes_results["half_y"]["states"])
                if logger:  logger.info("Modular S-matrix and topological statistics computed successfully.", color='green', lvl=3)
            else:
                mes_results["modular"] = None
                if logger:  logger.warning("Cannot compute modular S-matrix: missing MES results for 'half_x' or 'half_y' cuts.", color='yellow', lvl=3)
                
        return mes_results
    
    except Exception as e:
        raise IOError(f"Failed to load MES results from {save_path}: {e}")

# ---------------------------------------
#! Modular S-matrix and Topological Statistics
# ---------------------------------------

def compute_modular_s_matrix(
    mes_x   : List[np.ndarray], 
    mes_y   : List[np.ndarray], 
    tol     : float = 1e-8
) -> TopologicalResults:
    r"""
    Compute the modular S-matrix from two bases of Minimum Entangled States (MES)
    obtained from perpendicular cuts (e.g., along x and y).
    
    The modular S-matrix S_ij encodes the topological mutual statistics between anyons.
    It is proportional to the overlap matrix:
    S_ij = <Xi_i(x) | Xi_j(y)> / D
    
    From this matrix, we can extract the quantum dimensions d_i and the total quantum dimension D.
    The first row (or column) of S corresponds to the overlaps with the vacuum anyon
    meaning that S_0a = d_a / D, where d_a is the quantum dimension of anyon type a and D is the total quantum dimension.
    
    Otherwise, the total quantum dimension can be obtained from the largest value in the first row of the unnormalized overlap matrix, 
    which corresponds to the vacuum anyon.
    
    Parameters
    ----------
    mes_x : List[np.ndarray]
        List of MES obtained from a cut along the x-direction.
    mes_y : List[np.ndarray]
        List of MES obtained from a cut along the y-direction.
    tol : float
        Numerical tolerance for Abelian/Non-Abelian classification.
        
    Returns
    -------
    TopologicalResults
        Dataclass containing the modular S-matrix, quantum dimensions, and total quantum dimension.
    """
    
    # Compute the overlap matrix O_ij = <Xi_i(x) | Xi_j(y)> between the two MES bases.
    n_x     = len(mes_x)
    n_y     = len(mes_y)
    
    if n_x != n_y:
        raise ValueError(f"Number of MES in x and y bases must be equal (got {n_x} and {n_y}).")
    
    m       = n_x
    overlap = np.zeros((m, m), dtype=np.complex128)
    
    for i in range(m):
        for j in range(m):
            overlap[i, j] = np.vdot(mes_x[i], mes_y[j]) # Compute the inner product <Xi_i(x) | Xi_j(y)> to fill the overlap matrix.
            
    # The modular S-matrix is unitary. The first row (or column) contains d_i / D.
    # Since d_i >= 1, the largest values in the first row correspond to d_i.
    
    row1        = np.abs(overlap[0, :])
    d_tilde     = row1
    d_1_tilde   = d_tilde[0]
    
    if d_1_tilde < 1e-10:
        d_1_tilde = np.max(d_tilde)
    
    quantum_dims    = d_tilde / d_1_tilde
    total_D         = 1.0 / d_1_tilde                           # The total quantum dimension D can be obtained from the largest value in the first row of the unnormalized overlap matrix, which corresponds to the vacuum anyon. Since S_0a = d_a / D, we have D = 1 / S_00 = 1 / d_1_tilde.
    
    # Normalized S-matrix
    s_matrix        = overlap / total_D
    is_abelian      = np.all(np.abs(quantum_dims - 1.0) < tol)  # If all quantum dimensions are approximately 1, the system is Abelian.
    is_non_abelian  = np.any(quantum_dims > 1.0 + tol)          # If any quantum dimension is greater than 1, the system is Non-Abelian.
    
    return TopologicalResults(
        S_matrix                = s_matrix,                     # The normalized modular S-matrix, which encodes the mutual statistics of anyons.
        overlap_matrix          = overlap,
        quantum_dimensions      = quantum_dims,
        total_quantum_dimension = total_D,
        is_abelian              = is_abelian,
        is_non_abelian          = is_non_abelian
    )

# ---------------------------------------
#! EOF
# ---------------------------------------
