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

import  numpy as np
from    typing import Callable, List, Tuple, Optional, Dict, Any
from    dataclasses import dataclass
import  os
import  pickle
from    datetime import datetime
import  traceback

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax             = None
    jnp             = None
    JAX_AVAILABLE   = False

# ---------------------------------------
#! Topological Results Dataclass
# ---------------------------------------

@dataclass
class TopologicalResults:
    """
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
    '''Gauge-fix global phase so c[0] is real and non-negative.'''
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
    
    def __init__(self, V: np.ndarray, S_func: Callable[[np.ndarray], float]):
        """
        Initialize the MESFinder.
        
        Parameters
        ----------
        V : np.ndarray
            A matrix of shape (Ns, m) where m is the dimension of the degenerate manifold.
            Each column is a basis state of the manifold.
        S_func : Callable[[np.ndarray], float]
            A function that takes a state vector of size Ns and returns its entanglement entropy.
        """
        self.V          = V
        self.S_func     = S_func
        self.m          = V.shape[1]
        self.nh         = V.shape[0]
        # Pre-allocate buffer for state vector to avoid multiple large allocations.
        # Ensure it is complex as coefficients c are complex.
        # Use promoted dtype to avoid np.dot(..., out=...) dtype mismatch when V is complex64.
        self._psi_buf   = np.zeros(self.nh, dtype=np.result_type(V.dtype, np.complex128))

    @staticmethod
    def normalize(c):
        ''' Normalize a complex vector. '''
        nrm = np.linalg.norm(c)
        if nrm < 1e-15: return c
        return c / nrm

    @staticmethod
    def random_c(m, rng=None):
        ''' Generate random complex vector and normalize it. '''
        if rng is None:
            c = np.random.randn(m) + 1j*np.random.randn(m)
        else:
            c = rng.standard_normal(m) + 1j*rng.standard_normal(m)
        return MESFinder.normalize(c)

    @staticmethod
    def _default_checkpoint_path(prefix: str) -> str:
        '''Generate default checkpoint path for interruption/error recovery.'''
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{stamp}.pkl"

    def _save_checkpoint(
        self,
        payload: Dict[str, Any],
        checkpoint_path: Optional[str],
        prefix: str,
        verbose: bool = False
    ) -> Optional[str]:
        '''
        Save optimization state to disk for later recovery.
        The file is a pickle with a dictionary payload.
        '''
        path                        = checkpoint_path if checkpoint_path else self._default_checkpoint_path(prefix)
        path                        = os.path.abspath(path)
        payload["timestamp_utc"]    = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        payload["m"]                = self.m
        payload["nh"]               = self.nh
        try:
            with open(path, "wb") as fout:
                pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"Saved MES checkpoint to: {path}")
            return path
        except Exception as save_exc:
            if verbose:
                print(f"Failed to save MES checkpoint ({type(save_exc).__name__}: {save_exc})")
            return None

    # --------------------------------
    #! Find
    # --------------------------------
    
    def __call__(self, c):
        ''' Compute the entropy S(V @ c) for a given vector c. '''
        
        # Optimized matrix-vector multiplication into buffer
        # psi = self.V @ c
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
    ) -> Tuple[np.ndarray, float, int]:
        """
        Direct autodiff optimization in coefficient space with JAX value-and-grad.
        Requires `S_func` to be JAX-traceable.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available.")
        c_constraints = [] if c_constraints is None else c_constraints

        m       = self.m
        c0      = self.normalize(c0)
        if c_constraints:
            c0  = project_onto_complement(c0, c_constraints)
        c0      = _gauge_fix_first_component(c0)
        params  = jnp.concatenate([
                    jnp.asarray(c0.real),
                    jnp.asarray(c0.imag)
                ])
        Vj      = jnp.asarray(self.V)
        c_constraints_jax = [jnp.asarray(c_prev) for c_prev in c_constraints]

        def _params_to_c(theta: "jnp.ndarray") -> "jnp.ndarray":
            c   = theta[:m] + 1j * theta[m:]
            nrm = jnp.linalg.norm(c)
            c   = c / jnp.maximum(nrm, 1e-15)
            if c_constraints_jax:
                for c_prev in c_constraints_jax:
                    c = c - jnp.vdot(c_prev, c) * c_prev
                nrm2 = jnp.linalg.norm(c)
                c = c / jnp.maximum(nrm2, 1e-15)

            abs0 = jnp.abs(c[0])
            phase = jnp.where(abs0 > 1e-15, jnp.exp(-1j * jnp.angle(c[0])), 1.0 + 0.0j)
            c = c * phase
            c = jnp.where(jnp.real(c[0]) < 0.0, -c, c)
            c = c.at[0].set(jnp.real(c[0]) + 0.0j)
            return c

        def _loss(theta: "jnp.ndarray") -> "jnp.ndarray":
            c   = _params_to_c(theta)
            psi = Vj @ c
            val = self.S_func(psi)
            return jnp.real(jnp.asarray(val))

        try:
            value_and_grad  = jax.value_and_grad(_loss)
            val0, grad0     = value_and_grad(params)
            _               = float(val0)
            _               = np.asarray(grad0)
        except Exception as exc:
            raise RuntimeError("method='jax-grad' requires S_func to be JAX-traceable; build S_func with JAX-based density/entropy kernels.") from exc

        mom1        = jnp.zeros_like(params)
        mom2        = jnp.zeros_like(params)
        best_params = params
        best_val    = float(val0)
        prev_val    = best_val
        stall       = 0
        nfev        = 1

        for step in range(1, max_iter + 1):
            val, grad   = value_and_grad(params)
            nfev        += 1
            mom1        = beta1 * mom1 + (1.0 - beta1) * grad
            mom2        = beta2 * mom2 + (1.0 - beta2) * (grad * grad)
            mom1_hat    = mom1 / (1.0 - beta1 ** step)
            mom2_hat    = mom2 / (1.0 - beta2 ** step)
            params      = params - lr * mom1_hat / (jnp.sqrt(mom2_hat) + adam_eps)

            valf = float(val)
            if valf < best_val:
                best_val    = valf
                best_params = params

            if abs(valf - prev_val) < tol:
                stall += 1
            else:
                stall = 0
            prev_val = valf

            if verbose and (step == 1 or step % 25 == 0):
                print(f"\tJAX-GRAD step {step}/{max_iter}: val={valf:.8f}, best={best_val:.8f}")

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
        c_constraints       : Optional[List[np.ndarray]] = None,
        rng                 : Optional[np.random.Generator] = None,
        seed                : Optional[int] = None,
        method              : str = 'jax-grad',
        options             : Optional[Dict[str, Any]] = None,
        verbose             : bool = False,
        save_on_exception   : bool = True,
        checkpoint_path     : Optional[str] = None,
    ):
        r'''
        Minimize S(V @ c) over normalized complex coefficients using JAX autodiff.
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
        restart_idx = -1
        c0          = None
        try:
            for i in range(n_restarts):
                restart_idx = i
                c0          = self.random_c(self.m, rng=rng)
                if c_constraints:
                    c0      = project_onto_complement(c0, c_constraints)
                c0          = _gauge_fix_first_component(c0)

                c_opt, _, nfev = self._minimize_jax_grad(
                    c0=c0,
                    max_iter=max_iter,
                    tol=tol,
                    lr=options.get("lr", 0.05),
                    beta1=options.get("beta1", 0.9),
                    beta2=options.get("beta2", 0.999),
                    adam_eps=options.get("adam_eps", 1e-8),
                    patience=options.get("patience", 30),
                    c_constraints=c_constraints,
                    verbose=verbose
                )

                c_opt = self.normalize(c_opt)
                if c_constraints:
                    c_opt = project_onto_complement(c_opt, c_constraints)
                c_opt = _gauge_fix_first_component(c_opt)
                val   = float(self(c_opt))
                nfev_total += nfev

                if verbose:
                    print(f"\tRestart {i+1}/{n_restarts}: val={val:.8f}, nfev={nfev}")

                if val < best_val:
                    best_val = val
                    best_c   = c_opt
        except (KeyboardInterrupt, Exception) as exc:
            if save_on_exception:
                payload = {
                    "scope"                 : "minimize_entropy",
                    "exception_type"        : type(exc).__name__,
                    "exception_message"     : str(exc),
                    "traceback"             : traceback.format_exc(),
                    "method_requested"      : method,
                    "method_effective"      : "jax_grad_adam",
                    "max_iter"              : max_iter,
                    "tol"                   : tol,
                    "n_restarts"            : n_restarts,
                    "restart_in_progress"   : restart_idx + 1,
                    "nfev_total"            : nfev_total,
                    "best_val"              : float(best_val) if np.isfinite(best_val) else None,
                    "best_c"                : best_c.copy() if best_c is not None else None,
                    "current_c0"            : c0.copy() if isinstance(c0, np.ndarray) else None,
                    "options"               : dict(options),
                    "n_constraints"         : len(c_constraints),
                }
                ckpt = self._save_checkpoint(
                    payload=payload,
                    checkpoint_path=checkpoint_path,
                    prefix="mes_minimize_checkpoint",
                    verbose=verbose
                )
                if verbose and ckpt is not None:
                    print("Checkpoint saved after optimizer exception/interruption.")
            raise

        info = {
            "method"        : "jax_grad_adam",
            "nfev_total"    : nfev_total,
            "best_val"      : best_val,
            "best_c_norm"   : np.linalg.norm(best_c) if best_c is not None else None,
            "n_restarts"    : n_restarts
        }
        if verbose:
            print(f"Best value found: {best_val:.8f} with method={info['method']}")

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
        # Additional checkpoint paths...
        save_on_exception   : bool = True,
        checkpoint_path     : Optional[str] = None,
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
        '''    
        
        rng         = np.random.default_rng(seed) if rng is None else rng    
        minima      = []
        values      = []
        coeffs      = []
        c_constraints = []
        diagnostics = []

        trial_idx = -1
        try:
            # To find multiple MES, we perform multiple optimization runs with random initializations.
            for trial in range(n_trials):
                trial_idx = trial
                if verbose:
                    print(f"Trial {trial+1}/{n_trials}...")
                    
                c_opt, val, info = self.minimize_entropy(
                    max_iter=max_iter, tol=tol, n_restarts=n_restarts, c_constraints=c_constraints, rng=rng, seed=None,
                    method=method, options=options, verbose=verbose,
                    save_on_exception=False, checkpoint_path=None
                )
                
                # Check for uniqueness against found coefficients in the subspace
                # This is more efficient than checking full state vectors
                is_unique = True
                for c_prev in coeffs:
                    
                    # In the subspace, we check overlap |<c_prev|c_opt>|
                    if abs(np.vdot(c_prev, c_opt)) > 1.0 - overlap_tol:
                        is_unique = False
                        break
                
                if is_unique:
                    psi = self.V @ c_opt
                    minima.append(psi)
                    coeffs.append(c_opt)
                    c_constraints.append(self.normalize(c_opt))
                    values.append(float(val))
                    diagnostics.append(info)
                    if verbose:
                        print(f"  -> Found unique MES with S={val:.6f}")
                
                if len(minima) >= state_max:
                    break
        except (KeyboardInterrupt, Exception) as exc:
            if save_on_exception:
                payload = {
                    "scope": "find",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "trial_in_progress": trial_idx + 1,
                    "n_trials": n_trials,
                    "state_max": state_max,
                    "overlap_tol": overlap_tol,
                    "method": method,
                    "max_iter": max_iter,
                    "tol": tol,
                    "n_restarts": n_restarts,
                    "options": dict(options) if options is not None else None,
                    "values": list(values),
                    "coeffs": [c.copy() for c in coeffs],
                    "c_constraints": [c.copy() for c in c_constraints],
                    "minima": [psi.copy() for psi in minima],
                    "diagnostics": list(diagnostics),
                }
                ckpt = self._save_checkpoint(
                    payload=payload,
                    checkpoint_path=checkpoint_path,
                    prefix="mes_find_checkpoint",
                    verbose=verbose
                )
                if verbose and ckpt is not None:
                    print("Checkpoint saved with partial MES results.")
            raise

        return minima, values, coeffs, diagnostics

def find_mes(V: np.ndarray, S_func: Callable[[np.ndarray], float], **kwargs):
    """
    Convenience function to find MES using MESFinder.
    
    Parameters
    ----------
    V : np.ndarray
        A matrix of shape (Ns, m) where m is the dimension of the degenerate manifold. Each column is a basis state of the manifold.
    S_func : Callable[[np.ndarray], float]
        A function that takes a state vector of size Ns and returns its entanglement entropy.
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
    finder = MESFinder(V, S_func)
    return finder.find(**kwargs)

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
    n_x     = len(mes_x)
    n_y     = len(mes_y)
    
    if n_x != n_y:
        raise ValueError(f"Number of MES in x and y bases must be equal (got {n_x} and {n_y}).")
    
    m       = n_x
    overlap = np.zeros((m, m), dtype=np.complex128)
    
    for i in range(m):
        for j in range(m):
            overlap[i, j] = np.vdot(mes_x[i], mes_y[j])
            
    # The modular S-matrix is unitary. The first row (or column) contains d_i / D.
    # Since d_i >= 1, the largest values in the first row correspond to d_i.
    
    row1        = np.abs(overlap[0, :])
    d_tilde     = row1
    d_1_tilde   = d_tilde[0]
    
    if d_1_tilde < 1e-10:
        d_1_tilde = np.max(d_tilde)
    
    quantum_dims    = d_tilde / d_1_tilde
    total_D         = 1.0 / d_1_tilde
    
    # Normalized S-matrix
    s_matrix        = overlap / total_D
    
    is_abelian      = np.all(np.abs(quantum_dims - 1.0) < tol)
    is_non_abelian  = np.any(quantum_dims > 1.0 + tol)
    
    return TopologicalResults(
        S_matrix                = s_matrix,
        overlap_matrix          = overlap,
        quantum_dimensions      = quantum_dims,
        total_quantum_dimension = total_D,
        is_abelian              = is_abelian,
        is_non_abelian          = is_non_abelian
    )

# ---------------------------------------
#! EOF
# ---------------------------------------
