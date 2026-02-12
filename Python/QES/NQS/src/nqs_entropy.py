r"""
NQS Entanglement Entropy via Replica Methods
=============================================

This module implements entanglement entropy estimation for Neural Quantum States
using replica (swap) tricks. It supports:

1. **Standard bipartition Rényi-2 entropy** via the two-replica swap estimator:
   $S_2(A) = -\ln \mathrm{Tr}[\rho_A^2]$
   
2. **Rényi-q entropy** for integer $q \ge 2$ using q-replica generalizations:
   $S_q(A) = \frac{1}{1-q} \ln \mathrm{Tr}[\rho_A^q]$
   
3. **Topological Entanglement Entropy (TEE)** via the Kitaev-Preskill construction:
   $\gamma = S_A + S_B + S_C - S_{AB} - S_{BC} - S_{AC} + S_{ABC}$

Physics Background
------------------
For a pure quantum state $|\Psi\rangle$ on a lattice with Hilbert space 
$\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B$, the reduced density matrix is
$\rho_A = \mathrm{Tr}_B |\Psi\rangle\langle\Psi|$.

The $q$-th Rényi entropy is:
$$
S_q(A) = \frac{1}{1-q} \ln \mathrm{Tr}[\rho_A^q]
$$

**Replica swap trick for $S_2$**:
Given two independent replicas $|\Psi\rangle^{(1)} \otimes |\Psi\rangle^{(2)}$,
define the SWAP operator $\hat{S}_A$ that exchanges subsystem $A$ between replicas:
$$
\hat{S}_A |s_A^{(1)} s_B^{(1)}\rangle \otimes |s_A^{(2)} s_B^{(2)}\rangle
= |s_A^{(2)} s_B^{(1)}\rangle \otimes |s_A^{(1)} s_B^{(2)}\rangle
$$

Then $\mathrm{Tr}[\rho_A^2] = \langle \Psi \otimes \Psi | \hat{S}_A | \Psi \otimes \Psi \rangle$,
which gives the Monte Carlo estimator:
$$
\mathrm{Tr}[\rho_A^2] \approx \frac{1}{N_s} \sum_{i=1}^{N_s}
\frac{\Psi(s_A^{(2,i)}, s_B^{(1,i)}) \cdot \Psi(s_A^{(1,i)}, s_B^{(2,i)})}
     {\Psi(s_A^{(1,i)}, s_B^{(1,i)}) \cdot \Psi(s_A^{(2,i)}, s_B^{(2,i)})}
$$

where configurations $s^{(1)}$ and $s^{(2)}$ are independently sampled from $|\Psi|^2$.

**Generalization to $S_q$** ($q \ge 2$):
Use $q$ replicas and cyclic permutation of subsystem $A$:

$$
\mathrm{Tr}[\rho_A^q] = \left\langle \prod_{r=1}^{q}
\frac{\Psi(s_A^{(r+1 \bmod q)}, s_B^{(r)})}{\Psi(s_A^{(r)}, s_B^{(r)})}
\right\rangle
$$

This corresponds to the standard Rényi entropy because each replica is an independent
copy of the same wavefunction, and the cyclic permutation operator on $q$ replicas
has the property $\langle \hat{C}_A^{(q)} \rangle = \mathrm{Tr}[\rho_A^q]$.

Usage
-----
```python
from QES.NQS.src.nqs_entropy import (
    compute_renyi_entropy,
    compute_topological_entropy,
    bipartition_cuts_honeycomb,
)

# Standard bipartition S_2
s2, s2_err = compute_renyi_entropy(nqs, region=list(range(Ns//2)), q=2, return_error=True)

# Higher Rényi S_3
s3, s3_err = compute_renyi_entropy(nqs, region=region, q=3, return_error=True)

# Topological entropy
tee = compute_topological_entropy(nqs, lattice)
print(f"TEE gamma = {tee['gamma']:.4f}")
```

References
----------
- Hastings, Gonzalez, Tubman, Abrams, PRL 104, 157201 (2010)
- Flammia, Hamma, Hughes, Wen, PRL 103, 261601 (2009)
- Kitaev, Preskill, PRL 96, 110404 (2006)
- Levin, Wen, PRL 96, 110405 (2006)
- Humeniuk, Roscilde, PRB 86, 235116 (2012) — Rényi-q replicas in QMC

-------------------------------------------------------------------------------
File        : NQS/src/nqs_entropy.py
Author      : Maksymilian Kliczkowski
Date        : 2025-02-12
-------------------------------------------------------------------------------
"""

from    typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import  numpy as np

if TYPE_CHECKING:
    from QES.NQS.nqs import NQS

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jsp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ===========================================================================
#! Bipartition helpers for honeycomb lattices
# ===========================================================================

def bipartition_cuts_honeycomb(
    lattice,
    cut_type: str = "half_x",
) -> Dict[str, np.ndarray]:
    """
    Generate physically meaningful bipartition cuts for a honeycomb lattice.
    
    For a honeycomb with Lx x Ly unit cells (Ns = 2*Lx*Ly sites), we define several
    canonical cuts that are useful for entanglement studies:
    
    Parameters
    ----------
    lattice : HoneycombLattice
        The lattice object.
    cut_type : str
        Which cuts to return. Options:
        - "half_x"      : Split at x = Lx/2 (vertical cut)
        - "half_y"      : Split at y = Ly/2 (horizontal cut)
        - "quarter"     : A quarter of the system (corner region)
        - "all"         : Return dict with all standard cuts
        - "sweep"       : Return cuts for entanglement scaling (fractions 1/N...(N-1)/N)
    
    Returns
    -------
    dict
        Keys are cut labels, values are numpy arrays of site indices in region A.
    """
    Ns  = lattice.ns
    Lx  = lattice.Lx
    Ly  = lattice.Ly
    
    cuts = {}
    
    # Site indexing: site i has unit cell n = i//2, sublattice r = i%2
    # Unit cell (X, Y) where X = n % Lx, Y = n // Lx
    
    def sites_where(condition):
        """Return site indices satisfying a condition on (X, Y, sublattice)."""
        result = []
        for i in range(Ns):
            n   = i // 2
            X   = n % Lx
            Y   = n // Lx
            sub = i % 2
            if condition(X, Y, sub):
                result.append(i)
        return np.array(result, dtype=np.int32)
    
    if cut_type in ("half_x", "all"):
        cuts["half_x"] = sites_where(lambda X, Y, s: X < Lx // 2)
    
    if cut_type in ("half_y", "all"):
        cuts["half_y"] = sites_where(lambda X, Y, s: Y < Ly // 2)
    
    if cut_type in ("quarter", "all"):
        cuts["quarter"] = sites_where(lambda X, Y, s: X < Lx // 2 and Y < Ly // 2)
    
    if cut_type in ("sublattice_A", "all"):
        cuts["sublattice_A"] = sites_where(lambda X, Y, s: s == 0)
    
    if cut_type in ("sweep", "all"):
        # Sweep cuts for entanglement scaling: include first n unit cells
        for n_uc in range(1, Lx * Ly):
            label = f"sweep_{n_uc}_of_{Lx*Ly}"
            region = []
            uc_count = 0
            for Y in range(Ly):
                for X in range(Lx):
                    if uc_count >= n_uc:
                        break
                    idx = Y * Lx + X
                    region.extend([2 * idx, 2 * idx + 1])
                    uc_count += 1
                if uc_count >= n_uc:
                    break
            cuts[label] = np.array(region, dtype=np.int32)
    
    if cut_type not in ("half_x", "half_y", "quarter", "sublattice_A", "sweep", "all"):
        raise ValueError(
            f"Unknown cut_type '{cut_type}'. Use 'half_x', 'half_y', 'quarter', "
            f"'sublattice_A', 'sweep', or 'all'."
        )
    
    return cuts


def _default_bipartition(Ns: int) -> "jnp.ndarray":
    """Default half-system bipartition."""
    return jnp.arange(Ns // 2, dtype=jnp.int32)


# ===========================================================================
#! Core Rényi entropy estimators
# ===========================================================================

def compute_renyi_entropy(
    nqs             : "NQS",
    region          : Optional[Any]     = None,
    *,
    q               : int               = 2,
    num_samples     : int               = 4096,
    num_chains      : int               = 1,
    return_error    : bool              = False,
    return_raw      : bool              = False,
    min_trace_value : float             = 1e-15,
) -> Union[float, Tuple[float, float], Dict[str, Any]]:
    r"""
    Compute the q-th Rényi entanglement entropy of subsystem A using the 
    q-replica swap/cyclic-permutation estimator.
    
    For q=2, this reduces to the standard two-replica swap trick:
        $S_2(A) = -\ln \mathrm{Tr}[\rho_A^2]$
    
    For general integer q >= 2:
        $S_q(A) = \frac{1}{1-q} \ln \mathrm{Tr}[\rho_A^q]$
    
    The estimator uses q independent replicas. For each replica r, 
    we sample $s^{(r)} \sim |\Psi(s^{(r)})|^2$, then cyclically permute
    the subsystem A indices among replicas:
    
        $\mathrm{Tr}[\rho_A^q] = \langle \prod_{r=0}^{q-1}
        \frac{\Psi(s_A^{((r+1) \bmod q)}, s_B^{(r)})}
             {\Psi(s_A^{(r)}, s_B^{(r)})} \rangle$
    
    Parameters
    ----------
    nqs : NQS
        Trained NQS wavefunction.  
    region : array-like or None
        Site indices defining subsystem A. If None, uses canonical half-system cut.
    q : int
        Rényi index. Must be >= 2. Default: 2.
    num_samples : int
        Number of Monte Carlo samples per replica.
    num_chains : int
        Number of Markov chains for sampling.
    return_error : bool
        If True, also return the statistical error estimate.
    return_raw : bool
        If True, return full diagnostic dict (traces, per-sample values, errors).
    min_trace_value : float
        Minimum clamp for Tr[rho^q] to avoid log(0).
        
    Returns
    -------
    float or (float, float) or dict
        The Rényi-q entropy S_q, optionally with error bar or full diagnostics.
    """
    if not JAX_AVAILABLE:
        raise NotImplementedError("Rényi entropy estimation requires JAX backend.")
    if q < 2 or not isinstance(q, int):
        raise ValueError(f"Rényi index q must be an integer >= 2, got {q}.")
    
    Ns = nqs.nvisible
    
    if region is None:
        region = _default_bipartition(Ns)
    else:
        region = jnp.asarray(region, dtype=jnp.int32)
    
    if jnp.any(region < 0) or jnp.any(region >= Ns):
        raise ValueError(f"Region contains out-of-bounds indices for n_visible={Ns}.")
    
    # ---- Sample q independent replicas ----
    replicas_s      = []  # configurations
    replicas_log    = []  # log-amplitudes
    
    for _ in range(q):
        (_, _), (s_r, log_r), _ = nqs.sample(
            num_samples=num_samples, num_chains=num_chains
        )
        s_r     = jnp.asarray(s_r)
        log_r   = jnp.asarray(log_r).reshape(-1)
        if s_r.ndim == 1:
            s_r = s_r.reshape(1, -1)
        replicas_s.append(s_r)
        replicas_log.append(log_r)
    
    # Ensure all replicas have identical shapes
    n_samp = replicas_s[0].shape[0]
    for r in range(1, q):
        if replicas_s[r].shape[0] != n_samp:
            min_n   = min(replicas_s[r_].shape[0] for r_ in range(q))
            for r_ in range(q):
                replicas_s[r_]   = replicas_s[r_][:min_n]
                replicas_log[r_] = replicas_log[r_][:min_n]
            n_samp = min_n
            break
    
    # ---- Cyclic permutation of subsystem A ----
    # For replica r, create swapped config: take A from replica (r+1) % q, keep B from replica r
    log_ratio_sum = jnp.zeros(n_samp)
    
    for r in range(q):
        r_next = (r + 1) % q
        # Build swapped configuration
        s_swapped = replicas_s[r].at[:, region].set(replicas_s[r_next][:, region])
        # Evaluate log-psi on swapped config
        log_swapped = jnp.asarray(nqs.ansatz(s_swapped)).reshape(-1)
        # Accumulate log ratio: log[psi(swapped_r)] - log[psi(original_r)]
        log_ratio_sum = log_ratio_sum + (log_swapped - replicas_log[r])
    
    # ---- Estimate Tr[rho_A^q] ----
    # Tr[rho_A^q] = <exp(sum_r log_ratio_r)> where average over samples
    log_trace   = jsp.logsumexp(log_ratio_sum) - jnp.log(float(n_samp))
    trace_val   = jnp.real(jnp.exp(log_trace))
    trace_val   = jnp.maximum(trace_val, min_trace_value)
    
    # ---- Compute S_q ----
    sq_val = (1.0 / (1.0 - q)) * jnp.log(trace_val)
    
    # ---- Error estimation (chain-aware) ----
    swap_samples    = jnp.real(jnp.exp(log_ratio_sum))
    n_total         = int(n_samp)
    n_chain         = max(int(num_chains), 1)
    
    if n_chain > 1 and (n_total % n_chain) == 0:
        per_chain   = swap_samples.reshape(n_chain, n_total // n_chain).mean(axis=1)
        trace_err   = jnp.std(per_chain, ddof=1) / jnp.sqrt(float(n_chain))
    else:
        trace_err   = jnp.std(swap_samples, ddof=1) / jnp.sqrt(float(max(n_total, 1)))
    
    # Error propagation: dS_q = |d(Tr)/((1-q)*Tr)|
    sq_err = float(jnp.abs(trace_err / ((1.0 - q) * trace_val)))
    sq_val = float(jnp.real(sq_val))
    
    if return_raw:
        return {
            "sq"            : sq_val,
            "sq_err"        : sq_err,
            "q"             : q,
            "trace_rho_q"   : float(trace_val),
            "trace_err"     : float(trace_err),
            "n_samples"     : n_total,
            "n_chains"      : n_chain,
            "region_size"   : len(region),
            "system_size"   : Ns,
        }
    if return_error:
        return sq_val, sq_err
    return sq_val


# ===========================================================================
#! Topological Entanglement Entropy
# ===========================================================================

def compute_topological_entropy(
    nqs     : "NQS",
    lattice : Any,
    *,
    q           : int   = 2,
    radius      : Optional[float] = None,
    **renyi_kwargs,
) -> Dict[str, Any]:
    r"""
    Compute the Topological Entanglement Entropy (TEE) $\gamma$ using the 
    Kitaev-Preskill construction.
    
    $$
    \gamma = S_A + S_B + S_C - S_{AB} - S_{BC} - S_{AC} + S_{ABC}
    $$
    
    where regions A, B, C tile a disk around a reference point, and AB, BC, AC, ABC
    are their unions.
    
    Parameters
    ----------
    nqs : NQS
        Trained wavefunction.
    lattice : Lattice
        Lattice object with `.regions.region_kitaev_preskill(radius)` method.
    q : int
        Rényi index for the entropy measurement.
    radius : float, optional
        Radius for region construction.
    **renyi_kwargs
        Passed to `compute_renyi_entropy`.
        
    Returns
    -------
    dict
        {"gamma": float, "entropies": {key: float}, "errors": {key: float}, "regions": dict}
    """
    if not hasattr(lattice, 'regions'):
        raise ValueError("Lattice must have a 'regions' attribute for TEE calculation.")
    
    regions = lattice.regions.region_kitaev_preskill(radius=radius)
    
    required_keys   = ["A", "B", "C", "AB", "BC", "AC", "ABC"]
    entropies       = {}
    errors          = {}
    
    for key in required_keys:
        if key not in regions or len(regions[key]) == 0:
            entropies[key]  = 0.0
            errors[key]     = 0.0
            continue
        
        sq, sq_err = compute_renyi_entropy(
            nqs, region=regions[key], q=q, return_error=True, **renyi_kwargs
        )
        entropies[key]  = sq
        errors[key]     = sq_err
    
    try:
        gamma = (
            entropies["A"] + entropies["B"] + entropies["C"]
            - entropies["AB"] - entropies["BC"] - entropies["AC"]
            + entropies["ABC"]
        )
        # Propagate errors in quadrature
        gamma_err = np.sqrt(sum(errors[k]**2 for k in required_keys if k in errors))
    except KeyError as e:
        gamma       = float("nan")
        gamma_err   = float("nan")
    
    return {
        "gamma"     : gamma,
        "gamma_err" : gamma_err,
        "q"         : q,
        "entropies" : entropies,
        "errors"    : errors,
        "regions"   : regions,
    }


# ===========================================================================
#! ED entropy computations (exact, for benchmarking)
# ===========================================================================

def compute_ed_entanglement_entropy(
    eigenvectors    : np.ndarray,
    region          : np.ndarray,
    Ns              : int,
    *,
    q_values        : List[int]     = [2],
    n_states        : int           = 1,
    method          : str           = "svd",
) -> Dict[str, Any]:
    r"""
    Compute exact entanglement entropies from ED eigenvectors for benchmarking.
    
    Uses the Schmidt decomposition / reduced density matrix to compute
    both von Neumann and Rényi entropies *exactly* (no statistical error).
    
    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvectors from ED, shape (Hilbert_size, n_states).
    region : array-like
        Site indices defining subsystem A.
    Ns : int
        Total number of sites.
    q_values : list of int
        Rényi indices to compute (e.g., [2, 3, 4]).
    n_states : int
        Number of eigenstates to analyze (default: ground state only).
    method : str
        "svd" (default) or "eigh" for the Schmidt decomposition.
    
    Returns
    -------
    dict
        Keys: "von_neumann" (array), "renyi_{q}" (array for each q),
        "spectrum" (Schmidt values), "region", "region_size".
    """
    region = np.asarray(region, dtype=int)
    complement = np.array([i for i in range(Ns) if i not in region], dtype=int)
    
    dim_A   = 2 ** len(region)
    dim_B   = 2 ** len(complement)
    
    n_states = min(n_states, eigenvectors.shape[1] if eigenvectors.ndim > 1 else 1)
    
    results = {
        "von_neumann"   : np.zeros(n_states),
        "region"        : region,
        "region_size"   : len(region),
        "complement_size": len(complement),
        "system_size"   : Ns,
        "spectrum"      : [],
    }
    for q in q_values:
        results[f"renyi_{q}"] = np.zeros(n_states)
    
    for state_idx in range(n_states):
        if eigenvectors.ndim == 1:
            psi = eigenvectors
        else:
            psi = eigenvectors[:, state_idx]
        
        # Reorder to region|complement factorization
        # Build mapping: new_index -> old_index
        # For each computational basis state |b_0 b_1 ... b_{Ns-1}>,
        # reorder bits so region sites come first
        all_sites   = np.concatenate([region, complement])
        dim_H       = len(psi)
        
        # Build permuted state vector
        psi_perm = np.zeros(dim_H, dtype=psi.dtype)
        for old_idx in range(dim_H):
            # Extract bit pattern
            new_idx = 0
            for new_pos, old_site in enumerate(all_sites):
                bit = (old_idx >> old_site) & 1
                new_idx |= (bit << new_pos)
            psi_perm[new_idx] = psi[old_idx]
        
        # Reshape into (dim_A, dim_B) matrix
        psi_mat = psi_perm.reshape(dim_A, dim_B)
        
        # SVD / Schmidt decomposition
        if method == "svd":
            U, s, Vh = np.linalg.svd(psi_mat, full_matrices=False)
        else:
            # eigh method: compute rho_A = psi_mat @ psi_mat^dag
            rho_A = psi_mat @ psi_mat.conj().T
            eigvals = np.linalg.eigvalsh(rho_A)
            s = np.sqrt(np.maximum(eigvals[::-1], 0))
        
        # Schmidt values (squared = eigenvalues of rho_A)
        p = s ** 2
        p = p[p > 1e-15]  # remove numerical zeros
        p = p / p.sum()    # renormalize
        
        results["spectrum"].append(p)
        
        # Von Neumann entropy  
        results["von_neumann"][state_idx] = -np.sum(p * np.log(p))
        
        # Rényi entropies
        for q in q_values:
            results[f"renyi_{q}"][state_idx] = (1.0 / (1.0 - q)) * np.log(np.sum(p ** q))
    
    return results


# ===========================================================================
#! Entanglement entropy sweep (for scaling analysis)
# ===========================================================================

def compute_entropy_sweep(
    nqs_or_eigvecs  : Any,
    lattice         : Any,
    *,
    q_values        : List[int]     = [2],
    mode            : str           = "nqs",
    num_samples     : int           = 4096,
    num_chains      : int           = 1,
    n_states        : int           = 1,
) -> Dict[str, Any]:
    r"""
    Compute entanglement entropy for multiple bipartition cuts, useful for
    scaling analysis and area-law verification.
    
    Parameters
    ----------
    nqs_or_eigvecs : NQS or np.ndarray
        Either a trained NQS object (mode="nqs") or eigenvectors (mode="ed").
    lattice : Lattice
        Lattice object for generating cuts.
    q_values : list of int
        Rényi indices to compute.
    mode : str
        "nqs" for NQS replica method, "ed" for exact diagonalization.
    num_samples, num_chains : int
        MC parameters (NQS mode only).
    n_states : int
        Number of states (ED mode only).
    
    Returns
    -------
    dict
        {"cuts": dict of cut_label -> region, "results": dict of cut_label -> entropy_result}
    """
    cuts = bipartition_cuts_honeycomb(lattice, cut_type="all")
    
    results = {}
    
    for label, region in cuts.items():
        if len(region) == 0 or len(region) >= lattice.ns:
            continue
        
        if mode == "nqs":
            result_entry = {}
            for q in q_values:
                sq, sq_err = compute_renyi_entropy(
                    nqs_or_eigvecs, region=region, q=q,
                    num_samples=num_samples, num_chains=num_chains,
                    return_error=True,
                )
                result_entry[f"renyi_{q}"]      = sq
                result_entry[f"renyi_{q}_err"]   = sq_err
            result_entry["region_size"] = len(region)
            results[label] = result_entry
            
        elif mode == "ed":
            ed_result = compute_ed_entanglement_entropy(
                nqs_or_eigvecs, region, lattice.ns,
                q_values=q_values, n_states=n_states,
            )
            results[label] = ed_result
        else:
            raise ValueError(f"Unknown mode '{mode}', use 'nqs' or 'ed'.")
    
    return {"cuts": cuts, "results": results}


# ===========================================================================
#! End of file
# ===========================================================================
