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
    bipartition_cuts,
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

from    __future__ import annotations
from    typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import  warnings
import  numpy as np
from    QES.general_python.physics.density_matrix import mask_subsystem, psi_numpy
from    QES.general_python.physics.entropy import vn_entropy, renyi_entropy

if TYPE_CHECKING:
    from QES.NQS.nqs import NQS
    from QES.general_python.lattices import Lattice

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# ===========================================================================
#! Bipartition helpers
# ===========================================================================

def bipartition_cuts(lattice: "Lattice", cut_type: str = "half_x",) -> Dict[str, np.ndarray]:
    """
    Generate canonical bipartition cuts using the lattice region API.

    This function is lattice-agnostic and relies on:
    - ``lattice.get_entropy_cuts(...)`` (preferred), or
    - ``lattice.regions.get_entropy_cuts(...)`` as fallback.

    Parameters
    ----------
    lattice : Lattice
        Lattice object with region support.
    cut_type : str
        One of: ``half_x``, ``half_y``, ``quarter``, ``sublattice_A``,
        ``sweep``, ``all``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``cut_label -> region_indices``.
    """
    if hasattr(lattice, "get_entropy_cuts"):
        raw_cuts = lattice.get_entropy_cuts(cut_type=cut_type)
    elif hasattr(lattice, "regions") and hasattr(lattice.regions, "get_entropy_cuts"):
        raw_cuts = lattice.regions.get_entropy_cuts(cut_type=cut_type)
    else:
        raise ValueError(
            "Lattice does not expose entropy cut helpers. "
            "Expected `lattice.get_entropy_cuts` or `lattice.regions.get_entropy_cuts`."
        )

    return {label: np.asarray(region, dtype=np.int32) for label, region in raw_cuts.items()}

def _default_bipartition(Ns: int) -> "jnp.ndarray":
    """Default half-system bipartition."""
    return jnp.arange(Ns // 2, dtype=jnp.int32)

# ===========================================================================
#! Core Rényi entropy estimators
# ===========================================================================

def compute_renyi_entropy(
    nqs                     : "NQS",
    region                  : Optional[Any]     = None,
    *,
    q                       : int               = 2,
    num_samples             : int               = 4096,
    num_chains              : int               = 1,
    recompute_log_psi       : bool              = True,
    independent_replicas    : bool              = True,
    return_error            : bool              = False,
    return_raw              : bool              = False,
    min_trace_value         : float             = 1e-15,
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
    recompute_log_psi : bool
        If True (default), re-evaluate ``log(psi)`` on sampled configurations
        using ``nqs.ansatz`` for ratio consistency.
    independent_replicas : bool
        If True (default), reset sampler state before each replica draw.
        This improves independence between replicas in the swap estimator.
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

    # Replica-swap estimator assumes Born sampling, p(s) ~ |psi(s)|^2 (mu=2).
    sampler = getattr(nqs, "sampler", None)
    if sampler is not None:
        mu = getattr(sampler, "mu", getattr(sampler, "_mu", None))
        if mu is not None and not np.isclose(float(mu), 2.0, atol=1e-8):
            warnings.warn(
                f"compute_renyi_entropy assumes Born sampling (mu=2), got mu={mu}. "
                "Results may be biased unless reweighting is applied.",
                RuntimeWarning,
                stacklevel=2,
            )
    
    # Sample q independent replicas 
    replicas_s      = []  # configurations
    replicas_log    = []  # log-amplitudes
    
    for _ in range(q):
        try:
            (_, _), (s_r, log_r), _ = nqs.sample(num_samples=num_samples, num_chains=num_chains, reset=independent_replicas)
        except TypeError:
            raise RuntimeError("NQS sampler error: expected sample() to return ((s, log_psi), ...), got different format.")
        
        s_r = jnp.asarray(s_r)
        if s_r.ndim == 1:
            s_r     = s_r.reshape(1, -1)
        if recompute_log_psi:
            log_r   = jnp.asarray(nqs.ansatz(s_r)).reshape(-1)
        else:
            log_r   = jnp.asarray(log_r).reshape(-1)
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
    
    # Cyclic permutation of subsystem A
    # For replica r, create swapped config: take A from replica (r+1) % q, keep B from replica r
    log_ratio_sum = jnp.zeros(n_samp)
    
    for r in range(q):
        r_next          = (r + 1) % q
        # Build swapped configuration
        s_swapped       = replicas_s[r].at[:, region].set(replicas_s[r_next][:, region])
        # Evaluate log-psi on swapped config
        log_swapped     = jnp.asarray(nqs.ansatz(s_swapped)).reshape(-1)
        # Accumulate log ratio: log[psi(swapped_r)] - log[psi(original_r)]
        log_ratio_sum   = log_ratio_sum + (log_swapped - replicas_log[r])
    
    # Estimate Tr[rho_A^q] in a centered form for numerical stability:
    #   Tr = <exp(log_ratio_sum)>
    #      = exp(shift) * <exp(log_ratio_sum - shift)>,
    # where shift = max(real(log_ratio_sum)).
    # This avoids overflow in both mean and variance calculations for q >= 3.
    shift           = jnp.max(jnp.real(log_ratio_sum))
    swap_centered   = jnp.exp(log_ratio_sum - shift)
    mean_centered   = jnp.mean(swap_centered)
    trace_complex   = jnp.exp(shift) * mean_centered
    trace_val_real  = jnp.real(trace_complex)
    trace_val       = jnp.maximum(trace_val_real, min_trace_value)
    trace_val_f     = float(trace_val)
    
    # Compute S_q 
    sq_val          = float(jnp.real((1.0 / (1.0 - q)) * jnp.log(trace_val)))
    
    # Error estimation (chain-aware)
    n_total         = int(n_samp)
    n_chain         = max(int(num_chains), 1)
    trace_err       = 0.0
    sq_err          = 0.0

    if return_error or return_raw:
        if JAX_AVAILABLE:
            swap_centered_np = np.asarray(jax.device_get(swap_centered))
            mean_centered_np = np.asarray(jax.device_get(mean_centered))
        else:
            swap_centered_np = np.asarray(swap_centered)
            mean_centered_np = np.asarray(mean_centered)

        rel_err = 0.0
        if n_total > 1:
            if n_chain > 1 and (n_total % n_chain) == 0 and (n_total // n_chain) > 1:
                per_chain_centered = swap_centered_np.reshape(n_chain, n_total // n_chain).mean(axis=1)
                stderr_centered = float(np.std(per_chain_centered, ddof=1) / np.sqrt(float(n_chain)))
            else:
                stderr_centered = float(np.std(swap_centered_np, ddof=1) / np.sqrt(float(n_total)))

            mean_abs = max(float(np.abs(mean_centered_np)), np.finfo(float).tiny)
            rel_err = float(stderr_centered / mean_abs)

        trace_err = float(abs(trace_val_f) * rel_err)
    
    # Error propagation: dS_q = |d(Tr)/((1-q)*Tr)|
    if trace_val_f > 0.0:
        sq_err = float(abs(trace_err / ((1.0 - q) * trace_val_f)))
    
    if return_raw:
        return {
            "sq"            : sq_val,
            "sq_err"        : sq_err,
            "q"             : q,
            "trace_rho_q"   : trace_val_f,
            "trace_err"     : trace_err,
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
    nqs         : "NQS",
    lattice     : Any,
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
    
    regions         = lattice.regions.region_kitaev_preskill(radius=radius)
    required_keys   = ["A", "B", "C", "AB", "BC", "AC", "ABC"]
    entropies       = {}
    errors          = {}
    
    for key in required_keys:
        if key not in regions or len(regions[key]) == 0:
            entropies[key]  = 0.0
            errors[key]     = 0.0
            continue
        
        sq, sq_err      = compute_renyi_entropy(nqs, region=regions[key], q=q, return_error=True, **renyi_kwargs)
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
    region                  = np.asarray(region, dtype=int)
    (size_a, size_b), order = mask_subsystem(region, Ns, local_dim=2, contiguous=False)
    dim_A                   = 2 ** size_a
    dim_B                   = 2 ** size_b
    
    n_states    = min(n_states, eigenvectors.shape[1] if eigenvectors.ndim > 1 else 1)
    
    results     = {
        "von_neumann"       : np.zeros(n_states),
        "region"            : region,
        "region_size"       : size_a,
        "complement_size"   : size_b,
        "system_size"       : Ns,
        "spectrum"          : [],
    }
    for q in q_values:
        results[f"renyi_{q}"] = np.zeros(n_states)
    
    for state_idx in range(n_states):
        if eigenvectors.ndim == 1:
            psi = eigenvectors
        else:
            psi = eigenvectors[:, state_idx]
        
        psi_mat = psi_numpy(np.asarray(psi), order, size_a, Ns, local_dim=2)
        if psi_mat.shape != (dim_A, dim_B):
            raise RuntimeError(f"Unexpected ED reshape: got {psi_mat.shape}, expected {(dim_A, dim_B)}.")
        
        # SVD / Schmidt decomposition
        if method == "svd":
            s           = np.linalg.svd(psi_mat, full_matrices=False, compute_uv=False)
        else:
            # eigh method: compute rho_A = psi_mat @ psi_mat^dag
            rho_A       = psi_mat @ psi_mat.conj().T
            eigvals     = np.linalg.eigvalsh(rho_A)
            s           = np.sqrt(np.maximum(eigvals, 0.0))
        
        # Schmidt values (squared = eigenvalues of rho_A)
        p               = s ** 2
        p               = p[p > 1e-15]  # remove numerical zeros
        if p.size == 0:
            p           = np.array([1.0], dtype=float)
        else:
            p           = np.sort(p)[::-1]
            p           = p / p.sum() # renormalize
        
        results["spectrum"].append(p)
        
        # Von Neumann entropy  
        results["von_neumann"][state_idx] = float(vn_entropy(p))
        
        # Rényi entropies
        for q in q_values:
            results[f"renyi_{q}"][state_idx] = float(renyi_entropy(p, float(q)))
    
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
    cuts    = bipartition_cuts(lattice, cut_type="all")
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
