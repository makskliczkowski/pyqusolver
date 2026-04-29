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
Version     : 0.1 (Experimental)
-------------------------------------------------------------------------------
"""

from    __future__ import annotations
from    typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import  warnings
import  numpy as np
from    QES.general_python.physics.density_matrix import mask_subsystem, psi_numpy
from    QES.general_python.physics.entropy import vn_entropy, renyi_entropy

if TYPE_CHECKING:
    from QES.NQS.nqs                    import NQS
    from QES.general_python.lattices    import Lattice
    from QES.general_python.physics     import entropy as qes_entropy

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# ===========================================================================
#! Bipartition helpers
# ===========================================================================

def bipartition_cuts(lattice: Any, *, cut_type: str = "all", **kwargs) -> Dict[str, np.ndarray]:
    """
    Return entropy bipartitions from the lattice or a conservative fallback.

    Lattice implementations can provide domain-aware cuts through
    ``get_entropy_cuts``. If unavailable, this helper returns a single
    half-system cut so ``compute_entropy_sweep`` remains usable for simple
    chains and small regression systems.
    """
    if hasattr(lattice, "get_entropy_cuts"):
        return lattice.get_entropy_cuts(cut_type=cut_type, **kwargs)

    ns = int(getattr(lattice, "ns", 0))
    if ns <= 0:
        raise ValueError("lattice must expose ns or get_entropy_cuts().")
    half = np.arange(ns // 2, dtype=int)
    if str(cut_type).lower() in ("all", "half", "half_system"):
        return {"half": half}
    raise ValueError("Unsupported cut_type for fallback bipartition_cuts.")


def _enumerate_basis_states_nqs(nqs: "NQS") -> np.ndarray:
    """
    Enumerate the full computational basis for exact small-system diagnostics.

    This helper is used by the exact entropy path so that examples and tests can
    validate the Rényi machinery without Monte Carlo bias when the Hilbert space
    is tiny.
    """
    from QES.general_python.common.binary   import int2base
    from .network                          import resolve_nqs_state_defaults

    hilbert = getattr(nqs, "hilbert", None)
    if hilbert is None:
        raise ValueError("Exact NQS entropy requires an attached Hilbert space.")

    spin, spin_value = resolve_nqs_state_defaults(nqs, fallback_mode_repr=0.5)

    basis_int       = np.asarray(list(hilbert), dtype=np.int64).reshape(-1)
    ns              = int(getattr(hilbert, "ns", nqs.nvisible))
    # Convert integer basis states to spin configurations in the expected representation
    return np.stack([int2base(int(state), ns, spin=spin, spin_value=spin_value, backend="np") for state in basis_int], axis=0,).astype(np.float32, copy=False)

def _exact_nqs_wavefunction(nqs: "NQS") -> np.ndarray:
    """
    Materialize the full normalized NQS wavefunction on the computational basis.
    """

    states  = _enumerate_basis_states_nqs(nqs)
    log_psi = np.asarray(nqs.ansatz(states), dtype=np.complex128).reshape(-1)
    psi     = np.exp(log_psi)
    norm    = np.linalg.norm(psi)
    if norm <= 0.0:
        raise ValueError("Exact NQS entropy received a zero-norm wavefunction.")
    return psi / norm

# ===========================================================================
#! Core Rényi entropy estimators
# ===========================================================================

def _validate_renyi_index(q: int) -> int:
    if q < 2 or not isinstance(q, int):
        raise ValueError(f"Rényi index q must be an integer >= 2, got {q}.")
    return int(q)

def _normalize_region(region: Optional[Any], Ns: int) -> jnp.ndarray:
    if region is None:
        region_np = np.arange(Ns // 2, dtype=np.int32)
    else:
        region_np = np.asarray(region, dtype=np.int32).reshape(-1)
    if np.any(region_np < 0) or np.any(region_np >= Ns):
        raise ValueError(f"Region contains out-of-bounds indices for n_visible={Ns}.")
    return jnp.asarray(region_np, dtype=jnp.int32)

def _check_born_sampling(nqs: "NQS") -> None:
    sampler = getattr(nqs, "sampler", None)
    if sampler is None:
        return
    mu = getattr(sampler, "mu", getattr(sampler, "_mu", None))
    if mu is not None and not np.isclose(float(mu), 2.0, atol=1e-8):
        warnings.warn(f"compute_renyi_entropy assumes Born sampling (mu=2), got mu={mu}. Results may be biased unless reweighting is applied.", RuntimeWarning, stacklevel=3)

def _draw_replica_batches(nqs: "NQS", q: int, *, num_samples: Optional[int], num_chains: Optional[int], independent_replicas: bool) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    replicas_s_list   = []
    replicas_log_list = []

    for _ in range(q):
        try:
            (_, _), (s_r, log_r), _ = nqs.sample(num_samples=num_samples, num_chains=num_chains, reset=independent_replicas)
        except TypeError:
            raise RuntimeError("NQS sampler error: expected sample() to return ((s, log_psi), ...), got different format.")

        s_r = jnp.asarray(s_r)
        if s_r.ndim == 1:
            s_r = s_r.reshape(1, -1)
        replicas_s_list.append(s_r)
        replicas_log_list.append(jnp.asarray(log_r).reshape(-1))

    min_n               = min(s.shape[0] for s in replicas_s_list)
    replicas_s_list     = [s[:min_n] for s in replicas_s_list]
    replicas_log_list   = [log[:min_n] for log in replicas_log_list]
    replicas_s          = jnp.stack(replicas_s_list, axis=0)
    replicas_log        = jnp.stack(replicas_log_list, axis=0)
    return replicas_s, replicas_log, int(min_n)

def _maybe_recompute_replica_logs(nqs: "NQS", replicas_s: jnp.ndarray, replicas_log: jnp.ndarray, *, recompute_log_psi: bool) -> jnp.ndarray:
    ''' Optionally re-evaluate log(psi) on sampled configurations for ratio consistency. '''
    if not recompute_log_psi:
        return replicas_log
    q, n_samp, Ns   = replicas_s.shape
    s_flat          = replicas_s.reshape(q * n_samp, Ns)
    log_flat        = jnp.asarray(nqs.ansatz(s_flat)).reshape(-1)
    return log_flat.reshape(q, n_samp)

def _resolved_num_chains(nqs: "NQS", num_chains: Optional[int]) -> int:
    ''' Determine the number of Markov chains to use for error estimation, preferring the provided num_chains, then sampler attributes, and defaulting to 1. '''
    if num_chains is not None:
        return max(int(num_chains), 1)
    sampler             = getattr(nqs, "sampler", None)
    sampler_num_chains  = getattr(sampler, "_numchains", None) if sampler is not None else None
    if sampler_num_chains is None:
        sampler_num_chains = getattr(sampler, "numchains", None) if sampler is not None else None
    return max(int(sampler_num_chains) if sampler_num_chains is not None else 1, 1)

def _renyi_stats_from_log_ratio(log_ratio_sum: jnp.ndarray, *, q: int, n_chain: int, min_trace_value: float, compute_error: bool) -> Dict[str, Any]:
    ''' Compute Rényi entropy and error estimates from log ratio sums. '''
    log_ratio_sum = jnp.asarray(log_ratio_sum)
    if log_ratio_sum.ndim == 1:
        log_ratio_sum = log_ratio_sum.reshape(1, -1)

    shift           = jnp.max(jnp.real(log_ratio_sum), axis=1)
    swap_centered   = jnp.exp(log_ratio_sum - shift[:, None])
    mean_centered   = jnp.mean(swap_centered, axis=1)
    trace_complex   = jnp.exp(shift) * mean_centered
    trace_val       = jnp.maximum(jnp.real(trace_complex), min_trace_value)
    sq_val          = jnp.real((1.0 / (1.0 - q)) * jnp.log(trace_val))

    sq_np           = np.asarray(jax.device_get(sq_val), dtype=np.float64)
    trace_np        = np.asarray(jax.device_get(trace_val), dtype=np.float64)
    trace_err       = np.zeros_like(trace_np)
    sq_err          = np.zeros_like(trace_np)

    if compute_error:
        centered_np = np.asarray(jax.device_get(swap_centered))
        mean_np     = np.asarray(jax.device_get(mean_centered))
        n_total     = centered_np.shape[1]
        if n_total > 1:
            for idx in range(centered_np.shape[0]):
                if n_chain > 1 and (n_total % n_chain) == 0 and (n_total // n_chain) > 1:
                    per_chain       = centered_np[idx].reshape(n_chain, n_total // n_chain).mean(axis=1)
                    stderr_centered = float(np.std(per_chain, ddof=1) / np.sqrt(float(n_chain)))
                else:
                    stderr_centered = float(np.std(centered_np[idx], ddof=1) / np.sqrt(float(n_total)))
                mean_abs        = max(float(np.abs(mean_np[idx])), np.finfo(float).tiny)
                rel_err         = float(stderr_centered / mean_abs)
                trace_err[idx]  = float(abs(trace_np[idx]) * rel_err)
                if trace_np[idx] > 0.0:
                    sq_err[idx] = float(abs(trace_err[idx] / ((1.0 - q) * trace_np[idx])))

    return { "sq": sq_np, "sq_err": sq_err, "trace_rho_q": trace_np, "trace_err": trace_err }

def _log_ratio_for_region(nqs: "NQS", replicas_s: jnp.ndarray, replicas_log: jnp.ndarray, region: jnp.ndarray, q: int) -> jnp.ndarray:
    ''' Compute the log ratio sum for a single region using cyclic permutation of subsystem A across q replicas. '''
    replicas_s_q        = replicas_s[:q]
    replicas_log_q      = replicas_log[:q]
    next_indices        = (jnp.arange(q) + 1) % q
    s_swapped           = replicas_s_q.at[:, :, region].set(replicas_s_q[next_indices][:, :, region])
    q_eff, n_samp, Ns   = s_swapped.shape
    log_swapped_flat    = jnp.asarray(nqs.ansatz(s_swapped.reshape(q_eff * n_samp, Ns))).reshape(-1)
    log_swapped         = log_swapped_flat.reshape(q_eff, n_samp)
    return jnp.sum(log_swapped - replicas_log_q, axis=0)

def _log_ratio_for_regions(nqs: "NQS", replicas_s: jnp.ndarray, replicas_log: jnp.ndarray, regions: List[jnp.ndarray], q: int) -> jnp.ndarray:
    ''' Compute the log ratio sums for multiple regions in a batch by evaluating all cyclic permutations together and masking the appropriate sites for each region.  '''
    replicas_s_q        = replicas_s[:q]
    replicas_log_q      = replicas_log[:q]
    next_indices        = (jnp.arange(q) + 1) % q
    next_s              = replicas_s_q[next_indices]
    q_eff, n_samp, Ns   = replicas_s_q.shape

    masks_np = np.zeros((len(regions), Ns), dtype=bool)
    for idx, region in enumerate(regions):
        masks_np[idx, np.asarray(region, dtype=np.int32)] = True
    masks               = jnp.asarray(masks_np)
    swapped             = jnp.where(masks[:, None, None, :], next_s[None, :, :, :], replicas_s_q[None, :, :, :])
    log_swapped_flat    = jnp.asarray(nqs.ansatz(swapped.reshape(len(regions) * q_eff * n_samp, Ns))).reshape(-1)
    log_swapped         = log_swapped_flat.reshape(len(regions), q_eff, n_samp)
    return jnp.sum(log_swapped - replicas_log_q[None, :, :], axis=1)

# ==========================================================================
#! Rényi Entropy Estimation
# ==========================================================================

def compute_renyi_entropy(
    nqs                     : "NQS",
    region                  : Optional[Any]     = None,
    *,
    q                       : int               = 2,
    num_samples             : Optional[int]     = None,
    num_chains              : Optional[int]     = None,
    recompute_log_psi       : bool              = True,
    independent_replicas    : bool              = False,
    exact_sum               : bool              = False,
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
    num_samples : int or None
        Number of Monte Carlo samples per replica. If None, reuse the sampler's
        current configured value instead of forcing a reinitialization.
    num_chains : int or None
        Number of Markov chains for sampling. If None, reuse the sampler's
        current configured value instead of forcing a reinitialization.
    recompute_log_psi : bool
        If True (default), re-evaluate ``log(psi)`` on sampled configurations
        using ``nqs.ansatz`` for ratio consistency.
    independent_replicas : bool
        If True, reset sampler state before each replica draw. This may help
        decorrelate replicas, but for local MCMC samplers it can also destroy a
        well-equilibrated chain ensemble and bias the entropy estimate. The
        default keeps the current sampler state and draws successive replica
        batches from the warm chains.
    exact_sum : bool
        If True, bypass Monte Carlo and compute the Rényi entropy from the full
        NQS wavefunction on the computational basis. This is intended for tiny
        ED-style benchmarks where one wants deterministic validation.
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
    if exact_sum:
        if region is None:
            region_np = np.arange(nqs.nvisible // 2, dtype=int)
        else:
            region_np = np.asarray(region, dtype=int).reshape(-1)
        exact = compute_ed_entanglement_entropy(_exact_nqs_wavefunction(nqs),
            region_np, int(nqs.nvisible), q_values=[q], n_states=1,
        )
        sq_val = float(exact[f"renyi_{q}"][0])
        if return_raw:
            return {
                "sq"            : sq_val,
                "sq_err"        : 0.0,
                "q"             : q,
                "trace_rho_q"   : float(np.exp((1.0 - q) * sq_val)),
                "trace_err"     : 0.0,
                "n_samples"     : int(getattr(nqs.hilbert, "Nh", 2 ** nqs.nvisible)),
                "n_chains"      : 1,
                "region_size"   : int(region_np.size),
                "system_size"   : int(nqs.nvisible),
                "exact_sum"     : True,
            }
        if return_error:
            return sq_val, 0.0
        return sq_val

    if not JAX_AVAILABLE:
        raise NotImplementedError("Rényi entropy estimation requires JAX backend.")
    q       = _validate_renyi_index(q)

    Ns      = int(nqs.nvisible)
    region  = _normalize_region(region, Ns)
    _check_born_sampling(nqs)

    replicas_s, replicas_log, n_samp = _draw_replica_batches(
                        nqs,
                        q,
                        num_samples=num_samples,
                        num_chains=num_chains,
                        independent_replicas=independent_replicas,
                    )
    replicas_log = _maybe_recompute_replica_logs(
                        nqs,
                        replicas_s,
                        replicas_log,
                        recompute_log_psi=recompute_log_psi,
                    )
    log_ratio_sum   = _log_ratio_for_region(nqs, replicas_s, replicas_log, region, q)
    n_chain         = _resolved_num_chains(nqs, num_chains)
    stats           = _renyi_stats_from_log_ratio(
                        log_ratio_sum,
                        q=q,
                        n_chain=n_chain,
                        min_trace_value=min_trace_value,
                        compute_error=return_error or return_raw,
                    )
    sq_val          = float(stats["sq"][0])
    sq_err          = float(stats["sq_err"][0])
    trace_val_f     = float(stats["trace_rho_q"][0])
    trace_err       = float(stats["trace_err"][0])
    
    if return_raw:
        return {
            "sq"            : sq_val,
            "sq_err"        : sq_err,
            "q"             : q,
            "trace_rho_q"   : trace_val_f,
            "trace_err"     : trace_err,
            "n_samples"     : int(n_samp),
            "n_chains"      : n_chain,
            "region_size"   : len(region),
            "system_size"   : Ns,
        }
    if return_error:
        return sq_val, sq_err
    return sq_val

def compute_renyi_entropies(
    nqs                     : "NQS",
    regions                 : Union[Dict[Any, Any], List[Any], Tuple[Any, ...]],
    *,
    q_values                : List[int]         = [2],
    num_samples             : Optional[int]     = None,
    num_chains              : Optional[int]     = None,
    recompute_log_psi       : bool              = True,
    independent_replicas    : bool              = False,
    exact_sum               : bool              = False,
    return_error            : bool              = True,
    min_trace_value         : float             = 1e-15,
    region_batch_size       : Optional[int]     = 16,
) -> Dict[Any, Dict[str, Any]]:
    r"""
    Compute Rényi entropies for many regions and indices with shared replicas.

    This is the fast path for notebook sweeps such as all honeycomb cuts and
    ``q_values=[2, 3]``. It samples ``max(q_values)`` replicas once, recomputes
    original log amplitudes once, then batches the swapped-state ansatz
    evaluation across region batches for each q. Compared with repeatedly calling
    :func:`compute_renyi_entropy`, this removes repeated sampler warm-up/state
    advancement and greatly reduces Python/JAX dispatch overhead.

    The return format mirrors the manual notebook dictionaries:
    ``result[label]["renyi_2"]``, ``result[label]["renyi_2_err"]``, etc.
    """
    q_values = sorted({_validate_renyi_index(int(q)) for q in q_values})
    if not q_values:
        raise ValueError("q_values must contain at least one Rényi index.")

    if isinstance(regions, dict):
        region_items = list(regions.items())
    else:
        region_items = [(idx, region) for idx, region in enumerate(regions)]

    if exact_sum:
        results: Dict[Any, Dict[str, Any]] = {}
        for label, region in region_items:
            region_np               = np.asarray(region, dtype=int).reshape(-1)
            entry: Dict[str, Any]   = {"region_size": int(region_np.size)}
            exact                   = compute_ed_entanglement_entropy(
                                        _exact_nqs_wavefunction(nqs),
                                            region_np,
                                            int(nqs.nvisible),
                                            q_values=q_values,
                                            n_states=1,
                                        )
            for q in q_values:
                entry[f"renyi_{q}"] = float(exact[f"renyi_{q}"][0])
                if return_error:
                    entry[f"renyi_{q}_err"] = 0.0
            results[label] = entry
        return results

    if not JAX_AVAILABLE:
        raise NotImplementedError("Rényi entropy estimation requires JAX backend.")

    Ns                  = int(nqs.nvisible)
    normalized_items    = [(label, _normalize_region(region, Ns)) for label, region in region_items]
    if not normalized_items:
        return {}

    _check_born_sampling(nqs)
    q_max = max(q_values)
    replicas_s, replicas_log, n_samp = _draw_replica_batches(
                                        nqs,
                                        q_max,
                                        num_samples=num_samples,
                                        num_chains=num_chains,
                                        independent_replicas=independent_replicas,
                                    )
    replicas_log = _maybe_recompute_replica_logs(
                    nqs,
                    replicas_s,
                    replicas_log,
                    recompute_log_psi=recompute_log_psi,
                )
    n_chain = _resolved_num_chains(nqs, num_chains)

    labels          = [label for label, _ in normalized_items]
    region_arrays   = [region for _, region in normalized_items]
    results         = {
                        label: {"region_size": int(region.size), "n_samples": int(n_samp), "n_chains": n_chain}
                        for label, region in normalized_items
                    }
    if region_batch_size is None:
        region_batch_size = len(normalized_items)
    region_batch_size = max(int(region_batch_size), 1)

    for q in q_values:
        for start in range(0, len(region_arrays), region_batch_size):
            stop            = min(start + region_batch_size, len(region_arrays))
            batch_regions   = region_arrays[start:stop]
            batch_labels    = labels[start:stop]
            log_ratio       = _log_ratio_for_regions(nqs, replicas_s, replicas_log, batch_regions, q)
            stats           = _renyi_stats_from_log_ratio(
                                log_ratio,
                                q=q,
                                n_chain=n_chain,
                                min_trace_value=min_trace_value,
                                compute_error=return_error,
                            )
            for idx, label in enumerate(batch_labels):
                results[label][f"renyi_{q}"]        = float(stats["sq"][idx])
                results[label][f"trace_rho_{q}"]    = float(stats["trace_rho_q"][idx])
                if return_error:
                    results[label][f"renyi_{q}_err"]        = float(stats["sq_err"][idx])
                    results[label][f"trace_rho_{q}_err"]    = float(stats["trace_err"][idx])

    return results


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
    region_batch_size: Optional[int] = 16,
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
    cuts        = bipartition_cuts(lattice, cut_type="all")
    results     = {}

    valid_cuts  = { label: region for label, region in cuts.items() if len(region) != 0 and len(region) < lattice.ns }

    if mode == "nqs":
        results = compute_renyi_entropies(
            nqs_or_eigvecs,
            valid_cuts,
            q_values=q_values,
            num_samples=num_samples,
            num_chains=num_chains,
            return_error=True,
            region_batch_size=region_batch_size,
        )
    elif mode == "ed":
        for label, region in valid_cuts.items():
            ed_result       = compute_ed_entanglement_entropy(nqs_or_eigvecs, region, lattice.ns, q_values=q_values, n_states=n_states)
            results[label]  = ed_result
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'nqs' or 'ed'.")
    
    return {"cuts": cuts, "results": results}


# ===========================================================================
#! End of file
# ===========================================================================
