# Entanglement Entropy in NQS via Replica Methods

## Overview

This document describes the physics behind the entanglement entropy computation
in our NQS framework, implemented in `NQS/src/nqs_entropy.py`.

## 1. Entanglement Entropy Basics

For a pure quantum state $|\Psi\rangle$ on a lattice partitioned into regions $A$ and $B$,
the reduced density matrix is:

$$
\rho_A = \text{Tr}_B |\Psi\rangle\langle\Psi|
$$

### Von Neumann Entropy

$$
S_\text{vN}(A) = -\text{Tr}[\rho_A \ln \rho_A] = -\sum_i \lambda_i \ln \lambda_i
$$

### Rényi Entropy

$$
S_q(A) = \frac{1}{1-q} \ln \text{Tr}[\rho_A^q], \quad q \ge 2
$$

In the limit $q \to 1$, Rényi entropy reduces to von Neumann entropy.
For topological phases, the sub-leading correction (area law) is:
$$
S(A) = \alpha |\partial A| - \gamma + \ldots
$$
where $\gamma$ is the **topological entanglement entropy (TEE)**.

## 2. Replica Swap Trick for NQS

### Why replicas?

Direct access to $\rho_A$ requires exponential resources. Instead, we use the
**replica trick**: create $q$ independent copies of $|\Psi\rangle$ and use a
swap/cyclic-permutation operator to extract $\text{Tr}[\rho_A^q]$.

### $S_2$ (Two-Replica Estimator)

Sample two independent sets of configurations from $|\Psi|^2$:

- Replica 1: $\{s^{(1)}_i\} \sim |\Psi(s)|^2$
- Replica 2: $\{s^{(2)}_i\} \sim |\Psi(s)|^2$

For each pair $(s^{(1)}, s^{(2)})$, construct swapped configs by exchanging subsystem A:

- $\tilde{s}^{(1)} = (s^{(2)}_A, s^{(1)}_B)$
- $\tilde{s}^{(2)} = (s^{(1)}_A, s^{(2)}_B)$

Then:
$$
\text{Tr}[\rho_A^2] = \left\langle \frac{\Psi(\tilde{s}^{(1)}) \Psi(\tilde{s}^{(2)})}
{\Psi(s^{(1)}) \Psi(s^{(2)})} \right\rangle
$$

And $S_2 = -\ln \text{Tr}[\rho_A^2]$.

### $S_q$ (q-Replica Estimator)

For general integer $q \ge 2$, sample $q$ independent replicas and apply cyclic permutation:
$$
\text{Tr}[\rho_A^q] = \left\langle \prod_{r=0}^{q-1}
\frac{\Psi(s^{((r+1)\bmod q)}_A, s^{(r)}_B)}{\Psi(s^{(r)}_A, s^{(r)}_B)} \right\rangle
$$

**Correspondence to standard Rényi**: The cyclic permutation operator $\hat{C}_A^{(q)}$
acting on $q$ copies of $\mathcal{H}_A \otimes \mathcal{H}_B$ satisfies:
$$
\text{Tr}[\hat{C}_A^{(q)} (\rho^{\otimes q})] = \text{Tr}[\rho_A^q]
$$
This identity (proven by Hastings et al., PRL 104, 157201) makes the replica approach exact
in the infinite-sample limit.

### Practical considerations

1. **Log-space arithmetic**: We compute everything in log-space to avoid numerical overflow:
   $$\ln R_i = \sum_{r=0}^{q-1} [\ln\Psi(\text{swapped}_r) - \ln\Psi(\text{original}_r)]$$
   Then $\text{Tr}[\rho_A^q] = \text{logsumexp}(\{R_i\}) - \ln N_s$.

2. **Error estimation**: Chain-aware error bars using inter-chain variance when multiple
   Markov chains are used:
   $$\sigma_S = \frac{\sigma_{\text{Tr}}}{\text{Tr}[\rho_A^q]}$$

3. **Sample requirements**: For accurate $S_q$ with $q \ge 3$, more samples are needed
   because the estimator variance grows exponentially with $q$ and the region size.

## 3. Topological Entanglement Entropy (TEE)

### Kitaev-Preskill Construction

For a 2D topological phase, we compute:
$$
\gamma = S_A + S_B + S_C - S_{AB} - S_{BC} - S_{AC} + S_{ABC}
$$

where A, B, C are three wedge-shaped regions meeting at a point. This construction
cancels the area-law contribution and isolates the universal topological term.

For the Kitaev model on the honeycomb lattice, $\gamma = \ln 2$
(reflecting the $\mathbb{Z}_2$ topological order).

### Levin-Wen Construction

Alternative: $\gamma = S_\text{in} + S_\text{out} - S_\text{in+out}$
using an annular region.

## 4. Bipartitions for the Honeycomb Lattice

For a honeycomb lattice with $L_x \times L_y$ unit cells ($N_s = 2 L_x L_y$ sites):

- **half_x**: Vertical cut at $x = L_x/2$ — standard for entanglement scaling
- **half_y**: Horizontal cut at $y = L_y/2$
- **quarter**: Corner region (1/4 of system)
- **sublattice_A**: All A-sublattice sites
- **sweep**: Incremental cuts for entanglement entropy vs. subsystem size

## 5. Usage in Practice

### NQS (Monte Carlo sampling)

```python
from QES.NQS.src.nqs_entropy import compute_renyi_entropy, bipartition_cuts

cuts = bipartition_cuts(lattice, cut_type="all")
for label, region in cuts.items():
    s2, s2_err = compute_renyi_entropy(nqs, region=region, q=2, return_error=True)
    print(f"{label}: S_2 = {s2:.4f} ± {s2_err:.4f}")
```

### ED (exact benchmark)

```python
from QES.NQS.src.nqs_entropy import compute_ed_entanglement_entropy

result = compute_ed_entanglement_entropy(
    hamiltonian.eig_vec, region, lattice.ns,
    q_values=[2, 3], n_states=1
)
print(f"S_vN = {result['von_neumann'][0]:.6f}")
print(f"S_2  = {result['renyi_2'][0]:.6f}")
```

## References

1. Hastings, Gonzalez, Tubman, Abrams, PRL 104, 157201 (2010)
2. Flammia, Hamma, Hughes, Wen, PRL 103, 261601 (2009)
3. Kitaev, Preskill, PRL 96, 110404 (2006)
4. Levin, Wen, PRL 96, 110405 (2006)
5. Humeniuk, Roscilde, PRB 86, 235116 (2012)
