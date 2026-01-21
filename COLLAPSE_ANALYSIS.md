# Collapse Analysis

## The Problem: "Energy Collapse"

The original deep network architectures (CNN, GCNN, Transformer) were producing energies significantly lower than the Exact Diagonalization (ED) ground state, which is a variational violation ($E_{VMC} < E_{ED}$). This indicated a numerical instability in the local energy estimator.

The root cause was identified as the use of a `Dense(1)` output layer (often combined with non-holomorphic activations like ReLU/ELU and normalization layers).

1.  **Bottleneck Divergence**: A single dense output neuron acts as a bottleneck. In deep networks, the activations feeding into this final node can have large variances. The affine transformation $w \cdot x + b$ mapped these to the log-amplitude $\ln \psi(s)$.
2.  **Log-Amplitude Explosion**: Occasionally, for certain configurations $s$, the network would predict an extremely large negative or positive log-amplitude.
3.  **Local Energy Instability**: The local energy estimator involves terms like $H_{s,s'} \frac{\psi(s')}{\psi(s)} = H_{s,s'} e^{\ln \psi(s') - \ln \psi(s)}$. If $\psi(s) \to 0$ (very negative log-amplitude) while $\psi(s')$ remains finite, the ratio explodes. Conversely, if $\psi(s)$ is large, the ratio vanishes. The variance of the estimator becomes infinite, leading to spurious "low energy" measurements dominated by numerical outliers or underflow/overflow issues.

## The Solution: Physics Stabilizers

To fix this, we applied a series of "Physics Stabilizers" inspired by the stability of the RBM and physical principles:

1.  **Global Sum Pooling**: We removed the `Dense(1)` output head. Instead, the final log-amplitude is computed as an **extensive sum** of local hidden unit contributions (summing over all spatial sites and feature channels). This mimics the RBM structure ($\sum \ln \cosh$) and ensures that the log-amplitude scales extensively with system size, preventing localized bottlenecks from driving the global amplitude to divergence.

    $$ \ln \psi(s) = \sum_{i \in \text{sites}} \sum_{f \in \text{features}} h_{i,f}(s) $$

2.  **Holo-Spectral Activations (`log_cosh`)**: We replaced non-holomorphic activations (ReLU, GELU, ELU) with `log_cosh`. This function is:
    *   **Holomorphic**: Preserves complex differentiability, essential for correct gradients in complex-valued optimization (SR/TDVP).
    *   **Smooth & Saturation-Free**: Avoids dead neurons (ReLU) and ensures smooth derivatives.

3.  **Removal of Batch-Norm/LayerNorm**: Normalization layers introduce batch dependencies or break the local gauge symmetry of the wavefunction. They were stripped out to ensure the ansatz depends only on the physical state $s$.

4.  **Stability Epsilon**: We introduced a regularization term $\epsilon = 10^{-12}$ in the local energy calculation to bound the ratio $\psi(s')/\psi(s)$, preventing division-by-zero errors when the sampled probability is vanishingly small.

    $$ \text{weight} = \frac{\psi(s')}{\psi(s) + \epsilon} \approx \exp(\ln \psi(s') - \text{logaddexp}(\ln \psi(s), \ln \epsilon)) $$

These changes restore the variational principle and ensure numerical stability for deep architectures.
