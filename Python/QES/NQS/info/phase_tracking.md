# Global Phase Tracking in TDVP

## Overview

This document describes the implementation of global phase parameter tracking in the Time-Dependent Variational Principle (TDVP) solver, based on the dynamics research in neural quantum states.

## Physical Background

### Problem

When computing dynamical correlation functions (e.g., the dynamic structure factor), we need to track the relative phase between different time-evolved quantum states:

$$|\phi_\alpha^{r/q}(t)\rangle = e^{-i\hat{H}t}\hat{\sigma}^{\alpha}_{r/q}|\psi_0\rangle$$

While TDVP naturally handles parameter evolution, it only tracks up to a global phase and normalization. For dynamical correlators, the **relative phase matters**.

### Solution

The paper introduces an additional variational parameter $\theta_0$ that tracks the global phase:

$$|\psi_{\theta_0,\theta}\rangle = e^{\theta_0}|\psi_\theta\rangle$$

## Mathematical Framework

### TDVP Equation for Parameters (Eq. 4)

$$S_{k,k'}\dot{\theta}_{k'} = -iF_k$$

where:

- $S_{k,k'} = \langle\partial_{\theta_k}\psi_\theta|\partial_{\theta_{k'}}\psi_\theta\rangle_c$ is the quantum metric (Fisher matrix)
- $F_k = \langle\partial_{\theta_k}\psi_\theta|\hat{H}|\psi_\theta\rangle_c$ is the force vector

### Equation of Motion for Global Phase (Eq. 5)

$$\dot{\theta}_0 = -i\langle\hat{H}\rangle - \dot{\theta}_k\langle\psi_\theta|\partial_{\theta_k}\psi_\theta\rangle$$

where:

- $\langle\hat{H}\rangle$ is the mean energy
- $\dot{\theta}_k$ are the time derivatives computed from TDVP
- The second term accounts for phase shifts induced by parameter changes

## Implementation

### Class: TDVP

#### Initialization

```python
# Global phase tracking attributes
self._theta0         = 0.0    # Current global phase parameter
self._theta0_dot     = 0.0    # Time derivative of global phase
```

#### Key Methods

**1. `compute_global_phase_evolution(mean_energy, param_derivatives, log_derivatives_centered)`**

Computes $\dot{\theta}_0$ according to Equation (5):

- **Input**: Mean energy, parameter derivatives from TDVP, centered log derivatives
- **Output**: $\dot{\theta}_0$ time derivative

```python
term1 = -1j * mean_energy
term2 = -sum(param_derivatives * log_derivatives_centered)
theta0_dot = term1 + term2
```

**2. `update_global_phase(dt)`**

Integrates the global phase forward in time:

$$\theta_0(t+dt) = \theta_0(t) + dt \times \dot{\theta}_0(t)$$

**3. Properties**

- `global_phase`: Get current $\theta_0$
- `global_phase_dot`: Get current $\dot{\theta}_0$
- `set_global_phase(theta0)`: Set $\theta_0$ (for loading checkpoints)

#### Data Structure: TDVPStepInfo

Extended to include:

```python
theta0_dot : Optional[Array]  # Time derivative of global phase θ̇₀
theta0     : Optional[Array]  # Current global phase parameter θ₀
```

## Workflow

### Single Time Step

1. **Solve TDVP for parameters**

   ```python
   solution = tdvp.solve(e_loc, log_deriv)
   # Returns: θ̇_k (parameter time derivatives)
   ```

2. **Compute global phase evolution** (automatic in `solve()`)

   ```python
   theta0_dot = -i⟨H⟩ - θ̇_k ⟨∂_θk ψ⟩_c
   ```

3. **Update parameters and phase**

   ```python
   theta_k(t+dt) = theta_k(t) + dt * theta_dot_k
   theta0(t+dt) = theta0(t) + dt * theta0_dot
   ```

### Full Time Evolution Loop

```python
from QES.NQS.tdvp import TDVP

# Initialize TDVP
tdvp = TDVP(use_sr=True, backend='jax')

# Time evolution loop
t = 0
dt = 0.01
for step in range(n_steps):
    # Solve TDVP (includes global phase computation)
    param_updates, meta, shapes = tdvp(
        net_params, t,
        est_fn=lambda p, t, c, ca, pr: nqs.step(p, t, c, ca, pr),
        configs=configs,
        configs_ansatze=ansatze,
        probabilities=probs
    )
    
    # Access global phase evolution
    theta0_dot = meta.theta0_dot
    theta0_current = meta.theta0
    
    # Integrate time
    t += dt
```

## Usage Examples

### Example 1: Tracking Dynamical Correlations

```python
# Store state snapshots for correlation computation
snapshots = []
for step in range(n_steps):
    # Get current state
    params_dot, meta, _ = tdvp(...)
    
    # Store phase information
    snapshots.append({
        'params': net_params.copy(),
        'theta0': meta.theta0,        # Global phase
        'theta0_dot': meta.theta0_dot,  # Phase evolution rate
        'time': t
    })
    
    # Update state (including global phase)
    net_params = update_parameters(net_params, params_dot, dt)
    tdvp.update_global_phase(dt)
```

### Example 2: Computing Dynamic Structure Factor

```python
# After time evolution, compute correlations between snapshots
def compute_dsf(snapshot_i, snapshot_j, observable_op):
    """
    Compute correlation function accounting for relative phase.
    
    ⟨ϕ_\alpha(t_i)|O|ϕ_β(t_j)⟩ = e^{i(θ0_i - θ0_j)} ⟨ψ(t_i)|O|ψ(t_j)⟩
    """
    phase_factor = jnp.exp(1j * (snapshot_i['theta0'] - snapshot_j['theta0']))
    
    # Evaluate observable between snapshots
    corr = evaluate_observable(snapshot_i['params'], snapshot_j['params'], observable_op)
    
    # Apply phase correction
    return phase_factor * corr
```

## References

### Primary References

1. **Dynamical Correlations in the Quantum Ising Model using Neural Quantum States**
   - Authors: Schmitt et al.
   - Journal: Physical Review Letters, Vol. 131, 046501 (2023)
   - DOI: [10.1103/PhysRevLett.131.046501](https://doi.org/10.1103/PhysRevLett.131.046501)
   - **Key Content**: Introduces Eq. (5) for global phase evolution in TDVP

2. **Time-Dependent Variational Principle for Quantum Lattices and Applications to Quantum Simulators**
   - Authors: Haegeman et al.
   - Journal: Physical Review B, Vol. 88, 085118 (2013)
   - DOI: [10.1103/PhysRevB.88.085118](https://doi.org/10.1103/PhysRevB.88.085118)
   - **Key Content**: Foundation of TDVP method and quantum metric tensor

3. **The Variational Quantum Eigensolver: a review of best practices, applications, and perspectives**
   - Authors: Cao et al.
   - arXiv: [2012.09265](https://arxiv.org/abs/2012.09265)
   - **Key Content**: General VQE and parameter evolution concepts

### Related Work

- **Machine Learning of Accurate Energy-Conserving Molecular Potentials**
  - Authors: Schütt et al.
  - arXiv: [1902.08408](https://arxiv.org/abs/1902.08408)
  - Framework for neural quantum state representations

- **Neural Quantum States with Mixed-Spin Interactions**
  - Authors: Regy et al.
  - Journal: Physical Review B, Vol. 99, 064424 (2019)
  - **Key Content**: NQS ansatz design and parameter optimization

### Implementation References

- **Equation 4 (TDVP for parameters)**:
  - Derived from time-dependent Schrödinger equation
  - Minimizes Fubini-Study distance to exact evolution
  - Reference: Haegeman et al. (2013), Section II

- **Equation 5 (Global phase evolution)**:
  - Tracks relative phase between time-evolved states
  - Essential for computing dynamical correlation functions
  - Reference: Schmitt et al. (2023), Equation (5)

- **Quantum Metric Tensor** $S_{k,k'}$:
  - Classical Fisher information metric in parameter space
  - Related to state overlap: $\langle\partial_{\theta_k}\psi|\partial_{\theta_{k'}}\psi\rangle_c$
  - Reference: Amari, "Information Geometry and its Applications" (2016)

### Code Citations

The implementation in this module follows the conventions from:

- **JAX Documentation**: Automatic differentiation and compilation
- **Flax Framework**: Neural network building blocks (for potential NQS backends)
- **QES Framework**: Quantum many-body physics primitives

## Notes

- **Phase Accumulation**: The global phase continuously accumulates over time, tracking the overall phase shift
- **For Equal-Time Observables**: The global phase contribution cancels out, so it only matters for dynamical correlations
- **Numerical Precision**: Track both $\theta_0$ and $\dot{\theta}_0$ to maintain precision in long simulations
- **Checkpointing**: When saving/loading NQS state, also save `theta0` to preserve phase information

## Error Handling and Edge Cases

### Common Issues

1. **Phase Wrapping**: For very long simulations, $\theta_0$ may accumulate to large values
   - Solution: Use modular arithmetic if only phase differences matter
   - Keep both absolute phase and relative phases

2. **Numerical Overflow**: Complex exponentials $e^{i\theta_0}$ can overflow
   - Solution: Store phase and apply it only when computing observables
   - Current implementation avoids this by tracking $\theta_0$ separately

3. **Checkpoint Loading**: Ensure $\theta_0$ is preserved when resuming simulations
   - Use `set_global_phase(theta0_saved)` before resuming
   - Include in checkpoint metadata

## See Also

- **`TDVP` Class**: `/QES/NQS/tdvp.py` - Full implementation
- **`NQS` Class**: `/QES/NQS/nqs.py` - Neural quantum state solver using TDVP
- **Stochastic Reconfiguration**: Implementation details in SR module
- **Learning Phases**: `/QES/NQS/src/learning_phases.py` - Multi-phase training scheduling

## Future Enhancements

1. **Multiple Initial States**: Track separate θ₀ for each excited state (for excited state dynamics)
2. **Penalty Terms**: Extend to include lower-state penalty terms in global phase evolution
3. **Optimization**: Further optimize phase tracking for large simulations
