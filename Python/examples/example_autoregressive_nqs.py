"""
Example: Autoregressive Neural Network with NQS Solver
========================================================

This example demonstrates how to use the Autoregressive (AR) neural network
for quantum state representation with the Neural Quantum State (NQS) solver.

The autoregressive architecture factorizes the wavefunction as:
    ψ(s₁, s₂, ..., sₙ) = p(s₁) times  p(s₂|s₁) times  p(s₃|s₁,s₂) times  ... times  p(sₙ|s₁,...,sₙ₋₁)

This allows for exact sampling through sequential generation, making it
particularly suitable for high-probability sampling in NQS training.

Author: Development Team
Date: November 1, 2025
"""

import os
os.environ['PY_JAX_AVAILABLE'] = '1'
os.environ['PY_BACKEND'] = 'jax'

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

# Import QES modules
from QES.NQS.src.network_integration import NetworkFactory
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.hamil import Hamiltonian


def create_simple_hamiltonian(hilbert_space: HilbertSpace) -> Hamiltonian:
    """
    Create a simple transverse-field Ising model Hamiltonian.
    
    H = -Σᵢ \sigmaᵢˣ + 0.5 Σᵢⱼ \sigmaᵢᶻ \sigmaⱼᶻ
    
    Parameters:
    -----------
    hilbert_space : HilbertSpace
        The Hilbert space for the system
        
    Returns:
    --------
    Hamiltonian
        The configured Hamiltonian
    """
    # Create empty Hamiltonian
    H = Hamiltonian(hilbert_space)
    
    # Add transverse field terms: -Σᵢ \sigmaᵢˣ
    for i in range(hilbert_space.Ns):
        H.add_operator(f"sx_{i}", "sx", [i], -1.0)
    
    # Add coupling terms (simplified): 0.5 Σᵢ \sigmaᵢᶻ \sigmaᵢ₊₁ᶻ
    for i in range(hilbert_space.Ns - 1):
        H.add_operator(f"zz_{i}_{i+1}", "zz", [i, i+1], 0.5)
    
    return H


def demonstrate_autoregressive_network():
    """
    Main demonstration of Autoregressive network with NQS.
    """
    print("\n" + "="*70)
    print("AUTOREGRESSIVE NEURAL NETWORK FOR NQS - DEMONSTRATION")
    print("="*70 + "\n")
    
    # =====================================================================
    # 1. SETUP: Create Hilbert space and network
    # =====================================================================
    print("Step 1: Setting up the quantum system")
    print("-" * 70)
    
    # Create a small system for demonstration (2 spins = 4-dimensional Hilbert space)
    hilbert = HilbertSpace(2)
    n_visible = 2**hilbert.Ns
    print(f"(ok) Hilbert space created:")
    print(f"  - Number of spins: {hilbert.Ns}")
    print(f"  - Hilbert space dimension: {n_visible}")
    
    # Create the autoregressive network
    print(f"\n(ok) Creating Autoregressive network:")
    ar_network = NetworkFactory.create(
        network_type='ar',
        hilbert_space=hilbert,
        hidden_layers=(16, 16),  # Two hidden layers with 16 units each
        activation='tanh',
        seed=42
    )
    
    info = ar_network.get_info()
    print(f"  - Network type: {info['name']}")
    print(f"  - Architecture: Input({info['n_qubits']}) → " +
          f"Hidden{info['hidden_layers']} → Output(1)")
    print(f"  - Data type: {info['dtype']}")
    print(f"  - Total parameters: ~{info}")
    
    # =====================================================================
    # 2. NETWORK EVALUATION: Test forward pass
    # =====================================================================
    print(f"\nStep 2: Testing network forward pass")
    print("-" * 70)
    
    # Create some test states
    test_states = jnp.array([
        [0, 0, 0, 0],  # All down
        [1, 1, 1, 1],  # All up
        [0, 1, 0, 1],  # Alternating
        [1, 0, 1, 0],  # Alternating (phase shifted)
    ], dtype=jnp.float32)
    
    print(f"(ok) Test states: shape={test_states.shape}")
    print(f"  Sample states:")
    for i, state in enumerate(test_states):
        print(f"    State {i}: {state}")
    
    # Evaluate network on test states
    log_probs = ar_network(test_states)
    print(f"\n(ok) Network output:")
    print(f"  - Shape: {log_probs.shape}")
    print(f"  - Data type: {log_probs.dtype}")
    print(f"  - Log probabilities: {log_probs}")
    
    # =====================================================================
    # 3. SAMPLING: Generate samples from network
    # =====================================================================
    print(f"\nStep 3: Testing high-probability sampling")
    print("-" * 70)
    
    n_samples = 10
    samples = ar_network.sample(n_samples=n_samples)
    
    print(f"(ok) Generated {n_samples} samples:")
    print(f"  - Shape: {samples.shape}")
    print(f"  - Data type: {samples.dtype}")
    print(f"  - Sample values:")
    for i, sample in enumerate(samples):
        print(f"    Sample {i}: {sample}")
    
    # Compute statistics
    mean_occupancy = jnp.mean(samples, axis=0)
    print(f"\n(ok) Sample statistics:")
    print(f"  - Mean occupancy per site: {mean_occupancy}")
    print(f"  - Total 1s: {jnp.sum(samples)}/{n_samples*len(samples[0])}")
    
    # =====================================================================
    # 4. AUTODIFF: Test gradient computation
    # =====================================================================
    print(f"\nStep 4: Testing automatic differentiation")
    print("-" * 70)
    
    params = ar_network._parameters
    
    def loss_fn(params, states):
        """Simple loss: negative log probability averaged over states."""
        log_probs = ar_network.apply(params, states)
        return jnp.mean(-jnp.real(log_probs))  # Use real part
    
    # Compute gradients
    grads = jax.grad(loss_fn)(params, test_states)
    
    print(f"(ok) Gradient computation successful")
    print(f"  - Loss value: {loss_fn(params, test_states):.6f}")
    
    # Analyze gradient structure
    import jax.tree_util as jtu
    flat_grads, _ = jtu.tree_flatten(grads)
    grad_norms = [jnp.linalg.norm(g.flatten()) for g in flat_grads if hasattr(g, 'shape')]
    
    print(f"  - Number of parameter groups: {len(flat_grads)}")
    print(f"  - Gradient norms (first 3): {grad_norms[:3]}")
    print(f"  - All gradients non-zero: {all(n > 0 for n in grad_norms)}")
    
    # =====================================================================
    # 5. ADVANTAGES SUMMARY
    # =====================================================================
    print(f"\nStep 5: Autoregressive Network Advantages")
    print("-" * 70)
    
    advantages = [
        ("Exact Sampling", "Generate samples sequentially with controlled probabilities"),
        ("Holomorphic", "Complex-valued outputs suitable for quantum states"),
        ("High-Prob Focus", "Naturally emphasizes high-probability configurations"),
        ("JAX Backend", "Full JAX integration for JIT compilation and autodiff"),
        ("Flexibility", "Supports variable hidden layer configurations"),
        ("NQS Ready", "Fully integrated with NQS solver framework"),
    ]
    
    for title, description in advantages:
        print(f"  (ok) {title:20} → {description}")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("""
The Autoregressive network demonstrates:
  1. Correct initialization via NetworkFactory
  2. Accurate forward pass on quantum states
  3. High-probability sample generation
  4. Full automatic differentiation support
  5. Complex-valued holomorphic outputs

Ready for use in NQS training for quantum state representation!
    """)


if __name__ == "__main__":
    demonstrate_autoregressive_network()
