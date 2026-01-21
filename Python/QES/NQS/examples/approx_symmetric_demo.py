"""
Approximately Symmetric and Stacked Ansatz Demo

This script demonstrates how to use the new Approximately Symmetric and
Stacked Symmetry-Improved ansätze in the NQS module.

We show:
1. Creating an ApproxSymmetricNet with mean pooling.
2. Creating a StackedNet with convolution, symmetry, and readout.
3. Checking their output shapes and properties.
"""

import jax
import jax.numpy as jnp
import numpy as np

from QES.NQS.src.networks.net_approx_symmetric import AnsatzApproxSymmetric
from QES.NQS.src.networks.net_stacked import AnsatzStacked

def demo_approx_symmetric():
    print("\n=== Approximately Symmetric Ansatz Demo ===")

    L = 10
    print(f"System size L={L}")

    # Define a simple "mean" symmetry operation
    # In practice, this enforces translational invariance if the features are spatial
    sym_op = 'mean'

    net = AnsatzApproxSymmetric(
        chi_features=[32, 32],      # Nonsymmetric block layers
        symmetry_op=sym_op,         # Symmetry operation
        readout_act='log_cosh',     # Invariant nonlinearity
        input_shape=(L,),
        dtype=jnp.complex128
    )

    print(f"Created Ansatz: {net}")

    # Initialize parameters
    params = net.init()
    print(f"Number of parameters: {net.nparams}")

    # Forward pass
    x = jnp.ones((5, L)) # Batch of 5 states
    log_psi = net(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {log_psi.shape}")
    print(f"Output dtype: {log_psi.dtype}")
    print(f"Sample output: {log_psi[0]}")

def demo_stacked_symmetry():
    print("\n=== Stacked Symmetry-Improved Ansatz Demo ===")

    L = 10
    print(f"System size L={L}")

    # Define a custom symmetry stack
    # 1. Conv1D layer (local filters)
    # 2. Mean pooling (symmetry)
    # 3. Readout (log_cosh)

    stack_config = [
        {
            'type': 'Conv',
            'args': {'features': 8, 'kernel_size': 3, 'act': 'relu', 'padding': 'CIRCULAR'}
        },
        {
            'type': 'SymmetryGroup',
            'args': {'mode': 'mean'}
        },
        {
            'type': 'Readout',
            'args': {'act': 'log_cosh'}
        }
    ]

    net = AnsatzStacked(
        stack_config=stack_config,
        input_shape=(L,),
        dtype=jnp.complex128
    )

    print(f"Created Stacked Ansatz: {net}")

    # Initialize parameters
    net.init()
    print(f"Number of parameters: {net.nparams}")

    # Forward pass
    x = jax.random.normal(jax.random.PRNGKey(0), (3, L))
    log_psi = net(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {log_psi.shape}")
    print(f"Output dtype: {log_psi.dtype}")
    print(f"Sample output: {log_psi[0]}")

if __name__ == "__main__":
    demo_approx_symmetric()
    demo_stacked_symmetry()
