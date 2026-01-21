
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import sys
import os

# Add path to allow imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Python'))

try:
    from Python.QES.general_python.ml.net_impl.networks.net_cnn import CNN
    from Python.QES.general_python.ml.net_impl.networks.net_gcnn import GCNN
    from Python.QES.general_python.ml.net_impl.networks.net_transformer import Transformer
except ImportError:
    # Try alternate path if running from root
    from QES.general_python.ml.net_impl.networks.net_cnn import CNN
    from QES.general_python.ml.net_impl.networks.net_gcnn import GCNN
    from QES.general_python.ml.net_impl.networks.net_transformer import Transformer

def test_cnn():
    print("\nTesting CNN...")
    input_shape = (64,)
    # Note: FlaxInterface.init only takes 'key'. It handles dummy input internally.
    net = CNN(input_shape=input_shape, reshape_dims=(8,8), features=(8,), kernel_sizes=(3,), sum_pooling=True)

    # Initialize
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 64)) * 0.5 # Input 0.5

    # Correct Init Call for FlaxInterface (it overrides init(key))
    params = net.init(key)

    # Check output
    # FlaxInterface.apply(params, x)
    out = net.apply(params, x)
    print(f"Output shape: {out.shape}, Value: {out}")

    # Check for Dense output layer removal
    param_keys = params.keys()
    print(f"Param keys: {param_keys}")

    forbidden = ['dense_out', 'Dense_0', 'BatchNorm', 'LayerNorm']
    for k in param_keys:
        if k in forbidden:
            raise ValueError(f"Forbidden layer {k} found in CNN params!")

    if len(out.shape) != 1: # (Batch,)
         print(f"Warning: Output shape {out.shape} is not scalar per batch!")

def test_gcnn():
    print("\nTesting GCNN...")
    input_shape = (64,)
    edges = [(i, (i+1)%64) for i in range(64)]
    net = GCNN(input_shape=input_shape, graph_edges=edges, features=(8,), use_sum_pool=True)

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 64)) * 0.5
    params = net.init(key)

    out = net.apply(params, x)
    print(f"Output shape: {out.shape}, Value: {out}")

    param_keys = params.keys()
    print(f"Param keys: {param_keys}")

    forbidden = ['Readout', 'LN', 'LayerNorm']
    for k in param_keys:
        if any(f in k for f in forbidden):
             raise ValueError(f"Forbidden layer {k} found in GCNN params!")

def test_transformer():
    print("\nTesting Transformer...")
    input_shape = (64,)
    net = Transformer(input_shape=input_shape, patch_size=4, embed_dim=8, depth=1)

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 64)) * 0.5
    params = net.init(key)

    out = net.apply(params, x)
    print(f"Output shape: {out.shape}, Value: {out}")

    param_keys = params.keys()
    print(f"Param keys: {param_keys}")

    forbidden = ['head', 'cls_token', 'norm']
    for k in param_keys:
        if k in forbidden:
             raise ValueError(f"Forbidden layer {k} found in Transformer params!")

if __name__ == "__main__":
    try:
        test_cnn()
        test_gcnn()
        test_transformer()
        print("\nAll networks verified successfully.")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
