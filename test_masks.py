import jax.numpy as jnp
from QES.general_python.ml.net_impl.networks.net_autoregressive import create_masks

masks = create_masks(4, (16,), n_out_per_site=3)
for i, m in enumerate(masks):
    print(f"Mask {i} shape: {m.shape}")
