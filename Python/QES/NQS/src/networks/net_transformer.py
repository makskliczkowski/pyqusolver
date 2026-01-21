"""
general_python.ml.net_impl.networks.net_transformer
===================================================

Vision Transformer (ViT) Ansatz for Quantum States.

Refactored for VMC stability:
- Removes LayerNorm and CLS token.
- Enforces log_cosh.
- Uses Global Sum Pooling over patches.
- Removes Dropout.

"""

import jax
import jax.numpy    as jnp
import flax.linen   as nn
from typing         import Sequence, Any, Optional, Tuple, Callable

try:
    from QES.general_python.ml.net_impl.interface_net_flax import FlaxInterface
    from QES.general_python.ml.net_impl.utils.net_init_jax import cplx_variance_scaling
    from QES.general_python.ml.net_impl.activation_functions   import get_activation_jnp
    JAX_AVAILABLE = True
except ImportError:
    raise ImportError("Transformer requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Transformer Components
# ----------------------------------------------------------------------

class MLPBlock(nn.Module):
    hidden_dim  : int
    out_dim     : int
    dtype       : Any

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        # Replaced gelu with log_cosh
        log_cosh, _ = get_activation_jnp('log_cosh')
        x = log_cosh(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x

class EncoderBlock(nn.Module):
    num_heads   : int
    hidden_dim  : int
    mlp_dim     : int
    dtype       : Any

    @nn.compact
    def __call__(self, x):
        # Attention Block
        # REMOVED LayerNorm
        y = nn.MultiHeadAttention(
            num_heads       =   self.num_heads,
            dtype           =   self.dtype,
            kernel_init     =   nn.initializers.xavier_uniform(),
            deterministic   =   True
        )(x, x) # Note: x is passed as both query and key/value
        x = x + y

        # MLP Block
        # REMOVED LayerNorm
        y = MLPBlock(hidden_dim=self.mlp_dim, out_dim=self.hidden_dim, dtype=self.dtype)(x)
        x = x + y

        return x

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxTransformer(nn.Module):
    n_sites         : int
    patch_size      : int
    embed_dim       : int
    depth           : int
    num_heads       : int
    mlp_ratio       : float = 2.0
    dtype           : Any   = jnp.complex128

    def setup(self):
        self.num_patches = self.n_sites // self.patch_size

        # Ensure n_sites is divisible by patch_size
        if self.n_sites % self.patch_size != 0:
            raise ValueError(f"n_sites ({self.n_sites}) must be divisible by patch_size ({self.patch_size})")

        self.pos_embedding  = self.param('pos_embedding',
                                        nn.initializers.normal(stddev=0.02),
                                        (1, self.num_patches, self.embed_dim), # Removed +1 for CLS
                                        self.dtype)

        # REMOVED CLS token

        self.blocks         = [
                                EncoderBlock(
                                    num_heads=self.num_heads,
                                    hidden_dim=self.embed_dim,
                                    mlp_dim=int(self.embed_dim * self.mlp_ratio),
                                    dtype=self.dtype
                                ) for _ in range(self.depth)
                            ]

        # REMOVED Final LayerNorm
        # self.norm = ...

        # REMOVED Head
        # self.head = ...

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, n_sites)
        batch_size          = x.shape[0]

        # 1. Patch Embedding
        # Reshape to (batch, num_patches, patch_size)
        x_patches           = x.reshape((batch_size, self.num_patches, self.patch_size))

        # Linear projection of flattened patches
        x                   = nn.Dense(self.embed_dim, dtype=self.dtype)(x_patches) # (batch, num_patches, embed_dim)

        # REMOVED CLS token concatenation

        # 3. Add Position Embedding
        x                   = x + self.pos_embedding

        # 4. Transformer Encoder
        for block in self.blocks:
            x               = block(x)

        # REMOVED Final Norm

        # 5. Global Sum Pooling (over patches and embedding dim)
        # x: (Batch, Num_Patches, Embed_Dim)
        out                 = jnp.sum(x, axis=(1, 2))

        return out # (batch,)

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class Transformer(FlaxInterface):
    """
    Vision Transformer (ViT) Ansatz Interface.
    """
    def __init__(self,
                input_shape     : tuple,
                patch_size      : int   = 4,
                embed_dim       : int   = 32,
                depth           : int   = 2,
                num_heads       : int   = 4,
                mlp_ratio       : float = 2.0,
                dtype           : Any   = jnp.complex128,
                seed            : int   = 0,
                backend         : str   = 'jax',
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("Transformer requires JAX.")

        n_sites = input_shape[0] if len(input_shape) == 1 else input_shape[0] * input_shape[1]

        net_kwargs = {
            'n_sites'       : n_sites,
            'patch_size'    : patch_size,
            'embed_dim'     : embed_dim,
            'depth'         : depth,
            'num_heads'     : num_heads,
            'mlp_ratio'     : mlp_ratio,
            'dtype'         : dtype
        }

        super().__init__(
            net_module  =   _FlaxTransformer,
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   backend,
            dtype       =   dtype,
            seed        =   seed,
            **kwargs
        )
        self._name = 'transformer'

    def __repr__(self) -> str:
        mod = self._flax_module
        return f"Transformer(n={mod.n_sites}, patch={mod.patch_size}, dim={mod.embed_dim}, depth={mod.depth})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
