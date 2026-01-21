"""
Stacked Symmetry-Improved Ansatz Implementation

This module implements a composable stacked ansatz:
Ansatz = (Weakly-Breaking Preblock) o (Exact-Symmetry Block) o (Readout)

It provides a configuration-driven way to define these stacks, allowing for
pluggable symmetry groups and architectures.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Optional, Sequence, List, Dict, Union

try:
    from QES.general_python.ml.net_impl.interface_net_flax import FlaxInterface
    from QES.general_python.ml.net_impl.activation_functions import get_activation
    from QES.general_python.ml.net_impl.utils.net_init_jax import complex_he_init, real_he_init
    from .net_approx_symmetric import GroupAveragingOp, make_permutation_symmetry_op
except ImportError:
    raise ImportError("Required modules from general_python package are missing.")

# ----------------------------------------------------------------------
# Block Definitions
# ----------------------------------------------------------------------

class DenseBlock(nn.Module):
    features: int
    act: Callable = nn.relu
    dtype: Any = jnp.complex128
    kernel_init: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        init = self.kernel_init
        if init is None:
            if jnp.issubdtype(self.dtype, jnp.complexfloating):
                init = complex_he_init
            else:
                init = real_he_init

        x = nn.Dense(self.features, dtype=self.dtype, kernel_init=init)(x)
        return self.act(x)

class ConvBlock(nn.Module):
    """
    1D Convolutional Block.
    """
    features: int
    kernel_size: int = 3
    strides: int = 1
    padding: str = 'CIRCULAR' # 'SAME', 'VALID', 'CIRCULAR'
    act: Callable = nn.relu
    dtype: Any = jnp.complex128
    kernel_init: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        init = self.kernel_init
        if init is None:
             if jnp.issubdtype(self.dtype, jnp.complexfloating):
                init = complex_he_init
             else:
                init = real_he_init

        # x shape: (batch, spatial_dim) -> reshape to (batch, spatial_dim, 1) if needed

        is_flat = x.ndim == 2
        if is_flat:
            x_in = jnp.expand_dims(x, axis=-1)
        else:
            x_in = x

        # Handle CIRCULAR padding manually if needed, or rely on Flax if supported (Flax uses SAME/VALID)
        if self.padding == 'CIRCULAR':
             pad_w = (self.kernel_size - 1) // 2
             if pad_w > 0:
                 x_in = jnp.concatenate([x_in[:, -pad_w:, :], x_in, x_in[:, :pad_w, :]], axis=1)
             pad_mode = 'VALID'
        else:
             pad_mode = self.padding

        x_out = nn.Conv(features=self.features, kernel_size=(self.kernel_size,),
                        strides=(self.strides,), padding=pad_mode,
                        dtype=self.dtype, kernel_init=init)(x_in)

        x_out = self.act(x_out)
        return x_out

class IdentityBlock(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class FlattenBlock(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))

class ReadoutBlock(nn.Module):
    act: Callable = jnp.log
    output_dim: int = 1
    dtype: Any = jnp.complex128
    kernel_init: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        # Flatten everything except batch
        x_flat = x.reshape((x.shape[0], -1))

        init = self.kernel_init
        if init is None:
             if jnp.issubdtype(self.dtype, jnp.complexfloating):
                init = complex_he_init
             else:
                init = real_he_init

        x_out = nn.Dense(self.output_dim, dtype=self.dtype, kernel_init=init)(x_flat)
        x_out = self.act(x_out)
        return x_out.squeeze(-1)

# ----------------------------------------------------------------------
# The Stack Module
# ----------------------------------------------------------------------

class StackedNet(nn.Module):
    """
    A Flax module that executes a sequence of configured blocks.
    """
    blocks_config: List[Dict[str, Any]]
    dtype: Any = jnp.complex128

    @nn.compact
    def __call__(self, x):
        h = x.astype(self.dtype)

        for i, config in enumerate(self.blocks_config):
            block_type = config.get('type')
            # Ensure block_args is a mutable dict
            block_args = dict(config.get('args', {}))

            # Instantiate Block based on type or callable
            if callable(block_type):
                # If block_type is a class or factory passed directly
                h = block_type(**block_args)(h)

            elif block_type == 'Dense':
                act_str = block_args.pop('act', 'relu')
                act_fn, _ = get_activation(act_str)
                features = block_args.pop('features', 64)
                h = DenseBlock(features=features, act=act_fn, dtype=self.dtype, **block_args)(h)

            elif block_type == 'Conv':
                act_str = block_args.pop('act', 'relu')
                act_fn, _ = get_activation(act_str)
                features = block_args.pop('features', 8)
                h = ConvBlock(features=features, act=act_fn, dtype=self.dtype, **block_args)(h)

            elif block_type == 'Identity':
                h = IdentityBlock()(h)

            elif block_type == 'Flatten':
                h = FlattenBlock()(h)

            elif block_type == 'Readout':
                act_str = block_args.pop('act', 'log_cosh')
                act_fn, _ = get_activation(act_str)
                h = ReadoutBlock(act=act_fn, dtype=self.dtype, **block_args)(h)

            elif block_type == 'SymmetryGroup':
                # Use provided symmetry operation or factory
                op_callable = block_args.get('op')

                if op_callable is not None and callable(op_callable):
                    h = op_callable(h)
                else:
                    # Fallback to simple modes if no callable provided
                    mode = block_args.get('mode', 'identity')
                    indices = block_args.get('indices')

                    if indices is not None:
                        # Use permutation symmetry
                        sym_op = make_permutation_symmetry_op(indices)
                        h = sym_op(h)
                    elif mode in ['mean', 'sum']:
                        # Use simple averaging
                        sym_op = GroupAveragingOp(mode=mode)
                        h = sym_op(h)
                    else:
                        pass # Identity

            else:
                raise ValueError(f"Unknown block type: {block_type}")

        return h

# ----------------------------------------------------------------------
# The Interface
# ----------------------------------------------------------------------

class AnsatzStacked(FlaxInterface):
    """
    Interface for the Stacked Symmetry-Improved Ansatz.
    """

    def __init__(self,
                 stack_config: List[Dict[str, Any]],
                 input_shape: tuple = (10,),
                 backend: str = 'jax',
                 dtype: Any = jnp.complex128,
                 seed: int = 42,
                 **kwargs):

        # Validate config basic structure
        if not isinstance(stack_config, list):
            raise ValueError("stack_config must be a list of dictionaries.")

        net_kwargs = {
            'blocks_config': stack_config,
            'dtype': dtype
        }

        super().__init__(
            net_module=StackedNet,
            net_kwargs=net_kwargs,
            input_shape=input_shape,
            backend=backend,
            dtype=dtype,
            seed=seed,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"AnsatzStacked(blocks={len(self._net_kwargs_in['blocks_config'])}, dtype={self.dtype})"
