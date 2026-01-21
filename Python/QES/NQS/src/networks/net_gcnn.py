"""
general_python.ml.net_impl.networks.net_gcnn
==============================================

Graph Convolutional Neural Network (GCNN) for Quantum States.

Refactored for VMC stability:
- Removes LayerNorm.
- Enforces log_cosh.
- Uses Global Sum Pooling instead of Dense Readout.
- Ensures Symmetry Equivariance.

"""

import  numpy   as np
import  math
from    typing  import Sequence, Callable, Optional, Any, Union, List, Tuple

try:
    import jax
    import jax.numpy    as jnp
    import flax.linen   as nn
    from QES.general_python.ml.net_impl.interface_net_flax     import FlaxInterface
    from QES.general_python.ml.net_impl.utils.net_init_jax     import cplx_variance_scaling, lecun_normal
    from QES.general_python.ml.net_impl.activation_functions   import get_activation_jnp
    JAX_AVAILABLE       = True
except ImportError:
    raise ImportError("GCNN requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Graph Convolution Layer
# ----------------------------------------------------------------------

class GraphConv(nn.Module):
    """
    Graph Convolution Layer:
    out_i = W_self * x_i + W_neigh * sum_{j in N(i)} x_j + b
    """
    features    : int
    dtype       : Any
    param_dtype : Any
    use_bias    : bool
    kernel_init : Callable
    normalized  : bool      = True

    @nn.compact
    def __call__(self, x, adj_mat):
        # 1. Self interaction
        self_feat = nn.Dense(
            features    = self.features,
            use_bias    = False,
            dtype       = self.dtype,
            param_dtype = self.param_dtype,
            kernel_init = self.kernel_init,
            name        = "dense_self"
        )(x)

        # 1.5 Degree normalization (optional)
        if self.normalized:
            deg             = jnp.sum(adj_mat, axis=1, keepdims=True)  # (N, 1)
            deg             = jnp.maximum(deg, 1.0)
            deg_inv_sqrt    = 1.0 / jnp.sqrt(deg)
            adj_norm        = (deg_inv_sqrt.T * adj_mat) * deg_inv_sqrt
        else:
            adj_norm        = adj_mat

        # 2. Neighbor interaction
        x_neigh         = jnp.einsum('nm,bmc->bnc', adj_norm, x)

        neigh_feat      = nn.Dense(
                            features    = self.features,
                            use_bias    = False,
                            dtype       = self.dtype,
                            param_dtype = self.param_dtype,
                            kernel_init = self.kernel_init,
                            name        = "dense_neigh"
                        )(x_neigh)

        # 3. Combine
        out = self_feat + neigh_feat

        if self.use_bias:
            bias    = self.param('bias', nn.initializers.zeros, (self.features,), self.param_dtype)
            out     = out + bias

        return out

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxGCNN(nn.Module):
    adj_matrix      : jnp.ndarray
    features        : Sequence[int]
    activations     : Sequence[Callable]
    output_dim      : int
    use_bias        : bool                  = True
    dtype           : Any                   = jnp.complex128
    param_dtype     : Any                   = jnp.complex128
    split_complex   : bool                  = False
    input_trans     : Optional[Callable]    = None
    use_sum_pool    : bool                  = True

    def setup(self):
        # Dtype logic
        if self.split_complex:
            if jnp.issubdtype(self.dtype, jnp.complexfloating):
                c_dtype = jnp.float32 if self.dtype == jnp.complex64 else jnp.float64
            else:
                c_dtype = self.dtype
            if jnp.issubdtype(self.param_dtype, jnp.complexfloating):
                p_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
            else:
                p_dtype = self.param_dtype
        else:
            c_dtype = self.dtype
            p_dtype = self.param_dtype

        if jnp.issubdtype(p_dtype, jnp.complexfloating):
            k_init = cplx_variance_scaling(1.0, 'fan_in', 'normal', p_dtype)
        else:
            k_init = nn.initializers.lecun_normal()

        # Build Graph Layers
        self.gconv_layers   = [
            GraphConv(
                features    = feat,
                use_bias    = self.use_bias,
                dtype       = c_dtype,
                param_dtype = p_dtype,
                kernel_init = k_init,
                name        = f"GConv_{i}"
            ) for i, feat in enumerate(self.features)
        ]

        # REMOVED LayerNorms
        # self.layer_norms = ...

        # REMOVED Dense Readout
        # self.dense_out = ...

    @nn.compact
    def __call__(self, x):
        if x.ndim == 1:
            x = x[jnp.newaxis, :, jnp.newaxis]
        elif x.ndim == 2:
            x = x[:, :, jnp.newaxis]
        elif x.ndim == 3:
            pass
        else:
            raise ValueError(f"GCNN expected x ndim in {1,2,3}, got {x.ndim} with shape {x.shape}")

        # 1. Preprocessing
        if self.split_complex:
            x = x.real if jnp.iscomplexobj(x) else x

        dtype = self.gconv_layers[0].dtype
        x = x.astype(dtype)

        if self.input_trans:
            x = self.input_trans(x)

        adj = self.adj_matrix.astype(dtype)

        for i, (layer, act) in enumerate(zip(self.gconv_layers, self.activations)):
            x = layer(x, adj)
            # REMOVED LayerNorm
            x = act(x)

        # 3. Pooling (Global Sum Pooling)
        # x: (Batch, N, Feat)
        # Sum over sites and features to get extensive quantity
        if self.use_sum_pool:
            # We want an extensive sum -> jnp.sum, not jnp.mean
            # The prompt says: "The final lnψ(s) must be an extensive sum of local hidden contributions."
            # Original code used jnp.mean(x, axis=1) then Dense.
            # Here we sum over sites (axis=1) AND features (axis=2).
            x = jnp.sum(x, axis=(1, 2))
        else:
            # Fallback
            x = x.reshape((x.shape[0], -1))

        # 5. Combine
        if self.split_complex:
            # If backbone was real, x is real.
            pass

        return x

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class GCNN(FlaxInterface):
    """
    Graph Convolutional Neural Network Interface.
    """
    def __init__(self,
                input_shape    : tuple,
                *,
                graph_edges    : Optional[List[Tuple[int, int]]]    = None,
                adj_matrix     : Optional[np.ndarray]               = None,
                features       : Sequence[int]                      = (16, 32),
                activations    : Union[str, Sequence]               = 'log_cosh', # Default changed
                output_shape   : tuple                              = (1,),
                use_bias       : bool                               = True,
                use_sum_pool   : bool                               = True,
                split_complex  : bool                               = False,
                transform_input: bool                               = False,
                dtype          : Any                                = jnp.complex128,
                param_dtype    : Optional[Any]                      = None,
                seed           : int                                = 0,
                backend        : str                                = 'jax',
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("GCNN requires JAX.")

        # Build Adjacency Matrix
        n_sites = input_shape[0]
        if adj_matrix is not None:
            if adj_matrix.shape != (n_sites, n_sites):
                raise ValueError(f"Provided adjacency matrix shape {adj_matrix.shape} does not match number of sites {n_sites}.")
            adj = adj_matrix.astype(np.float32)
        else:
            if graph_edges is None:
                # Fallback to chain or raise?
                # Assume chain for testing if nothing provided, or require edges
                # For safety, let's just make zeros if no edges, but usually edges are needed.
                adj = np.zeros((n_sites, n_sites), dtype=np.float32)
            else:
                adj = np.zeros((n_sites, n_sites), dtype=np.float32)
                for i, j in graph_edges:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0 # Undirected

        if isinstance(activations, str):
            act_fn, _   = get_activation_jnp(activations)
            acts        = (act_fn,) * len(features)
        else:
            acts        = tuple(get_activation_jnp(a)[0] for a in activations)

        net_kwargs      = {
                            'adj_matrix'    : jnp.array(adj),
                            'features'      : features,
                            'activations'   : acts,
                            'output_dim'    : int(np.prod(output_shape)),
                            'use_bias'      : use_bias,
                            'dtype'         : dtype,
                            'param_dtype'   : param_dtype if param_dtype else dtype,
                            'split_complex' : split_complex,
                            'input_trans'   : (lambda x: 2*x) if transform_input else None, # Changed 2x-1 to 2x
                            'use_sum_pool'  : use_sum_pool
                        }

        super().__init__(
            net_module  = _FlaxGCNN,
            net_kwargs  = net_kwargs,
            input_shape = input_shape,
            backend     = backend,
            dtype       = dtype,
            seed        = seed,
            **kwargs
        )
        self._out_shape = output_shape
        self._name      = 'gcnn'

    def __call__(self, x):
        out = super().__call__(x)
        if self._out_shape == (1,):
            return out.reshape(-1)
        return out.reshape((-1,) + self._out_shape)

    def __repr__(self) -> str:
        return f"GCNN(sites={self.input_dim}, features={self._flax_module.features})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
