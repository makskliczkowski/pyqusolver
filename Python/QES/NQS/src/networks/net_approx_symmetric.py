r"""
Approximately Symmetric Ansatz Implementation

This module implements the "nonsymmetric χ -> symmetric Ω -> invariant \sigma" architecture pattern.
The idea is to have a flexible nonsymmetric block (chi) that learns a representation,
followed by a symmetrization block (Omega) that enforces physical invariances,
and finally an invariant nonlinearity (sigma) for the readout.

This architecture allows learning "unfattening" maps in the nonsymmetric block while
guaranteeing symmetry in the final wavefunction, as described in Kufel et al. (2025).
"""

import      jax
import      jax.numpy   as jnp
from        flax        import linen as nn
from        typing      import Any, Callable, Optional, Sequence, List

try:
    from QES.general_python.ml.net_impl.interface_net_flax     import FlaxInterface
    from QES.general_python.ml.net_impl.activation_functions   import get_activation
    from QES.general_python.ml.net_impl.utils.net_init_jax     import complex_he_init, real_he_init
except ImportError:
    raise ImportError("Required modules from general_python package are missing.")

# ----------------------------------------------------------------------

class ApproxSymmetricNet(nn.Module):
    """
    Flax module implementing the Approximately Symmetric Ansatz.

    Structure:
        x -> Chi(x) -> Omega(Chi(x)) -> Sigma(Omega(Chi(x)))

    Attributes:
        chi_features    : Sequence[int]
            Hidden layer sizes for the nonsymmetric block.
        symmetry_op     : Callable[[jnp.ndarray], jnp.ndarray]
            A callable that applies the symmetry operation (e.g., group averaging).
            Signature: (x: Array) -> Array.
            Should return invariant features or features averaged over the group.
        readout_act     : Callable[[jnp.ndarray], jnp.ndarray]
            The invariant nonlinearity (sigma). Usually log_cosh or similar.
        chi_act         : Callable[[jnp.ndarray], jnp.ndarray]
            Activation function for the chi block.
        dtype           : Any
            The dtype for computation.
    """
    chi_features    : Sequence[int]
    symmetry_op     : Callable[[jnp.ndarray], jnp.ndarray]
    readout_act     : Callable[[jnp.ndarray], jnp.ndarray]  = jnp.log           # default fallback
    chi_act         : Callable[[jnp.ndarray], jnp.ndarray]  = nn.relu           # for chi block
    dtype           : Any                                   = jnp.float32

    @nn.compact
    def __call__(self, x):

        # Determine initializers based on dtype
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            kernel_init = complex_he_init
        else:
            kernel_init = real_he_init

        # 1. Nonsymmetric Block (Chi)
        # ---------------------------
        # Learns an "unfattening" map.
        h = x.astype(self.dtype)

        # Simple MLP for Chi block
        for i, feat in enumerate(self.chi_features):
            h   = nn.Dense(feat, dtype=self.dtype, kernel_init=kernel_init, name=f'Chi_Dense_{i}')(h)
            h   = self.chi_act(h)

        # 2. Symmetric Block (Omega)
        # --------------------------
        # Enforces invariance via group averaging or similar operations.
        # h has shape (batch, features).
        sym_h   = self.symmetry_op(h)

        # 3. Readout (Sigma)
        # ------------------
        # Final projection to scalar and invariant nonlinearity.
        # If sym_h is already a scalar (or (batch, 1)), this is just a nonlinearity.
        # If sym_h is a vector (invariant features), we project it to a scalar first.

        out     = nn.Dense(1, dtype=self.dtype, kernel_init=kernel_init, name='Readout')(sym_h)

        # Apply invariant nonlinearity (Sigma)
        out     = self.readout_act(out)

        # Squeeze to scalar per sample if needed (batch, 1) -> (batch,)
        return out.squeeze(-1)

class AnsatzApproxSymmetric(FlaxInterface):
    """
    Interface for the Approximately Symmetric Ansatz.

    Parameters:
        chi_features : Sequence[int]
            Hidden layer sizes for the nonsymmetric block.
        symmetry_op : Callable or str
            The symmetry enforcing operation.
            - If Callable: (batch, features) -> (batch, features_out)
            - If 'mean': Averages over the last dimension (spatial).
            - If 'sum': Sums over the last dimension.
        readout_act : str or Callable
            Activation function for the readout (invariant nonlinearity). Default: 'log_cosh'.
        chi_act : str or Callable
            Activation function for the chi block. Default: 'relu'.
        input_shape : tuple
            Shape of the input configuration.
        dtype : Any
            Data type for computations (e.g. jnp.complex128).
    """

    def __init__(self,
                chi_features    : Sequence[int] = (64, 64),
                symmetry_op     : Optional[Any] = 'mean',
                readout_act     : Any           = 'log_cosh', # Common in NQS
                chi_act         : Any           = 'relu',
                input_shape     : tuple         = (10,),
                backend         : str           = 'jax',
                dtype           : Any           = jnp.complex128,
                seed            : int           = 42,
                **kwargs):

        # Resolve activations
        if isinstance(readout_act, str):
            readout_act, _      = get_activation(readout_act)

        if isinstance(chi_act, str):
            chi_act, _          = get_activation(chi_act)

        # Resolve symmetry op
        if isinstance(symmetry_op, str):
            if symmetry_op in ['mean', 'sum']:
                symmetry_op = GroupAveragingOp(mode=symmetry_op)
            elif symmetry_op == 'identity':
                def identity_sym(x): return x
                symmetry_op = identity_sym
            else:
                raise ValueError(f"Unknown symmetry_op string: {symmetry_op}")
        elif symmetry_op is None:
             def identity_sym(x): return x
             symmetry_op = identity_sym

        # Ensure symmetry_op is callable
        if not callable(symmetry_op):
             raise ValueError("symmetry_op must be a callable or valid string descriptor.")

        net_kwargs = {
            'chi_features'      : chi_features,
            'symmetry_op'       : symmetry_op,
            'readout_act'       : readout_act,
            'chi_act'           : chi_act,
            'dtype'             : dtype
        }

        super().__init__(
            net_module          = ApproxSymmetricNet,
            net_kwargs          = net_kwargs,
            input_shape         = input_shape,
            backend             = backend,
            dtype               = dtype,
            seed                = seed,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"AnsatzApproxSymmetric(chi_features={self._net_kwargs_in['chi_features']}, dtype={self.dtype})"

# ----------------------------------------------------------------------
# Symmetry Operations Factories
# ----------------------------------------------------------------------

class GroupAveragingOp:
    """
    Applies group averaging in feature space.
    Assumes that the input x has shape (batch, features) and that
    the group action permutes or transforms these features.
    """
    def __init__(self, mode='mean'):
        self.mode = mode

    def __call__(self, x):
        # Assumes x is (batch, ..., sites)
        if self.mode == 'mean':
            return jnp.mean(x, axis=-1, keepdims=True)
        elif self.mode == 'sum':
            return jnp.sum(x, axis=-1, keepdims=True)
        return x

def make_translation_symmetry_op(mode='mean'):
    op = GroupAveragingOp(mode=mode)
    return op

def make_permutation_symmetry_op(indices_list: List[List[int]]):
    indices_array = jnp.array(indices_list) # (n_sym, n_features)

    def op(x):
        def apply_single_perm(perm):
            return x[:, perm]
        all_perms = jax.vmap(apply_single_perm)(indices_array)
        return jnp.mean(all_perms, axis=0)

    return op

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------