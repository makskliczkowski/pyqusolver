"""
NQS-facing wrappers around reusable backbone networks.

These wrappers own the translation from NQS state conventions to generic
callable preprocessing hooks used by the backbone implementations in
``QES.general_python.ml.net_impl.networks``.

-----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-04-07
License         : MIT
-----------------------------------------------------------
"""

from functools import partial
from typing import Optional

import jax.numpy as jnp

try:
    from QES.general_python.ml.net_impl.utils.net_wrapper_utils     import configure_nqs_metadata
    from QES.general_python.ml.net_impl.networks.net_cnn            import CNN as _CNN
    from QES.general_python.ml.net_impl.networks.net_gcnn           import GCNN as _GCNN
    from QES.general_python.ml.net_impl.networks.net_mlp            import MLP as _MLP
    from QES.general_python.ml.net_impl.networks.net_rbm            import RBM as _RBM
    from QES.general_python.ml.net_impl.networks.net_res            import ResNet as _ResNet
    from QES.general_python.ml.net_impl.networks.net_transformer    import Transformer as _Transformer
    from QES.general_python.ml.net_impl.utils.net_state_repr_jax    import map_state_to_pm1
    from QES.general_python.common.binary                           import BACKEND_DEF_SPIN, BACKEND_REPR
except ImportError as e:
    raise ImportError("Failed to import necessary modules for NQS ansatz wrappers. Ensure JAX/Flax and general_python package are correctly installed.") from e

# --------------------------------------------------------------------------------

def _resolve_input_convention(*, input_is_spin: bool = BACKEND_DEF_SPIN, input_value: float = BACKEND_REPR) -> tuple[bool, float]:
    return bool(input_is_spin), float(input_value)

def _make_spin_adapter(*, input_is_spin: bool, input_value: float):
    return partial(map_state_to_pm1, input_is_spin=bool(input_is_spin), input_value=float(input_value),)

def _flip_local_values(values, *, input_is_spin: bool, input_value: float):
    arr = jnp.asarray(values)
    if bool(input_is_spin):
        return -arr
    return jnp.asarray(float(input_value), dtype=arr.dtype) - arr

# --------------------------------------------------------------------------------

class RBM(_RBM):
    """
    NQS-facing RBM wrapper.

    Accepts NQS state-convention kwargs and translates them to the
    generic RBM backbone constructor.
    """

    def __init__(self, *args,
        input_activation        = None,
        proposal_update         = None,
        input_is_spin: bool     = BACKEND_DEF_SPIN,
        input_value: float      = BACKEND_REPR,
        **kwargs,
    ):
        resolved_input_is_spin, resolved_input_value = _resolve_input_convention(input_is_spin=input_is_spin, input_value=input_value)

        if input_activation is True:
            input_activation = partial(map_state_to_pm1, input_is_spin=resolved_input_is_spin, input_value=resolved_input_value,)
            
        if proposal_update is None:
            proposal_update = partial(_flip_local_values, input_is_spin=resolved_input_is_spin, input_value=resolved_input_value,)

        super().__init__(*args, input_activation=input_activation, proposal_update=proposal_update, **kwargs,)
        
        # Configure NQS metadata for this RBM wrapper (family, native representation, etc.)
        configure_nqs_metadata(self, family="rbm", native_representation="spin_pm" if resolved_input_is_spin else "binary_01", supports_fast_updates=True)

# --------------------------------------------------------------------------------

class CNN(_CNN):
    """NQS-facing CNN wrapper with explicit state-convention parameters."""

    def __init__(self, *args, input_is_spin: bool = BACKEND_DEF_SPIN, input_value: float = BACKEND_REPR, input_adapter=None, **kwargs):
        resolved_input_is_spin, resolved_input_value = _resolve_input_convention(input_is_spin=input_is_spin, input_value=input_value)
        
        if input_adapter is None and (not resolved_input_is_spin):
            input_adapter = _make_spin_adapter(input_is_spin=resolved_input_is_spin, input_value=resolved_input_value)
        
        super().__init__(*args, input_adapter=input_adapter, **kwargs)
        
        # Configure NQS metadata for this CNN wrapper (family, native representation, etc.)
        configure_nqs_metadata(self, family="cnn", native_representation="spin_pm" if resolved_input_is_spin else "binary_01")

# --------------------------------------------------------------------------------

class MLP(_MLP):
    """NQS-facing MLP wrapper with explicit state-convention parameters."""

    def __init__(self, *args, input_is_spin: bool = BACKEND_DEF_SPIN, input_value: float = BACKEND_REPR, input_adapter=None, **kwargs):
        ''' MLP wrapper that accepts NQS state convention parameters and translates them to the MLP backbone. '''
        
        resolved_input_is_spin, resolved_input_value = _resolve_input_convention(input_is_spin=input_is_spin, input_value=input_value)
        if input_adapter is None and (not resolved_input_is_spin):
            input_adapter = _make_spin_adapter(input_is_spin=resolved_input_is_spin, input_value=resolved_input_value)
            
        super().__init__(*args, input_adapter=input_adapter, **kwargs)
        
        configure_nqs_metadata(self, family="mlp", native_representation="spin_pm" if resolved_input_is_spin else "binary_01")
        
# --------------------------------------------------------------------------------

class GCNN(_GCNN):
    """NQS-facing GCNN wrapper with explicit state-convention parameters."""

    def __init__(
        self,
        *args,
        input_is_spin: bool = BACKEND_DEF_SPIN,
        input_value: float = BACKEND_REPR,
        input_adapter=None,
        **kwargs,
    ):
        resolved_input_is_spin, resolved_input_value = _resolve_input_convention(
            input_is_spin=input_is_spin,
            input_value=input_value,
        )
        if input_adapter is None and (not resolved_input_is_spin):
            input_adapter = _make_spin_adapter(
                input_is_spin=resolved_input_is_spin,
                input_value=resolved_input_value,
            )
        super().__init__(
            *args,
            input_adapter=input_adapter,
            **kwargs,
        )
        configure_nqs_metadata(
            self,
            family="gcnn",
            native_representation="spin_pm" if resolved_input_is_spin else "binary_01",
        )

# --------------------------------------------------------------------------------

class ResNet(_ResNet):
    """NQS-facing ResNet wrapper with explicit state-convention parameters."""

    def __init__(
        self,
        *args,
        input_is_spin: bool = BACKEND_DEF_SPIN,
        input_value: float = BACKEND_REPR,
        input_adapter=None,
        **kwargs,
    ):
        resolved_input_is_spin, resolved_input_value = _resolve_input_convention(
            input_is_spin=input_is_spin,
            input_value=input_value,
        )
        if input_adapter is None and (not resolved_input_is_spin):
            input_adapter = _make_spin_adapter(
                input_is_spin=resolved_input_is_spin,
                input_value=resolved_input_value,
            )
        super().__init__(
            *args,
            input_adapter=input_adapter,
            **kwargs,
        )
        configure_nqs_metadata(
            self,
            family="resnet",
            native_representation="spin_pm" if resolved_input_is_spin else "binary_01",
        )

# --------------------------------------------------------------------------------

class Transformer(_Transformer):
    """NQS-facing transformer wrapper."""

# --------------------------------------------------------------------------------
#! FINAL
# --------------------------------------------------------------------------------

__all__ = ["RBM", "CNN", "ResNet", "MLP", "GCNN", "Transformer"]

# --------------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------------