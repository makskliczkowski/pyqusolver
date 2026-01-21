"""
Precision utilities for NQS mixed-precision policies.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


@dataclass(frozen=True)
class NQSPrecisionPolicy:
    """
    Precision policy for sampling and numerically sensitive paths.
    """
    state_dtype         : Any
    prob_dtype          : Any
    accum_real_dtype    : Any
    accum_complex_dtype : Any


def _jax_x64_enabled(is_jax: bool) -> bool:
    if not is_jax or jax is None:
        return False
    try:
        return bool(jax.config.read("jax_enable_x64"))
    except Exception:
        return False


def _is_double_precision_dtype(dtype) -> bool:
    dt = np.dtype(dtype)
    return dt == np.dtype("float64") or dt == np.dtype("complex128")


def _real_dtype_for(dtype, *, is_jax: bool):
    dt = np.dtype(dtype)
    if dt == np.dtype("float64") or dt == np.dtype("complex128"):
        return jnp.float64 if is_jax else np.float64
    return jnp.float32 if is_jax else np.float32


def _complex_dtype_for(dtype, *, is_jax: bool):
    dt = np.dtype(dtype)
    if dt == np.dtype("float64") or dt == np.dtype("complex128"):
        return jnp.complex128 if is_jax else np.complex128
    return jnp.complex64 if is_jax else np.complex64


def _default_accum_dtypes(net_dtype, *, prefer_x64: bool, is_jax: bool):
    if _is_double_precision_dtype(net_dtype):
        return _real_dtype_for(net_dtype, is_jax=is_jax), _complex_dtype_for(net_dtype, is_jax=is_jax)

    use_x64 = prefer_x64 and (not is_jax or _jax_x64_enabled(is_jax))
    if use_x64:
        return (
            jnp.float64 if is_jax else np.float64,
            jnp.complex128 if is_jax else np.complex128,
        )
    return (
        jnp.float32 if is_jax else np.float32,
        jnp.complex64 if is_jax else np.complex64,
    )


def _resolve_accum_dtypes(
    net_dtype,
    *,
    prefer_x64: bool,
    is_jax: bool,
    accum_dtype=None,
    accum_real_dtype=None,
    accum_complex_dtype=None,
):
    real_override = accum_real_dtype
    complex_override = accum_complex_dtype
    base_override = accum_dtype

    if base_override is not None:
        base_dt = np.dtype(base_override)
        if base_dt.kind == "c":
            complex_override = base_override
            if real_override is None:
                real_override = _real_dtype_for(base_override, is_jax=is_jax)
        else:
            real_override = base_override
            if complex_override is None:
                complex_override = _complex_dtype_for(base_override, is_jax=is_jax)

    if real_override is None or complex_override is None:
        default_real, default_complex = _default_accum_dtypes(
            net_dtype, prefer_x64=prefer_x64, is_jax=is_jax
        )
        if real_override is None:
            real_override = default_real
        if complex_override is None:
            complex_override = default_complex
    return real_override, complex_override


def resolve_precision_policy(net_dtype, *, is_jax: bool, **kwargs) -> NQSPrecisionPolicy:
    prefer_x64 = kwargs.get("mp_use_x64", True)

    state_dtype = kwargs.get("mp_state_dtype", None)
    if state_dtype is None:
        state_dtype = kwargs.get("s_statetype", None)
    if state_dtype is None:
        state_dtype = jnp.float32 if is_jax else np.float32

    prob_dtype = kwargs.get("mp_prob_dtype", None)
    if prob_dtype is None:
        prob_dtype = _real_dtype_for(net_dtype, is_jax=is_jax)

    accum_real_dtype, accum_complex_dtype = _resolve_accum_dtypes(
        net_dtype,
        prefer_x64=prefer_x64,
        is_jax=is_jax,
        accum_dtype=kwargs.get("mp_accum_dtype", None),
        accum_real_dtype=kwargs.get("mp_accum_real_dtype", None),
        accum_complex_dtype=kwargs.get("mp_accum_complex_dtype", None),
    )

    return NQSPrecisionPolicy(
        state_dtype=state_dtype,
        prob_dtype=prob_dtype,
        accum_real_dtype=accum_real_dtype,
        accum_complex_dtype=accum_complex_dtype,
    )


def cast_for_precision(array, real_dtype, complex_dtype, use_jax: bool):
    if real_dtype is None and complex_dtype is None:
        return array
    if use_jax:
        is_complex = jnp.iscomplexobj(array)
        dtype = complex_dtype if is_complex else real_dtype
        current_dtype = getattr(array, "dtype", None)
        if dtype is None or (current_dtype is not None and current_dtype == dtype):
            return array
        return jnp.asarray(array, dtype=dtype)
    is_complex = np.iscomplexobj(array)
    dtype = complex_dtype if is_complex else real_dtype
    current_dtype = getattr(array, "dtype", None)
    if dtype is None or (current_dtype is not None and current_dtype == dtype):
        return array
    return np.asarray(array, dtype=dtype)
