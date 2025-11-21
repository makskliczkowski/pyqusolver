'''
Factories for operator wrapping functions
'''

import numba
import numpy as np
import numba.typed
import inspect
from numba import njit
from inspect import signature

try:
    from QES.general_python.algebra.utils import JAX_AVAILABLE
    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

def _make_add_int_njit(f1, f2):
    """ Factory for (f1 + f2) integer backend """
    @numba.njit(inline='always')
    def add_int(s, *args):
        s1, v1 = f1(s, *args)
        s2, v2 = f2(s, *args)
        # Concatenate results: (A+B)|psi> = A|psi> + B|psi>
        return np.concatenate((s1, s2)), np.concatenate((v1, v2))
    return add_int

def _make_add_np_njit(f1, f2):
    """ Factory for (f1 + f2) numpy backend """
    @numba.njit(inline='always')
    def add_np(s, *args):
        s1, v1 = f1(s, *args)
        s2, v2 = f2(s, *args)
        return np.concatenate((s1, s2)), np.concatenate((v1, v2))
    return add_np

def _make_sub_int_njit(f1, f2):
    """ Factory for (f1 - f2) integer backend """
    @numba.njit(inline='always')
    def sub_int(s, *args):
        s1, v1 = f1(s, *args)
        s2, v2 = f2(s, *args)
        # Negate the values of the second operator
        return np.concatenate((s1, s2)), np.concatenate((v1, -v2))
    return sub_int

def _make_sub_np_njit(f1, f2):
    """ Factory for (f1 - f2) numpy backend """
    @numba.njit(inline='always')
    def sub_np(s, *args):
        s1, v1 = f1(s, *args)
        s2, v2 = f2(s, *args)
        return np.concatenate((s1, s2)), np.concatenate((v1, -v2))
    return sub_np

# JAX versions
if JAX_AVAILABLE:
    def _make_add_jax(f1, f2):
        @jax.jit
        def add_jax(s, *args):
            s1, v1 = f1(s, *args)
            s2, v2 = f2(s, *args)
            # JAX concatenation
            return jnp.concatenate([s1, s2], axis=0), jnp.concatenate([v1, v2], axis=0)
        return add_jax

    def _make_sub_jax(f1, f2):
        @jax.jit
        def sub_jax(s, *args):
            s1, v1 = f1(s, *args)
            s2, v2 = f2(s, *args)
            return jnp.concatenate([s1, s2], axis=0), jnp.concatenate([v1, -v2], axis=0)
        return sub_jax
    
# ----------------------------------------------------------------------------

####################################################################################################

def _make_mul_int_njit(outer_op_fun, inner_op_fun, allocator_m=2):
    """
    Creates a Numba-jitted function that composes two operator functions acting on integer-based quantum states.
    This function returns a compiled implementation that, given a quantum state and additional arguments, applies
    `inner_op_fun` to the state, then applies `outer_op_fun` to each resulting state, combining the coefficients
    appropriately. The result is a tuple of arrays containing the resulting states and their corresponding coefficients.
    If Numba compilation fails, a fallback implementation returning an empty result is provided.
    Args:
        outer_op_fun (callable):
            A function that takes a state and additional arguments, returning a tuple of
            (states, coefficients) as 1D numpy arrays.
        inner_op_fun (callable): 
            A function with the same signature as `outer_op_fun`, applied first.
    Returns:
        callable:
            A function with signature (state, *args) -> (states, coefficients), where `states` is a 1D numpy
            array of int64 and `coefficients` is a 1D numpy array of float64 or complex128.
    """
    try:
        # Wrapper that always returns complex128 - safe for all operator products
        # This avoids Numba's strict type unification issues with conditional dtypes
        @numba.njit(cache=True)
        def mul_int_impl(state, *args):
            g_states, g_coeffs = inner_op_fun(state, *args)
            if g_states.shape[0] == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.complex128)

            total_estimate  = g_states.shape[0] * allocator_m
            res_states      = np.empty(total_estimate, dtype=np.int64)
            res_coeffs      = np.empty(total_estimate, dtype=np.complex128)
            count           = 0

            for k in range(g_states.shape[0]):
                g_state_k   = g_states[k]
                g_coeff_k   = g_coeffs[k]
                f_states_k, f_coeffs_k = outer_op_fun(g_state_k, *args)

                for l in range(f_states_k.shape[0]):
                    if count >= res_states.shape[0]:
                        new_size = res_states.shape[0] * 2
                        tmp_s           = np.empty(new_size, dtype=np.int64)
                        tmp_c           = np.empty(new_size, dtype=np.complex128)
                        tmp_s[:count]   = res_states[:count]
                        tmp_c[:count]   = res_coeffs[:count]
                        res_states      = tmp_s
                        res_coeffs      = tmp_c

                    res_states[count]   = f_states_k[l]
                    res_coeffs[count]   = g_coeff_k * f_coeffs_k[l]
                    count              += 1

            return res_states[:count], res_coeffs[:count]

        return mul_int_impl
    except Exception as e: # Catch Numba errors
        print(f"Numba compilation failed for _make_mul_int_njit: {e}. Falling back to Python.")
        
        def mul_int_fallback(state, *args):
            return _empty_result_int()
        return mul_int_fallback

def _make_mul_np_njit(outer_op_fun, inner_op_fun, allocator_m=2):
    """
    Creates a function that applies two operator functions in sequence to a given state,
    efficiently handling the allocation of output arrays for states and coefficients.
    Parameters
    ----------
    outer_op_fun : callable
        A function that takes a state and additional arguments, and returns a tuple of
        (states, coefficients) as NumPy arrays. This function is applied after `inner_op_fun`.
    inner_op_fun : callable
        A function that takes a state and additional arguments, and returns a tuple of
        (states, coefficients) as NumPy arrays. This function is applied first.
    allocator_m : int, optional
        A multiplier used to estimate the maximum number of output states for pre-allocation.
        Default is 2.
    Returns
    -------
    mul_np_impl : callable
        A function that takes a state (as a NumPy array) and additional arguments, applies
        `inner_op_fun` followed by `outer_op_fun` to each resulting state, and returns a tuple:
        (resulting_states, resulting_coefficients), both as NumPy arrays.
    Notes
    -----
    - If the number of resulting states exceeds the pre-allocated size, the function will
      stop processing further states to avoid overflow.
    - If Numba compilation is intended but fails, the function falls back to a pure NumPy implementation.
    - The function assumes that the input state is a 1D NumPy array.
    """
    
    def mul_np_impl(state_np, *args_np):
        g_states, g_coeffs = inner_op_fun(state_np, *args_np)  # (M, D), (M,)
        if g_states.shape[0] == 0:
            return np.empty((0, state_np.shape[0]), dtype=state_np.dtype), np.empty((0,), dtype=g_coeffs.dtype)

        max_est     = g_states.shape[0] * allocator_m
        state_dim   = state_np.shape[0]  # assumes 1D input
        res_states  = np.empty((max_est, state_dim), dtype=state_np.dtype)
        res_coeffs  = np.empty((max_est,), dtype=g_coeffs.dtype)
        count       = 0

        for k in range(g_states.shape[0]):
            f_states_k, f_coeffs_k = outer_op_fun(g_states[k], *args_np)
            for j in range(f_states_k.shape[0]):
                if count >= max_est:
                    break
                res_states[count] = f_states_k[j]
                res_coeffs[count] = g_coeffs[k] * f_coeffs_k[j]
                count            += 1
        return res_states[:count], res_coeffs[:count]
    
    #! Attempt Numba compilation; if it fails, use the pure Python/NumPy version.
    try:
        return mul_np_impl
    except Exception as e:
        print("Numba compilation failed for _make_mul_np_njit. Falling back to pure NumPy version: ", e)
        return lambda state_np, *args_np: np.empty((0, state_np.shape[0]), dtype=state_np.dtype), np.empty((0,), dtype=np.float64)

def _make_mul_jax_vmap(outer_op_fun_jax, inner_op_fun_jax):

    if not JAX_AVAILABLE:
        return None 
        
    if outer_op_fun_jax is None or inner_op_fun_jax is None:
        return None

    @partial(jax.jit, static_argnums=())
    def mul_jax_impl(state_initial, *args):
        g_states, g_coeffs = inner_op_fun_jax(state_initial, *args)  # (M, D), (M,)

        def apply_outer(state, coeff):
            f_states, f_coeffs = outer_op_fun_jax(state, *args)  # (K, D), (K,)
            # Ensure complex dtype if either coefficient is complex
            # This handles cases like sigma_y which returns complex coefficients
            combined_coeffs = coeff * f_coeffs
            # Cast to complex128 if needed to avoid dtype mismatches in JAX operations
            if jnp.iscomplexobj(g_coeffs) or jnp.iscomplexobj(f_coeffs):
                combined_coeffs = jnp.asarray(combined_coeffs, dtype=jnp.complex128)
            return f_states, combined_coeffs

        # g_states: (M, D), g_coeffs: (M,)
        f_states, f_coeffs = jax.vmap(apply_outer, in_axes=(0, 0))(g_states, g_coeffs)

        s_dim = f_states.shape[-1]
        return f_states.reshape(-1, s_dim), f_coeffs.reshape(-1)
    
    try:
        # Example call to force compilation and catch errors early (optional)
        # This requires knowing typical shapes and dtypes for s_initial and args_for_ops
        # For instance, if s_initial is (D,) and args_for_ops is empty for global ops:
        # test_state = jnp.zeros((1,), dtype=jnp.float32) # Or a more representative state
        # if necessary_args == 0: mul_jax_impl.lower(test_state).compile()
        # elif necessary_args == 1: mul_jax_impl.lower(test_state, 0).compile()
        # This is complex to generalize here. JAX will compile on first actual call.
        pass
    except jax.errors.JAXTypeError as e:
        print(f"JAX compilation failed for _make_mul_jax_vmap: {e}. JAX operations will not be available for this composed operator.")
        return None
    return mul_jax_impl

# ----------------------------------------------------------------------------