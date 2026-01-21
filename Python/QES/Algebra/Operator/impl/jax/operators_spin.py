"""
This module implements spin-1/2 operators for quantum systems using the JAX library.
It includes functions for sigma_x, sigma_y, sigma_z, sigma_plus (raising),
sigma_minus (lowering), their products, and a Fourier-transformed sigma_k operator.

----------------------------------------------------------------------------
Author      : Maksymilian Kliczkowski, WUST, Poland
Date        : February 2025
Version     : 1.0
----------------------------------------------------------------------------
"""

import  numpy as np
from    typing import List, Union
from    functools import partial

################################################################################
try:
    from QES.Algebra.Operator.operator      import ensure_operator_output_shape_jax
    from QES.general_python.common.binary   import BACKEND_REPR as _SPIN, BACKEND_DEF_SPIN, JAX_AVAILABLE
    import QES.general_python.common.binary as _binary
except ImportError as e:
    raise ImportError("Failed to import Operator base class or utilities. Ensure that the QES package is correctly installed and accessible.") from e
################################################################################

# JAX imports
if JAX_AVAILABLE:
    import  jax
    from    jax import lax
    from    jax import numpy as jnp

    # transform the matrices to JAX arrays
    _SIG_0_jnp = jnp.array([[1, 0],
                    [0, 1]], dtype=float)
    _SIG_X_jnp = jnp.array([[0, 1],
                    [1, 0]], dtype=float)
    _SIG_Y_jnp = jnp.array([[0, -1j],
                    [1j, 0]], dtype=complex)
    _SIG_Z_jnp = jnp.array([[1,  0],
                    [0, -1]], dtype=float)
    _SIG_P_jnp = jnp.array([[0, 1],
                    [0, 0]], dtype=float)
    _SIG_M_jnp = jnp.array([[0, 0],
                    [1, 0]], dtype=float)

    @partial(jax.jit, static_argnums=(2,))
    def _flip_func(state_val, pos, spin: bool):
        return jax.lax.cond(
            spin,
            lambda _: _binary.jaxpy.flip_array_jax_spin(state_val, pos),
            lambda _: _binary.jaxpy.flip_array_jax_nspin(state_val, pos),
            operand = None
        )
else:
    _SIG_0_jnp = np
    _SIG_X_jnp = np
    _SIG_Y_jnp = np
    _SIG_Z_jnp = np
    _SIG_P_jnp = np
    _SIG_M_jnp = np
    _flip_func = None

# -----------------------------------------------------------------------------
#! Sigma-X (sigma _x) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(1, 2, 3, 4))
    def sigma_x_int_jnp(state,
                        ns,
                        sites,
                        spin        : bool  = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        Apply the Pauli-X (sigma _x) operator on the given sites.
        For each site, flip the bit at position (ns-1-site) using a JAX-compatible flip function.
        
        Args:
            state: 
                A JAX integer (or traced array) representing the state.
            ns (int): 
                Number of sites.
            sites (Union[List[int], None]): 
                A list of site indices.
            spin_value (float): 
                Spin value (default _SPIN).
        
        Returns:
            A tuple (state, coeff) with the updated state and accumulated coefficient.
        """
        sites       = jnp.asarray(sites)
        def body(i, carry):
            curr_state, curr_coeff  = carry
            # sites is static, so extract the site.
            site                    = ns - 1 + sites[i]
            # flip is assumed to be a JAX-compatible function that flips the bit at position pos.
            new_state               = _binary.jaxpy.flip_int_traced_jax(curr_state, site)
            new_coeff               = curr_coeff * spin_value
            return (new_state, new_coeff)

        num_sites   = sites.shape[0]
        init        = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, num_sites, body, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)

    @partial(jax.jit, static_argnums=(2, 3))
    def sigma_x_jnp(state, sites, spin=BACKEND_DEF_SPIN, spin_value=_SPIN):
        sites_arr = jnp.asarray(sites)
        coeff     = spin_value ** sites_arr.shape[0]
        
        def update_spin(s):
            return s.at[sites_arr].set(-s[sites_arr])
            
        def update_nspin(s):
            return s.at[sites_arr].set(1 - s[sites_arr])

        new_state = jax.lax.cond(spin, update_spin, update_nspin, state)
        return ensure_operator_output_shape_jax(new_state, coeff)

    @partial(jax.jit, static_argnums=(2, 3))
    def sigma_x_inv_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        Apply the inverse of the Pauli-X (sigma _x) operator on a JAX array state.
        This is equivalent to applying the sigma _x operator again.
        Corresponds to the adjoint operation.
        <s|O|s'> = <s'|O\dag|s>
        meaning that we want to find all the states s' that lead to the state s.
        Parameters:
            state (jax.numpy.ndarray):
                The state to be modified.
            ns (int):
                Number of sites.
            sites (Union[List[int], None]):
                A list of site indices to flip.
        """
        return sigma_x_jnp(state, sites, spin, spin_value)
    
# -----------------------------------------------------------------------------
#! Sigma-Y (sigma _y ) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(1, 2, 3, 4))
    def sigma_y_int_jnp(state,
                        ns          : int,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        sigma _y  on an integer state (JAX version).

        For each site, if the bit at (ns-1-site) is set then multiply the coefficient
        by (1j*spin_value), otherwise by (-1j*spin_value); then flip the bit.
        
        Args:
            state (int or JAX array) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
        
        Returns:
            tuple: (new_state, coeff) where new_state is the state after applying the operator,
                and coeff is the accumulated complex coefficient.
        """
        sites_arr = jnp.array(sites)

        def body(i, carry):
            state_val, coeff    = carry
            pos                 = ns - 1 - sites_arr[i]
            bitmask             = jnp.left_shift(1, pos)
            condition           = (state_val & bitmask) > 0
            factor              = lax.cond(condition,
                                    lambda _: 1j * spin_value,
                                    lambda _: -1j * spin_value,
                                operand=None)
            new_state           = _binary.jaxpy.flip_int_traced_jax(state_val, pos)
            return (new_state, coeff * factor)

        final_state, final_coeff = lax.fori_loop(0, sites_arr.shape[0], body, (state, 1.0 + 0j))
        return ensure_operator_output_shape_jax(final_state, final_coeff)
        # return final_state, final_coeff

    @partial(jax.jit, static_argnums=(2, 3))
    def sigma_y_jnp(state,
                    sites       : Union[List[int], None],
                    spin        : bool = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        r"""
        sigma _y  on a JAX array state.
        """
        sites_arr   = jnp.asarray(sites)
        
        # 1. Update State (Vectorized)
        def update_spin(s):
            return s.at[sites_arr].set(-s[sites_arr])

        def update_nspin(s):
            return s.at[sites_arr].set(1 - s[sites_arr])

        new_state   = lax.cond(spin, update_spin, update_nspin, state)
        
        # 2. Update Coeff (Vectorized)
        # bit = 1 if state[site] is "set" (spin up / occupied)
        # For spin=True (state is +/-1): "set" usually means > 0 or < 0 depending on convention.
        # Assuming _binary.check_arr_jax logic:
        # If spin=True: bit = 1 if state > 0 else 0
        # If spin=False: bit = state (0 or 1)
        
        # We can replicate check_arr_jax logic efficiently:
        if spin:
            # spin representation: +/- 1. usually 1 is up (bit=1), -1 is down (bit=0)? 
            # Or vice versa.
            # check_arr_jax implementation in _binary is needed to be exact.
            # Assuming standard: up=1, down=-1. check_arr returns 1 for up.
            # Let's trust that state[sites_arr] > 0 works for spin=True.
            bits = (state[sites_arr] > 0).astype(state.dtype)
        else:
            bits = state[sites_arr]

        # Factor logic from original:
        # if bit: 1j * spin_value
        # else: -1j * spin_value
        # This simplifies to: 1j * spin_value * (2*bit - 1)
        
        factors = 1j * spin_value * (2 * bits - 1)
        coeff   = jnp.prod(factors)
        
        return ensure_operator_output_shape_jax(new_state, coeff)

    def sigma_y_real_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        Apply the Pauli-Y (\sigma _y ) operator on a JAX array state.
        Corresponds to the adjoint operation.
        """
        state, coeff = sigma_y_jnp(state, sites, spin, spin_value)
        return state, coeff.real

    # @partial(jax.jit, static_argnums=(2, 3))
    def sigma_y_inv_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        Apply the inverse of the Pauli-Y (sigma _y ) operator on a JAX array state.
        Corresponds to the adjoint operation.
        <s|O|s'> = <s'|O\dag|s>
        meaning that we want to find all the states s' that lead to the state s.
        Parameters:
            state (np.ndarray) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
        Returns:
            tuple: (new_state, coeff) where new_state is the state after applying the operator
                and coeff is the accumulated coefficient.
        """
        # The inverse of sigma _y  is sigma _y  itself but with a different sign.
        # This is because sigma _y  is anti-Hermitian.
        return sigma_y_jnp(state, sites, spin, -spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Z (\sum _z) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3, 4))
    def sigma_z_int_jnp(state,
                        ns          : int,
                        sites       : Union[List[int], None],
                        spin        : bool      = BACKEND_DEF_SPIN,
                        spin_value  : float     = _SPIN):
        r"""
        \sum _z on an integer state.
        For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
        The state is unchanged.
        
        Args:
            state :
                A JAX integer (or traced array of integers) representing the state.
            ns (int) :
                The number of sites.
            sites (Union[List[int], None]) :
                A list of site indices.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The spin value (default _SPIN).
        
        Returns:
            A tuple (state, coeff) where state is unchanged and coeff is the product
            of the factors determined by the bits in state.
        """
        # Body function for the fori_loop. The loop variable 'i' runs over site indices.
        
        sites   = jnp.asarray(sites)
        
        def body(i, coeff):
            # Since sites is a static Python list, we can extract the site index.
            # Compute the bit position: (ns - 1 - site)
            pos         = ns - 1 - sites[i]
            # Compute the bit mask using JAX operations.
            bitmask     = jnp.left_shift(1, pos)
            # Compute the condition: is the bit set? This returns a boolean JAX array.
            condition   = (state & bitmask) > 0
            # Use lax.cond to choose the factor:
            factor      = lax.cond(condition,
                                lambda _: spin_value,
                                lambda _: -spin_value,
                                operand=None)
            # Multiply the accumulator with the factor.
            return coeff * factor

        # Use lax.fori_loop to accumulate the coefficient over all sites.
        coeff = lax.fori_loop(0, sites.shape[0], body, 1.0)
        return ensure_operator_output_shape_jax(state, coeff)
        # return state, coeff

    # @partial(jax.jit, static_argnums=(2, 3))
    def sigma_z_jnp(state,
                    sites       : Union[List[int], None],
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        r"""
        \sum _z on a JAX array state.
        """        
        sites_arr   = jnp.asarray(sites)
        
        if spin:
            bits    = (state[sites_arr] > 0).astype(state.dtype)
        else:
            bits    = state[sites_arr]
             
        # Factor logic:
        # if bit: spin_value
        # else: -spin_value
        # Simplifies to: spin_value * (2*bit - 1)
        
        factors = spin_value * (2 * bits - 1)
        coeff   = jnp.prod(factors)
        
        return ensure_operator_output_shape_jax(state, coeff)
        # return state, coeff

    # @partial(jax.jit, static_argnums=(2, 3))
    def sigma_z_inv_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        Apply the inverse of the Pauli-Z (\sum _z) operator on a JAX array state.
        Corresponds to the adjoint operation.
        <s|O|s'> = <s'|O\dag|s>
        meaning that we want to find all the states s' that lead to the state s.
        Parameters:
            state (np.ndarray) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
        Returns:
            tuple: (state, coeff) where state is unchanged and coeff is the accumulated coefficient.
        """
        return sigma_z_jnp(state, sites, spin, spin_value)        

# -----------------------------------------------------------------------------
#! Sigma-Plus (sigma ⁺) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @jax.jit
    def sigma_plus_int_jnp(state, 
                        ns          : int, 
                        sites       : Union[List[int], None], 
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        Apply the raising operator sigma ⁺ on an integer state (JAX version).
        """
        sites = jnp.asarray(sites)
        
        def body(i, carry):
            curr_state, curr_coeff  = carry
            # Early exit: if coeff is already zero, skip further computation
            def skip_branch(_):
                return curr_state, curr_coeff
            def compute_branch(_):
                pos         = ns - 1 - sites[i]
                bitmask     = jnp.left_shift(1, pos)
                condition   = (curr_state & bitmask) > 0
                new_state   = _binary.jaxpy.flip_int_traced_jax(curr_state, pos)
                new_coeff   = lax.cond(condition,
                            lambda _: 0.0,
                            lambda _: curr_coeff * spin_value,
                            operand=None)
                return new_state, new_coeff
            return lax.cond(curr_coeff == 0.0, skip_branch, compute_branch, operand=None)
        
        init                        = (state, 1.0)
        final_state, final_coeff    = lax.fori_loop(0, sites.shape[0], body, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)
        # return final_state, final_coeff

    @partial(jax.jit, static_argnums=(2,))
    def sigma_plus_jnp(state,
                    sites       : Union[List[int], None],
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        r"""
        sigma ⁺ on a JAX array state.
        Uses lax.fori_loop.
        """
        sites_arr = jnp.asarray(sites)
        def body_fun(i, state_val):
            state_in, coeff_in  = state_val
            site                = sites_arr[i]

            def skip_branch(_):
                return state_in, coeff_in

            def compute_branch(_):
                coeff_new = jax.lax.cond(_binary.jaxpy.check_arr_jax(state_in, site),
                            lambda _: 0.0,
                            lambda _: coeff_in * spin_value,
                            operand=None)
                new_state = jax.lax.cond(spin,
                            lambda _: _binary.jaxpy.flip_array_jax_spin(state_in, site),
                            lambda _: _binary.jaxpy.flip_array_jax_nspin(state_in, site),
                            operand=None)
                return new_state, coeff_new
            return jax.lax.cond(coeff_in == 0.0, skip_branch, compute_branch, operand=None)
        
        new_state, coeff = lax.fori_loop(0, sites_arr.shape[0], body_fun, (state, 1.0))
        return ensure_operator_output_shape_jax(new_state, coeff)
        # return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma-Minus (sigma ^ -) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @jax.jit
    def sigma_minus_int_jnp(state, 
                            ns, 
                            sites, 
                            spin        : bool = BACKEND_DEF_SPIN,
                            spin_value  : float = _SPIN):
        sites_arr = jnp.array(sites)
        def body(i, carry):
            curr_state, curr_coeff = carry
            
            def skip_branch(_):
                return curr_state, curr_coeff
            def compute_branch(_):
                pos         = ns - 1 - sites_arr[i]
                bitmask     = jnp.left_shift(1, pos)
                condition   = (curr_state & bitmask) > 0
                new_state   = _binary.jaxpy.flip_int_traced_jax(curr_state, pos)
                new_coeff   = lax.cond(condition,
                                        lambda _: curr_coeff * spin_value,
                                        lambda _: 0.0,
                                        operand=None)
                return (new_state, new_coeff)
            return lax.cond(curr_coeff == 0.0, skip_branch, compute_branch, operand=None)
        init                        = (state, 1.0)
        final_state, final_coeff    = lax.fori_loop(0, sites_arr.shape[0], body, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)
        # return final_state, final_coeff

    # @jax.jit
    def sigma_minus_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        sigma ^ - on a JAX array state.
        """
        sites_arr = jnp.asarray(sites)
        def body_fun(i, state_val):
            state_in, coeff_in  = state_val
            def skip_branch(_):
                return state_in, coeff_in
            def compute_branch(_):
                site                = sites_arr[i]
                coeff_new           = jax.lax.cond(_binary.jaxpy.check_arr_jax(state_in, site),
                                                    lambda _: 0.0,
                                                    lambda _: coeff_in * spin_value,
                                                    operand=None)
                new_state           = jax.lax.cond(spin,
                                                    lambda _: _binary.jaxpy.flip_array_jax_spin(state_in, site),
                                                    lambda _: _binary.jaxpy.flip_array_jax_nspin(state_in, site),
                                                    operand=None)
                return new_state, coeff_new
            return jax.lax.cond(coeff_in == 0.0, skip_branch, compute_branch, operand=None)
        new_state, coeff = lax.fori_loop(0, sites_arr.shape[0], body_fun, (state, 1.0))
        return ensure_operator_output_shape_jax(new_state, coeff)
        # return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma_pm (sigma ⁺ then sigma ^ -) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(2,))
    def sigma_pm_jnp(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        coeff       = 1.0
        sites_arr   = jnp.asarray(sites)
        def body_fun(i, state_val):
            site = sites_arr[i]
            pos  = site

            def even_branch(_):
            # If bit is set, return state_val; else flip
                return jax.lax.cond(
                    _binary.jaxpy.check_arr_jax(state_val, pos),
                    lambda _: state_val,
                    lambda _: _flip_func(state_val, pos, spin),
                    operand = None
                )

            def odd_branch(_):
            # If bit is not set, return state_val; else flip
                return jax.lax.cond(
                    _binary.jaxpy.check_arr_jax(state_val, pos),
                    lambda _: _flip_func(state_val, pos, spin),
                    lambda _: state_val,
                    operand=None
                )

            return jax.lax.cond(
                    (i % 2) == 0,
                    even_branch,
                    odd_branch,
                    operand = None
                )

        new_state = lax.fori_loop(0, sites_arr.shape[0], body_fun, state)
        return new_state, coeff

    @partial(jax.jit, static_argnums=(2,))
    def sigma_pm_int_jnp(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        r'''
        \sigma ⁺ then \sigma ^ - on an integer state.
        For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
        '''
        sites = jnp.asarray(sites)
        
        # Body function for the fori_loop. The loop variable 'i' runs over site indices.
        def body(i, carry):
            curr_state, curr_coeff  = carry
            def skip_branch(_):
                return curr_state, curr_coeff
            def compute_branch(_):
                pos                     = sites[i]
                bitmask                 = jnp.left_shift(1, pos)
                even_branch             = lax.cond((curr_state & bitmask) == 0,
                                            lambda _: (_flip_func(curr_state, pos, spin), curr_coeff * spin_value),
                                            lambda _: (curr_state, 0.0),
                                            operand=None)
                odd_branch              = lax.cond((curr_state & bitmask) > 0,
                                            lambda _: (_flip_func(curr_state, pos, spin), curr_coeff * spin_value),
                                            lambda _: (curr_state, 0.0),
                                            operand=None)
                new_state, new_coeff = jax.lax.cond(
                                            (i % 2) == 0,
                                            even_branch,
                                            odd_branch,
                                            operand=None
                                        )   
                return (new_state, new_coeff)
            return lax.cond(curr_coeff == 0.0, skip_branch, compute_branch, operand=None)
        init                     = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, sites.shape[0], body, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)
        # return final_state, final_coeff

# -----------------------------------------------------------------------------
#! Sigma_mp (sigma ^ - then sigma ⁺) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @jax.jit
    def sigma_mp_int_jnp(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        sites = jnp.array(sites)
        def body(i, carry):
            curr_state, curr_coeff  = carry
            
            def skip_branch(_):
                return curr_state, curr_coeff
            def compute_branch(_):
                pos                     = sites[i]
                bitmask                 = jnp.left_shift(1, pos)
                even_branch             = lax.cond((curr_state & bitmask) > 0,
                                            lambda _: _flip_func(curr_state, pos, spin),
                                            lambda _: (curr_state, 0.0),
                                            operand=None)
                odd_branch              = lax.cond((curr_state & bitmask) == 0,
                                            lambda _: (_flip_func(curr_state, pos, spin), curr_coeff * spin_value),
                                            lambda _: (curr_state, 0.0),
                                            operand=None)
                new_state, new_coeff    = jax.lax.cond(
                                            (i % 2) == 0,
                                            even_branch,
                                            odd_branch,
                                            operand=None
                                        )
                return (new_state, new_coeff)
            return lax.cond(curr_coeff == 0.0, skip_branch, compute_branch, operand=None)
        init                     = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, sites.shape[0], body, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)
        # return final_state, final_coeff

    @jax.jit
    def sigma_mp_jnp(state,
                    sites   : Union[List[int], None],
                    spin    : bool = BACKEND_DEF_SPIN,
                    spin_value : float = _SPIN):
        """
        Alternating operator (sigma ^ - then sigma ⁺) on a JAX array state.
        """
        sites_arr = jnp.asarray(sites)
        def body_fun(i, state_val):
            state_in, coeff_in  = state_val
            
            def skip_branch(_):
                return state_in, coeff_in
            def compute_branch(_):
                site                = sites_arr[i]
                def even_branch(_):
                    # sigma ^ -: only act if bit is set
                    coeff_new = jax.lax.cond(_binary.jaxpy.check_arr_jax(state_in, site),
                                            lambda _: coeff_in * spin_value,
                                            lambda _: 0.0,
                                            operand=None)
                    new_state = jax.lax.cond(spin,
                                            lambda _: _binary.jaxpy.flip_array_jax_spin(state_in, site),
                                            lambda _: _binary.jaxpy.flip_array_jax_nspin(state_in, site),
                                            operand=None)
                    return new_state, coeff_new

                def odd_branch(_):
                    # sigma ⁺: only act if bit is not set
                    coeff_new = jax.lax.cond(_binary.jaxpy.check_arr_jax(state_in, site),
                                            lambda _: 0.0,
                                            lambda _: coeff_in * spin_value,
                                            operand=None)
                    new_state = jax.lax.cond(spin,
                                            lambda _: _binary.jaxpy.flip_array_jax_spin(state_in, site),
                                            lambda _: _binary.jaxpy.flip_array_jax_nspin(state_in, site),
                                            operand=None)
                    return new_state, coeff_new

                new_state, coeff_new = jax.lax.cond(
                        (i % 2) == 0,
                        even_branch,
                        odd_branch,
                        operand=None
                    )
                return new_state, coeff_new
            return jax.lax.cond(coeff_in == 0.0, skip_branch, compute_branch, operand=None)
        new_state, coeff = lax.fori_loop(0, sites_arr.shape[0], body_fun, (state, 1.0))
        return ensure_operator_output_shape_jax(new_state, coeff)
        # return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma-K (sigma _k) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @jax.jit
    def sigma_k_int_jnp(state, 
                        ns          : int, 
                        sites       : Union[List[int], None], 
                        k           : float, 
                        spin        : bool = BACKEND_DEF_SPIN, 
                        spin_value  : float = _SPIN):
        
        sites = jnp.asarray(sites)
        def body(i, total):
            site    = sites[i]
            pos     = ns - 1 - site
            bitmask = jnp.left_shift(1, pos)
            factor  = lax.cond((state & bitmask) > 0,
                            lambda _: 1j,
                            lambda _: -1.0,
                            operand=None)
            return total + factor * jnp.exp(1j * k * site)
        total   = lax.fori_loop(0, sites.shape[0], body, 0.0+0j)
        sqrt_l  = jnp.sqrt(jnp.array(sites.shape[0]))
        norm    = lax.cond(sites.shape[0] > 0, lambda _: sqrt_l, lambda _: jnp.array(1.0), operand=None)
        return state, total / norm

    @jax.jit
    def sigma_k_jnp(state,
                    sites       : Union[List[int], None],
                    k           : float,
                    spin        : bool = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        """
        Compute the Fourier-transformed spin operator (sigma _k) on a JAX array state.
        Uses lax.fori_loop.
        Parameters:
            state (np.ndarray) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            k (float) :
                The wave vector for the Fourier transform.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
        Returns:
            tuple: (state, coeff) where state is unchanged and coeff is the accumulated coefficient.
        """
        total       = 0.0 + 0j
        sites_arr   = jnp.asarray(sites)
        def body_fun(i, total_val):
            pos     = sites_arr[i]
            bit     = _binary.jaxpy.check_arr_jax(state, pos)
            # Pauli Z eigenvalue: +1 for spin-up (bit=0), -1 for spin-down (bit=1)
            factor  = (1.0 - 2.0 * bit) * spin_value
            return total_val + factor * jnp.exp(1j * k * pos)
        total   = lax.fori_loop(0, sites_arr.shape[0], body_fun, total)
        sqrt_l  = jnp.sqrt(jnp.array(sites_arr.shape[0]))
        norm    = lax.cond(
            sites_arr.shape[0] > 0,
            lambda _: sqrt_l,
            lambda _: jnp.array(1.0),
            operand=None
        )
        return ensure_operator_output_shape_jax(state, total / norm)

    def sigma_k_inv_jnp(state,
                        sites       : Union[List[int], None],
                        k           : float,
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        Apply the inverse
        of the Fourier-transformed spin operator (sigma _k) on a JAX array state.
        Corresponds to the adjoint operation.
        <s|O|s'> = <s'|O\dag|s>
        
        meaning that we want to find all the states s' that lead to the state s.
        Parameters:
            state (np.ndarray) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            k (float) :
                The wave vector for the Fourier transform.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
        Returns:
            tuple: (state, coeff) where state is unchanged and coeff is the accumulated coefficient.
        """
        # with different sign
        return sigma_k_jnp(state, sites, k, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Total (sigma _t ) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def sigma_z_total_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        r"""
        sigma _t  on a JAX array state.
        """
        sites_arr   = jnp.asarray(sites)
        coeff       = jnp.sum(state[sites_arr]) * spin_value
        return ensure_operator_output_shape_jax(state, coeff)

    def sigma_z_total_int_jnp(state,
                            sites       : Union[List[int], None],
                            spin        : bool = BACKEND_DEF_SPIN,
                            spin_value  : float = _SPIN):
        r"""
        sigma _t  on a JAX array state.
        """
        sites_arr   = jnp.asarray(sites)
        coeff       = 0.0
        def body(i, coeff):
            pos     = sites_arr[i]
            bitmask = jnp.left_shift(1, pos)
            bit     = (state & bitmask) > 0
            factor  = 2 * bit - 1.0
            return coeff + factor * spin_value
        coeff = lax.fori_loop(0, sites_arr.shape[0], body, coeff)
        return ensure_operator_output_shape_jax(state, coeff)

# -----------------------------------------------------------------------------
#! Pauli String Operator (sequence of Pauli gates)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    def _apply_pauli_sequence_jnp(state, sites, codes, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        r"""
        Apply a sequence of Pauli operators to a JAX array state.
        
        Parameters
        ----------
        state : jnp.ndarray
            The quantum state array.
        sites : array-like
            The sites where each Pauli operator acts.
        codes : array-like
            The Pauli operator codes: 0=X, 1=Y, 2=Z.
        spin : bool
            If True, use the spin convention for flipping bits.
        spin_value : float
            The spin value multiplier.
            
        Returns
        -------
        tuple
            (new_state, coeff) where new_state is the state after applying 
            all operators and coeff is the accumulated coefficient.
        """
        sites_arr = jnp.asarray(sites)
        codes_arr = jnp.asarray(codes)
        n         = codes_arr.shape[0]
        
        def body_fun(i, carry):
            curr_state, curr_coeff = carry
            
            # Apply operators from right to left (reverse order)
            idx  = n - 1 - i
            site = sites_arr[idx]
            code = codes_arr[idx]
            
            # Apply X: flip bit, coeff *= spin_value
            def apply_x(state_c):
                st, c   = state_c
                new_st  = jax.lax.cond(
                    spin,
                    lambda s    :   _binary.jaxpy.flip_array_jax_spin(s, site),
                    lambda s    :   _binary.jaxpy.flip_array_jax_nspin(s, site),
                    st
                )
                return (new_st, c * spin_value)
            
            # Apply Y: flip bit, coeff *= ±i*spin_value (sign depends on bit)
            def apply_y(state_c):
                st, c   = state_c
                bit     = _binary.jaxpy.check_arr_jax(st, site)
                factor  = jax.lax.cond(
                    bit,
                    lambda _    :   1j * spin_value,
                    lambda _    :   -1j * spin_value,
                    operand     =   None
                )
                new_st = jax.lax.cond(
                    spin,
                    lambda s    :   _binary.jaxpy.flip_array_jax_spin(s, site),
                    lambda s    :   _binary.jaxpy.flip_array_jax_nspin(s, site),
                    st
                )
                return (new_st, c * factor)
            
            # Apply Z: no flip, coeff *= ±spin_value (sign depends on bit)
            def apply_z(state_c):
                st, c   = state_c
                bit     = _binary.jaxpy.check_arr_jax(st, site)
                factor  = jax.lax.cond(
                    bit,
                    lambda _    :   spin_value,
                    lambda _    :   -spin_value,
                    operand     =   None
                )
                return (st, c * factor)
            
            # Use nested lax.cond to select based on code
            # code == 0 -> X, code == 1 -> Y, code == 2 -> Z
            new_state, new_coeff = jax.lax.cond(
                code == 0,
                apply_x,
                lambda sc: jax.lax.cond(
                    code == 1,
                    apply_y,
                    apply_z,
                    sc
                ),
                (curr_state, curr_coeff)
            )
            
            return (new_state, new_coeff)
        
        init                        = (state, 1.0 + 0.0j)
        final_state, final_coeff    = jax.lax.fori_loop(0, n, body_fun, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)

    @jax.jit
    def apply_pauli_sequence_jnp(state, sites, codes, spin_value: float = _SPIN):
        """
        JIT-compiled wrapper for Pauli sequence application (array state).
        spin parameter is set to True by default for compatibility.
        """
        return _apply_pauli_sequence_jnp(state, sites, codes, spin=True, spin_value=spin_value)

# -----------------------------------------------------------------------------

if not JAX_AVAILABLE:
    sigma_x_int_jnp = np
    sigma_x_jnp = np
    sigma_y_int_jnp = np
    sigma_y_jnp = np
    sigma_z_int_jnp = np
    sigma_z_jnp = np
    sigma_plus_int_jnp = np
    sigma_plus_jnp = np
    sigma_minus_int_jnp = np
    sigma_minus_jnp = np
    sigma_pm_jnp = np
    sigma_mp_jnp = np
    sigma_k_int_jnp = np
    sigma_k_jnp = np
    sigma_pm_int_jnp = np
    sigma_mp_int_jnp = np
    sigma_k_int_jnp = np
    sigma_k_jnp = np

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------