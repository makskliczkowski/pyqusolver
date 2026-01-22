r"""
This module implements spin-1 operators for quantum systems using the JAX library.
It includes functions for S_x, S_y, S_z, S_plus (raising), S_minus (lowering),
and their products for spin-1 (S=1) systems with JAX acceleration.

Spin-1 systems have local dimension 3 with states |+1⟩, |0⟩, |-1⟩.
The representation uses integer encoding where each site uses 2 bits (trits):
    00 (0) -> |+1⟩  (m = +1)
    01 (1) -> | 0⟩  (m =  0)
    10 (2) -> |-1⟩  (m = -1)
    11 (3) -> invalid (not used)

Spin-1 Matrices (in |+1⟩, |0⟩, |-1⟩ basis):

    S_x = 1/sqrt2 * |0  1  0|      S_y = 1/sqrt2 *  |0  -i  0|      S_z = |1  0  0|
                    |1  0  1|                       |i   0 -i|            |0  0  0|
                    |0  1  0|                       |0   i  0|            |0  0 -1|

    S_+ = sqrt2 *   |0  1  0|        S_- = sqrt2 *  |0  0  0|
                    |0  0  1|                       |1  0  0|
                    |0  0  0|                       |0  1  0|

--------------------------------------------------------------
File        : QES/Algebra/Operator/jax/operators_spin_1.py
Description : JAX-accelerated spin-1 operator implementations
Author      : Maksymilian Kliczkowski
Date        : December 2025
--------------------------------------------------------------
"""

from functools import partial
from typing import List, Union

import numpy as np

################################################################################
# Local imports
################################################################################

try:
    import jax
    from jax import lax
    from jax import numpy as jnp
    from QES.Algebra.Operator.operator import ensure_operator_output_shape_jax

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    JAX_AVAILABLE = False
    Operator = None
    OperatorTypeActing = None
    ensure_operator_output_shape_jax = None

################################################################################
# JAX imports and matrix definitions
################################################################################

if JAX_AVAILABLE:

    # Spin-1 constants
    _SQRT2_INV_jnp = jnp.sqrt(0.5)
    _SQRT2_jnp = jnp.sqrt(2.0)

    # Spin-1 matrices as JAX arrays (3x3) in |+1⟩, |0⟩, |-1⟩ basis
    _S1_IDENTITY_jnp = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    _S1_X_jnp = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) * _SQRT2_INV_jnp

    _S1_Y_jnp = jnp.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) * _SQRT2_INV_jnp

    _S1_Z_jnp = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=float)

    _S1_PLUS_jnp = jnp.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float) * _SQRT2_jnp

    _S1_MINUS_jnp = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float) * _SQRT2_jnp

    _S1_Z2_jnp = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)

else:
    jnp = np
    jax = None
    lax = None
    _SQRT2_INV_jnp = np.sqrt(0.5)
    _SQRT2_jnp = np.sqrt(2.0)
    _S1_IDENTITY_jnp = np
    _S1_X_jnp = np
    _S1_Y_jnp = np
    _S1_Z_jnp = np
    _S1_PLUS_jnp = np
    _S1_MINUS_jnp = np
    _S1_Z2_jnp = np

################################################################################
#! Constants
################################################################################

_SPIN_1 = 1.0

################################################################################
#! Bit manipulation for spin-1 (using 2 bits per site) - JAX version
################################################################################

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2))
    def _get_spin1_state_jax(state: int, ns: int, site: int) -> int:
        """
        Get the spin-1 state (0, 1, or 2) at a given site.

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        site : int
            Site index (0 to ns-1).

        Returns
        -------
        int
            Local state: 0 = |+1⟩, 1 = |0⟩, 2 = |-1⟩
        """
        bit_pos = 2 * (ns - 1 - site)
        return (state >> bit_pos) & 0b11

    @partial(jax.jit, static_argnums=(1, 2))
    def _set_spin1_state_jax(state: int, ns: int, site: int, local_state: int) -> int:
        """
        Set the spin-1 state at a given site.

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        site : int
            Site index (0 to ns-1).
        local_state : int
            New local state: 0 = |+1⟩, 1 = |0⟩, 2 = |-1⟩

        Returns
        -------
        int
            Updated state.
        """
        bit_pos = 2 * (ns - 1 - site)
        mask = ~(0b11 << bit_pos)
        state = state & mask
        state = state | (local_state << bit_pos)
        return state

    @jax.jit
    def _spin1_magnetization_jax(local_state: int) -> float:
        """Get the m value for a local spin-1 state."""
        # 0 -> +1, 1 -> 0, 2 -> -1
        return 1.0 - local_state.astype(float)


# -----------------------------------------------------------------------------
#! S_z operator (diagonal)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_z_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Apply S_z operator to an integer state for spin-1 system.
        S_z |m⟩ = m |m⟩ where m ∈ {+1, 0, -1}

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices to apply the operator.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (state, coeff) - state is unchanged, coeff is the product of magnetizations.
        """
        sites_arr = jnp.array(sites)

        def body(i, coeff):
            site = sites_arr[i]
            bit_pos = 2 * (ns - 1 - site)
            local_s = (state >> bit_pos) & 0b11
            # m = 1 - local_s (0->+1, 1->0, 2->-1)
            m = 1.0 - local_s.astype(float)
            return coeff * m * spin_value

        coeff = lax.fori_loop(0, len(sites), body, 1.0)
        return ensure_operator_output_shape_jax(state, coeff)

    def spin1_z_jnp(state, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
        """
        Apply S_z operator to a JAX array state for spin-1 system.
        For array representation, each element represents the local state.

        Parameters
        ----------
        state : jnp.ndarray
            Array of local states (0, 1, or 2 for each site).
        sites : tuple
            Tuple of site indices to apply the operator.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (state, coeff) - state is unchanged, coeff is the product of magnetizations.
        """
        sites_arr = jnp.asarray(sites)

        # Extract local states and compute magnetizations
        local_states = state[sites_arr]
        magnetizations = 1.0 - local_states.astype(float)
        coeff = jnp.prod(magnetizations) * (spin_value ** len(sites))

        return ensure_operator_output_shape_jax(state, coeff)

    def spin1_z_inv_jnp(state, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
        """Inverse of S_z (same as S_z since it's Hermitian)."""
        return spin1_z_jnp(state, sites, spin_value)


# -----------------------------------------------------------------------------
#! S_+ (raising) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_plus_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Apply S_+ (raising) operator to an integer state for spin-1 system.
        S_+ |+1⟩ = 0
        S_+ |0⟩  = sqrt2 |+1⟩
        S_+ |-1⟩ = sqrt2 |0⟩

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices to apply the operator.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_state, coeff) - transformed state and coefficient.
        """
        sites_arr = jnp.array(sites)
        sqrt2 = jnp.sqrt(2.0)

        def body(i, carry):
            curr_state, curr_coeff = carry

            def skip_branch(_):
                return curr_state, curr_coeff

            def compute_branch(_):
                site = sites_arr[i]
                bit_pos = 2 * (ns - 1 - site)
                local_s = (curr_state >> bit_pos) & 0b11

                # S_+ transitions: 2->1 (sqrt2), 1->0 (sqrt2), 0->X (0)
                # local_s: 0=|+1⟩, 1=|0⟩, 2=|-1⟩
                new_local = local_s - 1

                # Coefficient is sqrt2 if valid transition, 0 if at |+1⟩
                valid = local_s > 0
                new_coeff = lax.cond(
                    valid, lambda _: curr_coeff * sqrt2 * spin_value, lambda _: 0.0, operand=None
                )

                # Update state only if valid
                mask = ~(0b11 << bit_pos)
                new_state = lax.cond(
                    valid,
                    lambda _: (curr_state & mask) | (new_local << bit_pos),
                    lambda _: curr_state,
                    operand=None,
                )
                return new_state, new_coeff

            return lax.cond(curr_coeff == 0.0, skip_branch, compute_branch, operand=None)

        init = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)

    def spin1_plus_jnp(state, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
        """
        Apply S_+ (raising) operator to a JAX array state for spin-1 system.

        Parameters
        ----------
        state : jnp.ndarray
            Array of local states (0, 1, or 2 for each site).
        sites : tuple
            Tuple of site indices to apply the operator.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_state, coeff) - transformed state and coefficient.
        """
        sites_arr = jnp.asarray(sites)
        sqrt2 = jnp.sqrt(2.0)

        def body_fun(i, state_val):
            state_in, coeff_in = state_val

            def skip_branch(_):
                return state_in, coeff_in

            def compute_branch(_):
                site = sites_arr[i]
                local_s = state_in[site]

                # S_+ transitions
                valid = local_s > 0
                new_coeff = lax.cond(
                    valid, lambda _: coeff_in * sqrt2 * spin_value, lambda _: 0.0, operand=None
                )

                new_state = lax.cond(valid, lambda s: s.at[site].add(-1), lambda s: s, state_in)
                return new_state, new_coeff

            return lax.cond(coeff_in == 0.0, skip_branch, compute_branch, operand=None)

        new_state, coeff = lax.fori_loop(0, len(sites), body_fun, (state, 1.0))
        return ensure_operator_output_shape_jax(new_state, coeff)


# -----------------------------------------------------------------------------
#! S_- (lowering) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_minus_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Apply S_- (lowering) operator to an integer state for spin-1 system.
        S_- |+1⟩ = sqrt2 |0⟩
        S_- |0⟩  = sqrt2 |-1⟩
        S_- |-1⟩ = 0

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices to apply the operator.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_state, coeff) - transformed state and coefficient.
        """
        sites_arr = jnp.array(sites)
        sqrt2 = jnp.sqrt(2.0)

        def body(i, carry):
            curr_state, curr_coeff = carry

            def skip_branch(_):
                return curr_state, curr_coeff

            def compute_branch(_):
                site = sites_arr[i]
                bit_pos = 2 * (ns - 1 - site)
                local_s = (curr_state >> bit_pos) & 0b11

                # S_- transitions: 0->1 (sqrt2), 1->2 (sqrt2), 2->X (0)
                new_local = local_s + 1

                # Coefficient is sqrt2 if valid transition, 0 if at |-1⟩
                valid = local_s < 2
                new_coeff = lax.cond(
                    valid, lambda _: curr_coeff * sqrt2 * spin_value, lambda _: 0.0, operand=None
                )

                # Update state only if valid
                mask = ~(0b11 << bit_pos)
                new_state = lax.cond(
                    valid,
                    lambda _: (curr_state & mask) | (new_local << bit_pos),
                    lambda _: curr_state,
                    operand=None,
                )
                return new_state, new_coeff

            return lax.cond(curr_coeff == 0.0, skip_branch, compute_branch, operand=None)

        init = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
        return ensure_operator_output_shape_jax(final_state, final_coeff)

    def spin1_minus_jnp(state, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
        """
        Apply S_- (lowering) operator to a JAX array state for spin-1 system.

        Parameters
        ----------
        state : jnp.ndarray
            Array of local states (0, 1, or 2 for each site).
        sites : tuple
            Tuple of site indices to apply the operator.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_state, coeff) - transformed state and coefficient.
        """
        sites_arr = jnp.asarray(sites)
        sqrt2 = jnp.sqrt(2.0)

        def body_fun(i, state_val):
            state_in, coeff_in = state_val

            def skip_branch(_):
                return state_in, coeff_in

            def compute_branch(_):
                site = sites_arr[i]
                local_s = state_in[site]

                # S_- transitions
                valid = local_s < 2
                new_coeff = lax.cond(
                    valid, lambda _: coeff_in * sqrt2 * spin_value, lambda _: 0.0, operand=None
                )

                new_state = lax.cond(valid, lambda s: s.at[site].add(1), lambda s: s, state_in)
                return new_state, new_coeff

            return lax.cond(coeff_in == 0.0, skip_branch, compute_branch, operand=None)

        new_state, coeff = lax.fori_loop(0, len(sites), body_fun, (state, 1.0))
        return ensure_operator_output_shape_jax(new_state, coeff)


# -----------------------------------------------------------------------------
#! S_x operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_x_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Apply S_x operator to an integer state for spin-1 system.
        S_x = (S_+ + S_-) / sqrt2

        For spin-1:
        S_x |+1⟩ = |0⟩
        S_x |0⟩  = |+1⟩ + |-1⟩  (superposition - branching!)
        S_x |-1⟩ = |0⟩

        Note: This returns multiple branches. For single-branch operations,
        use the composition function or matrix representation.

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices to apply the operator (typically single site).
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_states, coeffs) - arrays of possible output states and coefficients.

        Warning
        -------
        S_x on spin-1 creates superpositions! Use with caution for VMC/NQS.
        """
        # For integer state, handle single site case
        if len(sites) == 1:
            site = sites[0]
            bit_pos = 2 * (ns - 1 - site)
            local_s = (state >> bit_pos) & 0b11
            mask = ~(0b11 << bit_pos)

            # S_x action depends on initial state
            # |+1⟩ (0) -> |0⟩ (1) with coeff 1
            # |0⟩ (1) -> |+1⟩ (0) + |-1⟩ (2) with coeff 1 each
            # |-1⟩ (2) -> |0⟩ (1) with coeff 1

            def case_plus1(_):  # local_s == 0
                new_state = (state & mask) | (1 << bit_pos)
                return jnp.array([new_state]), jnp.array([spin_value])

            def case_zero(_):  # local_s == 1 - creates superposition
                state_up = (state & mask) | (0 << bit_pos)
                state_dn = (state & mask) | (2 << bit_pos)
                return jnp.array([state_up, state_dn]), jnp.array([spin_value, spin_value])

            def case_minus1(_):  # local_s == 2
                new_state = (state & mask) | (1 << bit_pos)
                return jnp.array([new_state]), jnp.array([spin_value])

            def case_invalid(_):
                return jnp.array([state]), jnp.array([0.0])

            return lax.switch(
                local_s, [case_plus1, case_zero, case_minus1, case_invalid], operand=None
            )
        else:
            raise ValueError(
                "spin1_x_int_jnp supports single site only. Use composition for multi-site."
            )


# -----------------------------------------------------------------------------
#! S_y operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_y_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Apply S_y operator to an integer state for spin-1 system.
        S_y = (S_+ - S_-) / (isqrt2)

        For spin-1:
        S_y |+1⟩ = -i |0⟩
        S_y |0⟩  = i(|+1⟩ - |-1⟩) / sqrt2  (superposition with complex coeff)
        S_y |-1⟩ = i |0⟩

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices to apply the operator (typically single site).
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_states, coeffs) - arrays of possible output states and complex coefficients.
        """
        if len(sites) == 1:
            site = sites[0]
            bit_pos = 2 * (ns - 1 - site)
            local_s = (state >> bit_pos) & 0b11
            mask = ~(0b11 << bit_pos)

            def case_plus1(_):  # |+1⟩ -> -i|0⟩
                new_state = (state & mask) | (1 << bit_pos)
                return jnp.array([new_state]), jnp.array([-1j * spin_value])

            def case_zero(_):  # |0⟩ -> i|+1⟩ - i|-1⟩
                state_up = (state & mask) | (0 << bit_pos)
                state_dn = (state & mask) | (2 << bit_pos)
                return jnp.array([state_up, state_dn]), jnp.array(
                    [1j * spin_value, -1j * spin_value]
                )

            def case_minus1(_):  # |-1⟩ -> i|0⟩
                new_state = (state & mask) | (1 << bit_pos)
                return jnp.array([new_state]), jnp.array([1j * spin_value])

            def case_invalid(_):
                return jnp.array([state]), jnp.array([0.0 + 0j])

            return lax.switch(
                local_s, [case_plus1, case_zero, case_minus1, case_invalid], operand=None
            )
        else:
            raise ValueError(
                "spin1_y_int_jnp supports single site only. Use composition for multi-site."
            )


# -----------------------------------------------------------------------------
#! S_+S_- and S_-S_+ (two-site correlators)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_pm_int_jnp(state, ns: int, sites: tuple, spin_value: float = _SPIN_1):
        """
        Apply S_+^i S_-^j operator to an integer state for spin-1 system.
        This is a two-site operator: raises site i, lowers site j.

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of (site_i, site_j) for S_+^i S_-^j.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_state, coeff) - transformed state and coefficient.
        """
        assert len(sites) == 2, "spin1_pm requires exactly 2 sites"
        site_i, site_j = sites[0], sites[1]

        # Get local states
        bit_pos_i = 2 * (ns - 1 - site_i)
        bit_pos_j = 2 * (ns - 1 - site_j)
        local_i = (state >> bit_pos_i) & 0b11
        local_j = (state >> bit_pos_j) & 0b11

        # S_+ on site i: need local_i > 0
        # S_- on site j: need local_j < 2
        valid = (local_i > 0) & (local_j < 2)

        # Coefficient is 2 (sqrt2 × sqrt2) for valid transition
        coeff = lax.cond(
            valid, lambda _: 2.0 * spin_value * spin_value, lambda _: 0.0, operand=None
        )

        # New local states
        new_local_i = local_i - 1
        new_local_j = local_j + 1

        mask_i = ~(0b11 << bit_pos_i)
        mask_j = ~(0b11 << bit_pos_j)

        new_state = lax.cond(
            valid,
            lambda _: (
                (state & mask_i & mask_j) | (new_local_i << bit_pos_i) | (new_local_j << bit_pos_j)
            ),
            lambda _: state,
            operand=None,
        )

        return ensure_operator_output_shape_jax(new_state, coeff)

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_mp_int_jnp(state, ns: int, sites: tuple, spin_value: float = _SPIN_1):
        """
        Apply S_-^i S_+^j operator to an integer state for spin-1 system.
        This is a two-site operator: lowers site i, raises site j.

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of (site_i, site_j) for S_-^i S_+^j.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (new_state, coeff) - transformed state and coefficient.
        """
        assert len(sites) == 2, "spin1_mp requires exactly 2 sites"
        site_i, site_j = sites[0], sites[1]

        # Get local states
        bit_pos_i = 2 * (ns - 1 - site_i)
        bit_pos_j = 2 * (ns - 1 - site_j)
        local_i = (state >> bit_pos_i) & 0b11
        local_j = (state >> bit_pos_j) & 0b11

        # S_- on site i: need local_i < 2
        # S_+ on site j: need local_j > 0
        valid = (local_i < 2) & (local_j > 0)

        coeff = lax.cond(
            valid, lambda _: 2.0 * spin_value * spin_value, lambda _: 0.0, operand=None
        )

        new_local_i = local_i + 1
        new_local_j = local_j - 1

        mask_i = ~(0b11 << bit_pos_i)
        mask_j = ~(0b11 << bit_pos_j)

        new_state = lax.cond(
            valid,
            lambda _: (
                (state & mask_i & mask_j) | (new_local_i << bit_pos_i) | (new_local_j << bit_pos_j)
            ),
            lambda _: state,
            operand=None,
        )

        return ensure_operator_output_shape_jax(new_state, coeff)


# -----------------------------------------------------------------------------
#! S_z^i S_z^j (ZZ correlator)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_zz_int_jnp(state, ns: int, sites: tuple, spin_value: float = _SPIN_1):
        """
        Apply S_z^i S_z^j operator to an integer state for spin-1 system.

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of (site_i, site_j) for S_z^i S_z^j.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (state, coeff) - state unchanged, coeff is product of magnetizations.
        """
        assert len(sites) == 2, "spin1_zz requires exactly 2 sites"
        site_i, site_j = sites[0], sites[1]

        # Get magnetizations
        bit_pos_i = 2 * (ns - 1 - site_i)
        bit_pos_j = 2 * (ns - 1 - site_j)
        local_i = (state >> bit_pos_i) & 0b11
        local_j = (state >> bit_pos_j) & 0b11

        m_i = 1.0 - local_i.astype(float)
        m_j = 1.0 - local_j.astype(float)

        coeff = m_i * m_j * spin_value * spin_value

        return ensure_operator_output_shape_jax(state, coeff)


# -----------------------------------------------------------------------------
#! S^2 operator (total spin squared at a site)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_squared_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Apply S^2 = S(S+1) = 2 operator to an integer state for spin-1 system.
        For S=1, S^2 = 1*2 = 2 (eigenvalue)

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (state, coeff) - state unchanged, coeff is 2^n where n is number of sites.
        """
        # For spin-1, S^2 eigenvalue is S(S+1) = 2
        n_sites = len(sites)
        coeff = (2.0 * spin_value) ** n_sites
        return ensure_operator_output_shape_jax(state, coeff)


# -----------------------------------------------------------------------------
#! Total magnetization (sum of S_z over all sites)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_z_total_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Compute total S_z = Σ_i S_z^i for spin-1 system.

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices to sum over.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (state, total_mz) - state unchanged, total magnetization.
        """
        sites_arr = jnp.array(sites)

        def body(i, total):
            site = sites_arr[i]
            bit_pos = 2 * (ns - 1 - site)
            local_s = (state >> bit_pos) & 0b11
            m = 1.0 - local_s.astype(float)
            return total + m * spin_value

        total_mz = lax.fori_loop(0, len(sites), body, 0.0)
        return ensure_operator_output_shape_jax(state, total_mz)

    def spin1_z_total_jnp(state, sites: Union[List[int], tuple], spin_value: float = _SPIN_1):
        """
        Compute total S_z = Σ_i S_z^i for spin-1 system (array state).

        Parameters
        ----------
        state : jnp.ndarray
            Array of local states.
        sites : tuple
            Tuple of site indices to sum over.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (state, total_mz) - state unchanged, total magnetization.
        """
        sites_arr = jnp.asarray(sites)
        local_states = state[sites_arr]
        magnetizations = 1.0 - local_states.astype(float)
        total_mz = jnp.sum(magnetizations) * spin_value
        return ensure_operator_output_shape_jax(state, total_mz)


# -----------------------------------------------------------------------------
#! (S_z)^2 operator (for quadrupolar interactions)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def spin1_z2_int_jnp(
        state, ns: int, sites: Union[List[int], tuple], spin_value: float = _SPIN_1
    ):
        """
        Apply (S_z)^2 operator to an integer state for spin-1 system.
        (S_z)^2 |m⟩ = m^2 |m⟩

        For spin-1: |+1⟩->1, |0⟩->0, |-1⟩->1

        Parameters
        ----------
        state : int
            The full system state as an integer.
        ns : int
            Number of sites.
        sites : tuple
            Tuple of site indices.
        spin_value : float
            Spin value multiplier (default 1.0).

        Returns
        -------
        tuple
            (state, coeff) - state unchanged, coeff is product of m^2 values.
        """
        sites_arr = jnp.array(sites)

        def body(i, coeff):
            site = sites_arr[i]
            bit_pos = 2 * (ns - 1 - site)
            local_s = (state >> bit_pos) & 0b11
            m = 1.0 - local_s.astype(float)
            m2 = m * m
            return coeff * m2 * spin_value

        coeff = lax.fori_loop(0, len(sites), body, 1.0)
        return ensure_operator_output_shape_jax(state, coeff)


# -----------------------------------------------------------------------------
#! Fallback for non-JAX environments
# -----------------------------------------------------------------------------

if not JAX_AVAILABLE:
    spin1_z_int_jnp = None
    spin1_z_jnp = None
    spin1_z_inv_jnp = None
    spin1_plus_int_jnp = None
    spin1_plus_jnp = None
    spin1_minus_int_jnp = None
    spin1_minus_jnp = None
    spin1_x_int_jnp = None
    spin1_y_int_jnp = None
    spin1_pm_int_jnp = None
    spin1_mp_int_jnp = None
    spin1_zz_int_jnp = None
    spin1_squared_int_jnp = None
    spin1_z_total_int_jnp = None
    spin1_z_total_jnp = None
    spin1_z2_int_jnp = None

# -----------------------------------------------------------------------------
#! Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Constants
    "_SPIN_1",
    "_SQRT2_INV_jnp",
    "_SQRT2_jnp",
    # Matrices
    "_S1_IDENTITY_jnp",
    "_S1_X_jnp",
    "_S1_Y_jnp",
    "_S1_Z_jnp",
    "_S1_PLUS_jnp",
    "_S1_MINUS_jnp",
    "_S1_Z2_jnp",
    # Operators - integer state
    "spin1_z_int_jnp",
    "spin1_plus_int_jnp",
    "spin1_minus_int_jnp",
    "spin1_x_int_jnp",
    "spin1_y_int_jnp",
    "spin1_pm_int_jnp",
    "spin1_mp_int_jnp",
    "spin1_zz_int_jnp",
    "spin1_squared_int_jnp",
    "spin1_z_total_int_jnp",
    "spin1_z2_int_jnp",
    # Operators - array state
    "spin1_z_jnp",
    "spin1_z_inv_jnp",
    "spin1_plus_jnp",
    "spin1_minus_jnp",
    "spin1_z_total_jnp",
]

# -----------------------------------------------------------------------------
#! End of file
# -----------------------------------------------------------------------------
