"""
Numerically stable Green's-function updates for DQMC.

This module is intentionally physics-agnostic. It only implements the matrix
identities used by DQMC:

- localized diagonal Green updates after an HS-field flip,
- imaginary-time propagation ``G(tau + 1) = B_tau G(tau) B_tau^{-1}``,
- stable equal-time Green's-function reconstruction
  ``G = (I + B_M ... B_1)^{-1}`` using factorized matrix stacks.
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import config, lax

    config.update("jax_enable_x64", True)
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "JAX is required for QES.pydqmc.stabilization. "
        "Install it via 'pip install jax jaxlib'."
    ) from exc

from functools import partial

try:
    from ..general_python.algebra.utilities import udt
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "UDT stabilization utilities are required for QES.pydqmc.stabilization."
    ) from exc

def calculate_green_stable_numpy(Bs: np.ndarray, n_stable: int = 10) -> np.ndarray:
    """
    Stable NumPy/SciPy reference for `G = (I + B_M ... B_1)^{-1}`.

    This path is not the hot Monte Carlo kernel.  It exists so that tests and
    debugging can use pivoted QR, which is the preferred stabilization strategy
    from the DQMC literature for robust reference calculations.
    """
    Bs = np.asarray(Bs)
    n_stable = max(1, int(n_stable))
    n_slices = Bs.shape[0]
    dim = Bs.shape[1]

    state = udt.UDTState(
        U=np.eye(dim, dtype=Bs.dtype),
        D=np.ones((dim,), dtype=Bs.dtype),
        T=np.eye(dim, dtype=Bs.dtype),
    )

    def multiply_group(group_Bs: np.ndarray) -> np.ndarray:
        """Multiply one stabilization window into the ordered product ``B_n ... B_1``."""
        group_product = np.eye(dim, dtype=Bs.dtype)
        for idx in range(group_Bs.shape[0]):
            group_product = group_Bs[idx] @ group_product
        return group_product

    for start in range(0, n_slices, n_stable):
        group = Bs[start : start + n_stable]
        state = udt.udt_fact_mult_pivoted(multiply_group(group), state)
    return udt.udt_inv_1p(state, backend="numpy")


@jax.jit
def localized_diagonal_update(G, indices, deltas):
    """
    Apply the exact DQMC equal-time update for a localized diagonal change.

    Math:
        if one slice changes by left multiplication with ``I + Delta`` where
        ``Delta`` is diagonal and supported only on a small set ``S``, then the
        equal-time Green's function changes by

            G' = G - G_{:,S} Delta_S
                 [I + (I - G)_{S,S} Delta_S]^{-1}
                 (I - G)_{S,:}.

        This is the small-rank Woodbury update used by DQMC.  For ``|S| = 1``
        it reduces to the standard Sherman-Morrison formula.

    Parameters
    ----------
    G : array, shape (N, N)
        Equal-time Green's function before the local update.
    indices : array, shape (k,)
        Site indices spanning the support ``S`` of the local HS term.
    deltas : array, shape (k,)
        Multiplicative diagonal changes ``exp(V'_i - V_i) - 1`` on ``S``.
        Entries may be zero, which is useful for padded fixed-width supports.
    """
    indices = jnp.asarray(indices, dtype=jnp.int32)
    deltas = jnp.asarray(deltas, dtype=G.dtype)
    k = indices.shape[0]

    sub_g = G[indices[:, None], indices[None, :]]
    delta_mat = jnp.diag(deltas)
    eye_k = jnp.eye(k, dtype=G.dtype)

    # R = I + (I - G)_{S,S} Delta_S is the small matrix whose determinant gives
    # the Metropolis ratio contribution of this channel.
    update_matrix = eye_k + (eye_k - sub_g) @ delta_mat

    rows_i_minus_g = -G[indices, :]
    rows_i_minus_g = rows_i_minus_g.at[jnp.arange(k), indices].add(1.0)

    cols_g_delta = G[:, indices] * deltas[None, :]
    correction = cols_g_delta @ jnp.linalg.solve(update_matrix, rows_i_minus_g)
    return G - correction


@jax.jit
def localized_diagonal_update_ratio(G, indices, deltas):
    """
    Return the DQMC determinant ratio for a localized diagonal HS update.

    Math:
        the local Metropolis factor for one fermion channel is

            det[I + (I - G)_{S,S} Delta_S],

        with the same small matrix that appears in the Woodbury correction.
    """
    indices = jnp.asarray(indices, dtype=jnp.int32)
    deltas = jnp.asarray(deltas, dtype=G.dtype)
    k = indices.shape[0]
    sub_g = G[indices[:, None], indices[None, :]]
    delta_mat = jnp.diag(deltas)
    eye_k = jnp.eye(k, dtype=G.dtype)
    update_matrix = eye_k + (eye_k - sub_g) @ delta_mat
    return jnp.linalg.det(update_matrix)


@jax.jit
def sherman_morrison_update(G, site, delta):
    """
    Apply the DQMC rank-1 Green's-function update for one local field change.

    Math:
        a local HS flip changes one diagonal entry of the slice propagator,
        so the updated equal-time Green's function can be written as a
        rank-1 correction to the old one.  The Sherman-Morrison identity gives

            G' = G - [delta / (1 + delta * (1 - G_ii))]
                     G[:, i] (e_i^T - G[i, :]),

        which is exactly the ``|S| = 1`` case of `localized_diagonal_update`.

    Parameters
    ----------
    G : array, shape (N, N)
        Equal-time Green's function before the local update.
    site : int
        Site index ``i`` of the changed HS field.
    delta : scalar
        Multiplicative diagonal change ``exp(V'_i - V_i) - 1`` for that site.
    """
    return localized_diagonal_update(
        G,
        jnp.asarray([site], dtype=jnp.int32),
        jnp.asarray([delta], dtype=G.dtype),
    )


@jax.jit
def propagate_green(G, B, iB):
    """
    Propagate the equal-time Green's function to the next Trotter slice.

    Math:
        if ``G(tau) = (I + B(tau, 0) B(beta, tau))^{-1}``, then moving the
        time origin by one slice gives the similarity transform

            G(tau + 1) = B_tau G(tau) B_tau^{-1}.
    """
    return B @ G @ iB


@partial(jax.jit, static_argnames=("n_stable",))
def calculate_green_stable(Bs, n_stable):
    """
    Reconstruct ``G = (I + B_M ... B_1)^{-1}`` from a stack of slice matrices.

    Math:
        the product ``B_M ... B_1`` becomes exponentially ill-conditioned at
        low temperature.  We therefore do not form the full product directly.
        Instead we multiply short blocks and repeatedly refactor the running
        product in UDT form.  The final inverse is then computed in factorized
        space via Loh's stable ``(I + U D T)^{-1}`` formula.

    Parameters
    ----------
    Bs : array, shape (M, N, N)
        Ordered imaginary-time slice propagators.
    n_stable : int
        Number of slices multiplied between refactorization checkpoints.
    """
    n_stable = max(1, int(n_stable))
    n_slices = Bs.shape[0]
    dim = Bs.shape[1]

    num_groups = n_slices // n_stable
    remainder = n_slices % n_stable

    def multiply_group(group_Bs):
        """Multiply one JAX stabilization window into the ordered product ``B_n ... B_1``."""
        # Compute ``B_group = B_n ... B_1`` inside one stabilization window.
        def mult_step(mat, idx):
            """Left-multiply the next slice matrix into the running block product."""
            return group_Bs[idx] @ mat, None

        group_product, _ = lax.scan(
            mult_step,
            jnp.eye(dim, dtype=Bs.dtype),
            jnp.arange(group_Bs.shape[0]),
        )
        return group_product

    def group_step(carry, group_idx):
        """Accumulate one stabilization window into the running UDT factorization."""
        group_start = group_idx * n_stable
        group_Bs = lax.dynamic_slice(Bs, (group_start, 0, 0), (n_stable, dim, dim))
        group_product = multiply_group(group_Bs)
        return udt.udt_fact_mult(group_product, carry, backend="jax"), None

    initial_state = udt.UDTState(
        U=jnp.eye(dim, dtype=Bs.dtype),
        D=jnp.ones((dim,), dtype=Bs.dtype),
        T=jnp.eye(dim, dtype=Bs.dtype),
    )
    final_state, _ = lax.scan(group_step, initial_state, jnp.arange(num_groups))

    if remainder > 0:
        rem_Bs = lax.dynamic_slice(Bs, (num_groups * n_stable, 0, 0), (remainder, dim, dim))
        final_state = udt.udt_fact_mult(
            multiply_group(rem_Bs),
            final_state,
            backend="jax",
        )

    return udt.udt_inv_1p(final_state, backend="jax")


@jax.jit
def stack_product(Bs):
    """
    Form the ordered DQMC slice product ``B_M ... B_1`` for one channel.

    Math:
        the equal-time Green's function is defined by

            G = (I + B_M ... B_1)^{-1},

        so diagnostics naturally need access to the same ordered stack product.
    """
    dim = Bs.shape[1]

    def mult_step(mat, idx):
        """Left-multiply one slice matrix into the ordered stack product."""
        return Bs[idx] @ mat, None

    product, _ = lax.scan(
        mult_step,
        jnp.eye(dim, dtype=Bs.dtype),
        jnp.arange(Bs.shape[0]),
    )
    return product


@jax.jit
def green_residual(G, B_product):
    """
    Return the consistency residual ``||(I + B) G - I||_F``.

    This is a lightweight numerical diagnostic used after refresh steps to
    measure how accurately the stabilized inverse satisfies the defining
    equation of the equal-time Green's function.
    """
    eye = jnp.eye(G.shape[0], dtype=G.dtype)
    return jnp.linalg.norm((eye + B_product) @ G - eye)


@jax.jit
def green_residual_from_stack(G, Bs):
    """
    Compute the equal-time inverse residual directly from a slice stack.

    This is the most convenient diagnostic during a DQMC run because the
    sampler already stores ``Bs`` and the current equal-time ``G``.
    """
    return green_residual(G, stack_product(Bs))


def stack_product_numpy(Bs: np.ndarray) -> np.ndarray:
    """
    NumPy reference for the ordered slice product ``B_M ... B_1``.
    """
    Bs = np.asarray(Bs)
    dim = Bs.shape[1]
    product = np.eye(dim, dtype=Bs.dtype)
    for idx in range(Bs.shape[0]):
        product = Bs[idx] @ product
    return product


def green_residual_numpy(G: np.ndarray, B_product: np.ndarray) -> float:
    """
    NumPy helper mirroring `green_residual` for tests and debugging.
    """
    eye = np.eye(G.shape[0], dtype=G.dtype)
    return float(np.linalg.norm((eye + B_product) @ G - eye))


def green_residual_from_stack_numpy(G: np.ndarray, Bs: np.ndarray) -> float:
    """
    NumPy helper mirroring `green_residual_from_stack` for tests and debugging.
    """
    return green_residual_numpy(G, stack_product_numpy(Bs))
