"""
JIT-compiled matrix builder applying an operator in a projected compact basis.

This module provides a Numba JIT-compiled function to efficiently apply an operator
to a set of input vectors in a reduced basis using symmetry projections.
The implementation leverages parallel processing and thread-local buffers for performance.

This is a twin module to QES/Algebra/Hilbert/matrix_builder.py but adapted for
the projected compact basis using symmetry groups.

------------------------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-05
Copyright       : (c) 2025-2026 Quantum EigenSolver
License         : MIT
------------------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

try:
    import numba
    from numba.typed import List

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    raise ImportError("Numba is required for JIT-compiled matrix building.")

# ------------------------------------------------------------------------------------------

try:
    from QES.Algebra.Symmetries.jit.symmetry_container_jit import (
        _INVALID_REPR_IDX_NB,
        _NUMBA_OVERHEAD_SIZE,
        apply_group_element_fast,
    )
except ImportError as e:
    raise ImportError(
        "QES.Algebra.Symmetries.symmetry_container module is required for matrix building: "
        + str(e)
    )

# ------------------------------------------------------------------------------------------


@numba.njit(fastmath=True, inline="always")
def _apply_op_batch_projected_compact_seq_jit(
    vecs_in: np.ndarray,  # (nh_in, n_batch)
    vecs_out: np.ndarray,  # (nh_out, n_batch)
    op_func: Callable,  # (state, *args) -> (new_states, values)
    args: Tuple,
    # Grouped basis and symmetry data
    basis_in_args: Tuple,  # (rep_list_in, norm_in)
    basis_out_args: Tuple,  # (repr_map_out, norm_out, rep_list_out)
    cg_args: Tuple,  # (n_group, n_ops, op_code, arg0, arg1, chi_in, chi_out)
    tb_args: Tuple,  # (trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
    ns: np.int64,
) -> None:
    """Sequential batch operator application in projected compact basis."""
    nh_in, n_batch = vecs_in.shape
    representative_list_in, normalization_in = basis_in_args
    repr_map_out, normalization_out, representative_list_out = basis_out_args
    n_group, _, _, _, _, chi_in, chi_out = cg_args
    inv_n_group = 1.0 / float(n_group)

    for k in range(nh_in):
        rep = np.int64(representative_list_in[k])
        norm_k = normalization_in[k]
        if norm_k == 0.0:
            continue
        inv_norm_k = 1.0 / norm_k

        for g in range(n_group):
            s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)
            w_g = np.conj(chi_in[g]) * ph_g * inv_norm_k
            new_states, values = op_func(s, *args)
            n_new = len(new_states)

            for i in range(n_new):
                new_state = new_states[i]
                val = values[i]
                if abs(val) < 1e-15:
                    continue
                val_w_g = val * w_g

                for h in range(n_group):
                    s_out, ph_h = apply_group_element_fast(
                        new_state, ns, np.int64(h), cg_args, tb_args
                    )
                    idx = repr_map_out[s_out]
                    if idx == _INVALID_REPR_IDX_NB:
                        continue
                    if representative_list_out[idx] != s_out:
                        continue

                    norm_new = normalization_out[idx]
                    if norm_new == 0.0:
                        continue

                    w_h = np.conj(chi_out[h]) * np.conj(ph_h) * inv_n_group / norm_new
                    factor = val_w_g * w_h
                    for b in range(n_batch):
                        vecs_out[idx, b] += factor * vecs_in[k, b]


# @numba.njit(fastmath=True, parallel=True, nogil=True)
@numba.njit(fastmath=True)
def _apply_op_batch_projected_compact_jit(
    vecs_in: np.ndarray,  # (nh_in, n_batch)
    vecs_out: np.ndarray,  # (nh_out, n_batch)
    op_func: Callable,  # (state, *args) -> (new_states, values)
    args: Tuple,
    # Grouped basis and symmetry data
    basis_in_args: Tuple,  # (rep_list_in, norm_in)
    basis_out_args: Tuple,  # (repr_map_out, norm_out, rep_list_out)
    cg_args: Tuple,  # (n_group, n_ops, op_code, arg0, arg1, chi_in, chi_out)
    tb_args: Tuple,  # (trans_perm, trans_cross_mask, refl_perm, inv_perm, parity_axis, boundary_phase)
    ns: np.int64,
    thread_buffers: np.ndarray,
    local_dim: np.int64 = 2,
    chunk_size: int = 6,
) -> None:
    """
    Apply operator in projected compact basis using symmetry group.
    """

    nh_in, n_batch = vecs_in.shape
    nh_out = vecs_out.shape[0]

    # Unpack basis
    representative_list_in, normalization_in = basis_in_args
    repr_map_out, normalization_out, representative_list_out = basis_out_args

    # Unpack cg
    n_group, _, _, _, _, chi_in, chi_out = cg_args

    # Fast path for small Hilbert spaces to avoid parallel overhead
    if nh_in <= _NUMBA_OVERHEAD_SIZE:
        inv_n_group = 1.0 / float(n_group)
        for k in range(nh_in):
            rep = np.int64(representative_list_in[k])
            norm_k = normalization_in[k]
            if norm_k == 0.0:
                continue

            for g in range(n_group):
                s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)
                w_g = np.conj(chi_in[g]) * ph_g / norm_k
                new_states, values = op_func(s, *args)

                for i in range(len(new_states)):
                    new_state = new_states[i]
                    val = values[i]
                    if abs(val) < 1e-15:
                        continue
                    for h in range(n_group):
                        s_out, ph_h = apply_group_element_fast(
                            new_state, ns, np.int64(h), cg_args, tb_args
                        )

                        idx = repr_map_out[s_out]
                        if idx == _INVALID_REPR_IDX_NB:
                            continue
                        if representative_list_out[idx] != s_out:
                            continue

                        norm_new = normalization_out[idx]
                        if norm_new == 0.0:
                            continue

                        w_h = np.conj(chi_out[h]) * np.conj(ph_h) * inv_n_group / norm_new
                        factor = val * w_g * w_h
                        for b in range(n_batch):
                            vecs_out[idx, b] += factor * vecs_in[k, b]
        return

    n_threads = thread_buffers.shape[0]
    chunk_size = min(chunk_size, n_batch)
    bufs = thread_buffers

    if (
        bufs is None
        or bufs.shape[0] < n_threads
        or bufs.shape[1] != nh_out
        or bufs.shape[2] < chunk_size
    ):
        bufs = np.zeros((n_threads, nh_out, chunk_size), dtype=vecs_out.dtype)

    inv_n_group = 1.0 / float(n_group)

    for b_start in range(0, n_batch, chunk_size):
        b_end = min(b_start + chunk_size, n_batch)
        actual_w = b_end - b_start
        bufs[:n_threads, :, :actual_w].fill(0.0)

        for k in numba.prange(nh_in):
            tid = numba.get_thread_id()
            rep = np.int64(representative_list_in[k])
            norm_k = normalization_in[k]
            if norm_k == 0.0:
                continue

            for g in range(n_group):
                s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)
                w_g = np.conj(chi_in[g]) * ph_g / norm_k
                new_states, values = op_func(s, *args)

                for i in range(len(new_states)):
                    new_state = new_states[i]
                    val = values[i]
                    if abs(val) < 1e-15:
                        continue

                    for h in range(n_group):
                        s_out, ph_h = apply_group_element_fast(
                            new_state, ns, np.int64(h), cg_args, tb_args
                        )
                        idx = repr_map_out[s_out]
                        if idx == _INVALID_REPR_IDX_NB:
                            continue
                        if representative_list_out[idx] != s_out:
                            continue

                        norm_new = normalization_out[idx]
                        if norm_new == 0.0:
                            continue

                        w_h = np.conj(chi_out[h]) * np.conj(ph_h) * inv_n_group / norm_new
                        factor = val * w_g * w_h

                        for b in range(actual_w):
                            bufs[tid, idx, b] += factor * vecs_in[k, b_start + b]

        # Reduction
        for row in numba.prange(nh_out):
            for b in range(actual_w):
                sum_val = 0.0
                for t in range(n_threads):
                    sum_val += bufs[t, row, b]
                vecs_out[row, b_start + b] += sum_val
        # Reduction
        for row in numba.prange(nh_out):
            for b in range(actual_w):
                sum_val = 0.0
                for t in range(n_threads):
                    sum_val += bufs[t, row, b]
                vecs_out[row, b_start + b] += sum_val


@numba.njit(fastmath=True, inline="always")
def _apply_fourier_batch_projected_compact_seq_jit(
    vecs_in: np.ndarray,  # (nh_in, n_batch)
    vecs_out: np.ndarray,  # (nh_out, n_batch)
    phases: np.ndarray,  # (ns,) complex128
    op_func: Callable,  # (state, site_idx) -> (new_states, values)
    # Grouped basis and symmetry data
    basis_in_args: Tuple,
    basis_out_args: Tuple,
    cg_args: Tuple,
    tb_args: Tuple,
    ns: np.int64,
) -> None:
    """Sequential Fourier batch operator application in projected compact basis."""
    nh_in, n_batch = vecs_in.shape
    n_sites = len(phases)
    representative_list_in, normalization_in = basis_in_args
    repr_map_out, normalization_out, representative_list_out = basis_out_args
    n_group, _, _, _, _, chi_in, chi_out = cg_args
    inv_n_group = 1.0 / float(n_group)

    for k in range(nh_in):
        rep = np.int64(representative_list_in[k])
        norm_k = normalization_in[k]
        if norm_k == 0.0:
            continue
        inv_norm_k = 1.0 / norm_k

        for g in range(n_group):
            s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)
            w_g = np.conj(chi_in[g]) * ph_g * inv_norm_k

            for site_idx in range(n_sites):
                c_site = phases[site_idx]
                new_states, values = op_func(s, site_idx)
                n_new = len(new_states)

                for i in range(n_new):
                    new_state = new_states[i]
                    val = values[i]
                    fourier_val = val * c_site
                    if abs(fourier_val) < 1e-15:
                        continue
                    fourier_w_g = fourier_val * w_g

                    for h in range(n_group):
                        s_out, ph_h = apply_group_element_fast(
                            new_state, ns, np.int64(h), cg_args, tb_args
                        )
                        idx = repr_map_out[s_out]
                        if idx == _INVALID_REPR_IDX_NB:
                            continue
                        if representative_list_out[idx] != s_out:
                            continue

                        norm_new = normalization_out[idx]
                        if norm_new == 0.0:
                            continue

                        w_h = np.conj(chi_out[h]) * np.conj(ph_h) * inv_n_group / norm_new
                        factor = fourier_w_g * w_h
                        for b in range(n_batch):
                            vecs_out[idx, b] += factor * vecs_in[k, b]


# @numba.njit(fastmath=True, parallel=True, nogil=True)
@numba.njit(fastmath=True)
def _apply_fourier_batch_projected_compact_jit(
    vecs_in: np.ndarray,  # (nh_in, n_batch)
    vecs_out: np.ndarray,  # (nh_out, n_batch)
    phases: np.ndarray,  # (ns,) complex128
    op_func: Callable,  # (state, site_idx) -> (new_states, values)
    # Grouped basis and symmetry data
    basis_in_args: Tuple,
    basis_out_args: Tuple,
    cg_args: Tuple,
    tb_args: Tuple,
    ns: np.int64,
    thread_buffers: np.ndarray,
    local_dim: np.int64 = 2,
    chunk_size: int = 4,
) -> None:
    """
    Apply Fourier operator in projected compact basis using symmetry group.
    """

    nh_in, n_batch = vecs_in.shape
    nh_out = vecs_out.shape[0]
    n_sites = len(phases)

    # Unpack basis
    representative_list_in, normalization_in = basis_in_args
    repr_map_out, normalization_out, representative_list_out = basis_out_args

    # Unpack cg
    n_group, _, _, _, _, chi_in, chi_out = cg_args

    # Fast path for small Hilbert spaces to avoid parallel overhead
    if nh_in <= _NUMBA_OVERHEAD_SIZE:
        inv_n_group = 1.0 / float(n_group)
        for k in range(nh_in):
            rep = np.int64(representative_list_in[k])
            norm_k = normalization_in[k]
            if norm_k == 0.0:
                continue
            for g in range(n_group):
                s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)
                w_g = np.conj(chi_in[g]) * ph_g / norm_k

                # Sum over sites for Fourier
                for site_idx in range(n_sites):
                    c_site = phases[site_idx]
                    new_states, values = op_func(s, site_idx)

                    for i in range(len(new_states)):
                        new_state = new_states[i]
                        val = values[i]
                        fourier_val = val * c_site
                        if abs(fourier_val) < 1e-15:
                            continue

                        for h in range(n_group):
                            s_out, ph_h = apply_group_element_fast(
                                new_state, ns, np.int64(h), cg_args, tb_args
                            )
                            idx = repr_map_out[s_out]
                            if idx == _INVALID_REPR_IDX_NB:
                                continue
                            if representative_list_out[idx] != s_out:
                                continue

                            norm_new = normalization_out[idx]
                            if norm_new == 0.0:
                                continue

                            w_h = np.conj(chi_out[h]) * np.conj(ph_h) * inv_n_group / norm_new
                            factor = fourier_val * w_g * w_h

                            for b in range(n_batch):
                                vecs_out[idx, b] += factor * vecs_in[k, b]
        return

    n_threads = thread_buffers.shape[0]
    chunk_size = min(chunk_size, n_batch)
    bufs = thread_buffers

    # CRITICAL: This should rarely trigger if thread_buffers are pre-allocated properly in matvec_fun
    # If this allocates on every call, memory usage explodes (e.g., 2GB × iterations)
    # See Operator.matvec_fun for proper pre-allocation pattern
    if (
        bufs is None
        or bufs.shape[0] < n_threads
        or bufs.shape[1] != nh_out
        or bufs.shape[2] < chunk_size
    ):
        bufs = np.zeros((n_threads, nh_out, chunk_size), dtype=vecs_out.dtype)

    inv_n_group = 1.0 / float(n_group)

    for b_start in range(0, n_batch, chunk_size):
        b_end = min(b_start + chunk_size, n_batch)
        actual_w = b_end - b_start
        bufs[:n_threads, :, :actual_w].fill(0.0)

        for k in numba.prange(nh_in):
            tid = numba.get_thread_id()
            rep = np.int64(representative_list_in[k])
            norm_k = normalization_in[k]
            if norm_k == 0.0:
                continue

            for g in range(n_group):
                s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)
                w_g = np.conj(chi_in[g]) * ph_g / norm_k

                for site_idx in range(n_sites):
                    c_site = phases[site_idx]
                    new_states, values = op_func(s, site_idx)

                    for i in range(len(new_states)):
                        new_state = new_states[i]
                        val = values[i]
                        fourier_val = val * c_site
                        if abs(fourier_val) < 1e-15:
                            continue

                        for h in range(n_group):
                            s_out, ph_h = apply_group_element_fast(
                                new_state, ns, np.int64(h), cg_args, tb_args
                            )
                            idx = repr_map_out[s_out]
                            if idx == _INVALID_REPR_IDX_NB:
                                continue
                            if representative_list_out[idx] != s_out:
                                continue

                            norm_new = normalization_out[idx]
                            if norm_new == 0.0:
                                continue

                            w_h = np.conj(chi_out[h]) * np.conj(ph_h) * inv_n_group / norm_new
                            factor = fourier_val * w_g * w_h

                            for b in range(actual_w):
                                bufs[tid, idx, b] += factor * vecs_in[k, b_start + b]

        # Reduction
        for row in numba.prange(nh_out):
            for b in range(actual_w):
                sum_val = 0.0
                for t in range(n_threads):
                    sum_val += bufs[t, row, b]
                vecs_out[row, b_start + b] += sum_val


# ----------------------------------------------------------------------
#! Sparse/Dense matrix builders for projected operators
# ----------------------------------------------------------------------


@numba.njit
def _build_sparse_projected_jit(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    data_idx: int,
    op_func: Callable,
    representative_list_in: np.ndarray,
    normalization_in: np.ndarray,
    repr_map_out: np.ndarray,
    normalization_out: np.ndarray,
    representative_list_out: np.ndarray,
    ns: np.int64,
    n_group: np.int64,
    cg_args: Tuple,
    tb_args: Tuple,
    local_dim: np.int64 = 2,
):
    """
    Sparse matrix builder for projected operators.
    """

    nh_in = len(representative_list_in)
    inv_n_group = 1.0 / float(n_group)
    chi_in = cg_args[5]
    chi_out = cg_args[6]

    for k in range(nh_in):
        rep = np.int64(representative_list_in[k])
        norm_k = normalization_in[k]
        if norm_k == 0.0:
            continue

        for g in range(n_group):
            s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)

            w_g = np.conj(chi_in[g]) * ph_g / norm_k

            new_states, values = op_func(s)

            for i in range(len(new_states)):
                new_state = new_states[i]
                val = values[i]
                if np.abs(val) < 1e-15:
                    continue

                for h in range(n_group):
                    s_out, ph_h = apply_group_element_fast(
                        new_state, ns, np.int64(h), cg_args, tb_args
                    )

                    idx = repr_map_out[s_out]
                    if idx == _INVALID_REPR_IDX_NB:
                        continue

                    if representative_list_out[idx] != s_out:
                        continue

                    norm_new = normalization_out[idx]
                    if norm_new == 0.0:
                        continue

                    w_h = np.conj(chi_out[h]) * ph_h * inv_n_group / norm_new
                    factor = val * w_g * w_h

                    if np.abs(factor) < 1e-15:
                        continue

                    rows[data_idx] = idx
                    cols[data_idx] = k
                    data[data_idx] = factor
                    data_idx += 1

    return data_idx


@numba.njit(parallel=True)
def _build_dense_projected_jit(
    matrix: np.ndarray,
    op_func: Callable,
    representative_list_in: np.ndarray,
    normalization_in: np.ndarray,
    repr_map_out: np.ndarray,
    normalization_out: np.ndarray,
    representative_list_out: np.ndarray,
    ns: np.int64,
    n_group: np.int64,
    cg_args: Tuple,
    tb_args: Tuple,
    local_dim: np.int64 = 2,
):
    """
    Dense matrix builder for projected operators.
    """

    nh_in = len(representative_list_in)
    inv_n_group = 1.0 / float(n_group)
    chi_in = cg_args[5]
    chi_out = cg_args[6]

    if nh_in <= _NUMBA_OVERHEAD_SIZE:
        for k in range(nh_in):
            rep = np.int64(representative_list_in[k])
            norm_k = normalization_in[k]
            if norm_k == 0.0:
                continue

            for g in range(n_group):
                s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)

                w_g = np.conj(chi_in[g]) * ph_g / norm_k

                new_states, values = op_func(s)

                for i in range(len(new_states)):
                    new_state = new_states[i]
                    val = values[i]
                    if np.abs(val) < 1e-15:
                        continue

                    for h in range(n_group):
                        s_out, ph_h = apply_group_element_fast(
                            new_state, ns, np.int64(h), cg_args, tb_args
                        )

                        idx = repr_map_out[s_out]
                        if idx == _INVALID_REPR_IDX_NB:
                            continue

                        if representative_list_out[idx] != s_out:
                            continue

                        norm_new = normalization_out[idx]
                        if norm_new == 0.0:
                            continue

                        w_h = np.conj(chi_out[h]) * ph_h * inv_n_group / norm_new
                        factor = val * w_g * w_h

                        if np.abs(factor) < 1e-15:
                            continue

                        matrix[idx, k] += factor
    else:
        for k in numba.prange(nh_in):
            rep = np.int64(representative_list_in[k])
            norm_k = normalization_in[k]
            if norm_k == 0.0:
                continue

            for g in range(n_group):
                s, ph_g = apply_group_element_fast(rep, ns, np.int64(g), cg_args, tb_args)

                w_g = np.conj(chi_in[g]) * ph_g / norm_k

                new_states, values = op_func(s)

                for i in range(len(new_states)):
                    new_state = new_states[i]
                    val = values[i]
                    if np.abs(val) < 1e-15:
                        continue

                    for h in range(n_group):
                        s_out, ph_h = apply_group_element_fast(
                            new_state, ns, np.int64(h), cg_args, tb_args
                        )

                        idx = repr_map_out[s_out]
                        if idx == _INVALID_REPR_IDX_NB:
                            continue

                        if representative_list_out[idx] != s_out:
                            continue

                        norm_new = normalization_out[idx]
                        if norm_new == 0.0:
                            continue

                        w_h = np.conj(chi_out[h]) * ph_h * inv_n_group / norm_new
                        factor = val * w_g * w_h

                        if np.abs(factor) < 1e-15:
                            continue

                        matrix[idx, k] += factor


# ----------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------
