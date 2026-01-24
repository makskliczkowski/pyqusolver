"""
Backend adapters for Operator operations.

This module isolates backend-specific logic (NumPy/Numba vs JAX) for
applying operators to states and vectors.

------------------------------------------------------------------------
File        : Algebra/Operator/adapters.py
Author      : Maksymilian Kliczkowski
Date        : 2025-12-01
------------------------------------------------------------------------
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional

# Try importing JAX
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


class BackendAdapter(ABC):
    """Abstract base class for backend adapters."""

    @abstractmethod
    def matvec(
        self,
        operator: "Operator",
        vecs: Any,
        hilbert_in: "HilbertSpace",
        hilbert_out: Optional["HilbertSpace"] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Apply operator to a vector of coefficients.

        Parameters:
            operator (Operator):
                The operator instance.
            vecs (Array):
                The input vectors to which the operator is applied. Shape (vec_size, n_vecs).
            hilbert_in (HilbertSpace):
                The Hilbert space of the input vectors.
            hilbert_out (HilbertSpace):
                The Hilbert space of the output vectors (target sector).
            *args:
                Additional arguments for the operator function.
            **kwargs:
                Additional keyword arguments (e.g. symmetry_mode, multithreaded).
        """
        pass

    @abstractmethod
    def apply(self, operator: "Operator", states: Any, *args) -> Any:
        """Apply operator to a state or list of states."""
        pass

    @abstractmethod
    def matvec_fourier(
        self, operator: "Operator", phases: Any, vec: Any, hilbert: "HilbertSpace", *args, **kwargs
    ) -> Any:
        """Apply Fourier operator O_q."""
        pass


class NumpyAdapter(BackendAdapter):
    """Adapter for NumPy/Numba backend."""

    def matvec(
        self,
        operator: "Operator",
        vecs: np.ndarray,
        hilbert_in: "HilbertSpace",
        hilbert_out: Optional["HilbertSpace"] = None,
        *args,
        symmetry_mode: str = "auto",
        multithreaded: bool = False,
        out: Optional[np.ndarray] = None,
        thread_buffer: Optional[np.ndarray] = None,
        chunk_size: int = 1,
        dtype: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply the operator matrix to a vector using Numba kernels.

        Parameters:
            vecs (Array):
                The input vectors to which the operator is applied. Shape (vec_size, n_vecs).

                vec_size can be:
                    - 1, nstates:           -> n integer states
                    - ns * nloc, n_vecs:    -> local Hilbert space vectors - usually basis vectors
                    - hilbert.nh, n_vecs:   -> full Hilbert space vectors
            hilbert_in (HilbertSpace):
                The Hilbert space of the input vectors.
            hilbert_out (HilbertSpace):
                The Hilbert space of the output vectors (target sector).
                If None, assumes same as input (hilbert).
            symmetry_mode (str):
                - "auto"                    -> use compact fast if possible, else fallback
                - "fast"                    -> assume operator preserves symmetry sector (commuting case)
                - "project"                 -> compute y = P O P x (always correct, slower)
                - "none"                    -> ignore symmetries (explicit basis / full space)
            *args:
                Additional arguments for the operator function.
            out (Array):
                Preallocated output array to store the result. If None, a new array is created.
            thread_buffer (Array):
                Preallocated thread buffer for JIT kernels. If None, a new buffer is created internally.
            chunk_size (int):
                Chunk size for JIT kernels. Larger sizes may improve performance.
            dtype:
                Desired data type for the output array. If None, defaults to complex128.

        Returns:
            Array: The resulting vectors after applying the operator.
        """

        # Import Numba kernels here to avoid circular imports if any
        try:
            from QES.Algebra.Hilbert.matrix_builder import (
                ensure_thread_buffer,
                canonicalize_args,
                _apply_op_batch_jit,
                _apply_op_batch_compact_jit,
                _apply_op_batch_seq_jit,
                _apply_op_batch_compact_seq_jit,
            )
            from QES.Algebra.Symmetries.jit.matrix_builder_jit import (
                _apply_op_batch_projected_compact_jit,
                _apply_op_batch_projected_compact_seq_jit,
            )
        except ImportError as e:
            raise ImportError(f"JIT matrix builder not available: {e}")

        # Determine Target Space
        target_space = hilbert_out if hilbert_out is not None else hilbert_in
        nh = target_space.nh if target_space is not None else vecs.shape[0]
        nhfull = hilbert_in.nhfull if hilbert_in is not None else vecs.shape[0]

        # Ensure 2D shape for batch kernel: (N_hilbert, N_batch)
        is_1d = vecs.ndim == 1
        vecs_in = vecs[:, np.newaxis] if is_1d else vecs

        if hilbert_in is not None and vecs_in.shape[0] != nh:
            raise ValueError(
                f"Input vector size {vecs_in.shape[0]} does not match Hilbert space dimension {hilbert_in.nh}."
            )

        # output
        dtype_use = dtype if dtype is not None else vecs_in.dtype
        vecs_out = out if out is not None else np.zeros((nh, vecs_in.shape[1]), dtype=dtype_use)

        # Check for CompactSymmetryData (O(1) lookup - fast)
        compact_data_out = getattr(target_space, "compact_symmetry_data", None)
        compact_data_in = getattr(hilbert_in, "compact_symmetry_data", None)

        # Logic to decide which kernel to use
        is_same_sector = hilbert_out is None or hilbert_out == hilbert_in
        force_project = symmetry_mode.startswith("p") and nh != nhfull

        use_projected_kernel = (compact_data_out is not None and compact_data_in is not None) and (
            force_project or not is_same_sector
        )
        use_fast_kernel = (
            compact_data_out is not None
            and is_same_sector
            and not force_project
            and symmetry_mode in ("auto", "fast")
        )

        # Operator function and args
        # NOTE: Operator must provide _fun_int for Numba backend
        op_func = operator._fun._fun_int
        if op_func is None:
            raise ValueError(
                "Integer function for the operator is not defined (required for Numba backend)."
            )

        # Handling args passed to matvec (e.g. for local operators)
        op_args = canonicalize_args(args)

        if not multithreaded:
            # SEQUENTIAL PATH
            if use_fast_kernel and nhfull != hilbert_in.nh:
                cd = compact_data_out
                basis_args = (
                    cd.representative_list,
                    cd.normalization,
                    cd.repr_map,
                    cd.phase_idx,
                    cd.phase_table,
                )
                _apply_op_batch_compact_seq_jit(vecs_in, vecs_out, op_func, op_args, basis_args)

            elif use_projected_kernel and nhfull != hilbert_in.nh:
                sc = hilbert_in.sym_container
                sc_out = (
                    target_space.sym_container
                    if getattr(target_space, "sym_container", None) is not None
                    else sc
                )
                cg = sc.compiled_group

                basis_in_args = (compact_data_in.representative_list, compact_data_in.normalization)
                basis_out_args = (
                    compact_data_out.repr_map,
                    compact_data_out.normalization,
                    compact_data_out.representative_list,
                )
                cg_args = (
                    cg.n_group,
                    cg.n_ops,
                    cg.op_code,
                    cg.arg0,
                    cg.arg1,
                    cg.chi,
                    sc_out.compiled_group.chi,
                )

                _apply_op_batch_projected_compact_seq_jit(
                    vecs_in,
                    vecs_out,
                    op_func,
                    op_args,
                    basis_in_args,
                    basis_out_args,
                    cg_args,
                    sc.tables.args,
                    np.int64(hilbert_in.ns),
                )
            else:
                _apply_op_batch_seq_jit(vecs_in, vecs_out, op_func, op_args)

        else:
            # MULTITHREADED PATH
            chunk_size_use = min(chunk_size, vecs_in.shape[1])
            th_buffer = ensure_thread_buffer(
                nh, chunk_size_use, dtype=dtype_use, thread_buffers=thread_buffer
            )

            if use_fast_kernel and nhfull != hilbert_in.nh:
                cd = compact_data_out
                basis_args = (
                    cd.representative_list,
                    cd.normalization,
                    cd.repr_map,
                    cd.phase_idx,
                    cd.phase_table,
                )
                _apply_op_batch_compact_jit(
                    vecs_in, vecs_out, op_func, op_args, basis_args, th_buffer, chunk_size_use
                )

            elif use_projected_kernel and nhfull != hilbert_in.nh:
                sc = hilbert_in.sym_container
                sc_out = (
                    target_space.sym_container
                    if getattr(target_space, "sym_container", None) is not None
                    else sc
                )

                cg = sc.compiled_group
                cg_args = (
                    cg.n_group,
                    cg.n_ops,
                    cg.op_code,
                    cg.arg0,
                    cg.arg1,
                    cg.chi,
                    sc_out.compiled_group.chi,
                )

                basis_in_args = (compact_data_in.representative_list, compact_data_in.normalization)
                basis_out_args = (
                    compact_data_out.repr_map,
                    compact_data_out.normalization,
                    compact_data_out.representative_list,
                )

                _apply_op_batch_projected_compact_jit(
                    vecs_in,
                    vecs_out,
                    op_func,
                    op_args,
                    basis_in_args,
                    basis_out_args,
                    cg_args,
                    sc.tables.args,
                    np.int64(hilbert_in.ns),
                    th_buffer,
                    local_dim=np.int64(hilbert_in.local_space.local_dim),
                    chunk_size=chunk_size_use,
                )

            else:
                _apply_op_batch_jit(vecs_in, vecs_out, op_func, op_args, th_buffer, chunk_size_use)

        return vecs_out.ravel() if is_1d else vecs_out

    def matvec_fourier(
        self,
        operator: "Operator",
        phases: np.ndarray,
        vec: np.ndarray,
        hilbert: "HilbertSpace",
        symmetry_mode: str = "auto",
        multithreaded: bool = False,
        out: Optional[np.ndarray] = None,
        thread_buffer: Optional[np.ndarray] = None,
        chunk_size: int = 4,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Computes |out> = O_q |in> without constructing the matrix O_q.
        O_q = (1/sqrt(N)) * sum_j exp(i * k * r_j) * sigma_j

        Parameters:
        -----------
        phases (Array):
            The phase factors exp(i * k * r_j) for each site.
        vec (Array):
            The input state vector |in>.
        hilbert (HilbertSpace):
            The Hilbert space for the system.
        symmetry_mode (str):
            - "auto"                    -> use compact fast if possible, else fallback
            - "fast"                    -> assume operator preserves symmetry sector (commuting case)
            - "project"                 -> compute y = P O P x (always correct, slower)
            - "none"                    -> ignore symmetries (explicit basis / full space)
        multithreaded (bool):
            If True, use multi-threaded execution.
        out (Array, optional):
            Preallocated output array for |out>. If None, a new array is created.
        thread_buffer (Array, optional):
            Buffer for thread-local storage to optimize performance.
        chunk_size (int, optional):
            Number of vectors to process in each chunk for performance optimization. Default is 4.
        Returns:
        --------
        Array:
            The resulting state vector |out> after applying the Fourier operator.
        """

        try:
            from QES.Algebra.Hilbert.matrix_builder import (
                _apply_fourier_batch_jit,
                _apply_fourier_batch_compact_jit,
                ensure_thread_buffer,
                _apply_fourier_batch_seq_jit,
                _apply_fourier_batch_compact_seq_jit,
            )
            from QES.Algebra.Symmetries.jit.matrix_builder_jit import (
                _apply_fourier_batch_projected_compact_seq_jit,
            )
        except ImportError as e:
            raise ImportError(f"JIT matrix builder not available: {e}")

        is_1d = vec.ndim == 1
        vecs_in = vec[:, np.newaxis] if is_1d else vec

        # output
        vecs_out = np.zeros_like(vecs_in, dtype=np.complex128) if out is None else out
        nh, n_batch = vecs_in.shape

        if not operator.type_acting.is_local():
            raise ValueError("Fourier operator application requires a local operator.")

        compact_data = getattr(hilbert, "compact_symmetry_data", None)
        symmetry_mode = symmetry_mode.lower() if symmetry_mode is not None else "auto"

        force_project = symmetry_mode.startswith("p") and hilbert.nh != hilbert.nhfull
        use_projected_kernel = (compact_data is not None) and force_project
        use_fast_kernel = (compact_data is not None) and (
            not force_project and symmetry_mode in ("auto", "fast")
        )

        op_func = operator._fun._fun_int

        if not multithreaded:
            # SEQUENTIAL PATH
            if use_fast_kernel and (hilbert.nhfull != hilbert.nh):
                cd = compact_data
                basis_args = (
                    cd.representative_list,
                    cd.normalization,
                    cd.repr_map,
                    cd.phase_idx,
                    cd.phase_table,
                )
                _apply_fourier_batch_compact_seq_jit(vecs_in, vecs_out, phases, op_func, basis_args)

            elif use_projected_kernel and (hilbert.nhfull != hilbert.nh):
                sc = hilbert.sym_container
                ns = np.int64(hilbert.ns)
                cg = sc.compiled_group
                basis_in_args = (compact_data.representative_list, compact_data.normalization)
                basis_out_args = (
                    compact_data.repr_map,
                    compact_data.normalization,
                    compact_data.representative_list,
                )
                cg_args = (cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, cg.chi)
                _apply_fourier_batch_projected_compact_seq_jit(
                    vecs_in,
                    vecs_out,
                    phases,
                    op_func,
                    basis_in_args,
                    basis_out_args,
                    cg_args,
                    sc.tables.args,
                    ns,
                )

            else:
                _apply_fourier_batch_seq_jit(vecs_in, vecs_out, phases, op_func)

        else:
            # MULTITHREADED PATH
            chunk_size_use = min(chunk_size, n_batch)
            th_buffer = ensure_thread_buffer(
                nh, chunk_size_use, dtype=np.complex128, thread_buffers=thread_buffer
            )

            if use_fast_kernel and (hilbert.nhfull != hilbert.nh):
                # Fast path with compact data
                cd = compact_data
                basis_args = (
                    cd.representative_list,
                    cd.normalization,
                    cd.repr_map,
                    cd.phase_idx,
                    cd.phase_table,
                )

                _apply_fourier_batch_compact_jit(
                    vecs_in, vecs_out, phases, op_func, basis_args, th_buffer, chunk_size_use
                )

            elif use_projected_kernel and (hilbert.nhfull != hilbert.nh):
                # Projected path
                try:
                    from QES.Algebra.Symmetries.jit.matrix_builder_jit import (
                        _apply_fourier_batch_projected_compact_jit,
                    )
                except ImportError:
                    raise ImportError("JIT projected compact matrix builder not available.")

                sc = hilbert.sym_container
                basis_in_args = (compact_data.representative_list, compact_data.normalization)
                basis_out_args = (
                    compact_data.repr_map,
                    compact_data.normalization,
                    compact_data.representative_list,
                )
                cg = sc.compiled_group
                cg_args = (cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, cg.chi)

                _apply_fourier_batch_projected_compact_jit(
                    vecs_in,
                    vecs_out,
                    phases,
                    op_func,
                    basis_in_args,
                    basis_out_args,
                    cg_args,
                    sc.tables.args,
                    np.int64(hilbert.ns),
                    th_buffer,
                    np.int64(hilbert.local_space.local_dim),
                    chunk_size_use,
                )
            else:
                # Fallback path (no symmetry or full space)
                _apply_fourier_batch_jit(
                    vecs_in, vecs_out, phases, op_func, th_buffer, chunk_size_use
                )

        return vecs_out.ravel() if is_1d else vecs_out

    def apply(self, operator: "Operator", states: Any, *args) -> Any:
        """Apply operator to state(s) using NumPy backend."""
        # Delegates to operator's callable interface
        return operator._fun(states, *args)


class JaxAdapter(BackendAdapter):
    """Adapter for JAX backend."""

    def matvec(
        self,
        operator: "Operator",
        vecs: Any,
        hilbert_in: "HilbertSpace",
        hilbert_out: Optional["HilbertSpace"] = None,
        *args,
        **kwargs,
    ) -> Any:

        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available.")

        raise NotImplementedError("JAX matvec adapter not yet implemented.")

    def matvec_fourier(self, *args, **kwargs) -> Any:
        raise NotImplementedError("JAX matvec_fourier adapter not yet implemented.")

    def apply(self, operator: "Operator", states: Any, *args) -> Any:
        # Use JAX function if available
        if operator._fun.jax is not None:
            return operator._fun.jax(states, *args)
        return operator._fun(states, *args)
