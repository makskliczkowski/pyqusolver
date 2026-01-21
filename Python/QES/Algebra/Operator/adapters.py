"""
Backend adapters for Operator operations.

This module isolates backend-specific logic (NumPy/Numba vs JAX) for
applying operators to states and vectors.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List

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
    def matvec(self,
               operator: 'Operator',
               vecs: Any,
               hilbert_in: 'HilbertSpace',
               hilbert_out: Optional['HilbertSpace'] = None,
               *args,
               **kwargs) -> Any:
        """Apply operator to a vector of coefficients."""
        pass

    @abstractmethod
    def apply(self, operator: 'Operator', states: Any, *args) -> Any:
        """Apply operator to a state or list of states."""
        pass

    @abstractmethod
    def matvec_fourier(self,
                       operator: 'Operator',
                       phases: Any,
                       vec: Any,
                       hilbert: 'HilbertSpace',
                       *args,
                       **kwargs) -> Any:
        """Apply Fourier operator O_q."""
        pass

class NumpyAdapter(BackendAdapter):
    """Adapter for NumPy/Numba backend."""

    def matvec(self,
               operator: 'Operator',
               vecs: np.ndarray,
               hilbert_in: 'HilbertSpace',
               hilbert_out: Optional['HilbertSpace'] = None,
               *args,
               symmetry_mode: str = "auto",
               multithreaded: bool = False,
               out: Optional[np.ndarray] = None,
               thread_buffer: Optional[np.ndarray] = None,
               chunk_size: int = 1,
               dtype: Optional[Any] = None,
               **kwargs) -> np.ndarray:

        # Import Numba kernels here to avoid circular imports if any
        try:
            from QES.Algebra.Hilbert.matrix_builder import (
                ensure_thread_buffer, canonicalize_args,
                _apply_op_batch_jit, _apply_op_batch_compact_jit,
                _apply_op_batch_seq_jit, _apply_op_batch_compact_seq_jit
            )
            from QES.Algebra.Symmetries.jit.matrix_builder_jit import (
                _apply_op_batch_projected_compact_jit,
                _apply_op_batch_projected_compact_seq_jit
            )
        except ImportError as e:
            raise ImportError(f"JIT matrix builder not available: {e}")

        # Determine Target Space
        target_space    = hilbert_out       if hilbert_out  is not None else hilbert_in
        nh              = target_space.nh   if target_space is not None else vecs.shape[0]
        nhfull          = hilbert_in.nhfull if hilbert_in   is not None else vecs.shape[0]

        # Ensure 2D shape for batch kernel: (N_hilbert, N_batch)
        is_1d = vecs.ndim == 1
        vecs_in = vecs[:, np.newaxis] if is_1d else vecs

        if hilbert_in is not None and vecs_in.shape[0] != nh:
            raise ValueError(f"Input vector size {vecs_in.shape[0]} does not match Hilbert space dimension {hilbert_in.nh}.")

        # output
        dtype_use = dtype if dtype is not None else vecs_in.dtype
        vecs_out = out if out is not None else np.zeros((nh, vecs_in.shape[1]), dtype=dtype_use)

        # Check for CompactSymmetryData (O(1) lookup - fast)
        compact_data_out = getattr(target_space, 'compact_symmetry_data', None)
        compact_data_in  = getattr(hilbert_in,   'compact_symmetry_data', None)

        # Logic to decide which kernel to use
        is_same_sector       = (hilbert_out is None or hilbert_out == hilbert_in)
        force_project        = symmetry_mode.startswith('p') and nh != nhfull

        use_projected_kernel = (compact_data_out is not None and compact_data_in is not None) and (force_project or not is_same_sector)
        use_fast_kernel      = (compact_data_out is not None and is_same_sector and not force_project and symmetry_mode in ("auto", "fast"))

        # Operator function and args
        # NOTE: Operator must provide _fun_int for Numba backend
        op_func = operator._fun._fun_int
        if op_func is None:
             raise ValueError("Integer function for the operator is not defined (required for Numba backend).")

        # Handling args passed to matvec (e.g. for local operators)
        op_args = canonicalize_args(args)

        if not multithreaded:
            # SEQUENTIAL PATH
            if use_fast_kernel and nhfull != hilbert_in.nh:
                cd          = compact_data_out
                basis_args  = (cd.representative_list, cd.normalization, cd.repr_map, cd.phase_idx, cd.phase_table)
                _apply_op_batch_compact_seq_jit(vecs_in, vecs_out, op_func, op_args, basis_args)

            elif use_projected_kernel and nhfull != hilbert_in.nh:
                sc      = hilbert_in.sym_container
                sc_out  = target_space.sym_container if getattr(target_space, 'sym_container', None) is not None else sc
                cg      = sc.compiled_group

                basis_in_args  = (compact_data_in.representative_list, compact_data_in.normalization)
                basis_out_args = (compact_data_out.repr_map, compact_data_out.normalization, compact_data_out.representative_list)
                cg_args        = (cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, sc_out.compiled_group.chi)

                _apply_op_batch_projected_compact_seq_jit(
                    vecs_in, vecs_out, op_func, op_args,
                    basis_in_args, basis_out_args,
                    cg_args, sc.tables.args,
                    np.int64(hilbert_in.ns))
            else:
                _apply_op_batch_seq_jit(vecs_in, vecs_out, op_func, op_args)

        else:
            # MULTITHREADED PATH
            chunk_size_use = min(chunk_size, vecs_in.shape[1])
            th_buffer = ensure_thread_buffer(nh, chunk_size_use, dtype=dtype_use, thread_buffers=thread_buffer)

            if use_fast_kernel and nhfull != hilbert_in.nh:
                cd          = compact_data_out
                basis_args  = (cd.representative_list, cd.normalization, cd.repr_map, cd.phase_idx, cd.phase_table)
                _apply_op_batch_compact_jit(vecs_in, vecs_out, op_func, op_args, basis_args, th_buffer, chunk_size_use)

            elif use_projected_kernel and nhfull != hilbert_in.nh:
                sc      = hilbert_in.sym_container
                sc_out  = target_space.sym_container if getattr(target_space, 'sym_container', None) is not None else sc

                cg      = sc.compiled_group
                cg_args = (cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, sc_out.compiled_group.chi)

                basis_in_args  = (compact_data_in.representative_list, compact_data_in.normalization)
                basis_out_args = (compact_data_out.repr_map, compact_data_out.normalization, compact_data_out.representative_list)

                _apply_op_batch_projected_compact_jit(
                    vecs_in, vecs_out, op_func, op_args,
                    basis_in_args, basis_out_args,
                    cg_args, sc.tables.args,
                    np.int64(hilbert_in.ns),
                    th_buffer,
                    local_dim=np.int64(hilbert_in.local_space.local_dim),
                    chunk_size=chunk_size_use)

            else:
                _apply_op_batch_jit(vecs_in, vecs_out, op_func, op_args, th_buffer, chunk_size_use)

        return vecs_out.ravel() if is_1d else vecs_out

    def matvec_fourier(self,
                       operator: 'Operator',
                       phases: np.ndarray,
                       vec: np.ndarray,
                       hilbert: 'HilbertSpace',
                       symmetry_mode: str = "auto",
                       multithreaded: bool = False,
                       out: Optional[np.ndarray] = None,
                       thread_buffer: Optional[np.ndarray] = None,
                       chunk_size: int = 4,
                       **kwargs) -> np.ndarray:

        try:
            from QES.Algebra.Hilbert.matrix_builder import (
                _apply_fourier_batch_jit, _apply_fourier_batch_compact_jit, ensure_thread_buffer,
                _apply_fourier_batch_seq_jit, _apply_fourier_batch_compact_seq_jit
            )
            from QES.Algebra.Symmetries.jit.matrix_builder_jit import (
                _apply_fourier_batch_projected_compact_seq_jit
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

        compact_data = getattr(hilbert, 'compact_symmetry_data', None)
        symmetry_mode = symmetry_mode.lower() if symmetry_mode is not None else "auto"

        force_project = symmetry_mode.startswith('p') and hilbert.nh != hilbert.nhfull
        use_projected_kernel = (compact_data is not None) and force_project
        use_fast_kernel = (compact_data is not None) and (not force_project and symmetry_mode in ("auto", "fast"))

        op_func = operator._fun._fun_int

        if not multithreaded:
            # SEQUENTIAL PATH
            if use_fast_kernel and (hilbert.nhfull != hilbert.nh):
                cd                  = compact_data
                basis_args          = (cd.representative_list, cd.normalization, cd.repr_map, cd.phase_idx, cd.phase_table)
                _apply_fourier_batch_compact_seq_jit(vecs_in, vecs_out, phases, op_func, basis_args)

            elif use_projected_kernel and (hilbert.nhfull != hilbert.nh):
                sc                  = hilbert.sym_container
                ns                  = np.int64(hilbert.ns)
                cg                  = sc.compiled_group
                basis_in_args       = (compact_data.representative_list, compact_data.normalization)
                basis_out_args      = (compact_data.repr_map, compact_data.normalization, compact_data.representative_list)
                cg_args             = (cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, cg.chi)
                _apply_fourier_batch_projected_compact_seq_jit(vecs_in, vecs_out, phases, op_func, basis_in_args, basis_out_args, cg_args, sc.tables.args, ns)

            else:
                _apply_fourier_batch_seq_jit(vecs_in, vecs_out, phases, op_func)

        else:
            # MULTITHREADED PATH
            chunk_size_use = min(chunk_size, n_batch)
            th_buffer = ensure_thread_buffer(nh, chunk_size_use, dtype=np.complex128, thread_buffers=thread_buffer)

            if use_fast_kernel and (hilbert.nhfull != hilbert.nh):
                # Fast path with compact data
                cd                  = compact_data
                basis_args          = (cd.representative_list, cd.normalization, cd.repr_map, cd.phase_idx, cd.phase_table)

                _apply_fourier_batch_compact_jit(
                                    vecs_in, vecs_out, phases, op_func, basis_args,
                                    th_buffer, chunk_size_use
                                    )

            elif use_projected_kernel and (hilbert.nhfull != hilbert.nh):
                # Projected path
                try:
                    from QES.Algebra.Symmetries.jit.matrix_builder_jit import _apply_fourier_batch_projected_compact_jit
                except ImportError:
                    raise ImportError("JIT projected compact matrix builder not available.")

                sc = hilbert.sym_container
                basis_in_args       = (compact_data.representative_list, compact_data.normalization)
                basis_out_args      = (compact_data.repr_map, compact_data.normalization, compact_data.representative_list)
                cg                  = sc.compiled_group
                cg_args             = (cg.n_group, cg.n_ops, cg.op_code, cg.arg0, cg.arg1, cg.chi, cg.chi)

                _apply_fourier_batch_projected_compact_jit(
                    vecs_in, vecs_out, phases, op_func,
                    basis_in_args, basis_out_args, cg_args, sc.tables.args,
                    np.int64(hilbert.ns), th_buffer,
                    np.int64(hilbert.local_space.local_dim), chunk_size_use,
                )
            else:
                # Fallback path (no symmetry or full space)
                _apply_fourier_batch_jit(vecs_in, vecs_out, phases, op_func, th_buffer, chunk_size_use)

        return vecs_out.ravel() if is_1d else vecs_out

    def apply(self, operator: 'Operator', states: Any, *args) -> Any:
        """Apply operator to state(s) using NumPy backend."""
        # This delegates back to Operator._apply_... methods which use _fun_int or _fun_np
        # Ideally we move logic here, but Operator._apply methods handle dispatching based on acting type.
        # For now, let's keep it simple: we assume apply() logic in Operator is fine,
        # or we implement specific apply logic if needed.
        # The Operator.apply() method does backend check itself.
        # If we refactor Operator to use adapter, this method should implement the logic.

        # Currently Operator.apply uses self._fun(states).
        # We can just return that.
        return operator._fun(states, *args)


class JaxAdapter(BackendAdapter):
    """Adapter for JAX backend."""

    def matvec(self,
               operator: 'Operator',
               vecs: Any,
               hilbert_in: 'HilbertSpace',
               hilbert_out: Optional['HilbertSpace'] = None,
               *args,
               **kwargs) -> Any:

        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available.")

        # Placeholder for JAX implementation
        # Assuming current Operator.matvec falls back to something or raises error for JAX if not implemented?
        # Operator.matvec implementation I saw earlier was heavily Numba centric.
        # It had `if self._is_jax`?
        # Let's check Operator.matvec again.

        # Operator.matvec in operator.py seems to force usage of Numba kernels (`_apply_op_batch_jit`).
        # It handles `vecs` as array.
        # If vecs is JAX array, Numba kernels might fail or force conversion.
        # There was no explicit JAX path in Operator.matvec in the file I read!
        # It seems JAX support for matvec might be missing or limited to conversion.

        # If I am to implement JAX adapter, I should use JAX primitives.
        # But if the kernels are not available, I can't fully implement it now.
        # I will leave it as a TODO or basic implementation if possible.

        raise NotImplementedError("JAX matvec adapter not yet implemented.")

    def matvec_fourier(self, *args, **kwargs) -> Any:
        raise NotImplementedError("JAX matvec_fourier adapter not yet implemented.")

    def apply(self, operator: 'Operator', states: Any, *args) -> Any:
        # Use JAX function if available
        if operator._fun.jax is not None:
            return operator._fun.jax(states, *args)
        return operator._fun(states, *args)
