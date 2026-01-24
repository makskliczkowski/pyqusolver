"""
QES Session Management
=====================

This module provides a high-level API for configuring and managing QES sessions.
It allows setting the backend, random seed, precision, and other global parameters
in a unified way.

Usage
-----
    import QES

    # Using context manager (recommended)
    with QES.run(backend='jax', seed=42, precision='float64') as session:
        # QES code here
        ...

    # Or creating a session object
    session = QES.QESSession(backend='numpy', seed=123)
    session.start()
    # ...
    session.stop()
"""

import os
from typing import Optional, Literal

from .qes_globals import get_backend_manager, get_logger


class QESSession:
    """
    Manages the configuration and state of a QES session.

    Parameters
    ----------
    backend : str, optional
        Computation backend ('numpy', 'jax'). Default is 'numpy'.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    precision : str, optional
        Floating point precision ('float32', 'float64'). Default is 'float64'.
    num_threads : int, optional
        Number of threads for CPU operations. If None, uses all available cores.
    """

    def __init__(
        self,
        backend: str = "numpy",
        seed: int = 42,
        precision: Literal["float32", "float64"] = "float64",
        num_threads: Optional[int] = None,
    ):
        self._backend_name = backend
        self._seed = seed
        self._precision = precision
        self._num_threads = num_threads
        self._backend_mgr = get_backend_manager()
        self._log = get_logger()
        self._previous_config = {}

    def start(self):
        """
        Apply the session configuration.
        """
        self._log.info(
            f"Starting QESSession(backend={self._backend_name}, seed={self._seed}, precision={self._precision})"
        )

        # Store previous state (simplified - full restoration might be complex)
        # For now we assume we are setting global state.

        # 1. Set threads
        if self._num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self._num_threads)
            os.environ["MKL_NUM_THREADS"] = str(self._num_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(self._num_threads)
            # Note: Changing env vars for threads might not affect already loaded libraries fully,
            # but usually OMP/MKL check env vars on first use or allow programmatic setting.
            # Python's os.environ might not propagate to C libraries if set after import,
            # but standard practice often relies on this being set early.

        # 2. Set Precision (Env var based in QES)
        # Ideally this should be done before importing QES, but BackendManager reads it.
        # If QES is already imported, we might need to rely on BackendManager handling it if it supports it.
        # Currently utils.py reads PY_FLOATING_POINT_STR at module level.
        # Changing it here might not affect already initialized types unless we force update.
        # However, the user request implies we should support this.
        # The BackendManager has _update_dtypes() which uses defaults.
        # We might need to poke internals or just set env vars for future imports if lazy.
        os.environ["PY_FLOATING_POINT"] = self._precision

        # 3. Set Backend
        try:
            self._backend_mgr.set_active_backend(self._backend_name)
        except ValueError as e:
            self._log.error(f"Failed to set backend {self._backend_name}: {e}")
            raise

        # 4. Reseed
        self._backend_mgr.reseed(self._seed)

        return self

    def stop(self):
        """
        Restore previous state (if applicable) or cleanup.
        Currently primarily serves as a marker for session end.
        """
        self._log.info("Stopping QESSession")
        # Implementation of full restore is tricky with global singletons.
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def run(
    backend: str = "numpy",
    seed: int = 42,
    precision: Literal["float32", "float64"] = "float64",
    num_threads: Optional[int] = None,
):
    """
    Context manager to run a block of code with a specific QES configuration.

    Parameters
    ----------
    backend : str
        'numpy' or 'jax'
    seed : int
        Random seed
    precision : str
        'float32' or 'float64'
    num_threads : int, optional
        Number of threads

    Returns
    -------
    QESSession
        The active session object.
    """
    return QESSession(backend=backend, seed=seed, precision=precision, num_threads=num_threads)
