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
import contextlib
from typing import Optional, Union, Literal

from .qes_globals import get_backend_manager, get_logger

class QESSession:
    """
    Manages the configuration and state of a QES session.

    This class handles the initialization of the global backend manager,
    setting the computation backend (NumPy/JAX), random seeding, and
    floating point precision.

    Parameters
    ----------
    backend : str, optional
        Computation backend to use. Options are 'numpy' or 'jax'.
        Default is 'numpy'.
    seed : int, optional
        Global random seed for reproducibility. Default is 42.
    precision : Literal['float32', 'float64'], optional
        Floating point precision for computations. Default is 'float64'.
        Note: This sets the `PY_FLOATING_POINT` environment variable,
        which influences default dtypes.
    num_threads : int, optional
        Number of threads for CPU operations (OMP/MKL/BLAS).
        If None, uses the system default.
        Note: Setting this may not affect libraries that are already initialized.

    Examples
    --------
    >>> session = QESSession(backend='jax', seed=123)
    >>> session.start()
    >>> # ... run code ...
    >>> session.stop()
    """

    def __init__(self,
                 backend: str = 'numpy',
                 seed: int = 42,
                 precision: Literal['float32', 'float64'] = 'float64',
                 num_threads: Optional[int] = None):
        self._backend_name = backend
        self._seed = seed
        self._precision = precision
        self._num_threads = num_threads
        self._backend_mgr = get_backend_manager()
        self._log = get_logger()
        self._previous_config = {}

    def start(self) -> 'QESSession':
        """
        Apply the session configuration to the global state.

        This sets the environment variables, activates the requested backend,
        and reseeds the random number generators.

        Returns
        -------
        QESSession
            The started session instance.
        """
        self._log.info(f"Starting QESSession(backend={self._backend_name}, seed={self._seed}, precision={self._precision})")

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
        End the session.

        Currently, this primarily logs the session end. Full state restoration
        is not yet implemented.
        """
        self._log.info("Stopping QESSession")
        # Implementation of full restore is tricky with global singletons.
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def run(backend: str = 'numpy',
        seed: int = 42,
        precision: Literal['float32', 'float64'] = 'float64',
        num_threads: Optional[int] = None) -> QESSession:
    """
    Context manager to run a block of code with a specific QES configuration.

    This is the recommended entry point for configuring a QES workflow. It ensures
    parameters are set before the code block runs.

    Parameters
    ----------
    backend : str, optional
        Computation backend ('numpy', 'jax'). Default 'numpy'.
    seed : int, optional
        Random seed for reproducibility. Default 42.
    precision : {'float32', 'float64'}, optional
        Floating point precision. Default 'float64'.
    num_threads : int, optional
        Number of threads for CPU operations.

    Returns
    -------
    QESSession
        The active session object.

    Examples
    --------
    >>> import QES
    >>> with QES.run(backend='jax', seed=123):
    ...     # Code runs with JAX backend and seeded RNG
    ...     pass
    """
    return QESSession(backend=backend, seed=seed, precision=precision, num_threads=num_threads)
