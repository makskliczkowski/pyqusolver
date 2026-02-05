"""
QES Quickstart Session
======================

This simple example demonstrates how to use the QES session context manager
to configure the global backend and precision settings.

To run:
    python examples/00_quickstart_session.py
"""

import os
import QES

if __name__ == "__main__":
    print("Initial backend:", QES.get_backend_manager().name)

    # Use QES.run to create a session with specific settings
    # This affects global singletons like the random number generator and backend
    with QES.run(backend='numpy', seed=123, precision='float32') as session:
        print("-" * 40)
        print("Inside session (numpy, seed=123, precision=float32)")
        print("Backend:", QES.get_backend_manager().name)
        print("Seed:", QES.get_backend_manager().default_seed)

        # Check environment variable set by the session
        print("Precision Env Var:", os.environ.get("PY_FLOATING_POINT"))
        print("-" * 40)

    print("Session ended.")
