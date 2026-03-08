'''Run the maintained Python example suite in a fixed order.

The examples are executed in separate subprocesses so each script gets a clean
environment and failures are isolated to the script that triggered them.
'''

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env():
    """Build a deterministic environment for example subprocesses."""
    env = os.environ.copy()
    py  = str(ROOT / "Python")
    env["PYTHONPATH"] = py if not env.get("PYTHONPATH") else py + os.pathsep + env["PYTHONPATH"]
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    return env

# Note:
# We use subprocesses so each example runs with the same environment a user
# would see from the command line. This also keeps one failing example from
# contaminating the rest of the run.

def main():
    # Section: Maintained example entrypoints
    scripts = [
        "examples/algebra/example_hilbert_and_custom_hamiltonian.py",
        "examples/algebra/example_operators_on_states.py",
        "examples/algebra/example_sparse_dense_matrix_build.py",
        "examples/algebra/example_quadratic_single_particle.py",
        "examples/algebra/example_quadratic_interop.py",
        "examples/physics/example_entropy_density_matrix.py",
        "examples/physics/example_time_evolution_and_spectral_stats.py",
        "examples/physics/example_spectral_and_statistical_tools.py",
        "examples/lattices/example_lattice_neighbors_and_honeycomb.py",
        "examples/models/example_random_spin_models.py",
        "examples/workflows/example_lattice_driven_hamiltonian.py",
    ]

    # Section: Execute examples one by one
    for rel in scripts:
        cmd = [sys.executable, rel]
        print("\n$", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)


if __name__ == "__main__":
    main()

# ------
#! EOF
# ------
