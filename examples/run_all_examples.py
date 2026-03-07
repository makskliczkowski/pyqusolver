'''Run all examples in one go. This is useful for testing and benchmarking.'''

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env():
    env = os.environ.copy()
    py  = str(ROOT / "Python")
    env["PYTHONPATH"] = py if not env.get("PYTHONPATH") else py + os.pathsep + env["PYTHONPATH"]
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    return env

# Note: We use subprocess to run the examples in separate processes, which is 
# more realistic and also allows us to test the environment variables and other 
# settings that might be needed for the examples to run correctly. It also allows us to 
# catch any exceptions or errors that might occur in the examples without affecting the main script.

def main():
    scripts = [
        "examples/algebra/example_hilbert_and_custom_hamiltonian.py",
        "examples/algebra/example_operators_on_states.py",
        "examples/algebra/example_sparse_dense_matrix_build.py",
        "examples/algebra/example_quadratic_single_particle.py",
        "examples/physics/example_entropy_density_matrix.py",
        "examples/physics/example_time_evolution_and_spectral_stats.py",
        "examples/physics/example_spectral_and_statistical_tools.py",
        "examples/lattices/example_lattice_neighbors_and_honeycomb.py",
        "examples/workflows/example_lattice_driven_hamiltonian.py",
    ]

    for rel in scripts:
        cmd = [sys.executable, rel]
        print("\n$", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)


if __name__ == "__main__":
    main()

# ------
#! EOF
# ------
