import sys
import os
import numpy as np
import traceback

# Adjust path to include Python/ directory if running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
python_path = os.path.abspath(os.path.join(current_dir, "../Python"))
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.general_python.lattices.square import SquareLattice
    from benchmarks.utils import benchmark
except ImportError as e:
    print(f"ImportError in bench_hamil: {e}")
    traceback.print_exc()
    sys.exit(1)

def run_hamil_benchmarks(sizes=[3, 4], heavy=False):
    """
    Run Hamiltonian benchmarks.
    sizes: List of linear dimensions L for LxL square lattice.
    """
    np.random.seed(42)  # Deterministic seed

    if heavy:
        sizes = [3, 4, 5]

    print("\n" + "="*60)
    print("Hamiltonian Benchmarks (TFIM on Square Lattice)")
    print("="*60)

    for L in sizes:
        Ns = L*L
        if Ns > 16 and not heavy:
            print(f"Skipping L={L} (Ns={Ns}) for standard benchmark (too large for matrix build). Use --heavy if sure.")
            continue
        if Ns > 20 and not heavy:
             print(f"Skipping L={L} (Ns={Ns}) - too large for exact matrix construction (enable --heavy to run).")
             continue
        if Ns > 26:
             print(f"Skipping L={L} (Ns={Ns}) - too large even for heavy benchmark.")
             continue

        _run_for_size(L)

def _run_for_size(L):
    # Setup lattice
    lattice = SquareLattice(lx=L, ly=L, dim=2)

    @benchmark(name=f"TFIM Build (L={L}x{L}, Ns={L*L})", n_repeat=5, n_warmup=1)
    def bench_build():
        # Force rebuild by creating new instance
        H = TransverseFieldIsing(lattice=lattice, j=1.0, hx=0.5, dtype=np.complex128)
        H.build(verbose=False, force=True)
        return H

    # Run build benchmark
    bench_build()

    # Prepare for Matvec
    H = TransverseFieldIsing(lattice=lattice, j=1.0, hx=0.5, dtype=np.complex128)
    H.build(verbose=False)

    # Create random vector
    v = np.random.rand(H.nh) + 1j * np.random.rand(H.nh)
    v /= np.linalg.norm(v)

    @benchmark(name=f"TFIM Matvec (L={L}x{L})", n_repeat=50, n_warmup=10)
    def bench_matvec():
        H.matvec(v)

    bench_matvec()

if __name__ == "__main__":
    run_hamil_benchmarks()
