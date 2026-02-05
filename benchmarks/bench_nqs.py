import sys
import os
import numpy as np
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
python_path = os.path.abspath(os.path.join(current_dir, "../Python"))
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.general_python.lattices.square import SquareLattice
    from QES.NQS.nqs import NQS
    from benchmarks.utils import benchmark
except ImportError as e:
    print(f"ImportError in bench_nqs: {e}")
    traceback.print_exc()
    sys.exit(1)

def run_nqs_benchmarks(sizes=[4, 6], heavy=False):
    np.random.seed(42)  # Deterministic seed

    if heavy:
        sizes = [4, 6, 8, 10]

    print("\n" + "="*60)
    print("NQS Benchmarks (RBM + TFIM + VMC)")
    print("="*60)

    for L in sizes:
        _run_for_size(L)

def _run_for_size(L):
    # Setup
    lattice = SquareLattice(lx=L, ly=L, dim=2)
    model = TransverseFieldIsing(lattice=lattice, j=1.0, hx=0.5, dtype=np.complex128)

    # print(f"DEBUG: model.lattice={getattr(model, 'lattice', 'MISSING')}")
    # print(f"DEBUG: model.hilbert={getattr(model, 'hilbert', 'MISSING')}")

    try:
        import jax
        backend = "jax"
    except ImportError:
        backend = "numpy"

    try:
        nqs = NQS(
            logansatz="rbm",
            model=model,
            sampler="vmc",
            backend=backend,
            batch_size=1024,
            verbose=False,
            shape=(lattice.ns,)
        )
        nqs.init_network()
    except Exception as e:
        print(f"Skipping NQS benchmark for L={L} due to initialization failure: {e}")
        return

    print(f"Running NQS benchmarks with backend: {nqs.backend_str}")

    @benchmark(name=f"NQS Sample (L={L}x{L}, batch=1024)", n_repeat=10, n_warmup=2)
    def bench_sample():
        nqs.sample(num_samples=1024, num_chains=1)

    bench_sample()

    @benchmark(name=f"NQS Step (Sample+Grad) (L={L}x{L})", n_repeat=5, n_warmup=1)
    def bench_step():
        # batch_size is already set in NQS init
        # num_samples/num_chains for sampling are set in sampler or default
        # We ensure they match what we want by setting them on sampler if needed
        # But for benchmarking step(), we rely on instance configuration.
        nqs.step()

    bench_step()

if __name__ == "__main__":
    run_nqs_benchmarks()
