import numpy as np
from .utils import run_benchmark, print_header, BenchmarkSquareLattice

try:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
    from QES.NQS.nqs import NQS
except ImportError as e:
    raise ImportError(f"Failed to import QES modules: {e}")

def benchmark_nqs_step(L, num_samples, num_chains, repeats):
    print(f"  Setting up NQS for L={L}x{L}...")

    lattice = BenchmarkSquareLattice(dim=2, lx=L, ly=L)
    hilbert = HilbertSpace(lattice=lattice)
    model = TransverseFieldIsing(lattice=lattice, hilbert_space=hilbert, j=1.0, hx=0.5, dtype=np.complex128)
    net = RBM(input_shape=(hilbert.ns,), n_hidden=hilbert.ns, bias=True)

    nqs = NQS(
        logansatz=net,
        model=model,
        hilbert=hilbert,
        batch_size=num_samples * num_chains, # Batch size for grad computation
        verbose=False,
        backend="jax"
    )

    # Configure sampler
    nqs.sampler.set_numsamples(num_samples)
    nqs.sampler.set_numchains(num_chains)
    nqs.sampler.set_therm_steps(10)
    nqs.sampler.set_sweep_steps(1)
    # Reset sampler to ensure arrays are resized to new chain count
    nqs.sampler.reset()

    # Init network
    nqs.init_network()

    # Warmup / Compile
    print("  Compiling NQS step...")
    try:
        nqs.step()
    except Exception as e:
        print(f"  Compilation failed: {e}")
        raise e

    def workload():
        nqs.step()

    run_benchmark(f"NQS Step (Sample+Grad+Upd) L={L}x{L}", workload, repeats=repeats)

def run(heavy=False, repeats=3):
    print_header("NQS Optimization Step Benchmark")

    sizes = [4, 6] if not heavy else [6, 8]

    for L in sizes:
        # Keep samples modest to allow measuring overhead and gradient scaling
        if not heavy:
            num_samples = 100
            num_chains = 4
        else:
            num_samples = 200
            num_chains = 16

        benchmark_nqs_step(L, num_samples, num_chains, repeats)
