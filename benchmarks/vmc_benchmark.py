import numpy as np
from .utils import run_benchmark, print_header, BenchmarkSquareLattice

# Imports
try:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    from QES.general_python.ml.net_impl.networks.net_rbm import RBM
    from QES.Solver.MonteCarlo.vmc import VMCSampler
except ImportError as e:
    raise ImportError(f"Failed to import QES modules: {e}")

def benchmark_vmc(L, num_samples, num_chains, repeats):
    print(f"  Setting up VMC for L={L}x{L}...")

    # Physics Setup
    lattice = BenchmarkSquareLattice(dim=2, lx=L, ly=L)
    hilbert = HilbertSpace(lattice=lattice)
    # Model not strictly needed for VMC sampling benchmark but good for completeness/context
    model = TransverseFieldIsing(lattice=lattice, hilbert_space=hilbert, j=1.0, hx=0.5, dtype=np.complex128)

    # Network Setup
    # RBM input shape is flat (Ns,)
    # hidden units: alpha=1.0 density
    net = RBM(input_shape=(hilbert.ns,), n_hidden=hilbert.ns, bias=True)

    # Initialize network parameters (JAX needs this usually, but VMCSampler might handle it via sample)
    # VMCSampler calls net.init if available or uses passed params.
    # RBM is a FlaxInterface, it has init().
    # However, VMCSampler expects a network it can call.
    # FlaxInterface has init() and apply().
    # VMCSampler handles Flax linen modules or objects with 'apply'.
    # We should let VMCSampler handle it.

    # Sampler Setup
    import jax
    rng_key = jax.random.PRNGKey(42)
    rng = np.random.default_rng(42)

    sampler = VMCSampler(
        net=net,
        shape=(L*L,),
        hilbert=hilbert,
        numsamples=num_samples,
        numchains=num_chains,
        therm_steps=10,
        sweep_steps=1,
        verbose=False,
        backend="jax",
        rng=rng,
        rng_k=rng_key,
        seed=42
    )

    # Force compilation (JIT) by running 1 sample
    print("  Compiling JIT kernels...")
    sampler.sample(num_samples=1)

    def workload():
        # Run sampling
        # sample() returns ((states, log_psi), (all_states, all_log_psi), probs)
        sampler.sample(num_samples=num_samples)

    run_benchmark(f"VMC Sampling L={L}x{L} ({num_chains} chains, {num_samples} samples)", workload, repeats=repeats)

def run(heavy=False, repeats=3):
    print_header("VMC Sampling Benchmark")

    # Sizes: L=4 (16 spins), L=6 (36 spins)
    # Heavy: L=8 (64 spins), L=10 (100 spins)
    sizes = [4, 6] if not heavy else [6, 8]

    for L in sizes:
        # Configuration
        # Total samples = num_samples * num_chains
        if not heavy:
            num_samples = 500
            num_chains = 16
        else:
            num_samples = 1000
            num_chains = 64

        benchmark_vmc(L, num_samples, num_chains, repeats)
