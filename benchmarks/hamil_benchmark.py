import numpy as np
from .utils import run_benchmark, print_header, BenchmarkSquareLattice

try:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
except ImportError as e:
    raise ImportError(f"Failed to import QES modules: {e}")

def benchmark_hamil_eval(L, batch_size, repeats):
    print(f"  Setting up Hamiltonian for L={L}x{L}...")

    lattice = BenchmarkSquareLattice(dim=2, lx=L, ly=L)
    hilbert = HilbertSpace(lattice=lattice)
    # Use complex128 to match Numba expectation in special_operator
    model = TransverseFieldIsing(lattice=lattice, hilbert_space=hilbert, j=1.0, hx=0.5, dtype=np.complex128)

    # We do NOT call build() because it attempts to construct the full Hamiltonian matrix,
    # which is impossible for L=6 (2^36 states).
    # TransverseFieldIsing.__init__ calls _set_local_energy_functions(), so loc_energy should work.

    # Generate random states (integers representing basis states)
    # For large systems, we can't use random integers in range [0, 2^N) if 2^N > int64 max.
    # Python ints handle arbitrary precision, but numpy arrays?
    # HilbertSpace handles states.
    # Let's use a batch of small integers just to test throughput of the operator logic,
    # assuming the logic doesn't depend on the specific value much (it might for sparsity).
    # Actually, for spins, we usually work with states as arrays of spins if we use NQS.
    # But Hamiltonian.loc_energy takes integer indices (mapping to states) or arrays if backend is JAX?
    # Reading Hamiltonian.loc_energy: "k (Union[int, np.ndarray]) : The k'th element of the Hilbert space"
    # If we pass an array of ints, it treats them as indices in the Hilbert space.

    # For L=6, N=36. 2^36 fits in int64.

    # However, creating a random array of 2^36 is tricky if we use np.random.randint which takes int.
    # np.random.randint(low, high, size, dtype=int64)
    # 2^36 < 2^63. Safe.

    limit = min(2**model.ns, 2**63 - 1)
    rng = np.random.default_rng(42)
    states = rng.integers(0, limit, size=batch_size, dtype=np.int64)

    # Warmup
    print("  Warmup local energy...")
    try:
        # Check if internal function is available (avoids wrapper argument mismatch bug)
        if hasattr(model, "_loc_energy_int_fun") and model._loc_energy_int_fun is not None:
            # Use direct call
            model._loc_energy_int_fun(states[0])
            workload_fn = lambda: [model._loc_energy_int_fun(s) for s in states]
        else:
            # Fallback
            model.loc_energy(states[0])
            workload_fn = lambda: model.loc_energy(states)

    except Exception as e:
        print(f"  Local energy execution failed: {e}")
        raise e

    def workload():
        workload_fn()

    run_benchmark(f"Hamiltonian Local Energy (Batch={batch_size}) L={L}x{L}", workload, repeats=repeats)

def run(heavy=False, repeats=3):
    print_header("Hamiltonian Local Energy Benchmark")

    sizes = [4, 6] if not heavy else [6, 8]

    for L in sizes:
        batch_size = 1000 if not heavy else 10000
        benchmark_hamil_eval(L, batch_size, repeats)
