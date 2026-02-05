# QES Benchmarks

This directory contains a benchmark suite for the QES (Quantum Energy Solver) library. It measures the performance of key components such as Hamiltonian construction and Neural Quantum State (NQS) sampling/optimization.

## Structure

- `benchmarks/run.py`: The main entry point to run benchmarks.
- `benchmarks/bench_hamil.py`: Benchmarks for `Hamiltonian` operations (build, matvec).
- `benchmarks/bench_nqs.py`: Benchmarks for `NQS` operations (sample, step).
- `benchmarks/utils.py`: Utility functions for timing and reporting.

## How to Run

Run the benchmarks from the root of the repository:

```bash
python benchmarks/run.py
```

### Options

- `--heavy`: Run benchmarks with larger system sizes. Warning: Hamiltonian construction for large sizes (e.g., 5x5 lattice) can be very slow or memory-intensive.
- `--filter {hamil,nqs,all}`: Run only a specific subset of benchmarks. Default is `all`.

Example:
```bash
python benchmarks/run.py --filter nqs --heavy
```

## What is Measured

### Hamiltonian (`bench_hamil.py`)
1.  **Construction (`build`)**: Time taken to construct the full Hamiltonian matrix (sparse). Scales with $2^N$ where $N$ is the number of spins.
2.  **Matrix-Vector Product (`matvec`)**: Time taken to apply the Hamiltonian to a random state vector. This is a critical operation for iterative diagonalization (e.g., Lanczos).

### Neural Quantum States (`bench_nqs.py`)
1.  **Sampling (`sample`)**: Throughput of the VMC sampler (samples per second logic, inferred from runtime for fixed batch). Uses an RBM ansatz on a Square Lattice TFIM.
2.  **Optimization Step (`step`)**: Time taken for one full optimization step, which includes sampling, local energy evaluation, gradient computation, and parameter update logic.

## Interpreting Results

The output shows the mean runtime and standard deviation over multiple repeats.

```
TFIM Build (L=4x4, Ns=16)                         : 0.123456 s +/- 0.012345 s (N=5)
```

- **Build time**: Lower is better. Increases exponentially with system size for matrix-based Hamiltonians.
- **Matvec time**: Lower is better. Critical for solver performance.
- **NQS Sample**: Lower time (for fixed samples) means higher throughput.
- **NQS Step**: Lower is better. Indicates faster training loops.

## Backend

Benchmarks try to use `jax` backend for NQS if available, falling back to `numpy`. The output will indicate which backend is being used. Hamiltonian benchmarks typically use `scipy.sparse` (via `numpy` backend logic) or `jax` if configured (defaulting to numpy/scipy logic for matrix build currently).
