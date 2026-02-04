# QES Benchmarks

This directory contains performance benchmarks for key components of the Quantum Eigen Solver (QES).

## Structure

*   `benchmarks/run_benchmarks.py`: The main entry point to run benchmarks.
*   `benchmarks/vmc_benchmark.py`: Benchmarks VMC sampling throughput.
*   `benchmarks/nqs_benchmark.py`: Benchmarks NQS optimization step (gradient computation + update).
*   `benchmarks/hamil_benchmark.py`: Benchmarks Hamiltonian local energy evaluation.

## Running Benchmarks

To run the full suite with default settings:

```bash
python3 benchmarks/run_benchmarks.py
```

### Options

*   `--heavy`: Run larger system sizes and more samples. Useful for testing scaling on powerful machines.
    ```bash
    python3 benchmarks/run_benchmarks.py --heavy
    ```
*   `--filter <name>`: Run only benchmarks matching the given name.
    ```bash
    python3 benchmarks/run_benchmarks.py --filter vmc
    ```
*   `--repeats <n>`: Number of times to repeat each measurement (default: 3).
    ```bash
    python3 benchmarks/run_benchmarks.py --repeats 5
    ```

## Benchmark Descriptions

### VMC Sampling
Measures the throughput of the Variational Monte Carlo sampler (`VMCSampler`) with an RBM ansatz on a Square Lattice with Transverse Field Ising Model.
*   **Metric**: Time to generate `N` samples.
*   **Scaling**: Tested on L=4x4, L=6x6 (and larger in heavy mode).

### NQS Optimization Step
Measures the time to perform a single optimization step of a Neural Quantum State (`NQS`). This includes sampling, gradient calculation (SR/MinSR), and parameter update.
*   **Metric**: Time per step.
*   **Scaling**: Tested on L=4x4, L=6x6.

### Hamiltonian Local Energy
Measures the time to calculate local energy matrix elements ($H_{s,s'}$ and indices $s'$) for a batch of states.
*   **Metric**: Time to process a batch of states.
*   **Scaling**: Tested on L=4x4, L=6x6.

## Interpretation

*   **VMC Sampling**: Lower time indicates faster sampling, which is crucial for convergence speed.
*   **NQS Step**: Lower time indicates faster training loops.
*   **Hamiltonian**: Lower time indicates faster energy estimation, which is the bottleneck in VMC.
