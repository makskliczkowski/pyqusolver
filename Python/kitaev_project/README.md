# Kitaev Honeycomb Impurity Workflow

This package provides a focused workflow for studying the Kitaev honeycomb Hamiltonian with on-site impurities, optional \( \Gamma \)-terms, and both neural-quantum-state (NQS) and exact-diagonalization (ED) solvers. The scaffold is intentionally lightweight but enforces a common interface for Hamiltonian generation, solver execution, observable evaluation, and HDF5 persistence.

## Directory Overview

- `models/`: Lattice builders and Hamiltonian factories that wrap the existing QES Heisenberg–Kitaev implementation and expose knobs for impurities and \( \Gamma \)-terms.
- `nqs/`: Neural-network ansatz definitions, a unified training interface, observable utilities, and workflow helpers (ground state, excited states, transfer learning, parameter sweeps).
- `ed/`: Sparse Lanczos / exact-diagonalization drivers for benchmarking up to \(3\times 4\) clusters.
- `io/`: Typed HDF5 schema definitions plus high-level readers/writers for storing states, observables, and training metadata.
- `workflows/`: Orchestrators that glue the `models/`, `nqs/`, `ed/`, and `io/` layers together for reproducible studies.
- `tests/`: Pytest-based smoke tests covering the shared abstractions.
- `notebooks/`: Interactive demos; `notebooks/kitaev_nqs_vs_ed.ipynb` shows the full workflow on a small cluster.

## Design Highlights

1. **Model-centric orchestration**  
   `models.kitaev_model.KitaevModelBuilder` constructs lattices (clean or impurity decorated) while keeping strong typing around couplings and Gamma terms.

2. **Backend-agnostic NQS stack**  
   Every ansatz implements the `NeuralAnsatz` protocol, so new architectures (autoregressive, RBM, simple CNN) plug seamlessly into `nqs.training.NQSTrainer`.

3. **Multi-state workflow**  
   `nqs.pipelines` exposes routines for ground states, excited states (via orthogonality constraints), transfer-learning on impurity configurations, and parameter sweeps.

4. **Shared HDF5 schema**  
   `io.hdf5_writer.KitaevResultWriter` enforces a single schema for NQS and ED outputs, ensuring downstream tooling can compare apples-to-apples.

5. **Benchmark parity**  
   The `ed.lanczos.LanczosSolver` consumes the same Hamiltonian objects and pushes results into the identical storage layout used by NQS runs.

## Getting Started

```bash
cd Python
python -m kitaev_project.workflows.pipeline_demo --config configs/example_clean.yaml
```

The CLI builds the clean Kitaev Hamiltonian, trains the configured NQS ansatz, benchmarks it against ED, and writes a fully structured HDF5 file with raw samples, observables, and metadata. From there you can:

1. Launch parameter sweeps (`--sweep Kz`),  
2. Activate impurities (`--impurities "[[0, +1.0], [3, -1.0]]"`),  
3. Enable excited-state targeting (`--excited 4`), or  
4. Switch architectures via YAML config files.
