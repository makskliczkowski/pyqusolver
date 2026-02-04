# New Examples

The following examples have been added to demonstrate modern scientific workflows using QES.

| Example File | Description | Goal | Prerequisites | Runtime | Validates |
|--------------|-------------|------|---------------|---------|-----------|
| `workflow_hamiltonian_analysis.py` | Constructs Spin (TFIM) and Quadratic (Free Fermion) Hamiltonians, checks properties, and diagonalizes them. | Demonstrate Hamiltonian API and basic analysis. | `numpy`, `scipy` | < 5s | Matrix construction, Hermiticity, Diagonalization, QuadraticHamiltonian API. |
| `workflow_variational_ground_state.py` | Runs a VMC optimization loop for a 1D TFIM and compares the result with Exact Diagonalization. | Verify VMC correctness against exact results. | `jax`, `flax`, `optax` | ~1 min | NQS Interface, VMC Sampling, TDVP/SR Optimization, Correctness Check. |

## Usage

These examples are designed to be run directly from the root of the repository:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/Python:$(pwd)/Python/QES
python examples/workflow_hamiltonian_analysis.py
python examples/workflow_variational_ground_state.py
```

## Details

### `workflow_hamiltonian_analysis.py`
- **System 1**: 1D Transverse Field Ising Model (Spin).
  - Checks sparsity and hermiticity.
  - Computes ground state energy and gap via sparse diagonalization.
- **System 2**: 1D Free Fermions (Quadratic).
  - Constructs using Bogoliubov-de Gennes (BdG) matrices.
  - Demonstrates `QuadraticHamiltonian` efficient spectrum calculation.

### `workflow_variational_ground_state.py`
- **System**: 1D TFIM (N=10).
- **Method**: Restricted Boltzmann Machine (RBM) with VMC.
- **Workflow**:
  1. Compute exact ground state $E_{ED}$.
  2. Initialize RBM ansatz.
  3. Optimize using Stochastic Reconfiguration (SR) for 10 epochs.
  4. Compare $E_{VMC}$ with $E_{ED}$.
- **Output**: Prints relative error and success/warning message.
