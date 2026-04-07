# DQMC Examples

This directory contains small runnable examples for the main `QES.pydqmc` workflows.
Location: `QES/pydqmc/examples/`

## Available Examples

### Default Onsite Hubbard

- **example_dqmc_default.py**            - Repulsive spinful onsite Hubbard with the default magnetic HS channel
- **example_dqmc_attractive_charge.py**  - Attractive spinful onsite Hubbard with the default charge HS channel

### Alternative HS Representations

- **example_dqmc_compact_hs.py**         - Spinful onsite Hubbard with the compact continuous interpolating HS field

### Spinless Density-Density

- **example_dqmc_spinless_bond.py**      - Spinless bond-density Hubbard with bond-centered HS fields

### Workflow and Scans

- **example_dqmc_parameter_scan.py**     - Minimal beta scan showing the structured result and energy postprocessing helpers

## Quick Start

Run examples from `Python/` with:

```bash
PYTHONPATH=. python QES/pydqmc/examples/example_dqmc_default.py
```

Each example prints:

- the HS setup actually used
- measured observables
- numerical diagnostics

The parameter-scan example additionally shows:

- `DQMCResult.summarize_energy(...)`
- how to loop over a small control parameter grid without ad hoc output parsing

## Which Example To Start With

Recommended order:

1. `example_dqmc_default.py`
2. `example_dqmc_attractive_charge.py`
3. `example_dqmc_compact_hs.py`
4. `example_dqmc_spinless_bond.py`
5. `example_dqmc_parameter_scan.py`

## Core Math

All examples use the same stabilized DQMC structure:

```text
B_tau = exp(-dtau K) exp(V_tau[s]),
G     = (I + B_M ... B_1)^(-1).
```

The examples differ only in how the auxiliary field `s` generates the diagonal potential `V_tau[s]`.

### Repulsive Onsite Hubbard

Default channel:

```text
n_up - n_dn
```

### Attractive Onsite Hubbard

Default channel:

```text
n_up + n_dn - 1
```

### Spinless Bond Density-Density

Current bond update:

```text
exp[-dtau V (n_i - 1/2)(n_j - 1/2)]
  = C sum_{s=+-1} exp[alpha s (n_i - n_j)].
```

## See Also

- [README.md](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/README.md) - `pydqmc` overview and usage notes
- [dqmc_solver.py](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/dqmc_solver.py) - high-level `run_dqmc(...)` entrypoint
- [hs.py](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/hs.py) - HS implementations and math comments
- [docs/interpreting_diagnostics.md](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/docs/interpreting_diagnostics.md) - how to read the run diagnostics
