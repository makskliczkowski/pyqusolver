# `QES.pydqmc`

Determinant Quantum Monte Carlo for QES Hamiltonians.

## Installation

Run from the repository `Python/` directory with:

```bash
PYTHONPATH=. python -c "from QES.pydqmc import run_dqmc"
```

Runtime requirements:

- `jax`
- `jaxlib`
- `numpy`

The current implementation expects JAX with 64-bit support enabled by the sampler.

## Simplest Path

For the common onsite Hubbard workflow, use [`run_dqmc(...)`](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/dqmc_solver.py#L199):

```python
from QES.Algebra.Model.Interacting.Fermionic.spinful_hubbard import SpinfulHubbardModel
from QES.general_python.lattices import choose_lattice
from QES.pydqmc import run_dqmc

lattice = choose_lattice("square", lx=4, ly=4, bc="pbc")
model   = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)

result  = run_dqmc(model, beta=4.0, M=40, warmup=20, sweeps=100)
print(result["observables"])
print(result["setup"]["hs"])
print(result.summarize_energy())
```

Default onsite convention:

- repulsive `U > 0` uses the magnetic HS channel,
- attractive `U < 0` uses the charge HS channel.

This is the intended default because it matches the standard DQMC sign-dependent decoupling.

## Output Workflow

The preferred output objects are now:

- `DQMCConfig`
  - typed runtime configuration for `run_dqmc(...)`
- `DQMCResult`
  - typed result object with dict-style compatibility

The high-level solver also enforces a conservative sign policy by default:

- `sign_policy="strict"`
  - only known sign-benign regimes are accepted
- `sign_policy="allow_unsupported"`
  - lets you run explicitly unsupported regimes, with explicit sign metadata and sign-reweighted built-in equal-time observables

Example:

```python
from QES.pydqmc import DQMCConfig, load_dqmc_result, run_dqmc

config = DQMCConfig(model=model, beta=4.0, M=40, sweeps=100, measure_every=2)
result = run_dqmc(config)
result.save("dqmc_result.json")

loaded = load_dqmc_result("dqmc_result.json")
print(loaded.observables)
```

Analysis-ready bundles and restart checkpoints are different:

- `result.save("run.json")`
  - JSON analysis bundle
- `result.solver.save_checkpoint("restart.npz")`
  - restart-oriented HS field checkpoint for a compatible solver

Common postprocessing helpers:

```python
from QES.pydqmc import summarize_series, derive_observables

energy_summary  = result.summarize_energy(warmup=5, bin_size=2)
derived         = derive_observables(result.observables, {
    "compressibility_proxy": lambda obs: obs["density"] - 2.0 * obs.get("double_occupancy", 0.0),
})
```

## Core Math

After Trotter decomposition, each imaginary-time slice is represented as

```text
B_tau = exp(-dtau K) exp(V_tau[s]),
```

where:

- `K` is the one-body hopping/chemical-potential matrix,
- `V_tau[s]` is the diagonal HS potential induced by the auxiliary field on slice `tau`.

The equal-time Green's function is

```text
G = (I + B_M B_{M-1} ... B_1)^(-1).
```

This product becomes ill-conditioned at low temperature, so `pydqmc` rebuilds `G` using stabilized factorizations in [`stabilization.py`](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/stabilization.py#L38).

For a local HS update, only a small diagonal block changes. The exact DQMC update is

```text
G' = G - G_{:,S} Delta_S [I + (I - G)_{S,S} Delta_S]^(-1) (I - G)_{S,:},
```

where `S` is the support of the changed HS term. This is implemented in [`localized_diagonal_update(...)`](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/stabilization.py#L69).

Interpretation:

- onsite discrete Hirsch updates have `|S| = 1`,
- bond-centered spinless density-density updates have `|S| = 2`,
- the sampler uses the same update kernel in both cases.

## Main User Paths

### 1. Default Spinful Onsite Hubbard

```python
result = run_dqmc(model, beta=4.0, M=40)
```

Use this unless you explicitly want another HS representation.

### 2. Explicit Compact Continuous Onsite HS

This uses the compact interpolating family from Karakuzu et al. with parameter `p`.

```python
result = run_dqmc(
    model,
    beta=4.0,
    M=40,
    hs="compact",
    p=2.0,
    proposal_sigma=0.2,
)
```

Math:

```text
a_p(s) = sqrt(c_p) atan(p sin s) / atan(p),   s in [-pi, pi].
```

Limits:

- `p -> 0` gives the sinusoidal compact field,
- `p -> inf` approaches the discrete Hirsch limit.

### 3. Spinless Bond Density-Density Hubbard

The existing QES spinless `HubbardModel` has bond interactions, so the faithful DQMC path uses bond HS fields:

```python
from QES.Algebra.Model.Interacting.Fermionic.hubbard import HubbardModel

model = HubbardModel(lattice=lattice, t=1.0, U=2.0)
result = run_dqmc(model, beta=4.0, M=40)
```

Math:

for one bond `(i, j)` the current implementation uses

```text
exp[-dtau V (n_i - 1/2)(n_j - 1/2)]
  = C sum_{s=+-1} exp[alpha s (n_i - n_j)],
```

with

```text
cosh(alpha) = exp(dtau V / 2).
```

So one local field update changes two diagonal entries with opposite sign.

## HS Layer

The HS layer is now split into reusable pieces in [`hs.py`](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/hs.py#L1):

- field distributions:
  - `DiscreteIsingDistribution`
  - `CompactUniformDistribution`
  - `GaussianDistribution`
- coupling rules:
  - `LinearSiteCouplingRule`
  - `OnsiteScalarCouplingRule`
- composed transformations:
  - `MagneticHubbardHS`
  - `ChargeHubbardHS`
  - `GaussianHubbardHS`
  - `CompactInterpolatingHubbardHS`
  - `BondDensityDifferenceHS`

The point of this split is that a new HS family should usually mean:

1. choose the field manifold and local proposal,
2. define the coupling map `a(s)`,
3. connect it to a model adapter.

It should not require a new sampler.

## Supported Sign Contract

`pydqmc` now exposes the current sign-handling contract explicitly through:

- `result.setup["sign"]`
- `result.diagnostics`

Current baseline:

- complex-phase one-body matrices are rejected,
- acceptance uses magnitude-weighted determinant ratios,
- built-in equal-time observables are reweighted with the measured configuration sign on the `|W|` ensemble,
- average sign is tracked in the result diagnostics and observables.

This means the current production envelope is still the sign-benign baseline, but unsupported regimes no longer silently drop the sign in the built-in equal-time estimator path.

By default, `run_dqmc(...)` and `DQMCSolver(...)` enforce this through `sign_policy="strict"`.
If you intentionally want to explore an unsupported regime, you must opt in explicitly:

```python
result = run_dqmc(
    model,
    beta=4.0,
    M=40,
    sign_policy="allow_unsupported",
)
```

Important limitation:

- built-in equal-time observables are sign-reweighted
- custom hooks are automatically reweighted only when they return one scalar value per chain
- unequal-time estimators collected through `collect_unequal=True` are now sign-reweighted
- arbitrary higher-rank custom hook outputs are not yet accumulated through the sign-reweighted path

## What To Use

Recommended starting points:

- `run_dqmc(spinful_model, beta=..., M=...)` for the standard onsite workflow.
- `hs="compact"` only when you intentionally want a continuous compact field.
- the spinless `HubbardModel` path as-is for bond density-density interactions.

If you are unsure, inspect

```python
result["setup"]["hs"]
```

to see exactly which HS transformation was used.

Runnable examples:

- example index: [examples/README.md](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/examples/README.md)
- default onsite path: [example_dqmc_default.py](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/examples/example_dqmc_default.py)
- attractive onsite charge path: [example_dqmc_attractive_charge.py](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/examples/example_dqmc_attractive_charge.py)
- compact continuous onsite path: [example_dqmc_compact_hs.py](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/examples/example_dqmc_compact_hs.py)
- spinless bond-density path: [example_dqmc_spinless_bond.py](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/examples/example_dqmc_spinless_bond.py)
- parameter scan: [example_dqmc_parameter_scan.py](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/examples/example_dqmc_parameter_scan.py)

Diagnostics guide:

- [docs/interpreting_diagnostics.md](/Users/makskliczkowski/Codes/QuantumEigenSolver/pyqusolver/Python/QES/pydqmc/docs/interpreting_diagnostics.md)
