# Neural Quantum States (`QES.NQS`)

`QES.NQS` provides a maintained neural-quantum-state workflow for variational
ground states, excited-state targeting, TDVP-based time evolution, and
observable evaluation on top of the broader QES model and lattice stack.

## Recommended Workflow

The preferred public path is config-first:

1. define the physical problem with `NQSPhysicsConfig`
2. define solver defaults with `NQSSolverConfig`
3. optionally define training defaults with `NQSTrainConfig`
4. build `model`, `hilbert`, and the ansatz
5. construct `NQS`
6. train and measure

This keeps runs reproducible and makes the physical and solver assumptions
explicit.

## Minimal Example

```python
from QES.NQS import NQS, NQSPhysicsConfig, NQSSolverConfig, NQSTrainConfig

# Accelerated path:
# pip install "QES[jax]"
# or:
# pip install "QES[all]"

p_cfg = NQSPhysicsConfig(
    model_type="tfim",
    lattice_type="chain",
    lx=16,
    bc="obc",
)

s_cfg = NQSSolverConfig(
    ansatz="rbm",
    backend="jax",
    dtype="complex128",
    n_samples=512,
    n_chains=32,
    epochs=200,
    lr=1e-2,
)

model, hilbert, lattice = p_cfg.make_hamiltonian()
net = s_cfg.make_net(p_cfg, alpha=1.0)

psi = NQS(
    logansatz=net,
    model=model,
    hilbert=hilbert,
    backend=s_cfg.backend,
    dtype=s_cfg.dtype,
    s_numchains=s_cfg.n_chains,
    s_numsamples=s_cfg.n_samples,
)

train_cfg = NQSTrainConfig.from_solver(s_cfg)
stats = psi.train(**train_cfg.to_train_kwargs())

energy = psi.compute_energy(num_samples=s_cfg.n_samples)
print("final_energy", stats.history[-1])
print("measured_energy", energy.mean, energy.error_of_mean)
```

## Compact Expert Path

If you already have a model, Hilbert space, and know exactly which ansatz you
want, you can construct the solver directly:

```python
from QES.NQS import NQS

psi = NQS(
    logansatz="rbm",
    model=model,
    hilbert=hilbert,
    backend="jax",
    dtype="complex128",
)

stats = psi.train(n_epochs=200, lr=1e-2)
```

## Core Objects

- `NQS`
  - main solver object owning the ansatz, sampler, state convention, and evaluation path
- `NQSPhysicsConfig`
  - physical model, lattice, geometry, and Hamiltonian-construction config
- `NQSSolverConfig`
  - ansatz, sampler, backend, and solver defaults
- `NQSTrainConfig`
  - training-time scheduling and checkpoint options
- `load_nqs(...)`
  - convenience path for config-driven reconstruction plus checkpoint restore

## Measurement

The low-level measurement API is:

- `compute_energy(...)`
- `compute_observable(...)`

There is also a higher-level helper:

- `measure(...)`

Examples:

```python
energy = psi.compute_energy(num_samples=1000)

obs = psi.measure(
    {
        "sx0": sig_x0.jax,
        "sz0": sig_z0.jax,
    },
    num_samples=1000,
)

print(obs["sx0"].mean, obs["sx0"].error_of_mean)
```

## Entanglement Entropy

The NQS entropy path lives in `QES.NQS.src.nqs_entropy` and is exposed through
the solver:

- `psi.compute_renyi2(region, ...)`
- `psi.compute_renyi_entropies(regions, q_values=[2, 3], ...)`
- `psi.compute_topological_entropy(...)`

Use `compute_renyi_entropies(...)` when comparing many cuts or many Renyi
indices. It shares replica samples across compatible regions and avoids the
slow one-call-per-cut loop:

```python
cuts = lattice.get_entropy_cuts(cut_type="all")
valid_cuts = {name: region for name, region in cuts.items() if 0 < len(region) < lattice.ns}

entropy = psi.compute_renyi_entropies(
    valid_cuts,
    q_values=[2, 3],
    num_samples=2000,
    return_error=True,
)
```

For exact ED comparisons, use
`QES.NQS.src.nqs_entropy.compute_ed_entanglement_entropy(...)` on ED
eigenvectors. That function is only a benchmark helper; production NQS entropy
uses the replica estimator.

## Spectral And Time-Domain Response

The NQS spectral path is variational and time-domain:

- `psi.time_evolve(times, ...)` runs real-time TDVP on the NQS parameters.
- `psi.compute_dynamical_correlator(times, ket_probe_operator=..., ...)`
  builds probe states and estimates time-domain correlators.
- `psi.dynamic_structure_factor(times, probe_operator=..., ...)` applies the
  probe-state workflow and FFT reconstruction.
- `psi.spectrum_from_correlator(times, correlator, ...)` only postprocesses an
  already sampled correlator.
- `psi.dynamic_structure_factor_kspace(...)` computes a probe/momentum map.

Exact-diagonalization and Lehmann spectral functions are deliberately not
duplicated in `QES.NQS.src.spectral`. For ED/Lanczos checks, use the shared
general physics backend:

```python
from QES.general_python.physics.spectral.spectral_backend import (
    find_spectral_peaks,
    integrated_spectral_weight,
    operator_spectral_function_multi_omega,
)

hamil.diagonalize()
ed_spectrum = operator_spectral_function_multi_omega(
    omega_grid,
    hamil.eig_val,
    hamil.eig_vec,
    operator_matrix,
    eta=0.05,
)
```

The internal `QES.NQS.src.spectral.exact` module exists only for tiny NQS
regression tests where the variational wavefunction is enumerated over the full
basis. It is not the ED spectral source of truth.

## Persistence

Three persistence levels are supported conceptually:

1. weights only
   - `psi.save_weights(...)`
   - `psi.load_weights(...)`
2. resumable training checkpoints
   - through `train(...)` checkpoint options
3. full config-driven reconstruction
   - `load_nqs(physics_config, solver_config, checkpoint_step=...)`

Use `load_nqs(...)` when the construction path itself should be reproducible.

## Ansatz Selection

Typical ansatz names include:

- `rbm`
- `cnn`
- `resnet`
- `mlp`
- `transformer`
- `ar`
- `pp`
- `rbmpp`
- `jastrow`
- `mps`
- `approx_symmetric`
- `eqgcnn`

These are resolved through the NQS ansatz/factory layer, while still allowing
user-provided networks and plain callables.

## Notes

- The generic ML backbones live under `QES.general_python.ml`.
- NQS-specific representation handling, ansatz wrappers, and faster sampler
  integration live in `QES.NQS`.
- If you need a printed walkthrough from inside a live solver instance, use:
  - `psi.help("general")`
  - `psi.help("usage")`
  - `psi.help("sampling")`
  - `psi.help("entropy")`
  - `psi.help("spectral")`
  - `psi.help("checkpoints")`
