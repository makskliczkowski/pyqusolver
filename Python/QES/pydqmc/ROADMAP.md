# pydqmc Roadmap

## Current baseline (working now)
- Lazy package exports in `QES.pydqmc` resolve correctly.
- Core objects are importable from one namespace:
  - `DQMCModel`, `HubbardDQMCModel`, `choose_dqmc_model`
  - `DQMCSampler`, `DQMCSolver`
  - `calculate_green_stable`, `sherman_morrison_update`, `propagate_green`
- Model parameter extraction is robust to scalar wrappers (Python/NumPy/JAX scalar-like containers).

## Phase 1: Correctness parity (highest priority)
- Port/stabilize the exact Loh/QR-UDT path from C++ (`UDT_QR::inv1P`, `factMult`) with pivoted QR.
- Add deterministic regression tests against:
  - direct inversion for small systems,
  - C++ reference outputs for fixed seeds,
  - particle-hole symmetric Hubbard benchmarks.
- Add consistency checks per sweep:
  - Green residual `|| (I + B)G - I ||`,
  - refresh drift before/after stabilization step.

## Phase 2: General model API
- Split model interface into:
  - kinetic builder (`K`),
  - HS channel specification,
  - per-channel propagator constructor,
  - local update deltas and acceptance ratio hooks.
- Add first-class support for:
  - spinful Hubbard (repulsive/attractive),
  - extended Hubbard (`V` terms),
  - multi-orbital density-density interactions.
- Keep model plugins backend-agnostic (NumPy/JAX) with one shared API.

## Phase 3: Performance engineering
- Replace per-site rank-1 update loops with blocked/delayed updates (`rank-k` Woodbury refresh).
- Add checkerboard kinetic decomposition for nearest-neighbor hopping models.
- Cache `exp(±dtau K)` by `(beta, M, model params)` and avoid redundant recomputation.
- Fuse chain/time kernels where possible (`vmap`/`scan` over chains and slices) and minimize Python control flow.
- Introduce optional mixed precision path (`float32` compute + `float64` stabilization checkpoints).

## Phase 4: Measurements and ergonomics
- Add standard observables with validated estimators:
  - kinetic, interaction, density, double occupancy, spin/charge structure factors.
- Add binning/jackknife and integrated autocorrelation-time reporting.
- Add checkpoint/resume with full RNG and field-configuration state.
- Add one public high-level entrypoint:
  - `run_dqmc(model=..., beta=..., M=..., warmup=..., sweeps=..., measure_every=...)`.

## Phase 5: Scale-out and advanced methods
- Multi-device parallel chains (`pmap`/`shard_map`) for throughput.
- Optional finite-size scaling helper utilities and parameter sweeps.
- Optional constrained-path / force-bias / Langevin HS variants as research extensions.

## Definition of done for "state-of-the-art" milestone
- Numerical agreement with reference implementations on standard Hubbard benchmarks.
- Stable low-temperature runs without catastrophic Green drift.
- Competitive wall-clock throughput per effective sample on CPU/GPU.
- Clear, model-agnostic API for adding new fermionic Hamiltonians.
