# FUTURE_BRANCHES.md

Compatibility-breaking or high-risk structural changes deferred from this branch.
Each item is tagged with **future branch**.

## [future branch] Package renames and path normalization

- Rename `QES.Algebra.Hamil` to `QES.Algebra.Hamiltonian` and migrate references.
- Flatten/reshape `QES.Algebra.Operator.impl` into clearer physics-domain packages.

## [future branch] Solver package reshaping

- Move `QES.Solver.MonteCarlo.vmc` into a dedicated top-level solver namespace.
- Consolidate sampler implementations under a unified `samplers/` package API.

## [future branch] `general_python` decomposition

- Split `QES.general_python` into focused packages/modules with deprecation shims.
- Migrate callers to stable non-monolithic namespaces over a staged cycle.

## [future branch] Global-state API reduction

- Reduce reliance on global singletons in `QES.qes_globals`.
- Transition to explicit context/config injection where practical.
