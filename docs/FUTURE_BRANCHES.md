# Future Branches / Breaking Changes

This document records potential refactors that were deferred to preserve backward compatibility (import paths). These changes would improve the library structure but require a major version bump or deprecation cycle.

## `QES.Algebra` Structure

*   **Rename `Hamil` to `Hamiltonian`**: The package `QES.Algebra.Hamil` should be renamed to `QES.Algebra.Hamiltonian` to match the class name and be more descriptive.
*   **Physical Operator Structure**: Move implementation files from `QES.Algebra.Operator.impl` to top-level `QES.Algebra.Operator` or dedicated subpackages like `QES.Algebra.Spin`, `QES.Algebra.Fermion`.

## `QES.Solver` Organization

*   **VMC Package**: Move `QES.Solver.MonteCarlo.vmc` to `QES.Solver.VMC`.
*   **Sampler consolidation**: Merge `sampler.py`, `arsampler.py`, `vmc.py` into a unified `samplers` subpackage.

## `QES.general_python` Decomposition

*   **Dissolve `general_python`**: This submodule is currently a catch-all. It should be broken down and its contents moved to:
    *   `QES.Utils`: Common utilities (`common`).
    *   `QES.Math`: Mathematical functions (`maths`).
    *   `QES.Physics`: General physics utilities (`physics`).
    *   `QES.ML`: Machine learning cores (`ml`).

## Global Namespaces

*   **Remove `QES.qes_globals`**: Use explicit dependency injection or a context object passed through the stack instead of global singletons.
