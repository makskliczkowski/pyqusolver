# Density Matrix Integration Plan

## Code Touchpoints

### 1. New Module: `Python/QES/Algebra/DensityMatrix`
*   **Status**: Safe (New files).
*   **Content**:
    *   `__init__.py`: Expose `DensityMatrix`.
    *   `density_matrix.py`: Core class implementation.
    *   `utils.py`: Helper functions for partial traces, entropy, etc. (Future).

### 2. Integration with `Python/QES/Algebra/Operator`
*   **Status**: Low Risk (Extension).
*   **Plan**:
    *   Operators currently act on vectors. No immediate change needed to `Operator` class if `DensityMatrix` handles the `matvec` application internally (by iterating/batching over columns/ensemble members).
    *   **Future**: Add `superoperator()` method to `Operator` to return the Liouville representation ($I \otimes A$ or $A \otimes I$).

### 3. Integration with `Python/QES/Algebra/Hamil/Hamiltonian`
*   **Status**: Safe (Extension).
*   **Plan**:
    *   Add `thermal_state(beta)` method to `Hamiltonian` (or a factory in `DensityMatrix` that takes a `Hamiltonian`).
    *   Ensure `Hamiltonian.matvec` works correctly when passed a matrix (JAX/NumPy broadcasting usually handles this, but verification is needed).

## Rollout Plan

### Phase 1: Foundation (Current PR)
*   Create `docs/CONCEPT_DENSITY_MATRIX.md`.
*   Create `Python/QES/Algebra/DensityMatrix` directory.
*   Add minimal `DensityMatrix` class scaffolding (inert).

### Phase 2: Core Implementation (Future Branch)
*   Implement `from_pure`, `from_ensemble`, `from_matrix`.
*   Implement `trace`, `purity`, `expectation`.
*   Connect `expectation` to `Operator.matvec`.
*   Add unit tests for basic properties.

### Phase 3: Dynamics and Thermal States (Future Branch)
*   Implement `evolve(hamiltonian, t)`.
*   Implement `thermal(hamiltonian, beta)` using exact diagonalization (for small systems).
*   Add tests against known thermal results (e.g., Ising model energy).

### Phase 4: Advanced Features (Future Branch)
*   Lindblad master equation support.
*   NQS density matrix ansatz (Neural Density Operator).
*   Vectorized Liouville space operations.

## Compatibility and Safety

*   **Import Paths**: No existing import paths will be changed. The new module is additive.
*   **Behavior**: Existing code using `Hamiltonian` and `Operator` with pure states will remain unaffected.
*   **Dependencies**: No new heavy dependencies introduced (standard NumPy/SciPy/JAX stack).

## Future Branches

Breaking changes or major features will be tracked in `docs/FUTURE_BRANCHES.md`.
The current plan does not require breaking changes to `general_python` or core `QES` logic.
