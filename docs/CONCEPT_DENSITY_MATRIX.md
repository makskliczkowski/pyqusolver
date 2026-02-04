# Density Matrix Concept: Mixed States in QES

## Motivation and Use Cases

The current QES architecture primarily focuses on pure states ($|\psi\rangle$) and their evolution under Hamiltonians. However, many physical scenarios require a description in terms of mixed states (density matrices, $\rho$):

1.  **Thermal States**: Systems in equilibrium at finite temperature $T > 0$, described by Gibbs ensembles $\rho \propto e^{-\beta H}$.
2.  **Decoherence and Open Systems**: Systems interacting with an environment, requiring Lindblad master equations or quantum channels.
3.  **Reduced Density Matrices**: Analyzing subsystems (entanglement entropy, partial traces) where the local state is mixed even if the global state is pure.
4.  **Ensemble Averages**: Statistical mixtures of pure states, often arising from classical uncertainty in preparation.

Introducing a `DensityMatrix` abstraction will allow QES to naturally handle these cases while leveraging the existing efficient operator stack.

## Proposed API

The API is designed to be small, sharp, and consistent with existing QES patterns.

### Construction

```python
class DensityMatrix(State): # Assuming a State base class exists or will exist
    """
    Represents a quantum density matrix.
    """

    @classmethod
    def from_pure(cls, psi: Union[np.ndarray, 'Array'], backend='default'):
        """Constructs rho = |psi><psi|."""
        pass

    @classmethod
    def from_ensemble(cls, states: List[Union[np.ndarray, 'Array']], weights: List[float]):
        """Constructs rho = sum_i w_i |psi_i><psi_i|."""
        pass

    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, 'Array']):
        """Constructs rho from a dense or sparse matrix."""
        pass

    @classmethod
    def thermal(cls, hamiltonian: 'Hamiltonian', beta: float, method='exact'):
        """
        Constructs a thermal state exp(-beta H) / Z.
        Method can be 'exact' (ED) or 'imaginary_time' (for MPO/NQS in future).
        """
        pass
```

### Core Operations

```python
    def trace(self) -> float:
        """Returns Tr(rho). Should be close to 1.0."""
        pass

    def purity(self) -> float:
        """Returns Tr(rho^2). 1.0 for pure states, < 1.0 for mixed."""
        pass

    def entropy_vn(self) -> float:
        """Returns von Neumann entropy -Tr(rho log rho)."""
        pass

    def expectation(self, op: 'Operator') -> complex:
        """
        Computes Tr(rho * op).
        If rho is an ensemble {(p_i, psi_i)}, computes sum p_i <psi_i|op|psi_i>.
        """
        pass

    def partial_trace(self, subsystem: List[int]) -> 'DensityMatrix':
        """Returns the reduced density matrix for the specified subsystem."""
        pass
```

### Evolution

#### Unitary Evolution
For closed systems starting in a mixed state:
$$ \rho(t) = U(t) \rho(0) U^\dagger(t) $$
where $U(t) = e^{-iHt}$.

```python
    def evolve(self, hamiltonian: 'Hamiltonian', t: float, method='krylov') -> 'DensityMatrix':
        """
        Evolves the density matrix under the given Hamiltonian.
        """
        pass
```

#### Liouville-Space Form (Concept)
For open systems or when vectorization is useful, $\rho$ can be flattened into a supervector $|\rho\rangle\rangle$.
The von Neumann equation $\dot{\rho} = -i[H, \rho]$ becomes $\frac{d}{dt}|\rho\rangle\rangle = \mathcal{L} |\rho\rangle\rangle$, where $\mathcal{L} = -i(H \otimes I - I \otimes H^T)$.
This allows reusing existing ODE solvers and Krylov methods designed for vectors.

### Interoperability with SpecialOperator

`SpecialOperator` and `Hamiltonian` are designed to act on vectors. To support density matrices without rewriting operators, we introduce the concept of **Left and Right Actions**:

*   **Left Action ($L_A$):** $L_A \rho = A \rho$. In vectorized form, $L_A \rightarrow A \otimes I$.
*   **Right Action ($R_A$):** $R_A \rho = \rho A$. In vectorized form, $R_A \rightarrow I \otimes A^T$.

Since `SpecialOperator` implements `matvec(psi)`, we can implement these actions on a `DensityMatrix`:

1.  **Dense Representation ($D \times D$ matrix):**
    *   $L_A$: Apply `matvec` to each *column* of $\rho$.
    *   $R_A$: Apply `matvec` (of $A^\dagger$) to each *row* of $\rho$ (or column of $\rho^\dagger$) and take adjoint.

2.  **Ensemble Representation ($\{p_i, |\psi_i\rangle\}$):**
    *   $L_A$: Return new ensemble $\{p_i, A|\psi_i\rangle\}$. (Note: this makes the "ket" unnormalized, or requires re-normalization and weight update).
    *   Expectation values are computed as averages over the ensemble.

This approach keeps the `Operator` API common: `DensityMatrix` consumes `Operator` methods, rather than requiring `Operator` to know about `DensityMatrix`.

## Contracts

*   **Shapes:**
    *   Dense: `(Nh, Nh)`
    *   Ensemble: List of `(Nh,)` vectors.
*   **Dtypes:** Consistent with `Hamiltonian` (float64/complex128).
*   **Hermiticity:** $\rho^\dagger = \rho$. Operations should preserve this (within numerical tolerance).
*   **Trace:** $\text{Tr}(\rho) = 1$. Evolution should preserve trace.
*   **Positivity:** $\rho \ge 0$. Hard to enforce exactly in all truncations, but checks should be available.

## Performance Strategy

1.  **Avoid Full Density Matrices**: For large systems ($N > 14$), full $D \times D$ matrices are infeasible.
    *   **Ensemble Method**: Use Monte Carlo sampling of pure states (e.g., minimally entangled typical thermal states - METTS) or just a collection of pure states if the rank is low.
    *   **Lazy Materialization**: Only compute elements or contractions when needed.

2.  **Numba/JAX Compilation**:
    *   The existing `SpecialOperator` infrastructure uses Numba/JAX for `matvec`. `DensityMatrix` operations should batch these calls.
    *   For JAX, use `vmap` to apply `matvec` across the columns of a density matrix or the members of an ensemble.
    *   Avoid recompiling kernels for every `rho`; reuse the existing `Hamiltonian` kernels.

3.  **Low-Rank Approximations**: Store $\rho \approx V V^\dagger$ where $V$ is $D \times k$ ($k \ll D$). Evolution acts on $V$.

## Testing Plan

*   **Invariants**:
    *   Trace = 1.
    *   Hermitian.
    *   Purity $\le 1$.
*   **Consistency**:
    *   $\text{Tr}(\rho H)$ should match thermal energy.
    *   `from_pure(psi).expectation(O)` == `psi.conj() @ O @ psi`.
    *   Evolution of `from_pure(psi)` should match evolution of `psi`.
