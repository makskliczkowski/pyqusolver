# Tutorials

This section summarizes the main supported workflows and points to maintained example scripts.

## Many-body custom Hamiltonian

- Build a `HilbertSpace` on a lattice.
- Create a `Hamiltonian` and add terms through operator factories.
- Build the sparse or dense matrix.
- Diagonalize and inspect the low-energy spectrum.

Reference example:

- `examples/algebra/example_hilbert_and_custom_hamiltonian.py`

## Quadratic single-particle workflow

- Construct a `QuadraticHamiltonian`.
- Add onsite, hopping, or pairing terms.
- Build either the single-particle matrix or the full BdG matrix.
- Use `diagonalizing_bogoliubov_transform` for the Qiskit-style orbital transform.
- Use `to_qiskit_hamiltonian`, `from_qiskit_hamiltonian`, or `from_openfermion_hamiltonian` for external interoperability when optional dependencies are installed.

Reference example:

- `examples/algebra/example_quadratic_single_particle.py`

## Density matrix and entropy workflow

- Build reduced density matrices with `general_python.physics.density_matrix`.
- Compute von Neumann or Renyi entropies with `general_python.physics.entropy`.
- Compute mutual information between sites.

Reference example:

- `examples/physics/example_entropy_density_matrix.py`

## Time evolution and spectral statistics

- Diagonalize a Hamiltonian.
- Use `ham.time_evo` for evolved states.
- Compute level-spacing diagnostics with `gap_ratio`.

Reference example:

- `examples/physics/example_time_evolution_and_spectral_stats.py`

## Lattice-driven construction

- Use `SquareLattice` or `HoneycombLattice`.
- Extract neighbors directly from the lattice.
- Feed those bonds into Hamiltonian assembly.

Reference example:

- `examples/workflows/example_lattice_driven_hamiltonian.py`
