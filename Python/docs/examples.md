# Usage Examples

This section provides cookbook-style recipes for common tasks in QES. These examples demonstrate how to combine Hilbert spaces, lattices, and Hamiltonians to simulate interesting physics.

## 1. Interacting Spin Model (Heisenberg-Kitaev)

This example shows how to set up a spin-1/2 system on a honeycomb lattice with complex interactions.

```python
import numpy as np
from QES.Algebra.Hilbert.hilbert_local import LocalSpace
from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.Algebra.Hilbert.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev

# 1. Define the Lattice and Local Space
# We create a small 2x2 honeycomb cluster (8 sites total)
local_space = LocalSpace.default_spin_half().with_catalog_operators()
lattice = HoneycombLattice(dim=2, lx=2, ly=2)

# 2. Create the Hilbert Space
# This manages the basis states and symmetries
hilbert = HilbertSpace(
    lattice=lattice,
    is_manybody=True,
    local_space=local_space,
    gen_mapping=True,  # Generate symmetry mappings automatically
)

# 3. Instantiate the Hamiltonian
# Heisenberg J + Kitaev K interactions
ham = HeisenbergKitaev(
    lattice=lattice,
    hilbert_space=hilbert,
    hx=0.0, hz=0.0,      # No external field
    j=1.0,               # Heisenberg interaction
    kx=1.0, ky=1.0, kz=1.0, # Kitaev interactions
)

# 4. Build and Solve
ham.build_hamiltonian()
ham.diagonalize(verbose=True)

print("Lowest 5 eigenvalues:", ham.eigenvalues[:5])
```

## 2. Thermodynamics of Free Fermions

QES has built-in tools for calculating thermal properties. Here, we calculate the free energy of a fermionic system over a range of temperatures.

```python
import numpy as np
from QES.Algebra import HamiltonianConfig, Hamiltonian
from QES.Algebra.Properties import quadratic_thermal

# 1. Create a simple 1D chain Hamiltonian using the Config system
# This is an alternative way to create objects using dictionaries
config = HamiltonianConfig(
    kind="quadratic",
    hilbert=None,  # Not needed for simple quadratic configs
    parameters={
        "ns": 10,
        "particle_conserving": True,
        "particles": "fermions",
    },
)
quad = Hamiltonian.from_config(config)

# Add hopping to make it a chain
for i in range(9):
    quad.add_hopping(i, i+1, -1.0)

quad.diagonalize()

# 2. Perform a Thermal Scan
energies = quad.eigenvalues
temperatures = np.linspace(0.1, 4.0, 40)

# Calculate properties like Free Energy, Entropy, Heat Capacity
scan = quadratic_thermal.quadratic_thermal_scan(
    energies,
    temperatures,
    particle_type="fermion",
    particle_number=5, # Half-filling
)

# Access results
t_idx = 10 # Pick a temperature index
T = temperatures[t_idx]
F = scan["free_energy"][t_idx]
print(f"Free energy at T={T:.2f}: {F:.4f}")
```

## 3. Using JAX for Acceleration

Switching to the JAX backend is simple and allows you to take advantage of GPU acceleration or automatic differentiation.

```python
from QES.Algebra import QuadraticHamiltonian

# Initialize with backend='jax'
# Note: You must have 'jax' installed
qh = QuadraticHamiltonian(ns=100, backend='jax')

# Setup a large random Hamiltonian
import jax.numpy as jnp
rng = jnp.random.PRNGKey(0)
random_vals = jnp.random.uniform(rng, shape=(99,))

for i in range(99):
    qh.add_hopping(i, i+1, random_vals[i])

# Diagonalization happens via JAX
qh.diagonalize()
print("First 5 eigenvalues (computed with JAX):")
print(qh.eig_val[:5])
```

## More Examples

For more advanced use cases, such as:
*   Training Neural Quantum States (NQS)
*   Time Evolution (TDVP)
*   Custom Symmetries

Please refer to the `examples/` directory in the source code repository.
