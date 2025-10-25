# QES Usage Examples

This short README collects runnable snippets that tie the modular building
blocks together—Hilbert spaces, Hamiltonians, and physics utilities.  The
code assumes you are working inside the editable `pyqusolver` checkout so
`python -m QES` style imports resolve correctly.

---

## 1. Interacting spin model (Heisenberg–Kitaev)

```python
import numpy as np

from QES.Algebra.Hilbert.hilbert_local import LocalSpace, LocalSpaceTypes
from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.Algebra.Hilbert.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev

# 1) Build a spin-1/2 local space and Hilbert space with symmetries.
local_space = LocalSpace.default_spin_half().with_catalog_operators()
lattice = HoneycombLattice(dim=2, lx=2, ly=2)  # 8-site honeycomb cluster
hilbert = HilbertSpace(
    lattice=lattice,
    is_manybody=True,
    local_space=local_space,
    sym_gen=[],                     # plug symmetry specs here if desired
    backend="default",
    gen_mapping=True,
)

# 2) Instantiate the Heisenberg–Kitaev Hamiltonian.
ham = HeisenbergKitaev(
    lattice=lattice,
    hilbert_space=hilbert,
    hx=0.0,
    hz=0.0,
    j=1.0,
    kx=ky=kz=1.0,
)

# 3) Build & diagonalise.
ham.build_hamiltonian()
ham.diagonalize(verbose=True)

print("Lowest 5 eigenvalues:", ham.eigenvalues[:5])
```

The interacting models still use their bespoke constructors, but they
benefit from the same `HilbertSpace`/operator infrastructure.  Supply a
`HilbertSpace` with the symmetries you require and the Hamiltonian takes
care of the rest.

---

## 2. Quadratic (free) fermions via the registry

The quadratic Hamiltonian is wired into the registry so you can describe
the problem declaratively:

```python
from QES.Algebra import HilbertConfig, HamiltonianConfig, Hamiltonian

# 1) Describe the single-particle Hilbert space (ns = 4 spinless fermions).
hilbert_cfg = HilbertConfig(ns=4, is_manybody=False, part_conserv=True)

# 2) Describe the Hamiltonian.  Parameters map directly to __init__.
quad_cfg = HamiltonianConfig(
    kind="quadratic",
    hilbert=hilbert_cfg,
    parameters={
        "ns": 4,
        "particle_conserving": True,
        "particles": "fermions",
    },
)

# 3) Materialise and populate couplings.
quad = Hamiltonian.from_config(quad_cfg)
quad.add_hopping(0, 1, 1.0)
quad.add_hopping(1, 2, 1.0)
quad.add_onsite(3, 0.5)

# Optional: BdG pairing
# quad.add_pairing(0, 1, 0.2)

quad.diagonalize()
print("Single-particle spectrum:", quad.eigenvalues)
```

Behind the scenes the registry calls into `QuadraticHamiltonian`, taking
care of BdG assembly when `particle_conserving=False`.  You can query the
current configuration with:

```python
from pprint import pprint
pprint(quad.info())
```

---

## 3. Thermodynamics of a quadratic spectrum

The thermodynamic helper in `QES.Algebra.Properties.quadratic_thermal`
works with any set of eigenvalues, including those returned by the
quadratic Hamiltonian above:

```python
from QES.Algebra.Properties import quadratic_thermal
import numpy as np

energies = quad.eigenvalues
temperatures = np.linspace(0.1, 4.0, 40)
scan = quadratic_thermal.quadratic_thermal_scan(
    energies,
    temperatures,
    particle_type="fermion",
    particle_number=2,
)

print("Free energy at T=1:", scan["free_energy"][temperatures.tolist().index(1.0)])
```

---

## Further Reading

- [`PHYSICS_MODULE.md`](./PHYSICS_MODULE.md) — layout of the physics package.
- [`PHYSICS_EXAMPLES.md`](./PHYSICS_EXAMPLES.md) — more cookbook-style snippets.
- [`QES/Algebra/README.md`](./QES/Algebra/README.md) — overview of the algebra layer.

Feel free to extend the registry with your own Hamiltonians using
`register_hamiltonian` so they surface in the same workflow as the built-in
quadratic model.
