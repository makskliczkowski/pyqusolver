"""
Minimal parameter-scan example for `QES.pydqmc`.

Run from `Python/` with:

    PYTHONPATH=. python QES/pydqmc/examples/example_dqmc_parameter_scan.py
"""

from __future__ import annotations

from QES.Algebra.Model.Interacting.Fermionic.spinful_hubbard import SpinfulHubbardModel
from QES.general_python.lattices import choose_lattice
from QES.pydqmc import run_dqmc


def main():
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    betas = [1.0, 2.0, 4.0]
    for beta in betas:
        model = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)
        result = run_dqmc(
            model,
            beta=beta,
            M=max(8, int(8 * beta)),
            warmup=2,
            sweeps=6,
            measure_every=2,
            num_chains=1,
            n_stable=2,
            seed=17,
        )
        energy = result.summarize_energy(warmup=0, bin_size=1)
        print(f"beta={beta:4.1f} energy={energy['mean']:.6f} +/- {energy['stderr']:.6f}")
        print("  diagnostics:", result.diagnostics)


if __name__ == "__main__":
    main()
