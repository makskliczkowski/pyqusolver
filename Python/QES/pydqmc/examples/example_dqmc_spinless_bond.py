#!/usr/bin/env python3
"""
Spinless bond-density DQMC example.

Usage:
    PYTHONPATH=. python QES/pydqmc/examples/example_dqmc_spinless_bond.py

Math:
    for one bond ``(i, j)`` the current faithful spinless path uses

        exp[-dtau V (n_i - 1/2)(n_j - 1/2)]
          = C sum_{s=+-1} exp[alpha s (n_i - n_j)],

    with

        cosh(alpha) = exp(dtau V / 2).

    One local field update therefore changes two diagonal entries with opposite
    signs, which is why these updates are rank-2 rather than rank-1.
"""

from __future__ import annotations

from QES.Algebra.Model.Interacting.Fermionic.hubbard import HubbardModel
from QES.general_python.lattices import choose_lattice
from QES.pydqmc import run_dqmc


def main():
    """Run the spinless bond-density DQMC example using the bond HS decoupling."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    model = HubbardModel(lattice=lattice, t=1.0, U=2.0)

    result = run_dqmc(
        model,
        beta=1.0,
        M=8,
        warmup=2,
        sweeps=4,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=19,
    )

    print("=== Spinless Bond-Density DQMC ===")
    print("HS setup:", result["setup"]["hs"])
    print("Observables:", result["observables"])
    print("Diagnostics:", result["diagnostics"])


if __name__ == "__main__":
    main()
