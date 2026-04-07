#!/usr/bin/env python3
"""
Default spinful onsite Hubbard DQMC example.

Usage:
    PYTHONPATH=. python QES/pydqmc/examples/example_dqmc_default.py

Math:
    the simple path calls ``run_dqmc(...)`` with no explicit HS keyword.
    For repulsive onsite Hubbard interactions ``U > 0``, the default HS
    channel is magnetic, so the slice matrix is

        B_tau = exp(-dtau K) exp[lambda s_tau (n_up - n_dn)].
"""

from __future__ import annotations

from QES.Algebra.Model.Interacting.Fermionic.spinful_hubbard import SpinfulHubbardModel
from QES.general_python.lattices import choose_lattice
from QES.pydqmc import run_dqmc


def main():
    """Run the default repulsive spinful-Hubbard DQMC example and print a compact summary."""
    lattice = choose_lattice("square", lx=2, ly=2, bc="pbc")
    model = SpinfulHubbardModel(lattice=lattice, t=1.0, U=4.0)

    result = run_dqmc(
        model,
        beta=1.0,
        M=8,
        warmup=2,
        sweeps=4,
        measure_every=1,
        num_chains=1,
        n_stable=2,
        seed=7,
    )

    print("=== Default Spinful Onsite Hubbard DQMC ===")
    print("HS setup:", result["setup"]["hs"])
    print("Observables:", result["observables"])
    print("Diagnostics:", result["diagnostics"])


if __name__ == "__main__":
    main()
