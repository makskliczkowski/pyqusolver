#!/usr/bin/env python3
"""
Spinful onsite Hubbard DQMC with compact continuous HS fields.

Usage:
    PYTHONPATH=. python QES/pydqmc/examples/example_dqmc_compact_hs.py

Math:
    this uses the compact interpolating family

        a_p(s) = sqrt(c_p) atan(p sin s) / atan(p),   s in [-pi, pi],

    so each slice remains

        B_tau = exp(-dtau K) exp(V_tau[s]),

    but the auxiliary field is continuous and compact rather than Ising-valued.
"""

from __future__ import annotations

from QES.Algebra.Model.Interacting.Fermionic.spinful_hubbard import SpinfulHubbardModel
from QES.general_python.lattices import choose_lattice
from QES.pydqmc import run_dqmc


def main():
    """Run the compact continuous onsite-Hubbard DQMC example and print the resulting observables."""
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
        seed=11,
        hs="compact",
        p=2.0,
        proposal_sigma=0.2,
    )

    print("=== Compact Continuous HS DQMC ===")
    print("HS setup:", result["setup"]["hs"])
    print("Observables:", result["observables"])
    print("Diagnostics:", result["diagnostics"])


if __name__ == "__main__":
    main()
