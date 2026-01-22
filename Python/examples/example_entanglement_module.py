"""
Example script for computing entanglement properties, including
topological entanglement entropy (TEE) and mutual information,
supporting both symmetric and non-symmetric many-body states.

------------------------------------------------------------------------------
File        : Python/examples/example_entanglement_module.py
Author      : Maksymilian Kliczkowski
Date        : 2025-12-30
------------------------------------------------------------------------------
"""

from typing import Dict, List, Tuple, Union

import numpy as np

try:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
    from QES.general_python.common.flog import Logger
    from QES.general_python.physics.entanglement_module import get_entanglement_module
except ImportError:
    raise ImportError("QES package is required to run this example.")

# Initialize logger
logger = Logger(name="EntanglementExample", verbose=True)


def get_entanglement_entropy(
    hamil: HeisenbergKitaev, hilbert: HilbertSpace, subsystem: Union[int, float]
) -> Tuple[List[float], Dict[Tuple[int, int], float], List[float], Dict[str, float]]:
    """
    Compute entanglement entropy for all eigenstates, mutual information for the ground state,
    and topological entanglement entropy (TEE).

    Supports symmetry-reduced states automatically via the improved EntanglementModule.
    """
    from QES.Algebra.Symmetries.jit.density_jit import (
        mutual_information_symmetries,
        rho_symmetries,
        topological_entropy_symmetries,
    )
    from QES.general_python.physics.density_matrix import rho_spectrum
    from QES.general_python.physics.entanglement_module import MaskGenerator
    from QES.general_python.physics.entropy import entropy, mutual_information

    has_sym = hilbert.has_sym
    ns = hamil.lattice.ns
    fraction = subsystem if subsystem < 1.0 else subsystem / ns
    va = int(fraction * ns)

    entropies = []
    mut_info = {}  # (i,j) -> mutual information between sites i and j
    purity = []
    topological = {}
    num_states = hamil.eig_val.shape[0]

    # 1. Compute Entanglement Entropy for all states
    for state_idx in range(num_states):
        psi = hamil.eig_vec[:, state_idx]
        try:
            # rho_symmetries now handles mask if va is an array, but here it is a contiguous size
            rho = rho_symmetries(psi, va=va, hilbert=hilbert)
            vals = rho_spectrum(rho)
            purity.append(np.sum(vals**2))
        except Exception as e:
            logger.error(
                f"Failed to compute Schmidt decomposition for state {state_idx}: {e}", lvl=0
            )
            entropies.append(np.nan)
            continue
        S_ent = entropy(vals, q=1)
        entropies.append(S_ent)

    logger.info(
        f"Computed entanglement entropy for {num_states} states on subsystem of size {va} sites.",
        color="green",
        lvl=1,
    )

    # 2. Compute Mutual Information for the Ground State (state 0)
    psi_gs = hamil.eig_vec[:, 0]

    if has_sym:
        logger.info(
            "Computing mutual information with symmetries for all site pairs.",
            color="green",
            lvl=1,
        )
        try:
            for i in range(ns):
                for j in range(i + 1, ns):
                    info = mutual_information_symmetries(psi_gs, i, j, hilbert, q=1)
                    mut_info[(i, j)] = info
        except Exception as e:
            logger.error(f"Failed to compute symmetric mutual information: {e}", lvl=0)
    else:
        logger.info("Computing mutual information for all site pairs.", color="green", lvl=1)
        for i in range(ns):
            for j in range(i + 1, ns):
                # Standard mutual information helper
                try:
                    info, _ = mutual_information(psi_gs, i, j, ns, q=1)
                    mut_info[(i, j)] = info
                except Exception as e:
                    logger.error(
                        f"Failed to compute mutual information for sites ({i}, {j}): {e}", lvl=0
                    )

    logger.info(
        "Successfully computed mutual information for all site pairs.", color="green", lvl=1
    )

    # 3. Add topological entanglement entropy (TEE) computation
    if ns >= 12:  # TEE typically requires larger systems for meaningful results
        try:
            # Generate Kitaev-Preskill regions
            regions = MaskGenerator.kitaev_preskill(ns)

            if has_sym:
                res_tee = topological_entropy_symmetries(psi_gs, regions, hilbert, q=1)
                gamma = res_tee["gamma"]
                topological = res_tee
            else:
                ent_mod = get_entanglement_module(hamil)
                # construction='kitaev_preskill' uses the regions A, B, C defined in Lattice or MaskGenerator
                tee_results = ent_mod.topological_entropy(
                    state=psi_gs, regions=regions, construction="kitaev_preskill"
                )
                gamma = tee_results["gamma"]
                topological = tee_results

            logger.info(
                f"Computed topological entanglement entropy: γ = {gamma:.6f}", color="green", lvl=1
            )
        except Exception as e:
            logger.error(f"Failed to compute TEE: {e}", lvl=0)
            topological = {"error": str(e)}
    else:
        logger.info("System size too small for reliable TEE calculation.", lvl=1)

    return entropies, mut_info, purity, topological


if __name__ == "__main__":
    # Example usage with a small Honeycomb Kitaev model
    from QES.Algebra.Symmetries.symmetry_container import SymmetryContainer
    from QES.general_python.lattices.honeycomb import HoneycombLattice

    # Create 2x2x1 Honeycomb lattice (8 sites)
    lat = HoneycombLattice(lx=2, ly=2, bc="pbc")

    # Create Heisenberg-Kitaev model
    model = HeisenbergKitaev(lattice=lat, K=1.0, J=0.1, hz=0.05)

    # Add symmetries (Translation)
    sym = SymmetryContainer(lattice=lat)
    sym.add_translation()
    # sym.add_parity() # optionally add more

    # Create Hilbert space with symmetries
    hilbert = HilbertSpace(lattice=lat, symmetry_container=sym)

    # Diagonalize
    logger.info(f"Diagonalizing {model.name} with symmetries...", color="blue")
    model.diagonalize(k=10, method="exact")  # compute first 10 states

    # Compute entanglement properties
    entropies, mut_info, purity, gamma = get_entanglement_entropy(model, hilbert, subsystem=0.5)

    print("\nResults:")
    print(f"Ground State EE: {entropies[0]:.6f}")
    print(f"Ground State Purity: {purity[0]:.6f}")
    if gamma is not None:
        print(f"Topological Entanglement Entropy γ: {gamma:.6f}")

    print("\nMutual Information (first few pairs):")
    for k, v in list(mut_info.items())[:5]:
        print(f"  I({k[0]}, {k[1]}) = {v:.6f}")
