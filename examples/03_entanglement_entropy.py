"""
Entanglement Entropy Example
============================

Computes entanglement entropy, mutual information, and topological entanglement entropy (TEE)
for a Heisenberg-Kitaev model on a honeycomb lattice.

Demonstrates:
- Using `EntanglementModule`
- Handling symmetries
- Computing TEE with Kitaev-Preskill construction

To run:
    python examples/03_entanglement_entropy.py
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import QES
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev
from QES.general_python.physics.entanglement_module import get_entanglement_module, MaskGenerator
from QES.general_python.lattices.honeycomb import HoneycombLattice
from QES.general_python.physics.entropy import entropy, mutual_information
from QES.general_python.physics.density_matrix import rho_spectrum

# JIT-compiled functions
# Updated imports for entropy functions
from QES.Algebra.Symmetries.jit.density_jit import rho_symmetries
from QES.Algebra.Symmetries.jit.entropy_jit import (
    mutual_information as mutual_information_sym,
    topological_entropy as topological_entropy_sym
)

def get_entanglement_entropy(
    hamil: HeisenbergKitaev, hilbert: HilbertSpace, subsystem: Union[int, float]
) -> Tuple[List[float], Dict[Tuple[int, int], float], List[float], Dict[str, float]]:
    """
    Compute entanglement entropy for all eigenstates, mutual information for the ground state,
    and topological entanglement entropy (TEE).
    """
    logger = QES.get_logger()
    has_sym = hilbert.has_sym
    ns = hamil.lattice.ns
    fraction = subsystem if subsystem < 1.0 else subsystem / ns
    va = int(fraction * ns)

    entropies = []
    mut_info = {}
    purity = []
    topological = {}
    num_states = hamil.eig_val.shape[0]

    # 1. Compute Entanglement Entropy for all states
    for state_idx in range(num_states):
        psi = hamil.eig_vec[:, state_idx]
        try:
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
                    info = mutual_information_sym(psi_gs, i, j, hilbert, q=1)
                    mut_info[(i, j)] = info
        except Exception as e:
            logger.error(f"Failed to compute symmetric mutual information: {e}", lvl=0)
    else:
        logger.info("Computing mutual information for all site pairs.", color="green", lvl=1)
        for i in range(ns):
            for j in range(i + 1, ns):
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
    # Note: TEE requires larger systems, but we run the code path here for demonstration
    if ns >= 8:
        try:
            regions = MaskGenerator.kitaev_preskill(ns)

            if has_sym:
                res_tee = topological_entropy_sym(psi_gs, regions, hilbert, q=1)
                gamma = res_tee["gamma"]
                topological = res_tee
            else:
                ent_mod = get_entanglement_module(hamil)
                tee_results = ent_mod.topological_entropy(
                    state=psi_gs, regions=regions, construction="kitaev_preskill"
                )
                gamma = tee_results["gamma"]
                topological = tee_results

            logger.info(
                f"Computed topological entanglement entropy: γ = {gamma:.6f}", color="green", lvl=1
            )
        except Exception as e:
            logger.warning(f"Skipping TEE calculation (system possibly too small or regions invalid): {e}", lvl=1)
            topological = {"error": str(e)}
            gamma = None
    else:
        logger.info("System size too small for TEE calculation.", lvl=1)
        gamma = None

    return entropies, mut_info, purity, topological, gamma

if __name__ == "__main__":
    logger = QES.get_logger()

    # Create 2x2x1 Honeycomb lattice (8 sites)
    # This is the smallest honeycomb lattice
    lat = HoneycombLattice(lx=2, ly=2, bc="pbc")

    # Create Heisenberg-Kitaev model
    # Use complex128 for stability
    model = HeisenbergKitaev(lattice=lat, K=1.0, J=0.1, hz=0.05, dtype=np.complex128)

    # Create Hilbert space with symmetries (Translation)
    # Using sym_gen dictionary to define symmetries
    # Sector kx=0, ky=0 (Gamma point)
    sym_gen = {'translation': {'kx': 0, 'ky': 0}}

    hilbert = HilbertSpace(lattice=lat, sym_gen=sym_gen, dtype=np.complex128)

    # Diagonalize
    logger.info(f"Diagonalizing {model.name} with symmetries...", color="blue")
    model.diagonalize(k=5, method="exact")  # compute first 5 states

    # Compute entanglement properties
    entropies, mut_info, purity, topological, gamma = get_entanglement_entropy(model, hilbert, subsystem=0.5)

    print("\nResults:")
    print(f"Ground State EE: {entropies[0]:.6f}")
    print(f"Ground State Purity: {purity[0]:.6f}")
    if gamma is not None:
        print(f"Topological Entanglement Entropy γ: {gamma:.6f}")

    print("\nMutual Information (first few pairs):")
    for k, v in list(mut_info.items())[:5]:
        print(f"  I({k[0]}, {k[1]}) = {v:.6f}")
