#!/usr/bin/env python3
"""
State-of-the-Art Demo: Minimum Entangled States (MES) and Topological Entropy.

This demo showcases:
1.  Construction of the Kitaev Honeycomb model in its topological phase.
2.  Identification of the degenerate ground state manifold.
3.  Finding Minimum Entangled States (MES) using entropy minimization.
4.  Extraction of Topological Entanglement Entropy (TEE) via Kitaev-Preskill and Levin-Wen partitions.
5.  Visualisation of the lattice regions, entanglement spectra, and MES properties.

Usage:
    python demo_mes.py           # saves PNGs to demo_mes_plots/
    python demo_mes.py --show    # also opens interactive windows
"""

import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as la
from pathlib import Path
from typing import Tuple, List, Dict, Any

# ── project import ──────────────────────────────────────────────────────
# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))

try:
    from QES.general_python.common.flog import get_global_logger
    from QES.general_python.physics.density_matrix import rho_numba_mask, mask_subsystem
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev import HeisenbergKitaev, HoneycombLattice
    from QES.Algebra.Properties.mes import find_mes
    from QES.general_python.physics.entropy import topological_entropy, entropy
except ImportError as e:
    print(f"Error: Could not import QES modules. Make sure the path is correct.\n{e}")
    sys.exit(1)

# setup
SAVE_DIR    = _CWD / "tmp"/ "demo_mes_plots"
SHOW        = "--show" in sys.argv
logger      = get_global_logger()

# Publication-ready plotting style
try:
    plt.style.use(['science', 'no-latex', 'colors5-light'])
except Exception:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.constrained_layout.use": True,
    })

def savefig(fig, name):
    path = SAVE_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot → {path}")
    if SHOW:
        plt.show()
    plt.close(fig)

# core functions

def get_entropy_function(region: List[int], ns: int, q: float = 1.0):
    """Factory for entropy function used by find_mes."""
    _, order    = mask_subsystem(np.array(region), ns)
    size_a      = len(region)
    
    def s_func(psi):
        rho     = rho_numba_mask(psi, order, size_a)
        eigvals = la.eigvalsh(rho)
        return entropy(eigvals, q=q)
    
    return s_func

def get_entanglement_spectrum(psi: np.ndarray, region: List[int], ns: int):
    """Compute the entanglement spectrum (log of eigenvalues of reduced density matrix)."""
    _, order    = mask_subsystem(np.array(region), ns)
    size_a      = len(region)
    rho         = rho_numba_mask(psi, order, size_a)
    eigvals     = la.eigvalsh(rho)
    eigvals     = eigvals[eigvals > 1e-14]
    return -np.log(np.sort(eigvals)[::-1])

# main

def run_demo():
    print("\n" + "="*60)
    print("  QES: MES & Topological Entropy Demo")
    print("="*60 + "\n")

    # 1. Model Setup
    # Using a 2x2x2 honeycomb lattice (8 sites) for quick ED.
    # Kitaev Honeycomb Model: Kx=Ky=Kz=1.0 is in the topological phase.
    lx, ly  = 3, 2
    lat     = HoneycombLattice(lx=lx, ly=ly, bc='pbc')
    model   = HeisenbergKitaev(lattice=lat, K=(1.0, 1.0, 1.0), logger=logger)
    
    logger.info(f"Lattice: {lat}")
    logger.info(f"Model: {model}")

    # 2. Diagonalization
    logger.info("Building Hamiltonian and diagonalizing...")
    model.build(verbose=False)
    model.diagonalize(method='exact', verbose=False)
    
    evals   = model.eigenvalues
    evecs   = model.eigenvectors
    logger.info(f"Ground state energy: {evals[0]:.6f}")
    
    # Identify degenerate ground states
    gs_tol      = 1e-8
    gs_indices  = np.where(np.abs(evals - evals[0]) < gs_tol)[0]
    n_gs        = len(gs_indices)
    logger.info(f"Found {n_gs} degenerate ground states.")
    
    V_gs        = evecs[:, gs_indices]

    # 3. MES Search
    # We define a cut (half of the system) to find MES
    region_half = lat.get_region(kind='half_x')
    logger.info(f"MES region (half_x): {region_half}")
    
    S_func      = get_entropy_function(region_half, lat.Ns)
    
    logger.info("Finding MES states...")
    mes_states, mes_values = find_mes(V_gs, S_func, n_trials=20, state_max=n_gs)
    
    for i, S in enumerate(mes_values):
        logger.info(f"MES {i}: Entanglement Entropy = {S:.6f}")

    # 4. Topological Entanglement Entropy (TEE)
    # TEE construction: S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    # For 8 sites, KP radius might need to be small.
    region_kp = lat.get_region(kind='kitaev_preskill', radius=1.2)
    region_lw = lat.get_region(kind='levin_wen', inner_radius=0.8, outer_radius=1.5)
    
    logger.info(f"Kitaev-Preskill regions: {list(region_kp.keys())}")
    logger.info(f"Levin-Wen regions: {list(region_lw.keys())}")

    # Compute TEE for the first MES state
    tee_kp = topological_entropy(mes_states[0], region_kp, lat.Ns, topological='kitaev_preskill')
    # region_lw also has A, B, C, ... keys, so we can use the same KP formula
    tee_lw = topological_entropy(mes_states[0], region_lw, lat.Ns, topological='kitaev_preskill')
    
    logger.info(f"TEE (Kitaev-Preskill) = {tee_kp['gamma']:.6f}")
    logger.info(f"TEE (Levin-Wen)      = {tee_lw['gamma']:.6f}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Visualisation
    # ═══════════════════════════════════════════════════════════════════════════

    # --- Plot 1: Regions ---
    print("\n--- Generating Figure 1: Lattice Regions ---")
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot KP regions
    kp_abc = {k: v for k, v in region_kp.items() if len(k) == 1}
    lat.plot.regions(
        kp_abc, ax=axes1[0], title="Kitaev-Preskill Partition",
        fill=True, fill_alpha=0.15, show_bonds=True
    )
    
    # Plot LW regions
    lw_abc = {k: v for k, v in region_lw.items() if k in ['A', 'B', 'C']}
    lat.plot.regions(
        lw_abc, ax=axes1[1], title="Levin-Wen Partition",
        fill=True, fill_alpha=0.15, show_bonds=True
    )
    
    fig1.suptitle(f"Lattice Partitions for TEE (Ns={lat.Ns})", fontsize=14)
    savefig(fig1, "01_lattice_regions.png")

    # --- Plot 2: Entanglement Spectrum ---
    print("--- Generating Figure 2: Entanglement Spectrum ---")
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    
    # MES Spectrum
    es_mes = get_entanglement_spectrum(mes_states[0], region_half, lat.Ns)
    ax2.plot(range(len(es_mes)), es_mes, 'o-', label=f"MES (S={mes_values[0]:.4f})", markersize=8)
    
    # Random State in GS Manifold Spectrum
    c_rand = np.random.randn(n_gs) + 1j*np.random.randn(n_gs)
    c_rand /= np.linalg.norm(c_rand)
    psi_rand = V_gs @ c_rand
    es_rand = get_entanglement_spectrum(psi_rand, region_half, lat.Ns)
    ax2.plot(range(len(es_rand)), es_rand, 's--', label=f"Random GS (S={S_func(psi_rand):.4f})", alpha=0.7)
    
    ax2.set_xlabel("Level Index $n$")
    ax2.set_ylabel(r"Entanglement Level $\xi_n = -\ln \lambda_n$")
    ax2.set_title("Entanglement Spectrum Comparison")
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    savefig(fig2, "02_entanglement_spectrum.png")

    # --- Plot 3: Entropy Minimization / Statistics ---
    print("--- Generating Figure 3: Entropy Statistics ---")
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    
    # Sample many random states in GS manifold and compute their entropy
    n_samples = 500
    entropies_rand = []
    for _ in range(n_samples):
        c = np.random.randn(n_gs) + 1j*np.random.randn(n_gs)
        c /= np.linalg.norm(c)
        entropies_rand.append(S_func(V_gs @ c))
        
    ax3.hist(entropies_rand, bins=30, alpha=0.6, color='tab:blue', label="Random Ground States", density=True)
    
    # Mark MES values
    for i, val in enumerate(mes_values):
        ax3.axvline(val, color='tab:red', linestyle='--', linewidth=2, label="MES" if i==0 else None)
        
    ax3.set_xlabel("Entanglement Entropy $S$")
    ax3.set_ylabel("Probability Density")
    ax3.set_title("Entropy Distribution in Ground State Manifold")
    ax3.legend()
    
    # Add text box with TEE values
    textstr = "\n".join((
        r"$\gamma_{KP} = %.4f$" % (tee_kp['gamma'], ),
        r"$\gamma_{LW} = %.4f$" % (tee_lw['gamma'], )
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    savefig(fig3, "03_entropy_statistics.png")

    print(f"\n✓ All plots saved to {SAVE_DIR}/")
    print("Done.\n")

if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    run_demo()
