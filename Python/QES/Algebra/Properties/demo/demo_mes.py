#!/usr/bin/env python3
"""
Demo: Minimum Entangled States (MES) and Topological Entropy.

This demo showcases:
1.  Construction of the Kitaev Honeycomb model in its topological phase.
2.  Identification of the degenerate ground state manifold.
3.  Finding Minimum Entangled States (MES) using entropy minimization.
4.  Extraction of Topological Entanglement Entropy (TEE) via Kitaev-Preskill and Levin-Wen partitions.
5.  Testing multiple cuts and region sizes to account for finite-size effects.
6.  Visualisation of the lattice regions, entanglement spectra, and MES properties.

Usage:
    python demo_mes.py                  # saves PNGs to demo_mes_plots/
    python demo_mes.py --show           # also opens interactive windows
    python demo_mes.py --flux 0.5 0.5   # add boundary fluxes in units of pi
"""

import sys, os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as la
from pathlib import Path
from typing import Tuple, List, Dict, Any

from traitlets import default

#! project import
# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))

try:
    from QES.general_python.common.flog                         import get_global_logger
    from QES.general_python.physics.density_matrix              import psi_numpy, mask_subsystem
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev   import HeisenbergKitaev, HoneycombLattice
    from QES.Algebra.Properties.mes                             import find_mes
    from QES.Algebra.Properties.entanglement_spectrum           import (
                                                                    calculate_entanglement_spectrum, 
                                                                    entanglement_entropy_from_spectrum
                                                                )
    from QES.general_python.physics.entropy                     import topological_entropy
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}. Please ensure the QES project structure is intact and all dependencies are installed.")

# setup
SAVE_DIR    = _CWD / "tmp"/ "demo_mes_plots"
logger      = get_global_logger()
ITER        = 0

# -------------------------------
# Utility functions
# -------------------------------

def savefig(fig, name, show=False, q: float = 1.0):
    ''' Save figure to disk with logging. '''
    global ITER
    
    ITER   += 1
    name    = f"{ITER:03d}_{name}_q{q:.2f}.png"
    path    = SAVE_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot -> {path}")
    
    if show:
        plt.show()
    
    plt.close(fig)

def get_entropy_function(region: List[int], ns: int, q: float = 1.0):
    """Factory for entropy function used by find_mes."""
    
    if len(region) == 0 or len(region) == ns:
        return lambda psi: 0.0
        
    _, order    = mask_subsystem(np.array(region), ns)
    size_a      = len(region)
    
    def s_func(psi):
        # Optimized: use SVD to get singular values directly
        psi_mat = psi_numpy(psi, order, size_a, ns, local_dim=2)
        s       = la.svdvals(psi_mat)
        eigvals = s**2
        
        # Avoid log(0)
        eigvals = eigvals[eigvals > 1e-15]
        
        if abs(q - 1.0) < 1e-9:
            return -np.sum(eigvals * np.log(eigvals))
        else:
            return np.log(np.sum(eigvals**q)) / (1.0 - q)
    
    return s_func

# -------------------------------------------------------------------------------
# Main functions for entanglement spectrum and Schmidt decomposition
# -------------------------------------------------------------------------------

def run_demo():
    parser = argparse.ArgumentParser(description="MES & Topological Entropy Demo")
    parser.add_argument("--show",               action="store_true",            help="Show plots interactively")
    parser.add_argument("--flux",   type=float, nargs=2,    default=[0.0, 0.0], help="Boundary fluxes in units of pi (phi_x, phi_y)")
    parser.add_argument("--lx",     type=int,               default=2,          help="Lattice size Lx")
    parser.add_argument("--ly",     type=int,               default=2,          help="Lattice size Ly")
    parser.add_argument("--gamma",  type=float,             default=0.0,        help="Gamma for the Kitaev model")
    parser.add_argument("--k",      type=float,             default=2.0,        help="Kitaev coupling strength (default: 2.0 for strong topological phase)")
    parser.add_argument("--q",      type=float,             default=1.0,        help="Rényi index for entropy (default: 1.0 for von Neumann)")
    args = parser.parse_args()
    logger.title("QES: MES & Topological Entropy Demo", desired_size=100, fill="=")

    # Model Setup
    # Kitaev Honeycomb Model: Kx=Ky=Kz=1.0 is in the topological phase.
    lx, ly      = args.lx, args.ly
    flux        = { 'x': args.flux[0] * np.pi, 'y': args.flux[1] * np.pi }
    if flux['x'] != 0.0 or flux['y'] != 0.0:
        logger.info(f"Applying boundary fluxes: phi_x={args.flux[0]:.2f}pi, phi_y={args.flux[1]:.2f}pi")
    else:
        flux    = None
    
    lat         = HoneycombLattice(lx=lx, ly=ly, bc='pbc', flux=flux)
    model       = HeisenbergKitaev(lattice=lat, K=(args.k, 1.0, 1.0), logger=logger, gamma=(args.gamma, args.gamma, args.gamma))
    
    logger.info(f"Lattice: {lat}")
    logger.info(f"Model: {model}")
    if lat.has_flux:
        logger.info(f"Flux: {lat.flux}")

    # Diagonalization
    logger.info("Building Hamiltonian and diagonalizing...")
    model.build(verbose=False)
    model.diagonalize(method='lanczos', k=10, tol=1e-10, maxiter=1000)
    
    evals       = model.eigenvalues
    evecs       = model.eigenvectors
    logger.info(f"Ground state energy: {evals[0]:.6f}")
    
    # Plot the energy differences from the ground state to identify degeneracies
    if True:
        fig0, ax0 = plt.subplots(figsize=(6, 4))
        ax0.plot(range(len(evals)), evals - evals[0], 'o-')
        ax0.set_xlabel("Eigenstate Index")
        ax0.set_ylabel("Energy - E0")
        ax0.set_title("Energy Spectrum")
        ax0.grid(True, which='both', linestyle='--', alpha=0.5)
        savefig(fig0, "energy_spectrum.png", show=args.show, q=args.q)
        
    # Identify degenerate ground states
    gs_tol                  = 1e-6 # Higher tolerance for finite size and flux
    gs_indices              = np.where(np.abs(evals - evals[0]) < gs_tol)[0]
    n_gs                    = len(gs_indices)
    logger.info(f"Found {n_gs} degenerate ground states.")
    
    #! Take the degenerate ground state manifold for MES search
    V_gs                    = evecs[:, gs_indices]

    # MES Search
    # We test multiple cuts to find the best MES (minimum entropy across different cuts)
    cuts                    = ['half_x', 'half_y']
    best_mes_across_cuts    = None
    min_S_across_cuts       = np.inf
    best_cut_kind           = None
    best_region             = None
    
    for cut_kind in cuts:
        try:
            region_cut      = lat.get_region(kind=cut_kind)
        except Exception as e:
            logger.warning(f"Could not get region {cut_kind}: {e}")
            continue
            
        logger.info(f"Searching MES for cut: {cut_kind} (sites: {region_cut})", lvl=1, color="red")
        S_func              = get_entropy_function(region_cut, lat.Ns, q=args.q)
        
        # find_mes is our optimized function
        mes_states, mes_values, _, _ = find_mes(V_gs, S_func, n_trials=15, state_max=n_gs, n_restarts=3)
        
        for i, S in enumerate(mes_values):
            logger.info(f"MES {i} ({cut_kind}): S = {S:.6f}", lvl=2, color="blue")
            
        if len(mes_values) > 0 and mes_values[0] < min_S_across_cuts:
            min_S_across_cuts       = mes_values[0]
            best_mes_across_cuts    = mes_states[0]
            best_region             = region_cut
            best_cut_kind           = cut_kind

    if best_mes_across_cuts is None:
        logger.error("Failed to find any MES states.")
        return

    logger.info(f"Best cut found: {best_cut_kind} with S={min_S_across_cuts:.6f}")

    #! Topological Entanglement Entropy (TEE)
    # We test multiple radii for KP and LW to see stability
    radii_kp    = [1.0, 1.1, 1.2]
    radii_lw    = [0.8, 1.0, 1.3]
    tee_results = []
    
    logger.info("Calculating Topological Entanglement Entropy (TEE) for multiple cuts and radii...", lvl=1, color="green")
    logger.info(f"Kitaev-Preskill radii: {radii_kp}", lvl=2)
    for r in radii_kp:
        try:
            reg = lat.get_region(kind='kitaev_preskill', radius=r)
            res = topological_entropy(best_mes_across_cuts, reg, lat.Ns, topological='kitaev_preskill', q=args.q)
            logger.info(f"TEE (Kitaev-Preskill, r={r:.1f}) = {res['gamma']:.6f}")
            tee_results.append(('KP', r, res))
        except Exception as e:
            logger.warning(f"KP radius {r} failed: {e}")

    logger.info(f"Levin-Wen radii: {radii_lw}", lvl=2)
    for r in radii_lw:
        try:
            # Levin-Wen usually uses inner and outer radius
            reg = lat.get_region(kind='levin_wen', inner_radius=r*0.7, outer_radius=r*1.2)
            res = topological_entropy(best_mes_across_cuts, reg, lat.Ns, topological='levin_wen', q=args.q)
            logger.info(f"TEE (Levin-Wen, r_mid={r:.1f}) = {res['gamma']:.6f}")
            tee_results.append(('LW', r, res))
        except Exception as e:
            logger.warning(f"LW radius {r} failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Visualisation
    # ═══════════════════════════════════════════════════════════════════════════

    # Plot 1: Best Partition Regions
    fig1, axes1     = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use the last successful KP and LW regions for plotting
    kp_res          = [res for kind, r, res in tee_results if kind == 'KP']
    lw_res          = [res for kind, r, res in tee_results if kind == 'LW']
    
    if kp_res:
        reg_kp      = kp_res[-1]['regions']
        kp_abc      = {k: v for k, v in reg_kp.items() if len(k) == 1}
        lat.plot.regions(
            kp_abc, ax=axes1[0], title=f"Kitaev-Preskill (r={radii_kp[-1]})",
            fill=True, fill_alpha=0.15, show_bonds=True
        )
    
    if lw_res:
        reg_lw      = lw_res[-1]['regions']
        lw_abc      = {k: v for k, v in reg_lw.items() if k in ['inner', 'outer']}
        lat.plot.regions(
            lw_abc, ax=axes1[1], title="Levin-Wen Partition",
            fill=True, fill_alpha=0.15, show_bonds=True
        )    
    savefig(fig1, "lattice_regions.png", show=args.show, q=args.q)

    # Plot 2: Entanglement Spectrum
    fig2, ax2       = plt.subplots(figsize=(7, 6))
    es_mes          = calculate_entanglement_spectrum(best_mes_across_cuts, best_region, lat.Ns)
    ax2.plot(range(len(es_mes)), es_mes, 'o-', label=f"MES ({best_cut_kind}, S={min_S_across_cuts:.4f})", markersize=8)
    
    # Random State in GS Manifold Spectrum
    c_rand          = np.random.randn(n_gs) + 1j*np.random.randn(n_gs)
    c_rand         /= np.linalg.norm(c_rand)
    psi_rand        = V_gs @ c_rand
    es_rand         = calculate_entanglement_spectrum(psi_rand, best_region, lat.Ns)
    S_rand          = entanglement_entropy_from_spectrum(es_rand, q=args.q)
    ax2.plot(range(len(es_rand)), es_rand, 's--', label=f"Random GS (S={S_rand:.4f})", alpha=0.7)
    
    ax2.set_xlabel("Level Index $n$")
    ax2.set_ylabel(r"Entanglement Level $\xi_n = -\ln \lambda_n$")
    ax2.set_title(f"Entanglement Spectrum ({best_cut_kind})")
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    savefig(fig2, "entanglement_spectrum.png", show=args.show, q=args.q)

    # Entropy Statistics
    fig3, ax3       = plt.subplots(figsize=(7, 6))
    
    # Sample many random states in GS manifold and compute their entropy
    S_func_best     = get_entropy_function(best_region, lat.Ns, q=args.q)
    if n_gs > 1:
        n_samples       = 300
        entropies_rand  = []
        for _ in range(n_samples):
            c           = np.random.randn(n_gs) + 1j*np.random.randn(n_gs)
            c          /= np.linalg.norm(c)
            entropies_rand.append(S_func_best(V_gs @ c))
        
        # Only plot histogram if there is a non-zero range to avoid ValueError
        if np.ptp(entropies_rand) > 1e-10:
            ax3.hist(entropies_rand, bins=30, density=True, alpha=0.7, label="Random GS Superpositions")
        else:
            logger.info("Entropy range is zero; skipping histogram.")
    else:
        logger.info("Only one ground state found; skipping random sampling.")
    
    # Mark MES value
    ax3.axvline(min_S_across_cuts, color='tab:red', linestyle='--', linewidth=2, label="MES")
        
    ax3.set_xlabel("Entanglement Entropy $S$")
    ax3.set_ylabel("Probability Density")
    ax3.set_title(f"Entropy Distribution ({best_cut_kind})")
    ax3.legend()
    
    # Add text box with average TEE values
    if kp_res and lw_res:
        gamma_kp    = np.mean([r['gamma'] for r in kp_res])
        gamma_lw    = np.mean([r['gamma'] for r in lw_res])
        textstr     = "\n".join((
            r"$\langle \gamma_{KP} \rangle = %.4f$" % (gamma_kp, ),
            r"$\langle \gamma_{LW} \rangle = %.4f$" % (gamma_lw, )
        ))
        props       = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    savefig(fig3, "entropy_statistics.png", show=args.show, q=args.q)

# ---------------------------------------

if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    run_demo()

# ---------------------------------------
#! EOF
# ---------------------------------------