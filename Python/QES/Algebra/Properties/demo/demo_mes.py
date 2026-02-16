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

import  sys, os
import  argparse
import  numpy as np
import  pandas as pd
import  matplotlib as mpl
import  matplotlib.pyplot as plt
import  scipy.linalg as la
from    pathlib import Path
from    typing import Tuple, List, Dict, Any, Union, Optional

from traitlets import default

#! project import
# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))

try:
    from QES.general_python.common.flog                         import get_global_logger
    from QES.general_python.physics.density_matrix              import schmidt
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev   import HeisenbergKitaev, HoneycombLattice
    from QES.Algebra.Properties.mes                             import find_mes, compute_modular_s_matrix
    from QES.Algebra.Properties.entanglement_spectrum           import (
                                                                    calculate_entanglement_spectrum, 
                                                                    entanglement_entropy_from_spectrum
                                                                )
    from QES.general_python.physics.entropy                     import topological_entropy, entropy
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
    name    = f"{ITER:03d}_demo_{name}_q{q:.2f}.png"
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
    
    def s_func(psi):
        # Use optimized schmidt call from density_matrix
        eigvals = schmidt(psi, va=region, ns=ns, eig=False, square=True, return_vecs=False)
        return entropy(eigvals, q=q)
    
    return s_func

# -------------------------------------------------------------------------------

def _handle_trials(args, model, logger):
    """Handle trial parameters based on system size to prevent long runtimes."""
    if model.lattice.Ns > 12:
        logger.warning(f"Large system (Ns={model.lattice.Ns}), reducing MES search trials to avoid long runtime.", color="yellow")
        n_trials    = min(args.n_trials, 8)
        n_restarts  = min(args.n_restarts, 1)
        max_iter    = min(args.max_iter, 50)
    else:
        n_trials    = args.n_trials
        n_restarts  = args.n_restarts
        max_iter    = args.max_iter
    
    return n_trials, n_restarts, max_iter

def _handle_cut(lat: HoneycombLattice, cut_kind: str, logger, args) -> Union[List[int], None]:
    """Handle cut region retrieval with error handling."""
    try:
        region_cut = lat.get_region(kind=cut_kind)
        
        # Plot the cut region on the lattice
        fig_cut, ax_cut = plt.subplots(figsize=(5, 5))
        
        # Visualize the cut region on the lattice
        fig_cut, ax_cut = plt.subplots(figsize=(5, 5))
        lat.plot.regions({cut_kind: region_cut}, ax=ax_cut, fill=True, fill_alpha=0.3, color="cyan", show_bonds=True)
        savefig(fig_cut, f"cut_{cut_kind}.png", show=args.show, q=args.q)
        
        return region_cut
    except Exception as e:
        logger.warning(f"Could not get region {cut_kind}: {e}")
        return None

# -------------------------------------------------------------------------------
# Main functions for entanglement spectrum and Schmidt decomposition
# -------------------------------------------------------------------------------

def run_demo():
    parser = argparse.ArgumentParser(description="MES & Topological Entropy Demo")
    parser.add_argument("--show",       action="store_true",                        help="Show plots interactively")
    parser.add_argument("--flux",       type=float, nargs=2,    default=[0.0, 0.0], help="Boundary fluxes in units of pi (phi_x, phi_y)")
    parser.add_argument("--lx",         type=int,               default=2,          help="Lattice size Lx")
    parser.add_argument("--ly",         type=int,               default=2,          help="Lattice size Ly")
    parser.add_argument("--gamma",      type=float,             default=0.0,        help="Gamma for the Kitaev model")
    parser.add_argument("--k",          type=float,             default=2.2,        help="Kitaev coupling strength (default: 2.2 for strong topological phase)")
    parser.add_argument("--q",          type=float,             default=1.0,        help="Rényi index for entropy (default: 1.0 for von Neumann)")
    parser.add_argument("--m",          type=int,               default=None,       help="Number of MES states to find (default: all degenerate GS)")
    
    # Optimization parameters to prevent running forever
    parser.add_argument("--n_trials",   type=int,               default=15,         help="Number of random trials for MES search")
    parser.add_argument("--n_restarts", type=int,               default=2,          help="Number of local restarts per trial")
    parser.add_argument("--max_iter",   type=int,               default=100,        help="Max iterations for entropy minimization")
    
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
    if args.m is not None:
        n_gs                = args.m
        gs_indices          = [0] + list(range(1, 1+n_gs-1)) # Take the first m states (including the ground state)
    else:
        gs_indices          = np.where(np.abs(evals - evals[0]) < gs_tol)[0]
        n_gs                = len(gs_indices)
    logger.info(f"Found {n_gs} degenerate ground states.")
    
    #! Take the degenerate ground state manifold for MES search
    V_gs                    = evecs[:, gs_indices]

    # MES Search
    # We test multiple cuts to find the best MES (minimum entropy across different cuts)
    cuts                    = ['half_x', 'half_y']
    mes_results             = {}
    
    # Scale parameters if system is large to avoid timeout
    n_trials, n_restarts, max_iter = _handle_trials(args, model, logger)
    
    for cut_kind in cuts:
        region_cut  = _handle_cut(lat, cut_kind, logger, args)
        
        logger.info(f"Searching MES for cut: {cut_kind} (sites: {region_cut})", lvl=1, color="red")
        S_func      = get_entropy_function(region_cut, lat.Ns, q=args.q)
        
        # find_mes is our optimized function
        # We need to find all unique MES states to build the modular S-matrix
        mes_states, mes_values, coeffs, _ = find_mes(
            V_gs, S_func, 
            n_trials=n_trials, state_max=n_gs, n_restarts=n_restarts, 
            max_iter=max_iter, verbose=False
        )
        
        for i, S in enumerate(mes_values):
            logger.info(f"MES {i} ({cut_kind}): S = {S:.6f}", lvl=2, color="blue")
            
        mes_results[cut_kind] = {
            'states'    : mes_states,
            'values'    : mes_values,
            'region'    : region_cut
        }

    # Best MES for TEE and plots
    available_cuts          = list(mes_results.keys())
    if not available_cuts:
        logger.error("No cuts available for MES search.")
        return

    best_cut_kind           = min(available_cuts, key=lambda k: mes_results[k]['values'][0] if mes_results[k]['values'] else np.inf)
    
    if not mes_results[best_cut_kind]['states']:
        logger.error("Failed to find any MES states.")
        return

    best_mes_across_cuts    = mes_results[best_cut_kind]['states'][0]
    min_S_across_cuts       = mes_results[best_cut_kind]['values'][0]
    best_region             = mes_results[best_cut_kind]['region']

    logger.info(f"Best cut found: {best_cut_kind} with S={min_S_across_cuts:.6f}")

    # Modular S-Matrix Analysis
    topo_res = None
    if 'half_x' in mes_results and 'half_y' in mes_results:
        states_x = mes_results['half_x']['states']
        states_y = mes_results['half_y']['states']
        if len(states_x) == len(states_y) and len(states_x) > 0:
            logger.info("Computing Modular S-matrix and topological statistics...", lvl=1, color="magenta")
            topo_res = compute_modular_s_matrix(states_x, states_y)
            
            logger.info(f"Topological Results: {topo_res}", lvl=2)
            logger.info(f"Quantum Dimensions: {topo_res.quantum_dimensions}", lvl=2)
            logger.info(f"Total Quantum Dimension D: {topo_res.total_quantum_dimension:.6f}", lvl=2)
            
            # Print S-matrix
            s_mat_str = np.array2string(topo_res.S_matrix, precision=4, suppress_small=True)
            logger.info(f"Modular S-matrix:\n{s_mat_str}", lvl=2)
            
            # Plot S-matrix heatmap
            fig_s, ax_s = plt.subplots(figsize=(6, 5))
            im = ax_s.imshow(np.abs(topo_res.S_matrix), cmap='magma')
            plt.colorbar(im, ax=ax_s, label='|S_ij|')
            ax_s.set_title("Modular S-matrix (Magnitude)")
            ax_s.set_xlabel("Basis Y")
            ax_s.set_ylabel("Basis X")
            savefig(fig_s, "modular_s_matrix.png", show=args.show, q=args.q)
            
        else:
            logger.warning(f"Bases size mismatch for S-matrix: x={len(states_x)}, y={len(states_y)}")

    #! Topological Entanglement Entropy (TEE)
    # We test multiple radii for KP and LW to see stability
    # For small lattices (2x2), we use origin=lat.Ns//2 to ensure non-empty regions
    radii_kp    = [1.1, 1.8, 2.7]
    radii_lw    = [1.1, 1.8, 2.7]
    tee_results = []
    
    logger.info("Calculating Topological Entanglement Entropy (TEE) for multiple cuts and radii...", lvl=1, color="green")
    logger.info(f"Kitaev-Preskill radii: {radii_kp}", lvl=2)
    
    for r in radii_kp:
        try:
            reg = lat.get_region(kind='kitaev_preskill', radius=r, origin=lat.Ns//2)
            res = topological_entropy(best_mes_across_cuts, reg, lat.Ns, topological='kitaev_preskill', q=args.q)
            logger.info(f"TEE (Kitaev-Preskill, r={r:.1f}) = {res['gamma']:.6f}", lvl=2, color="cyan")
            tee_results.append(('KP', r, res))
        except Exception as e:
            logger.warning(f"KP radius {r} failed: {e}")

    logger.info(f"Levin-Wen radii: {radii_lw}", lvl=2)
    for r in radii_lw:
        try:
            # Levin-Wen usually uses inner and outer radius
            # We tune these to split the distance groups from origin=lat.Ns//2
            reg     = lat.get_region(kind='levin_wen', origin=lat.Ns//2, inner_radius=r*0.8, outer_radius=r*1.2)
            res     = topological_entropy(best_mes_across_cuts, reg, lat.Ns, topological='levin_wen', q=args.q)
            
            # Debug site counts
            counts  = {k: len(v) for k, v in reg.items()}
            logger.info(f"TEE (Levin-Wen, r_mid={r:.1f}) = {res['gamma']:.6f} | Sizes: {counts}", lvl=2)
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
        lw_abc      = {k: v for k, v in reg_lw.items() if k in ['A', 'B', 'C']}
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

    try:
        # Save Results to CSV
        results = {
            'lx'            : lx,
            'ly'            : ly,
            'ns'            : lat.Ns,
            'k'             : args.k,
            'gamma'         : args.gamma,
            'flux_x'        : args.flux[0],
            'flux_y'        : args.flux[1],
            'q'             : args.q,
            'n_gs'          : n_gs,
            'best_cut'      : best_cut_kind,
            'min_S'         : min_S_across_cuts,
            'gamma_kp_avg'  : np.mean([r['gamma'] for r in kp_res]) if kp_res else np.nan,
            'gamma_lw_avg'  : np.mean([r['gamma'] for r in lw_res]) if lw_res else np.nan,
            'total_D'       : topo_res.total_quantum_dimension if topo_res else np.nan,
            'quantum_dims'  : str(topo_res.quantum_dimensions.tolist()) if topo_res else "",
            'is_abelian'    : topo_res.is_abelian if topo_res else np.nan,
            'is_non_abelian': topo_res.is_non_abelian if topo_res else np.nan,
        }
        
        csv_path    = SAVE_DIR / "demo_results.csv"
        df          = pd.DataFrame([results])
        
        if csv_path.exists():
            df_old  = pd.read_csv(csv_path)
            params  = ['lx', 'ly', 'k', 'gamma', 'flux_x', 'flux_y', 'q']
            
            # Ensure types match for comparison
            for p in params:
                if p in df_old.columns:
                    df_old[p] = df_old[p].astype(type(results[p]))

            # Check if row with same params exists
            matches = (df_old[params] == df[params].iloc[0]).all(axis=1)
            if matches.any():
                idx = df_old.index[matches][0]
                for col in df.columns:
                    df_old.at[idx, col] = df.at[0, col]
                df = df_old
            else:
                df = pd.concat([df_old, df], ignore_index=True)
                
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to CSV -> {csv_path}")
        
        # Also save S-matrix to a separate file for inspection
        if topo_res:
            s_path = SAVE_DIR / f"demo_s_matrix_lx{lx}_ly{ly}_k{args.k:.2f}.txt"
            np.savetxt(s_path, topo_res.S_matrix)
            logger.info(f"Saved S-matrix to -> {s_path}")

    except Exception as e:
        logger.warning(f"Failed to save results to CSV: {e}")

# ---------------------------------------

if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    run_demo()

# ---------------------------------------
#! EOF
# ---------------------------------------
