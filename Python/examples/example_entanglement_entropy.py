"""
Fixed entanglement entropy computation with proper imports and region handling.
"""

import numpy as np
from typing import Tuple, List, Dict, Union

def get_entanglement_entropy_fixed(hamil, hilbert, subsystem: Union[int, float]) -> Tuple[List[float], Dict, List[float], List[float], Dict]:
    """Compute entanglement entropy for all eigenstates with proper error handling."""
    
    # FIXED: Correct imports
    try:
        from QES.Algebra.Symmetries.jit.density_jit        import rho_symmetries
        from QES.general_python.physics.density_matrix     import rho_spectrum
        from QES.general_python.physics.entropy            import entropy as entropy_func
        from QES.Algebra.Symmetries.jit.entropy_jit        import mutual_information, topological_entropy
    except ImportError as e:
        raise ImportError(f"Failed to import entanglement entropy modules: {e}")
        
    # Determine subsystem size
    ns          = hamil.lattice.ns
    fraction    = subsystem if subsystem < 1.0 else subsystem / ns
    va          = int(fraction * ns)
    
    # Storage
    entropies       = []
    topological     = []
    purity_list     = []
    mut_info_list   = []
    regions_topo    = {}
    num_states      = hamil.eig_val.shape[0]
    
    logger = hamil.logger if hasattr(hamil, 'logger') else None
    
    # ============================================================================
    # 1. Bipartite Entanglement Entropy (contiguous subsystem)
    # ============================================================================
    for state_idx in range(num_states):
        psi = np.ascontiguousarray(hamil.eig_vec[:, state_idx])
        
        try:
            # Compute reduced density matrix for first va sites
            rho         = rho_symmetries(psi, va=va, hilbert=hilbert)
            vals        = rho_spectrum(rho)
            purity_val  = np.sum(vals**2)
            S_ent       = entropy_func(vals, q=1.0)  # Von Neumann entropy
            
            entropies.append(S_ent)
            purity_list.append(purity_val)
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to compute entropy for state {state_idx}: {e}", lvl=0)
            entropies.append(np.nan)
            purity_list.append(np.nan)
    
    if logger:
        logger.info(f"Computed entanglement entropy for {num_states} states (va={va} sites)", color='green', lvl=1)
    
    # ============================================================================
    # 2. Topological Entanglement Entropy (Kitaev-Preskill)
    # ============================================================================
    try:
        # FIXED: Check if lattice is large enough for meaningful TEE
        min_system_size = 6  # Need at least 6 sites for meaningful regions
        
        if ns >= min_system_size:
            # Get regions with explicit radius to ensure proper coverage
            # For TEE, we want regions to cover ~40-60% of system (not 100%)
            target_radius = 0.4 * min(hamil.lattice.lx, hamil.lattice.ly)
            
            regions_topo = hamil.lattice.regions.region_kitaev_preskill(
                origin=None,  # Use center
                radius=target_radius
            )
            
            # Verify regions are non-trivial
            total_sites_in_ABC = len(set(regions_topo['A']) | set(regions_topo['B']) | set(regions_topo['C']))
            
            if total_sites_in_ABC >= ns:
                if logger:
                    logger.warning(f"Regions A∪B∪C cover entire system ({total_sites_in_ABC}/{ns} sites). TEE will be unreliable.", lvl=1)
            
            if logger:
                logger.info(f"TEE regions: |A|={len(regions_topo['A'])}, |B|={len(regions_topo['B'])}, "
                           f"|C|={len(regions_topo['C'])}, |ABC|={total_sites_in_ABC}/{ns}", lvl=2)
            
            # Compute TEE for ground state and low-lying excited states
            states_to_calc_tee = min(num_states, 5)
            
            for state_idx in range(states_to_calc_tee):
                psi_cont = np.ascontiguousarray(hamil.eig_vec[:, state_idx])
                
                # This returns {'gamma': float, 'entropies': dict, 'regions': dict}
                tee_result = topological_entropy(
                    state=psi_cont,
                    regions=regions_topo,
                    hilbert=hilbert,
                    q=1.0  # Von Neumann
                )
                topological.append(tee_result)
                
                if logger and state_idx == 0:
                    gamma = tee_result['gamma']
                    logger.info(f"Ground state TEE: γ = {gamma:.6f} (expected ≈ 0.693 for toric code)", color='cyan', lvl=2)
        else:
            if logger:
                logger.warning(f"System too small (ns={ns}) for reliable TEE calculation. Skipping.", lvl=1)
                
    except Exception as e:
        if logger:
            logger.error(f"Failed to compute topological entanglement entropy: {e}", lvl=0)
        import traceback
        traceback.print_exc()
    
    # ============================================================================
    # 3. Mutual Information (all site pairs)
    # ============================================================================
    try:
        states_to_calc_mi = min(num_states, 5)  # Only compute for a few states
        
        for state_idx in range(states_to_calc_mi):
            psi = np.ascontiguousarray(hamil.eig_vec[:, state_idx])
            mut_info_state = {}
            
            # Compute for all pairs (can be slow for large systems)
            for i in range(min(ns, 9)):  # Limit to first 9 sites for large systems
                for j in range(i+1, min(ns, 9)):
                    info = mutual_information(
                        state=psi,
                        i=i,
                        j=j,
                        hilbert=hilbert,
                        q=1.0
                    )
                    mut_info_state[(i, j)] = info
                    
            mut_info_list.append(mut_info_state)
            
        if logger:
            logger.info(f"Computed mutual information for {states_to_calc_mi} states", color='green', lvl=1)
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to compute mutual information: {e}", lvl=0)
    
    # ============================================================================
    # Return results
    # ============================================================================
    gamma_values = np.array([t.get('gamma', np.nan) for t in topological]) if topological else np.array([])
    
    return entropies, mut_info_list, purity_list, gamma_values, regions_topo


# ============================================================================
# Diagnostic function to check region validity
# ============================================================================

def check_region_validity(lattice, regions: Dict[str, List[int]], logger=None):
    """
    Validate Kitaev-Preskill regions for TEE calculation.
    
    Returns
    -------
    bool
        True if regions are valid for TEE, False otherwise.
    """
    ns = lattice.ns
    
    # Extract individual regions
    A = set(regions.get('A', []))
    B = set(regions.get('B', []))
    C = set(regions.get('C', []))
    
    # Check for overlaps (should be disjoint)
    overlap_AB = A & B
    overlap_BC = B & C
    overlap_AC = A & C
    
    if overlap_AB or overlap_BC or overlap_AC:
        if logger:
            logger.error(f"Regions have overlaps: AB={len(overlap_AB)}, BC={len(overlap_BC)}, AC={len(overlap_AC)}", lvl=0)
        return False
    
    # Check sizes
    total = len(A | B | C)
    
    if total == 0:
        if logger:
            logger.error("All regions are empty!", lvl=0)
        return False
    
    if total >= ns:
        if logger:
            logger.warning(f"Regions cover entire system ({total}/{ns} sites). TEE will be 0 for pure states!", lvl=1)
        return False
    
    # For meaningful TEE, regions should cover 30-70% of system
    fraction = total / ns
    if fraction < 0.3 or fraction > 0.7:
        if logger:
            logger.warning(f"Regions cover {fraction*100:.1f}% of system. Optimal range is 30-70%.", lvl=1)
    
    # Check that regions are comparable in size (balanced Y-junction)
    sizes = [len(A), len(B), len(C)]
    max_size = max(sizes)
    min_size = min(sizes)
    
    if max_size > 3 * min_size:
        if logger:
            logger.warning(f"Region sizes are imbalanced: A={len(A)}, B={len(B)}, C={len(C)}", lvl=1)
    
    if logger:
        logger.info(f"✓ Regions valid: |A|={len(A)}, |B|={len(B)}, |C|={len(C)}, |ABC|={total}/{ns} ({fraction*100:.1f}%)", 
                   color='green', lvl=2)
    
    return True


# ============================================================================
# Alternative: Manual region definition for small lattices
# ============================================================================

def get_manual_kp_regions_3x3() -> Dict[str, List[int]]:
    """
    Manually define Kitaev-Preskill regions for a 3x3 lattice.
    
    Site layout (9 sites):
        0 1 2
        3 4 5
        6 7 8
    
    We create three regions meeting at the center (site 4):
    - A: left sector {0, 3, 6}
    - B: right sector {2, 5, 8}
    - C: top sector {1}
    
    This leaves {4, 7} as region D (complement).
    """
    regions = {
        'A': [0, 3, 6],
        'B': [2, 5, 8],
        'C': [1],
    }
    
    # Compute combinations
    regions['AB'] = sorted(list(set(regions['A']) | set(regions['B'])))
    regions['BC'] = sorted(list(set(regions['B']) | set(regions['C'])))
    regions['AC'] = sorted(list(set(regions['A']) | set(regions['C'])))
    regions['ABC'] = sorted(list(set(regions['A']) | set(regions['B']) | set(regions['C'])))
    
    return regions
