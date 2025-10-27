"""
Momentum sector analysis and construction for Hilbert spaces.

This module provides tools for analyzing and constructing momentum sectors in quantum many-body
systems with translation symmetry. It properly handles:
- 1D systems with single translation direction
- 2D/3D systems with multiple translation directions (k_x, k_y, k_z)
- Lattices with multiple sites per unit cell (honeycomb, kagome, etc.)
- Combined momentum quantum numbers for simultaneous symmetries

Integrates with:
- HilbertSpace class for automatic sector construction
- TranslationSymmetry operators for orbit generation
- Lattice objects for proper unit cell handling

---------------------------------------------------
File        : QES/Algebra/Symmetries/momentum_sectors.py
Description : Momentum sector analysis and construction utilities
Author      : Maksymilian Kliczkowski
Date        : 2025-10-27
---------------------------------------------------
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from math import gcd

try:
    from QES.general_python.lattices.lattice import Lattice, LatticeDirection
except ImportError:
    LatticeDirection = None  # type: ignore

from QES.Algebra.Symmetries.translation import TranslationSymmetry

class MomentumSectorAnalyzer:
    """
    Analyzes momentum sector structure for systems with translation symmetry.
    
    Handles:
    - 1D translation: single k quantum number
    - 2D translation: (k_x, k_y) quantum numbers
    - 3D translation: (k_x, k_y, k_z) quantum numbers
    - Multiple sites per unit cell (honeycomb, etc.)
    
    The analyzer properly determines:
    - Number of unit cells vs total sites
    - Orbit periods under each translation
    - Allowed momentum quantum numbers
    - Representative states for each sector
    """
    
    def __init__(self, lattice: 'Lattice'):
        """
        Initialize momentum sector analyzer for a lattice.
        
        Parameters
        ----------
        lattice : Lattice
            Lattice object with translation structure.
        """
        self.lattice = lattice
        
        # Get lattice parameters
        self.ns = getattr(lattice, 'Ns', None) or getattr(lattice, 'ns', None)
        if self.ns is None:
            raise ValueError("Could not determine number of sites from lattice")
        
        # Unit cell dimensions
        self.Lx = getattr(lattice, 'Lx', None) or getattr(lattice, '_lx', None) or getattr(lattice, 'lx', None)
        self.Ly = getattr(lattice, 'Ly', None) or getattr(lattice, '_ly', None) or getattr(lattice, 'ly', None) or 1
        self.Lz = getattr(lattice, 'Lz', None) or getattr(lattice, '_lz', None) or getattr(lattice, 'lz', None) or 1
        
        if self.Lx is None:
            raise ValueError("Could not determine lattice extent Lx")
        
        # Determine sites per unit cell
        expected_sites = self.Lx * self.Ly * self.Lz
        if self.ns == expected_sites:
            self.sites_per_cell = 1
        else:
            self.sites_per_cell = self.ns // expected_sites
        
        # Determine active dimensions
        self.active_directions = []
        if self.Lx > 1:
            self.active_directions.append(LatticeDirection.X)
        if self.Ly > 1:
            self.active_directions.append(LatticeDirection.Y)
        if self.Lz > 1:
            self.active_directions.append(LatticeDirection.Z)
        
        self.dim = len(self.active_directions)
    
    def get_extent(self, direction: LatticeDirection) -> int:
        """Get lattice extent (number of unit cells) in a direction."""
        if direction == LatticeDirection.X:
            return self.Lx
        elif direction == LatticeDirection.Y:
            return self.Ly
        elif direction == LatticeDirection.Z:
            return self.Lz
        else:
            return 1
    
    def analyze_1d_sectors(
        self,
        direction: LatticeDirection = LatticeDirection.X,
        verbose: bool = False
    ) -> Dict[int, List[Tuple[int, Dict]]]:
        """
        Analyze 1D momentum sectors for a single translation direction.
        
        Parameters
        ----------
        direction : LatticeDirection
            Translation direction to analyze.
        verbose : bool
            Print detailed analysis.
        
        Returns
        -------
        Dict[int, List[Tuple[int, Dict]]]
            Mapping from momentum index k to list of (representative, info) pairs.
        """
        extent = self.get_extent(direction)
        translator = TranslationSymmetry(self.lattice, direction=direction)
        
        representatives = {}
        visited = set()
        
        for state in range(2**self.ns):
            if state in visited:
                continue
            
            # Generate orbit
            orbit = []
            current = state
            seen = set()
            
            while current not in seen:
                orbit.append(current)
                seen.add(current)
                current, _ = translator.apply(current)
            
            period = len(orbit)
            visited.update(orbit)
            
            # Representative is minimum in orbit
            representative = min(orbit)
            
            # Allowed momenta: k where k·period ≡ 0 (mod extent)
            allowed_k = []
            for q in range(extent):
                if (q * period) % extent == 0:
                    allowed_k.append(q)
            
            representatives[representative] = {
                'orbit': orbit,
                'period': period,
                'allowed_k': allowed_k,
                'norm': np.sqrt(period)
            }
        
        # Organize by momentum sector
        momentum_sectors = defaultdict(list)
        for rep_state, info in representatives.items():
            for k in info['allowed_k']:
                momentum_sectors[k].append((rep_state, info))
        
        return dict(momentum_sectors)
    
    def analyze_2d_sectors(
        self,
        directions: Optional[Tuple[LatticeDirection, LatticeDirection]] = None,
        verbose: bool = False
    ) -> Dict[Tuple[int, int], List[Tuple[int, Dict]]]:
        """
        Analyze 2D momentum sectors for two translation directions.
        
        Parameters
        ----------
        directions : Tuple[LatticeDirection, LatticeDirection], optional
            Pair of translation directions. Defaults to (X, Y).
        verbose : bool
            Print detailed analysis.
        
        Returns
        -------
        Dict[Tuple[int, int], List[Tuple[int, Dict]]]
            Mapping from (k_x, k_y) to list of (representative, info) pairs.
        """
        if directions is None:
            if len(self.active_directions) < 2:
                raise ValueError("Lattice must have at least 2 active directions for 2D analysis")
            directions = (self.active_directions[0], self.active_directions[1])
        
        dir_x, dir_y = directions
        extent_x = self.get_extent(dir_x)
        extent_y = self.get_extent(dir_y)
        
        translator_x = TranslationSymmetry(self.lattice, direction=dir_x)
        translator_y = TranslationSymmetry(self.lattice, direction=dir_y)
        
        representatives = {}
        visited = set()
        
        for state in range(2**self.ns):
            if state in visited:
                continue
            
            # Generate orbit under first direction
            orbit_x = []
            current = state
            seen_x = set()
            
            while current not in seen_x:
                orbit_x.append(current)
                seen_x.add(current)
                current, _ = translator_x.apply(current)
            
            period_x = len(orbit_x)
            
            # Generate orbit under second direction
            orbit_y = []
            current = state
            seen_y = set()
            
            while current not in seen_y:
                orbit_y.append(current)
                seen_y.add(current)
                current, _ = translator_y.apply(current)
            
            period_y = len(orbit_y)
            
            # Generate full 2D orbit
            orbit_2d = set()
            for state_x in orbit_x:
                current = state_x
                for _ in range(period_y):
                    orbit_2d.add(current)
                    current, _ = translator_y.apply(current)
            
            orbit_2d = sorted(orbit_2d)
            visited.update(orbit_2d)
            
            # Representative is minimum in 2D orbit
            representative = min(orbit_2d)
            
            # Allowed momenta in each direction
            allowed_kx = []
            for q in range(extent_x):
                if (q * period_x) % extent_x == 0:
                    allowed_kx.append(q)
            
            allowed_ky = []
            for q in range(extent_y):
                if (q * period_y) % extent_y == 0:
                    allowed_ky.append(q)
            
            representatives[representative] = {
                'orbit_x': orbit_x,
                'orbit_y': orbit_y,
                'orbit_2d': orbit_2d,
                'period_x': period_x,
                'period_y': period_y,
                'allowed_kx': allowed_kx,
                'allowed_ky': allowed_ky,
                'norm': np.sqrt(len(orbit_2d))
            }
        
        # Organize by (k_x, k_y) momentum sector
        momentum_sectors = defaultdict(list)
        for rep_state, info in representatives.items():
            for kx in info['allowed_kx']:
                for ky in info['allowed_ky']:
                    momentum_sectors[(kx, ky)].append((rep_state, info))
        
        return dict(momentum_sectors)
    
    def get_sector_representatives(
        self,
        momentum_indices: Optional[Dict[LatticeDirection, int]] = None
    ) -> List[int]:
        """
        Get representative states for specified momentum sector(s).
        
        Parameters
        ----------
        momentum_indices : Dict[LatticeDirection, int], optional
            Momentum quantum numbers for each active direction.
            If None, returns all representatives.
        
        Returns
        -------
        List[int]
            Representative states in the specified sector.
        """
        if self.dim == 0:
            # No translation symmetry
            return list(range(2**self.ns))
        
        if momentum_indices is None:
            # Return all representatives across all sectors
            if self.dim == 1:
                sectors = self.analyze_1d_sectors(self.active_directions[0])
                all_reps = set()
                for reps_list in sectors.values():
                    all_reps.update(rep for rep, _ in reps_list)
                return sorted(all_reps)
            elif self.dim == 2:
                sectors = self.analyze_2d_sectors()
                all_reps = set()
                for reps_list in sectors.values():
                    all_reps.update(rep for rep, _ in reps_list)
                return sorted(all_reps)
            else:
                raise NotImplementedError("3D momentum analysis not yet implemented")
        
        # Return representatives for specific momentum sector
        if self.dim == 1:
            direction = self.active_directions[0]
            k = momentum_indices.get(direction, 0)
            sectors = self.analyze_1d_sectors(direction)
            return [rep for rep, _ in sectors.get(k, [])]
        
        elif self.dim == 2:
            kx = momentum_indices.get(self.active_directions[0], 0)
            ky = momentum_indices.get(self.active_directions[1], 0)
            sectors = self.analyze_2d_sectors()
            return [rep for rep, _ in sectors.get((kx, ky), [])]
        
        else:
            raise NotImplementedError("3D momentum analysis not yet implemented")


def build_momentum_basis(
    lattice: 'Lattice',
    momentum_indices: Dict[LatticeDirection, int],
    normalize: bool = True
) -> Dict[int, Dict[int, complex]]:
    """
    Build momentum-resolved basis states for a lattice system.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice with translation symmetry.
    momentum_indices : Dict[LatticeDirection, int]
        Momentum quantum number for each translation direction.
    normalize : bool
        Whether to normalize the momentum eigenstates.
    
    Returns
    -------
    Dict[int, Dict[int, complex]]
        Mapping from representative state to {basis_state: coefficient} dictionary.
    """
    analyzer = MomentumSectorAnalyzer(lattice)
    representatives = analyzer.get_sector_representatives(momentum_indices)
    
    # Build translators for each active direction
    translators = {}
    for direction in analyzer.active_directions:
        k = momentum_indices.get(direction, 0)
        translators[direction] = TranslationSymmetry(lattice, direction=direction, momentum_index=k)
    
    momentum_basis = {}
    
    for rep in representatives:
        # Build superposition from representative
        superposition = {rep: 1.0 + 0j}
        
        # Apply momentum projectors for each direction
        for direction, translator in translators.items():
            k = momentum_indices.get(direction, 0)
            new_superposition = defaultdict(complex)
            
            for state, amp in superposition.items():
                orbit = translator.orbit(state)
                period = len(orbit)
                
                for idx, orb_state in enumerate(orbit):
                    phase = np.exp(-1j * 2 * np.pi * k * idx / period)
                    new_superposition[orb_state] += amp * phase
            
            superposition = dict(new_superposition)
        
        # Normalize if requested
        if normalize:
            norm = np.sqrt(sum(abs(amp)**2 for amp in superposition.values()))
            if norm > 1e-10:
                superposition = {state: amp/norm for state, amp in superposition.items()}
        
        momentum_basis[rep] = superposition
    
    return momentum_basis


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
