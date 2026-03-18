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

IMPORTANT:
    CURRENTLY, INTEGER REPRESENTATION OF STATES IS ASSUMED (e.g., 64-bit integers for up to 64 sites for spin-1/2 systems).

---------------------------------------------------
File        : QES/Algebra/Symmetries/momentum_sectors.py
Description : Momentum sector analysis and construction utilities
Author      : Maksymilian Kliczkowski
Date        : 2025-10-27
---------------------------------------------------
"""


from numba import njit
from numba.typed import List as TypedList

@njit(cache=True)
def _find_orbits_dfs_numba(ns: int, base: int, g_invs) -> list[int]:
    num_perms = g_invs.shape[0]
    reps = []

    base_pow = np.zeros(ns, dtype=np.int64)
    p = 1
    for i in range(ns):
        base_pow[i] = p
        p *= base

    S = np.full(ns, -1, dtype=np.int32)

    idx = ns - 1
    v = 0
    prefix_val = 0

    while True:
        if v < base:
            S[idx] = v
            new_prefix = prefix_val + v * base_pow[idx]

            valid = True
            for i in range(num_perms):
                g_inv = g_invs[i]
                is_greater = False
                for k in range(ns - 1, idx - 1, -1):
                    val_s = S[k]
                    g_inv_k = g_inv[k]
                    if g_inv_k >= idx:
                        val_g = S[g_inv_k]
                        if val_s < val_g:
                            break
                        elif val_s > val_g:
                            is_greater = True
                            break
                    else:
                        break

                if is_greater:
                    valid = False
                    break

            if valid:
                if idx == 0:
                    reps.append(new_prefix)
                    v += 1
                else:
                    prefix_val = new_prefix
                    idx -= 1
                    v = 0
            else:
                v += 1
        else:
            S[idx] = -1
            idx += 1
            if idx >= ns:
                break
            v = S[idx] + 1
            prefix_val -= S[idx] * base_pow[idx]

    return reps

@njit(cache=True)
def _build_group_permutations(generators, ns: int):
    capacity = 16
    group = np.zeros((capacity, ns), dtype=np.int64)

    for i in range(ns):
        group[0, i] = i

    count = 1
    q_head = 0

    while q_head < count:
        current = group[q_head].copy()
        q_head += 1

        for g_idx in range(generators.shape[0]):
            g = generators[g_idx]
            new_p = np.zeros(ns, dtype=np.int64)
            for i in range(ns):
                new_p[i] = g[current[i]]

            is_in_group = False
            for c_idx in range(count):
                match = True
                for i in range(ns):
                    if group[c_idx, i] != new_p[i]:
                        match = False
                        break
                if match:
                    is_in_group = True
                    break

            if not is_in_group:
                if count == capacity:
                    new_capacity = capacity * 2
                    new_group = np.zeros((new_capacity, ns), dtype=np.int64)
                    for i in range(count):
                        for j in range(ns):
                            new_group[i, j] = group[i, j]
                    group = new_group
                    capacity = new_capacity

                for i in range(ns):
                    group[count, i] = new_p[i]
                count += 1

    res = np.zeros((count, ns), dtype=np.int64)
    for i in range(count):
        for j in range(ns):
            res[i, j] = group[i, j]

    return res

@njit(cache=True)
def _get_orbits_1d(reps, ns, base, perm_fwd):
    base_pow = np.zeros(ns, dtype=np.int64)
    p = 1
    for i in range(ns):
        base_pow[i] = p
        p *= base

    num_reps = len(reps)
    orbits = []

    for r_idx in range(num_reps):
        r = reps[r_idx]
        orbit = [r]
        seen = {r}

        cur_arr = np.zeros(ns, dtype=np.int32)
        temp = r
        for i in range(ns):
            cur_arr[i] = temp % base
            temp //= base

        while True:
            next_arr = np.zeros(ns, dtype=np.int32)
            next_val = 0
            for i in range(ns):
                val = cur_arr[perm_fwd[i]]
                next_arr[i] = val
                next_val += val * base_pow[i]

            if next_val in seen:
                break

            seen.add(next_val)
            orbit.append(next_val)
            for i in range(ns):
                cur_arr[i] = next_arr[i]

        orbits.append(orbit)

    return orbits

@njit(cache=True)
def _get_orbits_2d(reps, ns, base, perm_x, perm_y):
    base_pow = np.zeros(ns, dtype=np.int64)
    p = 1
    for i in range(ns):
        base_pow[i] = p
        p *= base

    num_reps = len(reps)
    orbits_x = []
    orbits_y = []
    orbits_2d = []

    for r_idx in range(num_reps):
        r = reps[r_idx]

        cur_arr = np.zeros(ns, dtype=np.int32)
        temp = r
        for i in range(ns):
            cur_arr[i] = temp % base
            temp //= base

        # orbit_x
        orb_x = [r]
        seen_x = {r}
        temp_arr = cur_arr.copy()
        while True:
            next_arr = np.zeros(ns, dtype=np.int32)
            next_val = 0
            for i in range(ns):
                val = temp_arr[perm_x[i]]
                next_arr[i] = val
                next_val += val * base_pow[i]
            if next_val in seen_x:
                break
            seen_x.add(next_val)
            orb_x.append(next_val)
            for i in range(ns):
                temp_arr[i] = next_arr[i]
        orbits_x.append(orb_x)

        # orbit_y
        orb_y = [r]
        seen_y = {r}
        temp_arr = cur_arr.copy()
        while True:
            next_arr = np.zeros(ns, dtype=np.int32)
            next_val = 0
            for i in range(ns):
                val = temp_arr[perm_y[i]]
                next_arr[i] = val
                next_val += val * base_pow[i]
            if next_val in seen_y:
                break
            seen_y.add(next_val)
            orb_y.append(next_val)
            for i in range(ns):
                temp_arr[i] = next_arr[i]
        orbits_y.append(orb_y)

        # orbit_2d
        orb_2d_set = {r}
        orb_2d_set.remove(r) # empty set

        for st in orb_x:
            st_arr = np.zeros(ns, dtype=np.int32)
            temp = st
            for i in range(ns):
                st_arr[i] = temp % base
                temp //= base

            temp_arr = st_arr.copy()
            for step_y in range(len(orb_y)):
                val_2d = 0
                for i in range(ns):
                    val_2d += temp_arr[i] * base_pow[i]

                orb_2d_set.add(val_2d)

                next_arr = np.zeros(ns, dtype=np.int32)
                for i in range(ns):
                    next_arr[i] = temp_arr[perm_y[i]]
                for i in range(ns):
                    temp_arr[i] = next_arr[i]

        orb_2d = []
        for val in orb_2d_set:
            orb_2d.append(val)

        orb_2d_arr = np.zeros(len(orb_2d), dtype=np.int64)
        for i in range(len(orb_2d)):
            orb_2d_arr[i] = orb_2d[i]
        orb_2d_arr.sort()

        orb_2d_list = []
        for i in range(len(orb_2d_arr)):
            orb_2d_list.append(orb_2d_arr[i])

        orbits_2d.append(orb_2d_list)

    return orbits_x, orbits_y, orbits_2d


from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from QES.general_python.lattices.lattice import Lattice, LatticeDirection
except ImportError:
    LatticeDirection = None  # type: ignore

# --------------------------------------------------

try:
    from QES.Algebra.Symmetries.translation import TranslationSymmetry
except ImportError:
    raise ImportError("TranslationSymmetry could not be imported. Check your QES installation.")

# --------------------------------------------------


class MomentumSectorAnalyzer:
    """
    Analyzes momentum sector structure for systems with translation symmetry.

    Handles:
    - 1D translation    : single k quantum number
    - 2D translation    : (k_x, k_y) quantum numbers
    - 3D translation    : (k_x, k_y, k_z) quantum numbers
    - Multiple sites per unit cell (honeycomb, etc.)

    The analyzer properly determines:
    - Number of unit cells vs total sites in lattice
        - This considers lattices with multiple sites per unit cell, which is important for correct momentum quantization
    - Orbit periods under each translation
        - Important for determining allowed momentum values
    - Allowed momentum quantum numbers
        - These are determined by the lattice geometry and the active translation directions
    - Representative states for each sector
    """

    def __init__(self, lattice: "Lattice") -> None:
        """
        Initialize momentum sector analyzer for a lattice.

        Parameters
        ----------
        lattice : Lattice
            Lattice object with translation structure. See general_python dependencies.
        """
        self.lattice = lattice
        self.ns = getattr(lattice, "Ns", None) or getattr(lattice, "ns", None)
        if self.ns is None:
            raise ValueError("Could not determine number of sites from lattice")

        # Unit cell dimensions
        self.Lx = (
            getattr(lattice, "Lx", None)
            or getattr(lattice, "_lx", None)
            or getattr(lattice, "lx", None)
        )
        self.Ly = (
            getattr(lattice, "Ly", None)
            or getattr(lattice, "_ly", None)
            or getattr(lattice, "ly", None)
            or 1
        )
        self.Lz = (
            getattr(lattice, "Lz", None)
            or getattr(lattice, "_lz", None)
            or getattr(lattice, "lz", None)
            or 1
        )

        if self.Lx is None:
            raise ValueError("Could not determine lattice extent Lx")

        # Determine sites per unit cell
        expected_sites = self.Lx * self.Ly * self.Lz
        if self.ns == expected_sites:
            self.sites_per_cell = 1
        else:
            self.sites_per_cell = self.ns // expected_sites  # Assume integer division

        # Determine active dimensions
        self.active_directions = []
        if self.Lx > 1:
            self.active_directions.append(LatticeDirection.X)
        if self.Ly > 1:
            self.active_directions.append(LatticeDirection.Y)
        if self.Lz > 1:
            self.active_directions.append(LatticeDirection.Z)

        # Dimension of active translations
        self.dim = len(self.active_directions)

    # --------------------------------------------------

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

    # --------------------------------------------------

    def analyze_1d_sectors(
        self,
        direction: LatticeDirection = LatticeDirection.X,
        verbose: bool = False,
        local_hilbert_base: int = 2,
    ) -> Dict[int, List[Tuple[int, Dict]]]:
        """
        Analyze 1D momentum sectors for a single translation direction.
        The analysis identifies representative states, their orbits, periods,
        and allowed momentum quantum numbers. This is irrespective of the number
        of sites per unit cell. In principle:

        - We start from a given state and generate its orbit under translation
        - The period of the orbit determines which momentum quantum numbers are allowed
        - The representative state is chosen as the minimum integer in the orbit

        Parameters
        ----------
        direction : LatticeDirection
            Translation direction to analyze.
        verbose : bool
            Print detailed analysis.
        local_hilbert_base : int
            Local Hilbert space base (e.g., 2 for spin-1/2).

        Returns
        -------
        Dict[int, List[Tuple[int, Dict]]]
            Mapping from momentum index k to list of (representative, info) pairs.
        """
        extent = self.get_extent(direction)
        # TranslationSymmetry now requires sector and ns; use neutral sector=0 for orbit analysis
        translator = TranslationSymmetry(self.lattice, sector=0, ns=self.ns, direction=direction)
        representatives = {}
        visited = set()

        # TODO: Optimize for large systems by avoiding full enumeration. Implement the orbit finding more efficiently.
        # Implement different local bases (e.g., fermionic occupation)

        # Optimize for large systems by avoiding full enumeration.
        perm_fwd = translator.get_site_permutation(self.ns)
        generators = np.array([np.argsort(perm_fwd)])
        full_group = _build_group_permutations(generators, self.ns)

        reps = _find_orbits_dfs_numba(self.ns, local_hilbert_base, full_group)

        t_reps = TypedList()
        for r in reps:
            t_reps.append(r)

        orbits = _get_orbits_1d(t_reps, self.ns, local_hilbert_base, np.argsort(perm_fwd))

        for rep, orbit in zip(reps, orbits):
            period = len(orbit)
            orbit_list = list(orbit)

            # Allowed momenta: k where k\cdot period ≡ 0 (mod extent)
            allowed_k = []

            # Find allowed momentum indices
            for q in range(extent):
                if (q * period) % extent == 0:
                    allowed_k.append(q)

            representatives[rep] = {
                "orbit": orbit_list,  # Full orbit states
                "period": period,  # Orbit period
                "allowed_k": allowed_k,  # Allowed momentum indices
                "norm": np.sqrt(period),  # Normalization factor
            }

        # Organize by momentum sector
        momentum_sectors = defaultdict(list)

        # Distribute representatives into momentum sectors
        for rep_state, info in representatives.items():
            for k in info["allowed_k"]:
                momentum_sectors[k].append((rep_state, info))

        return dict(momentum_sectors)

    def analyze_2d_sectors(
        self,
        directions: Optional[Tuple[LatticeDirection, LatticeDirection]] = None,
        verbose: bool = False,
        local_hilbert_base: int = 2,
    ) -> Dict[Tuple[int, int], List[Tuple[int, Dict]]]:
        """
        Analyze 2D momentum sectors for two translation directions.

        Parameters
        ----------
        directions : Tuple[LatticeDirection, LatticeDirection], optional
            Pair of translation directions. Defaults to (X, Y).
        verbose : bool
            Print detailed analysis.
        local_hilbert_base : int
            Local Hilbert space base (e.g., 2 for spin-1/2).

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

        # Use neutral sector for orbit generation; specify ns
        translator_x = TranslationSymmetry(self.lattice, sector=0, ns=self.ns, direction=dir_x)
        translator_y = TranslationSymmetry(self.lattice, sector=0, ns=self.ns, direction=dir_y)

        representatives = {}

        perm_x = translator_x.get_site_permutation(self.ns)
        perm_y = translator_y.get_site_permutation(self.ns)
        generators = np.array([np.argsort(perm_x), np.argsort(perm_y)])
        full_group = _build_group_permutations(generators, self.ns)

        reps = _find_orbits_dfs_numba(self.ns, local_hilbert_base, full_group)

        t_reps = TypedList()
        for r in reps:
            t_reps.append(r)

        orbits_x, orbits_y, orbits_2d = _get_orbits_2d(
            t_reps, self.ns, local_hilbert_base, np.argsort(perm_x), np.argsort(perm_y)
        )

        for rep, orb_x, orb_y, orb_2d in zip(reps, orbits_x, orbits_y, orbits_2d):
            orbit_x = list(orb_x)
            orbit_y = list(orb_y)
            orbit_2d = list(orb_2d)

            period_x = len(orbit_x)
            period_y = len(orbit_y)

            # Allowed momenta in each direction
            allowed_kx = []
            for q in range(extent_x):
                if (q * period_x) % extent_x == 0:
                    allowed_kx.append(q)

            allowed_ky = []
            for q in range(extent_y):
                if (q * period_y) % extent_y == 0:
                    allowed_ky.append(q)

            representatives[rep] = {
                "orbit_x": orbit_x,
                "orbit_y": orbit_y,
                "orbit_2d": orbit_2d,
                "period_x": period_x,
                "period_y": period_y,
                "allowed_kx": allowed_kx,
                "allowed_ky": allowed_ky,
                "norm": np.sqrt(len(orbit_2d)),
            }

        # Organize by (k_x, k_y) momentum sector
        momentum_sectors = defaultdict(list)
        for rep_state, info in representatives.items():
            for kx in info["allowed_kx"]:
                for ky in info["allowed_ky"]:
                    momentum_sectors[(kx, ky)].append((rep_state, info))

        return dict(momentum_sectors)

    # --------------------------------------------------
    #! Get representatives
    # --------------------------------------------------

    def get_sector_representatives(
        self, momentum_indices: Optional[Dict[LatticeDirection, int]] = None
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


# ------------------------------------------------------
#! Build momentum basis
# ------------------------------------------------------


def build_momentum_basis(
    lattice: "Lattice", momentum_indices: Dict[LatticeDirection, int], normalize: bool = True
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

    # Build translators for each active direction (provide sector and ns)
    translators = {}
    for direction in analyzer.active_directions:
        k = momentum_indices.get(direction, 0)
        translators[direction] = TranslationSymmetry(
            lattice, sector=int(k), ns=analyzer.ns, direction=direction
        )

    momentum_basis = {}

    for rep in representatives:
        # Build superposition from representative
        superposition = {rep: 1.0 + 0j}

        # Apply momentum projectors for each direction
        for direction, translator in translators.items():
            k = momentum_indices.get(direction, 0)
            new_superposition = defaultdict(complex)

            for state, amp in superposition.items():
                # Build orbit under translation using the translator callable
                orbit = []
                seen = set()
                cur = state
                while cur not in seen:
                    orbit.append(cur)
                    seen.add(cur)
                    cur, _ = translator(cur)
                period = len(orbit)

                for idx, orb_state in enumerate(orbit):
                    phase = np.exp(-1j * 2 * np.pi * k * idx / period)
                    new_superposition[orb_state] += amp * phase

            superposition = dict(new_superposition)

        # Normalize if requested
        if normalize:
            norm = np.sqrt(sum(abs(amp) ** 2 for amp in superposition.values()))
            if norm > 1e-10:
                superposition = {state: amp / norm for state, amp in superposition.items()}

        momentum_basis[rep] = superposition

    return momentum_basis


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#! EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
