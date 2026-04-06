''' 
Convenience utilities for impurity models, including geometry computations and impurity normalization.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import numpy as np

from QES.Algebra.hamil import Hamiltonian


@dataclass(frozen=True)
class SiteGeometry:
    site            : int
    r               : float
    spatial_theta   : float

def make_impurity(site: int, phi: float = 0.0, theta: float = 0.0, amplitude: float = 1.0) -> Tuple[int, float, float, float]:
    """
    Create an impurity tuple in the canonical ``(site, phi, theta, amplitude)`` format.
    """
    return int(site), float(phi), float(theta), float(amplitude)

def impurity_site_amplitude(impurity: Tuple) -> Tuple[int, float]:
    """
    Return the site index and scalar amplitude for one impurity specification.
    """
    return Hamiltonian.impurity_site_amplitude(impurity)

def compute_impurity_distances(lattice: Any, impurities: Iterable[Tuple], *, minimum_image: bool = True) -> np.ndarray:
    """
    Compute the real-space distance from every impurity site to every lattice site.
    """
    impurities_norm = Hamiltonian.normalize_impurities(list(impurities))
    ns              = int(lattice.ns)
    out             = np.zeros((len(impurities_norm), ns), dtype=float)

    for idx, (site, _phi, _theta, _amplitude) in enumerate(impurities_norm):
        for j in range(ns):
            if hasattr(lattice, "distance"):
                out[idx, j] = float(lattice.distance(site, j, minimum_image=minimum_image))
            else:
                pos_i       = np.asarray(lattice.rvectors[site], dtype=float)
                pos_j       = np.asarray(lattice.rvectors[j], dtype=float)
                out[idx, j] = float(np.linalg.norm(pos_j - pos_i))
    return out

def compute_site_geometries(lattice: Any, anchor_site: int, *, minimum_image: bool = True) -> List[SiteGeometry]:
    """
    Compute distance and polar angle of every site relative to one anchor site.
    """
    anchor = int(anchor_site)
    out: List[SiteGeometry] = []

    for site in range(int(lattice.ns)):
        if site == anchor:
            continue

        if hasattr(lattice, "displacement"):
            disp = np.asarray(lattice.displacement(anchor, site, minimum_image=minimum_image), dtype=float).reshape(-1)
        elif hasattr(lattice, "site_diff"):
            dx, dy, dz = lattice.site_diff(anchor, site, minimum_image=minimum_image, real_space=True)
            disp = np.asarray([dx, dy, dz], dtype=float).reshape(-1)
        else:
            positions = np.asarray(lattice.rvectors, dtype=float)
            disp = np.asarray(positions[site], dtype=float) - np.asarray(positions[anchor], dtype=float)

        r = float(np.linalg.norm(disp))
        theta = float(np.arctan2(disp[1], disp[0])) if disp.shape[0] >= 2 else 0.0
        out.append(SiteGeometry(site=int(site), r=r, spatial_theta=theta))

    return out

def group_site_geometries(site_geometries: Iterable[SiteGeometry], *, round_digits: int = 8, reverse: bool = True) -> List[List[SiteGeometry]]:
    """
    Group site geometries into radial shells using rounded distance as the shell key.
    """
    grouped = {}
    for geom in site_geometries:
        grouped.setdefault(round(float(geom.r), round_digits), []).append(geom)

    return [
        sorted(shell, key=lambda geom: geom.spatial_theta)
        for _radius, shell in sorted(grouped.items(), key=lambda item: item[0], reverse=reverse)
    ]

# ----------------------------------------------------------------------
# Utilities for picking representative sites from shells
# ----------------------------------------------------------------------

def pick_site_geometries_by_angle(shell: List[SiteGeometry], n_pick: int) -> List[SiteGeometry]:
    """
    Pick up to ``n_pick`` representative sites from one shell, distributed by angle.
    """
    if n_pick <= 0:
        return []
    if len(shell) <= n_pick:
        return shell[:]

    idx                     = np.linspace(0, len(shell) - 1, n_pick, dtype=int)
    used                    = set()
    out: List[SiteGeometry] = []

    for candidate in idx.tolist():
        probe = candidate
        while probe in used and probe + 1 < len(shell):
            probe += 1
        if probe in used:
            for fallback in range(len(shell)):
                if fallback not in used:
                    probe = fallback
                    break
        used.add(probe)
        out.append(shell[probe])

    return out

# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------