''' 
Convenience utilities for impurity models, including geometry computations and impurity normalization.

It includes utilities for computing distances and angles from impurity sites to lattice sites,
grouping sites into shells based on distance, and picking representative sites from each shell.

------------------------------------------------------------------------
Author      : Maksymilian Kliczkowski
Copyright   : (c) 2026 Maksymilian Kliczkowski
License     : MIT
Version     : 0.1.0
Changelog   : 
    - 2026-04-01 : Initial version.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from QES.Algebra.hamil              import Hamiltonian
    from QES.general_python.lattices    import Lattice

Impurity    = Tuple[int, float, float, float]  # (site, phi, theta, strength)
Impurities  = List[Impurity]

# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SiteGeometry:
    site            : int
    r               : float
    spatial_theta   : float # This is the angle in the xy-plane relative to the impurity site, computed as arctan2(dy, dx),
                            # where dx and dy are the x and y components of the displacement vector 
                            # from the impurity site to the site in question. Not to be confused with the spin angle theta of the impurity itself.

def make_impurity(site: int, phi: float = 0.0, theta: float = 0.0, amplitude: float = 1.0) -> Impurity:
    """
    Create an impurity tuple in the canonical ``(site, phi, theta, amplitude)`` format. 
    This is a convenience function to ensure consistent impurity specifications across the codebase.
    
    Parameters
    ----------
    site : int
        The index of the site where the impurity is located.
    phi : float, optional
        The azimuthal angle of the impurity's spin orientation in spherical coordinates (default is 0.0).
    theta : float, optional
        The polar angle of the impurity's spin orientation in spherical coordinates (default is 0.0).
    amplitude : float, optional
        The scalar amplitude of the impurity (default is 1.0).
        
    Returns
    -------
    Tuple[int, float, float, float]
        A tuple containing the site index, phi, theta, and amplitude of the impurity.
        
    Example
    -------
    >>> make_impurity(site=5, phi=np.pi/4, theta=np.pi/2, amplitude=0.5)
    (5, 0.7853981633974483, 1.5707963267948966, 0.5)
    
    Notes
    -----
    - The angles phi and theta are expected to be in radians. They correspond to 
    the spherical coordinate representation of the impurity's spin orientation, where phi is the azimuthal 
    angle in the xy-plane and theta is the polar angle from the z-axis.
    - The amplitude can be used to scale the strength of the impurity's effect on the system.
    - This function ensures that the impurity specification is always in a consistent format, 
    which can be important for downstream processing and computations involving impurities in the model.
    
    The impurities are always connected to the node index, either in the lattice or in the computational graph...
    """
    return int(site), float(phi), float(theta), float(amplitude)

def impurity_site_amplitude(impurity: Impurity) -> Tuple[int, float]:
    """
    Return the site index and scalar amplitude for one impurity specification.
    """
    try:
        from QES.Algebra.hamil import Hamiltonian
    except ImportError as e:
        raise ImportError(f"Failed to import Hamiltonian from QES.Algebra.hamil.\nOriginal error: {e}")
    
    return Hamiltonian.impurity_site_amplitude(impurity)

# ----------------------------------------------------------------------

def compute_impurity_distances(lattice: 'Lattice', impurities: Iterable[Impurity], *, minimum_image: bool = True) -> np.ndarray:
    """
    Compute the real-space distance from every impurity site to every lattice site.
    
    Parameters
    ----------
    lattice : 'Lattice'
        The lattice object which should have methods for computing distances or displacements between sites.
    impurities : Iterable[Impurity]
        An iterable of impurity specifications, where each impurity is an Impurity tuple containing at least the site index. The expected format is (site, phi, theta, amplitude), but only the site
        index is used for distance calculations.
    minimum_image : bool, optional
        Whether to use the minimum image convention when computing distances (default is True). This is relevant for periodic lattices, 
        where the distance between two sites should be computed considering the periodic boundaries to find the shortest distance.
        
    Returns
    -------
    np.ndarray
        A 2D array of shape (num_impurities, num_sites) containing the distances from each impurity site to each lattice site. The entry at [i, j] corresponds
        to the distance from the i-th impurity site to the j-th lattice site. 
        
    Notes
    -----
    - The function first normalizes the impurity specifications using the Hamiltonian's normalization method, which ensures that the impurities are in a consistent format. 
    - It then iterates over each impurity and computes the distance to each lattice site. 
    If the lattice object has a method for computing distances directly, 
    it uses that; otherwise, it falls back to computing the distance from the position vectors of the sites.
    """
    try:
        from QES.Algebra.hamil import Hamiltonian
    except ImportError as e:
        raise ImportError(f"Failed to import Hamiltonian from QES.Algebra.hamil.\nOriginal error: {e}")
    
    impurities_norm = Hamiltonian.normalize_impurities(list(impurities)) # Ensure impurities are in a consistent format
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

def compute_site_geometries(lattice: 'Lattice', anchor_site: int, *, minimum_image: bool = True) -> List[SiteGeometry]:
    """
    Compute distance and polar angle of every site relative to one anchor site.
    This means that we fix site ``anchor_site`` as the reference point (the impurity site) 
    and compute the distance and angle from this site to every other site in the lattice.
    
    Parameters
    ----------
    lattice : 'Lattice'
        The lattice object which should have methods for computing displacements between sites or provide site positions.
    anchor_site : int
        The index of the site that serves as the reference point (the impurity site) for the geometry calculations.
    minimum_image : bool, optional
        Whether to use the minimum image convention when computing displacements (default is True). This is relevant for periodic lattices, where the displacement between two sites should be computed considering the periodic boundaries to
        find the shortest displacement vector.
    Returns
    -------
    List[SiteGeometry]
        A list of SiteGeometry objects, where each object contains the site index, distance (r), and polar angle (spatial_theta) of a site relative to the anchor site. 
        The list includes all sites except the anchor site itself. 
        The distance is computed as the norm of the displacement vector, 
        and the polar angle is computed using arctan2 of the y and x components of the displacement.
    """
    anchor = int(anchor_site)
    out: List[SiteGeometry] = []

    for site in range(int(lattice.ns)):
        if site == anchor:
            continue

        if hasattr(lattice, "displacement"):
            disp        = np.asarray(lattice.displacement(anchor, site, minimum_image=minimum_image), dtype=float).reshape(-1)
        elif hasattr(lattice, "site_diff"):
            dx, dy, dz  = lattice.site_diff(anchor, site, minimum_image=minimum_image, real_space=True)
            disp        = np.asarray([dx, dy, dz], dtype=float).reshape(-1)
        else:
            positions   = np.asarray(lattice.rvectors, dtype=float)
            disp        = np.asarray(positions[site], dtype=float) - np.asarray(positions[anchor], dtype=float)

        # Compute distance and angle
        r       = float(np.linalg.norm(disp))
        theta   = float(np.arctan2(disp[1], disp[0])) if disp.shape[0] >= 2 else 0.0
        out.append(SiteGeometry(site=int(site), r=r, spatial_theta=theta))

    return out

# ----------------------------------------------------------------------

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
    
    The function attempts to pick sites that are evenly distributed in angle (spatial_theta) around the impurity site.
    
    Parameters
    ----------
    shell : List[SiteGeometry]
        A list of SiteGeometry objects representing sites in a radial shell.
    n_pick : int
        The number of sites to pick from the shell.

    Returns
    -------
    List[SiteGeometry]
        A list of the picked SiteGeometry objects.

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