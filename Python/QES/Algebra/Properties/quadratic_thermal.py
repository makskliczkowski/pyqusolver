"""
QES/Algebra/Properties/quadratic_thermal.py

Thermal properties for quadratic (non-interacting) Hamiltonians.

For quadratic Hamiltonians (free fermions, bosons), thermal properties
can be computed efficiently from single-particle eigenvalues without
diagonalizing the full many-body Hamiltonian.

This module is QES-specific as it relies on the quadratic structure
of certain Hamiltonians in the QES framework.

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp

try:
    from QES.general_python.algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    JAX_AVAILABLE = False
    Array = np.ndarray

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Fermionic Thermal Quantities
# =============================================================================

def fermi_occupation(epsilon: Array, beta: float, mu: float = 0.0) -> Array:
    r"""
    Compute Fermi-Dirac occupation number.
    
    f(\varepsilon) = 1 / (\exp(\beta(\varepsilon - \mu)) + 1)
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
    mu : float, optional
        Chemical potential (default: 0).
        
    Returns
    -------
    Array
        Occupation numbers f(\varepsilon).
    """
    epsilon = np.asarray(epsilon)
    
    # Prevent overflow in exp
    x = beta * (epsilon - mu)
    x = np.clip(x, -500, 500)
    
    return 1.0 / (np.exp(x) + 1.0)


def fermi_grand_potential(
        epsilon: Array,
        beta: float,
        mu: float = 0.0
) -> float:
    """
    Compute grand potential \Omega for non-interacting fermions.
    
    \Omega = -k_B T \sum _k ln(1 + exp(-\beta(\varepsilon_k - \mu)))
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies \varepsilon_k.
    beta : float
        Inverse temperature.
    mu : float, optional
        Chemical potential (default: 0).
        
    Returns
    -------
    float
        Grand potential \Omega (in units where k_B = 1).
    """
    epsilon = np.asarray(epsilon)
    
    x = beta * (epsilon - mu)
    x = np.clip(x, -500, 500)
    
    # \Omega = -(1/\beta) Σ ln(1 + exp(-x))
    return -np.sum(np.log(1.0 + np.exp(-x))) / beta


def fermi_particle_number(
        epsilon: Array,
        beta: float,
        mu: float = 0.0
) -> float:
    """
    Compute average particle number for fermions.
    
    <N> = \sum _k f(\varepsilon_k)
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies.
    beta : float
        Inverse temperature.
    mu : float, optional
        Chemical potential (default: 0).
        
    Returns
    -------
    float
        Average particle number.
    """
    return np.sum(fermi_occupation(epsilon, beta, mu))


def fermi_internal_energy(
        epsilon: Array,
        beta: float,
        mu: float = 0.0
) -> float:
    """
    Compute internal energy for fermions.
    
    U = \sum _k \varepsilon_k f(\varepsilon_k)
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies.
    beta : float
        Inverse temperature.
    mu : float, optional
        Chemical potential (default: 0).
        
    Returns
    -------
    float
        Internal energy U.
    """
    epsilon = np.asarray(epsilon)
    f = fermi_occupation(epsilon, beta, mu)
    
    return np.sum(epsilon * f)


def fermi_heat_capacity(
        epsilon: Array,
        beta: float,
        mu: float = 0.0
) -> float:
    """
    Compute heat capacity for fermions.
    
    C_V = -\beta^2 ∂^2\Omega/∂\beta^2 = \sum _k (\varepsilon_k - \mu)^2 f(\varepsilon_k)(1 - f(\varepsilon_k)) \beta^2
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies.
    beta : float
        Inverse temperature.
    mu : float, optional
        Chemical potential (default: 0).
        
    Returns
    -------
    float
        Heat capacity C_V.
    """
    epsilon = np.asarray(epsilon)
    f = fermi_occupation(epsilon, beta, mu)
    
    # C_V = \beta^2 Σ (\varepsilon - \mu)^2 f(1-f)
    return beta**2 * np.sum((epsilon - mu)**2 * f * (1.0 - f))


# =============================================================================
# Bosonic Thermal Quantities
# =============================================================================

def bose_occupation(epsilon: Array, beta: float, mu: float = 0.0) -> Array:
    """
    Compute Bose-Einstein occupation number.
    
    n(\varepsilon) = 1 / (exp(\beta(\varepsilon - \mu)) - 1)
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies (must have \varepsilon > \mu).
    beta : float
        Inverse temperature.
    mu : float, optional
        Chemical potential (default: 0, must be < min(\varepsilon)).
        
    Returns
    -------
    Array
        Occupation numbers n(\varepsilon).
        
    Notes
    -----
    For bosons, \mu < min(\varepsilon) to avoid divergence. Typically \mu ≤ 0.
    """
    epsilon = np.asarray(epsilon)
    
    x = beta * (epsilon - mu)
    
    # Check for validity
    if np.any(x <= 0):
        raise ValueError("For bosons, must have \varepsilon > \mu for all states")
    
    x = np.clip(x, 0.01, 500)  # Avoid division by zero and overflow
    
    return 1.0 / (np.exp(x) - 1.0)


def bose_grand_potential(
        epsilon: Array,
        beta: float,
        mu: float = 0.0
) -> float:
    """
    Compute grand potential \Omega for non-interacting bosons.
    
    \Omega = k_B T \sum _k ln(1 - exp(-\beta(\varepsilon_k - \mu)))
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies.
    beta : float
        Inverse temperature.
    mu : float, optional
        Chemical potential (must be < min(\varepsilon)).
        
    Returns
    -------
    float
        Grand potential \Omega.
    """
    epsilon = np.asarray(epsilon)
    
    x = beta * (epsilon - mu)
    if np.any(x <= 0):
        raise ValueError("For bosons, must have \varepsilon > \mu")
    
    x = np.clip(x, 0.01, 500)
    
    return np.sum(np.log(1.0 - np.exp(-x))) / beta


# =============================================================================
# Chemical Potential Calculation
# =============================================================================

def find_chemical_potential_fermions(
        epsilon: Array,
        beta: float,
        target_particle_number: float,
        mu_min: float = -10.0,
        mu_max: float = 10.0,
        tol: float = 1e-6
) -> float:
    """
    Find chemical potential \mu such that <N> = N_target for fermions.
    
    Uses bisection method to solve \sum _k f(\varepsilon_k, \mu) = N_target.
    
    Parameters
    ----------
    epsilon : array-like
        Single-particle energies.
    beta : float
        Inverse temperature.
    target_particle_number : float
        Desired average particle number.
    mu_min, mu_max : float, optional
        Search range for chemical potential.
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
        
    Returns
    -------
    float
        Chemical potential \mu.
    """
    from scipy.optimize import brentq
    
    def objective(mu):
        return fermi_particle_number(epsilon, beta, mu) - target_particle_number
    
    mu = brentq(objective, mu_min, mu_max, xtol=tol)
    
    return mu


# =============================================================================
# Quadratic Hamiltonian Thermal Scan
# =============================================================================

def quadratic_thermal_scan(
        single_particle_energies: Array,
        temperatures: Array,
        particle_type: str = 'fermion',
        particle_number: Optional[float] = None
) -> dict:
    """
    Compute thermal properties of a quadratic Hamiltonian over a temperature range.
    
    Parameters
    ----------
    single_particle_energies : array-like
        Single-particle eigenvalues \varepsilon_k.
    temperatures : array-like
        Array of temperatures.
    particle_type : str, optional
        'fermion' or 'boson' (default: 'fermion').
    particle_number : float, optional
        If provided, compute chemical potential to fix <N> = particle_number.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'T': temperatures
        - 'beta': inverse temperatures
        - 'mu': chemical potentials (if particle_number is set)
        - 'Omega': grand potentials
        - 'U': internal energies
        - 'N': particle numbers
        - 'C_V': heat capacities
        - 'S': entropies (calculated as S = \beta(U - \Omega))
        
    Examples
    --------
    >>> # Free fermion system
    >>> epsilon = np.linspace(-2, 2, 100)  # tight-binding band
    >>> temps = np.linspace(0.1, 5.0, 50)
    >>> results = quadratic_thermal_scan(epsilon, temps, particle_type='fermion', particle_number=50)
    >>> plt.plot(results['T'], results['C_V'], label='Heat Capacity')
    """
    epsilon = np.asarray(single_particle_energies)
    temperatures = np.asarray(temperatures)
    n_temps = len(temperatures)
    
    if particle_type.lower() not in ['fermion', 'boson']:
        raise ValueError("particle_type must be 'fermion' or 'boson'")
    
    is_fermion = (particle_type.lower() == 'fermion')
    
    results = {
        'T': temperatures,
        'beta': 1.0 / temperatures,
        'mu': np.zeros(n_temps),
        'Omega': np.zeros(n_temps),
        'U': np.zeros(n_temps),
        'N': np.zeros(n_temps),
        'C_V': np.zeros(n_temps),
        'S': np.zeros(n_temps),
    }
    
    for i, T in enumerate(temperatures):
        beta = 1.0 / T
        
        # Determine chemical potential
        if particle_number is not None and is_fermion:
            mu = find_chemical_potential_fermions(
                epsilon, beta, particle_number,
                mu_min=epsilon.min() - 5.0,
                mu_max=epsilon.max() + 5.0
            )
        else:
            mu = 0.0  # Default \mu=0
        
        results['mu'][i] = mu
        
        # Compute quantities
        if is_fermion:
            results['Omega'][i] = fermi_grand_potential(epsilon, beta, mu)
            results['U'][i] = fermi_internal_energy(epsilon, beta, mu)
            results['N'][i] = fermi_particle_number(epsilon, beta, mu)
            results['C_V'][i] = fermi_heat_capacity(epsilon, beta, mu)
        else:
            # Bosons (simplified - no particle number fixing)
            results['Omega'][i] = bose_grand_potential(epsilon, beta, mu)
            n = bose_occupation(epsilon, beta, mu)
            results['U'][i] = np.sum(epsilon * n)
            results['N'][i] = np.sum(n)
            # Approximate C_V for bosons
            results['C_V'][i] = beta**2 * np.sum((epsilon - mu)**2 * n * (n + 1))
        
        # Entropy: S = \beta(U - \Omega)
        results['S'][i] = beta * (results['U'][i] - results['Omega'][i])
    
    return results


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Fermions
    'fermi_occupation',
    'fermi_grand_potential',
    'fermi_particle_number',
    'fermi_internal_energy',
    'fermi_heat_capacity',
    'find_chemical_potential_fermions',
    
    # Bosons
    'bose_occupation',
    'bose_grand_potential',
    
    # Scans
    'quadratic_thermal_scan',
]
