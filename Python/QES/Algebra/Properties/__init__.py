"""
Properties Module
=================

This module contains calculations of physical properties of quantum systems.

Modules:
--------
- statistical   : Statistical properties and thermodynamic quantities
- time_evo      : Time evolution and dynamical properties

Usage:
------
Access these modules lazily through Hamiltonian properties:

    >>> hamil.statistical.ldos(overlaps)
    >>> hamil.statistical.fidelity_susceptibility(V_proj)
    >>> hamil.time_evo.evolve(psi0, t=1.0)
    >>> hamil.spectral.dynamic_structure_factor(omega, S_q)

Or import directly when needed:

    >>> from QES.Algebra.Properties.statistical import fidelity_susceptibility_low_rank
    >>> from QES.Algebra.Properties.time_evo import QuenchTypes

File    : QES/Algebra/Properties/__init__.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

# A short, user-facing description used by QES.registry
MODULE_DESCRIPTION  = "Physical properties: statistical mechanics, thermodynamics, and time evolution."
__all__             = ['statistical', 'time_evo']

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------